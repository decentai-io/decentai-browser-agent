"""The run: look, decide a few actions, act, see what changed, look
again — with the person's cards at the moments that are theirs, the
live view and its inspector throughout, and what was learned kept for
next time.

One model call gives up to six actions. They run in order, and the run
goes back to the model as soon as one of them needs its eyes: it
failed, it changed nothing, it opened another page or a dialog, or it
read something the model must see. Every result says what the action
changed, measured on the page's outline before and after.

An action that changed nothing is not done again on the same page: a
second try is refused and the model is told. Three actions in a row
that changed nothing (refused or not), or a run going in circles, get
one call at the chat's own effort to find another way; when that too
comes to nothing, the person is offered the browser, once for a page,
before the run gives up.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from standard import ResourceDenied, ToolBase

from . import guards
from .brain import Brain
from .driver import VIEWPORT, Driver, DriverError, Snapshot, changes
from .inspector import InspectError
from .judge import Judge
from .live import Live
from .memory import SiteNotes, cut
from .policy import domain_of

#: A frame this often while a browser is handed to the person and
#: their hand is off it.
HANDOFF_FRAME_SECONDS = 0.35
#: While the person drives, frames on their own clock, this often:
#: their hand is never made to wait for a picture.
HANDOFF_FRAME_MIN_SECONDS = 0.1
#: While a run works, a frame at least this often, between the ones
#: taken after each action.
RUN_FRAME_SECONDS = 1.0
#: How long the input loop waits for the person's next action before
#: looking again at whether it should end.
INPUT_WAIT_SECONDS = 0.05
#: What the notes of one site may hold (the platform's limit on a plain
#: field is a little above it).
NOTES_CHARS = 8000
#: What is kept of one stretch of the person's own doing, on the notes,
#: and how much of it the mind is told at once.
TRAIL_NOTE_MAX = 400
TRAIL_SHOWN = 12


async def show_frame(call, driver, image: bytes, width: int, height: int) -> None:
    """One frame to the live view, with the tabs that are open: a
    person who cannot see the tabs cannot tell where a click went, nor
    close the one they do not want."""
    await call.screen.show(image, width, height, await driver.tabs())


async def drive(call, driver, until, idle_interval: float = HANDOFF_FRAME_SECONDS,
                hear=None, may_drive=None):
    """The person's hand on a browser: inputs go to the page the moment
    they arrive, and frames flow on their own clock beside them — one
    every HANDOFF_FRAME_MIN_SECONDS while they hold control, lighter
    pictures, and one every ``idle_interval`` otherwise. Ends when
    ``until()`` says why. ``may_drive()`` false means the person only
    watches for now — a run holds the browser — and their input is
    dropped rather than fed under the run's hand. Returns (why, what
    they did in words)."""
    trail: List[str] = []
    stop = asyncio.Event()

    async def frames() -> None:
        while not stop.is_set():
            taken = call.screen.taken
            started = time.monotonic()
            if taken:
                # Their hand is on the browser: what the page fetches
                # now may be their sign-in, and is never opened.
                driver.inspector.guarded = True
            try:
                image = await driver.frame(fast=taken)
                if image and driver.page is not None:
                    size = driver.page.viewport_size
                    width, height = (size["width"], size["height"]) if size else VIEWPORT
                    await show_frame(call, driver, image, width, height)
            except Exception:
                pass
            interval = HANDOFF_FRAME_MIN_SECONDS if taken else idle_interval
            await asyncio.sleep(max(0.0, interval - (time.monotonic() - started)))

    pictures = asyncio.get_running_loop().create_task(frames())
    try:
        while True:
            # What is already waiting is the person's, and reaches the
            # page before it is asked whether they still hold it: a
            # take, a few actions and a release can all arrive inside
            # one of the run's steps.
            events = call.screen.inputs()
            if not events:
                why = until()
                if why:
                    return why, trail
                events = await call.screen.wait_input(INPUT_WAIT_SECONDS)
            if events and (may_drive is None or may_drive()):
                try:
                    trail += await driver.dispatch(events)
                except Exception:
                    pass
            if hear is not None:
                for text in call.screen.said():
                    hear(text)
    finally:
        stop.set()
        pictures.cancel()


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


#: A watch (the browser shown on request) ends on its own after this.
WATCH_MAX_SECONDS = 3600
#: While the person only watches, a frame this often; taking over
#: makes frames follow their hand (HANDOFF_FRAME_MIN_SECONDS).
WATCH_IDLE_FRAME_SECONDS = 1.0


#: A chat's browser nobody has used for this long is closed: Chromium is
#: most of a process's memory, and a sandbox holds many processes. Where
#: it was and what the last run did are kept, and its sign-ins are on the
#: site's session record, so the next run opens a browser there.
IDLE_CLOSE_SECONDS = int(os.environ.get("DECENTAI_BROWSER_IDLE_SECONDS") or 180)


class Kept:
    """The chat's browser, kept open between its runs: the next run
    finds the page the last one left, and is told what that run did.
    One at a time — a second run waits its turn on the lock."""

    def __init__(self, driver: Driver):
        self.driver = driver
        self.lock = asyncio.Lock()
        #: what the last run in this browser did, for the next one
        self.last = ""
        #: the browser's own idle clock, while no call uses it
        self.closing: Optional[asyncio.Task] = None


class Browser:
    """The one browser of this process. A process serves one chat, so
    nothing here is ever handed across chats; it goes when the process
    does, which the runtime ends once the chat stops using it. Where it
    was is kept as long as the process lives, so the next run or watch
    without an address starts there after it closed."""

    def __init__(self):
        self.kept: Optional[Kept] = None
        self.was = ""
        #: what the last run did, kept when its browser is closed for want of use
        self.last = ""

    def get(self) -> Optional[Kept]:
        if self.kept is not None and self.kept.driver.page is None:
            self.kept = None
        if self.kept is not None and self.kept.closing is not None:
            self.kept.closing.cancel()
            self.kept.closing = None
        return self.kept

    def idle(self) -> None:
        """No call uses the browser now: it is closed after IDLE_CLOSE_SECONDS
        unless one does meanwhile."""
        kept = self.kept
        if kept is None or IDLE_CLOSE_SECONDS <= 0:
            return
        if kept.closing is not None:
            kept.closing.cancel()

        async def close() -> None:
            await asyncio.sleep(IDLE_CLOSE_SECONDS)
            if self.kept is not kept or kept.lock.locked():
                return
            if kept.driver.page is not None:
                self.remember(kept.driver.page.url)
            self.last, self.kept = kept.last, None
            await kept.driver.stop()

        kept.closing = asyncio.get_running_loop().create_task(close())

    def keep(self, driver: Driver) -> Kept:
        if self.kept is None or self.kept.driver is not driver:
            self.kept = Kept(driver)
        return self.kept

    def remember(self, url: str) -> None:
        if str(url or "").startswith(("http://", "https://")):
            self.was = str(url)

    async def drop(self) -> None:
        kept, self.kept = self.kept, None
        if kept is not None:
            await kept.driver.stop()


BROWSER = Browser()


class BrowseTool(ToolBase):
    id = "browse"

    #: Actions in a row that changed nothing — done or refused — before
    #: the model is asked, at the chat's effort, for another way.
    IDLE_AFTER = 3
    #: Those calls for another way, before the person is offered the
    #: browser (once for a page) or the run ends.
    AGAIN_MOST = 1
    #: This many answers in a row that could not be read as actions end
    #: the run: the model is told each time what was wrong, and a model
    #: that cannot answer three times will not on the tenth.
    UNREADABLE_STOP = 3
    #: Answers in a row the provider refused to give (its filter): the
    #: same page will be refused again, so two end the run, said as that.
    REFUSED_STOP = 2
    #: Actions meant to change the page: what they changed is measured.
    MOVING = frozenset(("click", "type", "select", "press", "scroll", "goto", "back",
                        "tab", "hover", "login", "code", "download", "upload", "handoff"))
    #: Actions whose result the model must read before it does more:
    #: the rest of their answer's actions are not run.
    READING = frozenset(("read", "recall", "source", "network", "response", "console", "request",
                         "extract", "script", "ask", "look", "invalid", "login", "code", "handoff"))

    # -- browse_screenshot ---------------------------------------------------
    async def screenshot(self, call):
        url = str(call.inputs["url"]).strip()
        driver = Driver(headless=self._headless())
        try:
            await call.progress("Starting the browser")
            await driver.start()
            await call.progress(f"Opening {url}")
            await driver.goto(url)
            snapshot = await driver.snapshot(look=True)
            await call.progress("Saving the screenshot")
            saved = self._save_image(call, snapshot, "screenshot")
            await call.progress("Closing the browser")
            return {"file_ref": saved, "title": snapshot.title,
                    "final_url": snapshot.url}, "success"
        except DriverError as exc:
            return {"error": str(exc), "kind": "browser"}, "error"
        finally:
            await driver.stop()

    # -- browse_run ------------------------------------------------------------
    async def run(self, call):
        goal = str(call.inputs["goal"]).strip()
        start_url = str(call.inputs.get("start_url") or "").strip()
        if start_url and "://" not in start_url:
            start_url = "https://" + start_url
        max_steps = int(call.inputs.get("max_steps") or 60)

        # The chat's browser, if the last run left one open: the run
        # continues on its page unless sent elsewhere.
        kept = BROWSER.get()
        if kept is not None:
            async with kept.lock:
                try:
                    return await self._run_in(call, kept.driver, goal, start_url,
                                              max_steps, continuing=True, earlier=kept.last)
                finally:
                    BROWSER.idle()
        if not start_url:
            # No browser and no address: where the chat's browser was
            # when it last closed, if anywhere.
            start_url = BROWSER.was
            if start_url:
                await call.progress(f"Resuming at {start_url}")
        if not start_url:
            return {"outcome": "failed", "steps": 0, "final_url": "", "records": [],
                    "summary": "No browser is open in this chat and no "
                               "start_url was given: say where to start."}, "success"
        driver = Driver(headless=self._headless())
        try:
            # A browser closed for want of use: what its last run did is
            # still told to this one.
            return await self._run_in(call, driver, goal, start_url, max_steps, continuing=False,
                                      earlier=BROWSER.last)
        finally:
            BROWSER.idle()

    async def _run_in(self, call, driver: Driver, goal: str, start_url: str,
                      max_steps: int, continuing: bool, earlier: str = ""):
        here = driver.page.url if continuing and driver.page is not None else ""
        domain = domain_of(start_url or here)
        brain = Brain(call)
        # The files the person handed the run, by their paths in the
        # program's home: what upload may put into a page.
        brain.files = [str(given) for given in call.inputs.get("files") or []]
        brain.memory.earlier = earlier
        run = _Run(call, driver, brain, goal, domain, max_steps)
        try:
            # Whether the person asked only to look is the model's to
            # say from the goal alone, while the browser starts.
            mode = asyncio.ensure_future(brain.mode(goal))
            brain.notes, session = await asyncio.gather(run.load_notes(), run.load_session())
            if not continuing:
                await driver.start(storage_state=session)
            run.live.start()
            if start_url and (not continuing or domain_of(here) != domain
                              or start_url.rstrip("/") != here.rstrip("/")):
                await call.progress(f"Opening {start_url}")
                run.live.set(now=f"Opening {start_url}")
                await driver.goto(start_url)
            elif continuing:
                await call.progress(f"Continuing at {here}")
                brain.continuing = here
            run.live.set(read_only=await mode)
            outcome = await run.loop(max_steps)
        except DriverError as exc:
            outcome = {"outcome": "failed", "summary": str(exc)}
        except Exception as exc:  # the browser died, the model refused
            outcome = {"outcome": "failed", "summary": f"The run stopped: {exc}"}
        finally:
            result = await run.close(outcome)
            if driver.page is not None:
                kept = BROWSER.keep(driver)
                kept.last = brain.memory.handed_on(goal, result["outcome"], result["summary"])
                BROWSER.remember(result.get("final_url", ""))
            else:
                await driver.stop()
        return result, "success"

    async def watch(self, call):
        """The browser shown on request, before anything is asked: the
        chat's kept browser, or a fresh one at a blank page that is kept
        from now on. Frames flow, the person's hand reaches the page,
        and whatever they did — a sign-in above all — is the browser's
        state for the next run. Ends when they close the panel or on its
        own clock. Never holds the browser's lock: a run takes it
        whenever it needs to, and the person watches until the run is
        done."""
        kept = BROWSER.get()
        if str(call.inputs.get("action") or "open") == "quit":
            # The person closed the browser itself, not the panel: where
            # it was is remembered, and it goes now.
            url = kept.driver.page.url if kept is not None and kept.driver.page is not None else ""
            if kept is not None:
                BROWSER.remember(url)
                await BROWSER.drop()
                await call.progress("The browser is closed")
            return {"outcome": "quit", "final_url": url,
                    "summary": "The browser was closed" + (f" at {url}" if url else "") + "."}, "success"
        if kept is None:
            resume = BROWSER.was
            driver = Driver(headless=self._headless())
            await driver.start(storage_state=await _Run.session_for(call, domain_of(resume)) if resume else None)
            if resume:
                try:
                    await call.progress(f"Resuming at {resume}")
                    await driver.goto(resume)
                except DriverError as exc:
                    await call.progress(str(exc))
            kept = BROWSER.keep(driver)
        driver = kept.driver
        await call.progress("Showing the browser")
        started = time.monotonic()
        opened_at = driver.page.url if driver.page is not None else ""

        def until() -> str:
            if call.screen.closed:
                return "closed"
            if time.monotonic() - started > WATCH_MAX_SECONDS:
                return "timeout"
            if driver.page is None:
                return "browser_closed"
            return ""

        # A run of the same chat may take the browser while the person
        # watches: the picture goes on, their hand waits until the run
        # is done with it. A typed address goes through the same policy
        # a run's goto obeys.
        outcome, trail = await drive(call, driver, until, idle_interval=WATCH_IDLE_FRAME_SECONDS,
                                     may_drive=lambda: not kept.lock.locked())
        for line in trail:
            if "cannot be visited" in line:
                await call.progress(line)
        await self._note_person(call, driver, trail)
        if driver.page is not None:
            BROWSER.remember(driver.page.url)
            if trail:
                # What the person did by hand — a sign-in above all — is
                # the browser's alone until it is kept: a browser that
                # closes before the next run would take it along. Kept
                # under the site they began at and the one they ended
                # at, since a sign-in often starts on one and lands on
                # another, and the next run may start at either.
                for domain in dict.fromkeys((domain_of(driver.page.url), domain_of(opened_at))):
                    await _Run.keep_session(call, driver, domain)
        try:
            await call.screen.close()
        except Exception:
            pass
        BROWSER.idle()
        url = driver.page.url if driver.page is not None else ""
        return {"outcome": outcome, "final_url": url,
                "summary": {"closed": "The person closed the live view.",
                            "timeout": "The watch ended on its own clock.",
                            "browser_closed": "The browser closed."}.get(outcome, outcome)}, "success"

    async def _note_person(self, call, driver: Driver, trail: List[str]) -> None:
        """What the person did in a watched browser, kept on the site's
        notes so the next run there can try it itself: the sign-in they
        did by hand is the sign-in the agent will know."""
        lines = [t for t in trail if "cannot be visited" not in t]
        url = driver.page.url if driver.page is not None else ""
        domain = domain_of(url) if url and not url.startswith("about:") else ""
        if not lines or not domain:
            return
        try:
            rows = await call.records.list("notes", {"domain": domain})
            notes = SiteNotes(str(rows[0]["fields"].get("notes") or "") if rows else "")
            notes.add("person", now_iso()[:10] + ": " + "; ".join(lines)[:TRAIL_NOTE_MAX])
            fields = {"domain": domain, "updated_at": now_iso(),
                      "notes": notes.text()[:NOTES_CHARS]}
            if rows:
                await call.records.update("notes", rows[0]["ref"], fields)
            else:
                await call.records.create("notes", fields)
        except Exception:
            pass

    @staticmethod
    def _headless() -> bool:
        return os.environ.get("DECENTAI_BROWSER_HEADLESS", "1") != "0"

    @staticmethod
    def _save_image(call, snapshot: Snapshot, name: str) -> str:
        """The picture, returned to the chat as a file: its path, which the runtime makes the file's ref."""
        if not snapshot.image:
            return ""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return call.give(f"{name}-{stamp}.jpg", snapshot.image)

    async def close(self) -> None:
        """The process ends, and the chat's browser with it."""
        await BROWSER.drop()


class _Run:
    """One browse_run from start to result."""

    #: What one reading action may bring into the run's memory. How
    #: much of it the mind is shown at a step is the memory's budget.
    TAKEN_CHARS = 40000
    #: What a question may hold (the standard's limit on chat text),
    #: and each of its choices.
    CARD_CHARS = 4000
    CHOICE_CHARS = 100
    #: A press that commits, approved, is shown to the judge with the
    #: next one that commits on its site for this many steps: one that
    #: only confirms it is not asked again.
    CONFIRMS_WITHIN = 3
    #: The finish the check refuses this many times is taken anyway,
    #: said as not confirmed on the page.
    REFUSED_FINISH_MOST = 2

    def __init__(self, call, driver: Driver, brain: Brain, goal: str, domain: str, max_steps: int = 60):
        self.call = call
        self.driver = driver
        self.brain = brain
        self.judge = Judge(brain.meter)
        self.live = Live(call, driver, goal, max_steps)
        self.goal = goal
        self.domain = domain
        self.steps = 0
        #: answers in a row that could not be read as actions, and that
        #: the provider refused to give
        self.unreadable = 0
        self.refusals = 0
        self.approvals: List[str] = []
        self.handoffs = 0
        self.downloads: List[Dict[str, Any]] = []
        self.account = ""
        #: the names the person's values were asked under on this run,
        #: with their labels — kept on the site's notes, so the next
        #: run asks for the same field under the same name
        self.login_names: Dict[str, str] = {}
        self.session_dirty = False
        self.last: Optional[Snapshot] = None
        #: the pages where the browser was already offered to the
        #: person because the run was stuck: offered once, not for ever
        self.offered: List[str] = []
        #: actions that changed nothing, by what they were and the page
        #: they were done on: not done again there
        self.idle_marks: set = set()
        #: actions in a row that changed nothing, done or refused
        self.idle = 0
        #: calls for another way since the run last got anywhere
        self.agains = 0
        self.refused_repeats = 0
        self.refused_finishes = 0
        #: controls the person allowed for the rest of the run, and how
        #: often each was approved
        self.allowed: set = set()
        self.approved_counts: Dict[str, int] = {}
        #: the newest press that commits the person approved, while it
        #: may still be confirmed: {"words", "why", "domain", "step"}
        self.committed: Optional[Dict[str, Any]] = None
        #: extractions run unasked until one is seen to change the page
        self.extract_trusted = True
        self.cards = 0
        self.browser_ms = 0
        #: notes for the model's next step
        self.notes: List[str] = []
        #: the person's hand is on the browser, and frames follow it
        self.driving = False

    # -- what is kept between runs -----------------------------------------
    async def load_notes(self) -> SiteNotes:
        try:
            rows = await self.call.records.list("notes", {"domain": self.domain})
            if not rows:
                return SiteNotes()
            # The notes are encrypted (values) since 2.1.0, so a listing doesn't carry them: the record is read.
            # Notes kept by an earlier version in plain keys come back in the listing.
            listed = str(rows[0]["fields"].get("notes") or "")
            read = await self.call.records.read("notes", rows[0]["ref"])
            return SiteNotes(str((read.get("fields") or {}).get("notes") or listed))
        except Exception:
            return SiteNotes()

    async def save_notes(self, notes: SiteNotes) -> None:
        text = notes.text()
        if not text.strip():
            return
        fields = {"domain": self.domain, "updated_at": now_iso(), "notes": text[:NOTES_CHARS]}
        try:
            rows = await self.call.records.list("notes", {"domain": self.domain})
            if rows:
                await self.call.records.update("notes", rows[0]["ref"], fields)
            else:
                await self.call.records.create("notes", fields)
        except Exception:
            pass

    async def load_session(self) -> Optional[Dict[str, Any]]:
        return await self.session_for(self.call, self.domain)

    @staticmethod
    async def session_for(call, domain: str) -> Optional[Dict[str, Any]]:
        """A site's saved sign-in, as a browser context takes it."""
        if not domain:
            return None
        try:
            rows = await call.records.list("session", {"domain": domain})
            if not rows:
                return None
            state = (await call.records.read("session", rows[0]["ref"]))["fields"].get("state")
            return state if isinstance(state, dict) and state.get("cookies") else None
        except Exception:
            return None

    async def save_session(self) -> None:
        await self.keep_session(self.call, self.driver, self.domain, self.account)

    @staticmethod
    async def keep_session(call, driver: Driver, domain: str, account: str = "") -> None:
        """The browser's sign-ins, kept under a site's name for the next
        browser that opens there. Without an account to name, the one
        already on the row stays."""
        if not domain:
            return
        try:
            state = await driver.storage_state()
            if not state.get("cookies"):
                return
            rows = await call.records.list("session", {"domain": domain})
            if rows and not account:
                account = str(rows[0]["fields"].get("account") or "")
            fields = {"domain": domain, "account": account,
                      "saved_at": now_iso(), "state": state}
            if rows:
                await call.records.update("session", rows[0]["ref"], fields)
            else:
                await call.records.create("session", fields)
        except Exception:
            pass

    # -- the loop ------------------------------------------------------------
    async def loop(self, max_steps: int) -> Dict[str, Any]:
        memory, brain, live = self.brain.memory, self.brain, self.live
        framer = asyncio.get_running_loop().create_task(self._frames())
        fresh: Optional[Snapshot] = None
        pictured = ""
        look = True
        try:
            while self.steps < max_steps:
                # The person may speak, or take the browser, at any step.
                for text in self.call.screen.said():
                    brain.hear(text)
                took = getattr(self.call.screen, "took", lambda: False)()
                if took or self.call.screen.taken:
                    await self.yield_to_person()
                    fresh, look = None, True
                began = time.monotonic()
                snapshot = fresh if fresh is not None else await self.driver.snapshot()
                fresh = None
                # A picture when the outline is not enough: the run's
                # first look, a page the outline says little about, a
                # frame a person could act in, or the model asked. Not at
                # every new address: a step with a picture is never served
                # from the provider's cache, and the outline holds the page.
                sparse = len(snapshot.elements) < 3 and len(snapshot.outline) < 300
                if look or not pictured or sparse or snapshot.frames:
                    snapshot.image = await self.driver.picture()
                    pictured = snapshot.url
                look = False
                self.browser_ms += round((time.monotonic() - began) * 1000)
                self.last = snapshot
                memory.seen(snapshot.url)
                # While a sign-in is on the page, what the page fetches is
                # listed and never opened.
                self.driver.inspector.guarded = any(
                    e.get("tag") == "input" and str(e.get("type") or "").lower() == "password"
                    for e in snapshot.elements)
                again = self.idle >= BrowseTool.IDLE_AFTER or bool(brain.circling(memory.actions))
                if again:
                    if self.agains >= BrowseTool.AGAIN_MOST:
                        ended = await self.stuck(snapshot)
                        if ended is not None:
                            return ended
                        self.idle, self.agains, fresh, look = 0, 0, None, True
                        continue
                    self.agains += 1
                    self.idle = 0
                    if not snapshot.image:
                        snapshot.image = await self.driver.picture()

                self.steps += 1
                live.set(step=self.steps, status="thinking", now="Deciding what to do")
                self._figures()
                decision = await brain.decide(self.goal, snapshot, self.steps, max_steps, self.notes, again=again)
                self.notes = []
                model_ms = brain.meter.last_ms
                live.plan(brain.plan.steps)
                live.set(next=decision.get("next", ""), status="acting")
                ended, fresh, look = await self.batch(decision["actions"], snapshot, model_ms)
                if ended is not None:
                    return ended
                self._figures()
            summary = f"Ran out of steps ({max_steps}) at {self.last.url if self.last else ''}."
            return {"outcome": "budget", "summary": summary}
        finally:
            framer.cancel()

    async def batch(self, actions: List[Dict[str, Any]], snapshot: Snapshot,
                    model_ms: int) -> Tuple[Optional[Dict[str, Any]], Optional[Snapshot], bool]:
        """One answer's actions, in order, until one needs the model's
        eyes. (A result that ends the run, the page as the last action
        left it when that was looked at, whether the model asked for a
        picture.)"""
        memory, live = self.brain.memory, self.live
        before, look = snapshot, False
        for index, action in enumerate(actions):
            what = str(action.get("do") or "").lower()
            line = self._line(action, before)
            await self.call.progress(line)
            live.set(now=line)
            self.unreadable = self.unreadable + 1 if what == "invalid" else 0
            self.refusals = self.refusals + 1 if what == "invalid" and action.get("refused") else 0
            if self.refusals >= BrowseTool.REFUSED_STOP:
                return {"outcome": "failed",
                        "summary": f"The model's provider refused to answer about {before.url} "
                                   f"{self.refusals} times in a row (its content filter, most likely), after "
                                   f"{self.steps} steps; the page's content is what it would not take."}, None, False
            if self.unreadable >= BrowseTool.UNREADABLE_STOP:
                how = ("were cut off before they ended" if action.get("cut")
                       else "could not be read as actions")
                return {"outcome": "failed",
                        "summary": f"The model's last {self.unreadable} answers {how}, at "
                                   f"{before.url} after {self.steps} steps."}, None, False
            mark = self._mark(what, action, before)
            if mark in self.idle_marks:
                # Done on this very page a moment ago, and it changed
                # nothing: it is not done again.
                self.idle += 1
                self.refused_repeats += 1
                result = ("not done: the same action changed nothing on this same page a moment ago — "
                          "do something else")
                memory.did(line, result, "refused")
                live.acted(len(memory.actions), what, self._target(action, before), result, "refused", 0,
                           model_ms if index == 0 else 0)
                return None, before, look
            began = time.monotonic()
            verdict = ""
            try:
                ended, result, verdict = await self.act(what, action, before)
            except (DriverError, InspectError) as exc:
                elapsed = round((time.monotonic() - began) * 1000)
                self.browser_ms += elapsed
                memory.did(line, f"failed: {exc}", "failed")
                live.acted(len(memory.actions), what, self._target(action, before), f"failed: {exc}", "failed",
                           elapsed, model_ms if index == 0 else 0)
                self.idle += 1
                return None, None, look
            if ended is not None:
                return ended, None, look
            state, stop = "ok", what in BrowseTool.READING
            after: Optional[Snapshot] = None
            if what in BrowseTool.MOVING and result is not None:
                after = await self.driver.snapshot()
                changed = changes(before, after)
                result = changed["said"] if result == "ok" else f"{result}; {changed['said']}"
                if changed["moved"]:
                    self.idle, self.agains = 0, 0
                else:
                    state = "no_effect"
                    self.idle += 1
                    self.idle_marks.add(mark)
                stop = stop or not changed["moved"] or after.url != before.url \
                    or "dialog" in changed["said"] or len(after.tabs) != len(before.tabs)
                memory.seen(after.url)
                await self.show()
            elif result is None:
                # Refused by the person or the rules: said already.
                state, stop, result = "refused", True, ""
            elapsed = round((time.monotonic() - began) * 1000)
            self.browser_ms += elapsed
            if result:
                memory.did(line, result, state)
            live.acted(len(memory.actions), what, self._target(action, before), result or "not done",
                       state, elapsed, model_ms if index == 0 else 0, verdict)
            live.send()
            if what == "look":
                look = True
            if after is not None:
                before = after
            if stop:
                return None, (after if after is not None and what != "look" else None), look
        return None, (before if before is not snapshot else None), look

    @staticmethod
    def _mark(what: str, action: Dict[str, Any], snapshot: Snapshot) -> str:
        """What an action is, on the page it is done on: its kind, its
        control by what it says, what it carries, and the page's own
        signature."""
        element = snapshot.element(int(action.get("n") or 0)) if action.get("n") else None
        target = "|".join(str((element or {}).get(k) or "") for k in ("tag", "kind", "text", "label", "href")) \
            if element else str(action.get("n") or "")
        carried = "|".join(str(action.get(k) or "") for k in ("text", "option", "key", "direction", "url", "query"))
        return f"{what}#{target}#{carried}#{snapshot.signature}"

    @staticmethod
    def _target(action: Dict[str, Any], snapshot: Snapshot) -> str:
        n = int(action.get("n") or 0)
        element = snapshot.element(n) if n else None
        if element:
            words = element.get("text") or element.get("label") or ""
            return f"[{n}] {element.get('kind') or element.get('tag')} '{words}'"
        for key in ("url", "key", "direction", "query", "question", "reason", "why"):
            if action.get(key):
                return str(action[key])[:160]
        return ""

    def _figures(self) -> None:
        meter = self.brain.meter
        self.live.count(steps=self.steps, model_calls=meter.calls, model_ms=meter.ms, browser_ms=self.browser_ms,
                        cards=self.cards, repeats_refused=self.refused_repeats)

    async def _frames(self) -> None:
        """A frame at least every RUN_FRAME_SECONDS while the run works:
        the page moves while the model thinks."""
        while True:
            await asyncio.sleep(RUN_FRAME_SECONDS)
            try:
                if not self.driving:
                    await self.show()
            except Exception:
                pass

    async def stuck(self, snapshot: Snapshot) -> Optional[Dict[str, Any]]:
        """The run got nowhere, and another way did not help. The person
        is offered the browser, once for a page: what stops a script is
        often a thing a person sees at a glance. When they decline, or
        nobody is there to take it, the run ends."""
        failed = {"outcome": "failed",
                  "summary": f"The run got nowhere at {snapshot.url} after {self.steps} steps: its last "
                             f"actions changed nothing, and another way did not either."}
        if snapshot.signature in self.offered:
            return failed
        self.offered.append(snapshot.signature)
        took = await self.handoff("The last actions changed nothing on this page, and the way on is not clear "
                                  "from here.")
        return None if took else failed

    async def act(self, what: str, action: Dict[str, Any],
                  snapshot: Snapshot) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
        """One action: (a result that ends the run, what it gave — None
        when it was not done — and the judge's verdict on it)."""
        driver, memory, inspector = self.driver, self.brain.memory, self.driver.inspector
        line = self._line(action, snapshot)
        result = "ok"
        n = int(action.get("n") or 0)
        element = snapshot.element(n) if n else None
        if n and element is None and what in ("click", "type", "select", "hover", "read", "download", "upload",
                                              "source"):
            raise DriverError(f"there is no control [{n}] on this page now; the outline has the numbers")

        verdict = ""
        focused = None
        if what == "press":
            focused = snapshot.element(await self._focused())
        if Judge.asked_for(what, action, element, focused):
            ended, verdict, allowed = await self.judged(what, action, element, snapshot, line)
            if ended is not None:
                return ended, None, verdict
            if not allowed:
                return None, None, verdict

        if what == "click":
            await driver.click(n)
        elif what == "type":
            await driver.type(n, str(action.get("text") or ""), bool(action.get("submit")),
                              keyed=str((element or {}).get("kind") or "") in Driver.KEYED)
        elif what == "select":
            await driver.select(n, str(action.get("option") or ""))
        elif what == "press":
            await driver.press(str(action.get("key") or "Enter"))
        elif what == "scroll":
            await driver.scroll(str(action.get("direction") or "down"))
        elif what == "hover":
            await driver.hover(n)
        elif what == "goto":
            await driver.goto(str(action.get("url") or ""))
        elif what == "back":
            await driver.back()
        elif what == "tab":
            await driver.tab(str(action.get("action") or "new"), action.get("index"), str(action.get("url") or ""))
        elif what == "look":
            result = "a screenshot comes with your next turn"
        elif what == "read":
            text = await driver.read(n)
            result = f"read {len(text)} characters:\n" + cut(
                text, self.TAKEN_CHARS, "read of one element gives less")
        elif what == "remember":
            records = action.get("records")
            if not isinstance(records, list):
                records = [action.get("record")]
            kept = memory.keep(records)
            result = (f"kept {kept} ({len(memory.records)} records)" if kept
                      else "nothing to keep: give a record object, or records as a list")
        elif what == "recall":
            result = memory.recall(str(action.get("query") or ""), memory.share("actions"))
        elif what == "source":
            text = await inspector.source(n)
            result = f"the HTML of {'element ' + str(n) if n else 'the page'}, {len(text)} characters:\n" + cut(
                text, self.TAKEN_CHARS, "source of one element gives less")
        elif what == "network":
            lines = inspector.network(str(action.get("filter") or ""))
            result = (f"{len(lines)} fetched by this page (number, method, status, kind, address):\n"
                      + cut("\n".join(lines), self.TAKEN_CHARS, "a filter gives fewer")) if lines \
                else "nothing this page fetched matches; without a filter, all of it is listed"
        elif what == "response":
            text = await inspector.response(int(action.get("id") or 0))
            result = f"{len(text)} characters came back:\n" + cut(
                text, self.TAKEN_CHARS, "request with a narrower address gives less")
        elif what == "console":
            lines = inspector.console()
            result = cut("\n".join(lines), self.TAKEN_CHARS) if lines else "the page logged nothing"
        elif what == "request":
            key = self.live.pend("request", f"GET {str(action.get('url') or '')[:120]}",
                                 str(action.get("url") or ""), state="running")
            try:
                got = await inspector.request(str(action.get("url") or ""))
            finally:
                self.live.settle(key)
            result = (f"{got['status']} {got['type']} from {got['address']}, {got['size']} characters:\n"
                      + cut(str(got["text"]), self.TAKEN_CHARS, "a narrower address gives less"))
        elif what == "extract":
            result = await self.extract(action, snapshot)
        elif what == "script":
            result = await self.script(action, snapshot)
        elif what == "login":
            result = await self.login(action, snapshot)
        elif what == "code":
            # The older way to ask for a one-time code: one field, asked
            # every time.
            result = await self.login({"fields": [{"n": n, "name": "otp", "once": True,
                                                   "label": "One-time code"}]}, snapshot)
        elif what == "download":
            filename, content = await driver.download(n)
            # Returned to the chat with the result, as the chat's file.
            self.downloads.append({"filename": filename, "file_ref": self.call.give(filename, content),
                                   "size": len(content)})
            result = f"downloaded {filename} ({len(content)} bytes); it goes to the person with the result"
        elif what == "upload":
            # A file the person handed this run, by its path in the home.
            given = str(action.get("file") or "")
            if given not in self.brain.files:
                result = f"nothing was uploaded: {given or 'no file'} is not one of FILES"
            else:
                path = await self.call.file(given)
                await driver.upload(n, path.name, path.read_bytes())
                result = f"uploaded {path.name} ({path.stat().st_size} bytes)"
        elif what == "ask":
            choices = [str(c)[:self.CHOICE_CHARS] for c in (action.get("choices") or [])][:8]
            question = str(action.get("question") or "Which one?")[: self.CARD_CHARS]
            answer = await self._ask(question, choices or None)
            result = f"the person answered: {answer}" if answer is not None else "nobody answered"
        elif what == "handoff":
            if not await self.handoff(str(action.get("reason") or "the page needs a person")):
                return {"outcome": "stopped", "summary": "Stopped at a handoff the person did not complete."}, \
                    None, ""
            result = "the person handed the browser back"
        elif what == "finish":
            return await self.finish(action, snapshot, line), None, ""
        elif what == "invalid":
            if action.get("refused"):
                result = ("the model's provider refused to answer (its content filter): nothing was done; "
                          "what is on this page may be what it refuses")
            elif action.get("cut"):
                result = ("your answer was cut off before it ended, so nothing was done: "
                          "answer with less — fewer and shorter records in one remember")
            else:
                result = ("your answer could not be read as one JSON object with actions, "
                          "so nothing was done: answer with the JSON object only")
        else:
            result = f"unknown action {what!r}; use one of the actions listed"
        return None, result, verdict

    async def finish(self, action: Dict[str, Any], snapshot: Snapshot, line: str) -> Optional[Dict[str, Any]]:
        """The run's end, as the model claims it: a "done" is checked
        against the page, and one the check refuses is told back with
        what is missing — until it has been refused twice, when it is
        taken, said as not confirmed."""
        memory = self.brain.memory
        summary = str(action.get("summary") or "").strip()
        claimed = str(action.get("outcome") or "done")
        if claimed != "done":
            return {"outcome": "failed", "summary": summary}
        self.live.set(status="thinking", now="Checking the page shows the goal done")
        check = await self.driver.snapshot()
        evidence = " ".join(str(action.get("evidence") or "").split())
        if len(evidence) >= 12 and evidence.lower() in " ".join(check.text.split()).lower():
            # The words the model gave as its evidence are on the page as
            # it is now, word for word: nothing for a check to add.
            return {"outcome": "done", "summary": summary}
        if len(check.elements) < 3 and len(check.outline) < 300:
            check.image = await self.driver.picture()
        verdict = await self.brain.verify(self.goal, summary, str(action.get("evidence") or ""), check)
        if verdict["done"]:
            return {"outcome": "done", "summary": verdict["summary"]}
        self.refused_finishes += 1
        if self.refused_finishes > self.REFUSED_FINISH_MOST:
            return {"outcome": "done", "summary": f"{summary} (Not confirmed on the final page: "
                                                  f"{verdict['missing'] or verdict['summary']})"}
        missing = verdict["missing"] or verdict["summary"]
        memory.did(line, f"not finished: the page does not show it yet — {missing}", "refused")
        self.notes.append(f"the check of the page says the goal is not done yet: {missing}. Do what is "
                          f"missing, or finish as failed if it cannot be done.")
        return None

    async def _focused(self) -> int:
        try:
            return int(await self.driver.page.evaluate(
                "() => Number((document.activeElement && document.activeElement.getAttribute('data-dai-mark')) || 0)"))
        except Exception:
            return 0

    async def _ask(self, question: str, choices: Optional[List[str]]) -> Optional[str]:
        """A card in the chat, counted."""
        self.cards += 1
        self._figures()
        self.live.set(status="waiting")
        self.live.send(force=True)
        try:
            return await self.call.ask(question, choices=choices)
        finally:
            self.live.set(status="acting")

    # -- the guarded moments ---------------------------------------------------
    async def judged(self, what: str, action: Dict[str, Any], element: Optional[Dict[str, Any]],
                     snapshot: Snapshot, line: str) -> Tuple[Optional[Dict[str, Any]], str, bool]:
        """A press, judged before it lands. One that commits — or one
        that changes anything, when the person asked only to look —
        stops for their say, on the platform's own card. Nothing else
        approves a press: not a question the mind asked on its own, not
        a word in an answer. (A result that ends the run, the verdict,
        whether the press may go.)"""
        n = int((element or {}).get("n") or 0)
        about = await self.driver.inspector.about(0 if what == "press" else n)
        if about is None and element is not None:
            about = {"control": {k: element.get(k) for k in
                                 ("tag", "type", "role", "text", "label", "name", "href")},
                     "title": snapshot.title}
        domain = domain_of(snapshot.url)
        committed = self.committed if self.committed and self.committed["domain"] == domain \
            and self.steps - self.committed["step"] <= self.CONFIRMS_WITHIN else None
        found = await self.judge.verdict(self.goal, what, action, about, snapshot.url, approved=committed)
        control = (about or {}).get("control") or {}
        words = str(control.get("text") or control.get("label") or control.get("name")
                    or (f"element {n}" if n else "the field in use")).strip()[:120]
        if what in ("press", "type"):
            words = f"Enter in {words}"
        if not Judge.stops(found, self.brain.read_only):
            if found["verdict"] == "commits" and committed:
                # It only confirms what the person approved a moment ago.
                self.approvals.append(f"{words} (confirming “{committed['words']}”)")
                self.committed = None
            return None, found["verdict"], True
        ident = f"{domain}|{what}|{words}|{((about or {}).get('form') or {}).get('action', '')}"
        if ident in self.allowed:
            self.approvals.append(f"{words} (allowed for this run)")
            return None, found["verdict"], True
        where = snapshot.url.split("?", 1)[0][:200]
        choices = ["Go ahead", "Stop"]
        if self.approved_counts.get(ident):
            # The same press, approved before on this run: the person may
            # let the rest of them through.
            choices = ["Go ahead", "Go ahead for the rest of this run", "Stop"]
        key = self.live.pend("press", f"Press “{words}”", where, found["why"])
        answer = await self._ask(f"About to press “{words}” on {where}. {found['why']} Go ahead?"[: self.CARD_CHARS],
                                 choices)
        self.live.settle(key)
        if answer not in ("Go ahead", "Go ahead for the rest of this run"):
            self.brain.memory.did(line, "the person did not approve it", "refused")
            # Their own Stop is said as that, and not as a run that
            # merely ended, where a plain "stopped" reads as a result to
            # try again. Nobody answering is the other case.
            return ({"outcome": "stopped_by_person" if answer == "Stop" else "stopped",
                     "summary": f"Stopped before pressing “{words}” on {where}, as the person asked."
                     if answer == "Stop" else
                     f"Stopped before pressing “{words}” on {where}: nobody approved it."},
                    found["verdict"], False)
        self.approved_counts[ident] = self.approved_counts.get(ident, 0) + 1
        if answer == "Go ahead for the rest of this run":
            self.allowed.add(ident)
        self.approvals.append(words)
        self.brain.memory.approved.append(f"“{words}” on {where}")
        if found["verdict"] == "commits":
            self.committed = {"words": words, "why": found["why"], "domain": domain, "step": self.steps}
        return None, found["verdict"], True

    async def extract(self, action: Dict[str, Any], snapshot: Snapshot) -> str:
        """Code that only reads, run on a copy of the page with no way
        out: not asked of the person — they see it in the inspector —
        until one is seen to change the live page, after which each one
        is asked like a script."""
        code = str(action.get("code") or "").strip()
        why = " ".join(str(action.get("why") or "").split())[:200]
        if not code:
            return "nothing was run: give the code, as the body of an async function, and the why"
        if len(code) > 4000:
            return "nothing was run: an extraction is at most 4000 characters"
        if not self.extract_trusted and not await self._allows(code, why, snapshot, reads=True):
            return "the person did not allow the extraction; do not reach for the same by another way"
        key = self.live.pend("script", "Extract (reads a copy of the page)", code, why, state="running")
        try:
            found = await self.driver.inspector.extract(code)
        finally:
            self.live.settle(key)
        said = ""
        if found["touched"]:
            self.extract_trusted = False
            said = ("\n[the live page changed while it ran — by the page itself or by the code; from now on "
                    "each extraction is asked of the person first]")
        if found["failed"]:
            return f"the extraction failed: {found['failed'][:300]}" + said
        return "the extraction returned:\n" + cut(found["answer"], self.TAKEN_CHARS) + said

    async def script(self, action: Dict[str, Any], snapshot: Snapshot) -> str:
        """JavaScript in the page, when the person allows it — asked
        every time, shown the reason and the code whole. A script can do
        whatever the person can do on the site, so the last word on one
        is theirs."""
        inspector = self.driver.inspector
        code = str(action.get("code") or "").strip()
        why = " ".join(str(action.get("why") or "").split())[:200]
        if not code:
            return "nothing was run: give the code, as the body of a function, and the why"
        if len(code) > inspector.SCRIPT_CHARS:
            return (f"nothing was run: a script of more than {inspector.SCRIPT_CHARS} characters "
                    f"is not put to the person, who must be able to read what they allow; "
                    f"extract reads without one")
        if not await self._allows(code, why, snapshot):
            return "the person did not allow the script; do not reach for the same by another way"
        self.approvals.append(f"script: {why or code[:60]}")
        self.session_dirty = True
        key = self.live.pend("script", "Script (acts in the page)", code, why, state="running")
        try:
            return "the script returned:\n" + cut(await inspector.script(code), self.TAKEN_CHARS)
        finally:
            self.live.settle(key)

    async def _allows(self, code: str, why: str, snapshot: Snapshot, reads: bool = False) -> bool:
        """The person's answer to a script, asked in the chat with the
        code whole and the reason beside it."""
        where = snapshot.url.split("?", 1)[0][:120]
        kind = "this extraction (it reads a copy of the page)" if reads else "this script"
        key = self.live.pend("script", "Waiting for your approval", code, why)
        answer = await self._ask(
            f"Run {kind} on {where}?\nWhy: {why or 'no reason was given'}\n\n{code}"[: self.CARD_CHARS],
            ["Run it", "Stop"])
        self.live.settle(key)
        return answer == "Run it"

    def _own_fields(self, action: Dict[str, Any], snapshot: Snapshot) -> List[Dict[str, Any]]:
        """The fields on this page that take the person's own values,
        as the model named them — whatever the form's shape. With
        nothing named, the form's own password field and the field
        before it."""
        said = [dict(f) for f in (action.get("fields") or []) if isinstance(f, dict)]
        if not said:
            marks = {name: int(action.get(name) or 0) for name in ("username", "password")}
            if not any(marks.values()):
                marks = guards.login_fields(snapshot.elements)
            said = [{"n": marks[name], "name": name}
                    for name in ("username", "password") if marks.get(name)]
        fields, seen = [], set()
        for item in said:
            element = snapshot.element(int(item.get("n") or 0))
            if element is None:
                continue
            field = guards.field_of(element, item)
            if field["name"] not in seen:
                seen.add(field["name"])
                fields.append(field)
        return fields

    async def login(self, action: Dict[str, Any], snapshot: Snapshot) -> str:
        """The person's own values, put into the fields the model named.
        The site's login is asked for exactly those: what it holds it
        hands over, what it lacks the person fills in on the platform's
        form — once, or every time for a code. The values reach the page
        and nothing else: not the model, not the history."""
        fields = self._own_fields(action, snapshot)
        if not fields:
            return ("nothing was filled: name the fields on this page that take the person's "
                    "own values — login {fields: [{n, name, label}]} — by their numbers")
        page = self.driver.page
        key = self.live.pend("sign_in", f"Sign-in on {domain_of(snapshot.url)}",
                             ", ".join(f["label"] for f in fields), "the person's own values, never shown")
        try:
            values = await self._values(domain_of(snapshot.url), fields)
        finally:
            self.live.settle(key)
        if values is None:
            return "the person did not provide it; do not try to sign in another way"
        if self.driver.page is not page or domain_of(page.url) != domain_of(snapshot.url):
            # The form may wait minutes: a popup or a redirect meanwhile must not be handed site A's password.
            return ("nothing was filled: the page changed while the person answered; "
                    "look at the page again and ask for the login there")
        self.account = str(values.get("username") or self.account)
        filled = []
        for field in fields:
            value = str(values.get(field["name"]) or "")
            if value:
                await self.driver.fill(field["n"], value)
                filled.append(field["label"])
                self.login_names[field["name"]] = field["label"]
        if not filled:
            return "the person's login holds nothing for these fields; nothing was filled"
        submit = action.get("submit")
        marks = self.driver._marks()
        if submit is False:
            pass  # more of the form is the model's to fill before it goes
        elif int(submit or 0):
            await self.driver.click(int(submit))
        else:
            await self.driver.page.keyboard.press("Enter")
            await self.driver._after(*marks)
        self.session_dirty = True
        return (f"filled {', '.join(filled)} from the person's own values"
                + ("" if submit is False else " and sent the form"))

    async def _values(self, site: str, fields: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """The person's values for these fields, from the site's login
        (one per site, its answers by field name): those it keeps, and
        the rest asked as its form's questions, a code every time and
        never kept. None when the person declined."""
        asked = {f["name"]: {"type": "secret" if f["secret"] else "text", "remember": not f["once"]}
                 for f in fields}
        records = self.call.records
        try:
            found = await records.list("logins", {"site": site})
            if found:
                ref = found[0]["ref"]
                answers = (await records.read("logins", ref))["fields"].get("answers") or {}
            else:
                ref, answers = await records.create("logins", {"site": site, "answers": {}}), {}
            missing = {name: question for name, question in asked.items()
                       if not question["remember"] or "value" not in (answers.get(name) or {})}
            if missing:
                kept = {name: answer for name, answer in answers.items() if name not in missing}
                answers = (await records.update("logins", ref, {"answers": {**kept, **missing}}))[
                    "fields"].get("answers") or {}
        except ResourceDenied as exc:
            if exc.code != "refused":
                raise
            return None
        return {name: str((answers.get(name) or {}).get("value") or "") for name in asked}

    async def yield_to_person(self) -> None:
        """The person took the browser in the live view without being
        asked. The run waits: frames flow, their hand is fed to the
        page, until they hand it back — then the model is told and looks
        again at the page as they left it."""
        await self.call.progress("The person took the browser")
        self.live.set(status="waiting", now="The person has the browser")
        self.live.send(force=True)
        self.driving = True
        try:
            _why, trail = await drive(
                self.call, self.driver,
                until=lambda: "" if self.call.screen.taken else "released",
                hear=self.brain.hear)
        finally:
            self.driving = False
        self.session_dirty = True
        self.idle, self.agains = 0, 0
        # What they did, in words, is the run's to learn from and the
        # site's notes to keep: the next run tries it itself.
        self._person_did(trail)
        self.brain.memory.did(
            "the person took the browser for a while",
            ("they " + "; ".join(trail[-TRAIL_SHOWN:]) + "; then handed it back — look again"
             if trail else
             "they handed it back; the page is as they left it — look again"))
        await self.call.progress("The person handed the browser back")

    async def handoff(self, reason: str) -> bool:
        """The person takes the browser over. Frames keep flowing and
        their hand is fed to the page until they press Done."""
        self.handoffs += 1
        self.cards += 1
        self._figures()
        await self.call.progress(f"Handing the browser to you: {reason}")
        self.live.set(status="waiting", now=f"Handed to you: {reason}")
        self.live.send(force=True)
        before = (await self.driver.snapshot()).signature
        asking = asyncio.ensure_future(self.call.ask(
            (f"{reason} Take over the browser in the live view, do what it needs, "
             f"then choose Done.")[: self.CARD_CHARS], choices=["Done", "Stop"]))
        self.driving = True
        try:
            _why, trail = await drive(
                self.call, self.driver,
                until=lambda: "answered" if asking.done() else "",
                hear=self.brain.hear)
        finally:
            self.driving = False
            if not asking.done():
                asking.cancel()
        answer = asking.result() if asking.done() and not asking.cancelled() else None
        self.live.set(status="acting")
        if answer == "Done":
            self.session_dirty = True
            self.idle, self.agains = 0, 0
            snapshot = await self.driver.snapshot()
            if snapshot.signature == before:
                # Nothing visibly changed in this browser: the person may
                # have done what was needed elsewhere (their phone, another
                # window). Reload so the site's new state shows.
                try:
                    await self.driver.page.reload(wait_until="domcontentloaded", timeout=15000)
                    await self.driver.quiet()
                except Exception:
                    pass
            self._person_did(trail)
            self.brain.memory.did(
                "handoff",
                ("the person took over and " + "; ".join(trail[-TRAIL_SHOWN:]) + "; then handed back"
                 if trail else "the person took over and handed back"))
            return True
        return False

    def _person_did(self, trail: List[str]) -> None:
        """What the person did themselves, on the site's notes, so the
        next run tries it itself."""
        lines = [t for t in trail if "cannot be visited" not in t]
        if lines:
            self.brain.notes.add("person", now_iso()[:10] + ": " + "; ".join(lines)[:TRAIL_NOTE_MAX])

    # -- the live view -------------------------------------------------------------
    async def show(self) -> None:
        """One frame to the live view, the size the browser's viewport
        is, with the open tabs."""
        image = await self.driver.frame()
        if not image:
            return
        size = self.driver.page.viewport_size if self.driver.page else None
        width, height = (size["width"], size["height"]) if size else VIEWPORT
        await show_frame(self.call, self.driver, image, width, height)

    # -- the end -----------------------------------------------------------------------
    async def close(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        memory = self.brain.memory
        meter = self.brain.meter
        result: Dict[str, Any] = {
            "outcome": str(outcome.get("outcome") or "failed"),
            "summary": str(outcome.get("summary") or ""),
            "steps": self.steps, "final_url": "", "records": list(memory.records),
            "approvals": list(self.approvals), "handoffs": self.handoffs,
            "downloads": list(self.downloads),
        }
        self.live.set(status={"done": "done", "budget": "stopped", "stopped": "stopped",
                              "stopped_by_person": "stopped"}.get(result["outcome"], "failed"),
                      now=result["summary"][:200], next="")
        try:
            if self.driver.page is not None:
                final = await self.driver.snapshot(look=True)
                result["final_url"] = final.url
                result["file_ref"] = BrowseTool._save_image(self.call, final, "final")
                if self.session_dirty:
                    await self.save_session()
                if self.steps:
                    notes = await self.brain.write_notes(self.domain)
                    notes.names.update(self.login_names)
                    await self.save_notes(notes)
        except Exception:
            pass
        result["figures"] = {"steps": self.steps, "model_calls": meter.calls, "model_ms": meter.ms,
                             "browser_ms": self.browser_ms, "cards": self.cards,
                             "repeats_refused": self.refused_repeats,
                             "actions": len([a for a in memory.actions if a["action"] != "the person said"]),
                             "elapsed_ms": round((time.monotonic() - self.live.began) * 1000)}
        self._figures()
        self.live.stop()
        try:
            await self.call.screen.close()
        except Exception:
            pass
        if len(result["records"]) > 1:
            columns = sorted({k for r in result["records"] for k in r if isinstance(r, dict)})[:8]
            try:
                await self.call.say(table={"title": "What was found", "columns": columns,
                                           "rows": result["records"][:500]})
            except Exception:
                pass
        return result

    @staticmethod
    def _line(action: Dict[str, Any], snapshot: Snapshot) -> str:
        what = str(action.get("do") or "?")
        if what == "invalid":
            if action.get("refused"):
                return "the model's provider refused to answer"
            return ("the model's answer was cut off before it ended" if action.get("cut")
                    else "the model's answer could not be read")
        n = action.get("n")
        element = snapshot.element(int(n)) if n else None
        words = (element or {}).get("text") or (element or {}).get("label") or ""
        parts = [what]
        if n:
            parts.append(f"[{n}] {words!r}" if words else f"[{n}]")
        for key in ("text", "url", "option", "key", "direction", "question", "reason", "outcome",
                    "query", "filter", "id", "why"):
            if action.get(key) and not (what == "login" and key == "text"):
                parts.append(f"{key}={str(action[key])[:60]!r}")
        return " ".join(parts)
