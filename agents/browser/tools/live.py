"""What the person sees of the agent's own working: the inspector.

Beside the browser's picture, the live view has a tab that cannot be
closed: what the agent does now and next, its plan, what is about to
run (a script, a press waiting for approval, a request), each action
and what it changed, what the page fetched and logged, and the run's
figures. This is that tab's state, sent as the call's ``inspector``
note: whole each time (the newest replaces the one before, so a lost
one costs nothing), at most a few times a second, and only when
something changed.

Its shape is the inspector event's (version 1): every field optional,
lists oldest first, each list capped.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

#: The newest of each list the state carries.
ACTIONS_KEPT, NETWORK_KEPT, CONSOLE_KEPT = 100, 200, 50
#: How often the state may be sent, in seconds.
EVERY = 0.25


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def short(text: Any, most: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= most else text[:most - 1] + "…"


class Live:
    """The inspector of one run."""

    def __init__(self, call, driver, goal: str, max_steps: int):
        self.call, self.driver = call, driver
        self.began = time.monotonic()
        self.state: Dict[str, Any] = {"v": 1, "status": "planning", "goal": short(goal, 500), "read_only": False,
                                      "step": 0, "max_steps": int(max_steps), "now": "", "next": "", "plan": [],
                                      "pending": [], "actions": [], "network": [], "console": [], "figures": {}}
        self._actions: List[Dict[str, Any]] = []
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._ids = 0
        self._sent_at = 0.0
        self._dirty = True
        self._task: Optional[asyncio.Task] = None
        self.figures = {"steps": 0, "model_calls": 0, "model_ms": 0, "browser_ms": 0, "cards": 0,
                        "repeats_refused": 0, "elapsed_ms": 0}

    # -- what changed --------------------------------------------------------------------
    def set(self, **fields: Any) -> None:
        for key, value in fields.items():
            if key in ("now", "next"):
                value = short(value, 200)
            self.state[key] = value
        self._dirty = True

    def plan(self, steps: List[Dict[str, str]]) -> None:
        self.state["plan"] = [{"step": short(s.get("step"), 300), "state": s.get("state", "todo")}
                              for s in steps[:20]]
        self._dirty = True

    def pend(self, kind: str, title: str, detail: str = "", why: str = "", state: str = "waiting") -> str:
        """Something about to happen: its id, to settle it by."""
        self._ids += 1
        key = f"{kind[0]}{self._ids}"
        self._pending[key] = {"id": key, "kind": kind, "title": short(title, 160), "detail": str(detail)[:2000],
                              "why": short(why, 300), "state": state}
        self._dirty = True
        self.send(force=True)
        return key

    def settle(self, key: str, state: str = "") -> None:
        """A pending item done: gone from the list, or kept with a state
        while it waits on something else."""
        if key in self._pending:
            if state:
                self._pending[key]["state"] = state
            else:
                del self._pending[key]
            self._dirty = True

    def acted(self, n: int, do: str, target: str, result: str, state: str, ms: int, model_ms: int = 0,
              verdict: str = "") -> None:
        self._actions.append({"n": int(n), "at": now_iso(), "do": short(do, 40), "target": short(target, 160),
                              "result": short(result, 300), "state": state, "ms": int(ms),
                              "model_ms": int(model_ms), "verdict": verdict})
        del self._actions[:-ACTIONS_KEPT]
        self._dirty = True

    def count(self, **figures: int) -> None:
        for key, value in figures.items():
            self.figures[key] = int(value)
        self._dirty = True

    # -- sending -------------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        state = dict(self.state)
        state["pending"] = list(self._pending.values())
        state["actions"] = list(self._actions)
        inspector = self.driver.inspector
        network = []
        for entry in inspector.history[-NETWORK_KEPT:]:
            network.append({"id": entry["id"], "at": entry.get("at", ""), "method": entry["method"],
                            "status": entry["status"] or None, "kind": entry["kind"],
                            "url": str(entry["address"])[:300], "size": entry["size"] or None,
                            "ms": entry.get("ms") or None, "by": entry.get("by", "page"),
                            "blocked": entry.get("blocked", "")})
        state["network"] = network
        state["console"] = [{"at": entry.get("at", ""),
                             "level": "warning" if entry["kind"] == "warning" else
                             "error" if entry["kind"] == "error" else "log",
                             "text": short(entry["text"], 300)} for entry in inspector.logged[-CONSOLE_KEPT:]]
        figures = dict(self.figures)
        figures["elapsed_ms"] = round((time.monotonic() - self.began) * 1000)
        state["figures"] = figures
        return state

    def send(self, force: bool = False) -> None:
        if not (force or self._dirty) or self.driver.page is None:
            return
        if not force and time.monotonic() - self._sent_at < EVERY:
            return
        try:
            inspect = getattr(self.call.screen, "inspect", None)
            if inspect is not None:
                inspect(self.snapshot())
        except Exception:
            pass
        self._sent_at = time.monotonic()
        self._dirty = False

    def start(self) -> None:
        """Sent on its own clock while the run lasts: what the page
        fetches arrives between the agent's own changes."""
        async def beat() -> None:
            seen = -1
            while True:
                await asyncio.sleep(EVERY * 2)
                fetched = self.driver.inspector._count
                if fetched != seen:
                    seen, self._dirty = fetched, True
                self.send()

        self._task = asyncio.get_running_loop().create_task(beat())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._dirty = True
        self.send(force=True)
