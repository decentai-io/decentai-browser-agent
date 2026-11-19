"""The Browser agent as revised: a press judged by what it means, a
run that remembers by budget, the page looked at beneath its picture,
and the tabs in the person's sight.

Run the way production runs it where a browser is needed — the agent's
own program, a real Chromium against the loopback shop, the model
scripted — and as plain objects where none is."""

import asyncio
import json
import re

import pytest


from tests.harness import Chat, ChatSinks, Records, run
from tests.browser_site import VISITOR
from tests.test_browser import (  # noqa: F401  (shop is a fixture)
    Person, Shopper, _progress, browse, make, run, shop)


class Scripted(Shopper):
    """A mind that takes the steps it was given, in order, and answers
    the side calls as the shopper does. A step is an action, or a
    function of the prompt that returns one — for a step that depends
    on what the last one brought back."""

    def __init__(self, steps):
        super().__init__()
        self.steps = list(steps)

    async def __call__(self, messages, max_tokens=None, pictures=None):
        text = messages[-1]["text"]
        if not text.startswith("STEP "):
            return await super().__call__(messages, max_tokens=max_tokens, pictures=pictures)
        self.prompts.append("\n\n".join(m["text"] for m in messages))
        step = self.steps.pop(0) if self.steps else {
            "do": "finish", "outcome": "failed", "summary": "The script ran out."}
        action = step(text) if callable(step) else step
        if isinstance(action, tuple):
            return action  # the provider's own answer: words and why it stopped
        actions = action if isinstance(action, list) else [action]
        return json.dumps({"actions": actions, "next": "as scripted"})


def scripted(shop, steps, asker=None):
    provider = Records()
    mind, person = Scripted(steps), Person(shop)
    executor = Chat(provider, ChatSinks(llm=mind, ask=asker or person.ask, form=person.form, progress=_progress))
    return executor, provider, mind, person


def last_actions(prompt):
    """The actions a prompt shows, as the mind reads them."""
    head = "YOUR ACTIONS AND WHAT THEY CHANGED:"
    return prompt.split(head)[-1].split("\n\nPAGE:")[0].split("\n\nNOTE:")[0] if head in prompt else ""


class TestTheJudge:
    def test_whether_a_press_is_judged_is_a_matter_of_what_it_is(self):
        """Not of any word on it. A plain link is followed unjudged —
        a read, by the protocol's rule — unless the mind itself expects
        more of it; a button, a form being sent, a choice in a list
        and a file put in are judged always; reading and scrolling are
        not presses."""
        from agents.browser.tools.judge import Judge

        link = {"tag": "a", "href": "/delete/5", "text": "حذف"}
        button = {"tag": "button", "text": "Continue"}
        assert not Judge.asked_for("click", {"n": 1}, link)
        assert not Judge.asked_for("click", {"n": 1, "effect": "looks"}, link)
        assert Judge.asked_for("click", {"n": 1, "effect": "commits"}, link)
        assert Judge.asked_for("click", {"n": 1, "effect": "looks"}, button)
        assert Judge.asked_for("click", {"n": 1}, {"tag": "a", "href": "#", "text": "Pay"})
        assert Judge.asked_for("click", {"n": 1}, {"tag": "a", "href": "javascript:pay()"})
        assert Judge.asked_for("click", {"n": 1}, {"tag": "a", "role": "button", "href": "/x"})
        assert Judge.asked_for("type", {"n": 1, "text": "dumbbell", "submit": True}, None)
        assert not Judge.asked_for("type", {"n": 1, "text": "dumbbell"}, None)
        assert Judge.asked_for("press", {"key": "Enter"}, None)
        assert not Judge.asked_for("press", {"key": "Escape"}, None)
        # Whatever else may send: a line break typed, Ctrl+Enter, Space on a button, a site's shortcut.
        assert Judge.asked_for("type", {"n": 1, "text": "see you\n"}, None)
        assert Judge.asked_for("type", {"n": 1, "text": "see you\r"}, None)
        for key in ("Control+Enter", "Meta+Enter", "Space", "e", "Delete"):
            assert Judge.asked_for("press", {"key": key}, None), key
        for key in ("Tab", "Shift+Tab", "ArrowDown", "PageDown", "End"):
            assert not Judge.asked_for("press", {"key": key}, None), key
        # A search, and a form sent by GET, only ask for a page; so does
        # a choice in a list of no form. A form sent by POST is judged.
        search = {"tag": "input", "type": "search", "kind": "searchbox", "form": "get"}
        assert not Judge.asked_for("type", {"n": 1, "text": "dumbbell", "submit": True}, search)
        assert not Judge.asked_for("type", {"n": 1, "text": "x", "submit": True}, {"tag": "input", "form": "get"})
        assert Judge.asked_for("type", {"n": 1, "text": "x", "submit": True}, {"tag": "input", "form": "post"})
        assert not Judge.asked_for("press", {"key": "Enter"}, None, focused=search)
        assert not Judge.asked_for("select", {"n": 1, "option": "x"}, {"tag": "select", "form": ""})
        assert Judge.asked_for("select", {"n": 1, "option": "x"}, {"tag": "select", "form": "post"})
        assert Judge.asked_for("upload", {"n": 1}, {"tag": "input", "type": "file"})
        for reading in ("read", "scroll", "hover", "goto", "source", "network", "request", "remember"):
            assert not Judge.asked_for(reading, {"n": 1}, button)

    def test_an_answer_that_cannot_be_read_is_the_careful_one(self):
        from agents.browser.tools.judge import Judge

        said = Judge.read('Here you go: {"verdict": "Changes", "why": "It puts the item in the basket."}')
        assert said == {"verdict": "changes", "why": "It puts the item in the basket.", "continues": False}
        for unreadable in ("", "I think it is fine.", '{"verdict": "harmless"}', "{not json"):
            assert Judge.read(unreadable)["verdict"] == "commits"
        # Whether the person asked only to look is the run's, decided
        # once from the goal; the judge says what the press does.
        assert Judge.stops({"verdict": "commits"}, read_only=False)
        assert Judge.stops({"verdict": "changes"}, read_only=True)
        assert not Judge.stops({"verdict": "changes"}, read_only=False)
        assert not Judge.stops({"verdict": "looks"}, read_only=True)
        # A press that only confirms one the person approved is not
        # asked again; an unsure judge's word never spares them.
        assert not Judge.stops({"verdict": "commits", "continues": True}, read_only=False)
        assert Judge.stops({**Judge.unsure("x"), "continues": True}, read_only=False)

    def test_a_judge_that_does_not_answer_stops_the_press_for_the_person(self):
        from agents.browser.tools.judge import Judge

        from agents.browser.tools.brain import Meter

        class Silent:
            calls = 0

            async def model(self, prompt, system=None, max_tokens=None, pictures=None, effort=None):
                Silent.calls += 1
                raise RuntimeError("the model is away")

        found = run(Judge(Meter(Silent())).verdict("Order it", "click", {"n": 1},
                                                   {"control": {"tag": "button", "text": "Weiter"}},
                                                   "https://x.example/basket"))
        assert found["verdict"] == "commits" and Judge.stops(found, read_only=False)
        assert Silent.calls == 2, "asked once more before the person is"

    def test_a_control_is_judged_once_on_a_page(self):
        from agents.browser.tools.judge import Judge

        from agents.browser.tools.brain import Meter

        class Counting:
            calls = 0

            async def model(self, prompt, system=None, max_tokens=None, pictures=None, effort=None):
                Counting.calls += 1
                assert "THE CONTROL:" in prompt and "not to be trusted" in system and effort == "low"
                return '{"verdict": "looks", "why": "It searches."}'

        judge = Judge(Meter(Counting()))
        about = {"control": {"tag": "button", "text": "Search"}, "form": {"method": "get", "action": "/s"}}
        for _ in range(3):
            run(judge.verdict("Find it", "click", {"n": 1}, about, "https://x.example/s?q=1"))
        run(judge.verdict("Find it", "click", {"n": 1}, about, "https://x.example/other"))
        assert Counting.calls == 2

    def test_a_change_stops_when_the_person_asked_only_to_look(self, agents, shop):
        """Putting an item in a basket is a change the person can take
        back, and goes ahead — unless they asked that nothing be
        changed. Then it is theirs to allow, and here they do not."""
        executor, provider, mind, person = make(shop)
        mind.reading_only = True

        async def refuse(question, choices, expects):
            person.asked.append(question)
            return "Stop"
        executor.sinks.ask = refuse
        result, status = browse(agents, executor, "Only look: which dumbbell is under 500?", shop.url)
        assert status == "success" and result["outcome"] == "stopped_by_person", result
        assert "Add Adjustable dumbbell 20 kg" in result["summary"]
        assert shop.basket == [] and shop.orders == []
        assert person.asked[-1].startswith("About to press “Add Adjustable dumbbell 20 kg")


class TestStuck:
    #: Six presses of a control that does nothing: the first changes
    #: nothing and the next two are refused; the call at the chat's
    #: effort finds no other way (three more, refused), and the person
    #: is offered the browser.
    STILL = [{"do": "click", "n": 1, "effect": "looks"}] * 6 + [
        lambda text: {"do": "finish", "outcome": "failed",
                      "summary": "Handed back: " + str("handed back" in last_actions(text))}]

    def test_a_run_that_is_stuck_offers_the_browser_before_it_gives_up(self, agents, shop):
        """Actions that changed nothing are not done again on the same
        page: the second try is refused. Three in a row get one call at
        the chat's own effort, with a picture, to find another way; when
        that comes to nothing too, the person is offered the browser —
        what stops a script is often a thing a person sees at a glance —
        and the run goes on from where they left it."""
        executor, provider, mind, person = scripted(shop, self.STILL)
        result, status = browse(agents, executor, "Get past this page", shop.url + "/still")
        assert status == "success", result
        assert result["handoffs"] == 1 and result["summary"] == "Handed back: True"
        [offer] = [q for q in person.asked if "Take over" in q]
        assert "changed nothing" in offer
        assert result["figures"]["repeats_refused"] == 5
        again = [p for p in mind.prompts if "the way you were going is not working" in p]
        assert len(again) == 1
        assert None in executor.efforts, "finding another way thinks at the chat's own effort"
        assert any("not done: the same action changed nothing" in p for p in mind.prompts)

    def test_declined_it_ends_as_it_always_did(self, agents, shop):
        async def decline(question, choices, expects):
            return "Stop"
        executor, provider, mind, person = scripted(shop, self.STILL, asker=decline)
        result, status = browse(agents, executor, "Get past this page", shop.url + "/still")
        assert status == "success" and result["outcome"] == "failed", result
        assert "got nowhere" in result["summary"]


class TestWhatItRemembers:
    def test_a_cut_is_said_with_how_to_get_the_rest(self):
        from agents.browser.tools.memory import cut

        assert cut("short", 100) == "short"
        said = cut("x" * 500, 100, "read 0 gives the whole page")
        assert said.startswith("x" * 100) and "400 more characters" in said
        assert "read 0 gives the whole page" in said

    def test_what_is_shown_is_a_share_not_a_count(self):
        """Forty actions are all kept. The newest are shown whole, the
        older ones are said to be there, and what is shown stays within
        its share however long the run."""
        from agents.browser.tools.memory import WorkingMemory

        memory = WorkingMemory()
        for number in range(1, 41):
            memory.did(f"read page {number}", f"client {number}: " + "row " * 1500)
        shown = memory.actions_shown()
        assert len(memory.actions) == 40
        assert "read page 40" in shown and "client 40" in shown
        assert "older actions are not shown: recall finds them" in shown
        assert "cut here" in shown
        assert len(shown) < memory.share("actions") * 1.5
        # What left sight is found again.
        assert "client 3:" in memory.recall("client 3:", 4000)
        assert memory.recall("nothing like this", 4000).startswith("nothing kept holds")

    def test_the_ledger_keeps_facts_and_hands_them_on(self):
        """Pages visited and presses approved are kept by the code, never
        summarised away; a run tells the next one in its chat what it
        did."""
        from agents.browser.tools.memory import WorkingMemory

        memory = WorkingMemory()
        for page in ("https://x.example/", "https://x.example/a", "https://x.example/", "about:blank"):
            memory.seen(page)
        memory.approved.append("“Place order” on https://x.example/basket")
        memory.did("click [3] 'Place order'", "the address is now https://x.example/ordered")
        memory.keep([{"kind": "item", "name": "a"}])
        ledger = memory.ledger_shown()
        assert "https://x.example/a | https://x.example/" in ledger and "about:" not in ledger
        assert "Place order" in ledger
        handed = memory.handed_on("Order it", "done", "Ordered.")
        assert handed.startswith("Goal: Order it\nOutcome: done: Ordered.")
        assert "1 records kept" in handed and "click [3] 'Place order'" in handed

    def test_records_are_all_kept_and_counted_by_kind(self):
        from agents.browser.tools.memory import WorkingMemory

        memory = WorkingMemory()
        kept = memory.keep([{"kind": "ticket", "key": f"MSP-{n}", "note": "n" * 400}
                            for n in range(1, 121)] + [None, {}, "not a record"])
        assert kept == 120 and len(memory.records) == 120
        shown = memory.records_shown()
        assert shown.startswith("120 kept: 120 ticket")
        assert "MSP-120" in shown and "MSP-1\"" not in shown and "recall finds the others" in shown
        assert "MSP-7\"" in memory.recall("MSP-7\"", 4000)

    def test_the_plan_says_where_each_step_stands(self):
        from agents.browser.tools.memory import Plan

        plan = Plan()
        plan.take(["Client A", "Client B", "Client C"])
        plan.take([{"step": "Client A", "state": "done"}, {"step": "Client B", "state": "doing"},
                   "Client C", {"step": "Client D", "state": "nonsense"}])
        assert plan.shown() == ("1. [done] Client A\n2. [doing] Client B\n"
                                "3. [todo] Client C\n4. [todo] Client D")
        plan.take(["Client A", "Client B"])      # a state once said stays
        assert plan.shown() == "1. [done] Client A\n2. [doing] Client B"
        plan.take("not a list")
        assert len(plan.steps) == 2

    def test_a_sites_notes_are_kept_by_section(self):
        from agents.browser.tools.memory import SiteNotes

        notes = SiteNotes("Search is at the top; Place order is on /basket.")   # from before sections
        assert notes.sections["finding"].startswith("Search is at the top")
        notes.write("signing_in", "Email first, the password on the next page.")
        notes.write("avoid", "")                          # nothing said keeps what was there
        notes.add("person", "2026-09-28: clicked 'Sign in' (button)")
        notes.names["username"] = "Email"
        again = SiteNotes(notes.text())
        assert again.sections == notes.sections and again.names == {"username": "Email"}
        assert again.text().index("SIGNING IN") < again.text().index("FINDING THINGS")
        notes.write("finding", "x" * 5000)
        assert len(notes.sections["finding"]) == SiteNotes.SECTION_CHARS
        assert notes.sections["signing_in"].startswith("Email first")


class TestLookingBeneath:
    def test_what_a_page_fetched_is_read_as_data(self, agents, shop):
        """A page that lists things has fetched them as data. The mind
        finds that address among what the page fetched, reads what came
        back, asks the same address a narrower question with the
        person's session — and never sees the session itself."""
        def open_what_was_fetched(text):
            listed = re.search(r"^(\d+)  GET 200  \S+ application/json.*?/api/items\?under=1000$",
                               last_actions(text), re.M)
            return {"do": "response", "id": int(listed.group(1))}

        def keep_what_came_back(text):
            body = re.search(r"characters:\n(\{.*\})", last_actions(text)).group(1)
            data = json.loads(body)
            return {"do": "remember", "records": [
                {"kind": "item", "name": item["name"], "price": item["price"],
                 "known_visitor": data["known_visitor"]} for item in data["items"]]}

        executor, provider, mind, person = scripted(shop, [
            {"do": "network", "filter": "api"},
            open_what_was_fetched,
            {"do": "request", "url": "/api/items?under=200"},
            keep_what_came_back,
            {"do": "request", "url": "https://elsewhere.example/api/items"},
            {"do": "console"},
            {"do": "source", "n": 0},
            {"do": "finish", "outcome": "done", "summary": "Read the list as data."},
        ])
        result, status = browse(agents, executor, "List what costs under 200", shop.url + "/catalog")
        assert status == "success" and result["outcome"] == "done", result
        # Read from the address the page itself uses, with its session.
        assert result["records"] == [{"kind": "item", "name": "Hex dumbbell 10 kg pair",
                                      "price": 140, "known_visitor": True}]
        assert shop.api_calls == ["/api/items?under=1000", "/api/items?under=200"]
        # What the run showed of the page, without the instructions every call carries.
        from agents.browser.tools.brain import SYSTEM
        seen = "\n".join(p.replace(SYSTEM, "") for p in mind.prompts)
        assert "Adjustable dumbbell 30 kg" in seen          # the first answer, opened
        assert "is not of the site the page is on" in seen  # another site is not read
        assert "catalog: 3 items drawn" in seen             # the console
        assert 'data-source="/api/items"' in seen           # the HTML, attributes and all
        assert "console.log" not in seen                    # and nothing that runs
        # The session is the person's: no cookie, no header, in any prompt.
        assert VISITOR not in seen and "Set-Cookie" not in seen and "cookie" not in seen.lower()

    def test_what_was_fetched_during_a_sign_in_is_listed_and_not_opened(self, agents, shop):
        def open_the_first(text):
            listed = re.search(r"^(\d+)  GET", last_actions(text), re.M)
            return {"do": "response", "id": int(listed.group(1))}

        executor, provider, mind, person = scripted(shop, [
            {"do": "network"}, open_the_first,
            lambda text: {"do": "finish", "outcome": "failed",
                          "summary": "Refused: " + str("is not opened" in last_actions(text))}])
        result, status = browse(agents, executor, "Look at the sign-in", shop.url + "/login")
        assert status == "success" and result["summary"] == "Refused: True", result

    def test_a_script_runs_only_when_the_person_allows_it(self, agents, shop):
        """A script can do whatever the person can do on the site, so
        each one is theirs to allow, asked in the chat: the code whole,
        its reason, and the page it runs on. Allowed, its answer comes
        back; refused, or with nobody to answer, it never ran."""
        code = ("await fetch('/add', {method: 'POST', body: new URLSearchParams({id: 'd20'})});\n"
                "return document.title;")
        steps = [
            {"do": "script", "why": "To put the 20 kg dumbbell in the basket", "code": code},
            lambda text: {"do": "finish", "outcome": "failed", "summary": last_actions(text)[-200:]},
        ]
        asked = []

        async def allow(question, choices, expects):
            asked.append((question, choices, expects))
            return "Run it"

        async def refuse(question, choices, expects):
            return "Stop"

        async def nobody(question, choices, expects):
            return None

        for unwilling in (refuse, nobody):
            executor, provider, mind, person = scripted(shop, list(steps), asker=unwilling)
            result, status = browse(agents, executor, "Put it in the basket", shop.url + "/catalog")
            assert status == "success" and "the person did not allow the script" in result["summary"]
            assert shop.basket == [] and result["approvals"] == []

        executor, provider, mind, person = scripted(shop, list(steps), asker=allow)
        result, status = browse(agents, executor, "Put it in the basket", shop.url + "/catalog")
        assert status == "success" and '"Catalog — Sidra Fitness"' in result["summary"], result
        [(question, choices, expects)] = asked
        assert (choices, expects) == (["Run it", "Stop"], "choice")
        assert code in question and "Why: To put the 20 kg dumbbell in the basket" in question
        assert f"{shop.url}/catalog" in question
        assert result["approvals"] == ["script: To put the 20 kg dumbbell in the basket"]

    def test_a_script_is_put_to_the_person_as_a_plain_question(self):
        """The script is put to the person as words, and their word
        decides; the page is named without its query."""
        from types import SimpleNamespace

        from agents.browser.tools.browse_tool import _Run

        asked = []

        class Call:
            async def ask(self, question, choices=None, expects=None):
                asked.append((question, choices))
                return "Run it" if len(asked) == 1 else "Stop"

        run = _Run.__new__(_Run)
        run.call = Call()
        run.cards, run.steps, run.browser_ms, run.refused_repeats = 0, 0, 0, 0
        run.brain = SimpleNamespace(meter=SimpleNamespace(calls=0, ms=0))
        run.live = SimpleNamespace(pend=lambda *a, **k: "s1", settle=lambda *a, **k: None,
                                   set=lambda **k: None, send=lambda **k: None, count=lambda **k: None)
        page = SimpleNamespace(url="https://shop.example.com/catalog?page=2")
        allowed = asyncio.run(run._allows("return 1;", "To count", page))
        refused = asyncio.run(run._allows("return 1;", "To count", page))
        assert (allowed, refused) == (True, False)
        question, choices = asked[0]
        assert choices == ["Run it", "Stop"]
        assert "https://shop.example.com/catalog" in question and "page=2" not in question
        assert "Why: To count" in question and "return 1;" in question

    def test_a_script_too_long_to_read_is_not_put_to_the_person(self, agents, shop):
        async def never(*asked):
            raise AssertionError("the person was asked")

        executor, provider, mind, person = scripted(shop, [
            {"do": "script", "why": "x", "code": "return 1;" + " // and more" * 60},
            lambda text: {"do": "finish", "outcome": "failed", "summary": last_actions(text)[-300:]},
        ], asker=never)
        result, status = browse(agents, executor, "Anything", shop.url + "/catalog")
        assert status == "success" and "is not put to the person" in result["summary"], result

    def test_two_addresses_are_of_one_site_by_their_names(self):
        from agents.browser.tools.policy import same_site

        assert same_site("https://acme.atlassian.net/browse/X-1", "https://acme.atlassian.net/rest/api/3/search")
        assert same_site("https://www.example.com/", "https://api.example.com/v1")
        assert same_site("https://shop.example.co.uk/", "https://api.example.co.uk/")
        assert not same_site("https://one.example.co.uk/", "https://other.co.uk/")
        assert not same_site("https://www.example.com/", "https://example.org/")
        assert not same_site("http://127.0.0.1:8000/", "http://127.0.0.2:8000/")
        assert same_site("http://127.0.0.1:8000/a", "http://127.0.0.1:8000/b")
        assert not same_site("https://www.example.com/", "not an address")


class TestTheTabs:
    def test_a_page_a_click_opened_is_in_front_and_the_tabs_are_the_persons(self, shop):
        """A link that opens in a new tab opened it behind the page the
        person was shown, and they saw nothing happen. What a click
        opened comes to the front, the open tabs are told with every
        frame, and the person's hand goes to one, closes one, opens
        one."""
        pytest.importorskip("playwright")
        from agents.browser.tools.driver import Driver

        async def scenario():
            driver = Driver()
            try:
                await driver.start()
                await driver.goto(shop.url + "/catalog")
                snapshot = await driver.snapshot()
                link = next(e for e in snapshot.elements if "new tab" in e["text"])
                await driver.click(link["n"])
                opened = [(t["index"], t["title"], t["active"]) for t in await driver.tabs()]
                front = driver.page.url
                trail = await driver.dispatch([{"type": "tab", "action": "switch", "index": 1}])
                back = driver.page.url
                trail += await driver.dispatch([{"type": "tab", "action": "close", "index": 2}])
                left = [(t["index"], t["title"], t["active"]) for t in await driver.tabs()]
                trail += await driver.dispatch([{"type": "tab", "action": "close", "index": 1}])
                trail += await driver.dispatch([{"type": "tab", "action": "new"}])
                return opened, front, back, left, trail, len(await driver.tabs())
            finally:
                await driver.stop()

        opened, front, back, left, trail, at_the_end = run(scenario())
        assert opened == [(1, "Catalog — Sidra Fitness", False), (2, "Terms — Sidra Fitness", True)]
        assert front.endswith("/terms") and back.endswith("/catalog")
        assert left == [(1, "Catalog — Sidra Fitness", True)]
        assert trail == ["went to tab 1 'Catalog — Sidra Fitness'",
                         "closed tab 2 'Terms — Sidra Fitness'",
                         "the last tab stays open", "opened a new tab"]
        assert at_the_end == 2


class TestTheLoop:
    def test_one_answer_runs_several_actions_until_one_needs_eyes(self, agents, shop):
        """A page's rows remembered and its link followed come in one
        answer and one step; the link opened another page, so the rest
        of that answer waits for the model to see it."""
        def shop_link(text):
            n = int(re.search(r'^\[(\d+)\] link "Shop"', text, re.M).group(1))
            return [{"do": "remember", "records": [{"kind": "note", "name": "the front page"}]},
                    {"do": "click", "n": n, "effect": "looks"},
                    {"do": "remember", "records": [{"kind": "note", "name": "never kept"}]}]

        executor, provider, mind, person = scripted(shop, [
            shop_link,
            lambda text: {"do": "finish", "outcome": "failed", "summary": last_actions(text)[-300:]},
        ])
        result, status = browse(agents, executor, "Open the shop", shop.url)
        assert status == "success", result
        assert result["steps"] == 2 and [r["name"] for r in result["records"]] == ["the front page"]
        assert "'Shop' → the address is now " + shop.url in result["summary"]
        # What each action did is in the inspector, with what it changed.
        [click] = [a for a in executor.inspectors[-1]["actions"] if a["do"] == "click"]
        assert click["state"] == "ok" and shop.url in click["result"] and click["ms"] >= 0

    def test_an_extraction_reads_a_copy_of_the_page_without_asking(self, agents, shop):
        """Code that only reads runs on a copy of the page, in a world of
        its own, with no way to the network or the site's cookies: it is
        not asked of the person, who sees it in the inspector."""
        listed = "return [...document.querySelectorAll('#list li')].map((li) => li.textContent);"
        reach = ("const out = [];\n"
                 "try { await fetch('/api/items?under=50'); out.push('fetched'); } catch (e) { out.push('no fetch'); }\n"
                 "out.push(document.cookie || 'no cookie');\n"
                 "out.push(typeof window, typeof globalThis);\n"
                 "try { out.push((() => {}).constructor('return document')() ? 'escaped' : 'no'); }"
                 " catch (e) { out.push('no constructor'); }\n"
                 "return out;")

        async def never(*asked):
            raise AssertionError("the person was asked")

        executor, provider, mind, person = scripted(shop, [
            {"do": "extract", "code": listed, "why": "To read the list at once"},
            {"do": "extract", "code": reach, "why": "To try the walls"},
            lambda text: {"do": "finish", "outcome": "failed", "summary": last_actions(text)},
        ], asker=never)
        result, status = browse(agents, executor, "List the catalog", shop.url + "/catalog")
        assert status == "success", result
        said = result["summary"]
        assert "Hex dumbbell 10 kg pair — 140" in said
        assert '["no fetch","no cookie","undefined","undefined","no constructor"]' in said, said
        assert shop.api_calls == ["/api/items?under=1000"], "the extraction reached the network"
        assert VISITOR not in said and result["figures"]["cards"] == 0

    def test_a_follow_up_is_told_what_the_earlier_run_did(self, agents, shop):
        executor, provider, mind, person = make(shop)

        async def scenario():
            first = await executor.invoke(agents["browser"], "browse_run", {
                "goal": "Put the dumbbell under 500 in my basket and order it", "start_url": shop.url},
                chat_level=2)
            again = await executor.invoke(agents["browser"], "browse_run", {
                "goal": "Confirm the order went through"}, chat_level=2)
            return first, again

        (first, _), (again, _) = run(scenario())
        assert first["outcome"] == "done" and again["outcome"] == "done", (first, again)
        told = [p for p in mind.prompts if "THE EARLIER RUN IN THIS CHAT" in p]
        assert told and "Goal: Put the dumbbell under 500" in told[0] and "Outcome: done" in told[0]

    def test_a_control_known_only_by_its_pointer_is_in_the_outline(self, agents, shop):
        """A tree's arrow drawn as a span with a click handler has no
        role and no link: 2.0.0 left it out of the outline, and the run
        asked the person to allow a script that opened the tree. The
        outermost element with a pointer cursor is a control, named by
        the words beside it when it has none."""
        def open_home(text):
            n = int(re.search(r'^\[(\d+)\] button near "Home"', text, re.M).group(1))
            return {"do": "click", "n": n, "effect": "looks"}

        async def never(*asked):
            raise AssertionError("the person was asked")

        executor, provider, mind, person = scripted(shop, [
            open_home,
            lambda text: {"do": "finish", "outcome": "failed", "summary": text.split("OUTLINE:")[-1]},
        ], asker=never)
        result, status = browse(agents, executor, "Open the Home folder", shop.url + "/tree")
        assert status == "success", result
        assert '"Downloads"' in result["summary"], result["summary"]
        assert result["figures"]["cards"] == 0

    def test_a_providers_refusal_ends_the_run_said_as_that(self, agents, shop):
        """The provider refused to answer about a page (its content filter) and the run read the empty answer
        as one it could not read, three times, and said so. Two refusals in a row end it, saying what happened."""
        executor, provider, mind, person = scripted(shop, [("", "refusal"), ("", "refusal"), ("", "refusal")])
        result, status = browse(agents, executor, "Read the catalog", shop.url + "/catalog")
        assert status == "success" and result["outcome"] == "failed", result
        assert "refused to answer" in result["summary"] and result["steps"] == 2

    def test_a_finish_whose_evidence_is_on_the_page_is_not_checked_again(self, agents, shop):
        executor, provider, mind, person = scripted(shop, [
            {"do": "finish", "outcome": "done", "summary": "Saw the page.", "evidence": "Nothing moves here"}])
        result, status = browse(agents, executor, "Read the heading", shop.url + "/still")
        assert status == "success" and result["outcome"] == "done" and result["summary"] == "Saw the page."
        assert not any(p.startswith("The goal was:") for p in mind.prompts)
        assert result["figures"]["model_calls"] == 3  # the look-only check, the step, the site notes

    def test_a_browser_nobody_uses_is_closed_and_where_it_was_kept(self, monkeypatch):
        from agents.browser.tools import browse_tool

        class Driver:
            stopped = False

            class page:
                url = "https://shop.example/basket"

            async def stop(self):
                Driver.stopped = True

        async def scenario():
            monkeypatch.setattr(browse_tool, "IDLE_CLOSE_SECONDS", 0.05)
            browser = browse_tool.Browser()
            kept = browser.keep(Driver())
            kept.last = "Goal: order it"
            browser.idle()
            assert browser.get() is kept, "a call in time keeps it"
            browser.idle()
            await asyncio.sleep(0.2)
            return browser

        browser = run(scenario())
        assert Driver.stopped and browser.kept is None
        assert browser.was == "https://shop.example/basket" and browser.last == "Goal: order it"

