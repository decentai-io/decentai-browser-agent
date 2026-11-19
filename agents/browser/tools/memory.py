"""What a run remembers, and how much of it the mind is shown.

Everything is kept by the code, never rewritten by the model: every
action and what it changed, every record, every page visited, every
word the person said, and what an earlier run in this chat did. What is
*shown* at a step is budgeted — a share of the step's prompt each:

- **The plan.** Each step of it has a state, so where the run stands is
  in front of the mind at every step.
- **The ledger.** Pages visited and presses approved: facts, kept by
  the code, which a summary would blur.
- **The actions.** The newest whole; older ones shorter as room runs
  out, and the oldest counted, with how to ``recall`` them.

Wherever something is cut, the cut is said, with how to get the rest.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List


def cut(text: str, limit: int, rest: str = "") -> str:
    """The text within its limit — and when that cut it, a line saying
    how much is missing and how to get it."""
    text = str(text or "")
    limit = max(0, int(limit))
    if len(text) <= limit:
        return text
    note = f"\n[cut here: {len(text) - limit} more characters" + (f" — {rest}" if rest else "") + "]"
    return text[:max(0, limit)] + note


class Plan:
    """The steps of the work, each with where it stands."""

    STATES = ("todo", "doing", "done", "dropped")
    STEPS_MAX = 20

    def __init__(self):
        self.steps: List[Dict[str, str]] = []

    def take(self, raw: Any) -> None:
        """The plan as the mind wrote it: a list of steps, each a
        sentence or {"step", "state"}. A step already known keeps its
        state when none is given."""
        if not isinstance(raw, list):
            return
        known = {s["step"]: s["state"] for s in self.steps}
        steps = []
        for item in raw[: self.STEPS_MAX]:
            if isinstance(item, dict):
                text = str(item.get("step") or item.get("text") or "").strip()
                state = str(item.get("state") or "").strip().lower()
            else:
                text, state = str(item or "").strip(), ""
            if not text:
                continue
            steps.append({"step": text[:300],
                          "state": state if state in self.STATES else known.get(text, "todo")})
        if steps:
            self.steps = steps

    def shown(self) -> str:
        return "\n".join(f"{i}. [{s['state']}] {s['step']}" for i, s in enumerate(self.steps, 1))


class SiteNotes:
    """What is known of a site, by section, so that one long section
    does not push out the others. Kept as text a person can read and
    edit: a line naming the section, then its words."""

    SECTIONS = (
        ("signing_in", "SIGNING IN"),
        ("finding", "FINDING THINGS"),
        ("avoid", "WHAT TO AVOID"),
        ("person", "WHAT THE PERSON DID THEMSELVES"),
    )
    SECTION_CHARS = 900
    #: The names the person's values are saved under on this site: a
    #: field asked under another name would put a card in front of
    #: them for a value they already gave.
    NAMES = "NAMES OF THE SIGN-IN FIELDS"

    def __init__(self, text: str = ""):
        self.sections: Dict[str, str] = {key: "" for key, _ in self.SECTIONS}
        self.names: Dict[str, str] = {}
        self._read(str(text or ""))

    def _read(self, text: str) -> None:
        titles = {title: key for key, title in self.SECTIONS}
        current, lines = "", {key: [] for key in self.sections}
        loose: List[str] = []
        for line in text.splitlines():
            head = line.strip().rstrip(":")
            if head in titles:
                current = titles[head]
            elif head == self.NAMES:
                current = self.NAMES
            elif current == self.NAMES:
                name, _, label = line.partition(" = ")
                if name.strip() and label.strip():
                    self.names[name.strip()] = label.strip()
            elif current:
                lines[current].append(line)
            elif line.strip():
                loose.append(line)
        for key in self.sections:
            self.sections[key] = "\n".join(lines[key]).strip()
        if loose:
            # Notes from before there were sections: what is known of
            # the site, kept where finding things is.
            self.sections["finding"] = ("\n".join(loose) + "\n" + self.sections["finding"]).strip()

    def write(self, key: str, words: str) -> None:
        """One section anew. Nothing said keeps what was there."""
        words = str(words or "").strip()
        if key in self.sections and words:
            self.sections[key] = words[: self.SECTION_CHARS]

    def add(self, key: str, words: str) -> None:
        """More under a section, the oldest giving way when it is full."""
        words = str(words or "").strip()
        if key in self.sections and words:
            self.sections[key] = (self.sections[key] + "\n" + words).strip()[-self.SECTION_CHARS:]

    def text(self) -> str:
        parts = [f"{title}:\n{self.sections[key]}" for key, title in self.SECTIONS
                 if self.sections[key]]
        if self.names:
            parts.append(f"{self.NAMES}:\n" + "\n".join(
                f"{name} = {label}" for name, label in self.names.items()))
        return "\n\n".join(parts)


class WorkingMemory:
    #: What one step's prompt may hold, in characters — the model's to
    #: read, not the agent's to guess: a deployment with a larger or a
    #: smaller model says so.
    PROMPT_CHARS = int(os.environ.get("DECENTAI_BROWSER_PROMPT_CHARS") or 48000)
    #: What each part may take of it. The page's outline takes what the
    #: others leave, and never less than its own share.
    SHARES = {"actions": 0.20, "ledger": 0.06, "records": 0.08, "notes": 0.06, "said": 0.03,
              "earlier": 0.05, "page": 0.45}
    #: The newest actions are always shown whole.
    WHOLE = 4
    #: Pages the ledger names, the newest.
    VISITED_SHOWN = 12

    def __init__(self):
        self.actions: List[Dict[str, str]] = []
        self.records: List[Dict[str, Any]] = []
        self.said: List[str] = []
        #: every address the run looked at, oldest first, once each
        self.visited: List[str] = []
        #: presses the person approved, in words
        self.approved: List[str] = []
        #: what an earlier run in this chat did, when this one continues it
        self.earlier = ""

    def share(self, part: str) -> int:
        return int(self.PROMPT_CHARS * self.SHARES[part])

    # -- what is kept ------------------------------------------------------------------
    def did(self, action: str, result: str, state: str = "ok") -> None:
        self.actions.append({"action": str(action), "result": str(result), "state": state})

    def heard(self, text: str) -> None:
        self.said.append(str(text)[:2000])
        self.did("the person said", str(text)[:600])

    def keep(self, records: List[Dict[str, Any]]) -> int:
        kept = [r for r in records if isinstance(r, dict) and r]
        self.records.extend(kept)
        return len(kept)

    def seen(self, url: str) -> None:
        url = str(url or "")
        if not url or url.startswith("about:") or (self.visited and self.visited[-1] == url):
            return
        if url in self.visited:
            self.visited.remove(url)
        self.visited.append(url)

    def handed_on(self, goal: str, outcome: str, summary: str) -> str:
        """What this run did, as the next run in the chat is told it."""
        lines = [f"- {a['action']} → {cut(a['result'], 160)}" for a in self.actions[-15:]]
        return (f"Goal: {goal}\nOutcome: {outcome}: {cut(summary, 600)}\n"
                f"Pages: {', '.join(self.visited[-8:])}\n{len(self.records)} records kept.\n"
                "Its last actions:\n" + "\n".join(lines)).strip()

    # -- what is shown ---------------------------------------------------------------------
    def ledger_shown(self) -> str:
        parts = []
        if self.visited:
            pages = self.visited[-self.VISITED_SHOWN:]
            more = len(self.visited) - len(pages)
            parts.append("Pages visited (oldest first): " + (f"[{more} earlier] " if more else "")
                         + " | ".join(pages))
        if self.approved:
            parts.append("Approved by the person: " + "; ".join(self.approved[-8:]))
        return cut("\n".join(parts), self.share("ledger"))

    def actions_shown(self) -> str:
        """Every action, oldest first, within their share: the newest
        whole, older ones given less as room runs short, the oldest
        counted."""
        if not self.actions:
            return ""
        room = self.share("actions")
        lines: List[str] = []
        for index, action in enumerate(reversed(self.actions)):
            # The newest may take half of what is left; each one before
            # it, half again — a read a moment ago is whole, a read ten
            # actions ago is its beginning.
            allowed = max(200, room // 2) if index < self.WHOLE else max(160, room // 6)
            line = f"{len(self.actions) - index}. {action['action']} → " + cut(
                action["result"], allowed, "recall finds it again")
            if len(line) > room and index >= self.WHOLE:
                lines.append(f"[{len(self.actions) - index} older actions are not shown: recall finds them]")
                break
            lines.append(line)
            room = max(0, room - len(line))
        return "\n".join(reversed(lines))

    def records_shown(self) -> str:
        """How many records are kept, by kind, and the newest within
        their share."""
        if not self.records:
            return ""
        kinds: Dict[str, int] = {}
        for record in self.records:
            kind = str(record.get("kind") or "record")
            kinds[kind] = kinds.get(kind, 0) + 1
        head = f"{len(self.records)} kept: " + ", ".join(f"{n} {kind}" for kind, n in kinds.items())
        room, lines = self.share("records") - len(head), []
        for record in reversed(self.records):
            line = json.dumps(record, ensure_ascii=False)
            if len(line) > room:
                break
            lines.append(line)
            room -= len(line) + 1
        if len(lines) < len(self.records):
            head += f" — the newest {len(lines)} are shown; recall finds the others"
        return head + ("\n" + "\n".join(reversed(lines)) if lines else "")

    def said_shown(self) -> str:
        room, lines = self.share("said"), []
        for text in reversed(self.said):
            if len(text) > room and lines:
                break
            lines.append(f"- {cut(text, room)}")
            room -= len(text)
        return "\n".join(reversed(lines))

    def recall(self, query: str, limit: int) -> str:
        """What was kept that holds every word asked for: records
        first, then actions, oldest first. With no words, the oldest
        actions — what left sight first."""
        words = [w for w in re.split(r"\s+", str(query or "").lower()) if w]
        found: List[str] = []
        for record in self.records:
            line = json.dumps(record, ensure_ascii=False)
            if words and all(w in line.lower() for w in words):
                found.append("record: " + line)
        for number, action in enumerate(self.actions, 1):
            line = f"{action['action']} → {action['result']}"
            if not words or all(w in line.lower() for w in words):
                found.append(f"action {number}: " + line)
        if not found:
            return f"nothing kept holds {query!r}"
        each = max(300, limit // max(1, min(len(found), 12)))
        shown = [cut(line, each) for line in found[:12]]
        more = len(found) - len(shown)
        return "\n".join(shown) + (f"\n[{more} more match: ask with more words]" if more > 0 else "")
