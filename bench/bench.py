"""The browser agent's benchmark: the same tasks, each in a chat of its own, on a DecentAI deployment.

    DECENTAI_URL=https://decentai.example DECENTAI_KEY=<an API key> python bench/bench.py <label> [task ...]

For each task: a new chat asks the assistant to run the Browser agent with the task's goal and address, word for
word. Every card is answered the same way, as a person who allows everything would: approvals allowed, a press
"Go ahead", a script "Run it", a handoff "Done". A WebSocket stays open on each chat while it runs, as a browser
tab does, so the deployment sends no mail for it.

What is kept, per task, in bench/results/<label>.jsonl: the run's outcome, summary, steps and figures as the agent
reported them (figures only from 2.0.0 on), the wall time from the message sent to the run's end, and every card
by kind (the platform's approvals, the agent's questions, forms). Model calls and their time come from the
deployment's usage records, by chat id, which the operator reads apart (bench/usage.js).
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("DECENTAI_URL", "https://decentai.net").rstrip("/")
KEY = os.environ["DECENTAI_KEY"]
CTX = ssl.create_default_context()
HERE = Path(__file__).resolve().parent

TASKS = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))
#: How long one task may take before it is given up on.
LIMIT_SECONDS = 900


def call(endpoint: str, data: dict | None = None, timeout: int = 40) -> dict:
    body = json.dumps({"endpoint": endpoint, "data": data or {}}).encode()
    request = urllib.request.Request(BASE + "/app", body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=CTX) as answer:
            return json.loads(answer.read())
    except urllib.error.HTTPError as error:
        return {"_error": error.code, "_body": error.read().decode(errors="replace")[:500]}


def answer_for(card: dict) -> dict:
    """The same answer every time: allow everything, choose the first choice that is not a refusal."""
    kind, request = card.get("kind"), card.get("request") or {}
    if kind == "approval":
        return {"items": {item["id"]: True for item in request.get("items", [])}}
    if kind in ("network", "confirm"):
        return {"allowed": True}
    expects = request.get("expects")
    if expects == "choice":
        choices = request.get("choices") or []
        refusals = ("don", "no ", "no,", "cancel", "stop")
        return {"choice": next((c for c in choices if not c.lower().startswith(refusals)), choices[0])}
    if expects == "text":
        return {"text": "Choose what fits the goal best."}
    return {}


def hold(chat_id: str, stop: threading.Event) -> None:
    """A socket on the chat, held open as a person's tab would be."""
    from websockets.sync.client import connect
    while not stop.is_set():
        try:
            with connect(f"{BASE.replace('https://', 'wss://')}/chats/{chat_id}", open_timeout=10,
                         subprotocols=["bearer", KEY], ssl=CTX) as socket:
                while not stop.is_set():
                    try:
                        socket.recv(timeout=5)
                    except TimeoutError:
                        continue
        except Exception:
            time.sleep(1)


def one(task: dict, label: str) -> dict:
    chat = call("AI:Chat:create", {"title": f"Browser bench {label} {task['id']}"})["chat"]["chat_id"]
    stop = threading.Event()
    threading.Thread(target=hold, args=(chat, stop), daemon=True).start()
    time.sleep(2)
    text = (f"Use the Browser agent's Run function once, with start_url {task['url']} and this goal, word for word: "
            f"{task['goal']}\nDo nothing else; when it ends, tell me its outcome in one sentence.")
    began = time.time()
    call("AI:Chat:input", {"chat_id": chat, "text": text, "files": [], "client_id": f"bench-{label}-{task['id']}"})
    seen, cards, runs, replied, idle = 0, [], {}, False, False
    while time.time() - began < LIMIT_SECONDS:
        for event in call("AI:Event:list", {"chat_id": chat, "after": seen}).get("events", []):
            seen = max(seen, int(event.get("seq") or 0))
            kind, data = event.get("event"), event.get("data") or {}
            if kind == "card_opened":
                answer = answer_for(data)
                request = data.get("request") or {}
                cards.append({"kind": data.get("kind"), "agent": data.get("agent") or "",
                              "what": str(request.get("text") or [i.get("name") for i in request.get("items", [])])[:160],
                              "after_s": round(time.time() - began, 1)})
                call("AI:Approval:decide", {"card_id": data["card_id"], "answer": answer})
            elif kind == "work" and "browse" in json.dumps(data).lower():
                runs[data.get("id")] = data
            elif kind == "message" and data.get("actor") == "assistant":
                replied = True
            elif kind == "idle":
                idle = True
        done = [r for r in runs.values() if r.get("status") not in ("running", "waiting")]
        if runs and len(done) == len(runs) and replied and idle:
            break
        time.sleep(1)
    stop.set()
    browse = next(iter(runs.values()), {})
    output = {}
    for work in call("AI:Work:list", {"chat_id": chat}).get("work", []):
        if (work.get("means") or {}).get("function") == "browse_run":
            browse, ref = work, (work.get("output") or {}).get("ref")
            if ref:
                try:
                    output = json.loads(call("AI:Output:see", {"chat_id": chat, "ref": ref}).get("text") or "{}")
                except ValueError:
                    output = {}
    seconds = None
    try:
        from datetime import datetime
        seconds = round((datetime.fromisoformat(browse["ended"].replace("Z", "+00:00"))
                         - datetime.fromisoformat(browse["started"].replace("Z", "+00:00"))).total_seconds(), 1)
    except Exception:
        pass
    return {"label": label, "task": task["id"], "chat_id": chat, "status": browse.get("status"),
            "outcome": output.get("outcome"), "summary": str(output.get("summary") or "")[:400],
            "steps": output.get("steps"), "figures": output.get("figures"), "records": len(output.get("records") or []),
            "run_seconds": seconds, "wall_seconds": round(time.time() - began, 1), "cards": cards,
            "runs": len(runs)}


def main() -> None:
    label = sys.argv[1]
    wanted = set(sys.argv[2:])
    out = HERE / "results" / f"{label}.jsonl"
    out.parent.mkdir(exist_ok=True)
    for task in TASKS:
        if wanted and task["id"] not in wanted:
            continue
        row = one(task, label)
        with out.open("a", encoding="utf-8") as kept:
            kept.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({k: row[k] for k in ("task", "status", "outcome", "steps", "run_seconds", "wall_seconds")}
                         | {"cards": len(row["cards"])}), flush=True)


if __name__ == "__main__":
    main()
