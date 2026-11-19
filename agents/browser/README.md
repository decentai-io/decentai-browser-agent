# Browser

A real web browser, driven by the chat's model, with no site-specific
code. Given a goal and a page it reads the page, decides a few actions,
runs them, sees what each one changed, and reads again — up to a step
budget — while the person watches it live in the chat, with an
inspector beside the page, and takes it over when a site wants a human.

## What the model sees, and what it answers

The page as an **outline** of the whole document, in its order:
headings, words, forms, dialogs, and every control a person could act
on, numbered — `[12] button "Add to basket"`. A control keeps its
number for as long as the page lives, so the history stays meaningful.
Controls below the fold are in the outline too: the model acts on them
without scrolling. A control a framework drew without saying so (a
click handler on a span, no role, no link) is known by its pointer
cursor, and an icon with no words is named by the words beside it. A **picture** comes with the outline only when it
helps: the run's first look, a page the outline says little about (a
canvas, a map), a frame a person could act in, or when the model asks
(`look`). A step with a picture is never served from the provider's
cache, so the outline carries the rest.

Beside the page: the goal and whether the person asked only to look
(decided once, from the goal alone, while the browser starts), the plan
with where each step stands, the ledger (pages visited, presses the
person approved), the records kept, every action with **what it
changed** (a new address, a dialog, lines that came and went, or "no
visible effect"), what the person said, the site's notes, and what an
earlier run in the same chat did.

The model answers with one JSON object: up to six **actions**, run in
order; a few words on what it does **next** (the person reads them in
the inspector); and its plan when the plan changed. It writes no
reasoning into the answer: a step asks for low effort, and only finding
another way, when stuck, asks for the chat's own. What stays the same for
the whole run — the instructions, the goal, whether to only look, the
earlier run, the site's notes — is sent as the system text, the part a
provider caches; the plan, the ledger, the actions and the page follow
in the step's own text. A provider that refuses to answer (its content
filter) is said as that, and two refusals in a row end the run.

How much of each part it is shown is a share of the step's prompt
(`tools/memory.py`); everything is kept by the code and nothing is
summarised by the model. Where something is cut, the cut is said, with
how to get the rest (`recall`, `read`). `DECENTAI_BROWSER_PROMPT_CHARS`
(48,000) is the number a deployment may change.

## Not doing the same thing twice

The actions of one answer run until one needs the model's eyes: it
failed, it changed nothing, it opened another page, a tab or a dialog,
or it read something the model must see. An action that changed nothing
is not done again on the same page: the second try is refused, and the
model is told. Three actions in a row that changed nothing, or a run
going in circles, get one call at the chat's own effort, with a
picture, to find another way. When that comes to nothing too, the
person is offered the browser, once for a page, before the run gives
up.

## Waiting for the page

After an action the driver waits for what the action causes, not for a
fixed time: a new page to load when one is on its way, then the page to
stop changing for 120 ms: 0.7 seconds at most, 1.2 after a new page.
Only changes to what the page holds count, not attributes alone, so an
ad's animation or a carousel is not waited on. Fields are filled at
once; a search box or a list that suggests as one types is typed key by
key.

## Beneath the picture

The browser has no window, so there is no developer's window to open;
what that window shows is read through the driver and handed over as
words (`tools/inspector.py`).

| Action | What it gives | Rule |
|---|---|---|
| `source` | The HTML of an element or the page, without what draws or runs, and without any field's value | Free |
| `network` | What the page fetched since it opened: method, status, kind, address | Free; never a header |
| `response` | What came back for one of them, when it is data or text | Free; not while a sign-in is on the page, nor for what was fetched while the person held the browser |
| `console` | What the page logged, and its errors | Free |
| `request` | An address of the page's own site, fetched with the page's session | GET only, the same site only |
| `extract` | JavaScript that only reads, run on a copy of the page | Not asked: see below |
| `script` | JavaScript, run in the live page | The person's approval every time, asked in the chat with the reason and the code whole. 600 characters at most |

An **extraction** runs in a script world of its own, which the page's
scripts cannot reach. Its `document` is a copy of the page. It has no
`window`, `fetch`, `XMLHttpRequest`, `WebSocket`, storage or cookies, and
it cannot make code of its own (the function constructors are taken
away). The page's own requests are held while it runs. The live page is
watched meanwhile: if it changes — by the code reaching past its copy,
or by the page itself, which cannot be told apart — that is reported,
and every later extraction in the run is asked of the person like a
script. Each extraction shows in the inspector's Pending list while it
runs.

Cookies, stored values and headers are never shown: they are the
person's session, not the page's content.

## The inspector

Beside the page, the live view has a tab that cannot be closed:

- what the agent does now and plans next;
- the plan;
- what is about to happen: a script, an extraction, a press waiting for
  approval, a request, a sign-in;
- each action with what it changed and how long it took (and the model
  time it cost);
- the page's requests, the agent's own marked apart, and those the
  address policy refused;
- the console;
- the run's figures: steps, model calls and time, browser time, cards
  asked, repeats refused.

It is sent as the call's `inspector` note, whole each time (the newest
replaces the one before), at most a few times a second. Its shape is
the `inspector` event's, version 1 (`tools/live.py`). It holds no
control: approvals stay on the platform's cards.

## Functions

| Function | Level | What it does |
|---|---|---|
| `browse_run` | 1 | Does what the goal says from `start_url`, up to `max_steps` (60 by default, 200 at most). Without `start_url` it continues on the page the chat's browser was left at. `files` are what the person attached for a page to take (`upload`). Returns the outcome (`done`, `stopped`, `stopped_by_person` when the user answered Stop, `failed`, `budget`), a summary, the records found, the steps taken, the final URL, a screenshot's `file_ref`, what was approved, how many handoffs, any downloads (each returned to the chat as a file), and the run's figures |
| `browse_watch` | 0 | Shows the chat's browser in the live view before anything is asked of it, so the person can look, sign in or go somewhere; the model calls it when the person wants to see or use the browser, not as a step in a task. `action: quit` closes the browser and remembers where it was |
| `browse_screenshot` | 2 | Opens a page and returns a screenshot as a file, with its title and final URL |

A process serves one chat, so it holds one browser: it stays open after
a run, and a follow-up in the same chat continues where the last run
stopped. Chromium is most of a process's memory, so a browser nobody
has used for three minutes (`DECENTAI_BROWSER_IDLE_SECONDS`, 180) is
closed; where it was and what the last run did are kept, and the next
run opens a browser there, signed in by the site's session.

## The moments that are the person's

- **A sign-in, in whatever shape the site gives it.** An email on one
  page and the password on the next, a company number beside both, a
  PIN, a code: on each page the model names the fields that take the
  person's own values, and the site's login (a `logins` secret, one
  per domain) is asked for exactly those — handed over when it holds
  them; when it does not, they are its form's questions, which the
  person fills in on the platform's form. A field is named by what
  the form says it is (a password input, the form's own marking of a
  user name or a one-time code) or by what the model says it is, never
  by the words on the page, which are in any language. A code is asked
  each time and never stored. The values reach the page and nothing
  else: not the model's prompt, not the history.
- **A press that cannot be taken back.** Before a button is pressed or
  a form sent by POST, a second model call — the judge
  (`tools/judge.py`) — is shown the control, its form and the screen
  around it, and says what the press does: it looks, it changes
  something the person can take back, or it commits. One that commits
  stops for the person's approval (asked in the chat), and so does any
  change when they asked only to look. A judge that cannot be read is
  asked once more; then its answer is one that commits. A plain link,
  a search box, a form sent by GET and a choice in a list of no form are
  reads and are not judged. No list of words decides any of this, so it
  holds in every language. Only the agent's own question approves a
  press: a question the model asked on its own does not. The same press
  approved once may be allowed for the rest of the run when it comes
  again (the third "Add to basket"); and a press that only confirms one
  the person just approved on the same site ("Confirm order" after
  "Place order") is not asked again.
- **A page that wants a human.** A CAPTCHA, an approval on the person's
  phone, a sign-in a script may not do: the model sees it and hands the
  browser over, the agent feeds the person's hand to the page, and
  continues when they press Done. A run that got nowhere, and found
  no other way, offers the browser too, once for a page, before it
  gives up.
- **A script.** See above: theirs to allow, each time. An extraction is
  not, unless one was seen to change the page.
- **The tabs.** A page a click opened in a new tab comes to the front.
  Every frame tells of the open tabs, the live view shows them, and the
  person goes to one, closes one or opens one — which takes the
  browser, as any hand on it does.

## What it keeps

- **Sessions** (`session` records, encrypted): a site's cookies after a
  sign-in or a handoff, keyed by domain and account, so the next run
  skips the form.
- **Site notes** (`notes` records, editable by the person), by
  section — signing in, finding things, what to avoid, what the person
  did themselves, and the names the sign-in's fields are saved under —
  each rewritten on its own after a run, read before the next on that
  site.
- **Sign-ins** (`logins` secrets): what each site's sign-in asks, by
  field name, under the site's domain; a one-time code is never kept.
- **Where the browser was**, in the process only: the address the
  chat's browser was at when it closed, so the next run or watch in
  that chat without an address starts there.

The final screenshot of every run, and any file the model chose to
download, are returned to the chat as its files; nothing keeps them.

## Where it may go

Every navigation and every request the page makes is checked: loopback
and private addresses are refused, so a browser on the runtime host
cannot reach a metadata service or the platform itself.
`DECENTAI_WEB_ALLOW_LOOPBACK=1` is the tests' escape. In the sandbox
the browser is handed the sandbox's way out (`HTTPS_PROXY`), which
looks each name up and makes the same check; a page it refused is said
with its reason, asked of the way out with a tunnel.

## What it carries

The program carries Playwright and its driver, pinned in
`requirements.txt`, and Chromium with them, under `chromium/` at the
program's root (`publish/chromium.sh` makes it for each machine): the
build Playwright wants, the system libraries it loads that the sandbox's
base lacks, and fonts. Outside a program, in a checkout, Playwright's own
install serves (`playwright install --no-shell chromium`). Chromium runs without its own sandbox
(`--no-sandbox`: the sandbox it runs in refuses the namespaces it would
make) and keeps its shared memory out of the small `/dev/shm` (`--disable-dev-shm-usage`). Each Playwright
release wants one browser build: Playwright 1.59.0 and `chromium-1217`
go together. A missing browser is reported in the run's summary, not
hidden. Set `DECENTAI_BROWSER_HEADLESS=0` to watch it on a machine with
a screen.

## Tests

`tests/test_browser.py` runs the agent as its own program with a real
Chromium against a loopback shop: a login, items, a basket, an order
button, and a page that wants a human. The chat's model is scripted;
the person is a scripted asker who fills in the platform's form.
`tests/test_browser_revision.py` proves the judge, the memory, the
looking beneath and the tabs: the shop has a sign-in over two pages in
Arabic, a list it fetches as data, a link that opens a new tab, and a
page where nothing moves. `tests/test_browser_proxy.py` runs it
behind the tests' way out, which refuses a name that resolves inside.
