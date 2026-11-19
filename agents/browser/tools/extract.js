async function (code) {
  // An extraction: code that only reads. It runs in a world of its own
  // (the page's scripts cannot reach it, nor it them), is handed a copy
  // of the document as `document`, and has no way to the network or to
  // what the site stored. What it returns is the answer, as JSON. The
  // real document is watched while it runs: a change there is reported.
  const refuse = () => { throw new Error('an extraction reads the page only: no network, no storage'); };
  const later = setTimeout.bind(window);
  const address = location.href;
  // eval would run code against this world's global object, which holds
  // the real document: it is taken away with the network and storage.
  for (const name of ['fetch', 'XMLHttpRequest', 'WebSocket', 'EventSource', 'open', 'localStorage',
                      'sessionStorage', 'indexedDB', 'caches', 'cookieStore', 'Worker', 'SharedWorker', 'eval',
                      'setTimeout', 'setInterval', 'importScripts']) {
    try { Object.defineProperty(window, name, {get: refuse, configurable: false}); } catch (e) {}
  }
  try { Object.defineProperty(Navigator.prototype, 'sendBeacon', {value: refuse}); } catch (e) {}
  try { Object.defineProperty(Document.prototype, 'cookie', {get: () => '', set: refuse}); } catch (e) {}
  const copy = document.cloneNode(true);
  let touched = 0;
  const watch = new MutationObserver((found) => { touched += found.length; });
  watch.observe(document, {subtree: true, childList: true, attributes: true, characterData: true});
  const Async = Object.getPrototypeOf(async function () {}).constructor;
  const made = new Async('document', 'window', 'self', 'globalThis', 'top', 'parent', 'frames', 'opener',
                         'Function', 'location', '"use strict";\n' + code);
  // Past here no code may make code of its own, which would see the
  // world's global object and, through it, the real document.
  for (const kind of [Function, Async, Object.getPrototypeOf(function* () {}).constructor,
                      Object.getPrototypeOf(async function* () {}).constructor]) {
    try { Object.defineProperty(kind.prototype, 'constructor', {value: undefined}); } catch (e) {}
  }
  let value = null;
  let failed = '';
  try {
    value = await made(copy, undefined, undefined, undefined, undefined, undefined, undefined, undefined,
                       undefined, {href: address});
  } catch (e) {
    failed = String((e && e.message) || e);
  }
  await new Promise((resolve) => later(resolve, 0));
  watch.disconnect();
  let answer;
  try { answer = JSON.stringify(value === undefined ? null : value); } catch (e) { answer = String(value); }
  return {answer: answer === undefined ? 'null' : answer, failed, touched};
}
