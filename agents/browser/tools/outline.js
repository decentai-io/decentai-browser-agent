(opts) => {
  // The page as an outline, in document order: headings, words, forms,
  // dialogs, and every control a person could act on, numbered. A
  // control keeps its number (data-dai-mark) as long as the document
  // lives. Hidden things are left out; things out of view are not.
  if (!window.__daiSeq) Object.defineProperty(window, '__daiSeq', {value: {n: 0}});
  const seq = window.__daiSeq;
  const CONTROL = 'a[href], button, input:not([type=hidden]), select, textarea, summary, [role=button], ' +
    '[role=link], [role=checkbox], [role=radio], [role=tab], [role=menuitem], [role=menuitemcheckbox], ' +
    '[role=menuitemradio], [role=option], [role=combobox], [role=switch], [role=textbox], [role=searchbox], ' +
    '[role=slider], [role=spinbutton], [contenteditable=""], [contenteditable=true], [onclick]';
  const INLINE = new Set(['span', 'b', 'i', 'em', 'strong', 'small', 'label', 'abbr', 'code', 'sup', 'sub',
    'mark', 'time', 'bdi', 'u', 's', 'font', 'data', 'q', 'cite']);
  const SKIPPED = new Set(['script', 'style', 'noscript', 'template', 'svg', 'canvas', 'head', 'meta', 'link']);
  const words = (t) => String(t || '').replace(/\s+/g, ' ').trim();
  const cut = (t, n) => { t = words(t); return t.length > n ? t.slice(0, n - 1) + '…' : t; };
  const shown = (el) => {
    if (el.checkVisibility && !el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true})) return false;
    const box = el.getBoundingClientRect();
    return (box.width >= 2 && box.height >= 2) || getComputedStyle(el).display === 'contents';
  };
  const roleOf = (el, tag, type) => {
    const role = el.getAttribute('role');
    if (role) return role;
    if (tag === 'a') return 'link';
    if (tag === 'select') return el.multiple ? 'listbox' : 'combobox';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'input') {
      return {checkbox: 'checkbox', radio: 'radio', search: 'searchbox', range: 'slider', number: 'spinbutton',
              submit: 'button', button: 'button', reset: 'button', image: 'button', file: 'file'}[type] || 'textbox';
    }
    if (el.isContentEditable) return 'textbox';
    return 'button';
  };
  const labelOf = (el, tag) => {
    const by = el.getAttribute('aria-labelledby');
    if (by) {
      const t = by.split(/\s+/).map((id) => document.getElementById(id)).filter(Boolean)
        .map((e) => e.innerText).join(' ');
      if (words(t)) return t;
    }
    const tied = el.id ? document.querySelector('label[for="' + CSS.escape(el.id) + '"]') : null;
    const wrap = (tag === 'input' || tag === 'select' || tag === 'textarea') ? el.closest('label') : null;
    const img = el.querySelector ? el.querySelector('img[alt]') : null;
    return el.getAttribute('aria-label') || (tied ? tied.innerText : '') || (wrap ? wrap.innerText : '') ||
      el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('alt') || (img ? img.alt : '') ||
      el.getAttribute('name') || '';
  };
  const elements = [], lines = [];
  let used = 0, more = 0, full = false, text = '';
  const say = (line) => {
    if (full) return;
    if (used + line.length + 1 > opts.chars) { full = true; return; }
    lines.push(line); used += line.length + 1;
  };
  const flush = () => { if (text) { say(cut(text, 400)); text = ''; } };
  const control = (el) => {
    if (elements.length >= opts.controls || full) { more += 1; return; }
    const tag = el.tagName.toLowerCase();
    const type = tag === 'input' ? (el.getAttribute('type') || 'text').toLowerCase() : '';
    let n = Number(el.getAttribute('data-dai-mark') || 0);
    if (!n) { seq.n += 1; n = seq.n; el.setAttribute('data-dai-mark', String(n)); }
    const role = roleOf(el, tag, type);
    const inner = words(el.innerText || el.textContent || '');
    const label = words(labelOf(el, tag));
    const raw = (tag === 'input' || tag === 'textarea' || tag === 'select') ? String(el.value || '')
      : (el.isContentEditable ? inner : '');
    const value = type === 'password' ? (raw ? '••••' : '') : raw;
    const form = el.form || el.closest('form');
    const box = el.getBoundingClientRect();
    const ariaChecked = el.getAttribute('aria-checked');
    const item = {
      n, tag, role: el.getAttribute('role') || '', kind: role, type,
      text: cut(inner, 80), label: cut(label, 80), value: cut(value, 80),
      href: tag === 'a' ? String(el.getAttribute('href') || '').slice(0, 200) : '',
      name: String(el.getAttribute('name') || el.id || '').slice(0, 60),
      autocomplete: String(el.getAttribute('autocomplete') || '').toLowerCase().slice(0, 40),
      checked: (type === 'checkbox' || type === 'radio') ? !!el.checked
        : (ariaChecked ? ariaChecked === 'true' : undefined),
      disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true',
      form: form ? (form.getAttribute('method') || 'get').toLowerCase() : '',
      seen: box.bottom > 0 && box.top < innerHeight && box.right > 0 && box.left < innerWidth,
    };
    elements.push(item);
    const named = ['textbox', 'searchbox', 'combobox', 'spinbutton', 'listbox', 'file'].includes(role)
      ? (label || item.text) : (item.text || label);
    let line = '[' + n + '] ' + role + (type === 'password' ? '(password)' : '');
    if (named) line += ' "' + cut(named, 80) + '"';
    if (item.value && tag !== 'select') line += ' value="' + item.value + '"';
    if (tag === 'select') {
      const chosen = el.selectedOptions && el.selectedOptions[0] ? words(el.selectedOptions[0].text) : '';
      if (chosen) line += ' chosen="' + cut(chosen, 60) + '"';
      const options = Array.from(el.options || []).slice(0, 12).map((o) => cut(o.text, 30));
      if (options.length) line += ' options: ' + options.join(' | ') + (el.options.length > 12 ? ' | …' : '');
    }
    if (item.checked !== undefined) line += item.checked ? ' checked' : ' unchecked';
    if (item.disabled) line += ' disabled';
    const expanded = el.getAttribute('aria-expanded');
    if (expanded) line += expanded === 'true' ? ' expanded' : ' collapsed';
    if (item.href && !item.href.startsWith('javascript:') && !item.href.startsWith('#')) {
      line += ' -> ' + item.href.slice(0, 120);
    }
    say(line);
  };
  const walk = (node) => {
    for (let child = node.firstChild; child; child = child.nextSibling) {
      if (child.nodeType === 3) {
        const t = words(child.nodeValue);
        if (t) text = text ? text + ' ' + t : t;
        continue;
      }
      if (child.nodeType !== 1) continue;
      const el = child;
      const tag = el.tagName.toLowerCase();
      if (SKIPPED.has(tag) || el.id === 'dai-marks') continue;
      if (!shown(el)) continue;
      if (el.matches(CONTROL)) { flush(); control(el); continue; }
      const role = el.getAttribute('role') || '';
      if (/^h[1-6]$/.test(tag) || role === 'heading') {
        flush();
        say('#'.repeat(Number(tag[1]) || Number(el.getAttribute('aria-level')) || 2) + ' ' + cut(el.innerText, 160));
        continue;
      }
      if (tag === 'dialog' || role === 'dialog' || role === 'alertdialog') {
        flush();
        const name = el.getAttribute('aria-label');
        say('== dialog' + (name ? ' "' + cut(name, 60) + '"' : '') + ' ==');
        walk(el); flush(); say('== end of dialog ==');
        continue;
      }
      if (tag === 'form') {
        flush();
        say('== form ' + (el.getAttribute('method') || 'get').toUpperCase() + ' ' +
            cut(el.getAttribute('action') || '(this page)', 80) + ' ==');
        walk(el); flush();
        continue;
      }
      if (tag === 'img') {
        if (el.alt) text = (text ? text + ' ' : '') + '[image: ' + cut(el.alt, 60) + ']';
        continue;
      }
      if (tag === 'iframe') {
        flush(); say('[frame ' + cut(el.getAttribute('title') || el.getAttribute('src') || '', 80) + ']');
        continue;
      }
      const block = !INLINE.has(tag);
      if (block) flush();
      walk(el);
      if (block) flush();
    }
  };
  if (document.body) walk(document.body);
  flush();
  // Only frames a person could see and act in: an invisible captcha
  // frame (a login page's silent risk check) is not a human check.
  const frames = Array.from(document.querySelectorAll('iframe')).filter((f) => {
    const r = f.getBoundingClientRect();
    const s = getComputedStyle(f);
    return r.width >= 200 && r.height >= 60 && s.display !== 'none' && s.visibility !== 'hidden'
      && s.opacity !== '0' && r.right <= window.innerWidth + 1 && r.left >= -1;
  }).map((f) => f.getAttribute('src') || '').filter(Boolean);
  return {elements, outline: lines.join('\n'), more, cut: full ? 1 : 0, frames,
          scroll: [Math.round(window.scrollY), Math.round(document.documentElement.scrollHeight), innerHeight]};
}
