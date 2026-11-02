"""Login / checkout form control classifier for vault autofill.

Python port (~170 LOC) of Merit-Systems/OpenInstinct's
``lib/manager/server/kernel-login-autofill.ts`` (MIT). Classifies visible
input controls on a page into login-autofill tokens. The vault fill path
uses the classification to select the single best current-password control
(the identifier is agent-visible metadata and is typed by the agent
itself via normal input tools).

Scoring:
- exact autocomplete-token match ................ 100
- type=password (not new/confirm/create/repeat) .. 90
- type=email / type=tel .......................... 85
- label/name regex heuristics .................. 70-75
Hard exclusions: autocomplete ``new-password`` / ``one-time-code``, and
label/name text matching ``(new|confirm|create|repeat)\\s*password``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import itertools

LOGIN_AUTOFILL_TOKENS = ("username", "email", "tel", "current-password")

# Payment / address autocomplete tokens (WHATWG) the checkout fill targets. ``cc-exp`` (combined
# MM/YY) is derived at fill time from exp_month + exp_year. Field-name/label heuristics below back
# up sites that omit autocomplete attributes.
PAYMENT_AUTOFILL_TOKENS = ("cc-number", "cc-name", "cc-exp", "cc-exp-month", "cc-exp-year", "cc-csc")
ADDRESS_AUTOFILL_TOKENS = ("address-line1", "address-line2", "address-level2", "address-level1",
                           "postal-code", "country-name", "country")
_CHECKOUT_HEURISTICS = (
    (re.compile(r"\b(?:card\s*number|cardnumber|ccnumber|cc\s*num|pan)\b"), "cc-number"),
    (re.compile(r"\b(?:name\s*on\s*card|cardholder|cc\s*name|ccname)\b"), "cc-name"),
    (re.compile(r"\b(?:cvc|cvv|csc|security\s*code|card\s*code)\b"), "cc-csc"),
    (re.compile(r"\b(?:exp(?:iry|iration)?\s*month|exp\s*mm|ccmonth)\b"), "cc-exp-month"),
    (re.compile(r"\b(?:exp(?:iry|iration)?\s*year|exp\s*yy(?:yy)?|ccyear)\b"), "cc-exp-year"),
    (re.compile(r"\b(?:exp(?:iry|iration)?(?:\s*date)?|mm\s*yy|valid\s*thru)\b"), "cc-exp"),
    (re.compile(r"\b(?:address\s*(?:line\s*)?2|apt|suite|unit)\b"), "address-line2"),
    (re.compile(r"\b(?:address(?:\s*line\s*1)?|street)\b"), "address-line1"),
    (re.compile(r"\b(?:city|town|locality)\b"), "address-level2"),
    (re.compile(r"\b(?:state|province|region|county)\b"), "address-level1"),
    (re.compile(r"\b(?:zip|postal|postcode)\b"), "postal-code"),
    (re.compile(r"\b(?:country)\b"), "country-name"),
)

_EXCLUDED_AUTOCOMPLETE = {"new-password", "one-time-code"}

_RE_EXCLUDED_PASSWORD = re.compile(r"\b(?:new|confirm|create|repeat)\s*password\b")
_RE_EMAIL = re.compile(r"\b(?:e[\s-]?mail|email address)\b")
_RE_TEL = re.compile(r"\b(?:phone|telephone|mobile)\b")
_RE_USERNAME = re.compile(
    r"\b(?:user\s*name|username|login|account|member|membership|mileageplus)\b"
)


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


@dataclass(frozen=True)
class LoginControl:
    """Descriptor of a visible input control, as inspected in the page."""

    autocomplete: str
    form_index: Optional[int]
    index: int
    label: str
    name: str
    type: str
    max_length: Optional[int] = None
    frame_path: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LoginControl:
        form_index = raw.get("formIndex", raw.get("form_index"))
        max_length = raw.get("maxLength", raw.get("max_length"))
        raw_frame_path = raw.get("framePath", raw.get("frame_path")) or ()
        return cls(
            autocomplete=str(raw.get("autocomplete") or ""),
            form_index=int(form_index) if form_index is not None else None,
            index=int(raw.get("index") or 0),
            label=str(raw.get("label") or ""),
            name=str(raw.get("name") or ""),
            type=str(raw.get("type") or ""),
            max_length=int(max_length) if max_length is not None else None,
            frame_path=tuple(int(part) for part in raw_frame_path),
        )


@dataclass(frozen=True)
class ClassifiedLoginControl:
    control: LoginControl
    score: int
    token: str


def classify_login_control(control: LoginControl) -> Optional[ClassifiedLoginControl]:
    """Classify one control, or return None if it is not a login fill target."""
    autocomplete_tokens = [
        t for t in control.autocomplete.lower().split() if t
    ]
    if any(t in _EXCLUDED_AUTOCOMPLETE for t in autocomplete_tokens):
        return None

    for token in LOGIN_AUTOFILL_TOKENS:
        if token in autocomplete_tokens:
            return ClassifiedLoginControl(control, 100, token)

    searchable = _normalize_text(
        " ".join(part for part in (control.name, control.label) if part)
    )
    if _RE_EXCLUDED_PASSWORD.search(searchable):
        return None
    if control.type == "password":
        return ClassifiedLoginControl(control, 90, "current-password")
    if control.type == "email":
        return ClassifiedLoginControl(control, 85, "email")
    if control.type == "tel":
        return ClassifiedLoginControl(control, 85, "tel")
    if _RE_EMAIL.search(searchable):
        return ClassifiedLoginControl(control, 75, "email")
    if _RE_TEL.search(searchable):
        return ClassifiedLoginControl(control, 75, "tel")
    if _RE_USERNAME.search(searchable):
        return ClassifiedLoginControl(control, 70, "username")
    return None


_RE_OTP = re.compile(
    r"\b(?:one[\s-]?time|verification|security|auth(?:entication|enticator)?|2fa|two[\s-]?factor|mfa|totp|otp|"
    r"passcode|sms)\b.*\b(?:code|pin|token)\b|\b(?:otp|totp|2fa|mfa|verification\s*code|passcode)\b"
)


def classify_otp_controls(controls: list[LoginControl]) -> list[ClassifiedLoginControl]:
    """The controls that take a second-factor code. ``autocomplete=one-time-code`` is authoritative;
    otherwise a text/tel/number input whose name/label says code/OTP/2FA/verification. Some sites split
    the code into one input per digit (``maxlength=1`` boxes): they are returned in DOM order and the
    fill spreads the code across them."""
    out: list[ClassifiedLoginControl] = []
    for c in controls:
        tokens = c.autocomplete.lower().split()
        if "one-time-code" in tokens:
            out.append(ClassifiedLoginControl(c, 100, "one-time-code"))
            continue
        if c.type not in ("text", "tel", "number", "password", ""):
            continue
        if _RE_OTP.search(_normalize_text(" ".join(p for p in (c.name, c.label) if p))):
            out.append(ClassifiedLoginControl(c, 70, "one-time-code"))
    return out


def select_password_fill(
    classified: list[ClassifiedLoginControl],
    password: str,
) -> list[dict[str, Any]]:
    """Select the single best current-password control to fill.

    The vault fill path is password-only: the identifier is agent-visible
    metadata and is typed by the agent via normal input tools. This picks
    the highest-scoring ``current-password`` control (ties broken by DOM
    order) and returns ``[{"index": int, "token": "current-password",
    "value": password}]`` or ``[]`` when no password field exists. A control in a same-origin frame
    retains its inspected frame path so the write can revalidate that destination.
    """
    passwords = [c for c in classified if c.token == "current-password"]
    if not passwords or not password:
        return []
    best_password = min(passwords, key=lambda c: (-c.score, c.control.index))
    fill = {
        "index": best_password.control.index,
        "token": "current-password",
        "value": password,
    }
    if best_password.control.frame_path:
        fill["frame_path"] = best_password.control.frame_path
    return [fill]


def classify_checkout_control(control: LoginControl) -> Optional[ClassifiedLoginControl]:
    """Classify one control as a payment/address fill target (autocomplete token exact match 100,
    label/name heuristic 70), or None. Password/email inputs are never checkout targets."""
    tokens = [t for t in control.autocomplete.lower().split() if t]
    for token in PAYMENT_AUTOFILL_TOKENS + ADDRESS_AUTOFILL_TOKENS:
        if token in tokens:
            return ClassifiedLoginControl(control, 100, "country-name" if token == "country" else token)
    if control.type in ("password", "email"):
        return None
    searchable = _normalize_text(" ".join(part for part in (control.name, control.label) if part))
    for pattern, token in _CHECKOUT_HEURISTICS:
        if pattern.search(searchable):
            return ClassifiedLoginControl(control, 70, token)
    return None


def select_checkout_fills(classified: list[ClassifiedLoginControl], secret: dict[str, str],
                          field_tokens: dict[str, str]) -> list[dict[str, Any]]:
    """Map a payment/address secret payload onto the best control per autocomplete token.

    ``field_tokens`` is ``PAYMENT_FIELDS`` / ``ADDRESS_FIELDS`` (agent/vault_store.py). A combined
    ``cc-exp`` control gets ``MM/YY`` from exp_month + exp_year and then suppresses the separate
    month/year fills. Returns ``[{"index", "token", "value"}]``: one control per token, highest
    score then DOM order.
    """
    values: dict[str, str] = {tok: secret[f] for f, tok in field_tokens.items() if secret.get(f)}
    if "cc-exp-month" in values and "cc-exp-year" in values:
        values["cc-exp"] = f"{values['cc-exp-month'].zfill(2)}/{values['cc-exp-year'][-2:]}"
    fills: list[dict[str, Any]] = []
    for token, value in values.items():
        candidates = sorted((c for c in classified if c.token == token), key=lambda c: (-c.score, c.control.index))
        if candidates:
            control = candidates[0].control
            fill = {"index": control.index, "token": token, "value": value}
            if control.frame_path:
                fill["frame_path"] = control.frame_path
            fills.append(fill)
    if any(f["token"] == "cc-exp" for f in fills):
        fills = [f for f in fills if f["token"] not in ("cc-exp-month", "cc-exp-year")]
    return fills


# JS expression evaluated in the page to inspect candidate input controls.
# Ported from OpenInstinct's nativeLoginControlInspectionExpression.
# Inspection stamps every input and reachable same-origin frame under a per-inspection nonce; the fill
# script resolves the target and its frame chain from ITS OWN inspection instead of re-querying by position,
# so neither a DOM reflow nor a second inspection in between can redirect a secret into another field.
INSPECTION_STAMP_ATTR = "data-hermes-vault-slot"
FRAME_STAMP_ATTR = "data-hermes-vault-frame"


def build_otp_fills(otp_controls: list[ClassifiedLoginControl], code: str) -> list[dict[str, Any]]:
    """One fill per box. Default: the single best-scoring code field takes the whole code.

    Per-digit entry only when the page unmistakably uses it: exactly len(code) OTP controls that are all
    ``maxlength=1``, all in the same form, and adjacent in DOM order (the classic N-box widget). Anything
    looser (several code-like inputs scattered over a page) gets ONE field, never a digit sprayed across
    unrelated inputs."""
    best = max(otp_controls, key=lambda c: c.score)
    boxes = sorted((c for c in otp_controls if c.control.max_length == 1), key=lambda c: c.control.index)
    if (len(boxes) == len(code)
            and len({b.control.form_index for b in boxes}) == 1
            and len({b.control.frame_path for b in boxes}) == 1
            and all(b.control.index - a.control.index == 1 for a, b in itertools.pairwise(boxes))):
        fills = [{"index": b.control.index, "token": "one-time-code", "value": ch,
                  "frame_path": b.control.frame_path} for b, ch in zip(boxes, code)]
    else:
        fills = [{"index": best.control.index, "token": "one-time-code", "value": code,
                  "frame_path": best.control.frame_path}]
    for fill in fills:
        if not fill["frame_path"]:
            del fill["frame_path"]
    return fills


def build_inspection_js(nonce: str) -> str:
    return _LOGIN_CONTROL_INSPECTION_JS_TEMPLATE.replace("__NONCE__", json.dumps(nonce))


_LOGIN_CONTROL_INSPECTION_JS_TEMPLATE = """(() => {
  const nonce = __NONCE__;
  const expectedOrigin = window.location.origin;
  if (!expectedOrigin || expectedOrigin === "null") return JSON.stringify([]);
  const elements = [];
  const forms = [];
  const seenDocuments = new Set();
  let nextFrame = 0;
  const frameAllowsSameOrigin = (frame) =>
    !frame.hasAttribute("sandbox") || frame.sandbox.contains("allow-same-origin");
  const documentOrigin = (doc) => {
    try {
      if (!doc || !doc.defaultView) return "";
      const view = doc.defaultView;
      const origin = view.location.origin;
      if (origin !== "null") return origin;
      const frame = view.frameElement;
      if ((view.location.href === "about:blank" || view.location.href === "about:srcdoc") && frame &&
          frameAllowsSameOrigin(frame) && frame.contentDocument === doc) return documentOrigin(frame.ownerDocument);
      return origin;
    }
    catch (_) { return ""; }
  };
  const walkRoot = (root, framePath) => {
    for (const e of root.querySelectorAll("*")) {
      if (e.tagName === "INPUT" || e.tagName === "SELECT") elements.push({ element: e, framePath });
      else if (e.tagName === "FORM") forms.push(e);
      if (e.shadowRoot) walkRoot(e.shadowRoot, framePath);
      if (e.tagName !== "IFRAME" || !frameAllowsSameOrigin(e)) continue;
      try {
        const child = e.contentDocument;
        if (!child || documentOrigin(child) !== expectedOrigin) continue;
        const frameIndex = nextFrame++;
        e.setAttribute("data-hermes-vault-frame", nonce + ":" + frameIndex);
        walkDocument(child, [...framePath, frameIndex]);
      } catch (_) { /* cross-origin, opaque, or detached frame */ }
    }
  };
  const walkDocument = (doc, framePath) => {
    if (seenDocuments.has(doc)) return;
    seenDocuments.add(doc);
    walkRoot(doc, framePath);
  };
  walkDocument(document, []);
  elements.forEach(({ element }, index) => element.setAttribute("data-hermes-vault-slot", nonce + ":" + index));
  const out = elements.flatMap(({ element, framePath }, index) => {
    if (element.disabled || element.readOnly) return [];
    if (["hidden", "submit", "button", "reset", "file", "image", "checkbox", "radio"].includes(element.type)) return [];
    const view = element.ownerDocument.defaultView;
    if (!view || documentOrigin(element.ownerDocument) !== expectedOrigin) return [];
    const style = view.getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden" || element.getClientRects().length === 0) return [];
    const labels = element.labels ? Array.from(element.labels, (l) => l.textContent || "") : [];
    const ariaText = (element.getAttribute("aria-labelledby") || "")
      .split(/\\s+/).filter(Boolean)
      .map((id) => { const r = element.getRootNode(); const n = (r.getElementById ? r.getElementById(id) : null) || document.getElementById(id); return n ? (n.textContent || "") : ""; })
      .join(" ");
    const resolvedFormIndex = element.form ? forms.indexOf(element.form) : -1;
    return [{
      autocomplete: element.autocomplete || "",
      framePath,
      formIndex: resolvedFormIndex >= 0 ? resolvedFormIndex : null,
      index,
      maxLength: element.maxLength > 0 ? element.maxLength : null,
      label: [
        ...labels,
        element.getAttribute("aria-label") || "",
        ariaText,
        element.getAttribute("placeholder") || "",
        element.getAttribute("title") || "",
      ].join(" "),
      name: [element.name, element.id].join(" "),
      type: element.tagName === "SELECT" ? "select" : (element.type || ""),
    }];
  });
  return JSON.stringify(out);
})()"""


def build_fill_js(fills: list[dict[str, Any]], expected_origin: str, nonce: str = "") -> str:
    """Build a JS expression that fills the selected controls and reports only a count. The
    returned expression never echoes the values back.

    ``expected_origin`` is asserted against the root and stamped destination frame chain inside the SAME
    evaluated script, immediately before any write. If either navigated between inspection and fill
    (TOCTOU), the script writes nothing to that destination and returns
    ``{"refused": "origin_changed", "found": <actual>}``: proof scope equals mutation scope (#88706).
    Cross-origin, opaque and sandboxed-without-allow-same-origin frames are unreachable. A
    ``current-password`` fill additionally requires ``type=password``; ``<select>`` controls (country,
    state, expiry month) match an option by value or visible text. No marker is left on filled controls.
    """
    payload = json.dumps(
        [{"index": f["index"], "token": f.get("token", "current-password"), "value": f["value"],
          "framePath": list(f.get("frame_path") or f.get("framePath") or ())} for f in fills]
    )
    return (_FILL_JS_TEMPLATE.replace("__EXPECTED_ORIGIN__", json.dumps(expected_origin))
            .replace("__FILLS__", payload).replace("__NONCE__", json.dumps(nonce)))


def build_cleanup_js(nonce: str, expected_origin: str) -> str:
    """Remove this inspection's slot and frame markers from every reachable destination document."""
    return (_CLEANUP_JS_TEMPLATE.replace("__EXPECTED_ORIGIN__", json.dumps(expected_origin))
            .replace("__NONCE__", json.dumps(nonce)))


_FILL_JS_TEMPLATE = """(() => {
  const expectedOrigin = __EXPECTED_ORIGIN__;
  if (window.location.origin !== expectedOrigin) {
    return JSON.stringify({ refused: "origin_changed", found: window.location.origin });
  }
  const fills = __FILLS__;
  const nonce = __NONCE__;
  const frameAllowsSameOrigin = (frame) =>
    !frame.hasAttribute("sandbox") || frame.sandbox.contains("allow-same-origin");
  const documentOrigin = (doc) => {
    try {
      if (!doc || !doc.defaultView) return "";
      const view = doc.defaultView;
      const origin = view.location.origin;
      if (origin !== "null") return origin;
      const frame = view.frameElement;
      if ((view.location.href === "about:blank" || view.location.href === "about:srcdoc") && frame &&
          frameAllowsSameOrigin(frame) && frame.contentDocument === doc) return documentOrigin(frame.ownerDocument);
      return origin;
    }
    catch (_) { return ""; }
  };
  const rootsFor = (doc) => {
    const roots = [];
    const walk = (root) => {
      roots.push(root);
      for (const e of root.querySelectorAll("*")) if (e.shadowRoot) walk(e.shadowRoot);
    };
    walk(doc);
    return roots;
  };
  const findIn = (doc, selector) => rootsFor(doc).flatMap((root) => Array.from(root.querySelectorAll(selector)));
  const refuse = (found) => {
    cleanup();
    return JSON.stringify({ refused: "origin_changed", found: found || "unavailable" });
  };
  const destinationFor = (framePath) => {
    let doc = document;
    for (const frameIndex of framePath) {
      const frame = findIn(doc, '[data-hermes-vault-frame="' + nonce + ':' + frameIndex + '"]')[0];
      if (!frame || frame.tagName !== "IFRAME") return { refused: "missing-frame" };
      if (!frameAllowsSameOrigin(frame)) return { refused: "sandboxed" };
      try {
        const child = frame.contentDocument;
        const found = documentOrigin(child);
        if (!child || found !== expectedOrigin) return { refused: found || "cross-origin" };
        doc = child;
      } catch (_) { return { refused: "cross-origin" }; }
    }
    return { doc };
  };
  const targets = [];
  for (const f of fills) {
    const destination = destinationFor(f.framePath || []);
    if (destination.refused) return refuse(destination.refused);
    const el = findIn(destination.doc, '[data-hermes-vault-slot="' + nonce + ':' + f.index + '"]')[0];
    if (el && documentOrigin(el.ownerDocument) !== expectedOrigin) return refuse(documentOrigin(el.ownerDocument));
    targets.push({ f, el });
  }
  let filled = 0;
  const norm = (t) => String(t || "").trim().toLowerCase();
  const originStillBound = (el) =>
    window.location.origin === expectedOrigin && documentOrigin(el.ownerDocument) === expectedOrigin;
  for (const { f, el } of targets) {
    if (!el || (f.token === "current-password" && el.type !== "password")) continue;
    try {
      if (el.tagName === "SELECT") {
        if (!originStillBound(el)) return refuse(documentOrigin(el.ownerDocument));
        const want = norm(f.value);
        const opt = Array.from(el.options).find((o) => [o.value, o.textContent].some((t) => norm(t) === want || norm(t) === want.replace(/^0/, "")));
        if (opt) {
          const view = el.ownerDocument.defaultView;
          el.value = opt.value;
          el.dispatchEvent(new view.Event("change", { bubbles: true, composed: true }));
          filled += 1;
        }
        continue;
      }
      el.focus();
      if (!originStillBound(el)) return refuse(documentOrigin(el.ownerDocument));
      const view = el.ownerDocument.defaultView;
      const setter = Object.getOwnPropertyDescriptor(view.HTMLInputElement.prototype, "value");
      // one-time-code split into single-character boxes: f.value is the slice for THIS box (see build_otp_fills)
      if (setter && setter.set) { setter.set.call(el, f.value); } else { el.value = f.value; }
      // composed: a component listening on its shadow host must see the value change too.
      el.dispatchEvent(new view.InputEvent("input", { bubbles: true, composed: true, inputType: "insertText" }));
      el.dispatchEvent(new view.Event("change", { bubbles: true, composed: true }));
      if (el.value.length > 0) filled += 1;
    } catch (e) { /* skip */ }
  }
  function cleanup() {
    const seen = new Set();
    const walkDocument = (doc) => {
      if (!doc || seen.has(doc) || documentOrigin(doc) !== expectedOrigin) return;
      seen.add(doc);
      for (const root of rootsFor(doc)) {
        for (const node of root.querySelectorAll("[data-hermes-vault-slot], [data-hermes-vault-frame]")) {
          if ((node.getAttribute("data-hermes-vault-slot") || "").startsWith(nonce + ":")) {
            node.removeAttribute("data-hermes-vault-slot");
          }
          if ((node.getAttribute("data-hermes-vault-frame") || "").startsWith(nonce + ":")) {
            node.removeAttribute("data-hermes-vault-frame");
          }
        }
        for (const frame of root.querySelectorAll("iframe")) {
          if (!frameAllowsSameOrigin(frame)) continue;
          try { walkDocument(frame.contentDocument); } catch (_) { /* inaccessible destination */ }
        }
      }
    };
    walkDocument(document);
  }
  cleanup();
  return JSON.stringify({ filled });
})()"""


_CLEANUP_JS_TEMPLATE = """(() => {
  const expectedOrigin = __EXPECTED_ORIGIN__;
  const nonce = __NONCE__;
  const frameAllowsSameOrigin = (frame) =>
    !frame.hasAttribute("sandbox") || frame.sandbox.contains("allow-same-origin");
  const documentOrigin = (doc) => {
    try {
      if (!doc || !doc.defaultView) return "";
      const view = doc.defaultView;
      const origin = view.location.origin;
      if (origin !== "null") return origin;
      const frame = view.frameElement;
      if ((view.location.href === "about:blank" || view.location.href === "about:srcdoc") && frame &&
          frameAllowsSameOrigin(frame) && frame.contentDocument === doc) return documentOrigin(frame.ownerDocument);
      return origin;
    }
    catch (_) { return ""; }
  };
  if (documentOrigin(document) !== expectedOrigin) return JSON.stringify({ cleaned: 0 });
  const seen = new Set();
  let cleaned = 0;
  const walkRoot = (root) => {
    for (const node of root.querySelectorAll("*")) {
      if ((node.getAttribute("data-hermes-vault-slot") || "").startsWith(nonce + ":")) {
        node.removeAttribute("data-hermes-vault-slot");
        cleaned += 1;
      }
      if ((node.getAttribute("data-hermes-vault-frame") || "").startsWith(nonce + ":")) {
        node.removeAttribute("data-hermes-vault-frame");
        cleaned += 1;
      }
      if (node.shadowRoot) walkRoot(node.shadowRoot);
      if (node.tagName !== "IFRAME" || !frameAllowsSameOrigin(node)) continue;
      try { walkDocument(node.contentDocument); } catch (_) { /* inaccessible destination */ }
    }
  };
  const walkDocument = (doc) => {
    if (!doc || seen.has(doc) || documentOrigin(doc) !== expectedOrigin) return;
    seen.add(doc);
    walkRoot(doc);
  };
  walkDocument(document);
  return JSON.stringify({ cleaned });
})()"""
