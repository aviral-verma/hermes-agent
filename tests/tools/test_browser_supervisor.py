"""Integration tests for tools.browser_supervisor.

Exercises the supervisor end-to-end against a real local Chrome
(``--remote-debugging-port``).  Skipped when Chrome is not installed
— these are the tests that actually verify the CDP wire protocol
works, since mock-CDP unit tests can only prove the happy paths we
thought to model.

These tests spawn a **real Chrome process** on the machine running them.
They are therefore opt-in, twice over:

* ``@pytest.mark.integration`` — excluded by the default
  ``addopts = "-m 'not integration'"`` in ``pyproject.toml``, so a bare
  ``pytest`` cannot launch a browser on a developer's desktop by accident.
* ``HERMES_E2E_BROWSER=1`` — the env gate this docstring has always claimed.
  It previously existed only in this prose: nothing read the variable, and
  the sole real gate was "is a Chrome binary on PATH", which is true on most
  desktops and on ``ubuntu-latest``. Now it is enforced.

Run manually:
    HERMES_E2E_BROWSER=1 scripts/run_tests.sh -m integration \\
        tests/tools/test_browser_supervisor.py

(``scripts/run_tests.sh`` runs under ``env -i`` and forwards
``HERMES_E2E_BROWSER`` explicitly; ``-m integration`` overrides the default
marker filter.)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


_MAC_CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def _installed_chrome() -> str | None:
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        if path := shutil.which(candidate):
            return path
    # scripts/run_tests.sh deliberately supplies an isolated HOME. On macOS,
    # also discover Playwright's pinned Chromium under the account's real cache;
    # unlike the system Chrome singleton this always honors our disposable profile.
    if sys.platform == "darwin":
        import pwd

        account_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
        playwright = sorted(
            account_home.glob(
                "Library/Caches/ms-playwright/chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium"
            ),
            reverse=True,
        )
        if playwright:
            return str(playwright[0])
    return str(_MAC_CHROME) if _MAC_CHROME.is_file() else None


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("HERMES_E2E_BROWSER", "").strip() != "1",
        reason="real-browser E2E: set HERMES_E2E_BROWSER=1 to opt in",
    ),
    pytest.mark.skipif(
        not _installed_chrome(),
        reason="Chrome/Chromium not installed",
    ),
]


def _find_chrome() -> str:
    if path := _installed_chrome():
        return path
    pytest.skip("no Chrome binary found")


@pytest.fixture
def chrome_cdp(tmp_path):
    """Start a headless Chrome with --remote-debugging-port, yield its WS URL.

    Binds an ephemeral port (bind port 0, read back the real one) so
    concurrently-running test files in the per-file parallel runner can
    never collide on a fixed port.
    Always launches with ``--site-per-process`` so cross-origin iframes
    become real OOPIFs (needed by the iframe interaction tests).
    """
    profile = tempfile.mkdtemp(prefix="hermes-supervisor-test-")
    stderr = (tmp_path / "chrome.stderr").open("w+b")
    chrome = _find_chrome()
    browser_flags = ["--headless=new", "--disable-extensions"]
    launch_env = os.environ.copy()
    if "ms-playwright" in chrome and sys.platform == "darwin":
        import pwd

        # The pinned macOS Chromium stalls before opening CDP when HOME points
        # at run_tests.sh's synthetic home. Its profile and password store remain
        # disposable; HOME is restored only for framework startup resources.
        launch_env["HOME"] = pwd.getpwuid(os.getuid()).pw_dir
        browser_flags = [
            "--headless",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-background-networking",
            "--disable-component-update",
            "--password-store=basic",
            "--use-mock-keychain",
            "--disable-extensions",
        ]
    proc = subprocess.Popen(
        [
            chrome,
            "--remote-debugging-port=0",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            *browser_flags,
            "--disable-gpu",
            "--site-per-process",  # force OOPIFs for cross-origin iframes
        ],
        stdout=subprocess.DEVNULL,
        stderr=stderr,
        env=launch_env,
    )

    ws_url = None
    port = None
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            port = int((Path(profile) / "DevToolsActivePort").read_text(encoding="utf-8").splitlines()[0])
            import urllib.request
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/version", timeout=1
            ) as r:
                info = json.loads(r.read().decode())
                ws_url = info["webSocketDebuggerUrl"]
                break
        except Exception:
            time.sleep(0.25)
    if ws_url is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, AssertionError, Exception):
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except (AssertionError, Exception):
                pass
        shutil.rmtree(profile, ignore_errors=True)
        stderr.seek(0)
        diagnostic = stderr.read().decode("utf-8", errors="replace")
        stderr.close()
        pytest.fail(f"Chrome didn't expose CDP in time: {diagnostic}")

    yield ws_url, port

    # Tear down Chrome. The stdlib `subprocess._wait()` POSIX implementation
    # has a known race (https://bugs.python.org/issue38630): when SIGCHLD
    # arrives concurrently with `proc.wait()`, `_try_wait(WNOHANG)` can
    # return a foreign pid and the `assert pid == self.pid or pid == 0`
    # fires. We saw this in CI on slice 1 after this fixture's teardown
    # (PR #33661 follow-up). Swallow the stdlib race + force-kill if wait
    # hangs, then always reap so we don't leak a zombie.
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=3)
    except (subprocess.TimeoutExpired, AssertionError, Exception):
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except (AssertionError, Exception):
            pass
    shutil.rmtree(profile, ignore_errors=True)
    stderr.close()


def _test_page_url() -> str:
    html = """<!doctype html>
<html><head><title>Supervisor pytest</title></head><body>
<h1>Supervisor pytest</h1>
<iframe id="inner" srcdoc="<body><h2>frame-marker</h2></body>" width="400" height="100"></iframe>
</body></html>"""
    return "data:text/html;base64," + base64.b64encode(html.encode()).decode()


def _fire_on_page(cdp_url: str, expression: str, page_url: str | None = None) -> None:
    """Navigate the first page target to a data URL and fire `expression`."""
    import websockets as _ws_mod

    async def run():
        async with _ws_mod.connect(cdp_url, max_size=50 * 1024 * 1024) as ws:
            next_id = [1]

            async def call(method, params=None, session_id=None):
                cid = next_id[0]
                next_id[0] += 1
                p = {"id": cid, "method": method}
                if params:
                    p["params"] = params
                if session_id:
                    p["sessionId"] = session_id
                await ws.send(json.dumps(p))
                async for raw in ws:
                    m = json.loads(raw)
                    if m.get("id") == cid:
                        return m

            targets = (await call("Target.getTargets"))["result"]["targetInfos"]
            page = next(t for t in targets if t.get("type") == "page")
            attach = await call(
                "Target.attachToTarget", {"targetId": page["targetId"], "flatten": True}
            )
            sid = attach["result"]["sessionId"]
            await call("Page.navigate", {"url": page_url or _test_page_url()}, session_id=sid)
            await asyncio.sleep(1.5)  # let the page load
            await call(
                "Runtime.evaluate",
                {"expression": expression, "returnByValue": True},
                session_id=sid,
            )

    asyncio.run(run())


def _create_loaded_target(cdp_url: str, url: str) -> None:
    """Create another synthetic tab and wait for its load event without touching a user browser."""
    import websockets as _ws_mod

    async def run():
        async with _ws_mod.connect(cdp_url, max_size=50 * 1024 * 1024) as ws:
            next_id = 1

            async def call(method, params=None, session_id=None, events=None):
                nonlocal next_id
                cid = next_id
                next_id += 1
                payload = {"id": cid, "method": method}
                if params:
                    payload["params"] = params
                if session_id:
                    payload["sessionId"] = session_id
                await ws.send(json.dumps(payload))
                async for raw in ws:
                    message = json.loads(raw)
                    if message.get("id") == cid:
                        return message
                    if events is not None:
                        events.append(message)

            created = await call("Target.createTarget", {"url": "about:blank"})
            attached = await call("Target.attachToTarget", {
                "targetId": created["result"]["targetId"],
                "flatten": True,
            })
            sid = attached["result"]["sessionId"]
            events = []
            await call("Page.enable", session_id=sid, events=events)
            await call("Page.navigate", {"url": url}, session_id=sid, events=events)
            if any(message.get("sessionId") == sid and message.get("method") == "Page.loadEventFired"
                   for message in events):
                return
            async for raw in ws:
                message = json.loads(raw)
                if message.get("sessionId") == sid and message.get("method") == "Page.loadEventFired":
                    return

    asyncio.run(run())


@pytest.fixture
def synthetic_vault_sites():
    """Two loopback HTTP origins serving only in-memory dummy form fixtures."""
    pages: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = pages.get(self.path, "<h1>not found</h1>").encode()
            self.send_response(200 if self.path in pages else 404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, message, *args):
            return

    servers = [ThreadingHTTPServer(("127.0.0.1", 0), Handler) for _ in range(2)]
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
    for thread in threads:
        thread.start()
    bases = [f"http://127.0.0.1:{server.server_port}" for server in servers]
    try:
        yield bases[0], bases[1], pages
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)


@pytest.fixture
def supervisor_registry():
    """Yield the global registry and tear down any supervisors after the test."""
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    yield SUPERVISOR_REGISTRY
    SUPERVISOR_REGISTRY.stop_all()


def _wait_for_dialog(supervisor, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = supervisor.snapshot()
        if snap.pending_dialogs:
            return snap.pending_dialogs
        time.sleep(0.1)
    return ()


def test_supervisor_start_and_snapshot(chrome_cdp, supervisor_registry):
    """Supervisor attaches, exposes an active snapshot with a top frame."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-1", cdp_url=cdp_url)

    # Navigate so the frame tree populates.
    _fire_on_page(cdp_url, "/* no dialog */ void 0")

    # Give a moment for frame events to propagate
    time.sleep(1.0)
    snap = supervisor.snapshot()
    assert snap.active is True
    assert snap.task_id == "pytest-1"
    assert snap.pending_dialogs == ()
    # At minimum a top frame should exist after the navigate.
    assert snap.frame_tree.get("top") is not None


def test_main_frame_alert_detection_and_dismiss(chrome_cdp, supervisor_registry):
    """alert() in the main frame surfaces and can be dismissed via the sync API."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-2", cdp_url=cdp_url)

    _fire_on_page(cdp_url, "setTimeout(() => alert('PYTEST-MAIN-ALERT'), 50)")
    dialogs = _wait_for_dialog(supervisor)
    assert dialogs, "no dialog detected"
    d = dialogs[0]
    assert d.type == "alert"
    assert "PYTEST-MAIN-ALERT" in d.message

    result = supervisor.respond_to_dialog("dismiss")
    assert result["ok"] is True
    # State cleared after dismiss
    time.sleep(0.3)
    assert supervisor.snapshot().pending_dialogs == ()


def test_iframe_contentwindow_alert(chrome_cdp, supervisor_registry):
    """alert() fired from inside a same-origin iframe surfaces too."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-3", cdp_url=cdp_url)

    _fire_on_page(
        cdp_url,
        "setTimeout(() => document.querySelector('#inner').contentWindow.alert('PYTEST-IFRAME'), 50)",
    )
    dialogs = _wait_for_dialog(supervisor)
    assert dialogs, "no iframe dialog detected"
    assert any("PYTEST-IFRAME" in d.message for d in dialogs)

    result = supervisor.respond_to_dialog("accept")
    assert result["ok"] is True


def test_prompt_dialog_with_response_text(chrome_cdp, supervisor_registry):
    """prompt() gets our prompt_text back inside the page."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-4", cdp_url=cdp_url)

    # Fire a prompt and stash the answer on window
    _fire_on_page(
        cdp_url,
        "setTimeout(() => { window.__promptResult = prompt('give me a token', 'default-x'); }, 50)",
    )
    dialogs = _wait_for_dialog(supervisor)
    assert dialogs
    d = dialogs[0]
    assert d.type == "prompt"
    assert d.default_prompt == "default-x"

    result = supervisor.respond_to_dialog("accept", prompt_text="PYTEST-PROMPT-REPLY")
    assert result["ok"] is True


def test_browser_dialog_tool_end_to_end(chrome_cdp, supervisor_registry):
    """Full agent-path check: fire an alert, call the tool handler directly."""
    from tools.browser_dialog_tool import browser_dialog

    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-tool", cdp_url=cdp_url)

    _fire_on_page(cdp_url, "setTimeout(() => alert('PYTEST-TOOL-END2END'), 50)")
    assert _wait_for_dialog(supervisor), "no dialog detected via wait_for_dialog"

    r = json.loads(browser_dialog(action="dismiss", task_id="pytest-tool"))
    assert r["success"] is True
    assert r["action"] == "dismiss"
    assert "PYTEST-TOOL-END2END" in r["dialog"]["message"]



def test_evaluate_runtime_unserializable_value(chrome_cdp, supervisor_registry):
    """``Infinity``/``NaN``/``BigInt`` come back via ``unserializableValue``."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-eval-5", cdp_url=cdp_url)

    _fire_on_page(cdp_url, "void 0")
    time.sleep(0.5)

    out = supervisor.evaluate_runtime("Infinity")
    assert out["ok"] is True
    assert out["result"] == "Infinity"


def _evaluated_json(supervisor, expression: str):
    result = supervisor.evaluate_runtime(expression)
    assert result["ok"] is True, result
    value = result["result"]
    return json.loads(value) if isinstance(value, str) else value


def test_vault_fill_keeps_top_level_behavior(chrome_cdp, supervisor_registry, synthetic_vault_sites):
    from agent.vault_login_classifier import build_fill_js, build_inspection_js

    base, _, pages = synthetic_vault_sites
    pages["/top"] = ('<input id="password" type="password" autocomplete="current-password">'
                     '<iframe id="unused" src="/unused"></iframe>')
    pages["/unused"] = '<input id="unused-password" type="password" autocomplete="current-password">'
    cdp_url, _ = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="vault-top", cdp_url=cdp_url)
    _fire_on_page(cdp_url, "void 0", f"{base}/top")

    controls = _evaluated_json(supervisor, build_inspection_js("top-nonce"))
    assert len(controls) == 2 and controls[0]["framePath"] == []
    filled = _evaluated_json(supervisor, build_fill_js([{
        "index": controls[0]["index"],
        "token": "current-password",
        "value": "dummy-top-password",
    }], expected_origin=base, nonce="top-nonce"))

    assert filled == {"filled": 1}
    assert supervisor.evaluate_runtime("document.querySelector('#password').value")["result"] == "dummy-top-password"
    markers = ("document.querySelectorAll('[data-hermes-vault-slot], [data-hermes-vault-frame]').length + "
               "document.querySelector('#unused').contentDocument"
               ".querySelectorAll('[data-hermes-vault-slot], [data-hermes-vault-frame]').length")
    assert supervisor.evaluate_runtime(markers)["result"] == 0


def test_vault_fill_reaches_nested_iframe_inside_shadow_root(
    chrome_cdp, supervisor_registry, synthetic_vault_sites
):
    from agent.vault_login_classifier import build_fill_js, build_inspection_js

    base, _, pages = synthetic_vault_sites
    pages["/nested"] = """<div id="host"></div><script>
      const frame = document.createElement('iframe'); frame.src = '/middle';
      document.querySelector('#host').attachShadow({mode: 'open'}).append(frame);
    </script>"""
    pages["/middle"] = ('<iframe id="inner" srcdoc="&lt;input id=&quot;password&quot; type=&quot;password&quot; '
                        'autocomplete=&quot;current-password&quot;&gt;"></iframe>')
    cdp_url, _ = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="vault-nested", cdp_url=cdp_url)
    _fire_on_page(cdp_url, "void 0", f"{base}/nested")

    controls = _evaluated_json(supervisor, build_inspection_js("nested-nonce"))
    assert len(controls) == 1 and len(controls[0]["framePath"]) == 2
    filled = _evaluated_json(supervisor, build_fill_js([{
        "index": controls[0]["index"],
        "token": "current-password",
        "value": "dummy-nested-password",
        "frame_path": tuple(controls[0]["framePath"]),
    }], expected_origin=base, nonce="nested-nonce"))

    assert filled == {"filled": 1}
    read_value = "document.querySelector('#host').shadowRoot.querySelector('iframe').contentDocument" \
                 ".querySelector('#inner').contentDocument.querySelector('#password').value"
    assert supervisor.evaluate_runtime(read_value)["result"] == "dummy-nested-password"
    marker_count = ("document.querySelectorAll('[data-hermes-vault-slot], [data-hermes-vault-frame]').length + "
                    "document.querySelector('#host').shadowRoot"
                    ".querySelectorAll('[data-hermes-vault-slot], [data-hermes-vault-frame]').length + "
                    "document.querySelector('#host').shadowRoot.querySelector('iframe').contentDocument"
                    ".querySelectorAll('[data-hermes-vault-slot], [data-hermes-vault-frame]').length + "
                    "document.querySelector('#host').shadowRoot.querySelector('iframe').contentDocument"
                    ".querySelector('#inner').contentDocument"
                    ".querySelectorAll('[data-hermes-vault-slot], [data-hermes-vault-frame]').length")
    assert supervisor.evaluate_runtime(marker_count)["result"] == 0


def test_vault_probe_selects_the_same_origin_iframe_tab(
    chrome_cdp, supervisor_registry, synthetic_vault_sites
):
    from tools import browser_vault_tool
    from tools.browser_use_cli import BrowserExecRoute

    base, _, pages = synthetic_vault_sites
    pages["/plain"] = "<h1>No login here</h1>"
    pages["/framed"] = '<iframe src="/password"></iframe>'
    pages["/password"] = '<input type="password" autocomplete="current-password">'
    cdp_url, _ = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="vault-tabs", cdp_url=cdp_url)
    _fire_on_page(cdp_url, "void 0", f"{base}/plain")
    _create_loaded_target(cdp_url, f"{base}/framed")

    route = BrowserExecRoute(cdp_url=cdp_url, session={"cdp_url": cdp_url, "features": {}})
    token = browser_vault_tool._browser_exec_route.set(route)
    try:
        assert browser_vault_tool._focus_bound_origin("vault-tabs", base, "login") == base
        assert supervisor.evaluate_runtime("window.location.pathname")["result"] == "/framed"
    finally:
        browser_vault_tool._browser_exec_route.reset(token)


def test_vault_skips_cross_origin_opaque_and_unsafe_sandbox_frames(
    chrome_cdp, supervisor_registry, synthetic_vault_sites
):
    from agent.vault_login_classifier import build_inspection_js
    from tools.browser_vault_tool import _TAB_PROBES

    base, other, pages = synthetic_vault_sites
    opaque = base64.b64encode(b'<input type="password" autocomplete="current-password">').decode()
    pages["/blocked"] = f"""
      <iframe src="{other}/password"></iframe>
      <iframe src="data:text/html;base64,{opaque}"></iframe>
      <iframe sandbox src="/password"></iframe>
    """
    pages["/password"] = '<input type="password" autocomplete="current-password">'
    cdp_url, _ = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="vault-blocked", cdp_url=cdp_url)
    _fire_on_page(cdp_url, "void 0", f"{base}/blocked")

    assert _evaluated_json(supervisor, build_inspection_js("blocked-nonce")) == []
    probe = supervisor.evaluate_runtime(_TAB_PROBES["login"])
    assert probe["ok"] is True and probe["result"] is False


def test_vault_refuses_when_the_destination_frame_changes_origin(
    chrome_cdp, supervisor_registry, synthetic_vault_sites
):
    from agent.vault_login_classifier import build_fill_js, build_inspection_js

    base, other, pages = synthetic_vault_sites
    pages["/change"] = '<iframe id="destination" src="/password"></iframe>'
    pages["/password"] = '<input type="password" autocomplete="current-password">'
    cdp_url, _ = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="vault-change", cdp_url=cdp_url)
    _fire_on_page(cdp_url, "void 0", f"{base}/change")
    [control] = _evaluated_json(supervisor, build_inspection_js("change-nonce"))

    navigated = supervisor.evaluate_runtime(
        "new Promise((resolve) => { const frame = document.querySelector('#destination'); "
        f"frame.onload = () => resolve(true); frame.src = {json.dumps(other + '/password')}; }})"
    )
    assert navigated["ok"] is True and navigated["result"] is True
    refused = _evaluated_json(supervisor, build_fill_js([{
        "index": control["index"],
        "token": "current-password",
        "value": "dummy-refused-password",
        "frame_path": tuple(control["framePath"]),
    }], expected_origin=base, nonce="change-nonce"))

    assert refused["refused"] == "origin_changed"
    assert refused["found"] in {"cross-origin", "unavailable"}
    assert supervisor.evaluate_runtime(
        "document.querySelectorAll('[data-hermes-vault-slot], [data-hermes-vault-frame]').length"
    )["result"] == 0
