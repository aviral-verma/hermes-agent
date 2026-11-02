"""The bot's browser tools obey the screen lease: while a human holds the shared browser, nothing is
dispatched, and a command whose run crossed a takeover loses its result."""

from __future__ import annotations

import json
import subprocess

import pytest

from tools.bot_desktop import lease, runtime


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    lease._reset_for_tests()
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":37"})
    yield
    lease._reset_for_tests()


def _wire_browser_exec(monkeypatch, run_cli):
    """Route browser_exec through the local session layer without starting a real browser."""
    from tools import browser_lightpanda, browser_session, browser_session_cdp, browser_session_cloud
    from tools import browser_use_cli as browser_use

    monkeypatch.setattr(browser_use, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(browser_use, "_base_subprocess_env", dict)
    monkeypatch.setattr(browser_use, "_real_profile_consented", lambda: False)
    monkeypatch.setattr(browser_lightpanda, "using_lightpanda_engine", lambda: False)
    monkeypatch.setattr(browser_session_cdp, "get_cdp_override", lambda: "")
    monkeypatch.setattr(browser_session_cdp, "get_cdp_override_raw", lambda: "")
    monkeypatch.setattr(browser_session_cloud, "get_cloud_provider", lambda: None)
    monkeypatch.setattr(browser_session, "get_session_info", lambda *_a, **_kw: {
        "session_name": "review", "cdp_url": "http://127.0.0.1:9222", "features": {"local": True}})
    monkeypatch.setattr(browser_use, "_attach_vault_supervisor", lambda *a: None)
    monkeypatch.setattr(browser_use, "_run_cli_killing_process_group", run_cli)
    return browser_use


def test_browser_exec_is_fenced_while_human_controls_shared_browser(monkeypatch):
    dispatched: list[str] = []

    def run_cli(*_args):
        dispatched.append("browser-use")
        return subprocess.CompletedProcess([], 0, "WHAT-THE-HUMAN-TYPED", "")

    browser_use = _wire_browser_exec(monkeypatch, run_cli)
    lease.acquire("human-viewer")
    raw = browser_use.browser_exec("print(page_info())", task_id="review")
    assert isinstance(raw, str)
    result = json.loads(raw)
    assert dispatched == [], "human holds the lease, yet browser_exec was dispatched"
    assert "WHAT-THE-HUMAN-TYPED" not in raw
    assert result.get("code") == "human_has_control"


def test_browser_exec_result_crossing_a_takeover_is_discarded(monkeypatch):
    def run_cli(*_args):
        lease.acquire("human-viewer")
        lease.release("human-viewer")
        return subprocess.CompletedProcess([], 0, "WHAT-THE-HUMAN-TYPED", "")

    browser_use = _wire_browser_exec(monkeypatch, run_cli)
    raw = browser_use.browser_exec("print(page_info())", task_id="review")
    assert isinstance(raw, str)
    result = json.loads(raw)
    assert "WHAT-THE-HUMAN-TYPED" not in raw
    assert result.get("code") == "human_has_control"


def test_run_fenced_refuses_a_local_session_while_human_controls_shared_browser():
    from tools.browser_session import run_fenced

    dispatched: list = []
    lease.acquire("human-viewer")
    result = run_fenced({"session_name": "review", "cdp_url": None, "features": {"local": True}},
                        lambda: dispatched.append("cmd") or {"success": True})
    assert dispatched == [], f"human holds the lease, yet a browser command was dispatched: {dispatched}"
    assert result.get("code") == "human_has_control"


def test_run_fenced_discards_a_result_crossing_a_takeover():
    from tools.browser_session import run_fenced

    def run_then_takeover():
        lease.acquire("human-viewer")
        lease.release("human-viewer")  # a full cycle, control is back — the frame is still theirs
        return {"success": True, "data": {"secret": "WHAT-THE-HUMAN-TYPED"}}

    result = run_fenced({"session_name": "review", "cdp_url": None, "features": {"local": True}},
                        run_then_takeover)
    assert "WHAT-THE-HUMAN-TYPED" not in json.dumps(result)


def test_nested_fence_keeps_the_original_epoch_after_takeover_and_release():
    from tools.browser_session import run_fenced

    session = {"session_name": "review", "cdp_url": None, "features": {"local": True}}
    dispatched = []

    def outer():
        lease.acquire("human-viewer")
        lease.release("human-viewer")
        return run_fenced(session, lambda: dispatched.append("page-access") or {"success": True})

    result = run_fenced(session, outer)

    assert result.get("code") == "human_has_control"
    assert dispatched == [], "a nested fence must not admit the epoch created after takeover"


def test_real_profile_local_browser_is_fenced_by_provenance_even_without_a_live_display(monkeypatch):
    """A real-profile session attaches over a loopback cdp_url but is launched with the Bot Desktop
    DISPLAY, so it IS the human's browser: the fence keys on the ``local`` feature, not on the
    transport. And a stranded human lease with the screen already down must still fence (computer_use
    does), not silently unfence the browser."""
    from tools.browser_session import run_fenced

    dispatched: list = []
    monkeypatch.setattr(runtime, "published_env", dict)
    lease.acquire("human-viewer")
    result = run_fenced({"session_name": "rp_1", "cdp_url": "ws://127.0.0.1:9222/devtools/browser/x",
                         "features": {"local": True, "real_profile": True}},
                        lambda: dispatched.append("cmd") or {"success": True})
    assert dispatched == [], f"human holds the lease, yet a real-profile browser command was dispatched: {dispatched}"
    assert result.get("code") == "human_has_control"


def test_vault_page_operations_respect_the_human_lease(monkeypatch):
    """The vault tools reach the page over the supervisor socket, outside `_run_browser_command`: while a
    human holds the screen they must be refused like every other page access, and never focus, inspect or
    write to the form the human is typing into."""
    from tools import browser_use_cli
    from tools import browser_vault_tool as vault

    route = browser_use_cli.BrowserExecRoute("", {"session_name": "review", "features": {"local": True}})
    monkeypatch.setattr(browser_use_cli, "_default_browser_exec_route", lambda _task_id: route)
    touched = []
    for name in ("browser_vault_fill", "browser_vault_enter_code", "browser_vault_save_login"):
        monkeypatch.setattr(vault, name, lambda *a, **k: touched.append(name) or json.dumps({"success": True}))
    lease.acquire("human")
    for action in ("fill", "enter_code", "save_login"):
        res = json.loads(vault._handle_vault({"action": action, "handle": "vault_x"}, task_id="default"))
        assert res["code"] == "human_has_control", action
    assert touched == [], "no vault page access while the human holds the screen"
    lease.release("human")
    assert json.loads(vault._handle_vault({"action": "fill", "handle": "vault_x"}, task_id="default"))["success"] is True


def _route_resolution_fakes(monkeypatch):
    from tools import browser_lightpanda, browser_session, browser_session_cdp, browser_use_cli

    monkeypatch.setattr(browser_session, "build_browser_env", dict)
    monkeypatch.setattr(browser_session_cdp, "get_cdp_override", lambda: "")
    monkeypatch.setattr(browser_session_cdp, "get_cdp_override_raw", lambda: "")
    monkeypatch.setattr(browser_lightpanda, "using_lightpanda_engine", lambda: False)
    monkeypatch.setattr(browser_use_cli, "_live_generation", lambda: 0)
    monkeypatch.setattr(browser_use_cli, "_read_browser_cfg", dict)
    monkeypatch.setattr(browser_use_cli, "_exec_route_is_sidecar", {})
    monkeypatch.setattr(browser_use_cli, "_exec_resolved_routes", {})
    return browser_session, browser_use_cli


def test_vault_fences_the_exact_hybrid_sidecar_route(monkeypatch):
    """The bare task's primary may be cloud while its sticky ``::local`` sidecar is the browser on the
    Bot Desktop screen. Vault must fence the sidecar returned by route resolution, not the cloud primary."""
    from tools import browser_session_cloud
    from tools import browser_vault_tool as vault

    browser_session, browser_use_cli = _route_resolution_fakes(monkeypatch)
    route_key = browser_use_cli._backend_cache_key("hybrid")
    browser_use_cli._exec_route_is_sidecar[route_key] = True
    monkeypatch.setattr(browser_use_cli, "_real_profile_consented", lambda: False)
    monkeypatch.setattr(browser_session_cloud, "get_cloud_provider", lambda: object())
    resolved = []
    page_operations = []

    def session_info(key):
        resolved.append(key)
        assert key == "hybrid::local"
        return {"session_name": "sidecar", "cdp_url": "ws://sidecar.invalid", "features": {"local": True}}

    monkeypatch.setattr(browser_session, "get_session_info", session_info)
    lease.acquire("human")

    result = json.loads(vault._fenced_page_op(
        "hybrid", lambda: page_operations.append("page") or json.dumps({"success": True})))

    assert resolved == ["hybrid::local"]
    assert result["code"] == "human_has_control"
    assert page_operations == []


def test_vault_fences_a_direct_real_profile_route(monkeypatch):
    """The real-profile route bypasses browser_session entirely, but its returned provenance is still local
    and must fence vault access."""
    from tools import browser_session_cloud, browser_session_real_profile
    from tools import browser_vault_tool as vault

    browser_session, browser_use_cli = _route_resolution_fakes(monkeypatch)
    monkeypatch.setattr(browser_use_cli, "_real_profile_consented", lambda: True)
    monkeypatch.setattr(browser_session_cloud, "get_cloud_provider", lambda: object())
    monkeypatch.setattr(browser_session_real_profile, "real_profile_cdp", lambda: ("ws://real.invalid", None))
    monkeypatch.setattr(browser_session, "get_session_info",
                        lambda _key: pytest.fail("direct real-profile route must not create a session-layer route"))
    route_key = browser_use_cli._backend_cache_key("real")
    env, error, _sidecar, _generation = browser_use_cli._route_exec_current(
        {}, "", "real", True, False, route_key)
    assert error is None
    route = browser_use_cli._remember_exec_route(route_key, browser_use_cli._route_from_env(env), _generation)
    assert route.session["features"] == {"local": True, "real_profile": True}
    page_operations = []
    lease.acquire("human")

    result = json.loads(vault._fenced_page_op(
        "real", lambda: page_operations.append("page") or json.dumps({"success": True})))

    assert result["code"] == "human_has_control"
    assert page_operations == []


def test_vault_none_task_uses_browser_exec_default_route_provenance(monkeypatch):
    """A missing task id is cache key ``browser-exec-default``. Looking up ``default`` misses this local
    session and used to let vault page access bypass the human lease."""
    from tools import browser_session_cloud
    from tools import browser_vault_tool as vault

    browser_session, browser_use_cli = _route_resolution_fakes(monkeypatch)
    monkeypatch.setattr(browser_use_cli, "_real_profile_consented", lambda: False)
    monkeypatch.setattr(browser_session_cloud, "get_cloud_provider", lambda: None)
    resolved = []

    def session_info(key):
        resolved.append(key)
        return {"session_name": "default", "cdp_url": "ws://default.invalid", "features": {"local": True}}

    monkeypatch.setattr(browser_session, "get_session_info", session_info)
    page_operations = []
    lease.acquire("human")

    result = json.loads(vault._fenced_page_op(
        None, lambda: page_operations.append("page") or json.dumps({"success": True})))

    assert resolved == ["browser-exec-default"]
    assert result["code"] == "human_has_control"
    assert page_operations == []


def test_vault_route_re_resolves_when_publish_races_old_route_insertion(monkeypatch):
    """An unnamed route resolved on generation E must not enter the cache after generation E+1 is
    published. Vault attach retries and receives the newly published endpoint."""
    _browser_session, browser_use_cli = _route_resolution_fakes(monkeypatch)

    generation = [1]
    resolved = []

    def route_current(_env, _session, _task_id, _local, _sidecar, _route_key):
        endpoint = {1: "ws://old.invalid", 2: "ws://new.invalid", 3: "ws://newest.invalid"}[generation[0]]
        resolved.append(endpoint)
        return {"BU_CDP_WS": endpoint}, None, False, generation[0]

    real_route_from_env = browser_use_cli._route_from_env

    def publish_after_old_resolve(env):
        route = real_route_from_env(env)
        if route.cdp_url == "ws://old.invalid":
            generation[0] = 2
        return route

    monkeypatch.setattr(browser_use_cli, "_live_generation", lambda: generation[0])
    monkeypatch.setattr(browser_use_cli, "_route_exec_current", route_current)
    monkeypatch.setattr(browser_use_cli, "_route_from_env", publish_after_old_resolve)

    route = browser_use_cli._default_browser_exec_route("vault-race")

    assert route.cdp_url == "ws://new.invalid"
    assert resolved == ["ws://old.invalid", "ws://new.invalid"]

    generation[0] = 3
    newest = browser_use_cli._default_browser_exec_route("vault-race")

    assert newest.cdp_url == "ws://newest.invalid"
    assert resolved == ["ws://old.invalid", "ws://new.invalid", "ws://newest.invalid"]


def test_secret_write_re_admits_after_a_takeover_during_the_code_prompt(monkeypatch):
    """`enter_code` blocks on the user's code INSIDE the handler fence. A takeover during that wait must
    refuse the write itself: the outer fence only discards the result afterwards, and by then the code is
    already in the page the human is typing into."""
    from tools import browser_use_cli
    from tools import browser_vault_tool as vault

    route = browser_use_cli.BrowserExecRoute("ws://browser.invalid", {"features": {"local": True}})
    token = vault._browser_exec_route.set(route)
    evaluated = []

    class Sup:
        def evaluate_runtime(self, expr):
            evaluated.append(expr)
            return {"ok": True, "result": "{}"}

    try:
        monkeypatch.setattr(vault, "_ensure_supervisor", lambda tid: Sup())
        assert vault._eval_js_secret("default", "fill()")["success"] is True  # agent holds: writes
        lease.acquire("human")  # takeover while the prompt was open
        res = vault._eval_js_secret("default", "fill()")
        assert res["success"] is False and res["error_type"] == "human_has_control"
        assert evaluated == ["fill()"], "the credential must not reach the page under a human lease"
    finally:
        vault._browser_exec_route.reset(token)


def _wire_prompt_page(monkeypatch, controls, origin="https://acme.test"):
    from tools import browser_use_cli
    from tools import browser_vault_tool as vault

    route = browser_use_cli.BrowserExecRoute(
        "ws://browser.invalid", {"cdp_url": "ws://browser.invalid", "features": {"local": True}})
    monkeypatch.setattr(browser_use_cli, "_default_browser_exec_route", lambda _task_id: route)
    events = []

    class Supervisor:
        def focus_page(self, requested_origin, *, accept=None):
            events.append(("focus", requested_origin))
            return {"ok": True, "url": origin + "/form"}

        def evaluate_runtime(self, expression):
            if expression == "window.location.href":
                events.append(("origin", expression))
                return {"ok": True, "result": origin + "/form"}
            events.append(("inspection-or-write-or-cleanup", expression))
            return {"ok": True, "result": json.dumps(controls)}

    monkeypatch.setattr(vault, "_ensure_supervisor", lambda _task_id: Supervisor())
    return vault, events


def test_takeover_during_save_login_prompt_blocks_later_focus_and_origin_reads(monkeypatch):
    from agent.vault_backends import unlock
    from agent.vault_store import get_vault_store

    vault, events = _wire_prompt_page(monkeypatch, [{"index": 0, "type": "password"}])
    before = {item.id for item in get_vault_store().list_items()}

    def prompt(_origin, _site):
        lease.acquire("human")
        lease.release("human")
        return {"identifier": "person@acme.test", "password": "save-prompt-secret"}

    unlock.set_save_login_prompt_callback(prompt)
    monkeypatch.setattr(unlock, "can_prompt_here", lambda: True)
    try:
        result = json.loads(vault._handle_vault({"action": "save_login"}, task_id="save"))
    finally:
        unlock.set_save_login_prompt_callback(None)

    assert result["code"] == "human_has_control"
    assert [kind for kind, _ in events] == ["focus", "origin"]
    assert {item.id for item in get_vault_store().list_items()} == before


def test_takeover_during_payment_prompt_blocks_inspection_cleanup_and_write(monkeypatch):
    from agent.vault_store import get_vault_store
    from tools import approval_prompt

    controls = [{"index": 0, "type": "text", "autocomplete": "cc-number"}]
    vault, events = _wire_prompt_page(monkeypatch, controls, origin="https://shop.test")
    meta = get_vault_store().add_item(
        "payment", "Visa", {"card_number": "4111111111111111", "exp_month": "01", "exp_year": "30",
                            "cvc": "123"}, origin="https://shop.test")

    def approve(*_args, **_kwargs):
        lease.acquire("human")
        lease.release("human")
        return "accept"

    monkeypatch.setattr(approval_prompt, "request_elicitation_consent", approve)
    result = json.loads(vault._handle_vault({"action": "fill", "handle": meta.id}, task_id="payment"))

    assert result["code"] == "human_has_control"
    assert events == [("focus", "https://shop.test")], "inspection and cleanup must not touch DOM after takeover"


def test_takeover_during_unlock_prompt_blocks_later_focus_and_origin_reads(monkeypatch):
    from agent import vault_backends
    from agent.vault_backends import unlock
    from agent.vault_store import VaultItemMeta

    vault, events = _wire_prompt_page(monkeypatch, [])
    meta = VaultItemMeta(id="manager:item", kind="login", label="Acme", origin="https://acme.test",
                         created_at="now", allowed_origins=("https://acme.test",))

    class Backend:
        name = "manager"
        display_name = "Manager"
        needs_unlock = True
        unlocked = False

        def is_unlocked(self):
            return self.unlocked

        def unlock(self, _master):
            self.unlocked = True

        def get_meta(self, _handle):
            return meta

    backend = Backend()

    def prompt(_name, _display_name):
        lease.acquire("human")
        lease.release("human")
        return "manager-master-secret"

    monkeypatch.setattr(vault_backends, "enabled_backends", lambda: [backend])
    monkeypatch.setattr(vault_backends, "backend_for_handle", lambda _handle: backend)
    monkeypatch.setattr(unlock, "can_prompt_here", lambda: True)
    unlock.set_unlock_prompt_callback(prompt)
    try:
        result = json.loads(vault._handle_vault(
            {"action": "fill", "handle": meta.id}, task_id="unlock"))
    finally:
        unlock.set_unlock_prompt_callback(None)

    assert result["code"] == "human_has_control"
    assert events == []
    assert backend.unlocked is False


def test_takeover_during_otp_prompt_blocks_write_and_marker_cleanup(monkeypatch):
    from agent.vault_backends import unlock

    controls = [{"index": 0, "type": "text", "name": "otp", "label": "Code",
                 "autocomplete": "one-time-code"}]
    vault, events = _wire_prompt_page(monkeypatch, controls)

    def prompt(_site, _hint):
        lease.acquire("human")
        lease.release("human")
        return "246810"

    monkeypatch.setattr(unlock, "can_prompt_here", lambda: True)
    unlock.set_code_prompt_callback(prompt)
    try:
        result = json.loads(vault._handle_vault({"action": "enter_code"}, task_id="otp"))
    finally:
        unlock.set_code_prompt_callback(None)

    assert result["code"] == "human_has_control"
    assert [kind for kind, _ in events] == ["focus", "origin", "inspection-or-write-or-cleanup"]
