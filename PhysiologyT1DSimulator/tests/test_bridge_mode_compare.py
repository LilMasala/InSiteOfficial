from pathlib import Path

from scripts.compare_chamelia_modes import build_run_command, build_variant_specs
from t1d_sim.chamelia_client import ChameliaClient


class _Args:
    modes = ["v1.1", "v1.5", "v3"]
    include_legacy_jepa_compat = True
    namespace_prefix = "cmp"
    comparison_label = "athlete-42"
    uid_prefix = "cmp-user"
    local_root = "/tmp/local-runs"
    persona = "athlete"
    days = 14
    seed = 42
    chamelia_url = "http://127.0.0.1:8080"
    timeout = 60
    profile_policy = "single"
    coldstart_targets = "synthetic"
    python_bridge_url = "http://127.0.0.1:8090"
    bridge_model_version = "bridge-sim-v1"
    weights_dir = "/tmp/weights"
    verbose = False


def test_chamelia_client_initialize_and_load_include_bridge_runtime_fields():
    client = ChameliaClient("http://example.test")
    calls: list[tuple[str, dict]] = []

    def fake_post(path: str, body: dict):
        calls.append((path, body))
        return {"ok": True}

    client._post = fake_post  # type: ignore[method-assign]

    client.initialize(
        "patient-1",
        {"persona": "athlete"},
        weights_dir="/tmp/weights",
        bridge_url="http://bridge.test",
        bridge_mode="v3",
        bridge_model_version="bridge-model-v2",
        legacy_jepa_compat=True,
    )
    client.load(
        "patient-1",
        bridge_url="http://bridge.test",
        bridge_mode="v1.5",
        bridge_model_version="bridge-model-v3",
        legacy_jepa_compat=False,
    )

    init_path, init_body = calls[0]
    assert init_path == "/chamelia_initialize_patient"
    assert init_body["bridge_url"] == "http://bridge.test"
    assert init_body["bridge_mode"] == "v3"
    assert init_body["bridge_model_version"] == "bridge-model-v2"
    assert init_body["legacy_jepa_compat"] is True

    load_path, load_body = calls[1]
    assert load_path == "/chamelia_load_patient"
    assert load_body["bridge_url"] == "http://bridge.test"
    assert load_body["bridge_mode"] == "v1.5"
    assert load_body["bridge_model_version"] == "bridge-model-v3"
    assert load_body["legacy_jepa_compat"] is False


def test_compare_mode_variants_include_optional_legacy_run():
    variants = build_variant_specs(_Args())
    assert [variant["label"] for variant in variants] == ["v1.1", "v1.5", "v3", "legacy-jepa-compat"]
    assert variants[-1]["bridge_mode"] == "v3"
    assert variants[-1]["legacy_jepa_compat"] is True


def test_compare_mode_command_uses_isolated_namespace_and_runtime_flags(tmp_path: Path):
    args = _Args()
    variant = {"label": "v3", "bridge_mode": "v3", "legacy_jepa_compat": False}
    report_path = tmp_path / "report.json"

    command = build_run_command(args, variant, report_path=report_path)

    assert "--no-firebase" in command
    assert "--bridge-mode" in command
    assert command[command.index("--bridge-mode") + 1] == "v3"
    assert "--bridge-model-version" in command
    assert command[command.index("--bridge-model-version") + 1] == "bridge-sim-v1"
    assert "--python-bridge-url" in command
    assert command[command.index("--python-bridge-url") + 1] == "http://127.0.0.1:8090"
    assert "--uid" in command
    assert command[command.index("--uid") + 1] == "cmp-user-v3"
    assert "--namespace" in command
    assert command[command.index("--namespace") + 1] == "cmp-athlete-42-v3"
    assert command[command.index("--report-file") + 1] == str(report_path)
