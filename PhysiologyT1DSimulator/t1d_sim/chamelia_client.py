"""HTTP client for the deployed Chamelia Cloud Run service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(slots=True)
class ChameliaError(RuntimeError):
    path: str
    status: int
    body: str

    def __str__(self) -> str:
        return f"Chamelia {self.path} returned {self.status}: {self.body}"


class ChameliaClient:
    def __init__(self, base_url: str, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health(self) -> bool:
        response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
        return response.ok

    def initialize(
        self,
        patient_id: str,
        preferences: dict[str, Any],
        weights_dir: str | None = None,
        *,
        bridge_url: str | None = None,
        bridge_mode: str | None = None,
        bridge_model_version: str | None = None,
        legacy_jepa_compat: bool | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "patient_id": patient_id,
            "preferences": preferences,
        }
        if weights_dir is not None:
            body["weights_dir"] = weights_dir
        if bridge_url is not None:
            body["bridge_url"] = bridge_url
        if bridge_mode is not None:
            body["bridge_mode"] = bridge_mode
        if bridge_model_version is not None:
            body["bridge_model_version"] = bridge_model_version
        if legacy_jepa_compat is not None:
            body["legacy_jepa_compat"] = legacy_jepa_compat
        return self._post("/chamelia_initialize_patient", body)

    def observe(self, patient_id: str, timestamp: float, signals: dict[str, Any]) -> dict[str, Any]:
        return self._post("/chamelia_observe", {
            "patient_id": patient_id,
            "timestamp": timestamp,
            "signals": signals,
        })

    def step(
        self,
        patient_id: str,
        timestamp: float,
        signals: dict[str, Any],
        connected_app_capabilities: dict[str, Any] | None = None,
        connected_app_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "patient_id": patient_id,
            "timestamp": timestamp,
            "signals": signals,
        }
        if connected_app_capabilities is not None:
            body["connected_app_capabilities"] = connected_app_capabilities
        if connected_app_state is not None:
            body["connected_app_state"] = connected_app_state
        return self._post("/chamelia_step", body)

    def record_outcome(
        self,
        patient_id: str,
        rec_id: int,
        response: str | None,
        signals: dict[str, Any],
        cost: float,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "patient_id": patient_id,
            "rec_id": rec_id,
            "signals": signals,
            "cost": cost,
        }
        if response is not None:
            body["response"] = response
        return self._post("/chamelia_record_outcome", body)

    def save(self, patient_id: str) -> dict[str, Any]:
        return self._post("/chamelia_save_patient", {"patient_id": patient_id})

    def load(
        self,
        patient_id: str,
        *,
        bridge_url: str | None = None,
        bridge_mode: str | None = None,
        bridge_model_version: str | None = None,
        legacy_jepa_compat: bool | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"patient_id": patient_id}
        if bridge_url is not None:
            body["bridge_url"] = bridge_url
        if bridge_mode is not None:
            body["bridge_mode"] = bridge_mode
        if bridge_model_version is not None:
            body["bridge_model_version"] = bridge_model_version
        if legacy_jepa_compat is not None:
            body["legacy_jepa_compat"] = legacy_jepa_compat
        return self._post("/chamelia_load_patient", body)

    def graduation_status(self, patient_id: str) -> dict[str, Any]:
        return self._post("/chamelia_graduation_status", {"patient_id": patient_id})

    def free(self, patient_id: str) -> dict[str, Any]:
        return self._post("/chamelia_free_patient", {"patient_id": patient_id})

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = self.session.post(url, json=body, timeout=self.timeout)
        if not response.ok:
            raise ChameliaError(path=path, status=response.status_code, body=response.text)
        return response.json()
