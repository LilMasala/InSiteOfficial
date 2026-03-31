from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud import secretmanager


@dataclass
class TandemConnectionRecord:
    uid: str
    email: str
    password: str
    region: str = "US"
    pump_serial_number: str | None = None
    connected: bool = False
    needs_reauth: bool = False
    source: str = "tandem"
    last_successful_sync: str | None = None
    last_error: str | None = None


class TandemConnectionStore:
    """
    Secret Manager-backed Tandem connection store.

    Each user gets one secret containing the latest connection/session payload.
    The adapter service account should own access to these secrets on the VM.
    """

    def __init__(
        self,
        *,
        project_id: str | None = None,
        secret_prefix: str | None = None,
        client: secretmanager.SecretManagerServiceClient | None = None,
    ):
        self.project_id = (
            project_id
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT")
        )
        self.secret_prefix = secret_prefix or os.environ.get("TANDEM_SECRET_PREFIX", "tandem-conn-")
        self.client = client or secretmanager.SecretManagerServiceClient()

    def save(self, record: TandemConnectionRecord) -> TandemConnectionRecord:
        self._ensure_project_id()
        secret_name = self._secret_name(record.uid)
        parent = f"projects/{self.project_id}"
        payload = json.dumps(asdict(record), sort_keys=True).encode("utf-8")

        self._ensure_secret(secret_name, parent)
        self.client.add_secret_version(
            request={
                "parent": secret_name,
                "payload": {"data": payload},
            }
        )
        return record

    def load(self, uid: str) -> TandemConnectionRecord | None:
        self._ensure_project_id()
        secret_name = self._secret_name(uid)
        try:
            response = self.client.access_secret_version(
                request={"name": f"{secret_name}/versions/latest"}
            )
        except NotFound:
            return None
        raw = json.loads(response.payload.data.decode("utf-8"))
        return TandemConnectionRecord(**raw)

    def update_status(
        self,
        uid: str,
        *,
        connected: bool | None = None,
        needs_reauth: bool | None = None,
        last_successful_sync: datetime | None = None,
        last_error: str | None = None,
    ) -> TandemConnectionRecord | None:
        record = self.load(uid)
        if record is None:
            return None
        if connected is not None:
            record.connected = connected
        if needs_reauth is not None:
            record.needs_reauth = needs_reauth
        if last_successful_sync is not None:
            record.last_successful_sync = last_successful_sync.astimezone(UTC).isoformat().replace("+00:00", "Z")
        record.last_error = last_error
        return self.save(record)

    def _ensure_project_id(self) -> None:
        if not self.project_id:
            raise RuntimeError(
                "Missing GCP project configuration. Set GOOGLE_CLOUD_PROJECT on the Tandem adapter service."
            )

    def _secret_name(self, uid: str) -> str:
        safe_uid = re.sub(r"[^a-zA-Z0-9-_]", "-", uid)
        secret_id = f"{self.secret_prefix}{safe_uid}"
        return f"projects/{self.project_id}/secrets/{secret_id}"

    def _ensure_secret(self, secret_name: str, parent: str) -> None:
        secret_id = secret_name.rsplit("/", 1)[-1]
        try:
            self.client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": secret_id,
                    "secret": {
                        "replication": {"automatic": {}},
                        "labels": {
                            "service": "tandem-adapter",
                            "source": "tandem",
                        },
                    },
                }
            )
        except AlreadyExists:
            return
