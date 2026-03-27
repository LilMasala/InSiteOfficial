"""Firestore backend writer."""
from __future__ import annotations

from t1d_sim.writers.base_writer import BaseWriter


class FirebaseWriter(BaseWriter):
    def __init__(self, project: str) -> None:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.ApplicationDefault(), {"projectId": project})
        self.db = firestore.client()

    def write_patient(self, payload: dict) -> None:
        uid = payload["patient"]["patient_id"]
        for kind, entries in payload["firestore_docs"].items():
            for doc_id, data in entries:
                self.db.document(f"users/{uid}/{kind}/items/{doc_id}").set({k: v for k, v in data.items() if v is not None})

    def finalize(self) -> None:
        return
