# Tandem Adapter Service

This folder is the first implementation slice for direct Tandem ingestion using the local
[`tconnectsync`](/Users/anandparikh/Desktop/InSiteOfficial/tconnectsync) clone as the source-specific adapter.

The goal is not to run Tandem logic inside Chamelia or iOS. The goal is:

- reuse `tconnectsync` auth/session and Tandem parsing
- normalize Tandem data into the same canonical Firestore shapes already used by Nightscout
- keep InSite and Chamelia source-agnostic downstream

Current files:

- `canonical.py`
  - pure mapping layer from `tconnectsync` profile/event objects into canonical record dictionaries
- `worker.py`
  - service/worker implementation showing the connection and sync interface
- `store.py`
  - Secret Manager-backed connection store
- `firestore_writer.py`
  - canonical Firestore writer
- `service.py`
  - HTTP adapter service exposing `connect`, `sync`, `status`, and `health`
- `deploy/tandem-adapter.service`
  - systemd unit for a VM deployment

Target canonical outputs:

1. `therapy_settings_log`
2. `insulin_context/hourly/items/{hourStartUtc}`
3. `insulin_context/events/items/{eventId}`

This service now:

- authenticates to Tandem with `tconnectsync`
- fetches profiles / events / IOB
- maps them through `canonical.py`
- persists canonical records using the same downstream schema as Nightscout
- stores Tandem connection state in Secret Manager instead of a local JSON file

Operational direction:

- run as a small VM service or always-on worker
- keep credentials and session state outside Chamelia and outside Firestore
- have InSite call the service for:
  - connect
  - sync
  - status

Recommended deployment docs:

- [`Documents/tandem_adapter_vm_deploy.md`](/Users/anandparikh/Desktop/InSiteOfficial/Documents/tandem_adapter_vm_deploy.md)
- [`Documents/tconnectsync_integration_notes.md`](/Users/anandparikh/Desktop/InSiteOfficial/Documents/tconnectsync_integration_notes.md)
