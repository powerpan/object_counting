"""Continue the stage ledger in a new file, preserving the audited M3 snapshot."""
import hashlib
import json
from pathlib import Path
from planb.round3_accounting import StageLedger


def inherit_ledger(source, destination, expected_sha, caps, total):
    source, destination = Path(source), Path(destination)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Previous stage ledger changed')
    previous = StageLedger(source)
    if previous.data['active'] is not None or previous.data['state'] != 'open':
        raise RuntimeError('Previous stage has unsettled or failed work')
    if previous.data['caps'] != caps or previous.data['total_cap'] != total:
        raise ValueError('Stage limits changed')
    data = json.loads(raw)
    data['inherited_ledger_sha256'] = expected_sha
    data['inherited_entry_count'] = len(data['entries'])
    with destination.open('x') as f:
        json.dump(data, f, indent=2)
    return StageLedger(destination)
