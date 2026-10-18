"""Persistent accounting for sequential M4 work; unresolved work fails closed."""
import json
import math
from pathlib import Path


def seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Expected finite, nonnegative elapsed seconds')
    if not math.isfinite(value) or value < 0:
        raise ValueError('Expected finite, nonnegative elapsed seconds')
    return float(value)


class BudgetLedger:
    """The caller must hold the shared M4 process lock for all mutations."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text())
        if self.data['version'] != 'round2-m4-ledger-v1':
            raise ValueError('Unknown budget ledger')
        self.cap = seconds(self.data['cap_seconds'])
        ids = [e['id'] for e in self.data['entries']]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate budget entry')
        for entry in self.data['entries']:
            seconds(entry['elapsed_seconds'])

    @classmethod
    def create(cls, path, cap, workflow, workflow_sha256):
        cap = seconds(cap)
        if workflow.get('state') != 'awaiting_validation':
            raise ValueError('Training controller has not completed its handoff')
        if not {'head-S-seed42', 'head-L-seed42'} <= set(workflow['completed']):
            raise ValueError('Both seed-42 heads must be complete')
        if 'stage_pid' in workflow:
            raise ValueError('Training controller still owns a child')
        elapsed = seconds(workflow['spent_seconds'])
        if elapsed >= cap:
            raise TimeoutError('M4 aggregate budget exhausted')
        # This entry ALREADY includes preflight and both heads. Never add the
        # per-head status.elapsed_seconds to it a second time.
        data = {'version': 'round2-m4-ledger-v1', 'cap_seconds': cap,
                'state': 'open', 'active': None, 'automatic_retry': False,
                'entries': [{'id': 'seed42-training-workflow', 'outcome': 'passed',
                             'elapsed_seconds': elapsed,
                             'evidence': {'workflow_budget_v1.json': workflow_sha256}}]}
        with Path(path).open('x') as stream:
            stream.write(json.dumps(data, indent=2) + '\n')
        return cls(path)

    @property
    def spent(self):
        return math.fsum(seconds(e['elapsed_seconds']) for e in self.data['entries'])

    @property
    def remaining(self):
        return max(0., self.cap - self.spent)

    def save(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.data, indent=2) + '\n')
        temporary.replace(self.path)

    def begin(self, stage, stage_cap, started_utc):
        if self.data['state'] != 'open' or self.data['active'] is not None:
            raise RuntimeError('Stopped or unresolved work must be inspected before continuing')
        if stage in {e['id'] for e in self.data['entries']}:
            raise FileExistsError('Budget stage has already been recorded')
        limit = min(seconds(stage_cap), self.remaining)
        if limit <= 10:
            raise TimeoutError('Insufficient M4 budget, including process shutdown allowance')
        self.data['active'] = {'id': stage, 'limit_seconds': limit, 'started_utc': started_utc}
        self.save()
        return limit

    def finish(self, stage, elapsed, passed, evidence, finished_utc):
        active = self.data['active']
        if active is None or active['id'] != stage:
            raise RuntimeError('Budget settlement does not match active work')
        elapsed = seconds(elapsed)
        accepted = bool(passed) and elapsed <= active['limit_seconds'] and elapsed <= self.remaining
        self.data['entries'].append({**active, 'elapsed_seconds': elapsed,
                                     'outcome': 'passed' if accepted else 'stopped',
                                     'evidence': evidence, 'finished_utc': finished_utc})
        self.data['active'] = None
        if not accepted:
            self.data['state'] = 'stopped'
        self.save()
        return accepted
