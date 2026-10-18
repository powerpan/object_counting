"""Additional immutable source receipt for R3-2; previous evidence remains intact."""
from round3_common import sources as m3_sources, read, sha, write_json, utc, assets, runtime


def sources(root):
    cfg, splits, previous = m3_sources(root)
    path = root/'provenance/round3_tta_receipt_v1.json'
    receipt = read(path)
    if not receipt['remote_main_verified'] or receipt['registration_commit'] != previous['registration_commit']:
        raise ValueError('TTA has not been preregistered and pushed')
    if len(receipt['code_commit']) != 40 or not receipt['files_sha256']:
        raise ValueError('Missing TTA implementation receipt')
    for name, digest in receipt['files_sha256'].items():
        if sha(root/name) != digest:
            raise ValueError('TTA source receipt differs: '+name)
    return cfg, splits, {**receipt, 'prior_receipts_sha256':previous['receipts_sha256'],
                         'tta_receipt_sha256':sha(path)}
