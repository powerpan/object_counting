"""Verify the additive M2-v2 receipt while retaining all prior remote sources."""
from round3_tta_common import sources as tta_sources,read,sha,write_json,utc,assets,runtime


def sources(root):
    cfg,splits,previous=tta_sources(root)
    p=root/'provenance/round3_m2_receipt_v1.json';receipt=read(p)
    if not receipt['remote_main_verified'] or receipt['registration_commit']!=previous['registration_commit']:
        raise ValueError('M2-v2 was not registered and pushed')
    if len(receipt['code_commit'])!=40 or not receipt['files_sha256']:
        raise ValueError('Missing M2 implementation receipt')
    for n,h in receipt['files_sha256'].items():
        if sha(root/n)!=h:raise ValueError('M2 source differs: '+n)
    if any(receipt['files_sha256'].get(n)!=h for n,h in previous['files_sha256'].items()):
        raise ValueError('M2 changed frozen TTA sources')
    m2=read(root/'configs/round3_m2_v1.json')
    if m2['test_enabled'] or m2['stage_seconds']!=cfg['stage_seconds']['R3-5']:
        raise ValueError('M2 scope or stage budget changed')
    return cfg,splits,m2,{**receipt,'prior_receipts_sha256':previous['prior_receipts_sha256'],
                          'tta_receipt_sha256':previous['tta_receipt_sha256'],'m2_receipt_sha256':sha(p)}
