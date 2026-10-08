"""Report observed attention implementations from local engine logs."""
import os
from pathlib import Path
import re


def attention_health():
    requested = os.environ.get('JEV_ATTENTION_BACKEND') or 'auto'
    text = ''
    path = os.environ.get('JEV_ATTENTION_LOG')
    if path:
        try:
            text = Path(path).read_text(encoding='utf-8', errors='replace')
        except OSError:
            pass
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    pattern = (r'\bUsing\s+(?:AttentionBackendEnum\.([A-Za-z][A-Za-z0-9_]*)\s+backend\b'
               r'|([A-Za-z][A-Za-z0-9_ -]*?)\s+attention\s+backend\b)')
    aliases = {'FLASH_ATTENTION': 'FLASH_ATTN', 'TRITON_ATTENTION': 'TRITON_ATTN',
               'FLASH_INFER': 'FLASHINFER', 'FLASHINFER_ATTENTION': 'FLASHINFER'}
    names = []
    for enum_name, plain_name in re.findall(pattern, text, re.I):
        name = (enum_name or plain_name).strip().upper().replace('-', '_').replace(' ', '_')
        names.append(aliases.get(name, name))
    names = list(dict.fromkeys(names))
    return {'attention_backend': names[-1] if names else None,
            'attention_backends': names,
            'attention_backend_requested': requested,
            'attention_backend_source': 'engine_log' if names else 'unreported'}
