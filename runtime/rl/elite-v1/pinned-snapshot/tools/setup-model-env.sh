#!/usr/bin/env bash
# New project environment only. Existing project dependencies and model weights
# are read-only inputs; no original environment executable is used for install.
set -euo pipefail
target=/home/<user>/alpharush/.venv
source_env=/home/<user>/strategy_brain_sft_20260922/runtime/venv
if [[ ! -e "$target/pyvenv.cfg" ]]; then
    python3 -m venv "$target"
    cp -a --reflink=auto "$source_env/lib/python3.12/site-packages/." "$target/lib/python3.12/site-packages/"
fi
"$target/bin/python" -m pip check
"$target/bin/python" - <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
assert sys.prefix == '/home/<user>/alpharush/.venv'
versions = {name: metadata.version(name) for name in
            ('torch', 'transformers', 'accelerate', 'bitsandbytes', 'safetensors', 'mistral-common', 'tokenizers')}
receipt = {'prefix': sys.prefix, 'python': sys.version, 'versions': versions,
           'dependency_source': '/home/<user>/strategy_brain_sft_20260922/runtime/venv',
           'isolation': 'independent filesystem copy, no shared site-packages link',
           'old_environment_modified': False, 'learning_updates': 0}
Path('/home/<user>/alpharush/environment-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
PY
