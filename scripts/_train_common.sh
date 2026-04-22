#!/bin/bash
set -euo pipefail

infer_n_gpus() {
    local cuda_devices="${1:-}"
    local n_gpus="${2:-}"

    if [ -n "$n_gpus" ]; then
        echo "$n_gpus"
        return
    fi

    if [ -n "$cuda_devices" ]; then
        python - "$cuda_devices" <<'PY'
import sys
devices = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
print(len(devices) if devices else 1)
PY
        return
    fi

    echo "1"
}

write_train_config_json() {
    local output_path="$1"
    shift

    mkdir -p "$(dirname "$output_path")"

    local env_args=()
    local key
    for key in "$@"; do
        env_args+=("$key=${!key-}")
    done

    env "${env_args[@]}" python - "$output_path" "$@" <<'PY'
import json
import os
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
keys = sys.argv[2:]
payload = {key: os.environ.get(key) for key in keys}
payload["_saved_from"] = "scripts/_train_common.sh"

output_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}
