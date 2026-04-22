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

infer_torchrun_port() {
    local requested_port="${1:-}"

    if [ -n "$requested_port" ]; then
        echo "$requested_port"
        return
    fi

    python - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

infer_torchrun_run_id() {
    local requested_id="${1:-}"

    if [ -n "$requested_id" ]; then
        echo "$requested_id"
        return
    fi

    python - <<'PY'
import uuid

print(uuid.uuid4().hex)
PY
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
