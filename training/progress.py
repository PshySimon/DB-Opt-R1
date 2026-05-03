import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime


def progress_enabled() -> bool:
    value = os.environ.get("GRPO_PROGRESS_LOG", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _progress_line(message: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"[progress] {timestamp} {message}"


def _write_progress_file(line: str) -> None:
    path = os.environ.get("GRPO_PROGRESS_LOG_FILE")
    if not path:
        return

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "a", encoding="utf-8", buffering=1) as handle:
        handle.write(line + "\n")
        handle.flush()


def progress_log(message: str) -> None:
    if not progress_enabled():
        return

    line = _progress_line(message)
    print(line, flush=True)
    _write_progress_file(line)


@contextmanager
def progress_heartbeat(message: str):
    if not progress_enabled():
        yield
        return

    interval = float(os.environ.get("GRPO_PROGRESS_HEARTBEAT_INTERVAL", "5"))
    if interval <= 0:
        yield
        return

    stop_event = threading.Event()
    start_time = time.perf_counter()

    def emit_heartbeat():
        while not stop_event.wait(interval):
            elapsed = time.perf_counter() - start_time
            progress_log(f"{message} still_running elapsed_s={elapsed:.1f}")

    thread = threading.Thread(target=emit_heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=0.2)
