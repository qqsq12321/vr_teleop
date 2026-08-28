import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "example" / "pico4_daemon.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_tcp(port: int, process: subprocess.Popen[str]) -> socket.socket:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(f"daemon exited early: {stdout}{stderr}")
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=0.2)
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"daemon did not listen on port {port}")


def test_daemon_republishes_direct_tracking_to_local_relay() -> None:
    direct_port = _free_port()
    relay_port = _free_port()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--direct-port",
            str(direct_port),
            "--relay-port",
            str(relay_port),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        relay = _wait_for_tcp(relay_port, process)
        direct = _wait_for_tcp(direct_port, process)
        payload = b'{"Hand":{}}'
        frame = (
            bytes([0x3F, 0x6D])
            + struct.pack("<I", len(payload))
            + payload
            + struct.pack("<Q", 0)
            + bytes([0xA5])
        )
        relay.settimeout(0.1)
        deadline = time.monotonic() + 2
        data = b""
        while time.monotonic() < deadline and not data:
            direct.sendall(frame)
            try:
                data = relay.recv(4096)
            except TimeoutError:
                pass

        assert data
        device_id_length = struct.unpack_from("<I", data, 0)[0]
        json_length_offset = 4 + device_id_length
        json_length = struct.unpack_from("<I", data, json_length_offset)[0]

        assert data[4:json_length_offset] == b"pico4"
        assert data[json_length_offset + 4:json_length_offset + 4 + json_length] == payload
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_daemon_reports_direct_port_conflict() -> None:
    relay_port = _free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 0))
        direct_port = blocker.getsockname()[1]
        blocker.listen()
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--direct-port",
                str(direct_port),
                "--relay-port",
                str(relay_port),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=5,
        )

    assert result.returncode == 2
    assert "address already in use" in result.stderr
