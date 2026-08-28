"""Keep Pico 4 connected and relay tracking data to local applications.

The daemon broadcasts this PC's USB/Wi-Fi addresses, accepts the Pico client
on TCP 63901, and republishes tracking JSON to local clients on TCP 63902.
"""

from __future__ import annotations

import argparse
import errno
import logging
import socket
import struct
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.adapters.input.pico4 import (
    _CMD_BATTERY,
    _CMD_CONNECT,
    _CMD_DEVICE_STATE_JSON,
    _CMD_HEARTBEAT,
    _CMD_SENSOR,
    _DirectFrameParser,
    _build_broadcast_packet,
    _get_local_ips,
)

logger = logging.getLogger("pico4_daemon")

DEFAULT_DIRECT_PORT = 63901
DEFAULT_RELAY_HOST = "127.0.0.1"
DEFAULT_RELAY_PORT = 63902
DEFAULT_BROADCAST_PORT = 29888
DEFAULT_DEVICE_ID = "pico4"
HEARTBEAT_TIMEOUT_S = 20.0
BROADCAST_INTERVAL_S = 5.0


def encode_relay_frame(device_id: str, payload: bytes) -> bytes:
    device_id_bytes = device_id.encode("utf-8")
    return (
        struct.pack("<I", len(device_id_bytes))
        + device_id_bytes
        + struct.pack("<I", len(payload))
        + payload
    )


class PortBindingError(RuntimeError):
    def __init__(self, failures: list[tuple[str, str, int, OSError]]) -> None:
        details = []
        for role, host, port, exc in failures:
            reason = (
                "address already in use"
                if exc.errno == errno.EADDRINUSE
                else exc.strerror or str(exc)
            )
            details.append(f"  - {role} TCP {host}:{port}: {reason}")
        super().__init__(
            "Cannot start Pico 4 daemon:\n"
            + "\n".join(details)
            + "\nCheck for an existing process with: "
            "ss -ltnp | grep -E ':(63901|63902)\\b'"
        )


class RelayHub:
    def __init__(self, host: str, port: int, device_id: str) -> None:
        self.host = host
        self.port = port
        self._device_id = device_id
        self._stop = threading.Event()
        self._clients: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def bind(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.host, self.port))
            server.listen(8)
            server.settimeout(1.0)
        except OSError:
            server.close()
            raise
        self._server = server

    def start(self) -> None:
        if self._server is None:
            raise RuntimeError("RelayHub.bind() must be called before start()")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        server, self._server = self._server, None
        if server is not None:
            server.close()
        with self._lock:
            for client in self._clients:
                client.close()
            self._clients.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def publish(self, payload: bytes) -> None:
        frame = encode_relay_frame(self._device_id, payload)
        dead_clients = []
        with self._lock:
            for client in self._clients:
                try:
                    client.sendall(frame)
                except OSError:
                    dead_clients.append(client)
            for client in dead_clients:
                self._clients.discard(client)
                client.close()

    def _run(self) -> None:
        server = self._server
        if server is None:
            return
        logger.info("Relay hub listening on %s:%d", self.host, self.port)
        while not self._stop.is_set():
            try:
                client, address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            logger.info("Relay client connected: %s", address)
            with self._lock:
                self._clients.add(client)


class Pico4Daemon:
    def __init__(
        self,
        direct_port: int,
        relay_host: str,
        relay_port: int,
        broadcast_port: int,
        device_id: str,
    ) -> None:
        self._direct_port = direct_port
        self._broadcast_port = broadcast_port
        self._hub = RelayHub(relay_host, relay_port, device_id)
        self._stop = threading.Event()

    def run(self) -> None:
        server, failures = self._bind_servers()
        if failures:
            raise PortBindingError(failures)
        assert server is not None

        self._hub.start()
        broadcaster = threading.Thread(target=self._broadcast_loop, daemon=True)
        broadcaster.start()
        logger.info("Direct server listening on 0.0.0.0:%d", self._direct_port)

        try:
            while not self._stop.is_set():
                try:
                    client, address = server.accept()
                except socket.timeout:
                    continue
                logger.info("Pico 4 connected from %s", address)
                self._handle_direct_client(client)
                logger.info("Pico 4 disconnected")
        except KeyboardInterrupt:
            logger.info("Stopping daemon...")
        finally:
            self._stop.set()
            server.close()
            self._hub.stop()
            broadcaster.join(timeout=2.0)

    def _bind_servers(
        self,
    ) -> tuple[socket.socket | None, list[tuple[str, str, int, OSError]]]:
        failures: list[tuple[str, str, int, OSError]] = []
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(("0.0.0.0", self._direct_port))
            server.listen(1)
            server.settimeout(1.0)
        except OSError as exc:
            failures.append(("direct", "0.0.0.0", self._direct_port, exc))
            server.close()
            server = None

        try:
            self._hub.bind()
        except OSError as exc:
            failures.append(("relay", self._hub.host, self._hub.port, exc))

        if failures:
            if server is not None:
                server.close()
            self._hub.stop()
            return None, failures
        return server, failures

    def _handle_direct_client(self, client: socket.socket) -> None:
        parser = _DirectFrameParser()
        client.settimeout(1.0)
        last_heartbeat = time.monotonic()
        try:
            while not self._stop.is_set():
                if time.monotonic() - last_heartbeat > HEARTBEAT_TIMEOUT_S:
                    logger.warning("Pico 4 heartbeat timeout")
                    break
                try:
                    data = client.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                parser.feed(data)
                while True:
                    frame = parser.try_parse()
                    if frame is None:
                        break
                    if frame["cmd"] in (
                        _CMD_HEARTBEAT,
                        _CMD_CONNECT,
                        _CMD_BATTERY,
                        _CMD_SENSOR,
                    ):
                        last_heartbeat = time.monotonic()
                    if frame["cmd"] == _CMD_DEVICE_STATE_JSON:
                        self._hub.publish(frame["payload"])
        finally:
            client.close()

    def _broadcast_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            while not self._stop.is_set():
                for ip in _get_local_ips():
                    parts = ip.split(".")
                    if len(parts) != 4:
                        continue
                    packet = _build_broadcast_packet(ip)
                    try:
                        sock.sendto(
                            packet,
                            (".".join(parts[:3]) + ".255", self._broadcast_port),
                        )
                    except OSError:
                        pass
                self._stop.wait(BROADCAST_INTERVAL_S)
        finally:
            sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep Pico 4 connected and relay tracking data locally."
    )
    parser.add_argument("--direct-port", type=int, default=DEFAULT_DIRECT_PORT)
    parser.add_argument("--relay-host", default=DEFAULT_RELAY_HOST)
    parser.add_argument("--relay-port", type=int, default=DEFAULT_RELAY_PORT)
    parser.add_argument("--broadcast-port", type=int, default=DEFAULT_BROADCAST_PORT)
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    daemon = Pico4Daemon(
        direct_port=args.direct_port,
        relay_host=args.relay_host,
        relay_port=args.relay_port,
        broadcast_port=args.broadcast_port,
        device_id=args.device_id,
    )
    try:
        daemon.run()
    except PortBindingError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
