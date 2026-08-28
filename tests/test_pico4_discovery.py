import json
import subprocess
from pathlib import Path

from utils.adapters.input.pico4 import _get_local_ips


def _active_physical_ipv4s() -> set[str]:
    output = subprocess.check_output(
        ["ip", "-j", "-4", "address", "show", "up"],
        text=True,
    )
    interfaces = json.loads(output)
    result: set[str] = set()
    for interface in interfaces:
        name = interface["ifname"]
        if name == "lo" or not Path(f"/sys/class/net/{name}/device").exists():
            continue
        for address in interface.get("addr_info", []):
            if address.get("family") == "inet" and address.get("scope") == "global":
                result.add(address["local"])
    return result


def test_get_local_ips_includes_every_active_physical_interface() -> None:
    expected = _active_physical_ipv4s()

    assert expected
    assert expected <= set(_get_local_ips())
