"""Factory for configured input devices."""

from __future__ import annotations

import argparse


def create_input_device(args: argparse.Namespace):
    if args.input_source == "avp":
        from utils.adapters.input.avp import AVPInput

        return AVPInput(ip=args.avp_ip)
    if args.input_source == "pico4":
        from utils.adapters.input.pico4 import Pico4

        return Pico4(
            mode=args.pico4_mode,
            relay_host=args.pico4_relay_host,
            relay_port=args.pico4_relay_port,
            port=args.pico4_port,
            broadcast_port=args.pico4_broadcast_port,
        )
    from utils.adapters.input.quest3 import make_socket

    return make_socket(args.port)
