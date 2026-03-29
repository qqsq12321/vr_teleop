"""Move Kinova Gen3 to official Home position and exit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_KORTEX_EXAMPLES_DIR = (
    Path(__file__).resolve().parent
    / "Kinova-kortex2_Gen3_G3L"
    / "api_python"
    / "examples"
)
if str(_KORTEX_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_KORTEX_EXAMPLES_DIR))

import argparse
import threading

import utilities as kortex_utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description="Move Kinova Gen3 to Home.")
    parser.add_argument("--kinova-ip", default="192.168.1.10")
    parser.add_argument("--kinova-username", default="admin")
    parser.add_argument("--kinova-password", default="admin")
    args = parser.parse_args()

    kinova_args = SimpleNamespace(
        ip=args.kinova_ip,
        username=args.kinova_username,
        password=args.kinova_password,
    )

    with kortex_utilities.DeviceConnection.createTcpConnection(kinova_args) as router:
        base = BaseClient(router)

        servo_mode = Base_pb2.ServoingModeInformation()
        servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(servo_mode)

        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = base.ReadAllActions(action_type)

        action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle
                break

        if action_handle is None:
            print("Home action not found!")
            return

        finished = threading.Event()

        def on_notification(notification):
            if notification.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                finished.set()

        handle = base.OnNotificationActionTopic(on_notification, Base_pb2.NotificationOptions())
        try:
            print("Moving to Home...")
            base.ExecuteActionFromReference(action_handle)
            if finished.wait(30.0):
                print("Home reached.")
            else:
                print("Timeout!")
        finally:
            base.Unsubscribe(handle)


if __name__ == "__main__":
    main()
