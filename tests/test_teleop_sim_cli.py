import sys

from example import teleop_sim


def test_no_arguments_preserve_quest3_relay_defaults(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(sys, "argv", ["teleop_sim.py"])
    monkeypatch.setattr(
        teleop_sim,
        "_run_bimanual",
        lambda args: captured.update(vars(args)),
    )

    teleop_sim.main()

    assert captured["input_source"] == "quest3"
    assert captured["pico4_mode"] == "relay"
