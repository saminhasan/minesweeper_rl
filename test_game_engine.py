from __future__ import annotations

import numpy as np

from game_engine import Minesweeper, game_mode


def test_first_click_is_safe() -> None:
    b = Minesweeper("hard", seed=123)
    b.reveal(0, 0)
    assert b.mine_count[0, 0] != -1


def test_shapes_and_dtypes() -> None:
    for diff, cfg in game_mode.items():
        b = Minesweeper(diff, seed=123)
        x = b.get_input()
        y = b.get_output()
        assert x.shape == (cfg["rows"], cfg["columns"])
        assert y.shape == (cfg["rows"], cfg["columns"])
        assert x.dtype == np.int8
        assert y.dtype == np.float32


def test_output_zero_on_uncovered() -> None:
    b = Minesweeper("easy", seed=123)
    b.reveal(0, 0)
    y = b.get_output()
    uncovered = b.state == b.states.UNCOVERED
    assert np.all(y[uncovered] == 0.0)


def test_rule_generation_runs() -> None:
    b = Minesweeper("intermediate", seed=123)
    b.reveal(0, 0)
    rules = b.create_rules_from_minefield()
    assert isinstance(rules, list)


def run_all() -> None:
    tests = [
        test_first_click_is_safe,
        test_shapes_and_dtypes,
        test_output_zero_on_uncovered,
        test_rule_generation_runs,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("All game engine tests passed.")


if __name__ == "__main__":
    run_all()
