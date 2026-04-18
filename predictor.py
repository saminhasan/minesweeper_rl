from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from game_engine import Minesweeper

_probability_sources = ["bayesian", "cnn"]
_PROBABILITY_SOURCE = _probability_sources[0]  # ← change index: 0=bayesian, 1=cnn

_level_cnn_paths: dict[str, Path] = {
    "easy":         Path("models/CNN/easy/model.keras"),
    "intermediate": Path("models/CNN/intermediate/model.keras"),
    "hard":         Path("models/CNN/hard/model.keras"),
}


class Predictor:
    def __init__(self, level: str, source: str | None = None) -> None:
        self._source = source if source is not None else _PROBABILITY_SOURCE
        self._cnn: tf.keras.Model | None = None
        if self._source == "cnn":
            path = _level_cnn_paths[level]
            if not path.exists():
                raise FileNotFoundError(f"CNN model not found: {path}")
            self._cnn = tf.keras.models.load_model(path, compile=False)
            print(f"Loaded CNN model: {path}")



    def _build_input_channels(self, x: np.ndarray, covered: np.ndarray) -> np.ndarray:
        x_f = x.astype(np.float32, copy=False)
        cv = covered.astype(np.float32, copy=False)
        board_scaled = np.where(x_f < 0.0, -1.0, x_f / 8.0).astype(np.float32, copy=False)
        revealed_clue = np.where(cv > 0.5, 0.0, np.clip(x_f, 0.0, 8.0) / 8.0).astype(np.float32, copy=False)
        uncovered = (cv <= 0.5).astype(np.float32, copy=False)
        padded = np.pad(uncovered, ((1, 1), (1, 1)), mode="constant")
        near_uncovered = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        )
        frontier_mask = ((cv > 0.5) & (near_uncovered > 0.0)).astype(np.float32, copy=False)
        return np.stack([board_scaled, cv, revealed_clue, frontier_mask], axis=-1)

    def predict(self, board: Minesweeper) -> tuple[np.ndarray, np.ndarray]:
        covered = np.asarray(board.state == board.states.COVERED, dtype=np.float32)
        if self._source == "cnn":
            assert self._cnn is not None
            x = np.asarray(board.get_input(), dtype=np.float32)
            base = self._build_input_channels(x, covered)[None, ...]
            pred = self._cnn(base, training=False).numpy()[0]
            mine_prob = pred[..., 0].astype(np.float32, copy=False)
            safe_prob = pred[..., 1].astype(np.float32, copy=False)
            return mine_prob, safe_prob
        mine_prob = np.asarray(board.get_output(), dtype=np.float32)
        return mine_prob, (1.0 - mine_prob).astype(np.float32, copy=False)

    def build_state(self, board: Minesweeper) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(board.get_input(), dtype=np.float32)
        covered = np.asarray(board.state == board.states.COVERED, dtype=np.float32)
        base = self._build_input_channels(x, covered)
        mine_prob, safe_prob = self.predict(board)
        state = np.concatenate(
            [base, mine_prob[..., None], safe_prob[..., None]], axis=-1
        ).astype(np.float32)
        action_mask = (covered > 0.5).reshape(-1).astype(np.float32)
        return state, action_mask
