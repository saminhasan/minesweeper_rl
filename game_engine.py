import time
import numpy as np
from statistics import mean
from collections import deque
from dataclasses import dataclass
from solver import MineCount, Rule, solve
from joblib import Parallel, delayed

from typing import Any, Dict, List, Set, Tuple, cast


game_mode: Dict[str, Dict[str, int]] = {
    # "test": {"rows": 5, "columns": 5, "mines": 5}, # mine density = (mines*100/(rows*columns))%  = 20.0
    "easy": {"rows": 10, "columns": 10, "mines": 10}, # mine density = (mines*100/(rows*columns))%  = 10.0
    "intermediate": {"rows": 16, "columns": 16, "mines": 40}, # mine density = (mines*100/(rows*columns))%  = 15.625
    "hard": {"rows": 16, "columns": 30, "mines": 99}, # mine density = (mines*100/(rows*columns))%  = 20.625
    # "xtreme": {"rows": 12, "columns": 12, "mines": 36}, # mine density = (mines*100/(rows*columns))%  = 25.0
}


@dataclass(frozen=True)
class State:
    UNCOVERED: int = 0
    COVERED: int = -1


class Minesweeper:
    """Efficient Minesweeper board with Bayesian-solver adapters."""

    _NEIGHBOR_OFFSETS: Tuple[Tuple[int, int], ...] = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    def __init__(self, difficulty: str, seed: int | None = None) -> None:
        if difficulty not in game_mode:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        cfg = game_mode[difficulty]
        self.game_over: bool = False
        self.game_won: bool = False
        self.states = State()

        self.n_rows: int = cfg["rows"]
        self.n_cols: int = cfg["columns"]
        self.shape: Tuple[int, int] = (self.n_rows, self.n_cols)
        self.total_cells: int = self.n_rows * self.n_cols
        self.n_mines: int = cfg["mines"]

        self._rng = np.random.default_rng(seed)
        self._mines_placed = False
        self.covered_count: int = self.total_cells

        # Store fields in separate contiguous arrays for faster hot-path operations.
        self.mine_count: np.ndarray = np.zeros(self.shape, dtype=np.int8)
        self.state: np.ndarray = np.full(self.shape, self.states.COVERED, dtype=np.int8)

        self.mines: Set[Tuple[int, int]] = set()
        self.tag_to_index: Dict[int, Tuple[int, int]] = {}

        self.mine_count.fill(0)
        self._neighbors: List[List[Tuple[int, int]]] = [
            self._build_neighbors(i, j)
            for i in range(self.n_rows)
            for j in range(self.n_cols)
        ]

    def _build_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        r0 = i - 1 if i else 0
        r1 = i + 1 if i + 1 < self.n_rows else self.n_rows - 1
        c0 = j - 1 if j else 0
        c1 = j + 1 if j + 1 < self.n_cols else self.n_cols - 1

        for ni in range(r0, r1 + 1):
            for nj in range(c0, c1 + 1):
                if ni != i or nj != j:
                    out.append((ni, nj))
        return out

    def _flat(self, i: int, j: int) -> int:
        return i * self.n_cols + j

    def place_mines(self, safe_i: int, safe_j: int) -> None:
        total_cells = self.n_rows * self.n_cols
        safe_flat = safe_i * self.n_cols + safe_j

        # Sample from [0, total_cells-2] then shift indices >= safe_flat by +1
        # so safe cell is excluded without constructing a large filtered array.
        flat_idx = self._rng.choice(total_cells - 1, size=self.n_mines, replace=False)
        flat_idx = flat_idx + (flat_idx >= safe_flat)

        mine_rows = flat_idx // self.n_cols
        mine_cols = flat_idx % self.n_cols

        self.mine_count.fill(0)
        self.mine_count[mine_rows, mine_cols] = -1

        self.mines = set(zip(mine_rows.tolist(), mine_cols.tolist()))

        mine_mask = self.mine_count == -1
        padded = np.pad(mine_mask.astype(np.int8), 1, mode="constant")
        neighbors = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ).astype(np.int8)

        self.mine_count[:, :] = neighbors
        self.mine_count[mine_mask] = -1
        self._mines_placed = True

    def reveal(self, i: int, j: int) -> None:
        if not self._in_bounds(i, j):
            return
        if self.game_over or self.game_won:
            return
        if self.state[i, j] != self.states.COVERED:
            return

        if not self._mines_placed:
            self.place_mines(i, j)

        if self.mine_count[i, j] == -1:
            self.reveal_all_mines()
            self.game_over = True
            self.game_won = False
            return

        q: deque[Tuple[int, int]] = deque([(i, j)])
        while q:
            x, y = q.popleft()
            if self.state[x, y] != self.states.COVERED:
                continue

            self.state[x, y] = self.states.UNCOVERED
            self.covered_count -= 1

            if self.mine_count[x, y] != 0:
                continue

            for nx, ny in self.get_neighbors(x, y):
                if self.state[nx, ny] == self.states.COVERED:
                    q.append((nx, ny))

        if self.check_win():
            self.game_won = True
    
    def _in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.n_rows and 0 <= j < self.n_cols


    def random_reveal(self) -> None:
        if self.game_over or self.game_won:
            return
        covered_flat = np.flatnonzero(self.state.ravel() == self.states.COVERED)
        if covered_flat.size == 0:
            return
        flat = int(covered_flat[self._rng.integers(covered_flat.size)])
        self.reveal(flat // self.n_cols, flat % self.n_cols)

    def random_safe_reveal(self) -> None:
        if self.game_over or self.game_won:
            return
        if not self._mines_placed:
            # Before mines are placed, any covered cell can be selected safely.
            self.random_reveal()
            return

        safe_flat = np.flatnonzero(((self.state == self.states.COVERED) & (self.mine_count != -1)).ravel())
        if safe_flat.size == 0:
            return
        flat = int(safe_flat[self._rng.integers(safe_flat.size)])
        self.reveal(flat // self.n_cols, flat % self.n_cols)

    def reveal_all_mines(self) -> None:
        mine_mask = self.mine_count == -1
        newly_uncovered = np.count_nonzero((self.state == self.states.COVERED) & mine_mask)
        self.state[mine_mask] = self.states.UNCOVERED
        self.covered_count -= int(newly_uncovered)

    def check_win(self) -> bool:
        return self.covered_count == self.n_mines

    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        return self._neighbors[self._flat(i, j)]
    def get_frontier_cells(self) -> np.ndarray:
        uncovered = self.state == self.states.UNCOVERED
        padded = np.pad(uncovered.astype(np.int8), 1, mode="constant")
        near_uncovered = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        )
        frontier = (self.state == self.states.COVERED) & (near_uncovered > 0)
        return frontier.astype(np.int8)

    def create_rules_from_minefield(self) -> List[Rule]:
        rules: List[Rule] = []
        self.tag_to_index = {}

        uncovered_cells = np.argwhere(self.state == self.states.UNCOVERED)
        for i_raw, j_raw in uncovered_cells:
            i, j = int(i_raw), int(j_raw)
            mine_count = int(self.mine_count[i, j])
            if mine_count < 0:
                continue

            covered_neighbors: List[int] = []
            for x, y in self.get_neighbors(i, j):
                if self.state[x, y] != self.states.COVERED:
                    continue
                covered_neighbors.append(self._flat(x, y))

            if covered_neighbors:
                rules.append(Rule(mine_count, covered_neighbors))

        return rules

    def decode_solution(self, solution: Dict[Any, float]) -> Tuple[Dict[Tuple[int, int], float], np.ndarray]:
        decoded: Dict[Tuple[int, int], float] = {}

        default_probability = float(solution.get(None, 0.0))
        probability = np.full(self.shape, default_probability, dtype=np.float32)

        for tag, p in solution.items():
            if tag is None:
                continue
            if isinstance(tag, (int, np.integer)) and 0 <= int(tag) < self.total_cells:
                flat = int(tag)
                idx = (flat // self.n_cols, flat % self.n_cols)
                decoded[idx] = float(p)
                probability[idx] = float(p)

        return decoded, probability

    def solve_minefield(self) -> Tuple[Dict[Tuple[int, int], float], np.ndarray]:
        rules = self.create_rules_from_minefield()
        total_cells = self.n_rows * self.n_cols
        results = solve(set(rules), MineCount(total_cells=total_cells, total_mines=self.n_mines))
        return self.decode_solution(results)

    def get_input(self) -> np.ndarray:
        """Return v2 datagen input: covered=-1, uncovered=cell value (0..8)."""
        return np.where(self.state == self.states.COVERED, self.states.COVERED, self.mine_count).astype(np.int8)

    def get_output(self) -> np.ndarray:
        _, probability = self.solve_minefield()
        probability[self.state == self.states.UNCOVERED] = 0.0
        return probability

    @staticmethod
    def display_minefield(board: "Minesweeper") -> None:
        rows, cols = board.shape
        for i in range(rows):
            for j in range(cols):
                if board.state[i, j] == State.COVERED:
                    print("#", end=" ")
                elif board.state[i, j] == State.UNCOVERED:
                    if board.mine_count[i, j] == -1:
                        print("*", end=" ")
                    else:
                        print(int(board.mine_count[i, j]), end=" ")
            print()


def play_game(game_id: int, difficulty: str = "easy", safe_reveal: bool = False) -> Tuple[int, float, int, bool]:
    """Simulate one game and return (moves, seconds, game_id, won)."""
    start = time.perf_counter()
    board = Minesweeper(difficulty)
    moves = 0

    while not (board.game_over or board.game_won):
        if safe_reveal:
            board.random_safe_reveal()
        else:
            board.random_reveal()
        if board.game_over or board.game_won:
            break
        board.solve_minefield()
        moves += 1

    elapsed = time.perf_counter() - start
    return moves, elapsed, game_id, board.game_won


def benchmark_games(
    total_games: int = 1000,
    n_jobs: int = 12,
    difficulty: str = "easy",
    safe_reveal: bool = False,
) -> List[Tuple[int, float, int, bool]]:
    """Run benchmark games in parallel when joblib is available, else serially."""
    if Parallel is not None and delayed is not None:
        results = Parallel(n_jobs=n_jobs)(
            delayed(play_game)(game_id, difficulty=difficulty, safe_reveal=safe_reveal)
            for game_id in range(total_games)
        )
        return cast(List[Tuple[int, float, int, bool]], results)

    return [
        play_game(game_id, difficulty=difficulty, safe_reveal=safe_reveal)
        for game_id in range(total_games)
    ]


def _quick_self_test() -> None:
    board = Minesweeper("easy", seed=123)
    board.reveal(0, 0)

    if board.mine_count[0, 0] == -1:
        raise AssertionError("First reveal should never be a mine")

    inp = board.get_input()
    if inp.shape != board.shape:
        raise AssertionError("Input shape mismatch")

    out = board.get_output()
    if out.shape != board.shape:
        raise AssertionError("Output shape mismatch")

    uncovered_mask = board.state == board.states.UNCOVERED
    if np.any(out[uncovered_mask] != 0.0):
        raise AssertionError("Output probabilities must be zero on uncovered cells")

    print("Quick self-test passed.")


if __name__ == "__main__":
    _quick_self_test()

    TOTAL_GAMES = 1000
    N_JOBS = 12
    DIFFICULTY = "hard"
    SAFE_REVEAL = False

    mode = "safe-reveal" if SAFE_REVEAL else "random-reveal"
    print(
        f"Simulating {TOTAL_GAMES} games on {DIFFICULTY} using {N_JOBS} workers "
        f"({mode})...\n"
    )

    results = benchmark_games(
        total_games=TOTAL_GAMES,
        n_jobs=N_JOBS,
        difficulty=DIFFICULTY,
        safe_reveal=SAFE_REVEAL,
    )

    moves_list = [r[0] for r in results]
    time_list = [r[1] for r in results]
    wins = sum(1 for r in results if r[3])

    print("\nSummary Statistics:")
    print(f"Total Games: {TOTAL_GAMES}")
    print(f"Wins: {wins} ({100.0 * wins / TOTAL_GAMES:.2f}%)")
    print(f"Moves: Min={min(moves_list)}, Max={max(moves_list)}, Avg={mean(moves_list):.2f}")
    print(
        f"Time: Min={min(time_list):.3f}s, Max={max(time_list):.3f}s, "
        f"Avg={mean(time_list):.3f}s"
    )
