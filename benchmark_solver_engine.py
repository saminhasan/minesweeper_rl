from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from game_engine import Minesweeper


DIFFICULTIES: Tuple[str, ...] = ("easy", "intermediate", "hard")
GAMES_PER_LEVEL: int = 10
SAFE_REVEAL: bool = True
VERBOSE: bool = True
RUN_WARMUP: bool = True
WARMUP_SEED_OFFSET: int = 50_000_000

FIXED_SEEDS: Dict[str, Tuple[int, ...]] = {
    "easy": (
        1326614145,
        1987572157,
        1145179090,
        1524455654,
        785653375,
        1226168024,
        1257968761,
        1571732188,
        355810735,
        1664728517,
    ),
    "intermediate": (
        1046116841,
        89600439,
        733018785,
        1494025571,
        1250690770,
        630379384,
        255694498,
        858501564,
        730789637,
        1597102226,
    ),
    "hard": (
        1827640484,
        1614048804,
        820003138,
        77118010,
        873660890,
        1866207564,
        451193426,
        1976623494,
        451453888,
        257224422,
    ),
}


@dataclass
class GameResult:
    difficulty: str
    won: bool
    moves: int
    solve_calls: int
    seconds: float


def run_single_game(difficulty: str, seed: int, safe_reveal: bool) -> GameResult:
    board = Minesweeper(difficulty, seed=seed)
    moves = 0
    solve_calls = 0
    start = time.perf_counter()

    while not (board.game_over or board.game_won):
        if VERBOSE:
            print('.', end='', flush=True)
        if safe_reveal:
            board.random_safe_reveal()
        else:
            board.random_reveal()

        if board.game_over or board.game_won:
            break

        board.solve_minefield()
        solve_calls += 1
        moves += 1

    elapsed = time.perf_counter() - start
    if VERBOSE:
        print(
            f"\nGame finished: difficulty={difficulty}, won={board.game_won}, "
            f"moves={moves}, solve_calls={solve_calls}, seconds={elapsed:.6f}"
        )
    return GameResult(
        difficulty=difficulty,
        won=board.game_won,
        moves=moves,
        solve_calls=solve_calls,
        seconds=elapsed,
    )


def get_benchmark_seeds(difficulty: str, games: int) -> List[int]:
    seeds = list(FIXED_SEEDS[difficulty])
    if games > len(seeds):
        raise ValueError(
            f"Requested {games} games for {difficulty}, but only {len(seeds)} fixed seeds are configured."
        )
    return seeds[:games]


def warmup_seed_for(difficulty: str) -> int:
    return FIXED_SEEDS[difficulty][0] + WARMUP_SEED_OFFSET


def summarize(results: Iterable[GameResult]) -> Dict[str, float]:
    rows = list(results)
    wins = sum(1 for r in rows if r.won)
    times = [r.seconds for r in rows]
    moves = [r.moves for r in rows]
    solve_calls = [r.solve_calls for r in rows]

    return {
        "games": float(len(rows)),
        "wins": float(wins),
        "win_rate": wins / float(len(rows)) if rows else 0.0,
        "avg_seconds": statistics.mean(times) if times else 0.0,
        "p50_seconds": statistics.median(times) if times else 0.0,
        "avg_moves": statistics.mean(moves) if moves else 0.0,
        "avg_solve_calls": statistics.mean(solve_calls) if solve_calls else 0.0,
    }


def run_difficulty_benchmark(difficulty: str, games: int, safe_reveal: bool) -> Dict[str, float]:
    seeds = get_benchmark_seeds(difficulty, games)

    # Warm-up uses a dedicated seed so measured runs are exactly FIXED_SEEDS.
    if RUN_WARMUP:
        _ = run_single_game(difficulty, warmup_seed_for(difficulty), safe_reveal)

    all_results = [run_single_game(difficulty, s, safe_reveal) for s in seeds]
    return summarize(all_results)


def run_all_difficulties_benchmark(
    difficulties: Tuple[str, ...] = DIFFICULTIES,
    games_per_level: int = GAMES_PER_LEVEL,
    safe_reveal: bool = SAFE_REVEAL,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for difficulty in difficulties:
        out[difficulty] = run_difficulty_benchmark(
            difficulty=difficulty,
            games=games_per_level,
            safe_reveal=safe_reveal,
        )
    return out


def print_benchmark_report(report: Dict[str, Dict[str, float]], safe_reveal: bool) -> None:
    print("Benchmark summary")
    print(f"safe_reveal: {safe_reveal}")
    print("seed_mode: fixed")
    print(f"warmup: {RUN_WARMUP}")
    print(f"games_per_level: {GAMES_PER_LEVEL}")
    print()

    for difficulty in DIFFICULTIES:
        summary = report[difficulty]
        print(f"difficulty: {difficulty}")
        print(f"games: {int(summary['games'])}")
        print(f"wins: {int(summary['wins'])} ({summary['win_rate'] * 100.0:.2f}%)")
        print(f"avg_seconds: {summary['avg_seconds']:.6f}")
        print(f"p50_seconds: {summary['p50_seconds']:.6f}")
        print(f"avg_moves: {summary['avg_moves']:.2f}")
        print(f"avg_solve_calls: {summary['avg_solve_calls']:.2f}")
        print()


def run_default_benchmark() -> Dict[str, Dict[str, float]]:
    report = run_all_difficulties_benchmark(
        difficulties=DIFFICULTIES,
        games_per_level=GAMES_PER_LEVEL,
        safe_reveal=SAFE_REVEAL,
    )
    print_benchmark_report(report, SAFE_REVEAL)
    return report


if __name__ == "__main__":
    run_default_benchmark()
