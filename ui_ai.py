
import os
import ctypes
from collections import deque
from typing import List, Set, Tuple, cast

import numpy as np
import pygame as pg
import pygame.gfxdraw as gfxdraw
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon
from shapely.ops import unary_union

from game_engine import Minesweeper


FONT_PATH: str = "./assets/fonts"
IMAGE_PATH: str = "./assets/images"
CELL_SIZE: int = 64
LINE_WIDTH: int = 0
BORDER_SIZE: int = 0
COVERED: int = -1

cell_color: Tuple[int, int, int] = (30, 30, 30)
line_color: Tuple[int, int, int] = (125, 125, 125)
background_color: Tuple[int, int, int] = (5, 5, 5)
text_color: Tuple[int, int, int] = (220, 220, 220)
font_size: int = 18

levels = {0: "test", 1: "easy", 2: "intermediate", 3: "hard", 4: "xtreme"}


# -------------------------
# Inline render utilities
# -------------------------
def blur_bg(screen: pg.Surface, sigma: float = 0.5) -> None:
    """Apply a Gaussian filter in-place to each RGB channel."""
    pixels = pg.surfarray.pixels3d(screen)
    for channel in range(3):
        gaussian_filter(pixels[:, :, channel], sigma=sigma, mode="nearest", output=pixels[:, :, channel])


def get_custom_rgb(value: float) -> Tuple[int, int, int]:
    if not 0 <= value <= 1:
        raise ValueError("Value must be between 0 and 1")

    if value <= 0.5:
        return int(255 * (2 * value)), 255, 0
    return 255, int(255 * (2 * (1 - value))), 0


def find_clusters(board: np.ndarray, flag: int) -> List[List[Tuple[int, int]]]:
    rows, cols = board.shape
    visited = np.zeros_like(board, dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def flood_fill_iterative(start_r: int, start_c: int) -> List[Tuple[int, int]]:
        cluster: List[Tuple[int, int]] = []
        queue: deque[Tuple[int, int]] = deque([(start_r, start_c)])
        visited[start_r, start_c] = True

        while queue:
            r, c = queue.popleft()
            cluster.append((r, c))

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and board[nr, nc] == flag:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

        return cluster

    return [
        flood_fill_iterative(row, col)
        for row in range(rows)
        for col in range(cols)
        if board[row, col] == flag and not visited[row, col]
    ]


def get_rects_from_cluster(cluster: List[Tuple[int, int]], cell_size: int, border_size: int, line_width: int) -> List[pg.Rect]:
    return [
        pg.Rect(
            border_size + c * (cell_size + border_size * 2 + line_width) + 1,
            border_size + r * (cell_size + border_size * 2 + line_width) + 1,
            cell_size,
            cell_size,
        )
        for r, c in cluster
    ]


def rects_to_polygon(rects: List[pg.Rect]) -> Polygon:
    return cast(
        Polygon,
        unary_union(
        [
            Polygon(
                [
                    rect.topleft,
                    rect.topright,
                    rect.bottomright,
                    rect.bottomleft,
                ]
            )
            for rect in rects
        ]
        ),
    )


def draw_polygon_with_holes(
    surface: pg.Surface,
    polygon: Polygon,
    fill_color: Tuple[int, int, int],
    hole_color: Tuple[int, int, int],
    cell_size: int,
) -> None:
    resolution = max(16, int(polygon.length / 10))
    dilate_distance = cell_size // 9

    smoothed_exterior = (
        polygon.buffer(dilate_distance, cap_style="round", join_style="round", resolution=resolution)
        .buffer(-dilate_distance * 3.0, cap_style="round", join_style="round", resolution=resolution)
        .buffer(dilate_distance, cap_style="round", join_style="round", resolution=resolution)
    )

    gfxdraw.aapolygon(surface, list(map(tuple, smoothed_exterior.exterior.coords)), fill_color)
    gfxdraw.filled_polygon(surface, list(map(tuple, smoothed_exterior.exterior.coords)), fill_color)

    for interior in polygon.interiors:
        smoothed_hole = (
            Polygon(interior.coords)
            .buffer(dilate_distance * 1.5, cap_style="round", join_style="round", resolution=resolution)
            .buffer(-dilate_distance * 2, cap_style="round", join_style="round", resolution=resolution)
            .buffer(dilate_distance * 1.5, cap_style="round", join_style="round", resolution=resolution)
        )
        gfxdraw.aapolygon(surface, list(map(tuple, smoothed_hole.exterior.coords)), hole_color)
        gfxdraw.filled_polygon(surface, list(map(tuple, smoothed_hole.exterior.coords)), hole_color)


# -------------------------
# UI logic
# -------------------------
def predict(board: Minesweeper) -> np.ndarray:
    _, probability = board.solve_minefield()
    return probability


class GUI:
    def __init__(self, level: int) -> None:
        self.level: str = levels[level]
        self._initialized: bool = False
        self.init_game()

    def init_game(self) -> None:
        if self._initialized:
            pg.quit()

        self.running: bool = True
        self.fps: int = 240
        self.help: bool = True
        self.flagged: Set[Tuple[int, int]] = set()

        self.board = Minesweeper(self.level)
        rows, cols = self.board.shape
        self.probability = predict(self.board)

        global CELL_SIZE
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)

        if rows >= cols:
            CELL_SIZE = (screen_height - 100 - BORDER_SIZE * 2 - (rows - 1) * (LINE_WIDTH + BORDER_SIZE * 2)) // rows
        else:
            CELL_SIZE = (screen_width - 100 - BORDER_SIZE * 2 - (cols - 1) * (LINE_WIDTH + BORDER_SIZE * 2)) // cols

        self.width: int = BORDER_SIZE * 2 + cols * CELL_SIZE + (cols - 1) * (LINE_WIDTH + BORDER_SIZE * 2) + 1
        self.height: int = BORDER_SIZE * 2 + rows * CELL_SIZE + (rows - 1) * (LINE_WIDTH + BORDER_SIZE * 2) + 1

        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{(screen_width - self.width) // 2},{(screen_height - self.height - 20) // 2}"

        pg.init()
        pg.font.init()
        self.clock: pg.time.Clock = pg.time.Clock()
        self.screen: pg.Surface = pg.display.set_mode((self.width, self.height))

        # self.font: pg.font.Font = pg.font.Font(f"{FONT_PATH}/orbitron.ttf", font_size)
        self.font : pg.font.Font = pg.font.SysFont("Consolas", font_size)
        self.mine_image: pg.Surface = pg.image.load(f"{IMAGE_PATH}/mine.png").convert_alpha()
        self.flag_image: pg.Surface = pg.image.load(f"{IMAGE_PATH}/flag.png").convert_alpha()
        self.scaled_mine_image = pg.transform.scale(self.mine_image, (CELL_SIZE // 2, CELL_SIZE // 2))
        self.scaled_flag_image = pg.transform.scale(self.flag_image, (CELL_SIZE // 2, CELL_SIZE // 2))

        pg.display.set_caption("Minesweeper")
        pg.display.set_icon(self.mine_image)
        self._initialized = True

    def quit(self) -> None:
        self.running = False

    def reset_game(self) -> None:
        self.init_game()

    def handle_events(self) -> None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            elif event.type == pg.KEYDOWN:
                self.handle_key_event(event.key)
            elif event.type == pg.MOUSEBUTTONDOWN:
                self.handle_mouse_event(event)

    def handle_key_event(self, key: int) -> None:
        if key == pg.K_ESCAPE:
            self.quit()
        elif key == pg.K_r:
            self.reset_game()
        elif key == pg.K_h:
            self.help = not self.help
            if self.help:
                self.probability = predict(self.board)
        elif key == pg.K_z:
            self.board.random_safe_reveal()
            self.probability = predict(self.board)
        elif key in [pg.K_0, pg.K_1, pg.K_2, pg.K_3, pg.K_4]:
            self.level = levels[key - pg.K_0]
            self.reset_game()

    def handle_mouse_event(self, event: pg.event.Event) -> None:
        if self.board.game_over or self.board.game_won:
            return

        mouse_x, mouse_y = event.pos
        col = (mouse_x - BORDER_SIZE) // (CELL_SIZE + LINE_WIDTH + BORDER_SIZE * 2)
        row = (mouse_y - BORDER_SIZE) // (CELL_SIZE + LINE_WIDTH + BORDER_SIZE * 2)

        if not (0 <= row < self.board.n_rows and 0 <= col < self.board.n_cols):
            return

        if event.button == pg.BUTTON_LEFT:
            if (row, col) not in self.flagged:
                self.board.reveal(row, col)
                if not (self.board.game_over or self.board.game_won):
                    self.probability = predict(self.board)
                    self.flagged = {
                        flag for flag in self.flagged if self.board.minefield[flag]["state"] != self.board.states.UNCOVERED
                    }

        elif event.button == pg.BUTTON_RIGHT:
            if self.board.minefield[row, col]["state"] != self.board.states.UNCOVERED:
                if (row, col) in self.flagged:
                    self.flagged.remove((row, col))
                else:
                    self.flagged.add((row, col))

    def draw(self) -> None:
        self.screen.fill(background_color)

        if self.board.game_over:
            self.draw_mines()
            overlay = pg.Surface(self.screen.get_size(), pg.SRCALPHA)
            overlay.fill((128, 0, 0, 64))
            self.screen.blit(overlay, (0, 0))
            main_text = "Game Over, Press 'R' to Restart"
        elif self.board.game_won:
            self.draw_clusters()
            self.draw_flags()
            overlay = pg.Surface(self.screen.get_size(), pg.SRCALPHA)
            overlay.fill((0, 128, 0, 64))
            self.screen.blit(overlay, (0, 0))
            main_text = "Game Won, Press 'R' to Restart"
        else:
            self.draw_clusters()
            self.draw_cells_bayes()
            self.draw_markers()
            self.draw_lines()
            blur_bg(self.screen, sigma=0.32)
            return

        self.draw_lines()
        blur_bg(self.screen, sigma=2)

        text_lines = [
            main_text,
            f"Current Level: {self.level.capitalize()}",
            "Press 1 - Easy",
            "Press 2 - Intermediate",
            "Press 3 - Hard",
            "Press 4 - Extreme",
            "Press H in game to Toggle Help",
        ]

        start_y = self.height // 4
        end_y = (3 * self.height) // 4
        vertical_spacing = (end_y - start_y) // (len(text_lines) - 1)

        for i, text in enumerate(text_lines):
            text_surface = self.font.render(text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.width // 2, start_y + i * vertical_spacing))
            self.screen.blit(text_surface, text_rect)

    def draw_clusters(self) -> None:
        for cluster in find_clusters(self.board.minefield["state"], COVERED):
            draw_polygon_with_holes(
                self.screen,
                rects_to_polygon(get_rects_from_cluster(cluster, CELL_SIZE, BORDER_SIZE, LINE_WIDTH)),
                cell_color,
                background_color,
                CELL_SIZE,
            )
        blur_bg(self.screen, sigma=0.8)

    def draw_mines(self) -> None:
        for row, col in self.board.mines:
            self.screen.blit(
                self.scaled_mine_image,
                (
                    BORDER_SIZE + col * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1 + self.scaled_mine_image.get_width() // 2,
                    BORDER_SIZE + row * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1 + self.scaled_mine_image.get_height() // 2,
                ),
            )

    def draw_flags(self) -> None:
        for row, col in self.board.mines:
            self.screen.blit(
                self.scaled_flag_image,
                (
                    BORDER_SIZE + col * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1 + self.scaled_flag_image.get_width() // 2,
                    BORDER_SIZE + row * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1 + self.scaled_flag_image.get_height() // 2,
                ),
            )

    def draw_markers(self) -> None:
        for row, col in self.flagged:
            self.screen.blit(
                self.scaled_flag_image,
                (
                    BORDER_SIZE + col * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1 + self.scaled_flag_image.get_width() // 2,
                    BORDER_SIZE + row * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1 + self.scaled_flag_image.get_height() // 2,
                ),
            )

    def draw_cells_bayes(self) -> None:
        for row in range(self.board.n_rows):
            for col in range(self.board.n_cols):
                x = BORDER_SIZE + col * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1
                y = BORDER_SIZE + row * (CELL_SIZE + BORDER_SIZE * 2 + LINE_WIDTH) + 1

                if self.board.minefield[row, col]["state"] == self.board.states.UNCOVERED:
                    mc = int(self.board.minefield[row, col]["mine_count"])
                    if mc > 0:
                        text_surface = self.font.render(f"{mc}", True, text_color)
                        text_rect = text_surface.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                        self.screen.blit(text_surface, text_rect)

                if (
                    self.help
                    and (row, col) not in self.flagged
                    and self.board.minefield[row, col]["state"] == self.board.states.COVERED
                ):
                    probability = float(self.probability[row, col])
                    if probability == 0:
                        display_text = "S"
                    elif probability == 1:
                        display_text = "X"
                    else:
                        display_text = f"{probability:.2f}"

                    text_surface = self.font.render(display_text, True, get_custom_rgb(probability))
                    text_rect = text_surface.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                    self.screen.blit(text_surface, text_rect)

    def draw_lines(self) -> None:
        gap_size = CELL_SIZE // 8
        segment_length = CELL_SIZE - gap_size

        [
            pg.draw.line(
                self.screen,
                line_color,
                (
                    BORDER_SIZE * 2 * (col + 1) + CELL_SIZE * (col + 1) + LINE_WIDTH * col,
                    seg_start + gap_size,
                ),
                (
                    BORDER_SIZE * 2 * (col + 1) + CELL_SIZE * (col + 1) + LINE_WIDTH * col,
                    min(seg_start + segment_length, self.height),
                ),
                1,
            )
            for col in range(self.board.n_cols - 1)
            for seg_start in range(0, self.height, segment_length + gap_size)
        ]

        [
            pg.draw.line(
                self.screen,
                line_color,
                (
                    seg_start + gap_size,
                    BORDER_SIZE * 2 * (row + 1) + CELL_SIZE * (row + 1) + LINE_WIDTH * row,
                ),
                (
                    min(seg_start + segment_length, self.width),
                    BORDER_SIZE * 2 * (row + 1) + CELL_SIZE * (row + 1) + LINE_WIDTH * row,
                ),
                1,
            )
            for row in range(self.board.n_rows - 1)
            for seg_start in range(0, self.width, segment_length + gap_size)
        ]


def main() -> None:
    game = GUI(4)

    while game.running:
        game.handle_events()
        game.draw()
        pg.display.update()

    pg.quit()


if __name__ == "__main__":
    main()
