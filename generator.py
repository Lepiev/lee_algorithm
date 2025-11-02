# generator.py
from __future__ import annotations
from typing import List, Tuple, Optional
import random
from collections import deque

# Базовые типы, чтобы можно было импортировать в других файлах
Grid = List[List[int]]
Coord = Tuple[int, int]


def _random_free_cell(grid: Grid, rng: random.Random) -> Optional[Coord]:
    """Возвращает случайную свободную (0) клетку или None, если таких нет."""
    rows, cols = len(grid), len(grid[0])
    free = [(i, j) for i in range(rows) for j in range(cols) if grid[i][j] == 0]
    if not free:
        return None
    return rng.choice(free)


def generate_grid(
    rows: int,
    cols: int,
    obstacle_probability: float,
    *,
    seed: Optional[int] = None,
    ensure_path: bool = False,
    max_tries: int = 1000,
) -> Tuple[Grid, Coord, Coord]:
    """
    Генерирует решётку rows x cols со случайными препятствиями и
    случайными стартом/финишем на свободных клетках.

    Возвращает:
      grid, start, end
    """
    rng = random.Random(seed)

    def _one_attempt() -> Tuple[Grid, Coord, Coord]:
        # 1) сгенерировать сетку
        grid = [
            [1 if rng.random() < obstacle_probability else 0 for _ in range(cols)]
            for _ in range(rows)
        ]

        # 2) выбрать случайный старт
        start = _random_free_cell(grid, rng)
        if start is None:
            # сетка полностью забита, освободим две клетки вручную
            i1, j1 = rng.randrange(rows), rng.randrange(cols)
            grid[i1][j1] = 0
            i2, j2 = rng.randrange(rows), rng.randrange(cols)
            while i2 == i1 and j2 == j1:
                i2, j2 = rng.randrange(rows), rng.randrange(cols)
            grid[i2][j2] = 0
            return grid, (i1, j1), (i2, j2)

        # 3) выбрать случайный финиш, отличный от старта
        end = _random_free_cell(grid, rng)
        tries = 0
        while (end is None or end == start) and tries < rows * cols:
            end = _random_free_cell(grid, rng)
            tries += 1
        if end is None or end == start:
            # на всякий случай вручную освободим другую клетку
            i2, j2 = rng.randrange(rows), rng.randrange(cols)
            while (i2, j2) == start:
                i2, j2 = rng.randrange(rows), rng.randrange(cols)
            grid[i2][j2] = 0
            end = (i2, j2)
        return grid, start, end

    if not ensure_path:
        return _one_attempt()

    # Режим ensure_path: добиваемся существования пути (обычный BFS)
    def _has_path(grid: Grid, s: Coord, t: Coord) -> bool:
        if grid[s[0]][s[1]] == 1 or grid[t[0]][t[1]] == 1:
            return False
        rows_, cols_ = len(grid), len(grid[0])
        q = deque([s])
        seen = {s}
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while q:
            x, y = q.popleft()
            if (x, y) == t:
                return True
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows_ and 0 <= ny < cols_ and grid[nx][ny] == 0:
                    if (nx, ny) not in seen:
                        seen.add((nx, ny))
                        q.append((nx, ny))
        return False

    last = None
    for _ in range(max_tries):
        grid, start, end = _one_attempt()
        last = (grid, start, end)
        if _has_path(grid, start, end):
            return grid, start, end

    # если не нашли за max_tries — возвращаем последнюю попытку как есть
    assert last is not None
    return last
