# lee_algo.py
from __future__ import annotations
from collections import deque
from typing import List, Tuple, Optional, Dict, Any, Union

from generator import Grid, Coord  # берём типы и будем использовать их тут

LeeResult = Union[Optional[List[Coord]], Tuple[Optional[List[Coord]], Dict[str, Any]]]


def lee_shortest_path(
    grid: Grid,
    start: Coord,
    goal: Coord,
    *,
    allow_diagonals: bool = False,
    return_metrics: bool = False,
) -> LeeResult:
    """
    Алгоритм Ли (BFS) на решётке: кратчайший путь от start к goal.
    grid: 0 — свободно, 1 — препятствие.
    return_metrics=True -> вернёт (path, {"visited":..., "queue_max":..., "path_len":...})
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    if rows == 0 or cols == 0:
        return (None, {"visited": 0, "queue_max": 0, "path_len": 0}) if return_metrics else None

    sx, sy = start
    gx, gy = goal

    # Проверка границ
    if not (0 <= sx < rows and 0 <= sy < cols and 0 <= gx < rows and 0 <= gy < cols):
        return (None, {"visited": 0, "queue_max": 0, "path_len": 0}) if return_metrics else None

    # Препятствия
    if grid[sx][sy] == 1 or grid[gx][gy] == 1:
        return (None, {"visited": 0, "queue_max": 0, "path_len": 0}) if return_metrics else None

    # Тривиальный случай
    if start == goal:
        if return_metrics:
            return [start], {"visited": 1, "queue_max": 1, "path_len": 1}
        return [start]

    # Соседства
    if allow_diagonals:
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]
    else:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Матрицы расстояний и родителей
    dist = [[-1] * cols for _ in range(rows)]
    parent: List[List[Optional[Coord]]] = [[None] * cols for _ in range(rows)]

    q = deque([start])
    dist[sx][sy] = 0
    parent[sx][sy] = start

    visited = 0
    queue_max = 1

    # Волна
    while q:
        x, y = q.popleft()
        visited += 1
        if (x, y) == goal:
            break
        nd = dist[x][y] + 1
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] == 0 and dist[nx][ny] == -1:
                    dist[nx][ny] = nd
                    parent[nx][ny] = (x, y)
                    q.append((nx, ny))
        if len(q) > queue_max:
            queue_max = len(q)

    # Не дошли
    if dist[gx][gy] == -1:
        return (None, {"visited": visited, "queue_max": queue_max, "path_len": 0}) if return_metrics else None

    # Восстановление пути
    path: List[Coord] = []
    cx, cy = gx, gy
    while True:
        path.append((cx, cy))
        if (cx, cy) == start:
            break
        p = parent[cx][cy]
        if p is None:
            return (None, {"visited": visited, "queue_max": queue_max, "path_len": 0}) if return_metrics else None
        cx, cy = p
    path.reverse()

    if return_metrics:
        return path, {
            "visited": visited,
            "queue_max": queue_max,
            "path_len": len(path),
        }
    return path
