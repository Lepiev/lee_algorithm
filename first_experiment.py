# first_experiment.py
from __future__ import annotations
import time
from statistics import mean

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generator import generate_grid
from lee_algo import lee_shortest_path


def main() -> None:
    # --- ПАРАМЕТРЫ ЭКСПЕРИМЕНТА ---
    N = 50
    P_VALUES = [round(p, 2) for p in np.arange(0.0, 1.0001, 0.05)]  # 0.00 ... 1.00
    RUNS_PER_P = 100
    BASE_SEED = 42             # для воспроизводимости (можно None)
    ENSURE_PATH = True         # если True, будет перегенерировать до существования пути
    ALLOW_DIAGONALS = False    # 4-соседей

    # --- ХРАНИЛИЩА РЕЗУЛЬТАТОВ ---
    avg_time_ms = []
    avg_visited = []
    avg_queue_max = []
    success_rate = []
    avg_path_len_ok = []

    raw_records = []

    for idx_p, p in enumerate(P_VALUES):
        times_ms = []
        visited_list = []
        qmax_list = []
        success_list = []
        path_len_list_ok = []

        for r in range(RUNS_PER_P):
            seed = BASE_SEED + idx_p * RUNS_PER_P + r if BASE_SEED is not None else None

            grid, start, end = generate_grid(
                N, N, p,
                seed=seed,
                ensure_path=ENSURE_PATH
            )

            t0 = time.perf_counter()
            path, metrics = lee_shortest_path(
                grid, start, end,
                allow_diagonals=ALLOW_DIAGONALS,
                return_metrics=True
            )
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) * 1000.0
            times_ms.append(elapsed_ms)

            v = metrics.get("visited", np.nan)
            q = metrics.get("queue_max", np.nan)
            pl = metrics.get("path_len", 0)

            visited_list.append(v)
            qmax_list.append(q)

            if path is not None:
                success_list.append(1)
                if pl:
                    path_len_list_ok.append(pl)
            else:
                success_list.append(0)

            raw_records.append({
                "p": p,
                "time_ms": elapsed_ms,
                "visited": v,
                "queue_max": q,
                "success": 1 if path is not None else 0,
                "path_len": pl if path is not None else 0
            })

        # Агрегация по p
        avg_time_ms.append(mean(times_ms))
        avg_visited.append(mean(visited_list))
        avg_queue_max.append(mean(qmax_list))
        success_rate.append(mean(success_list))
        avg_path_len_ok.append(mean(path_len_list_ok) if path_len_list_ok else 0)

    # --- Таблица-итог ---
    df = pd.DataFrame({
        "p": P_VALUES,
        "avg_time_ms": avg_time_ms,
        "avg_visited": avg_visited,
        "avg_queue_max": avg_queue_max,
        "success_rate": success_rate,
        "avg_path_len_success": avg_path_len_ok
    })
    print(df.round(3).to_string(index=False))

    # --- ВИЗУАЛИЗАЦИИ ---

    # 1) Время vs плотность препятствий
    plt.figure(figsize=(7, 4))
    plt.plot([p * 100 for p in P_VALUES], avg_time_ms, marker="o")
    plt.xlabel("Плотность препятствий, %")
    plt.ylabel("Среднее время, мс")
    plt.title(f"Время работы алгоритма Ли при N={N} (runs={RUNS_PER_P}, ensure_path={ENSURE_PATH})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) Кол-во посещённых клеток vs плотность препятствий
    plt.figure(figsize=(7, 4))
    plt.plot([p * 100 for p in P_VALUES], avg_visited, marker="o")
    plt.xlabel("Плотность препятствий, %")
    plt.ylabel("Среднее число обработанных клеток (visited)")
    plt.title(f"Объём работы BFS при N={N}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3) Доля успешных прогонов
    plt.figure(figsize=(7, 4))
    plt.plot([p * 100 for p in P_VALUES], success_rate, marker="o")
    plt.xlabel("Плотность препятствий, %")
    plt.ylabel("Доля успешных запусков")
    plt.title(f"Вероятность существования пути при N={N}")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4) Пиковый размер очереди
    plt.figure(figsize=(7, 4))
    plt.plot([p * 100 for p in P_VALUES], avg_queue_max, marker="o")
    plt.xlabel("Плотность препятствий, %")
    plt.ylabel("Средний пик очереди")
    plt.title(f"Пиковая нагрузка очереди BFS при N={N}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
