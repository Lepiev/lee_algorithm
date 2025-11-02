# experiment_sizes.py
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
    SIZES = [20, 40, 80, 160, 320, 640, 1280]
    P = 0.2
    RUNS_PER_SIZE = 20
    BASE_SEED = 42
    ENSURE_PATH = True
    ALLOW_DIAGONALS = False

    # --- ХРАНИЛИЩА РЕЗУЛЬТАТОВ ---
    avg_time_ms = []
    avg_visited = []
    avg_queue_max = []
    success_rate = []
    avg_path_len = []

    raw_records = []

    for idxN, N in enumerate(SIZES):
        times_ms = []
        visiteds = []
        qmaxes = []
        succs = []
        pathlens = []

        for r in range(RUNS_PER_SIZE):
            seed = BASE_SEED + idxN * RUNS_PER_SIZE + r if BASE_SEED is not None else None

            grid, start, end = generate_grid(
                N, N, P,
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
            visiteds.append(metrics.get("visited", np.nan))
            qmaxes.append(metrics.get("queue_max", np.nan))
            succs.append(1 if path is not None else 0)
            pathlens.append(metrics.get("path_len", 0) if path is not None else 0)

            raw_records.append({
                "N": N,
                "time_ms": elapsed_ms,
                "visited": metrics.get("visited", np.nan),
                "queue_max": metrics.get("queue_max", np.nan),
                "success": 1 if path is not None else 0,
                "path_len": metrics.get("path_len", 0) if path is not None else 0
            })

        avg_time_ms.append(mean(times_ms))
        avg_visited.append(mean(visiteds))
        avg_queue_max.append(mean(qmaxes))
        success_rate.append(mean(succs))
        avg_path_len.append(
            mean([pl for pl in pathlens if pl > 0]) if any(pathlens) else 0
        )

    # --- ТАБЛИЦА ДЛЯ ОТЧЁТА ---
    df = pd.DataFrame({
        "N": SIZES,
        "avg_time_ms": avg_time_ms,
        "avg_visited": avg_visited,
        "avg_queue_max": avg_queue_max,
        "success_rate": success_rate,
        "avg_path_len_success": avg_path_len
    })
    print(df.round(3).to_string(index=False))
    # при необходимости: df.to_csv("exp2_sizes_p02.csv", index=False)

    # --- ВИЗУАЛИЗАЦИИ ---

    # 1) Время vs размер сетки
    plt.figure(figsize=(7, 4))
    plt.plot(SIZES, avg_time_ms, marker="o")
    plt.xlabel("Размер сетки N (N×N)")
    plt.ylabel("Среднее время, мс")
    plt.title(f"Алгоритм Ли: время vs размер (p={P}, runs={RUNS_PER_SIZE}, ensure_path={ENSURE_PATH})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) Посещённые вершины vs размер сетки
    plt.figure(figsize=(7, 4))
    plt.plot(SIZES, avg_visited, marker="o")
    plt.xlabel("Размер сетки N (N×N)")
    plt.ylabel("Среднее число посещённых клеток (visited)")
    plt.title(f"Алгоритм Ли: объём работы BFS vs размер (p={P})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3) Лог-лог: время vs N
    logN = np.log(np.array(SIZES, dtype=float))
    logT = np.log(np.array(avg_time_ms, dtype=float))
    slope, intercept = np.polyfit(logN, logT, 1)  # log T ≈ intercept + slope * log N

    plt.figure(figsize=(7, 4))
    plt.plot(logN, logT, marker="o", linestyle="none", label="измерения")
    plt.plot(logN, intercept + slope * logN, label=f"аппроксимация, наклон ≈ {slope:.2f}")
    plt.xlabel("log N")
    plt.ylabel("log (время, мс)")
    plt.title(f"Лог-лог график времени (оценка степени роста ~ N^{slope:.2f})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4) Отношения при удвоении N
    pairs = [(20, 40), (40, 80), (160, 320), (640, 1280)]
    print("Отношения средних времён при удвоении N:")
    for a, b in pairs:
        ta = df.loc[df["N"] == a, "avg_time_ms"].values[0]
        tb = df.loc[df["N"] == b, "avg_time_ms"].values[0]
        ratio = tb / ta if ta > 0 else float("nan")
        print(f"N: {a} → {b}: ×{ratio:.2f} (ожидание ~×4 при квадратичном росте)")


if __name__ == "__main__":
    main()
