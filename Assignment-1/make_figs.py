"""Figures for the report (reads results.json and belief_results.json)."""
import json
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hospital_search import generate_hospital, bfs_distances
from belief_search import Floor

os.makedirs("figs", exist_ok=True)
R = json.load(open("results.json"))
B = json.load(open("belief_results.json"))
SIZES = R["sizes"]
COL = {"BFS": "#0072B2", "UCS": "#E69F00", "IDDFS": "#009E73"}
MK = {"BFS": "o", "UCS": "s", "IDDFS": "^"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})


def series(metric, algo):
    return [R["main"][str(n)]["rows"][algo][metric] for n in SIZES]


def loglog(ax, ys, label_y, title):
    for a in ("BFS", "UCS", "IDDFS"):
        ax.plot(SIZES, ys[a], marker=MK[a], color=COL[a], label=a, lw=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("State-space size V (locations)"); ax.set_ylabel(label_y); ax.set_title(title, fontsize=11)
    ax.legend(frameon=False)


# 1. nodes generated / expanded
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
loglog(ax[0], {a: series("generated", a) for a in COL}, "Nodes generated", "Nodes generated vs size")
ax[0].plot(SIZES, SIZES, "k--", lw=1, label="y = V"); ax[0].legend(frameon=False)
loglog(ax[1], {a: series("expanded", a) for a in COL}, "Nodes expanded", "Nodes expanded vs size")
plt.tight_layout(); plt.savefig("figs/fig_nodes.png"); plt.close()

# 2. memory proxies
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
mem = {"BFS": series("max_frontier", "BFS"), "UCS": series("max_frontier", "UCS"),
       "IDDFS": series("max_frontier", "IDDFS")}
loglog(ax[0], mem, "Max frontier / stack size", "Memory proxy: maximum frontier")
ax[0].plot(SIZES, [0.32 * n for n in SIZES], "k:", lw=1)
ax[0].text(SIZES[1], 0.32 * SIZES[1] * 1.6, "~0.32 V", fontsize=8)
ax[1].plot(SIZES, series("max_frontier", "IDDFS"), marker="^", color=COL["IDDFS"], lw=2, label="IDDFS recursion stack (b*d part)")
ax[1].plot(SIZES, series("max_seen", "IDDFS"), marker="v", color="#7a3e9d", lw=2, label="IDDFS visited map (graph-search)")
ax[1].plot(SIZES, series("max_frontier", "BFS"), marker="o", color=COL["BFS"], lw=2, label="BFS frontier")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("State-space size V")
ax[1].set_ylabel("Entries held in memory"); ax[1].set_title("Where IDDFS memory really goes", fontsize=11)
ax[1].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig("figs/fig_memory.png"); plt.close()

# 3. time
fig, ax = plt.subplots(figsize=(5.2, 3.8))
loglog(ax, {a: series("time", a) for a in COL}, "Mean search time (s), 5 runs", "Execution time vs size")
plt.tight_layout(); plt.savefig("figs/fig_time.png"); plt.close()

# 4. best / average / worst
cases = [k for k in R["cases"] if not k.startswith("_")]
short = ["best\n(1 hop)", "average\n(200 wards)", "worst reachable\n(deepest)", "worst\n(unreachable)"]
fig, ax = plt.subplots(figsize=(8, 3.8))
w = 0.26
for i, a in enumerate(("BFS", "UCS", "IDDFS")):
    ys = [R["cases"][c][a]["generated"] for c in cases]
    bars = ax.bar(np.arange(4) + (i - 1) * w, ys, w, color=COL[a], label=a)
    for x, y in zip(np.arange(4) + (i - 1) * w, ys):
        ax.text(x, y * 1.15, f"{y:,.0f}", ha="center", fontsize=6.5, rotation=90)
ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(short, fontsize=8)
ax.set_ylabel("Nodes generated (log scale)"); ax.set_title("Case study on the 10,000-state hospital", fontsize=11)
ax.set_ylim(1, 6e5); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig("figs/fig_cases.png"); plt.close()

# 5. belief states vs k
ks = [r["k"] for r in B["rows"]]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(ks, [r["sensorless"]["generated"] for r in B["rows"]], marker="o", color="#D55E00", lw=2, label="no sensor (BFS over beliefs)")
ax[0].plot(ks, [r["sensor"]["generated"] for r in B["rows"]], marker="s", color="#0072B2", lw=2, label="local wall sensor (AND-OR)")
ax[0].set_yscale("log"); ax[0].set_xscale("log", base=2); ax[0].set_xticks(ks); ax[0].set_xticklabels(ks)
ax[0].set_xlabel("k = possible junctions"); ax[0].set_ylabel("Belief states generated (mean of 20)")
ax[0].set_title("Belief states generated", fontsize=11); ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(ks, [r["sensorless"]["plan_len"] for r in B["rows"]], marker="o", color="#D55E00", lw=2, label="no sensor: fixed plan length")
ax[1].plot(ks, [r["sensor"]["worst_len"] for r in B["rows"]], marker="s", color="#0072B2", lw=2, label="sensor: worst-case moves")
ax[1].plot(ks, [r["sensor"]["avg_len"] for r in B["rows"]], marker="^", color="#009E73", lw=2, ls="--", label="sensor: average moves")
ax[1].set_xscale("log", base=2); ax[1].set_xticks(ks); ax[1].set_xticklabels(ks)
ax[1].set_xlabel("k = possible junctions"); ax[1].set_ylabel("Plan length (moves)"); ax[1].set_title("Plan length", fontsize=11)
ax[1].legend(frameon=False, fontsize=8); ax[1].set_ylim(0, 14)
plt.tight_layout(); plt.savefig("figs/fig_belief.png"); plt.close()

# 6. floor map
fl = Floor(7, 7, 0.10, 5)
fig, ax = plt.subplots(figsize=(3.8, 3.8))
for r in range(fl.R):
    for c in range(fl.C):
        free = (r, c) in fl.free
        ax.add_patch(plt.Rectangle((c, fl.R - 1 - r), 1, 1, fc="#f2f2f2" if free else "#555", ec="white"))
init = random.Random(1004).sample(sorted(fl.free - {fl.dock}), 4)
for (r, c) in init:
    ax.plot(c + .5, fl.R - 1 - r + .5, "o", color="#D55E00", ms=11)
ax.plot(fl.dock[1] + .5, fl.R - 1 - fl.dock[0] + .5, "*", color="#0072B2", ms=18)
ax.set_xlim(0, fl.C); ax.set_ylim(0, fl.R); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Floor: dock (star), k=4 candidate junctions (dots)", fontsize=8)
plt.tight_layout(); plt.savefig("figs/fig_floor.png"); plt.close()

# 7. hospital graph illustration (80 locations, rings by BFS distance from the pharmacy)
adj = generate_hospital(80, 3, island=0)
d = bfs_distances(adj, 0)
rng = random.Random(1)
pos = {}
levels = {}
for v, dv in enumerate(d):
    levels.setdefault(dv, []).append(v)
for dv, vs in levels.items():
    for i, v in enumerate(vs):
        ang = 2 * math.pi * (i + rng.random() * 0.4) / len(vs)
        pos[v] = (dv * math.cos(ang), dv * math.sin(ang)) if dv else (0, 0)
fig, ax = plt.subplots(figsize=(4.6, 4.6))
for u in range(80):
    for v in adj[u]:
        if u < v:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#bbbbbb", lw=0.7, zorder=1)
ax.scatter([pos[v][0] for v in range(80)], [pos[v][1] for v in range(80)], s=14, c=[d[v] for v in range(80)],
           cmap="Blues_r", zorder=2, edgecolors="#444", linewidths=0.3)
ax.scatter([0], [0], s=160, marker="*", color="#D55E00", zorder=3)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Sample 80-location hospital graph (rings = corridors\nfrom pharmacy, star = pharmacy)", fontsize=8)
plt.tight_layout(); plt.savefig("figs/fig_graph.png"); plt.close()

# 8. recursion tree for T(l) = b T(l-1) + c   (b = 3)
fig, ax = plt.subplots(figsize=(7, 3.3))
levels = 3
b = 3
xs = {0: [0.0]}
for l in range(1, levels + 1):
    xs[l] = []
    for i in range(b ** l):
        xs[l].append((i + 0.5) / b ** l * 8 - 4)
for l in range(levels):
    for pi, px in enumerate(xs[l]):
        for j in range(b):
            cx = xs[l + 1][pi * b + j]
            ax.plot([px, cx], [-l, -l - 1], color="#999", lw=0.8)
for l in range(levels + 1):
    ax.scatter(xs[l], [-l] * len(xs[l]), s=60 if l < 2 else 22, color="#0072B2", zorder=3)
    lab = f"T(l-{l})" if l else "T(l)"
    ax.text(-4.8, -l, f"{lab}\n{b ** l} calls", ha="right", va="center", fontsize=8)
ax.set_xlim(-6.2, 4.4); ax.set_ylim(-3.5, 0.5); ax.axis("off")
ax.set_title("Recursion tree of depth-limited search (b = 3): level i has b^i calls, each doing c work", fontsize=9)
plt.tight_layout(); plt.savefig("figs/fig_rectree.png"); plt.close()
print("figures done", os.listdir("figs"))
