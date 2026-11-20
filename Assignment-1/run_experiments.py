"""
CSA1717 AI Assignment 1 - experiments for Tasks 8, 9, 10 (and Task 4 evidence)
Author : ANGELIN MARY C  (192521098, B.Tech IT)
Run    : python3 run_experiments.py      -> results.json + console log
Seeds  : graph seeds 101,102,103,104 for 1K,10K,100K,1M states ; goal sampling seed 2026
"""
import json
import math
import random
import sys
import time

from hospital_search import (ALGOS, bfs, ucs, iddfs, iddfs_tree, dls_count_calls, check_path,
                             generate_hospital, num_edges, bfs_distances, timed, edge_cost)

SIZES = [1000, 10000, 100000, 1000000]
SEEDS = {1000: 101, 10000: 102, 100000: 103, 1000000: 104}
RUNS = 5
out = {"sizes": SIZES, "seeds": SEEDS, "runs": RUNS}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ Task 8 / 9
main, tests = {}, {}
for n in SIZES:
    t0 = time.perf_counter()
    adj = generate_hospital(n, SEEDS[n])
    pre = time.perf_counter() - t0
    E = num_edges(adj)
    dist = bfs_distances(adj, 0)
    far = max(range(n), key=lambda v: (dist[v], v))          # farthest ward (deepest)
    sealed = n + 2                                             # ward inside the sealed wing
    log(f"== n={n}  seed={SEEDS[n]}  E={E}  preprocessing(build adjacency)={pre:.3f}s  far={far} depth={dist[far]}")
    rows = {}
    for name, fn in ALGOS.items():
        r, tm = timed(fn, adj, 0, far, RUNS)
        ok = check_path(adj, 0, far, r)
        rows[name] = {"found": r["found"], "depth": r["depth"], "cost": r["cost"],
                      "generated": r["generated"], "expanded": r["expanded"], "succ": r["succ"],
                      "max_frontier": r["max_frontier"], "time": tm, "valid": ok}
        if name == "IDDFS":
            rows[name]["iterations"] = r["iterations"]
            rows[name]["max_seen"] = r["max_seen"]
        log(f"   {name:6s} found={r['found']} depth={r['depth']} cost={r['cost']} gen={r['generated']} "
            f"exp={r['expanded']} frontier={r['max_frontier']} time={tm:.4f}s valid={ok}")
    best_cost = rows["UCS"]["cost"]
    best_depth = rows["BFS"]["depth"]
    for name in rows:
        rows[name]["cost_optimal"] = rows[name]["cost"] == best_cost
        rows[name]["depth_optimal"] = rows[name]["depth"] == best_depth
    main[n] = {"pre_time": pre, "E": E, "far": far, "far_depth": dist[far], "rows": rows}

    # ---- Task 8 test cases : reachable, unreachable, start == goal
    tc = {}
    for label, g in (("reachable", far), ("unreachable", sealed), ("start==goal", 0)):
        tc[label] = {}
        for name, fn in ALGOS.items():
            r = fn(adj, 0, g)
            ok = check_path(adj, 0, g, r)
            if label == "reachable":
                exp_ok = r["found"] is True
            elif label == "unreachable":
                exp_ok = r["found"] is False
            else:
                exp_ok = r["found"] is True and r["path"] == [0] and r["cost"] == 0
            tc[label][name] = {"found": r["found"], "depth": r["depth"], "cost": r["cost"],
                               "generated": r["generated"], "expanded": r["expanded"],
                               "pass": bool(ok and exp_ok)}
    tests[n] = tc
    log("   tests:", {k: all(v["pass"] for v in d.values()) for k, d in tc.items()})

    # keep the 10K graph for the best/average/worst-case study
    if n == 10000:
        adj10 = adj
        dist10 = dist
out["main"] = main
out["tests"] = tests

# ------------------------------------------------------------------ Task 4 evidence (n = 10,000)
rng = random.Random(2026)
n = 10000
adj, dist = adj10, dist10
nbr = adj[0][0]                                  # a ward 1 corridor from the pharmacy
reachable = [v for v in range(1, n) if dist[v] > 0]
sample = [rng.choice(reachable) for _ in range(200)]
cases = {}
for label, goals in (("best (goal 1 hop away)", [nbr]),
                     ("average (200 random wards)", sample),
                     ("worst reachable (deepest ward)", [main[n]["far"]]),
                     ("worst (unreachable ward)", [n + 2])):
    cases[label] = {}
    for name, fn in ALGOS.items():
        gens, exps = [], []
        for g in goals:
            r = fn(adj, 0, g)
            gens.append(r["generated"])
            exps.append(r["expanded"])
        cases[label][name] = {"generated": sum(gens) / len(gens), "expanded": sum(exps) / len(exps)}
# how often is BFS / IDDFS path costlier than UCS (the "cheap-edge detour" effect)
subopt = 0
extra = []
for g in sample:
    c_b = bfs(adj, 0, g)["cost"]
    c_u = ucs(adj, 0, g)["cost"]
    if c_b > c_u:
        subopt += 1
        extra.append(c_b - c_u)
cases["_bfs_suboptimal_fraction"] = subopt / len(sample)
cases["_bfs_mean_extra_cost_when_suboptimal"] = (sum(extra) / len(extra)) if extra else 0
out["cases"] = cases
log("== best/avg/worst study done; BFS non-optimal on", subopt, "of", len(sample), "goals")

# ------------------------------------------------------------------ recursive tree-search variant (theory check)
tree_rows = {}
for n in SIZES:
    adj = generate_hospital(n, SEEDS[n])
    far = main[n]["far"]
    t0 = time.perf_counter()
    r = iddfs_tree(adj, 0, far, budget=3_000_000)
    el = time.perf_counter() - t0
    tree_rows[n] = {"found": r["found"], "calls": r["calls"], "limit": r["limit"], "time": el,
                    "graph_search_expanded": main[n]["rows"]["IDDFS"]["expanded"]}
    log(f"   tree-IDDFS n={n}: found={r['found']} calls={r['calls']} limit={r['limit']} t={el:.2f}s")
    del adj
out["tree_variant"] = tree_rows

# ------------------------------------------------------------------ recurrence verification (Task 6/7)
rec_rows = []
for b in (2, 3, 4):
    for l in range(0, 9):
        if b ** l > 70000 and l > 0:
            continue
        calls = dls_count_calls(None, l, b)
        formula = (b ** (l + 1) - 1) // (b - 1)
        rec_rows.append({"b": b, "l": l, "calls": calls, "formula": formula, "match": calls == formula})
out["recurrence_check"] = rec_rows
log("== recurrence check all match:", all(r["match"] for r in rec_rows))


def complete_tree(b, d):
    adj = [[]]
    level = [0]
    for _ in range(d):
        nxt = []
        for u in level:
            for _ in range(b):
                v = len(adj)
                adj.append([u])
                adj[u].append(v)
                nxt.append(v)
        level = nxt
    return adj, level[-1]


idd_rows = []
for b, d in ((2, 8), (3, 6), (4, 5)):
    adj, goal = complete_tree(b, d)
    # root has no parent edge, so every node has b children: matches the model
    r = iddfs_tree(adj, 0, goal, budget=10 ** 8)
    predicted = sum((b ** (l + 1) - 1) // (b - 1) for l in range(d))      # full iterations 0..d-1
    last = None
    idd_rows.append({"b": b, "d": d, "calls": r["calls"], "found": r["found"],
                     "sum_full_iterations": predicted,
                     "bound_b_over_b_minus_1": (b / (b - 1)) * (b ** d)})
out["iddfs_tree_check"] = idd_rows
log("== IDDFS on complete trees:", idd_rows)

json.dump(out, open("results.json", "w"), indent=1, default=str)
log("saved results.json")
