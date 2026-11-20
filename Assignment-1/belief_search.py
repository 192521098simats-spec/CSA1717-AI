"""
CSA1717 AI Assignment 1 - Day-4 Task 12 : localisation failure -> search over BELIEF STATES
Author : ANGELIN MARY C  (192521098, B.Tech IT)

World   : one hospital floor modelled as a grid of junctions (R x C) with blocked cells (walls / equipment).
Actions : U, D, L, R  (a move into a wall or the floor boundary leaves the robot where it is)
Belief  : frozenset of junctions the robot might be at.  Goal : belief == {dock}.
Case 1 (no sensor)    : sensorless / conformant planning  -> plain BFS over belief states -> fixed plan
Case 2 (local sensor) : after every move the robot senses which of its 4 neighbours are blocked
                        (wall pattern).  The percept splits the belief into smaller beliefs ->
                        AND-OR search (iterative deepening on worst-case plan length) -> conditional plan.
Run : python3 belief_search.py  -> belief_results.json
"""
import json
import random
import time
from collections import deque

ACTIONS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def make_floor(R=10, C=10, wall_p=0.12, seed=7):
    rng = random.Random(seed)
    free = {(r, c) for r in range(R) for c in range(C) if rng.random() > wall_p}
    # keep the biggest connected region only (all junctions must be able to reach the dock)
    seen, best = set(), set()
    for s in sorted(free):
        if s in seen:
            continue
        comp, q = {s}, deque([s])
        while q:
            r, c = q.popleft()
            for dr, dc in ACTIONS.values():
                t = (r + dr, c + dc)
                if t in free and t not in comp:
                    comp.add(t)
                    q.append(t)
        seen |= comp
        if len(comp) > len(best):
            best = comp
    dock = min(best, key=lambda p: (p[0] + p[1], p))      # pharmacy docking station near a corner
    return best, dock


class Floor:
    def __init__(self, R=10, C=10, wall_p=0.12, seed=7):
        self.R, self.C = R, C
        self.free, self.dock = make_floor(R, C, wall_p, seed)
        self.step = {}                                    # (cell, action) -> next cell
        self.percept = {}                                 # cell -> wall pattern
        for p in self.free:
            pat = []
            for a, (dr, dc) in ACTIONS.items():
                t = (p[0] + dr, p[1] + dc)
                self.step[(p, a)] = t if t in self.free else p
                pat.append(0 if t in self.free else 1)
            self.percept[p] = tuple(pat)

    def predict(self, belief, a):
        return frozenset(self.step[(p, a)] for p in belief)

    def partition(self, belief):
        groups = {}
        for p in belief:
            groups.setdefault(self.percept[p], set()).add(p)
        return [frozenset(g) for g in groups.values()]


# --------------------------------------------------------------------------- the ORIGINAL design applied naively
def physical_bfs(fl, start):
    """Original design: BFS over single physical states (fully observable, known start)."""
    parent = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        if u == fl.dock:
            plan = []
            while parent[u]:
                u, a = parent[u]
                plan.append(a)
            return plan[::-1]
        for a in ACTIONS:
            v = fl.step[(u, a)]
            if v not in parent:
                parent[v] = (u, a)
                q.append(v)
    return None


def run_plan(fl, p, plan):
    for a in plan:
        p = fl.step[(p, a)]
    return p


# --------------------------------------------------------------------------- Case 1 : sensorless belief-state BFS
def sensorless_bfs(fl, init, cap=2_000_000):
    start = frozenset(init)
    goal = frozenset([fl.dock])
    parent = {start: None}
    q = deque([start])
    generated, expanded = 1, 0
    if start == goal:
        return {"solved": True, "plan": [], "generated": 1, "expanded": 0}
    while q:
        b = q.popleft()
        expanded += 1
        for a in ACTIONS:
            nb = fl.predict(b, a)
            if nb not in parent:
                parent[nb] = (b, a)
                generated += 1
                if nb == goal:
                    plan, cur = [], nb
                    while parent[cur]:
                        cur, act = parent[cur]
                        plan.append(act)
                    return {"solved": True, "plan": plan[::-1], "generated": generated, "expanded": expanded}
                q.append(nb)
                if generated > cap:
                    return {"solved": None, "plan": None, "generated": generated, "expanded": expanded}
    return {"solved": False, "plan": None, "generated": generated, "expanded": expanded}


# --------------------------------------------------------------------------- Case 2 : AND-OR search with local sensor
def sensor_andor(fl, init, max_len=60):
    goal = frozenset([fl.dock])
    seen_states = set()                       # distinct belief states generated (memory proxy)
    solved = {}                               # belief -> (min limit proven solvable, action)
    failed = {}                               # belief -> largest limit proven unsolvable
    expansions = [0]

    def solve(b, limit):
        if b == goal:
            return True
        if limit == 0:
            return False
        if b in solved and solved[b][0] <= limit:
            return True
        if failed.get(b, -1) >= limit:
            return False
        expansions[0] += 1
        for a in ACTIONS:
            nb = fl.predict(b, a)
            parts = fl.partition(nb)          # AND-node: every percept outcome must be solved
            for q in parts:
                seen_states.add(q)
            if all(solve(q, limit - 1) for q in parts):
                if b not in solved or limit < solved[b][0]:     # keep the best (shallowest) proof only;
                    solved[b] = (limit, a)                       # a self-loop move must never overwrite it
                return True
        failed[b] = limit
        return False

    top = fl.partition(frozenset(init))        # robot senses once right after restart
    for q in top:
        seen_states.add(q)
    worst = 0
    for q in top:
        for limit in range(0, max_len + 1):
            if solve(q, limit):
                worst = max(worst, limit)
                break
        else:
            return {"solved": False, "generated": len(seen_states), "expanded": expansions[0]}
    return {"solved": True, "worst_len": worst, "generated": len(seen_states), "expanded": expansions[0],
            "top": top, "table": solved}


def execute_conditional(fl, res, true_start, max_steps=200):
    """Simulate the conditional plan from the true (hidden) start; returns number of moves."""
    table = res["table"]
    goal = frozenset([fl.dock])
    b = None
    for q in res["top"]:
        if true_start in q:
            b = q
    p, steps = true_start, 0
    while b != goal and steps < max_steps:
        limit, a = table[b]
        # choose the action stored for the smallest proven limit; re-derive using current best
        p = fl.step[(p, a)]
        nb = fl.predict(b, a)
        b = [q for q in fl.partition(nb) if p in q][0]
        steps += 1
    assert p == fl.dock, "conditional plan failed to reach the dock"
    return steps


def main(R=7, C=7, seeds=20):
    fl = Floor(R, C, 0.10, seed=5)
    out = {"R": R, "C": C, "free_cells": len(fl.free), "dock": fl.dock, "trials": seeds, "rows": []}
    cells = sorted(fl.free)
    for k in (2, 4, 8):
        agg = {"naive_ok": [], "sl_gen": [], "sl_len": [], "sl_exp": [], "sl_solved": 0,
               "se_gen": [], "se_len": [], "se_avg": [], "se_exp": [], "se_solved": 0, "se_t": [], "sl_t": []}
        for s in range(seeds):
            rng = random.Random(1000 * k + s)
            init = rng.sample(cells, k)
            # (a) original design: plan for ONE assumed position, execute from every possible position
            plan = physical_bfs(fl, init[0])
            ok = sum(1 for p in init if run_plan(fl, p, plan) == fl.dock) / k
            agg["naive_ok"].append(ok)
            # (b) sensorless belief search
            t0 = time.perf_counter()
            r1 = sensorless_bfs(fl, init)
            agg["sl_t"].append(time.perf_counter() - t0)
            if r1["solved"]:
                agg["sl_solved"] += 1
                agg["sl_len"].append(len(r1["plan"]))
            agg["sl_gen"].append(r1["generated"])
            agg["sl_exp"].append(r1["expanded"])
            # (c) local sensor AND-OR search
            t0 = time.perf_counter()
            r2 = sensor_andor(fl, init)
            agg["se_t"].append(time.perf_counter() - t0)
            agg["se_gen"].append(r2["generated"])
            agg["se_exp"].append(r2["expanded"])
            if r2["solved"]:
                agg["se_solved"] += 1
                agg["se_len"].append(r2["worst_len"])
                agg["se_avg"].append(sum(execute_conditional(fl, r2, p) for p in init) / k)
        m = lambda x: (sum(x) / len(x)) if x else None
        out["rows"].append({
            "k": k,
            "naive_success_rate": m(agg["naive_ok"]),
            "sensorless": {"solved": agg["sl_solved"], "generated": m(agg["sl_gen"]), "expanded": m(agg["sl_exp"]),
                           "plan_len": m(agg["sl_len"]), "time": m(agg["sl_t"])},
            "sensor": {"solved": agg["se_solved"], "generated": m(agg["se_gen"]), "expanded": m(agg["se_exp"]),
                       "worst_len": m(agg["se_len"]), "avg_len": m(agg["se_avg"]), "time": m(agg["se_t"])},
        })
        print(out["rows"][-1], flush=True)
    # theoretical bound on belief-space size for n junctions and k initial candidates
    from math import comb
    n = len(fl.free)
    out["n"] = n
    out["bound"] = {k: sum(comb(n, i) for i in range(1, k + 1)) for k in (2, 4, 8)}
    out["full_power_set"] = 2 ** n - 1
    # scale-up demonstration on a bigger floor (10 x 10): sensorless beliefs explode, sensor stays small
    big = Floor(10, 10, 0.12, seed=7)
    bcells = sorted(big.free)
    out["scaleup"] = {"n": len(big.free), "rows": []}
    for k in (2, 4, 8):
        init = random.Random(5).sample(bcells, k)
        r1 = sensorless_bfs(big, init, cap=300_000)
        r2 = sensor_andor(big, init, max_len=40)
        row = {"k": k, "sensorless_solved": r1["solved"], "sensorless_generated": r1["generated"],
               "sensor_solved": r2["solved"], "sensor_generated": r2["generated"],
               "sensor_worst_len": r2.get("worst_len")}
        out["scaleup"]["rows"].append(row)
        print(row, flush=True)
    json.dump(out, open("belief_results.json", "w"), indent=1, default=str)
    return out, fl


if __name__ == "__main__":
    main()
