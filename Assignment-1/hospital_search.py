"""
CSA1717 - Artificial Intelligence | Assignment 1 (CO1)
Hospital Medicine-Delivery Robot : uninformed search strategies
Author : ANGELIN MARY C  (Reg No 192521098, B.Tech IT)

Contents
  generate_hospital()  - reproducible hospital graph generator (preprocessing)
  bfs()                - Strategy A : Breadth-First Search          (iterative, queue)
  ucs()                - Strategy B : Uniform-Cost Search           (iterative, priority queue)
  iddfs()              - Strategy C : Iterative-Deepening DFS       (recursive, depth-limited)
  iddfs_tree()         - Strategy C (tree-search variant, path check only) used for theory checks
  check_path()         - independent validator of any returned route

Every search returns a dict with the same metrics:
  found, path, cost, depth, generated, expanded, succ (successors examined), max_frontier
"""
import heapq
import random
import sys
import time
from collections import deque

INF = float("inf")
sys.setrecursionlimit(10000)


# ----------------------------------------------------------------------------
# Environment : hospital graph
# ----------------------------------------------------------------------------
def edge_cost(u, v):
    """Travel time (1..9 minutes) of corridor u-v. Symmetric and deterministic,
    so the weights need not be stored (saves memory for 10^6 states)."""
    a, b = (u, v) if u < v else (v, u)
    return 1 + ((a * 73856093) ^ (b * 19349663)) % 9


def generate_hospital(n, seed, extra_ratio=0.5, island=5):
    """Return adjacency lists for a hospital with n connected locations plus
    `island` locations in a sealed wing (unreachable from the pharmacy, id n..n+island-1).

    Backbone : random recursive tree rooted at the pharmacy (node 0) - corridors
               that fan out from the central pharmacy.
    Extra    : extra_ratio*n cross links (loops, lifts, bridges between blocks).
    Preprocessing cost = O(V + E)   (counted in the efficiency analysis)."""
    rng = random.Random(seed)
    adj = [[] for _ in range(n + island)]
    for v in range(1, n):
        p = rng.randrange(v)
        adj[v].append(p)
        adj[p].append(v)
    for _ in range(int(extra_ratio * n)):
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v and v not in adj[u]:
            adj[u].append(v)
            adj[v].append(u)
    for i in range(n, n + island - 1):          # sealed wing (a small chain)
        adj[i].append(i + 1)
        adj[i + 1].append(i)
    return adj


def num_edges(adj):
    return sum(len(a) for a in adj) // 2


# ----------------------------------------------------------------------------
# helper
# ----------------------------------------------------------------------------
def _result(found, path, generated, expanded, succ, max_frontier):
    cost = None
    if found:
        cost = sum(edge_cost(path[i], path[i + 1]) for i in range(len(path) - 1))
    return {
        "found": found,
        "path": path,
        "cost": cost,
        "depth": (len(path) - 1) if found else None,
        "generated": generated,
        "expanded": expanded,
        "succ": succ,
        "max_frontier": max_frontier,
    }


def _build_path(parent, goal):
    path = []
    while goal != -1:
        path.append(goal)
        goal = parent[goal]
    path.reverse()
    return path


# ----------------------------------------------------------------------------
# Strategy A : Breadth-First Search (iterative, FIFO queue)
# ----------------------------------------------------------------------------
def bfs(adj, start, goal):
    if start == goal:
        return _result(True, [start], 1, 0, 0, 1)
    parent = [-2] * len(adj)                  # -2 = never generated (duplicate-detection structure)
    parent[start] = -1
    frontier = deque([start])
    generated, expanded, succ, max_frontier = 1, 0, 0, 1
    while frontier:
        u = frontier.popleft()
        expanded += 1                          # basic operation: node expansion
        for v in adj[u]:
            succ += 1                          # successor generation
            if parent[v] == -2:
                parent[v] = u
                generated += 1
                if v == goal:                  # goal test at generation time
                    return _result(True, _build_path(parent, v), generated, expanded, succ, max_frontier)
                frontier.append(v)
                if len(frontier) > max_frontier:
                    max_frontier = len(frontier)
    return _result(False, None, generated, expanded, succ, max_frontier)


# ----------------------------------------------------------------------------
# Strategy B : Uniform-Cost Search (iterative, priority queue)
# ----------------------------------------------------------------------------
def ucs(adj, start, goal):
    dist = [INF] * len(adj)
    parent = [-2] * len(adj)
    dist[start] = 0
    parent[start] = -1
    heap = [(0, start)]
    generated, expanded, succ, max_frontier = 1, 0, 0, 1
    while heap:
        g, u = heapq.heappop(heap)
        if g > dist[u]:                        # stale entry (a cheaper path was found later)
            continue
        if u == goal:                          # goal test at expansion -> optimal cost
            return _result(True, _build_path(parent, u), generated, expanded, succ, max_frontier)
        expanded += 1                          # basic operation: expansion = heappop + relaxations
        for v in adj[u]:
            succ += 1
            ng = g + edge_cost(u, v)
            if ng < dist[v]:
                dist[v] = ng
                parent[v] = u
                heapq.heappush(heap, (ng, v))
                generated += 1
                if len(heap) > max_frontier:
                    max_frontier = len(heap)
    return _result(False, None, generated, expanded, succ, max_frontier)


# ----------------------------------------------------------------------------
# Strategy C : Iterative-Deepening DFS (recursive depth-limited search)
#   graph-search variant: a per-iteration map {node: shallowest depth seen}
#   stops repeated states (a node is re-entered only via a strictly shallower path).
# ----------------------------------------------------------------------------
def iddfs(adj, start, goal, max_limit=10 ** 6):
    stats = {"generated": 1, "expanded": 0, "succ": 0, "max_stack": 0, "max_seen": 0}
    path = [start]

    def dls(u, depth, limit, seen):
        stats["expanded"] += 1                 # basic operation: one recursive call = one node expansion
        if len(path) > stats["max_stack"]:
            stats["max_stack"] = len(path)
        if u == goal:
            return True
        if depth == limit:                     # base case (cut-off)
            return False
        for v in adj[u]:
            stats["succ"] += 1
            if seen.get(v, INF) > depth + 1:   # not seen, or reached by a shallower path now
                if v not in seen:
                    stats["generated"] += 1
                seen[v] = depth + 1
                path.append(v)
                if dls(v, depth + 1, limit, seen):
                    return True
                path.pop()
        return False

    iterations = 0
    for limit in range(0, max_limit + 1):
        iterations += 1
        seen = {start: 0}
        path[:] = [start]
        if dls(start, 0, limit, seen):
            r = _result(True, list(path), stats["generated"], stats["expanded"], stats["succ"],
                        stats["max_stack"])
            r["iterations"] = iterations
            r["max_seen"] = max(stats["max_seen"], len(seen))
            return r
        stats["max_seen"] = max(stats["max_seen"], len(seen))
        if len(seen) == stats.get("last_seen", -1) and limit > 0:   # nothing new at a deeper limit
            break                                                   # -> whole component explored
        stats["last_seen"] = len(seen)
    r = _result(False, None, stats["generated"], stats["expanded"], stats["succ"], stats["max_stack"])
    r["iterations"] = iterations
    r["max_seen"] = stats["max_seen"]
    return r


# ----------------------------------------------------------------------------
# Strategy C (tree-search variant) - only the current path is checked for loops.
# Memory O(b*d) but time can be exponential on graphs with cycles. Used to
# validate the recurrence T(l) = b*T(l-1) + c and to show why a visited map is needed.
# ----------------------------------------------------------------------------
class BudgetExceeded(Exception):
    pass


def iddfs_tree(adj, start, goal, budget=3_000_000, max_limit=60):
    stats = {"calls": 0}
    on_path = set()

    def dls(u, limit):
        stats["calls"] += 1
        if stats["calls"] > budget:
            raise BudgetExceeded
        if u == goal:
            return [u]
        if limit == 0:
            return None
        on_path.add(u)
        for v in adj[u]:
            if v not in on_path:
                sub = dls(v, limit - 1)
                if sub:
                    on_path.discard(u)
                    return [u] + sub
        on_path.discard(u)
        return None

    for limit in range(max_limit + 1):
        on_path.clear()
        try:
            res = dls(start, limit)
        except BudgetExceeded:
            return {"found": None, "calls": stats["calls"], "limit": limit}
        if res:
            return {"found": True, "path": res, "calls": stats["calls"], "limit": limit}
    return {"found": False, "calls": stats["calls"], "limit": max_limit}


def dls_count_calls(children, root_depth_limit, b):
    """Plain depth-limited recursion on a complete b-ary tree (no goal) that counts
    calls; used to verify T(l) = b T(l-1) + 1 exactly."""
    cnt = [0]

    def rec(l):
        cnt[0] += 1
        if l == 0:
            return
        for _ in range(b):
            rec(l - 1)

    rec(root_depth_limit)
    return cnt[0]


# ----------------------------------------------------------------------------
# Validator
# ----------------------------------------------------------------------------
def check_path(adj, start, goal, res):
    """True if the returned route is a legal walk start->goal over existing corridors
    and the reported cost equals the recomputed cost."""
    if not res["found"]:
        return True
    p = res["path"]
    if p[0] != start or p[-1] != goal:
        return False
    for i in range(len(p) - 1):
        if p[i + 1] not in adj[p[i]]:
            return False
    return res["cost"] == sum(edge_cost(p[i], p[i + 1]) for i in range(len(p) - 1))


ALGOS = {"BFS": bfs, "UCS": ucs, "IDDFS": iddfs}


def timed(fn, adj, s, g, runs=5):
    """Run `runs` times, return (result of first run, mean seconds)."""
    first, total = None, 0.0
    for i in range(runs):
        t0 = time.perf_counter()
        r = fn(adj, s, g)
        total += time.perf_counter() - t0
        if first is None:
            first = r
    return first, total / runs


def bfs_distances(adj, s):
    """Helper for test design only (not part of the timed algorithms)."""
    d = [-1] * len(adj)
    d[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if d[v] < 0:
                d[v] = d[u] + 1
                q.append(v)
    return d
