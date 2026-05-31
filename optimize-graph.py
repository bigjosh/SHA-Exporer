"""optimize-graph.py — minimize a NAND/C/O circuit via an And-Inverter Graph.

The on-disk format is always NAND/C/O (see plan.md), but we optimize on an
internal AIG (AND nodes + complemented edges), which makes NOT free and lets
structural hashing dedup a node and its complement together.

Baseline passes implemented here (every step is a sound boolean identity, so
the result is provably equivalent to the input):

  1. parse NAND/C/O -> AIG.  A NAND node becomes an AND node whose value is
     consumed complemented; constants map to the const literal; free inputs
     (undefined references, e.g. MESSAGE-*) become primary inputs.
  2. strash (structural hashing): hash-cons AND nodes by canonical fanin pair,
     folding trivial cases  AND(x,0)=0, AND(x,1)=x, AND(x,x)=x, AND(x,~x)=0.
     This performs constant folding (including over unknown X inputs) and common
     -subexpression elimination during construction.
  3. dead-logic elimination: only AIG nodes reachable from outputs are emitted.
  4. lowering AIG -> NAND/C/O: each AND node materializes as a NAND (its
     complemented value); an inverter (NAND(x,x)) is added only where a node's
     positive value or a negated input/PI is actually consumed.

Interface IDs are preserved exactly: free inputs keep their names (MESSAGE-*)
and outputs keep theirs (HASH-*). Internal node IDs are freshly generated and
deterministic (cosmetic renaming is a separate future tool).

Usage:
    python optimize-graph.py <input.nodes> <output.nodes>
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

import numpy as np

import nodes
import sim

# Literal encoding: lit = (node_index << 1) | complement_bit.
# Node 0 is the constant node:  lit 0 = const 0,  lit 1 = const 1.
CONST0 = 0
CONST1 = 1

# Wall-clock cap on the whole FRAIG SAT-confirm phase (checked between miters) —
# a safety net for the easy-but-many case. Note: partial evaluation (--pin) skips
# FRAIG by default, because pinning can create a single hard miter and Cadical
# (fast, but pysat cannot bound it mid-solve) would grind on it indefinitely;
# strash already captures most of the pinned gain.
SAT_PHASE_TIME_S = 1800

# maj cut-rewrite: enumerate <=3-input cuts and rewrite nodes whose exact function
# is majority into the optimal OR/MUX form (FIPS keeps Maj as XOR-of-ANDs, which
# strash cannot restructure). Sound: only exact-TT majority nodes are rewritten.
CUT_CAP = 64         # max (non-dominated) cuts kept per node during enumeration


def _maj_class() -> dict[int, tuple]:
    """Map every truth table in the 3-input majority NPN class to its phasing:
    tt -> (in0_neg, in1_neg, in2_neg, out_neg). Covers all input/output negations,
    since the round Maj's inputs are often complemented literals."""
    proj = (0xAA, 0xCC, 0xF0)
    cls: dict[int, list[tuple]] = {}
    for s0 in (0, 1):
        for s1 in (0, 1):
            for s2 in (0, 1):
                a = proj[0] ^ (0xFF if s0 else 0)
                b = proj[1] ^ (0xFF if s1 else 0)
                c = proj[2] ^ (0xFF if s2 else 0)
                t = ((a & b) | (a & c) | (b & c)) & 0xFF
                cls.setdefault(t, []).append((s0, s1, s2, 0))
                cls.setdefault(t ^ 0xFF, []).append((s0, s1, s2, 1))
    return cls  # each tt maps to BOTH self-dual phasings; pick the realizable one


MAJ_CLASS = _maj_class()


class Aig:
    """And-Inverter Graph with on-the-fly structural hashing."""

    def __init__(self) -> None:
        # Index 0 is the constant node. Parallel arrays indexed by node id.
        self.fanin0: list[int] = [0]
        self.fanin1: list[int] = [0]
        self.is_pi: list[bool] = [False]
        self.pi_name: list[str | None] = [None]
        self.strash: dict[tuple[int, int], int] = {}
        self.num_and = 0

    def _new_node(self) -> int:
        idx = len(self.fanin0)
        self.fanin0.append(0)
        self.fanin1.append(0)
        self.is_pi.append(False)
        self.pi_name.append(None)
        return idx

    def pi(self, name: str) -> int:
        idx = self._new_node()
        self.is_pi[idx] = True
        self.pi_name[idx] = name
        return idx << 1  # positive literal

    def AND(self, a: int, b: int) -> int:
        # Trivial-case folding (sound boolean identities).
        if a == CONST0 or b == CONST0:
            return CONST0
        if a == CONST1:
            return b
        if b == CONST1:
            return a
        if a == b:
            return a
        if a == (b ^ 1):
            return CONST0
        if a > b:
            a, b = b, a
        key = (a, b)
        idx = self.strash.get(key)
        if idx is not None:
            return idx << 1
        idx = self._new_node()
        self.fanin0[idx] = a
        self.fanin1[idx] = b
        self.strash[key] = idx
        self.num_and += 1
        return idx << 1

    def NAND(self, a: int, b: int) -> int:
        return self.AND(a, b) ^ 1


def build_aig(g: nodes.Graph) -> tuple[Aig, list[tuple[str, int]]]:
    """Translate a NAND/C/O graph into an AIG. Returns (aig, outputs) where
    outputs is a list of (output_id, literal)."""
    aig = Aig()
    lit: dict[str, int] = {}      # base node id -> literal
    pi_lit: dict[str, int] = {}   # free input name -> literal

    def resolve(ref: str) -> int:
        l = lit.get(ref)
        if l is not None:
            return l
        l = pi_lit.get(ref)
        if l is None:
            l = aig.pi(ref)
            pi_lit[ref] = l
        return l

    outputs: list[tuple[str, int]] = []
    for nid in g.topo_order():
        n = g.nodes[nid]
        if n.type == "C":
            lit[nid] = CONST1 if n.value == 1 else CONST0
        elif n.type == "N":
            a, b = n.inputs
            lit[nid] = aig.NAND(resolve(a), resolve(b))
        else:  # O — value is a copy of its input
            l = resolve(n.input)
            lit[nid] = l
            outputs.append((nid, l))
    return aig, outputs


def reachable_ands(aig: Aig, outputs: list[tuple[str, int]]) -> set[int]:
    """Indices of AND nodes reachable from the output literals."""
    reach: set[int] = set()
    stack: list[int] = []
    for _, l in outputs:
        idx = l >> 1
        if idx != 0 and not aig.is_pi[idx]:
            stack.append(idx)
    while stack:
        idx = stack.pop()
        if idx in reach:
            continue
        reach.add(idx)
        for fl in (aig.fanin0[idx], aig.fanin1[idx]):
            fi = fl >> 1
            if fi != 0 and not aig.is_pi[fi] and fi not in reach:
                stack.append(fi)
    return reach


def aig_simulate(aig: Aig, n_words: int, seed: int) -> np.ndarray:
    """Bit-parallel sim of every AIG node: returns sig[node] = (n_words,) uint64.
    PIs get random words; node 0 (const 0) stays all-zero."""
    n = len(aig.fanin0)
    sig = np.zeros((n, n_words), dtype=np.uint64)
    rng = np.random.default_rng(seed)
    f0, f1, ispi = aig.fanin0, aig.fanin1, aig.is_pi
    for i in range(n):
        if ispi[i]:
            sig[i] = np.frombuffer(rng.bytes(n_words * 8), dtype=np.uint64)
    for i in range(1, n):
        if ispi[i]:
            continue
        a, b = f0[i], f1[i]
        sa = sig[a >> 1]
        if a & 1:
            sa = np.invert(sa)
        sb = sig[b >> 1]
        if b & 1:
            sb = np.invert(sb)
        sig[i] = sa & sb
    return sig


def _encode_cnf(aig: Aig, reach: set[int]):
    """Tseitin-encode the reachable AND cone into a fresh incremental SAT solver.
    SAT var (i+1) holds the boolean value of AIG node i; const node 0 forced false.
    Returns the solver (caller must .delete() it)."""
    from pysat.solvers import Cadical153
    s = Cadical153()  # fast incremental solving on the large shared CNF

    def satlit(l: int) -> int:
        node = l >> 1
        return (node + 1) if (l & 1) == 0 else -(node + 1)

    s.add_clause([-1])  # node 0 (const 0) is false
    for i in sorted(reach):
        a, b = aig.fanin0[i], aig.fanin1[i]
        la, lb, vi = satlit(a), satlit(b), i + 1
        s.add_clause([-vi, la])
        s.add_clause([-vi, lb])
        s.add_clause([-la, -lb, vi])
    return s


def fraig(aig: Aig, outputs: list[tuple[str, int]], n_words: int = 16,
          seed_a: int = 12345, seed_b: int = 67890, use_sat: bool = True,
          verbose: bool = False
          ) -> tuple[Aig, list[tuple[str, int]], int]:
    """One round of functional merging (FRAIG).

    Simulation PROPOSES merges (nodes with identical canonical signatures over an
    independent two-seed check); SAT then CONFIRMS each one soundly (a node-pair
    miter: equivalent iff both differing assignments are UNSAT). Simulation alone
    is unsound — two nodes can agree on thousands of vectors yet differ on rare
    inputs — so nothing is merged without a SAT proof. Confirmed nodes are merged
    into the lowest-index representative (possibly the constant or a PI) by
    rebuilding through strash, which re-canonicalizes and cascades (e.g. XOR with
    a now-constant input collapses). PIs are always preserved (interface).
    Returns (new_aig, new_outputs, merged)."""
    reach = reachable_ands(aig, outputs)
    n = len(aig.fanin0)
    one = np.uint64(1)

    sig = aig_simulate(aig, n_words, seed_a)
    cand = [0]
    for i in range(1, n):
        if aig.is_pi[i] or i in reach:
            cand.append(i)
    buckets: dict[bytes, list[int]] = defaultdict(list)
    for i in cand:  # increasing index -> member lists stay sorted
        s = sig[i]
        canon = np.invert(s) if (s[0] & one) else s
        buckets[canon.tobytes()].append(i)

    def pol(s: np.ndarray) -> bool:
        return bool(s[0] & one)

    sig2 = aig_simulate(aig, n_words, seed_b)
    proposed: list[tuple[int, int, bool]] = []  # (rep, member, rel polarity)
    for members in buckets.values():
        if len(members) < 2:
            continue
        rep = members[0]
        prep1 = pol(sig[rep])
        srep2 = sig2[rep]
        nsrep2 = np.invert(srep2)
        for m in members[1:]:
            if aig.is_pi[m]:
                continue  # never merge away a free input
            rel = pol(sig[m]) ^ prep1            # m == rep ^ rel  (per seed_a)
            target = nsrep2 if rel else srep2
            if np.array_equal(sig2[m], target):  # survive the independent set
                proposed.append((rep, m, rel))
    if not proposed:
        return aig, outputs, 0

    if use_sat:
        solver = _encode_cnf(aig, reach)
        if verbose:
            print(f"  fraig: SAT-confirming {len(proposed)} candidates...", flush=True)

        def equiv(rep: int, m: int, rel: bool) -> bool:
            # Both nodes must be constrained in the CNF (const0, a PI, or a
            # reachable AND); an unconstrained dead node would make UNSAT unsound.
            assert rep == 0 or aig.is_pi[rep] or rep in reach
            assert m in reach
            vr, vm = rep + 1, m + 1
            pairs = [(vr, -vm), (-vr, vm)] if not rel else [(vr, vm), (-vr, -vm)]
            for assum in pairs:
                if solver.solve(assumptions=list(assum)):  # SAT -> can differ
                    return False
            return True  # both halves UNSAT -> proven equivalent

        merge = {}
        t0 = last = time.time()
        for k, (rep, m, rel) in enumerate(proposed):
            if time.time() - t0 > SAT_PHASE_TIME_S:
                if verbose:
                    print(f"  fraig: SAT phase time cap ({SAT_PHASE_TIME_S}s) hit at "
                          f"{k}/{len(proposed)}; proceeding with {len(merge)} confirmed",
                          flush=True)
                break
            if equiv(rep, m, rel):
                merge[m] = (rep, rel)
            now = time.time()
            if verbose and now - last >= 60:
                print(f"    SAT progress: {k + 1}/{len(proposed)} checked, "
                      f"{len(merge)} confirmed, {now - t0:.0f}s elapsed", flush=True)
                last = now
        solver.delete()
    else:
        merge = {m: (rep, rel) for (rep, m, rel) in proposed}
    if not merge:
        return aig, outputs, 0

    new = Aig()
    new_lit = [0] * n  # new literal for each old node index
    for i in range(1, n):
        if aig.is_pi[i]:
            new_lit[i] = new.pi(aig.pi_name[i])
        elif i in merge:
            rep, rel = merge[i]
            new_lit[i] = new_lit[rep] ^ (1 if rel else 0)
        elif i in reach:
            a, b = aig.fanin0[i], aig.fanin1[i]
            new_lit[i] = new.AND(new_lit[a >> 1] ^ (a & 1),
                                 new_lit[b >> 1] ^ (b & 1))
        # dead nodes keep new_lit 0 (never referenced)
    new_outputs = [(oid, new_lit[l >> 1] ^ (l & 1)) for oid, l in outputs]
    return new, new_outputs, len(merge)


def _build_maj(aig: Aig, x: int, y: int, z: int) -> int:
    """Build majority(x, y, z) = (x&y)|(x&z)|(y&z) over literals, via strash."""
    xy = aig.AND(x, y)
    xz = aig.AND(x, z)
    yz = aig.AND(y, z)
    or1 = aig.AND(xy ^ 1, xz ^ 1) ^ 1          # OR(xy, xz)
    return aig.AND(or1 ^ 1, yz ^ 1) ^ 1        # OR(or1, yz)


def _lit_and_exists(aig: Aig, la: int, lb: int) -> bool:
    """True if AND(la, lb) needs no new node: either it's a trivial fold, or the
    strash node already exists."""
    if la == lb or la == (lb ^ 1) or la < 2 or lb < 2:
        return True  # AND(x,x)=x, AND(x,~x)=0, or a constant -> no new node
    a, b = (la, lb) if la < lb else (lb, la)
    return (a, b) in aig.strash


def _expand_tt(tt: int, positions: list[int], k: int) -> int:
    """Expand a truth table over len(positions) vars to k vars; source variable i
    maps to bit positions[i] of the k-variable minterm index."""
    out = 0
    nsrc = len(positions)
    for m in range(1 << k):
        sm = 0
        for si in range(nsrc):
            if (m >> positions[si]) & 1:
                sm |= 1 << si
        if (tt >> sm) & 1:
            out |= 1 << m
    return out


def _find_maj_nodes(aig: Aig, reach: set[int]) -> dict[int, tuple]:
    """Enumerate <=3-input cuts with exact truth tables; return {idx: (leaves, phase)}
    for AND nodes whose function over a 3-leaf cut is majority (phase 0) or its
    complement (phase 1). Sound: detection is by exact truth table."""
    cuts: dict[int, list[tuple[tuple, int]]] = {0: [((0,), 0)]}
    marks: dict[int, tuple] = {}
    f0a, f1a, ispi = aig.fanin0, aig.fanin1, aig.is_pi
    for idx in range(1, len(f0a)):
        if ispi[idx]:
            cuts[idx] = [((idx,), 2)]
            continue
        if idx not in reach:
            continue
        a, b = f0a[idx], f1a[idx]
        n0, c0 = a >> 1, a & 1
        n1, c1 = b >> 1, b & 1
        seen = {(idx,): 2}  # trivial (identity) cut
        for (l0, t0) in cuts[n0]:
            for (l1, t1) in cuts[n1]:
                u = tuple(sorted(set(l0) | set(l1)))
                if len(u) > 3 or u in seen:
                    continue
                k = len(u)
                mask = (1 << (1 << k)) - 1
                e0 = _expand_tt(t0, [u.index(x) for x in l0], k)
                e1 = _expand_tt(t1, [u.index(x) for x in l1], k)
                if c0:
                    e0 ^= mask
                if c1:
                    e1 ^= mask
                seen[u] = e0 & e1
        matched = False
        for u, tt in seen.items():  # detect over ALL candidate cuts
            if len(u) != 3 or 0 in u or tt not in MAJ_CLASS:
                continue
            g0, g1, g2 = u
            for (s0, s1, s2, so) in MAJ_CLASS[tt]:  # try both self-dual phasings
                l0, l1, l2 = (g0 << 1) ^ s0, (g1 << 1) ^ s1, (g2 << 1) ^ s2
                # Only rewrite when all three pairwise products already exist (the
                # generator's XOR-of-ANDs Maj). Adder carries are maj too but are
                # built as OR(a&b, c&(a^b)) -> a&c, b&c absent -> OR-form would ADD
                # nodes, so skip them.
                if (_lit_and_exists(aig, l0, l1) and _lit_and_exists(aig, l0, l2)
                        and _lit_and_exists(aig, l1, l2)):
                    marks[idx] = (l0, l1, l2, so)
                    matched = True
                    break
            if matched:
                break
        # keep only non-dominated cuts (drop any cut that has a proper subset cut);
        # the majority's true 3-leaf cut is never dominated, so it survives.
        sets = {u: frozenset(u) for u in seen}
        keep = [(u, tt) for u, tt in seen.items()
                if not any(sets[v] < sets[u] for v in seen if v != u)]
        cuts[idx] = keep[:CUT_CAP]
    return marks


def maj_rewrite(aig: Aig, outputs: list[tuple[str, int]], verbose: bool = False
                ) -> tuple[Aig, list[tuple[str, int]], int]:
    """Rewrite majority-class nodes into the optimal OR/MUX form. Sound (exact-TT
    detection + functionally-identical replacement); accepted only if it reduces
    the reachable AND count."""
    reach = reachable_ands(aig, outputs)
    marks = _find_maj_nodes(aig, reach)
    if not marks:
        return aig, outputs, 0
    new = Aig()
    n = len(aig.fanin0)
    new_lit = [0] * n
    for i in range(1, n):
        if aig.is_pi[i]:
            new_lit[i] = new.pi(aig.pi_name[i])
        elif i in marks:
            l0, l1, l2, so = marks[i]
            x = new_lit[l0 >> 1] ^ (l0 & 1)
            y = new_lit[l1 >> 1] ^ (l1 & 1)
            z = new_lit[l2 >> 1] ^ (l2 & 1)
            new_lit[i] = _build_maj(new, x, y, z) ^ so
        elif i in reach:
            a, b = aig.fanin0[i], aig.fanin1[i]
            new_lit[i] = new.AND(new_lit[a >> 1] ^ (a & 1), new_lit[b >> 1] ^ (b & 1))
    new_outputs = [(oid, new_lit[l >> 1] ^ (l & 1)) for oid, l in outputs]
    old_n, new_n = len(reach), len(reachable_ands(new, new_outputs))
    if new_n >= old_n:  # never regress
        return aig, outputs, 0
    if verbose:
        print(f"  maj-rewrite: {len(marks)} majority nodes; "
              f"{old_n} -> {new_n} AND nodes", flush=True)
    return new, new_outputs, len(marks)


def lower(aig: Aig, outputs: list[tuple[str, int]]) -> nodes.Graph:
    """Lower the AIG back to a NAND/C/O graph, emitting only nodes reachable
    from the outputs."""
    out = nodes.Graph()
    reachable = reachable_ands(aig, outputs)

    const_id: dict[int, str] = {}
    and_neg: dict[int, str] = {}  # idx -> id computing ~AND (the NAND output)
    and_pos: dict[int, str] = {}  # idx -> id computing AND (= NOT of and_neg)
    pi_neg: dict[str, str] = {}   # pi name -> id computing its negation
    counter = [0]

    def fresh() -> str:
        counter[0] += 1
        return f"g{counter[0]}"

    def get_const(comp: int) -> str:
        cid = const_id.get(comp)
        if cid is None:
            cid = "CONST-1" if comp == 1 else "CONST-0"
            out.add(nodes.Node("C", cid, value=comp))
            const_id[comp] = cid
        return cid

    def litval(l: int) -> str:
        """The NAND-graph node id whose value equals literal l."""
        idx = l >> 1
        comp = l & 1
        if idx == 0:
            return get_const(comp)
        if aig.is_pi[idx]:
            name = aig.pi_name[idx]
            if comp == 0:
                return name
            nid = pi_neg.get(name)
            if nid is None:
                nid = fresh()
                out.add(nodes.Node("N", nid, inputs=(name, name)))
                pi_neg[name] = nid
            return nid
        # AND node: and_neg[idx] computes ~AND; the positive form needs an inverter.
        if comp == 1:
            return and_neg[idx]
        nid = and_pos.get(idx)
        if nid is None:
            nid = fresh()
            out.add(nodes.Node("N", nid, inputs=(and_neg[idx], and_neg[idx])))
            and_pos[idx] = nid
        return nid

    # Emit AND nodes in topological (increasing index) order; fanins precede.
    for idx in sorted(reachable):
        a_id = litval(aig.fanin0[idx])
        b_id = litval(aig.fanin1[idx])
        nid = fresh()
        out.add(nodes.Node("N", nid, inputs=(a_id, b_id)))
        and_neg[idx] = nid

    for oid, l in outputs:
        out.add(nodes.Node("O", oid, input=litval(l)))
    return out


def optimize(g: nodes.Graph, fraig_rounds: int = 8, verbose: bool = True) -> nodes.Graph:
    aig, outputs = build_aig(g)
    if verbose:
        print(f"  strash: {aig.num_and} AND nodes "
              f"({len(reachable_ands(aig, outputs))} reachable)", flush=True)
    total_merged = 0
    for r in range(fraig_rounds):
        aig, outputs, merged = fraig(aig, outputs, verbose=verbose)
        total_merged += merged
        if verbose and merged:
            print(f"  fraig round {r + 1}: merged {merged} "
                  f"({len(reachable_ands(aig, outputs))} reachable ANDs)", flush=True)
        if merged == 0:
            break
    if verbose:
        print(f"  fraig total merged: {total_merged}", flush=True)
    # majority cut-rewrite (sound, SAT-free; safe with or without pins)
    aig, outputs, n_maj = maj_rewrite(aig, outputs, verbose=verbose)
    return lower(aig, outputs)


def merge_pins(g: nodes.Graph, pin_paths: list[str]) -> int:
    """Add constant definitions from pin files into g (partial evaluation).

    A pin file is a .nodes file of `C,<id>,<0|1>` lines that pin otherwise-free
    inputs (e.g. MESSAGE-*). They are added as constant nodes; strash then folds
    the now-constant cones during optimize(). g.add raises if a pin collides with
    an already-defined node (intentional). Returns the number of pins added.
    """
    pinned: list[str] = []
    for path in pin_paths:
        pg = nodes.parse(path)
        for node in pg.nodes.values():
            if node.type != "C":
                raise ValueError(f"pin file {path} may only contain C nodes; "
                                 f"got {node.type} for {node.id}")
            g.add(node)
            pinned.append(node.id)
    referenced = g.referenced_ids()
    unused = [pid for pid in pinned if pid not in referenced]
    if unused:
        print(f"  warning: {len(unused)} pinned id(s) are not referenced by any "
              f"node (typo?): {unused[:5]}{' ...' if len(unused) > 5 else ''}")
    return len(pinned)


def _stats(g: nodes.Graph) -> str:
    from collections import Counter
    c = Counter(n.type for n in g.nodes.values())
    return (f"{len(g.nodes)} nodes "
            f"(N={c.get('N', 0)}, C={c.get('C', 0)}, O={c.get('O', 0)})")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Minimize a NAND/C/O circuit via an AIG.")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--pin", action="append", default=[], metavar="FILE",
                   help="pin free inputs to constants (partial evaluation); repeatable")
    p.add_argument("--vectors", type=int, default=4096,
                   help="random vectors for the equivalence gate (default 4096)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force-fraig", action="store_true",
                   help="run FRAIG even with --pin (may hit a slow SAT miter)")
    args = p.parse_args(argv[1:])

    g = nodes.parse(args.input)
    if args.pin:
        n_pins = merge_pins(g, args.pin)
        print(f"pinned {n_pins} free inputs to constants")
    print(f"input:  {_stats(g)}")

    # Partial evaluation skips FRAIG by default: pinning can create a hard SAT
    # miter that Cadical (unbounded) would grind on. strash still captures most
    # of the pinned gain; --force-fraig opts back in.
    rounds = 0 if (args.pin and not args.force_fraig) else 8
    if args.pin and rounds == 0:
        print("  (partial eval: FRAIG skipped; use --force-fraig to enable)")
    g_out = optimize(g, fraig_rounds=rounds)
    print(f"output: {_stats(g_out)}")
    n_in = sum(1 for n in g.nodes.values() if n.type == "N")
    n_out = sum(1 for n in g_out.nodes.values() if n.type == "N")
    if n_in:
        print(f"NAND reduction: {n_in} -> {n_out} "
              f"({100 * (n_in - n_out) / n_in:.1f}% fewer)")

    # Equivalence gate: the input graph (already specialized with any pins) must
    # match the output over many random vectors. sim.equivalence requires both
    # graphs to share the same free-input set, which holds because pins were
    # merged into g before optimizing.
    print(f"equivalence gate ({args.vectors} random vectors)...", end=" ")
    ok, msg = sim.equivalence(g, g_out, args.vectors, seed=args.seed)
    print("PASS" if ok else f"FAIL — {msg}")
    if not ok:
        return 1
    g_out.write(args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
