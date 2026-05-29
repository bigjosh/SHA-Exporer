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
from collections import defaultdict

import numpy as np

import nodes
import sim

# Literal encoding: lit = (node_index << 1) | complement_bit.
# Node 0 is the constant node:  lit 0 = const 0,  lit 1 = const 1.
CONST0 = 0
CONST1 = 1


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
    s = Cadical153()

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
          seed_a: int = 12345, seed_b: int = 67890, use_sat: bool = True
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

        def equiv(rep: int, m: int, rel: bool) -> bool:
            vr, vm = rep + 1, m + 1
            if not rel:  # rep == m : both differing assignments must be UNSAT
                if solver.solve(assumptions=[vr, -vm]):
                    return False
                if solver.solve(assumptions=[-vr, vm]):
                    return False
            else:        # rep == ~m
                if solver.solve(assumptions=[vr, vm]):
                    return False
                if solver.solve(assumptions=[-vr, -vm]):
                    return False
            return True

        merge = {m: (rep, rel) for (rep, m, rel) in proposed if equiv(rep, m, rel)}
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
              f"({len(reachable_ands(aig, outputs))} reachable)")
    total_merged = 0
    for r in range(fraig_rounds):
        aig, outputs, merged = fraig(aig, outputs)
        total_merged += merged
        if verbose and merged:
            print(f"  fraig round {r + 1}: merged {merged} "
                  f"({len(reachable_ands(aig, outputs))} reachable ANDs)")
        if merged == 0:
            break
    if verbose:
        print(f"  fraig total merged: {total_merged}")
    return lower(aig, outputs)


def merge_pins(g: nodes.Graph, pin_paths: list[str]) -> int:
    """Add constant definitions from pin files into g (partial evaluation).

    A pin file is a .nodes file of `C,<id>,<0|1>` lines that pin otherwise-free
    inputs (e.g. MESSAGE-*). They are added as constant nodes; strash then folds
    the now-constant cones during optimize(). g.add raises if a pin collides with
    an already-defined node (intentional). Returns the number of pins added.
    """
    n = 0
    for path in pin_paths:
        pg = nodes.parse(path)
        for node in pg.nodes.values():
            if node.type != "C":
                raise ValueError(f"pin file {path} may only contain C nodes; "
                                 f"got {node.type} for {node.id}")
            g.add(node)
            n += 1
    return n


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
    args = p.parse_args(argv[1:])

    g = nodes.parse(args.input)
    if args.pin:
        n_pins = merge_pins(g, args.pin)
        print(f"pinned {n_pins} free inputs to constants")
    print(f"input:  {_stats(g)}")

    g_out = optimize(g)
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
