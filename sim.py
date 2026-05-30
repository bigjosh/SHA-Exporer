"""Bit-parallel simulation for NAND/C/O graphs (numpy uint64 lanes).

Packs many input vectors into uint64 numpy arrays (64 vectors per word) and
evaluates the whole graph with vectorized bitwise ops, so equivalence checking
and signature computation are far faster and more memory-compact than per-vector
scalar evaluation. 2-valued (concrete 0/1 inputs).
"""

from __future__ import annotations

import numpy as np

import nodes

_U64_MAX = np.uint64(np.iinfo(np.uint64).max)


def random_input_arrays(names, n_words: int, rng: np.random.Generator
                        ) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in names:
        out[name] = np.frombuffer(rng.bytes(n_words * 8), dtype=np.uint64).copy()
    return out


def simulate_outputs(g: nodes.Graph, input_arrays: dict[str, np.ndarray],
                     n_words: int) -> dict[str, np.ndarray]:
    """Simulate and return only the output-node arrays (internal arrays are
    dropped when this returns, keeping peak memory to one graph's worth)."""
    vals: dict[str, np.ndarray] = {}
    zero = np.zeros(n_words, dtype=np.uint64)
    ones = np.full(n_words, _U64_MAX, dtype=np.uint64)

    def get(ref: str) -> np.ndarray:
        v = vals.get(ref)
        if v is not None:
            return v
        return input_arrays.get(ref, zero)

    for nid in g.topo_order():
        n = g.nodes[nid]
        if n.type == "C":
            vals[nid] = ones if n.value == 1 else zero
        elif n.type == "N":
            vals[nid] = np.invert(get(n.inputs[0]) & get(n.inputs[1]))
        else:  # O
            vals[nid] = get(n.input)
    return {n.id: vals[n.id] for n in g.outputs()}


def equivalence(g_a: nodes.Graph, g_b: nodes.Graph, n_vectors: int = 4096,
                seed: int = 0) -> tuple[bool, str]:
    """Random-vector equivalence over shared free inputs / outputs."""
    fa, fb = g_a.free_inputs(), g_b.free_inputs()
    oa = sorted(n.id for n in g_a.outputs())
    ob = sorted(n.id for n in g_b.outputs())
    if oa != ob:
        return False, "different outputs"

    # Compare over the UNION of free inputs: drive shared inputs identically and
    # let each graph ignore names it doesn't reference. This still catches real
    # inequivalence (if one graph actually depends on an input the other dropped,
    # varying it changes only one graph's outputs -> mismatch), but tolerates the
    # legitimate case where optimization/partial-eval renders an input irrelevant.
    note = "" if fa == fb else f" (free-input sets differ: |A|={len(fa)}, |B|={len(fb)})"
    n_words = (n_vectors + 63) // 64
    rng = np.random.default_rng(seed)
    iw = random_input_arrays(sorted(fa | fb), n_words, rng)
    out_a = simulate_outputs(g_a, iw, n_words)
    out_b = simulate_outputs(g_b, iw, n_words)
    for oid in oa:
        diff = out_a[oid] ^ out_b[oid]
        nz = np.nonzero(diff)[0]
        if nz.size:
            w = int(nz[0])
            bit = int(diff[w] & (~diff[w] + np.uint64(1))).bit_length() - 1
            return False, f"mismatch at output {oid}, vector {w * 64 + bit}"
    return True, f"equivalent over {n_words * 64} random vectors{note}"


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: python sim.py <graphA.nodes> <graphB.nodes> [n_vectors]")
        return 2
    n = int(argv[3]) if len(argv) > 3 else 4096
    ga = nodes.parse(argv[1])
    gb = nodes.parse(argv[2])
    ok, msg = equivalence(ga, gb, n)
    print(("EQUIVALENT: " if ok else "NOT EQUIVALENT: ") + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # self-test: bit-parallel sim must match the scalar evaluator
        sample = [
            "N,t,x,y", "N,andxy,t,t", "O,OUT,andxy",
            "N,nx,x,x", "N,ny,y,y", "N,orxy,nx,ny", "O,OR,orxy",
        ]
        g = nodes.parse_lines(sample)
        rng = np.random.default_rng(7)
        iw = random_input_arrays(sorted(g.free_inputs()), 4, rng)
        out = simulate_outputs(g, iw, 4)
        for wi in range(4):
            for bit in range(64):
                xv = int((int(iw["x"][wi]) >> bit) & 1)
                yv = int((int(iw["y"][wi]) >> bit) & 1)
                ev = nodes.evaluate(g, {"x": xv, "y": yv})
                assert int(out["OUT"][wi] >> np.uint64(bit)) & 1 == ev["OUT"] == (xv & yv)
                assert int(out["OR"][wi] >> np.uint64(bit)) & 1 == ev["OR"] == (xv | yv)
        print("sim.py self-test OK")
        sys.exit(0)
    sys.exit(main(sys.argv))
