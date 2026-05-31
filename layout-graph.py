"""layout-graph.py — preprocess a .nodes circuit into a binary layout for the viewer.

The web viewer must pan/zoom a ~230k-node graph at 60fps, so we do all the heavy
lifting here, offline, and hand the browser something it can load and render
directly:

  * **Topological reindex.** Nodes are renumbered so that index order *is* a valid
    evaluation order (every fanin has a smaller index). In-browser simulation then
    becomes a single forward loop over typed arrays — no per-frame graph walk.
  * **Layered layout.** Y = logic depth (longest path from the inputs); free inputs
    sit in the top row (layer 0), all outputs in a single bottom row. X within each
    layer is ordered by an iterative **barycenter** sweep to cut wire crossings
    (the input/output rows stay pinned in bit order — they're the meaningful ends).
  * **Compact binary.** A small JSON header (offsets, bounds, metadata) followed by
    raw typed-array blocks: positions (f32), node type (u8), the two fanin indices
    (i32), and bit indices for inputs/outputs.

As a full-pipeline self-check it simulates SHA-256("abc") over the emitted arrays
and compares against hashlib — if the reindex/fanin/type mapping were wrong, this
fails loudly *before* the browser ever sees the file.

Usage:
    python layout-graph.py SHA256-opt.nodes SHA256-opt.glayout [--sweeps N] [--dx F] [--dy F]

Format (.glayout):
    bytes  0..3    magic  "GLAY"
    u32    4..7    version
    u32    8..11   header length (bytes of UTF-8 JSON that follow the 16-byte preamble)
    u32    12..15  reserved (0)
    JSON   16..    header: {nodeCount, inputCount, outputCount, layerCount, dx, dy,
                            bounds, dataStart, arrays:[{name,dtype,offset,count}]}
    arrays at  dataStart + offset   (offsets/element-aligned; see header)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time

import numpy as np

import nodes

# Node type codes (must match viewer.js).
T_INPUT, T_NAND, T_OUTPUT, T_CONST0, T_CONST1 = 0, 1, 2, 3, 4

_DTYPE_NAME = {
    np.dtype("float32"): "float32",
    np.dtype("int32"): "int32",
    np.dtype("uint8"): "uint8",
}


def _log(msg: str, t0: float) -> None:
    print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)


def _bit_of(nid: str) -> int:
    """MESSAGE-463 -> 463, HASH-007 -> 7."""
    return int(nid.rsplit("-", 1)[1])


def build_arrays(g: nodes.Graph, t0: float):
    """Reindex (free inputs first, then defined nodes in topo order) and build the
    flat node arrays. Returns (idx_of, type, fanin0, fanin1, bit, F, N)."""
    free = sorted(g.free_inputs())          # 'MESSAGE-000'.. -> already bit order
    defined = g.topo_order()                # dependencies before dependents
    F, D = len(free), len(defined)
    N = F + D
    _log(f"vertices: {F} free inputs + {D} defined = {N} total", t0)

    idx_of: dict[str, int] = {}
    for i, fid in enumerate(free):
        idx_of[fid] = i
    for j, nid in enumerate(defined):
        idx_of[nid] = F + j

    typ = np.zeros(N, np.uint8)
    fi0 = np.full(N, -1, np.int32)
    fi1 = np.full(N, -1, np.int32)
    bit = np.full(N, -1, np.int32)

    for fid in free:
        i = idx_of[fid]
        typ[i] = T_INPUT
        bit[i] = _bit_of(fid)

    for nid in defined:
        n = g.nodes[nid]
        i = idx_of[nid]
        if n.type == "N":
            typ[i] = T_NAND
            fi0[i] = idx_of[n.inputs[0]]
            fi1[i] = idx_of[n.inputs[1]]
        elif n.type == "O":
            typ[i] = T_OUTPUT
            fi0[i] = idx_of[n.input]
            bit[i] = _bit_of(nid)
        elif n.type == "C":
            typ[i] = T_CONST1 if n.value == 1 else T_CONST0
    return idx_of, typ, fi0, fi1, bit, F, N


def compute_layers(typ, fi0, fi1, F, N, t0):
    """Longest-path layer (depth) for every node; outputs forced to a bottom row."""
    f0 = fi0.tolist()                       # python ints: faster scalar access in loop
    f1 = fi1.tolist()
    lay = [0] * N
    for i in range(F, N):                   # F..N-1 is topo order of defined nodes
        a, b = f0[i], f1[i]
        la = lay[a] if a >= 0 else 0
        lb = lay[b] if b >= 0 else 0
        lay[i] = 1 + (la if la >= lb else lb)
    layer = np.array(lay, np.int32)

    out_mask = typ == T_OUTPUT
    non_out_max = int(layer[~out_mask].max())
    layer[out_mask] = non_out_max + 1       # all outputs in one bottom row
    L = non_out_max + 1
    _log(f"layers: {L + 1} (depth 0..{L}); outputs pinned to row {L}", t0)
    return layer, L


def barycenter_layout(layer, bit, fi0, fi1, L, N, sweeps, t0):
    """Order nodes within each layer to reduce crossings (barycenter sweeps).
    Layer 0 (inputs/consts) and layer L (outputs) stay pinned in bit order.
    Returns within-layer rank x[] (float)."""
    layer_count = np.bincount(layer, minlength=L + 1).astype(np.int64)
    layer_start = np.concatenate([[0], np.cumsum(layer_count)])  # start of layer l in sorted order

    # edge list (producer -> consumer) over real edges, for both sweep directions
    prod = []
    cons = []
    for fi in (fi0, fi1):
        m = fi >= 0
        prod.append(fi[m])
        cons.append(np.nonzero(m)[0].astype(np.int32))
    p = np.concatenate(prod)
    c = np.concatenate(cons)
    _log(f"edges: {len(p)}", t0)

    pin = (layer == 0) | (layer == L)
    # pinned nodes order by bit (inputs/outputs); consts (bit<0) trail by index
    pinned_key = np.where(bit >= 0, bit.astype(np.float64),
                          1_000_000.0 + np.arange(N))

    def rerank(key):
        order = np.lexsort((key, layer))               # primary: layer, secondary: key
        ranks = np.arange(N) - layer_start[layer[order]]
        x = np.empty(N, np.float64)
        x[order] = ranks
        return x

    # initial order: pinned by bit, others by topo index (deterministic)
    init_key = np.where(pin, pinned_key, np.arange(N).astype(np.float64))
    x = rerank(init_key)

    for s in range(sweeps):
        if s % 2 == 0:                                  # down: key = mean of fanins' x
            sumw = np.bincount(c, weights=x[p], minlength=N)
            cnt = np.bincount(c, minlength=N)
        else:                                           # up: key = mean of fanouts' x
            sumw = np.bincount(p, weights=x[c], minlength=N)
            cnt = np.bincount(p, minlength=N)
        bary = np.where(cnt > 0, sumw / np.maximum(cnt, 1), x)
        key = np.where(pin, pinned_key, bary)
        x = rerank(key)
        _log(f"  barycenter sweep {s + 1}/{sweeps} ({'down' if s % 2 == 0 else 'up'})", t0)

    return x, layer_count


def abc_self_check(typ, fi0, fi1, bit, F, N, in_by_bit, out_by_bit, t0):
    """Simulate SHA-256('abc') over the emitted arrays; compare to hashlib."""
    msg = b"abc"
    block = bytearray(64)
    block[: len(msg)] = msg
    block[len(msg)] = 0x80
    block[56:64] = (len(msg) * 8).to_bytes(8, "big")

    state = np.zeros(N, np.int8)
    for byte_i in range(64):
        for k in range(8):                              # k=0 -> MSB (matches verify-graph.py)
            b_idx = byte_i * 8 + k
            ni = in_by_bit[b_idx] if b_idx < len(in_by_bit) else -1
            if ni >= 0:
                state[ni] = (block[byte_i] >> (7 - k)) & 1

    st = state.tolist()
    t = typ.tolist()
    a = fi0.tolist()
    b = fi1.tolist()
    for i in range(F, N):
        ti = t[i]
        if ti == T_NAND:
            st[i] = 1 - (st[a[i]] & st[b[i]])
        elif ti == T_OUTPUT:
            st[i] = st[a[i]]
        elif ti == T_CONST1:
            st[i] = 1
        elif ti == T_CONST0:
            st[i] = 0

    value = 0
    for hb in range(256):
        ni = out_by_bit[hb] if hb < len(out_by_bit) else -1
        value = (value << 1) | ((st[ni] & 1) if ni >= 0 else 0)
    got = f"{value:064x}"
    exp = hashlib.sha256(msg).hexdigest()
    ok = got == exp
    _log(f"abc self-check: {'PASS' if ok else 'FAIL'}  got={got}", t0)
    if not ok:
        _log(f"               expected={exp}", t0)
    return ok


def pack(path, header_meta, arrays):
    """Write the .glayout binary: 16-byte preamble + JSON header + aligned arrays."""
    # lay out arrays 4-byte aligned; record offsets relative to dataStart
    blob = bytearray()
    arr_meta = []
    for name, arr in arrays:
        while len(blob) % 4 != 0:
            blob.append(0)
        offset = len(blob)
        arr_meta.append({
            "name": name,
            "dtype": _DTYPE_NAME[arr.dtype],
            "offset": offset,
            "count": int(arr.size),
        })
        blob += arr.tobytes()

    header = dict(header_meta)
    header["arrays"] = arr_meta
    header_json = json.dumps(header).encode("utf-8")

    preamble = bytearray()
    preamble += b"GLAY"
    preamble += struct.pack("<III", 1, len(header_json), 0)
    # dataStart = 16 + header, padded to 4
    head_end = 16 + len(header_json)
    pad = (-head_end) % 4
    data_start = head_end + pad

    with open(path, "wb") as fh:
        fh.write(preamble)
        fh.write(header_json)
        fh.write(b"\0" * pad)
        fh.write(blob)
    return data_start, len(blob)


def main(argv):
    ap = argparse.ArgumentParser(description="Preprocess a .nodes circuit into a .glayout binary for the viewer.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--sweeps", type=int, default=8, help="barycenter crossing-reduction sweeps (default 8)")
    ap.add_argument("--dx", type=float, default=1.0, help="horizontal spacing between gates in a layer")
    ap.add_argument("--dy", type=float, default=4.0, help="vertical spacing between layers")
    args = ap.parse_args(argv[1:])

    t0 = time.perf_counter()
    _log(f"parsing {args.input} ...", t0)
    g = nodes.parse(args.input)

    idx_of, typ, fi0, fi1, bit, F, N = build_arrays(g, t0)
    layer, L = compute_layers(typ, fi0, fi1, F, N, t0)
    x, layer_count = barycenter_layout(layer, bit, fi0, fi1, L, N, args.sweeps, t0)

    # within-layer rank -> centered world coordinates
    cnt_per = layer_count[layer].astype(np.float64)
    pos_x = ((x - (cnt_per - 1.0) / 2.0) * args.dx).astype(np.float32)
    pos_y = (layer.astype(np.float64) * args.dy).astype(np.float32)

    # bit -> node-index lookup tables for inputs and outputs
    in_mask = typ == T_INPUT
    out_mask = typ == T_OUTPUT
    n_in_bits = int(bit[in_mask].max()) + 1 if in_mask.any() else 0
    n_out_bits = int(bit[out_mask].max()) + 1 if out_mask.any() else 0
    in_by_bit = np.full(n_in_bits, -1, np.int32)
    out_by_bit = np.full(n_out_bits, -1, np.int32)
    in_by_bit[bit[in_mask]] = np.nonzero(in_mask)[0].astype(np.int32)
    out_by_bit[bit[out_mask]] = np.nonzero(out_mask)[0].astype(np.int32)

    # inputs/outputs must be a bijection onto dense bit indices 0..n-1 (no gaps, no
    # duplicates); otherwise the viewer would index a -1 slot and read a wrong digest.
    if ((in_by_bit < 0).any() or int(in_mask.sum()) != n_in_bits or
            (out_by_bit < 0).any() or int(out_mask.sum()) != n_out_bits):
        _log("ABORT: MESSAGE-/HASH- bit indices are not dense & unique; not writing.", t0)
        return 1

    ok = abc_self_check(typ, fi0, fi1, bit, F, N, in_by_bit, out_by_bit, t0)
    if not ok:
        _log("ABORT: self-check failed; not writing output.", t0)
        return 1

    header_meta = {
        "version": 1,
        "nodeCount": int(N),
        "inputCount": int(in_mask.sum()),
        "outputCount": int(out_mask.sum()),
        "layerCount": int(L + 1),
        "dx": args.dx,
        "dy": args.dy,
        "bounds": {
            "minX": float(pos_x.min()), "maxX": float(pos_x.max()),
            "minY": float(pos_y.min()), "maxY": float(pos_y.max()),
        },
    }
    arrays = [
        ("posX", pos_x), ("posY", pos_y),
        ("type", typ), ("fanin0", fi0), ("fanin1", fi1), ("bit", bit),
        ("inputIdxByBit", in_by_bit), ("outputIdxByBit", out_by_bit),
    ]
    data_start, blob_len = pack(args.output, header_meta, arrays)
    # patch dataStart into header is not needed by JS if it recomputes; but include it
    # (the viewer recomputes dataStart from header length, so this is informational)
    total = 16 + blob_len  # approx; real file includes header json
    import os
    sz = os.path.getsize(args.output)
    _log(f"wrote {args.output}: {sz / 1e6:.2f} MB  "
         f"(N={N}, layers={L + 1}, maxLayerWidth={int(layer_count.max())})", t0)
    _log(f"bounds x[{header_meta['bounds']['minX']:.0f},{header_meta['bounds']['maxX']:.0f}] "
         f"y[{header_meta['bounds']['minY']:.0f},{header_meta['bounds']['maxY']:.0f}]", t0)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
