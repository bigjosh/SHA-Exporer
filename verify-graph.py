"""verify-graph.py — CLI verifier for SHA256 .nodes graphs.

Modes:
  evaluate <graph.nodes> <message>
      Build the padded 64-byte single block for `message` (<=55 bytes),
      set MESSAGE-* free inputs per the big-endian MSB-first mapping, evaluate,
      read HASH-* bits into a 64-char hex digest, and compare to
      hashlib.sha256(message).hexdigest().

  equivalence <graphA.nodes> <graphB.nodes> [K]
      Both graphs must share identical free inputs and outputs. Run K (default
      256) random input assignments through both and assert identical outputs.

  selftest
      Smoke-test equivalence: a graph equals itself; a mutated copy does not.
"""

from __future__ import annotations

import hashlib
import random
import sys

import nodes


# --------------------------------------------------------------------------
# Padding helper (single block, message <= 55 bytes)
# --------------------------------------------------------------------------
def pad_block(msg: bytes) -> bytes:
    """msg ‖ 0x80 ‖ 0x00... ‖ uint64_be(bitlen) -> exactly 64 bytes."""
    if len(msg) > 55:
        raise ValueError("message must be <= 55 bytes for a single block")
    bitlen = len(msg) * 8
    block = bytearray(64)
    block[: len(msg)] = msg
    block[len(msg)] = 0x80
    block[56:64] = bitlen.to_bytes(8, "big")
    return bytes(block)


def message_bits(block: bytes) -> dict[str, int]:
    """Map the 64-byte block to MESSAGE-000..511 (big-endian, MSB-first).

    MESSAGE-000 = bit 7 (MSB) of byte 0; MESSAGE-007 = bit 0 of byte 0;
    MESSAGE-008 = MSB of byte 1; ...
    """
    assert len(block) == 64
    bits: dict[str, int] = {}
    for byte_i in range(64):
        byte = block[byte_i]
        for k in range(8):  # k = 0 -> MSB (bit 7), k = 7 -> LSB (bit 0)
            idx = byte_i * 8 + k
            bit = (byte >> (7 - k)) & 1
            bits[f"MESSAGE-{idx:03d}"] = bit
    return bits


def digest_from_values(vals: dict[str, int | None]) -> str:
    """Read HASH-000..255 from evaluated values into a 64-char hex digest.

    HASH-000 = MSB of H0 (high bit of first hex nibble); HASH-031 = LSB of H0;
    HASH-255 = LSB of H7. So the 256 bits HASH-000..255, MSB-first, are the
    digest as a big-endian 256-bit integer.
    """
    value = 0
    for i in range(256):
        bit = vals[f"HASH-{i:03d}"]
        if bit is None:
            raise ValueError(f"HASH-{i:03d} evaluated to X (undefined)")
        value = (value << 1) | (bit & 1)
    return f"{value:064x}"


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def evaluate_mode(graph_path: str, message: str) -> int:
    msg = message.encode("utf-8")
    g = nodes.parse(graph_path)
    block = pad_block(msg)
    pi = message_bits(block)

    # Sanity: the graph's free inputs must be exactly the 512 MESSAGE-* bits.
    free = g.free_inputs()
    expected_free = {f"MESSAGE-{i:03d}" for i in range(512)}
    if free != expected_free:
        missing = expected_free - free
        extra = free - expected_free
        raise ValueError(
            f"free inputs mismatch: missing={sorted(missing)[:5]} "
            f"extra={sorted(extra)[:5]}"
        )

    vals = nodes.evaluate(g, pi)
    got = digest_from_values(vals)
    expected = hashlib.sha256(msg).hexdigest()
    ok = got == expected
    print(f"message={message!r} ({len(msg)} bytes)")
    print(f"  got      = {got}")
    print(f"  expected = {expected}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def equivalence_mode(path_a: str, path_b: str, k: int = 256, seed: int = 0) -> int:
    ga = nodes.parse(path_a)
    gb = nodes.parse(path_b)

    fa, fb = ga.free_inputs(), gb.free_inputs()
    if fa != fb:
        raise ValueError("graphs have different free inputs")
    oa = sorted(n.id for n in ga.outputs())
    ob = sorted(n.id for n in gb.outputs())
    if oa != ob:
        raise ValueError("graphs have different outputs")

    rng = random.Random(seed)
    free = sorted(fa)
    out_ids = oa
    for trial in range(k):
        pi = {fid: rng.randint(0, 1) for fid in free}
        va = nodes.evaluate(ga, pi)
        vb = nodes.evaluate(gb, pi)
        for oid in out_ids:
            if va[oid] != vb[oid]:
                print(f"MISMATCH on trial {trial} at output {oid}: "
                      f"{va[oid]} != {vb[oid]}")
                return 1
    print(f"equivalence PASS: {k} random vectors, {len(out_ids)} outputs match")
    return 0


def selftest_mode(graph_path: str) -> int:
    """A graph equals itself; a deliberately mutated copy does not."""
    g = nodes.parse(graph_path)

    # identity
    rc = equivalence_mode(graph_path, graph_path, k=32)
    if rc != 0:
        print("SELFTEST FAIL: graph not equivalent to itself")
        return 1

    # mutated copy: flip one constant node's value (or invert an output).
    g2 = nodes.parse(graph_path)
    mutated = False
    for nid, n in g2.nodes.items():
        if n.type == "C":
            g2.nodes[nid] = nodes.Node("C", nid, value=1 - n.value)
            mutated = True
            break
    if not mutated:
        # fall back: invert an output's source by pointing it at a const
        out = g2.outputs()[0]
        c0 = nodes.Node("C", "MUT-CONST-0", value=0)
        g2.add(c0)
        g2.nodes[out.id] = nodes.Node("O", out.id, input="MUT-CONST-0")

    tmp = graph_path + ".mutated"
    g2.write(tmp)
    rc2 = equivalence_mode(graph_path, tmp, k=64)
    import os
    os.remove(tmp)
    if rc2 == 0:
        print("SELFTEST FAIL: mutated graph reported equivalent")
        return 1
    print("SELFTEST PASS: identity equivalent, mutation distinguished")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    mode = argv[1]
    if mode == "evaluate":
        return evaluate_mode(argv[2], argv[3])
    if mode == "equivalence":
        k = int(argv[4]) if len(argv) > 4 else 256
        return equivalence_mode(argv[2], argv[3], k)
    if mode == "selftest":
        return selftest_mode(argv[2])
    print(f"unknown mode {mode!r}")
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
