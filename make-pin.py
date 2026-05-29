"""make-pin.py — generate a pin file for partial evaluation of the SHA256 circuit.

A pin file is a .nodes file of `C,MESSAGE-xxx,<bit>` lines that fix some of the
512 free message bits to constants. Feed it to optimize-graph.py via --pin.

Bit mapping is big-endian, MSB-first (matches plan.md / the base circuit):
  MESSAGE-{8*byte + k} = bit (7-k) of block byte `byte`.

Modes:
  python make-pin.py message <text> [out.pin]
      Pin ALL 512 bits to the padded single block for <text> (<=55 bytes).
      Optimizing with this collapses the circuit to constant HASH outputs.

  python make-pin.py pad <nbytes> [out.pin]
      Pin ONLY the deterministic padding bits for a message of <nbytes> bytes
      (the 0x80 marker, the zero fill, and the 64-bit length). The nbytes*8
      message-content bits stay free, so optimizing folds the padding-dependent
      logic while keeping a circuit over just the real input bits.
"""

from __future__ import annotations

import sys


def padded_block(msg: bytes) -> bytes:
    if len(msg) > 55:
        raise ValueError("message must be <= 55 bytes for a single block")
    block = bytearray(64)
    block[: len(msg)] = msg
    block[len(msg)] = 0x80
    block[56:64] = (len(msg) * 8).to_bytes(8, "big")
    return bytes(block)


def bit_lines(block: bytes, pin_byte_indices) -> list[str]:
    """C-lines for the bits in the given byte positions (big-endian MSB-first)."""
    lines = []
    for byte_i in pin_byte_indices:
        byte = block[byte_i]
        for k in range(8):
            idx = byte_i * 8 + k
            bit = (byte >> (7 - k)) & 1
            lines.append(f"C,MESSAGE-{idx:03d},{bit}")
    return lines


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    mode = argv[1]
    if mode == "message":
        msg = argv[2].encode("utf-8")
        out = argv[3] if len(argv) > 3 else "message.pin"
        block = padded_block(msg)
        lines = bit_lines(block, range(64))  # all 64 bytes -> 512 bits
    elif mode == "pad":
        nbytes = int(argv[2])
        out = argv[3] if len(argv) > 3 else "pad.pin"
        block = padded_block(b"\x00" * nbytes)  # content is zero but those bytes stay free
        lines = bit_lines(block, range(nbytes, 64))  # pin only padding bytes
    else:
        print(__doc__)
        return 2

    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} pinned bits)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
