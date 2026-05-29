"""Emit SHA256-base.nodes: a single-block SHA256 circuit built from NAND gates.

Single 512-bit block, fixed IV, including the final feed-forward add. No padding,
no multi-block chaining inside the circuit. Free inputs are exactly MESSAGE-000..511;
outputs are exactly HASH-000..255.

Constants K-* and H-* are DERIVED from first principles (not pasted from memory):
  K = first 32 fractional bits of the cube roots of the first 64 primes.
  H = first 32 fractional bits of the square roots of the first 8 primes.

Bit ordering: big-endian, MSB-first.
  MESSAGE-000 = bit 7 (MSB) of byte 0; MESSAGE-007 = bit 0 of byte 0; ...
  32-bit word w = bytes 4w..4w+3 big-endian; MESSAGE-{32w} is its MSB.
  HASH-000 = MSB of H0; HASH-031 = LSB of H0; HASH-255 = LSB of H7.

We represent a 32-bit value internally as a Python list of 32 node IDs,
index 0 = MSB (bit 31), index 31 = LSB (bit 0). This "big-endian list"
matches the MESSAGE/HASH bit numbering directly.
"""

from __future__ import annotations

import nodes


# --------------------------------------------------------------------------
# Constant derivation (from first principles)
# --------------------------------------------------------------------------
def first_primes(n: int) -> list[int]:
    primes: list[int] = []
    cand = 2
    while len(primes) < n:
        is_p = True
        for p in primes:
            if p * p > cand:
                break
            if cand % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(cand)
        cand += 1
    return primes


def frac_bits_of_root(prime: int, root: int) -> int:
    """First 32 fractional bits of prime ** (1/root), as a uint32.

    Use integer arithmetic for exactness: take the integer cube/square root of
    prime * 2**(32*root), which equals floor(frac * 2**32) for the fractional
    part (the integer part contributes whole multiples that vanish mod 2**32).
    """
    # x = floor((prime * 2**(32*root)) ** (1/root)) = floor(prime**(1/root) * 2**32)
    scaled = prime << (32 * root)
    x = integer_root(scaled, root)
    return x & 0xFFFFFFFF


def integer_root(n: int, k: int) -> int:
    """Floor of the k-th root of nonnegative integer n (exact, integer-only)."""
    if n < 0:
        raise ValueError("negative")
    if n == 0:
        return 0
    # initial guess from bit length
    x = 1 << ((n.bit_length() + k - 1) // k)
    while True:
        # Newton iteration for k-th root
        t = ((k - 1) * x + n // (x ** (k - 1))) // k
        if t >= x:
            break
        x = t
    # x is now an upper-ish estimate; fix off-by-one
    while x ** k > n:
        x -= 1
    while (x + 1) ** k <= n:
        x += 1
    return x


def derive_K() -> list[int]:
    primes = first_primes(64)
    return [frac_bits_of_root(p, 3) for p in primes]


def derive_H() -> list[int]:
    primes = first_primes(8)
    return [frac_bits_of_root(p, 2) for p in primes]


# --------------------------------------------------------------------------
# Circuit builder
# --------------------------------------------------------------------------
class Builder:
    def __init__(self) -> None:
        self.g = nodes.Graph()
        self._uid = 0
        self._const0: str | None = None
        self._const1: str | None = None

    def _new_id(self, prefix: str) -> str:
        self._uid += 1
        return f"{prefix}_{self._uid}"

    def const0(self) -> str:
        if self._const0 is None:
            self._const0 = "CONST-0"
            self.g.add(nodes.Node("C", self._const0, value=0))
        return self._const0

    def const1(self) -> str:
        if self._const1 is None:
            self._const1 = "CONST-1"
            self.g.add(nodes.Node("C", self._const1, value=1))
        return self._const1

    # --- NAND macro library ------------------------------------------------
    def nand(self, a: str, b: str) -> str:
        nid = self._new_id("N")
        self.g.add(nodes.Node("N", nid, inputs=(a, b)))
        return nid

    def NOT(self, a: str) -> str:
        return self.nand(a, a)

    def AND(self, a: str, b: str) -> str:
        return self.NOT(self.nand(a, b))

    def OR(self, a: str, b: str) -> str:
        return self.nand(self.NOT(a), self.NOT(b))

    def XOR(self, a: str, b: str) -> str:
        t = self.nand(a, b)
        return self.nand(self.nand(a, t), self.nand(b, t))

    # --- full adder --------------------------------------------------------
    def full_adder(self, a: str, b: str, cin: str) -> tuple[str, str]:
        """Returns (sum, cout). sum = a^b^cin; cout = majority(a,b,cin)."""
        axb = self.XOR(a, b)
        s = self.XOR(axb, cin)
        # cout = (a&b) | (cin & (a^b))
        ab = self.AND(a, b)
        cab = self.AND(cin, axb)
        cout = self.OR(ab, cab)
        return s, cout

    # --- 32-bit operations (lists are MSB-first, index 0 = bit 31) --------
    def add32(self, x: list[str], y: list[str]) -> list[str]:
        """32-bit ripple-carry add, mod 2**32. cin0 = const 0, drop final carry.

        x and y are MSB-first lists (index 0 = bit 31, index 31 = bit 0).
        Ripple from LSB (index 31) up to MSB (index 0).
        """
        assert len(x) == 32 and len(y) == 32
        out = [None] * 32  # type: ignore[var-annotated]
        carry = self.const0()
        for i in range(31, -1, -1):  # bit 0 (LSB) first
            s, carry = self.full_adder(x[i], y[i], carry)
            out[i] = s
        # final carry dropped (mod 2**32)
        return out  # type: ignore[return-value]

    def add32_many(self, terms: list[list[str]]) -> list[str]:
        acc = terms[0]
        for t in terms[1:]:
            acc = self.add32(acc, t)
        return acc

    def xor32(self, x: list[str], y: list[str]) -> list[str]:
        return [self.XOR(x[i], y[i]) for i in range(32)]

    def and32(self, x: list[str], y: list[str]) -> list[str]:
        return [self.AND(x[i], y[i]) for i in range(32)]

    def not32(self, x: list[str]) -> list[str]:
        return [self.NOT(x[i]) for i in range(32)]

    # --- free rotations / shifts (MSB-first list semantics) ---------------
    def rotr(self, x: list[str], n: int) -> list[str]:
        """ROTR^n: bit at output position j gets input bit (j+n) mod 32 (LSB sense).

        With MSB-first lists, position in list index `li` corresponds to bit
        (31 - li). Output bit b = input bit (b + n) mod 32. Equivalently, a
        right-rotate by n of the 32-bit word. For an MSB-first list this is a
        cyclic shift: out[li] = x[(li - n) mod 32]? Let's derive carefully.

        Let in_list = x (index 0 = bit31 ... index31 = bit0).
        bit value at logical bit b is x[31 - b].
        ROTR^n output bit b = input bit (b + n) mod 32.
        So out_list[31 - b] = x[31 - ((b + n) mod 32)].
        Let li = 31 - b  => b = 31 - li.
        out_list[li] = x[31 - ((31 - li + n) mod 32)].
        """
        out = [None] * 32  # type: ignore[var-annotated]
        for li in range(32):
            b = 31 - li
            src_b = (b + n) % 32
            out[li] = x[31 - src_b]
        return out  # type: ignore[return-value]

    def shr(self, x: list[str], n: int) -> list[str]:
        """SHR^n: logical right shift; vacated high bits become const 0.

        Output bit b = input bit (b + n) if (b+n) < 32 else 0.
        out_list[li] (b = 31 - li): src_b = b + n; if src_b < 32 use x[31-src_b]
        else const0.
        """
        out = [None] * 32  # type: ignore[var-annotated]
        zero = self.const0()
        for li in range(32):
            b = 31 - li
            src_b = b + n
            if src_b < 32:
                out[li] = x[31 - src_b]
            else:
                out[li] = zero
        return out  # type: ignore[return-value]

    # --- SHA256 functions --------------------------------------------------
    def Ch(self, e: list[str], f: list[str], g: list[str]) -> list[str]:
        # (e & f) ^ (~e & g)
        return self.xor32(self.and32(e, f), self.and32(self.not32(e), g))

    def Maj(self, a: list[str], b: list[str], c: list[str]) -> list[str]:
        # (a&b) ^ (a&c) ^ (b&c)
        return self.xor32(self.xor32(self.and32(a, b), self.and32(a, c)),
                          self.and32(b, c))

    def Sigma0(self, x: list[str]) -> list[str]:
        return self.xor32(self.xor32(self.rotr(x, 2), self.rotr(x, 13)),
                          self.rotr(x, 22))

    def Sigma1(self, x: list[str]) -> list[str]:
        return self.xor32(self.xor32(self.rotr(x, 6), self.rotr(x, 11)),
                          self.rotr(x, 25))

    def sigma0(self, x: list[str]) -> list[str]:
        return self.xor32(self.xor32(self.rotr(x, 7), self.rotr(x, 18)),
                          self.shr(x, 3))

    def sigma1(self, x: list[str]) -> list[str]:
        return self.xor32(self.xor32(self.rotr(x, 17), self.rotr(x, 19)),
                          self.shr(x, 10))

    # --- constant word as MSB-first list of constant nodes ----------------
    def const_word(self, value: int, id_for_bit) -> list[str]:
        """Create 32 constant nodes for `value`. id_for_bit(bb) -> node id,
        where bb is the bit index from LSB (0..31). Returns MSB-first list."""
        out = [None] * 32  # type: ignore[var-annotated]
        for bb in range(32):  # bb = bit index from LSB
            bit = (value >> bb) & 1
            nid = id_for_bit(bb)
            self.g.add(nodes.Node("C", nid, value=bit))
            li = 31 - bb  # MSB-first list index
            out[li] = nid
        return out  # type: ignore[return-value]


def build() -> nodes.Graph:
    K = derive_K()
    H = derive_H()
    b = Builder()

    # Message words W[0..15] from MESSAGE-* free inputs.
    # Word w = bytes 4w..4w+3 big-endian; MESSAGE-{32w} = MSB of word w.
    # MSB-first list: index 0 = bit 31 = MESSAGE-{32w}; index 31 = bit 0 = MESSAGE-{32w+31}.
    W: list[list[str]] = []
    for w in range(16):
        word = []
        for li in range(32):  # li 0 = MSB
            idx = 32 * w + li
            word.append(f"MESSAGE-{idx:03d}")
        W.append(word)

    # Message schedule W[16..63].
    for t in range(16, 64):
        s1 = b.sigma1(W[t - 2])
        s0 = b.sigma0(W[t - 15])
        Wt = b.add32_many([s1, W[t - 7], s0, W[t - 16]])
        W.append(Wt)

    # K constants as constant-node words.
    Kw: list[list[str]] = []
    for t in range(64):
        Kw.append(b.const_word(K[t], lambda bb, t=t: f"K-{t:02d}-{bb:02d}"))

    # Initial hash values H[0..7] as constant-node words.
    Hw: list[list[str]] = []
    for i in range(8):
        Hw.append(b.const_word(H[i], lambda bb, i=i: f"H-{i}-{bb:02d}"))

    # Working variables a..h initialized to H0..H7.
    a, bb_, c, d, e, f, g, h = (Hw[0], Hw[1], Hw[2], Hw[3],
                                Hw[4], Hw[5], Hw[6], Hw[7])

    for t in range(64):
        T1 = b.add32_many([h, b.Sigma1(e), b.Ch(e, f, g), Kw[t], W[t]])
        T2 = b.add32(b.Sigma0(a), b.Maj(a, bb_, c))
        h = g
        g = f
        f = e
        e = b.add32(d, T1)
        d = c
        c = bb_
        bb_ = a
        a = b.add32(T1, T2)

    # Feed-forward add: H_i += working var.
    final = [
        b.add32(Hw[0], a),
        b.add32(Hw[1], bb_),
        b.add32(Hw[2], c),
        b.add32(Hw[3], d),
        b.add32(Hw[4], e),
        b.add32(Hw[5], f),
        b.add32(Hw[6], g),
        b.add32(Hw[7], h),
    ]

    # Emit HASH outputs. HASH-000 = MSB of H0 (list index 0 of final[0]).
    for i in range(8):
        word = final[i]
        for li in range(32):  # li 0 = MSB
            hash_idx = 32 * i + li
            oid = f"HASH-{hash_idx:03d}"
            b.g.add(nodes.Node("O", oid, input=word[li]))

    return b.g


if __name__ == "__main__":
    g = build()
    g.write("SHA256-base.nodes")
    free = g.free_inputs()
    outs = [n.id for n in g.outputs()]
    print(f"nodes: {len(g.nodes)}")
    print(f"free inputs: {len(free)}")
    print(f"outputs: {len(outs)}")
