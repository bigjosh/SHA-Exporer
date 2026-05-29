# SHA256 Circuit Explorer

Goal: a browser-based UI to help humans explore and understand the SHA256 algorithm.
This document specs the **graph tooling** that runs at setup time to produce the data the
viewer will later display. The web UI itself is out of scope here; decisions below are
made so as not to box in that later work.

---

## 1. Representation

SHA256 is represented as an **acyclic directed circuit** of three node types. Every node
carries a single bit value that can drive any number of inputs lower in the graph. Every
node has a unique, human-readable ID.

| Type | Code | Inputs | Value |
|------|------|--------|-------|
| Constant | `C` | none | statically `0` or `1` |
| NAND     | `N` | exactly 2 | `NAND(in0, in1)` |
| Output   | `O` | exactly 1 | a copy of its input |

- **Constants** are the only source of defined `0`/`1`. (Algorithm constants `K-*`, `H-*`
  are constant nodes.)
- **NAND** is the sole logic primitive — it is universal, so `NOT`, `AND`, `OR`, `XOR`, and
  full adders are all built from it (see §6).
- **Output** nodes are the only nodes that may not be optimized away. They pin everything
  they (transitively) depend on into the graph. Each `HASH-*` bit is an output node.

### Free / undefined inputs
A node ID may be *referenced* (as a NAND/output input) without being *defined* by any line
in the file. Such IDs are **free inputs** with unknown value (`X`). The 512 message bits are
free inputs in the base graph. The optimizer treats `X` symbolically — circuits over unknown
values can still simplify (e.g. `NAND(X, 0) = 1`).

---

## 2. Conventions (locked)

**Bit ordering — big-endian, MSB-first (matches FIPS 180-4 and hex digests).**
- `MESSAGE-000 … MESSAGE-511`: the 512-bit block as a byte stream. Byte 0 first; within each
  byte the MSB comes first. So `MESSAGE-000` = bit 7 of byte 0, `MESSAGE-007` = bit 0 of
  byte 0, `MESSAGE-008` = bit 7 of byte 1, … This makes 32-bit word *w* = bytes `4w..4w+3`
  big-endian, and `MESSAGE-{32w}` its MSB.
- `HASH-000 … HASH-255`: the digest `H0‖H1‖…‖H7` (8 big-endian 32-bit words). `HASH-000` =
  MSB of `H0` (high bit of the first hex nibble); `HASH-031` = LSB of `H0`; `HASH-255` = LSB
  of `H7`. (Sanity: digest of `"abc"` starts `0xba = 1011…`, so `HASH-000 = 1`.)

**Index zero-padding — fixed 3 digits:** `MESSAGE-000..511`, `HASH-000..255`.

**Internal constant IDs:** `K-<rr>-<bb>` and `H-<i>-<bb>`, where `rr`/`i` is the word index
and `bb` is the bit index counting from the **LSB** (per the spec example: `C,K-00-00,0` is
the bottom bit of `K[0]=0x428a2f98`). These IDs are cosmetic; only the `MESSAGE-*`/`HASH-*`
interface IDs carry external meaning.

**Scope:** a **single 512-bit block** compressed with the fixed IV (`H-*` = SHA256 initial
hash values, `K-*` = the 64 round constants), including the final feed-forward add that
produces the digest. **No** multi-block chaining and **no** padding inside the circuit —
padding is a test/helper concern (§5).

---

## 3. Tools & pipeline

```
                 create-sha256-base.py
   (constants) ─────────────────────────▶  SHA256-base.nodes   (free MESSAGE-* inputs)
                                                   │
        (optional) prepend a constants file  ◀─────┤  e.g. C,MESSAGE-000,0  …  to pin inputs
                                                   ▼
                 optimize-graph.py
   SHA256-base.nodes ───────────────────▶  SHA256-opt.nodes     (minimized, same format)
                                                   │
                 verify-graph.py (run before AND after every optimization)
        compare HASH-* outputs to hashlib / check two graphs are equivalent
```

All tools share **`nodes.py`** (the format parser/serializer + graph model + 3-valued
evaluator). Input and output files are the same type/format, so files may be concatenated
(e.g. prepend a file pinning the top *n* message bits to `0`, then re-optimize — fewer free
inputs unlocks far more simplification).

| File | Role |
|------|------|
| `nodes.py` | Shared library: parse, serialize, graph model, topo sort, dead-cone, 3-valued eval. **Frozen interface** (§7). |
| `create-sha256-base.py` | Emits `SHA256-base.nodes` by directly translating SHA256 into NAND macros. Favors *simple, uniform, possibly redundant* construction — the optimizer cleans up. |
| `verify-graph.py` | Simulator/verifier: evaluate a graph on a message → compare HASH to `hashlib`; also random-vector equivalence between two graphs; includes the single-block padding helper. |
| `optimize-graph.py` | The graph compiler (§4) — the centerpiece. |
| *(future)* `make-pretty.py` | Cosmetic, semantics-preserving renaming of internal node IDs. The optimizer does **not** do this. |

---

## 4. Graph optimization (`optimize-graph.py`) — the centerpiece

Aggressively minimize total node count, and secondarily complexity (depth / size of each
node's dependency cone). The problem is NP, so these are heuristics; the effort is justified
because every downstream use rides on this output.

**Internal model: And-Inverter Graph (AIG) with complemented edges.** The on-disk format is
always `N/C/O`, but internally we use AND nodes with a per-edge "complemented" flag. This
makes `NOT` free, lets structural hashing dedup a node and its complement together, and is
the representation industry tools (e.g. ABC) optimize on. We build a *purpose-built* AIG
optimizer (per experiments, beats general ABC/SAT on this specific circuit).

Passes:
1. **Strash (structural hashing):** hash-cons AND nodes by canonicalized `(fanin0, fanin1,
   compl)`. Folds trivial cases on construction: `AND(x,0)=0`, `AND(x,1)=x`, `AND(x,x)=x`,
   `AND(x,¬x)=0` — this subsumes constant folding and common-subexpression elimination.
   `X` (free input) propagates: `AND(0,X)=0`, `AND(1,X)=X`, etc.
2. **Dead-cone elimination:** keep only nodes reachable from output nodes; sweep the rest
   (recursively). Any non-output node with no fanout is deletable.
3. **Rewrite / refactor / balance** (iterate to fixpoint):
   - *rewrite* — cut enumeration (≈4-input) + replacement with precomputed size-optimal
     subgraphs per NPN-class (DAG-aware).
   - *refactor* — larger-cut collapse + factored-form re-synthesis.
   - *balance* — tree balancing to reduce depth.
   Start with strash + a curated local rewrite-rule set + dead-cone; add cut-rewriting and
   refactoring as the heavy machinery matures.
4. **Lowering AIG → NAND/C/O:** per node choose to materialize the AND, its complement, or
   both, to minimize inserted inverters; complemented output edges become a `NAND(x,x)`
   inverter only when needed; constants → `C`; outputs → `O`.

**Objective:** primary = AND/NAND count; secondary = depth (levels). Seedable / deterministic.

**Stopping criterion (NP, "good enough"):** iterate passes to fixpoint, then run bounded
randomized local search (randomized rewrites, accept-if-not-worse / light annealing) under a
**wall-clock budget**, checkpointing the best graph to disk. Because the format round-trips,
a run is resumable and can be continued for longer later.

**Identity & IDs (per decision):** the optimizer is purely semantics-preserving and does
**not** prettify/anonymize. Interface IDs (`MESSAGE-*`, `HASH-*`, any caller-supplied named
inputs) are preserved exactly. Internal nodes get deterministic generated IDs; making those
human-friendly is the separate future `make-pretty.py` tool.

**Correctness safety net:** re-run `verify-graph.py` before and after optimization — both the
end-to-end `hashlib` check and random-vector equivalence between input and output graphs. A
SAT/BDD miter could be added later for a formal guarantee.

---

## 5. Verification (`verify-graph.py`)

- **Evaluate mode:** load a graph, assign `MESSAGE-*` from a message (or from a prepended
  constants file), evaluate in topological order (3-valued), read `HASH-*`, compare to
  `hashlib.sha256`. Vectors: `""` → `e3b0c442…`, `"abc"` → `ba7816bf…`, plus a 55-byte
  (max single-block) message and random short messages.
- **Equivalence mode:** given two graphs with identical PIs/POs, run K random input
  assignments through both and assert identical outputs. Cheap regression gate for the
  optimizer.
- **Padding helper:** build the 64-byte block for a message ≤ 55 bytes:
  `msg ‖ 0x80 ‖ 0x00… ‖ uint64_be(bitlen)`.

---

## 6. NAND macro library (used by the generator)

Build everything from NAND; favor uniformity and let the optimizer fold redundancy.

| Macro | Construction | NAND gates |
|-------|--------------|-----------:|
| `NOT(a)` | `NAND(a,a)` | 1 |
| `AND(a,b)` | `NOT(NAND(a,b))` | 2 |
| `OR(a,b)` | `NAND(NOT a, NOT b)` | 3 |
| `XOR(a,b)` | `t=NAND(a,b); NAND(NAND(a,t), NAND(b,t))` | 4 |
| full adder | `sum = a⊕b⊕cin`, `cout = maj` | ≈9 |
| 32-bit add | 32 ripple full adders, `cin0 = const 0`, drop final carry (mod 2³²) | ≈288 |

- **ROTR / ROTL are free** — a rotation is just a relabeling of which wires feed where (0
  gates). **SHR** is also free but injects `const 0` into the vacated high bits.
- SHA256 functions: `Ch(e,f,g)=(e∧f)⊕(¬e∧g)`, `Maj(a,b,c)=(a∧b)⊕(a∧c)⊕(b∧c)`,
  `Σ0=ROTR2⊕ROTR13⊕ROTR22`, `Σ1=ROTR6⊕ROTR11⊕ROTR25`,
  `σ0=ROTR7⊕ROTR18⊕SHR3`, `σ1=ROTR17⊕ROTR19⊕SHR10`.
- Message schedule: `W[t]=M[t]` for `t<16`; else `W[t]=σ1(W[t-2])+W[t-7]+σ0(W[t-15])+W[t-16]`.
- Round: `T1=h+Σ1(e)+Ch(e,f,g)+K[t]+W[t]`, `T2=Σ0(a)+Maj(a,b,c)`; shift state, `e=d+T1`,
  `a=T1+T2`. After 64 rounds, `H_i += working vars`; emit `H0..H7` as `HASH-*` outputs.

Expect the base graph to be on the order of 10⁵ NAND nodes pre-optimization (XOR/adder
dominated) — shrinking it is the optimizer's job.

---

## 7. Node file format

A plain text file, one node per line, comma-separated values. **Blank lines and lines
starting with `#` are ignored.** Otherwise the format is strict: a malformed line, a
duplicate node ID, or other unexpected condition should raise and crash (favor simple code
over defensive checks). IDs use conservative variable naming (`[A-Za-z0-9_.-]`).

- First value: node type `C` / `N` / `O`.
- Second value: node ID.
- Then one or two more values by type.

**Constant** — one extra value `[0,1]`:
```
C,K-00-00,0
```
(bottom/LSB bit of the first K constant)

**NAND** — two extra values, the IDs of its two inputs:
```
N,RND8-4-566,ROT-27-18,SFT-88-04
```

**Output** — one extra value, the ID of its input:
```
O,HASH-251,RND8-4-566
```
(bit 251 of the final SHA256 output)

### `nodes.py` public interface (frozen — build against this)
- `parse(path) -> Graph` / `parse_lines(iterable) -> Graph`
- `Graph.nodes: dict[str, Node]` — defined nodes only
- `Graph.free_inputs() -> set[str]` — referenced-but-undefined IDs (e.g. `MESSAGE-*`)
- `Graph.outputs() -> list[Node]` — the `O` nodes
- `Graph.topo_order() -> list[str]` — topological order of defined nodes
- `Graph.serialize() -> str` / `Graph.write(path)`
- `evaluate(graph, pi_values: dict[str, int|None]) -> dict[str, int|None]` — 3-valued
  (`None` = `X`), NAND semantics: any `0` input ⇒ `1`; both `1` ⇒ `0`; else `X`.
- `Node` fields: `.type` (`'C'|'N'|'O'`), `.id`, and `.value` (C) / `.inputs` tuple (N) /
  `.input` (O).

---

## 8. Optimizer — implementation status & results

The base circuit is **395,009 nodes (392,448 NAND)**. Optimization is sound (every
transform preserves the boolean function; results are verified by random-vector
equivalence + the `hashlib` battery, and FRAIG merges are individually SAT-proven).

| Pass | NAND | vs base | Notes |
|------|-----:|--------:|-------|
| baseline (strash) | 253,620 | −35.4% | structural hashing = constant fold + CSE + dead-cone; all 2,305 constants fold away |
| + FRAIG (Phase 2) | 239,780 | −38.9% | 11,856 SAT-proven functional merges (incl. ~3,950 functionally-constant nodes) |

**Soundness of FRAIG.** Simulation alone is *unsound* — two nodes can agree on
thousands of random vectors yet differ on rare inputs (an early sim-only merge
diverged at vector 809; the gate caught it). So simulation only *proposes*
candidates; each is *confirmed* by a `pysat` node-pair miter (equivalent ⇔ both
differing assignments UNSAT) before merging. Nothing is merged without a proof.

**Partial evaluation** (`--pin`) folds pinned inputs to constants and re-runs
strash, collapsing whole cones. Fully pinning a message → constant outputs equal
to its digest; pinning only a fixed-length message's padding (`make-pin.py pad N`)
specializes the circuit (e.g. a 3-byte message → 230,353 NAND, −41.3%, still
computing correct SHA-256 over the 24 free bits).

**Tools:** `optimize-graph.py <in> <out> [--pin FILE ...] [--vectors N] [--seed S]`;
`make-pin.py {message <text>|pad <nbytes>} [out]`; `sim.py <a> <b> [N]` (fast
bit-parallel equivalence). The optimizer is deterministic (same input ⇒ byte-
identical output).

**Why ~239k and not lower:** the cost is dominated by ~19,200 modular adders,
which are already near-optimal in the AND-only AIG after strash+FRAIG. Pushing
substantially lower needs the higher-complexity passes below.

**Roadmap (future work, ordered by ROI/risk):**
- **XAIG** — first-class XOR/MAJ nodes + optimal XOR3/MAJ3/MUX NAND lowering
  (recognizes that `Ch` is a MUX, adder sum is XOR3, carry is MAJ3); reduces both
  representation size and the lowered NAND count for the adder/`Ch`/`Maj` logic.
- **GF(2)-linear (P-LIN)** — collapse the pure-XOR Σ/σ/schedule cones to parity
  sets and re-synthesize a shared minimal XOR network (Boyar–Peralta-style).
- **CSA** — carry-save re-synthesis of the multi-operand adds (depth, and exposes
  shared compressor cells).
- **NPN-4 cut rewriting** — curated optimal subgraphs for recurring 4-input cuts
  (sound via exact local truth tables); a polisher for residual local slack.
- **Outer driver** — fixpoint loop + wall-clock budget + checkpoint/resume +
  randomized local search (lets a run be continued for a bigger result later).
- **Depth balancing** — size-preserving; balance *associative* XOR/AND trees only
  (carry ripples are not associative). Improves graph explorability for the UI.
