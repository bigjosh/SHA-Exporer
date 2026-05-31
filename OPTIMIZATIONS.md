# SHA-256 circuit optimizer — what we run, results, and dead ends

This documents `optimize-graph.py`: the passes it runs, in what order, the results
on the base SHA-256 circuit, and — importantly — what we tried that **didn't**
work, so the next person doesn't re-walk the same paths. For the data model and
node-file format see [`plan.md`](plan.md).

## TL;DR

- Base circuit: **392,448 NAND** (`SHA256-base.nodes`, single 512-bit block, fixed IV).
- After the full pipeline: **~227,000 NAND (≈ −42%)**, provably equivalent.
- We are **near the NAND-realization floor** (estimated hard floor ~190–200k). A 2×
  reduction is not achievable: the circuit is XOR/adder-dominated, and in a NAND
  basis a 2-input XOR costs 4 NAND and a full adder 9 NAND — both irreducible.

## The pipeline (in order)

The optimizer works on an internal **And-Inverter Graph** (AIG: 2-input AND nodes,
inverters as an edge flag), then lowers back to NAND/C/O. Order matters; each pass
feeds the next.

### 1. Strash (structural hashing) — `build_aig`
Hash-cons every AND node by its canonical fanin pair, folding trivial cases
(`AND(x,0)=0`, `AND(x,1)=x`, `AND(x,x)=x`, `AND(x,¬x)=0`). This does **constant
folding + common-subexpression elimination + dead-cone elimination** in one pass.
Sound (boolean identities only).
> **392,448 → 253,620 NAND (−35.4%)**. All 2,305 constant nodes fold away.

### 2. FRAIG — SAT-confirmed functional merging — `fraig`
Finds nodes that compute the **same function built differently** (which strash
can't see) and merges them. **Simulation proposes** (group nodes by a 1024-vector
signature, two independent seeds); **SAT confirms** each candidate via a pysat
node-pair miter (equivalent ⇔ both differing assignments UNSAT). Nothing is merged
without a proof. Encoding the CNF is ~1s; each miter ~20 ms (Cadical).
> **253,620 → 239,780 NAND (−38.9%)**. 11,856 SAT-proven merges (incl. ~3,950
> functionally-constant nodes). ~27 simulation-proposed candidates were correctly
> **rejected** by SAT — those were the unsound ones (see below).

### 3. Majority cut-rewrite — `maj_rewrite`
3-input cut enumeration with **exact truth tables**; rewrites nodes whose function
is in the majority NPN class into the optimal `(a&b)|(a&c)|(b&c)` OR/MUX form. The
generator emits `Maj` as `XOR-of-3-ANDs` (FIPS-faithful) which strash/FRAIG can't
restructure across the XOR boundary; this pass does. Sound (exact TT + a global
gain-guard that never regresses).
> Rewrites **1,952 round-`Maj` nodes** (−~11.7k AND). Full pipeline → **~227k NAND**.

### Partial evaluation (`--pin`)
Pin some input bits to constants (e.g. a known-length message's padding) and
re-optimize; strash then folds the now-constant cones. Pinning a 5-byte message's
padding (472 of 512 bits) → 233,474 NAND, still computing correct SHA-256 over the
40 free bits. Fully pinning a message → constant outputs equal to its digest.
**FRAIG is skipped under `--pin` by default** (see dead ends) — use `--force-fraig`
to enable it.

## Results on the base circuit

| Stage | NAND | vs base |
|---|---:|---:|
| base | 392,448 | — |
| strash | 253,620 | −35.4% |
| + FRAIG | 239,780 | −38.9% |
| + maj-rewrite | ~227,000 | ~−42% |

Verified throughout against Python `hashlib` (`""`, `"abc"`, the standard pangram,
55-byte blocks) plus 8,192-vector random equivalence between input and output.
Deterministic: same input ⇒ byte-identical output.

## What we tried that did NOT work (read this before "improving" it)

The circuit was profiled block-by-block; **the adders, `Ch`, and `Σ`/`σ` are all
already optimal**, so most classical logic-synthesis tricks find nothing here.

- **Inverter-minimizing lowering — ~0%.** The lowering is *already* inverter-minimal
  for a fixed AIG (one shared inverter per node whose AND-value is consumed; you
  cannot produce `a∧b` in fewer than 2 NANDs). Reducing inverters needs to change
  the AIG (i.e. rewriting), not the lowering. Measured, not guessed.
- **SAT don't-care optimization (ODC/SDC, ABC `mfs`-style) — ~0%.** The 512 message
  bits are *free* inputs (no satisfiability don't-cares) and all 256 hash outputs
  are observed (no observability don't-cares). The don't-care sets are nearly empty.
- **CSA / Wallace adder trees — ~0% node count** (depth only). A multi-operand
  carry-save tree has the *same full-adder count* as the ripple chain — it only cuts
  depth. Worth it if you later care about depth/explorability, not size.
- **GF(2)-linear (P-LIN) re-synthesis of Σ/σ — ~0%.** Each `Σ`/`σ` is a circulant XOR
  of three rotated copies with *distinct* rotation offsets, which makes it provably
  **sharing-free** — there is no common XOR sub-term to extract.
- **General NPN-4 cut rewriting beyond `Maj` — ~0%.** A full adder is already its
  9-NAND optimum and `Ch` is already the optimal 4-NAND MUX, so wider cut rewriting
  re-discovers the same optima. Only `Maj` had slack (hence pass 3).
- **Glucose3 + conflict-budget for FRAIG — reverted.** We tried making the SAT
  bounded (to stop a hard miter from hanging). Glucose3 was **~14× slower** on the
  large shared CNF *and* hit the budget on miters Cadical proves trivially (losing
  real merges). Cadical is far better here but **pysat cannot bound it mid-solve**
  ("Limited solve is currently unsupported by CaDiCaL").
- **Pure-simulation FRAIG — unsound.** Merging on simulation agreement alone is
  wrong: a candidate that matched on 2,048 random vectors **diverged at vector 809**.
  The equivalence gate caught it, but the lesson stands — SAT confirmation is
  mandatory.
- **FRAIG under `--pin` — disabled by default.** Pinning can create a single *hard*
  SAT miter, and since Cadical can't be bounded, it once spun a full CPU core for
  ~8 hours. strash already captures most of the pinned gain, so `--pin` skips FRAIG
  (opt back in with `--force-fraig`).
- **XOR-native representation (XAIG/XMG) — the only >10% long-shot, not pursued.**
  Keeping XOR native could shrink the *internal* graph, but the gain tends to
  **evaporate at NAND lowering** (XOR is 4 NAND regardless). A research spike, not a
  sure win.

### Gotchas inside the passes (subtle, cost real time)
- **maj-rewrite must skip adder carries.** A carry is *also* `maj(a,b,cin)`, but it's
  already optimal (built as `OR(a&b, c&(a^b))`, sharing `a^b` with the sum). Rewriting
  it to OR-of-3-products would *add* nodes. Discriminator: only rewrite when all three
  pairwise products already exist (true for the round `Maj`, false for the carry).
- **Majority NPN phasing is two-valued.** A truth table maps to *two* self-dual
  phasings (negate-all-inputs ≡ negate-output); you must try both and pick the one
  whose products are actually realizable, or you'll match the wrong one and reject
  everything.
- **Cut enumeration: prune dominated cuts, don't "keep smallest".** Keeping the
  smallest-K cuts severs the deep cuts needed to reach the majority's true leaves.

### Operational lessons (cost us a 12-hour stall)
- **Never run two full-circuit jobs at once** — it caused both an OOM (pre-numpy) and
  a silent multi-hour stall from CPU/memory contention. Serialize heavy runs.
- **`TaskStop` on a background shell orphans its Python child** — kill the process
  directly (PowerShell `Stop-Process`) and verify it's gone.
- **Long runs need a progress indicator + a periodic check.** A silent run + reliance
  on the completion notification hid a hang for 12 hours. The FRAIG SAT loop now
  prints per-minute progress, and there's an 1800s phase cap so it always terminates.

## How to run / verify

```
python create-sha256-base.py                              # -> SHA256-base.nodes
python optimize-graph.py SHA256-base.nodes SHA256-opt.nodes [--pin FILE] [--vectors N]
python verify-graph.py evaluate SHA256-opt.nodes "abc"    # compare to hashlib
python sim.py SHA256-base.nodes SHA256-opt.nodes 8192     # random-vector equivalence
python make-pin.py {message <text> | pad <nbytes>} out.pin
```

## Runtime note
strash and maj-rewrite are fast (tens of seconds). FRAIG's SAT phase is ~minutes on
an idle machine, but a few hard miters can stretch to minutes *each* under heavy CPU
load; the 1800s phase cap bounds it (soundly dropping unconfirmed merges) so it
always finishes.
