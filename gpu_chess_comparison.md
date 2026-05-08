# Comparing chess_gpu.py (ours) vs chessvec (friend's)

A practical, learning-oriented look at two batched GPU chess engines, focused on tensor programming choices and why each leads to specific perf characteristics.

| | Ours (`src/games/chess_gpu.py`) | Friend (`phulin/chessvec`) |
|---|---|---|
| Backend | PyTorch + `torch.compile` | PyTorch fallback + hand-written **Triton kernels** |
| Board representation | 12 bitboards × int64 | 64-square int8 array |
| Sliding attacks | Kogge-Stone bit-fill | F.pad shift loop / Triton ray walk |
| Jump attacks | Bitboard OR + precomputed bitboards per square | `[64,64]` bool table @ matmul |
| Threefold | Zobrist hash + ring buffer | Not implemented (deferred) |
| Speed (per-step, N=128, fused) | ~0.33 ms (compiled CUDA, our bench) | ~0.33 ms ([6] in friend's bench, Triton) |

The headline: both implementations land in the same ballpark when fully optimized, but they get there via different routes — we lean on `torch.compile` to fuse a PyTorch graph; the friend hand-writes Triton kernels. The Triton path bypasses several constraints the PyTorch IR can't escape. This doc walks through the lessons.

---

## 1. Board representation: bitboards vs piece-array

### Ours: `pieces[B, 12] int64`
12 bitboards per game — one per `(color, piece-type)`. Each `int64` is a 64-bit pattern; bit `i` = 1 means that piece occupies square `i`.

```python
# pieces[g, plane=0] is white pawns as a bitboard
# bit-extract: which squares have white pawns?
sq_has_wp = ((pieces[:, 0:1] >> arange(64)) & 1).bool()  # [B, 64]
```

**Pros**
- Compact: 12 × 8 bytes = **96 bytes/game**.
- Bitwise ops (`&`, `|`, `<<`, `>>`) on 64 squares in **one int64 instruction**. This is huge for sliding-piece attack generation (Kogge-Stone, see §2).
- Zero indexing logic — just AND/OR with masks.

**Cons**
- PyTorch's `>>` on `int64` is **arithmetic** (sign-extends from bit 63). We needed a `_lsr` helper that masks the sign-extended bits. The friend never hits this because they don't do shift-based attack generation.
- "Which piece is at square X?" requires probing 12 planes. Friend's representation can answer in one load.
- Uint64 is awkward in PyTorch — bitboards are stored as `int64` with the bit pattern, which introduces a few `_u64_to_int64_tensor` reinterpret-casts.

### Friend's: `pieces[B, 64] int8`
A flat array indexed by square. Each entry is a piece code 0..12 (0 = empty, 1..6 = white P/N/B/R/Q/K, 7..12 = black).

```python
# Which squares have white pawns?
sq_has_wp = (pieces == 1)  # [B, 64]
# Which planes? Broadcast equality.
codes = arange(1, 13).view(1, 12, 1)              # [1, 12, 1]
planes = pieces.view(B, 1, 64) == codes           # [B, 12, 64] bool
```

**Pros**
- 64 bytes/game (1.5× smaller).
- Reading the moving piece is a single gather: `pieces[from_sq]`.
- `[B, 64]` is naturally a **vector inside one Triton program** — each of 64 lanes processes its own square. Bitboards-as-int64 don't map this way.
- **Branchless to derive 12-plane occupancy** via broadcast equality. `torch.compile` and Triton both handle this cleanly.

**Cons**
- Sliding-piece attacks need ray walks (F.pad shifts, or `tl.gather` in Triton); no Kogge-Stone shortcut.
- 64 lanes vs 1 int64 means a slider attack reads 64 bytes instead of doing a 64-bit shift. On a CPU that's worse; on a GPU with 64-wide tile ops, it's *better* because the 64 lanes run in parallel anyway.

### Why the friend's choice is GPU-friendlier
Bitboards compress 64 squares into one 64-bit integer. CPUs love this. GPUs already have wide vector lanes — a `[64]` tile of int8 maps to a warp-level operation, and Triton lets you express that natively. The information density of bitboards mostly buys nothing on a 32-thread warp where each lane handles one square anyway.

That said: **for slider attacks**, our Kogge-Stone fill is `O(log distance)` (3 levels of halving × 8 directions = 24 ops), versus the friend's iterative ray fill (7 steps × 8 directions = 56 ops). On GPU the launch-overhead of those individual ops dominates either way, but the friend gets to fuse all 56 into one Triton kernel and we don't get to fuse our 24 ops into one kernel via `torch.compile` reliably.

---

## 2. Sliding-piece attacks: three different ideas

### Ours — Kogge-Stone fill on bitboards

```python
def _slider_attacks(sliders, empty, direction=DIR_N):
    g, e = sliders, empty
    g = g | (e & (g << 8));  e = e & (e << 8)
    g = g | (e & (g << 16)); e = e & (e << 16)
    g = g | (e & (g << 32))
    return (g << 8)
```

This computes "for every slider, the set of squares it attacks along direction N" in **3 doublings**. The trick: at each step, double both the ray length we've already extended (`g`) and the empty mask (`e`). After 3 levels, every reachable square (up to 7 along a rank/file/diagonal) is covered.

**Tradeoff:** elegant on bitboards, useless without them. The 8-direction sliding fill is 8 calls × 3 levels = 24 PyTorch ops, each launching a CUDA kernel. `torch.compile` will fuse adjacent ones, but our kernel-trace shows it doesn't fuse all 24 into one launch.

### Friend's PyTorch path — F.pad shift loop on `[B, 8, 8]`

```python
def _shift(x, df, dr):
    if df > 0: x = F.pad(x[..., :, :-df], (df, 0, 0, 0))
    if dr > 0: x = F.pad(x[..., :-dr, :], (0, 0, dr, 0))
    return x

attacks = zeros_like(pieces88)
for df, dr in dirs:                     # 8 directions
    ray = _shift(pieces88, df, dr)
    attacks |= ray
    ray &= empty88
    for _ in range(6):                  # 7 steps total
        ray = _shift(ray, df, dr)
        attacks |= ray
        ray &= empty88
```

Each shift is a `[B, 8, 8]` slice + pad — clean PyTorch. 8 × 7 = 56 ops. **About the same launch-overhead pile as our path** but expressed in array semantics rather than bitwise tricks. This is the [1] **PyTorch baseline** in their bench — 9.99 ms/iter.

### Friend's Triton path — per-target ray walk in one kernel

Inside one Triton program (= one game), with `sq = tl.arange(0, 64)` (a 64-lane vector):

```python
slider_atk = tl.zeros([64], tl.int32)
for d_idx in tl.static_range(0, 8):              # unrolled
    df, dr = DFS[d_idx], DRS[d_idx]
    walking = tl.full([64], 1, tl.int32)
    for step in tl.static_range(1, 8):           # unrolled
        target_sq = (rank_sq + dr*step) * 8 + (file_sq + df*step)
        target_piece = tl.gather(pieces_nk, target_sq, axis=0)
        is_piece = (target_piece != 0).to(tl.int32)
        first_piece = walking * is_piece
        matches = (target_piece == rook_code) | (target_piece == queen_code)
        slider_atk = slider_atk | (first_piece * matches)
        walking = walking * (1 - is_piece) * on_board
```

Conceptually identical to the PyTorch ray walk, but **all 64 squares run in parallel** within the kernel. Critical Triton features at play:

- `tl.static_range` unrolls the loop at compile time → the compiler sees 56 explicit ops.
- `tl.gather(pieces_nk, target_sq, axis=0)` does a **scattered read inside the kernel** with no host round-trip.
- All intermediate tensors (`walking`, `slider_atk`, `target_piece`) live in registers/shared memory — never round-tripped to global memory.
- The `[64]` tile size matches the warp-level vector unit; the GPU runs all 64 squares' work in one warp.

This is the [2]→[5] step in their bench: from PyTorch (9.99 ms) → Triton fused (1.15 ms) → with bitboard packing (0.42 ms). The win comes from fusion: one kernel launch instead of ~56.

---

## 3. Jump-piece attacks: bitwise OR vs matmul

### Ours

```python
KNIGHT_ATTACKS[sq]  # int64 bitboard — squares a knight on `sq` attacks
attacks = 0
for sq in arange(64):
    if knight_at_sq[g, sq]: attacks |= KNIGHT_ATTACKS[sq]
```

Implemented as a tensor reduce: for each from-square that has a knight, OR in its attack bitboard.

### Friend's: matmul on a `[64, 64]` bool table

```python
KNIGHT_TABLE  = zeros(64, 64, bool)   # [from, to]
for from_sq in range(64):
    for offset in KNIGHT_DIRS:
        if on_board(...): KNIGHT_TABLE[from_sq, to_sq] = True

# At runtime:
N = (pieces == WN_CODE)               # [B, 64] bool
n_attacks = (N.float() @ KNIGHT_TABLE.float()) > 0  # [B, 64] bool
```

This treats the knight-attack table as a **matrix, and applies it as a matmul**. The intuition: matrix-vector multiply over `{0,1}` is exactly the union-of-rows operation. `N` is the "input" indicator vector; `KNIGHT_TABLE.T @ N` collects all squares attacked by any knight.

**Why matmul on a chess board?** Because cuBLAS/CUTLASS *eats* `[B, 64] @ [64, 64]` matmuls — they're tiny, cache-resident, and hit hundreds of TFLOPS even at small B. A bitwise reduce can't beat that on a GPU.

This trick generalizes: any "for each from-square, contribute its attack set" computation can be a matmul on a precomputed lookup table. On CPU, bitboard-or wins; on GPU, matmul wins.

In the Triton kernel the same idea appears, but the table is loaded once into the kernel (`tl.load` of a `[64, 64]` tile) and used as a 2D operand to a `tl.sum` reduction — same algorithm, different syntax.

---

## 4. Triton: what it is, and why it's fast here

Triton is a Python-like DSL for writing GPU kernels. You write a function decorated with `@triton.jit`; Triton compiles it to PTX (NVIDIA assembly), schedules registers, picks tile sizes, and handles shared-memory layouts. It sits between CUDA C (low-level, manual everything) and PyTorch (high-level, automatic but with kernel launch overhead).

The friend's `_step_kernel` looks like this at a high level:

```python
@triton.jit
def _step_kernel(pieces_ptr, ..., out_pieces_ptr, ..., BLOCK_SQ: tl.constexpr=64):
    pid = tl.program_id(axis=0)              # pid = which game
    sq = tl.arange(0, BLOCK_SQ)              # 64-wide vector of square indices
    pieces = tl.load(pieces_ptr + pid*64 + sq)  # load all 64 squares for this game
    # ... do the entire step (find from_sq, mutate pieces, update castling/EP/clocks) ...
    tl.store(out_pieces_ptr + pid*64 + sq, new_pieces)
```

**The important bits:**
- **`pid`** = thread block ID = game index. Each program handles one game.
- **`sq`** is a vector of 64 lanes. Operations on `sq`-shaped values run on all 64 squares in lockstep.
- **`tl.where(sq == from_sq, ...)` extracts a per-game scalar from a per-square vector** via masked sum: `tl.sum(tl.where(sq == from_sq, pieces, 0))` is the moving piece. No CPU round-trip, no separate kernel.
- **All intermediate tensors are register-resident.** The 64-wide `pieces` vector lives in registers; `tl.load`/`tl.store` happens once at the boundaries.

Compare to our `_step_batch_impl`: that function has roughly **40 PyTorch ops**, each launching its own kernel, each round-tripping intermediate state through HBM (GPU global memory). Even with `torch.compile`, the Inductor backend fuses some but not all of those — the conditional updates (`torch.where(...)` chained 5–10 times for castling-rook moves, EP captures, promotion) are exactly the patterns Inductor doesn't fuse aggressively.

This is **why one Triton kernel beats ~40 PyTorch ops by an order of magnitude even with compile**: every kernel launch is ~5–10 µs of overhead on RTX 4090, and a `step_batch` at N=256 needs the work in <1 ms total. 40 launches = 200–400 µs in pure overhead; the friend's kernel is one launch.

### The bench tells the story

| Step | Per-iter | Speedup | What changed |
|---|---|---|---|
| [1] PyTorch | 9.99 ms | 1× | baseline; everything is per-op kernel launch |
| [2] Triton fused | 1.15 ms | 8.7× | one kernel = one launch, shared memory keeps everything resident |
| [3] Compact pin | 0.76 ms | 13× | replaced `[64,64]` pin matrix with `[64]` direction vector → less shared mem → higher occupancy → `num_warps=16` viable |
| [4] Between-bb | 0.78 ms | 13× | precomputed "squares between from and to" as a 64-bit bitboard → constant-time lookup vs per-step ray walk |
| [5] EA/BoC as u64 | 0.42 ms | 24× | packed 64-square enemy-attack and block-or-capture masks into single `int64` registers → bit tests instead of 64-element compares |
| [6] Plane chunks 2×64 | 0.33 ms | 30× | process 64 of 73 planes in one tile, 9 in the next → smaller working set → `num_warps=4` (less warp-coordination overhead) |
| [7] Plane-allowed bb | 0.34 ms | 29× | per-piece-type "valid plane" set as a bitmask → branchless legality check |

Each of these is a tensor-programming lesson worth understanding:

**[2] Fusion = one kernel launch.** The single biggest lever. The kernel doesn't compute anything *cleverer* than the PyTorch baseline; it just doesn't pay launch overhead 40 times.

**[3] Memory is the limit, not compute.** Triton kernels have a ~48 KB shared-memory budget per thread block. A `[64, 64]` int32 pin matrix is 16 KB — half the budget. Cutting it to `[64]` int32 (256 bytes) frees up shared memory for more in-flight warps, which means more parallelism, which means `num_warps=16` can actually run instead of being throttled. **Less data in shared memory → higher occupancy → more warps in flight.**

**[4] Precompute when the structure is fixed.** "Squares between from and to" is a function of `(from_sq, to_sq)` only, which has 64×64 = 4096 entries — fits in 32 KB as int64 bitboards. Loading the answer is faster than computing it on every step.

**[5] Pack into a register when the data fits in 64 bits.** A 64-square boolean mask = an `int64`. Instead of a 64-element vector compare, do a single `(mask >> sq) & 1` bit test. The `int64` lives in **one register** (vs 64 registers for the vector). Lower register pressure → higher occupancy → faster.

**[6] Tile size matters for occupancy.** A `[64, 73]` legal-action mask is 4672 elements = 18 KB at int32. Two `[64, 64]` chunks (16 KB + ~2 KB) keeps each chunk under the L1 cache size, and lets `num_warps=4` (default) run smoothly. With `num_warps=16` you have 16 × 32 = 512 threads in flight per block, which competes for the same register file — sometimes fewer warps with less work each is faster.

**[7] Bitmask lookups beat conditionals.** Instead of `if piece == ROOK: planes_ok = ROOK_PLANES`, store a `[7] int64` table where each entry is a bitmask of allowed planes for that piece type. The "is plane P allowed for this piece" test becomes `(table[piece] >> P) & 1`. Branchless, no warp divergence.

---

## 5. Specific tensor-programming idioms to steal

### a) Broadcast equality replaces one-hot scatter
Going from `pieces[B, 64]` int8 codes to `[B, 12, 64]` per-piece planes:

```python
codes = arange(1, 13).view(1, 12, 1)          # [1, 12, 1]
planes = pieces.view(B, 1, 64) == codes        # [B, 12, 64] bool
```

No loop, no scatter, no data-dependent indexing. This is **friendly to torch.compile, CUDA Graph capture, and Triton**. Worth using in our codebase wherever we do `for pt in range(6)`.

### b) Matmul-as-table-lookup
For any "from-set → to-set" lookup with a small static table:

```python
n_attacks = (N.float() @ KNIGHT_TABLE.float()) > 0   # [B, 64]
```

vs

```python
attacks = zeros_like(N)
for sq in range(64):
    attacks |= where(N[:, sq:sq+1], KNIGHT_ATTACKS_BB[sq], 0)
```

The matmul version is one cuBLAS call. The bitboard-OR version is 64 (or one `torch.compile`'d) — but the matmul still wins when GPUs have free Tensor Cores.

### c) Masked-sum to extract scalars from vectors
Inside Triton (and equivalent in PyTorch with `.sum(dim=-1)`):

```python
moving_piece = tl.sum(tl.where(sq == from_sq, pieces, 0))
```

Instead of `pieces[from_sq]` (a gather, which has its own overhead in Triton if the index isn't compile-time known), build a one-hot mask and reduce. Cheap when you already have the vector.

### d) Pack booleans into bitboards inside kernels
```python
# 64 lanes of 0/1 → one int64 register
bit_per_sq = (1 << sq.to(int64))                   # [64] int64, one bit per lane
mask_bb = tl.sum(tl.where(mask != 0, bit_per_sq, 0))   # scalar int64
```

Once packed, set/clear/test individual bits with `<<`, `|`, `&`, `>>`. **No more vector compares for bitset ops** — one ALU op per test.

### e) `tl.static_range` for unrolling
```python
for d_idx in tl.static_range(0, 8):
    df, dr = DFS[d_idx], DRS[d_idx]
    ...
```

`tl.static_range` unrolls at compile time. The kernel sees 8 explicit copies of the loop body, no branch, register-friendly. **Use whenever the loop bound is fixed and small** (≤16 typical).

---

## 6. Where we differ on chess semantics (not perf-related)

| | Ours | Friend |
|---|---|---|
| Threefold repetition | Zobrist hash + 160-slot ring buffer | **Not implemented** |
| Halfmove draw | 75-move (auto, halfmove ≥ 150) | 50-move (halfmove ≥ 100) |
| Insufficient material | Not checked | Implemented (K vs K, K+minor vs K, KB vs KB same color) |
| Action encoding directions | E, NE, N, NW, W, SW, S, SE | N, NE, E, SE, S, SW, W, NW |
| Underpromo order | R, B, N | N, B, R |

These are independent design choices — both are AlphaZero-style 4672-action spaces, just with different conventions. Action indices are NOT interchangeable between the two, so a model trained on one engine can't be loaded on the other without remapping.

---

## 7. What we'd port next

Ordered by EV-per-effort:

1. **Switch to `[B, 64] int8` board representation** — simpler code, GPU-friendlier, and unlocks the matmul-attack and Triton paths below. Cost: large refactor of `chess_gpu.py`. (Maybe not worth it given current speedups, but worth knowing as the alternative.)

2. **Matmul-as-lookup for jump-piece attacks** — even keeping bitboards, we could `(P.float() @ KNIGHT_TBL.float()) > 0` and probably gain a few percent on jump pieces. Tiny code change, easy A/B test.

3. **Hand-write `step_batch` and `legal_mask` as Triton kernels** — the friend's bench shows ~30× over a clean PyTorch baseline. Our compiled CUDA bench shows we're already getting most of that via `torch.compile`. The remaining gap is probably 2–3×, which translates to bigger N being viable for parallel self-play. **Real cost: 200–500 LOC of Triton, plus debugging/profiling.**

4. **Threefold detection on GPU** — the friend skipped this entirely; we have it. If we move to Triton we'd want to keep our Zobrist/ring-buffer approach (port to a `tl.gather`-based scan). Don't drop the feature.

5. **The compact-pin / between-bb / packed-bb tricks** — these are all ~5–10 line code changes once the Triton kernel exists. Each is worth 1.5–2× per the bench. Land progressively.

---

## TL;DR of the lesson

- **Bitboards** are a CPU-era optimization. On GPUs, **a flat `[64]` int8 vector with matmul lookups** is often equally fast and much easier to fuse.
- **Kernel launch overhead is the enemy.** A 40-op PyTorch graph at N=128 burns ~300 µs in launch overhead alone. One Triton kernel pays that overhead **once**. This is the big lever (8.7× in the friend's bench).
- **`torch.compile` gets you most of the way.** The friend's bench [1]→[2] is 8.7×. Our bench shows compiled CUDA at ~3–4× over PyTorch eager. The remaining 2–3× is what hand-written Triton buys.
- **Memory is the bottleneck, not arithmetic.** Compact pin descriptors, packed bitboards, and chunked planes are all about **fitting into shared memory and registers**. Less data in flight → more warps in flight → faster.
- **Triton is the right tool for "I want one fused kernel that does this pile of conditional state updates."** It's not magic — you have to think about tile sizes, register pressure, and shared-memory budget — but it removes the launch-overhead floor that PyTorch can't escape.
