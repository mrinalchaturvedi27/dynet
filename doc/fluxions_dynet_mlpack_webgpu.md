# Fluxions: DyNet + mlpack Integration Research Analysis
## Phase 3: WebGPU Acceleration

> **Scope**: Concrete research analysis of how DyNet's dynamic autograd engine and
> mlpack's classical-ML toolkit can be combined to build *Fluxions*, and how WebGPU
> can accelerate the resulting system in a third engineering phase.
>
> **No code changes are part of this document.**  The analysis draws directly from
> the architectural notes already present in `doc/mlpack.md` and
> `ARCHITECTURAL_ANALYSIS.md`.

---

## Table of Contents

1. [What Is Fluxions?](#1-what-is-fluxions)
2. [DyNet Capabilities Relevant to Fluxions](#2-dynet-capabilities-relevant-to-fluxions)
3. [mlpack Capabilities Relevant to Fluxions](#3-mlpack-capabilities-relevant-to-fluxions)
4. [Integration Architecture: DyNet ⊕ mlpack](#4-integration-architecture-dynet--mlpack)
   - 4.1 Data-Convention Bridge
   - 4.2 Feature-Pipeline Pattern
   - 4.3 Hybrid Training Loop
   - 4.4 Serialisation Strategy
5. [Concrete Leverage Points per mlpack Layer](#5-concrete-leverage-points-per-mlpack-layer)
6. [Phase 3 – WebGPU Acceleration](#6-phase-3--webgpu-acceleration)
   - 6.1 What WebGPU Adds
   - 6.2 DyNet Execution Engine → WebGPU Compute
   - 6.3 mlpack Linear Algebra → WebGPU BLAS Kernels
   - 6.4 Memory Layout Alignment
   - 6.5 Implementation Roadmap
7. [Risk Register](#7-risk-register)
8. [Decision Matrix](#8-decision-matrix)
9. [Global Invariants to Preserve](#9-global-invariants-to-preserve)

---

## 1. What Is Fluxions?

The term *fluxions* is Newton's original word for **instantaneous rates of change**
(i.e., derivatives).  In this engineering context the name signals intent:
a differentiable-programming system that:

* Builds **dynamic computation graphs** whose topology can change per sample
  (from DyNet's design philosophy).
* Draws on **classical ML primitives** for data processing, spatial queries,
  distributions, and optimisation (from mlpack's design philosophy).
* Targets **broad hardware** including GPU-less browsers and edge devices via
  WebGPU (Phase 3).

Throughout `doc/mlpack.md` every `NOTES FOR FLUXIONS:` annotation marks where an
architectural decision in mlpack is load-bearing for any reimplementation, and those
notes collectively define the invariants Fluxions must respect.

---

## 2. DyNet Capabilities Relevant to Fluxions

The following table maps DyNet's internal components to concrete Fluxions needs.
See `ARCHITECTURAL_ANALYSIS.md` for the full per-file analysis.

| DyNet Component | File(s) | What Fluxions Gets |
|---|---|---|
| **ComputationGraph** | `dynet/dynet.h/.cc` | Per-sample graph reset (`renew_cg`); DAG stored in strict topological order; one-active-graph-per-thread guarantee |
| **Expression API** | `dynet/expr.h/.cc` | Composable differentiable operations; graph-id staleness detection prevents silent correctness bugs |
| **Tensor / Dim** | `dynet/tensor.h/.cc`, `dim.h` | CPU-backed (Eigen) dense arrays with batch broadcasting; zero-copy aliasing into sub-tensors |
| **SimpleExecutionEngine** | `dynet/exec.h/.cc` | Forward/backward in topological order; gradient **accumulation** (not replacement) for DAG fan-in |
| **BatchedExecutionEngine** | `dynet/exec.h/.cc` | Automatic batching of structurally identical sub-graphs (v2 feature) |
| **AlignedMemoryPool** | `dynet/aligned-mem-pool.h` | Linear allocation + bulk-free after each graph reset; eliminates per-tensor `malloc`/`free` |
| **ParameterCollection / Parameter** | `dynet/model.h/.cc` | Parameters survive graph resets; sparse update via `LookupParameter` for embedding tables |
| **RNN Builders** | `dynet/rnn.h`, `lstm.h`, `gru.h` | Pre-built recurrent cells that compose with the graph; BPTT is automatic |
| **Trainers** | `dynet/training.h/.cc` | 9 optimisers (SGD, Adam, AdaGrad, RMSProp, AMSGrad, …) with weight decay and gradient clipping |
| **Python Bindings** | `python/` | Rapid prototyping; the C++ core runs unchanged underneath |

### Key DyNet Design Decisions to Carry Forward

1. **Graph reset at sample boundary** – `renew_cg()` resets the computation
   graph and returns all node memory to the pool in O(1).  Fluxions must preserve
   this to achieve DyNet-level throughput on variable-structure inputs.

2. **Parameters are graph-external** – `ParameterCollection` is not destroyed
   with the graph.  This strict separation makes gradient accumulation across
   micro-batches trivial.

3. **Gradient accumulation semantics** – `exec.cc` uses `+=` not `=` when
   writing gradients.  This is the correct DAG semantics and must not change.

4. **No Eigen temporary allocation** – `EIGEN_NO_MALLOC` is enforced at
   compile-time.  Any WebGPU or mlpack integration must not reintroduce heap
   allocations on the hot path.

---

## 3. mlpack Capabilities Relevant to Fluxions

The full reference is `doc/mlpack.md`.  The table below selects only the
components whose `NOTES FOR FLUXIONS:` annotations identify them as directly
load-bearing.

| mlpack Component | File(s) | Why Fluxions Needs It |
|---|---|---|
| **base.hpp / config.hpp** | `src/mlpack/base.hpp`, `config.hpp` | Linear-algebra backend swap point; `mlpack_force_inline` materially affects hot-path throughput |
| **Armadillo arma_extend** | `core/arma_extend/find_nan.hpp` | NaN detection for safe gradient computations (no equivalent in Armadillo core) |
| **cereal serialisation** | `core/arma_extend/serialize_armadillo.hpp` | Model save/load; any custom GPU tensor type needs its own cereal specialisation here |
| **math utilities** | `core/math/math.hpp` | Reproducible RNG (`math::RandomSeed`); log-sum-exp; digamma – all needed for probabilistic layers |
| **make_alias / unwrap_alias** | `core/math/make_alias.hpp` | Zero-copy Armadillo alias into layer weight vectors; the "single flat weight vector → aliased layers" pattern is the ANN memory layout |
| **data::Load / data::Save** | `core/data/data.hpp` | Column-major I/O; auto-detect file format; the `n_features × n_points` convention is a hard invariant |
| **DatasetMapper** | `core/data/dataset_mapper.hpp` | Categorical → integer encoding before feeding into DyNet embeddings |
| **Scaler methods** | `core/data/scaler_methods/` | Fit-then-transform preprocessing; inverse-transform is exact |
| **Binary space trees (kd-tree)** | `core/tree/binary_space_tree/` | Sub-linear kNN for retrieval-augmented pipelines; policy-class design (swap bounds/split-rules as templates) |
| **DualTreeTraverser** | `core/tree/binary_space_tree/dual_tree_traverser.hpp` | Branch-and-bound pruning responsible for kNN sub-linear scaling; do not replace with brute force until v2 |
| **Kernels** | `core/kernels/kernels.hpp` | Duck-typed kernel concept; any class with `Evaluate(a,b)` works; FastMKS for maximum kernel search |
| **FFN / layer contract** | `methods/ann/ffn.hpp`, `layer/layer.hpp` | "Weights live outside the layer" invariant; ensmallen optimizer integration via `EnsembleFunction` |
| **NSModel (kNN)** | `methods/neighbor_search/` | Serialises tree + algorithm choice together; cross-language round-trip |
| **Python binding transpose** | `bindings/python/mlpack_main.hpp` | Automatic row-major ↔ column-major transpose at the Python boundary; has copy cost for large datasets |
| **Log::Fatal / Log::Warn** | `core/util/log.hpp` | Error-reporting mechanism; unrecoverable conditions call `std::exit()` via `Log::Fatal` |

### Key mlpack Design Decisions to Carry Forward

1. **Column-major convention** – `arma::mat` is column-major; one data point per
   column (`n_features × n_points`).  Every DyNet ↔ mlpack data transfer must
   respect or explicitly transpose this.

2. **Weights live outside layers** – `FFN`/`RNN` allocates one flat weight vector;
   each `Layer` receives an alias via `make_alias`.  Breaking this invariant
   destroys single-allocation gradient storage.

3. **Policy / strategy via templates** – trees, kernels, distances, optimisers,
   loss functions are template parameters, not virtual bases.  Fluxions should
   adopt the same pattern for its own pluggable components.

4. **ensmallen is the optimiser substrate** – mlpack algorithms implement
   `Evaluate()` + `Gradient()` and pass themselves to an ensmallen optimiser.
   Fluxions can reuse this interface for any component that does not need
   DyNet's dynamic graph.

---

## 4. Integration Architecture: DyNet ⊕ mlpack

### 4.1 Data-Convention Bridge

The single most important integration concern is **memory layout**:

```
DyNet tensors (Eigen, row-major internally)
        ↕  explicit transpose / zero-copy alias
Armadillo matrices (column-major)
```

The `core/math/make_alias.hpp` mechanism lets Fluxions create a zero-copy
Armadillo alias backed by DyNet's pool-allocated memory **only when the strides
match**.  When they do not match (e.g., a batch dimension), an explicit copy is
required.  This is the same cost already paid at mlpack's Python binding boundary.

**Guideline**: expose a `FluxionsTensor` adapter that carries both a DyNet
`Tensor` reference and an `arma::mat` alias.  Construction is O(1) when strides
align; it forces a copy otherwise.  All Fluxions APIs accept this adapter, routing
to DyNet or mlpack internals as needed.

### 4.2 Feature-Pipeline Pattern

A typical Fluxions forward pass has three zones:

```
[Raw data]
    ↓ mlpack data::Load (column-major I/O)
    ↓ mlpack scaler (StandardScaler / MinMaxScaler)
    ↓ mlpack DatasetMapper (categorical → integer)
[Preprocessed Armadillo matrix]
    ↓ FluxionsTensor bridge (transpose if needed)
[DyNet ComputationGraph]
    ↓ dy.inputTensor / dy.lookup
    ↓ … model forward pass …
    ↓ loss.backward()
    ↓ trainer.update()
[Trained parameters in ParameterCollection]
    ↓ mlpack cereal serialisation (model save)
[Checkpoint on disk]
```

This pipeline ensures:

* mlpack owns **data ingestion and classical preprocessing**.
* DyNet owns **differentiable computation and gradient flow**.
* Serialisation is handled by mlpack's cereal infrastructure (which can already
  round-trip Armadillo matrices and is extensible to DyNet tensors by adding a
  cereal specialisation as noted in `doc/mlpack.md` §2).

### 4.3 Hybrid Training Loop

For models that mix a classical mlpack head (e.g., a kNN retriever) with a DyNet
neural encoder:

```
for each mini-batch:
    dy.renew_cg()                       # DyNet: reset graph
    enc = encoder_forward(batch)        # DyNet: dynamic graph
    enc_arma = bridge.to_arma(enc)      # Bridge: DyNet → Armadillo
    knn_scores = knn.Search(enc_arma)   # mlpack: kNN retrieval
    scores_dy = bridge.to_dynet(knn_scores)  # Bridge: Armadillo → DyNet
    loss = loss_fn(scores_dy, labels)   # DyNet: differentiable loss
    loss.backward()
    trainer.update()                    # DyNet: update encoder params
```

The kNN retrieval is **not differentiable** through mlpack; only the encoder
gradients propagate.  For differentiable retrieval (MIPS, learned metrics),
Fluxions would need to implement the kNN gradient natively in DyNet's node system.

### 4.4 Serialisation Strategy

DyNet uses its own binary format (`dynet::TextFileLoader`/`BinaryLoader`) for
`ParameterCollection`.  mlpack uses cereal.  Fluxions should adopt a **two-file
checkpoint**:

* `checkpoint.dynet` – DyNet `ParameterCollection` (encoder, decoder weights)
* `checkpoint.mlpack` – mlpack cereal archive (kNN tree, scaler state, mapper)

Both files are written atomically (write to `.tmp`, then rename) to prevent
half-written checkpoints.

---

## 5. Concrete Leverage Points per mlpack Layer

This section maps each `doc/mlpack.md` architectural layer to a concrete Fluxions
use case, guided by the `NOTES FOR FLUXIONS:` annotations.

### Layer 1 – Foundation

| File | Fluxions Use |
|---|---|
| `base.hpp` | If Fluxions swaps Armadillo for a GPU-native tensor (Phase 3), this is the **only** file to change.  Keep `mlpack_force_inline` to preserve hot-path throughput. |
| `config.hpp` | Add `FLUXIONS_HAS_WEBGPU` flag here (Phase 3). |
| `prereqs.hpp` | Keep `base.hpp` (no cereal) / `prereqs.hpp` (cereal) split; use `base.hpp` in environments without cereal (e.g., WebAssembly). |

### Layer 2 – Serialisation

| File | Fluxions Use |
|---|---|
| `serialize_armadillo.hpp` | Any custom Fluxions tensor type must add a cereal specialisation here to participate in model persistence. |
| `low_precision.hpp` | Use for compact on-device checkpoints (e.g., float16 weights for browser deployment). |

### Layer 3 – Utilities

| File | Fluxions Use |
|---|---|
| `log.hpp` | Replace `NullOutStream` with a thread-local async buffer for Fluxions' async training mode. |
| `params.hpp` | Reuse as the binding handshake point for any new Fluxions language binding (WASM, Swift, Kotlin). |

### Layer 4 – Mathematics

| File | Fluxions Use |
|---|---|
| `math.hpp` | `math::RandomSeed(seed)` for reproducible experiments; log-sum-exp for numerical stability in attention. |
| `make_alias.hpp` | Zero-copy weight sharing between the DyNet tensor pool and mlpack layer parameters – the core memory bridge. |

### Layer 5 – Data

| File | Fluxions Use |
|---|---|
| `data.hpp` | All dataset I/O; respect column-major hard invariant throughout Fluxions. |
| `dataset_mapper.hpp` | Encode categorical features before passing to DyNet's `LookupParameter`. |
| `scaler_methods/` | Fit scalers on training split; apply identically to test/inference.  Inverse-transform predictions for interpretability. |
| `split_data.hpp` | Reproducible train/test splits via `math::RandomSeed`. |

### Layer 6 – Spatial Indices

| File | Fluxions Use |
|---|---|
| `binary_space_tree/` | kNN retrieval over encoder output embeddings.  The policy-class design lets Fluxions swap L2 distance for a learned Mahalanobis distance without changing the tree. |
| `dual_tree_traverser.hpp` | Responsible for sub-linear kNN scaling; do not replace with brute-force batch matrix multiply until dataset ≤ 10 K points. |

### Layer 7 – Kernels

| File | Fluxions Use |
|---|---|
| `kernels.hpp` | Gaussian kernel for RBF attention weights; FastMKS for maximum inner-product search over vocabulary embeddings. |

### Layer 9 – ANN Framework

| File | Fluxions Use |
|---|---|
| `layer.hpp` | If Fluxions needs a pure-mlpack (non-DyNet) sub-network (e.g., a tabular head), implement it as a `Layer<MatType>` to preserve the flat-weight-vector invariant. |
| `ffn.hpp` | Use for tabular / structured-data sub-networks; pass to any ensmallen optimiser. |
| `loss_functions/` | Share loss semantics (MSE, cross-entropy) between DyNet and mlpack layers for consistent evaluation. |

### Layer 10 – Classical ML

| Method | Fluxions Use |
|---|---|
| `linear_regression` | Baseline; interpretability probes on encoder representations. |
| `random_forest` | Feature importance analysis; stacking head on top of DyNet encoder. |
| `gmm` | Density estimation on latent space; anomaly detection. |
| `hmm` | Sequence labelling where DyNet is too expensive; emission priors for structured prediction. |
| `kmeans` | Cluster encoder embeddings for pseudo-label generation; centroid initialisation for retrieval index. |
| `pca` | Dimensionality reduction for visualisation; whitening preprocessing. |

### Layer 11 – RL Framework

| File | Fluxions Use |
|---|---|
| `q_learning.hpp` | DyNet as the Q-network (dynamic graph per state); mlpack provides the experience replay buffer. |
| `replay/` | Decouples data collection from training; uniform and prioritised replay both available. |

---

## 6. Phase 3 – WebGPU Acceleration

> **Prerequisite**: Phases 1 and 2 (DyNet + mlpack integration) are stable and
> tested.  Phase 3 is a performance optimisation; it must not change observable
> semantics.

### 6.1 What WebGPU Adds

WebGPU is the W3C successor to WebGL for general-purpose GPU compute.  Unlike
CUDA or Metal, WebGPU:

* Runs in **browsers** (Chrome, Firefox, Safari) via the `GPUDevice` API.
* Is available as a **native C/C++ backend** via `wgpu` (Rust) or `dawn` (Google
  C++) outside the browser.
* Uses **WGSL** (WebGPU Shading Language) for compute shaders – a structured,
  statically-typed language that avoids the pitfalls of raw GLSL/HLSL.
* Provides **compute pipelines** with explicit workgroup sizing, storage buffers,
  and uniform buffers – sufficient for all BLAS-level operations.

For Fluxions the value proposition is:

| Aspect | Current (CPU) | With WebGPU |
|---|---|---|
| Matrix multiply (FFN forward) | Eigen/BLAS on CPU | WGSL matmul kernel, parallelised across GPU cores |
| Batch kNN search | Dual-tree traversal on CPU | Brute-force WGSL kernel competitive up to ~10 M points |
| Memory bandwidth | DDR4/5 | GPU VRAM (≥ 4× bandwidth on discrete GPU) |
| Portability | x86/ARM native | Browser + native via `dawn`/`wgpu` |
| Programming model | C++ threads / OpenMP | WGSL workgroups + storage buffers |

### 6.2 DyNet Execution Engine → WebGPU Compute

DyNet's `SimpleExecutionEngine` dispatches each node's `forward()` and
`backward()` sequentially on CPU.  The Phase 3 integration introduces a
`WebGPUExecutionEngine` that:

1. **Analyses the computation graph** after `renew_cg()` to identify sub-graphs
   of WebGPU-acceleratable nodes (matmul, element-wise ops, reductions).
2. **Allocates GPU buffers** once per graph reset via a `GPUMemoryPool` that
   mirrors DyNet's `AlignedMemoryPool` design.  Linear allocation + bulk-free
   preserves the O(1) reset invariant.
3. **Records a `GPUCommandEncoder`** with `dispatchWorkgroups()` calls for each
   node batch.  Nodes with data dependencies between them are separated by
   pipeline barriers.
4. **Submits the command buffer** to the GPU queue.  Results are read back to
   CPU only when `Expression::value()` is called explicitly (lazy readback).
5. **Falls back** to `SimpleExecutionEngine` for nodes without a WebGPU kernel
   (e.g., custom user-defined nodes).

**Critical invariants preserved**:

* Gradient accumulation (`+=`) semantics are enforced in the WGSL backward
  kernels via `atomicAdd` on `f32` storage buffers (WebGPU 2 adds native float
  atomics; WebGPU 1 requires `i32` tricks or serialised workgroups).
* `EIGEN_NO_MALLOC` spirit is preserved: no GPU allocations on the forward path;
  only the pool is used.
* The `graph_id` staleness mechanism remains CPU-side; it is not moved to the
  GPU.

### 6.3 mlpack Linear Algebra → WebGPU BLAS Kernels

mlpack's `base.hpp` is the single backend-swap point noted in `doc/mlpack.md` §1.
Phase 3 introduces a conditional compilation block:

```
#ifdef FLUXIONS_HAS_WEBGPU
  // Replace Armadillo BLAS calls with WebGPU storage-buffer matmul
  #include <fluxions/webgpu_arma_backend.hpp>
#else
  #include <armadillo>
#endif
```

The `webgpu_arma_backend.hpp` provides drop-in replacements for the Armadillo
matrix operations used on mlpack's hot paths:

| Armadillo Operation | WGSL Replacement | Notes |
|---|---|---|
| `C = A * B` (gemm) | `wgpu_sgemm(A, B, C)` | Tiled shared-memory matmul; tile size tuned per GPU |
| `arma::norm(v)` | `wgpu_norm_reduction(v)` | Two-pass reduction; first pass computes partial sums per workgroup |
| `arma::sum(A, 1)` | `wgpu_row_sum(A)` | One compute dispatch per column block |
| `arma::max(A)` | `wgpu_max_reduction(A)` | Tree reduction; matches standard parallel reduce pattern |

**The column-major invariant** is preserved in GPU buffers: each matrix is stored
column-by-column in a `GPUBuffer` exactly as Armadillo does in CPU memory.  This
means the transpose already applied at the Python binding boundary is **not**
re-applied at the GPU boundary.

### 6.4 Memory Layout Alignment

The critical insight for Phase 3 memory management:

```
DyNet AlignedMemoryPool (CPU)
    → slab of aligned CPU pages
    → zero-copy GPU upload via wgpuQueueWriteBuffer()
    → GPUBuffer (VRAM)
    → WGSL compute shader reads/writes
    → wgpuBufferMapAsync() readback (lazy, only when needed)
```

For the DyNet-side, `AlignedMemoryPool`'s 32-byte CPU alignment satisfies
WebGPU's buffer alignment requirement (minimum 4 bytes; 256 bytes for uniform
buffers).  Only storage buffers are needed for tensor data, so the 32-byte
alignment is sufficient.

For the mlpack-side, Armadillo's column-major storage is directly uploadable to
a `GPUBuffer` with `COPY_SRC | STORAGE` usage flags.  The `make_alias` zero-copy
pattern works on CPU; for GPU the alias becomes a `GPUBuffer` view (buffer offset
+ size) which WebGPU supports natively via `setBindGroup` with buffer offsets.

### 6.5 Implementation Roadmap

| Sub-phase | Deliverable | Prerequisite |
|---|---|---|
| **3.0** | `GPUDevice` initialisation + buffer pool; WGSL matmul kernel; element-wise ReLU, sigmoid, tanh kernels | Phase 2 complete |
| **3.1** | `WebGPUExecutionEngine` for DyNet (forward only); fallback to CPU for unsupported nodes | 3.0 |
| **3.2** | Backward pass kernels (gradient accumulation via atomic add); WebGPU trainer update step | 3.1 |
| **3.3** | mlpack `webgpu_arma_backend.hpp`; FFN forward/backward entirely on GPU | 3.2 |
| **3.4** | Brute-force WebGPU kNN kernel (competitive with dual-tree below 1 M points); toggle via feature flag | 3.3 |
| **3.5** | WebAssembly + WebGPU browser build via Emscripten; lazy CPU readback; reduced-precision (f16) storage | 3.4 |

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Column-major ↔ row-major confusion at DyNet/mlpack boundary | High | Data corruption, wrong results | `FluxionsTensor` adapter enforces explicit transpose; add assert on shape |
| Gradient accumulation bug when porting to WebGPU atomics | Medium | Silent numerical errors | Compare CPU and GPU results on fixed seed; maintain CPU golden reference |
| WebGPU float32 atomic unavailability in WebGPU 1 | High (browser) | Incorrect backward pass | Use i32 bit-cast workaround (IEEE 754) or serialise per-element workgroups |
| `mlpack_force_inline` lost after backend swap | Low | 10–30% throughput regression | Keep the macro in `base.hpp`; add benchmark gate in CI |
| Cereal round-trip broken for custom Fluxions tensors | Medium | Checkpoints unreadable | Add cereal specialisation test in Phase 1; run it on every checkpoint change |
| kNN tree invalidated when encoder weights change | High | Stale retrieval index | Rebuild tree after each optimizer epoch or use HNSW with incremental updates |
| WebGPU not available at runtime (old browser / driver) | High (browser) | Feature unavailable | Graceful fallback to CPU engine; report via `config.hpp` `FLUXIONS_HAS_WEBGPU` flag |

---

## 8. Decision Matrix

| Architectural Decision | Option A | Option B | Recommendation |
|---|---|---|---|
| Tensor bridge (DyNet ↔ mlpack) | Explicit copy every time | Zero-copy alias when strides align | **Zero-copy alias**; copy only when strides misalign |
| Optimiser for classical mlpack layers | DyNet trainer | ensmallen | **ensmallen** for mlpack layers; DyNet trainer for autograd layers |
| Serialisation format | Unified (one cereal archive) | Two-file (DyNet + mlpack) | **Two-file** to avoid tying DyNet's binary format to cereal versioning |
| kNN during training | Rebuild per epoch | Incremental HNSW | **Rebuild per epoch** for v0/v1; HNSW for v2 when index rebuilds become bottleneck |
| WebGPU matmul tile size | Fixed 16×16 | Auto-tuned per GPU | **Auto-tuned** (WebGPU `maxComputeWorkgroupSizeX` query at init) |
| WebGPU fallback | Abort | CPU fallback | **CPU fallback** always; never abort for missing GPU |
| Weight precision in browser (Phase 3.5) | float32 | float16 | **float32** by default; opt-in float16 via `low_precision.hpp` equivalent |

---

## 9. Global Invariants to Preserve

Drawn from `doc/mlpack.md` §16 and `ARCHITECTURAL_ANALYSIS.md`, these invariants
must survive every integration phase:

1. **Column-major data** (`n_features × n_points`) – enforced in all Armadillo
   paths; explicitly transposed at every language-binding boundary.

2. **Weights live outside layers** – mlpack ANN layers receive `make_alias`
   views into a flat weight vector; DyNet parameters are owned by
   `ParameterCollection`, not the graph.  Both must remain true in Fluxions.

3. **Graph reset is O(1)** – `AlignedMemoryPool::free()` frees the entire slab
   in one call.  Phase 3's `GPUMemoryPool` must replicate this with a single
   `wgpuBufferDestroy()` or pool reset rather than per-tensor deallocations.

4. **Gradient accumulation (`+=`)** – DyNet's `exec.cc` and any WebGPU backward
   kernel must use accumulation semantics, not assignment.

5. **Policy / strategy via templates** – trees, kernels, distances, split rules
   remain template parameters.  No virtual dispatch on the hot path (except the
   existing `Layer` polymorphism in mlpack's `MultiLayer`).

6. **ensmallen is the mlpack optimiser substrate** – mlpack algorithms expose
   `Evaluate()` + `Gradient()`.  Fluxions does not reimplement mlpack optimisers.

7. **Log::Fatal is the unrecoverable-error mechanism** in mlpack code paths.
   Fluxions should not silently swallow `Log::Fatal` conditions by wrapping them
   in try/catch.

8. **Serialisation is mandatory** – every public Fluxions model type must
   implement cereal `serialize()` (mlpack side) and DyNet `save()`/`load()` (DyNet
   side) to support the two-file checkpoint strategy.

---

*Document version: 1.0 | Analysis basis: `doc/mlpack.md` (mlpack architecture),
`ARCHITECTURAL_ANALYSIS.md` (DyNet architecture) | No source code was modified.*
