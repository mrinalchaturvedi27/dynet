# DyNet Reverse Engineering Documentation

This directory contains comprehensive architectural analysis documentation for rebuilding DyNet from scratch.

## 📚 Documentation Overview

Three complementary documents provide everything needed to rebuild DyNet:

### 1. [ARCHITECTURAL_ANALYSIS.md](./ARCHITECTURAL_ANALYSIS.md) (1583 lines)
**Purpose**: Deep architectural analysis of every component

**Structure**: File-by-file breakdown following strict format:
- **ROLE**: Single-sentence responsibility
- **DEPENDS ON**: Upstream dependencies
- **USED BY**: Downstream consumers
- **CORE OR AUX**: Classification (CP/OC/PO/DE/DV/EO)
- **INVARIANTS**: Critical assumptions
- **REBUILD PHASE**: v0/v1/v2
- **WHAT BREAKS**: Impact of removal
- **NOTES FOR REBUILDING**: Implementation guidance

**Coverage**:
- ✅ Core computation engine (graph, expressions, execution)
- ✅ Memory management (allocators, pools, devices)
- ✅ Node implementations (50+ operation types)
- ✅ RNN infrastructure (LSTM, GRU, TreeLSTM)
- ✅ Training system (optimizers, initialization, regularization)
- ✅ I/O and utilities (serialization, dictionaries, timing)
- ✅ GPU/CUDA infrastructure (kernels, cuDNN)
- ✅ Python bindings
- ✅ Build system and tests

**When to use**: Reference when implementing specific components; understand dependencies and invariants.

---

### 2. [REBUILD_GUIDE.md](./REBUILD_GUIDE.md) (559 lines)
**Purpose**: Practical rebuild roadmap with implementation details

**Structure**:
- **Executive Summary**: One-sentence architecture description
- **Rebuild Strategy**: Bottom-up dependency order
- **Phase 0-2**: Week-by-week implementation plan
- **Critical Implementation Details**: Common pitfalls and patterns
- **Testing Strategy**: Validation approach
- **Performance Checklist**: Optimization verification
- **Debugging Tools**: Diagnostic utilities
- **Decision Points**: When to use DyNet vs alternatives

**Coverage**:
- ✅ 3-month rebuild timeline (1 engineer)
- ✅ Phased milestones (XOR → LSTM LM → SOTA)
- ✅ Critical implementation patterns (gradient accumulation, memory pools)
- ✅ Common pitfalls (batch dimensions, Eigen temporaries)
- ✅ Testing strategy (correctness → features → performance)
- ✅ MVP checklists (2K LOC → 5K LOC → 15K LOC)
- ✅ Architecture comparison (DyNet vs PyTorch vs TensorFlow)

**When to use**: Starting rebuild project; planning implementation order; debugging common issues.

---

### 3. [DEPENDENCY_MAP.md](./DEPENDENCY_MAP.md) (900+ lines)
**Purpose**: Visual dependency relationships and execution flows

**Structure**:
- **Component Index**: Alphabetical quick reference (50+ components)
- **Dependency Graph**: Layer-by-layer bottom-up visualization
- **Execution Flow Diagram**: Forward/backward pass step-by-step
- **Memory Layout Diagram**: Pool organization and lifecycle
- **File Organization**: Directory structure explanation
- **Detailed Rebuild Checklist**: Day-by-day tasks

**Coverage**:
- ✅ 9-layer dependency hierarchy (foundation → high-level)
- ✅ Critical path diagrams (forward pass, backward pass)
- ✅ Memory architecture visualization
- ✅ File-to-component mapping
- ✅ 12-week detailed checklist

**When to use**: Understanding system architecture; visualizing dependencies; planning rebuild order.

---

## 🚀 Quick Start: How to Use These Docs

### Scenario 1: "I want to rebuild DyNet from scratch"
1. **Start**: Read `REBUILD_GUIDE.md` executive summary
2. **Plan**: Follow `DEPENDENCY_MAP.md` Layer 0-9 dependency graph
3. **Implement**: Use `ARCHITECTURAL_ANALYSIS.md` for each component
4. **Validate**: Follow testing strategy in `REBUILD_GUIDE.md`

**Timeline**: 3 months (1 engineer) or 1 month (team of 3)

---

### Scenario 2: "I need to understand how DyNet works"
1. **Overview**: Read `REBUILD_GUIDE.md` executive summary + architecture comparison
2. **Core concepts**: Study `DEPENDENCY_MAP.md` execution flow diagram
3. **Deep dive**: Read `ARCHITECTURAL_ANALYSIS.md` core components:
   - ComputationGraph (dynet.h/cc)
   - Expression (expr.h/cc)
   - Tensor (tensor.h/cc)
   - ExecutionEngine (exec.h/cc)

**Time**: 2-3 hours for solid understanding

---

### Scenario 3: "I'm implementing a specific feature (e.g., custom node)"
1. **Locate**: Find component in `DEPENDENCY_MAP.md` component index
2. **Understand**: Read corresponding section in `ARCHITECTURAL_ANALYSIS.md`
3. **Dependencies**: Check `DEPENDS ON` and `USED BY` fields
4. **Patterns**: Follow `nodes-def-macros.h` implementation pattern
5. **Test**: Use gradient checking from `REBUILD_GUIDE.md`

**Time**: 30 minutes to understand, hours to implement

---

### Scenario 4: "I'm debugging an issue"
1. **Common issues**: Check `REBUILD_GUIDE.md` → Common Pitfalls
2. **Tools**: Use debugging tools from `REBUILD_GUIDE.md`
3. **Invariants**: Verify invariants from `ARCHITECTURAL_ANALYSIS.md`
4. **Flow**: Trace execution in `DEPENDENCY_MAP.md` flow diagrams

**Time**: Varies by issue complexity

---

## 📊 Documentation Statistics

| Document | Lines | Components | Diagrams | Est. Read Time |
|----------|-------|------------|----------|----------------|
| ARCHITECTURAL_ANALYSIS.md | 1,583 | 60+ | 0 | 3-4 hours |
| REBUILD_GUIDE.md | 559 | Summary | Tables | 1-2 hours |
| DEPENDENCY_MAP.md | 900+ | 50+ | 5+ | 1-2 hours |
| **TOTAL** | **3,042+** | **60+** | **5+** | **5-8 hours** |

---

## 🎯 Key Insights (TL;DR)

### Core Architecture
**DyNet in one sentence**: Dynamic computation graph framework with linear memory allocation, topological execution, and automatic differentiation, optimized for NLP research.

### Key Innovation
**Linear allocation + bulk freeing**: AlignedMemoryPool enables fast graph construction/destruction without fragmentation.

### Critical Components (Must rebuild in v0)
1. **Memory**: CPUAllocator → AlignedMemoryPool → Device (4 pools)
2. **Data**: Dim → Tensor (Eigen wrapper)
3. **Graph**: Node → ComputationGraph → Expression
4. **Execution**: SimpleExecutionEngine (forward/backward)
5. **Parameters**: Parameter → ParameterCollection
6. **Training**: SimpleSGDTrainer

### Most Complex Component
**BatchedExecutionEngine**: Autobatching requires sophisticated graph analysis and memory checkpointing.

### Most Delicate Component
**Gradient accumulation**: Must use `+=` not `=` in backward pass (DAG fan-in).

### Biggest Performance Win
**AlignedMemoryPool**: Linear allocation is 10-100× faster than malloc/free for graph workloads.

### Architecture vs Competitors

| Feature | DyNet | PyTorch | TensorFlow 2 |
|---------|-------|---------|--------------|
| Graph | Dynamic (per-example) | Dynamic | Dynamic (eager) |
| Memory | Pooled linear | Caching | BFC allocator |
| Batching | Automatic | Manual | Manual |
| Focus | NLP/Research | General | Production |

---

## 📖 Reading Order

### For Builders (Implementing DyNet clone)
1. ✅ **REBUILD_GUIDE.md** → Executive Summary (5 min)
2. ✅ **DEPENDENCY_MAP.md** → Dependency Graph (15 min)
3. ✅ **ARCHITECTURAL_ANALYSIS.md** → Core Computation Engine (30 min)
4. ✅ **REBUILD_GUIDE.md** → Phase 0 checklist (10 min)
5. ✅ Start coding! Reference **ARCHITECTURAL_ANALYSIS.md** per component

### For Learners (Understanding DyNet)
1. ✅ **REBUILD_GUIDE.md** → Executive Summary + Architecture Comparison (10 min)
2. ✅ **DEPENDENCY_MAP.md** → Execution Flow Diagram (15 min)
3. ✅ **ARCHITECTURAL_ANALYSIS.md** → Core components (1 hour)
4. ✅ Study examples in `examples/` directory
5. ✅ Reference docs as needed for specific questions

### For Contributors (Adding features)
1. ✅ **DEPENDENCY_MAP.md** → Component Index (find relevant files)
2. ✅ **ARCHITECTURAL_ANALYSIS.md** → Read specific component section
3. ✅ **REBUILD_GUIDE.md** → Check common pitfalls
4. ✅ Implement feature
5. ✅ Use gradient checking and tests from **REBUILD_GUIDE.md**

---

## 🔧 Rebuild Milestones

### v0: Minimal Viable Framework (~2 weeks, ~2K LOC)
**Goal**: Train XOR

**Components**:
- Memory management (allocators, pools, devices)
- Tensor abstraction (Dim, Tensor, Eigen)
- Graph infrastructure (Node, ComputationGraph, Expression)
- Basic execution (SimpleExecutionEngine)
- Basic operations (MatMul, Tanh, Softmax, Sum)
- Simple training (SimpleSGDTrainer)

**Success criteria**:
- [x] XOR converges in <100 iterations
- [x] No memory leaks
- [x] Gradients match numerical approximation

---

### v1: Production Features (~4 weeks, ~5K LOC)
**Goal**: Train LSTM language model

**Add**:
- RNN infrastructure (RNNBuilder, LSTM, GRU)
- Sparse parameters (LookupParameter, embeddings)
- Extended operations (full arithmetic, activations, losses)
- Advanced optimizers (Adam, Momentum)
- I/O system (save/load models)
- GPU support (Device_GPU, CUDA kernels)
- Python bindings

**Success criteria**:
- [x] PTB language model perplexity < 100
- [x] GPU 5-10× faster than CPU
- [x] Models save/load correctly

---

### v2: Advanced Features (~6 weeks, ~15K LOC)
**Goal**: Match state-of-the-art

**Add**:
- Autobatching (BatchedExecutionEngine)
- CNN support (Conv2D, Pooling, cuDNN)
- Optimizations (fused ops, hierarchical softmax)
- Tree structures (TreeLSTM)
- Advanced features (weight decay, cyclical LR)

**Success criteria**:
- [x] Autobatch 2-3× speedup
- [x] cuDNN performance competitive
- [x] SOTA results on NLP benchmarks

---

## 🛠️ Tools and Resources

### Documentation Tools Used
- **Code analysis**: grep, ripgrep, ag
- **Dependency tracing**: Manual code reading + ctags
- **Architecture extraction**: Bottom-up component analysis

### Recommended Dev Tools
- **C++ IDE**: CLion, VSCode with C++ extensions
- **Debugger**: gdb, lldb
- **Profiler**: perf, Valgrind, NVIDIA Nsight
- **Build**: CMake, Make
- **Testing**: GoogleTest, custom test harness

### Learning Resources
- **DyNet examples**: `examples/` directory
- **DyNet tests**: `tests/` directory
- **Research papers**: Dynamic computation graphs (DyNet paper)
- **Similar frameworks**: PyTorch (dynamic graphs), Chainer (inspiration)

---

## 🎓 Educational Value

These documents are designed to teach:
- **System architecture**: How to design a deep learning framework
- **Memory management**: Linear allocation patterns
- **Automatic differentiation**: Reverse-mode gradient computation
- **Graph execution**: Topological traversal algorithms
- **Optimization**: Performance tuning strategies

Can be used for:
- ✅ University course on DL frameworks
- ✅ Industrial training on systems programming
- ✅ Research project foundation
- ✅ Interview preparation (systems design)

---

## 🤝 Contributing

If you find errors or want to improve these docs:
1. Read the relevant section
2. Verify against source code
3. Propose corrections via PR
4. Update all affected documents

---

## 📜 License

These documents are derived from analyzing the DyNet codebase, which is licensed under Apache 2.0.

Documentation © 2026, following the same license as DyNet.

---

## 🙏 Acknowledgments

This reverse engineering documentation was created by systematically analyzing the DyNet codebase to extract architectural knowledge for rebuilding purposes.

**DyNet authors**: Graham Neubig, Chris Dyer, Yoav Goldberg, et al.

**Original repository**: https://github.com/clab/dynet

**Analysis date**: February 2026

---

## 📞 Contact

For questions about this documentation:
- Open an issue in the repository
- Reference the specific document and section

For questions about DyNet itself:
- See the original DyNet repository
- Check DyNet documentation

---

**Ready to rebuild? Start with `REBUILD_GUIDE.md` and good luck! 🚀**
