# DyNet Rebuild Guide: Quick Reference

This document complements `ARCHITECTURAL_ANALYSIS.md` with a focused rebuild strategy.

## Executive Summary

**DyNet Architecture in One Sentence**: Dynamic computation graph framework with linear memory allocation, topological execution, and automatic differentiation, optimized for NLP research.

**Core Innovation**: Linear memory allocation with bulk freeing (AlignedMemoryPool) + dynamic graph reconstruction per example.

**Key Differentiator vs PyTorch/TensorFlow**: Dynamic graphs rebuilt per example (like PyTorch eager mode), with aggressive memory pooling optimization.

---

## Rebuild Strategy: Bottom-Up Dependency Order

### Phase 0: Core Primitives (Week 1-2)
Build these in order:

1. **Memory Layer** (foundational)
   ```
   CPUAllocator → InternalMemoryPool → AlignedMemoryPool → Device_CPU (4 pools)
   ```

2. **Data Structures** (parallel to memory)
   ```
   Dim → Tensor (wraps Eigen) → ParameterStorage
   ```

3. **Graph Infrastructure** (depends on above)
   ```
   Node (abstract) → ComputationGraph → Expression (API wrapper)
   ```

4. **Execution** (depends on graph)
   ```
   SimpleExecutionEngine (forward/backward topological traversal)
   ```

5. **Leaf Nodes** (entry points)
   ```
   InputNode → ParameterNode
   ```

6. **Basic Operations** (minimal set)
   ```
   MatrixMultiply, Sum, CwiseMultiply, Tanh, Softmax, PickNegLogSoftmax
   ```

7. **Training** (closes the loop)
   ```
   SimpleSGDTrainer → update() method
   ```

8. **Framework Setup**
   ```
   initialize() → DynetParams → cleanup()
   ```

**Milestone**: Train XOR (2 inputs → 1 hidden layer → 1 output)

---

### Phase 1: Production Features (Week 3-6)

9. **Sparse Parameters**
   ```
   LookupParameterStorage → LookupNode
   ```

10. **RNN Infrastructure**
    ```
    RNNBuilder (abstract) → RNNStateMachine → VanillaLSTMBuilder
    ```

11. **Extended Operations**
    ```
    All arithmetic nodes, activations, losses, concat, select
    ```

12. **Advanced Optimizers**
    ```
    ShadowParameters → AdamTrainer, MomentumSGDTrainer
    ```

13. **I/O System**
    ```
    TextFileSaver, TextFileLoader → model checkpointing
    ```

14. **GPU Support**
    ```
    GPUAllocator → Device_GPU → gpu-ops.cu (custom kernels)
    ```

15. **Python Bindings**
    ```
    Cython wrappers → _dynet.pyx → dynet Python module
    ```

16. **Utilities**
    ```
    Dict, ParameterInit variants, grad_check
    ```

**Milestone**: Train LSTM language model on PTB dataset

---

### Phase 2: Advanced Features (Week 7-12)

17. **Autobatching**
    ```
    BatchedExecutionEngine → automatic minibatching
    ```

18. **CNN Support**
    ```
    Conv2D, MaxPooling2D → cuDNN integration
    ```

19. **Fused Operations**
    ```
    VanillaLSTMGates/C/H, optimized pickneglogsoftmax
    ```

20. **Large Vocabulary Optimizations**
    ```
    ClassFactoredSoftmaxBuilder, HierarchicalSoftmaxBuilder
    ```

21. **Tree Structures**
    ```
    TreeLSTMBuilder variants
    ```

22. **Advanced Features**
    ```
    Weight decay, cyclical learning rates, EMA
    ```

**Milestone**: Match state-of-the-art NLP benchmark results

---

## Critical Implementation Details

### 1. Gradient Accumulation (MUST GET RIGHT)
```cpp
// WRONG - overwrites gradients
dEdxi[arg] = gradient;

// CORRECT - accumulates gradients (DAG has fan-in)
dEdxi[arg] += gradient;
```

**Why**: Single node may be used by multiple downstream nodes; gradients must sum.

---

### 2. Memory Pools (CORE OPTIMIZATION)
```cpp
// 4 pools per device
pools[0]: ForwardMemory   - intermediate forward values
pools[1]: BackwardMemory  - gradient tensors
pools[2]: ParameterMemory - model weights (persistent)
pools[3]: ScratchMemory   - temporary workspace
```

**Pattern**: Allocate linearly during forward pass, bulk free at end of graph.

**Critical**: Parameters persist; everything else freed when graph invalidates.

---

### 3. Topological Order (INVARIANT)
```cpp
// Adding node to graph
nodes.push_back(new_node);  // MUST be topological order
// All node arguments reference indices < nodes.size()
```

**Violation symptom**: Forward pass uses uncomputed values; backward fails.

---

### 4. Expression Staleness (SAFETY)
```cpp
struct Expression {
  ComputationGraph* pg;
  VariableIndex i;
  unsigned graph_id;  // Must match get_current_graph_id()
};
```

**Check**: Every Expression operation validates `graph_id` before use.

**Prevents**: Using expressions from old graphs (dangling pointers).

---

### 5. Batch Dimension (ERGONOMICS)
```cpp
struct Dim {
  unsigned d[7];   // Feature dimensions
  unsigned nd;     // Number of feature dimensions
  unsigned bd;     // Batch dimension (separate!)
};
```

**Why separate**: Simplifies broadcasting; batch_elems() × batch_size() = total elements.

---

## Testing Strategy

### v0 Tests (Correctness)
1. ✅ Forward pass produces correct values (XOR golden outputs)
2. ✅ Backward pass gradients match numerical gradients (finite difference)
3. ✅ SGD reduces loss over iterations
4. ✅ Memory pools free correctly (no leaks)
5. ✅ Topological order maintained

### v1 Tests (Features)
1. ✅ LSTM forward/backward correctness
2. ✅ LookupParameter sparse updates
3. ✅ Adam optimizer momentum/variance updates
4. ✅ Model save/load preserves weights
5. ✅ GPU results match CPU (tolerance 1e-5)

### v2 Tests (Performance)
1. ✅ Autobatching achieves expected speedup
2. ✅ cuDNN convolution faster than manual
3. ✅ Large vocabulary softmax scales
4. ✅ Memory usage stays bounded
5. ✅ Benchmark throughput (examples/sec)

---

## Common Pitfalls

### Pitfall 1: Forgetting Batch Dimension
```cpp
// WRONG - ignores batching
for (int i = 0; i < tensor.d[0]; ++i)

// CORRECT - handles batches
for (int b = 0; b < tensor.d.bd; ++b)
  for (int i = 0; i < tensor.d[0]; ++i)
```

### Pitfall 2: Eigen Temporaries
```cpp
// WRONG - allocates intermediate
result = (A * B) + C;  // Eigen allocates temporary for (A*B)

// CORRECT - pre-allocate or use noalias()
result.noalias() = A * B;
result += C;
```

**DyNet solution**: `EIGEN_NO_MALLOC` compile flag enforces no allocation.

### Pitfall 3: Node Ownership
```cpp
// WRONG - memory leak
ComputationGraph::add_function<MyNode>() {
  return nodes.size();  // Who deletes the node?
}

// CORRECT - graph owns nodes
~ComputationGraph() {
  for (auto* node : nodes) delete node;
}
```

### Pitfall 4: Stale Gradients
```cpp
// WRONG - gradients from previous graph
Parameter p = model.add_parameters({10});
ComputationGraph cg1;
Expression e1 = parameter(cg1, p);
// ... backward on cg1 ...

ComputationGraph cg2;  // New graph!
Expression e2 = parameter(cg2, p);
// p.gradients still has old values!

// CORRECT - zero gradients before backward
trainer.update();  // Applies then zeros gradients
```

---

## Performance Checklist

### Memory Efficiency
- [x] Linear allocation (no fragmentation)
- [x] Bulk freeing (no per-node deallocation)
- [x] Parameter memory persists across graphs
- [x] Forward/backward/scratch pools separate
- [x] Lazy expansion (16MB chunks)

### Computation Efficiency
- [x] Topological execution (no redundant computation)
- [x] Incremental forward (reuse from checkpoints)
- [x] Sparse updates for embeddings
- [x] Gradient clipping before update
- [x] Device dispatch (CPU/GPU specialization)

### Advanced Optimizations
- [x] Autobatching (combine identical subgraphs)
- [x] cuDNN integration (vendor-optimized ops)
- [x] Fused operations (single kernel vs multiple)
- [x] Hierarchical softmax (O(log V) vs O(V))
- [x] SIMD functors (vectorized element-wise ops)

---

## Debugging Tools

### 1. Print Node Info
```cpp
node->as_string(arg_names);  // Human-readable operation
```

### 2. Gradient Checking
```cpp
check_grad(model, loss_expression);  // Finite difference validation
```

### 3. Timing
```cpp
NamedTimer timer("forward");
// ... operations ...
timer.report();  // Cumulative time
```

### 4. Memory Usage
```cpp
device->pools[DeviceMempool::FXS].used();  // Forward memory
```

### 5. Graph Visualization
```cpp
cg.print_graphviz();  // DOT format for visualization
```

---

## Decision Points

### When to Use DyNet
✅ Research prototyping (dynamic graphs)  
✅ NLP tasks (strong RNN support)  
✅ Variable-length sequences  
✅ Rapidly changing model architectures  
✅ Memory-constrained environments  

### When NOT to Use DyNet
❌ Production serving (static graphs faster)  
❌ Computer vision (PyTorch/TF better CNN support)  
❌ Distributed training (limited multi-GPU)  
❌ Large team (PyTorch has bigger ecosystem)  

---

## Architecture Comparison

| Aspect | DyNet | PyTorch | TensorFlow 2 |
|--------|-------|---------|--------------|
| **Graph** | Dynamic (per-example) | Dynamic | Dynamic (eager) |
| **Memory** | Pooled linear allocation | Caching allocator | BFC allocator |
| **Batching** | Automatic (autobatch) | Manual | Manual |
| **NLP Focus** | Strong (RNN builders) | Moderate | Weak |
| **CV Focus** | Weak | Strong | Strong |
| **Adoption** | Research | Industry + Research | Industry |

---

## Key Files Reference (Quick Lookup)

| Need | File |
|------|------|
| Add new operation | Create `nodes-myop.h`, subclass `Node` |
| Add optimizer | Subclass `Trainer`, implement `update_rule` |
| Add RNN variant | Subclass `RNNBuilder` |
| Debug gradients | Use `grad-check.h::check_grad()` |
| Profile performance | Use `timing.h::NamedTimer` |
| Save/load model | Use `io.h::TextFileSaver/Loader` |
| Initialize parameters | Use `param-init.h::ParameterInit*` |
| Manage vocabulary | Use `dict.h::Dict` |
| GPU operation | Add to `gpu-ops.cu` |
| Python API | Edit `python/_dynet.pyx` |

---

## Minimum Viable Product (MVP) Checklist

### Can it train XOR? (MVP-1)
- [x] ComputationGraph with topological order
- [x] Expression API (input, parameter, *, +)
- [x] Tensor + Dim
- [x] SimpleExecutionEngine (forward/backward)
- [x] MatrixMultiply, Tanh, Sum nodes
- [x] SimpleSGDTrainer
- [x] Memory pools (basic)

**Lines of code**: ~2000 (core only)

### Can it train LSTM language model? (MVP-2)
- [x] All of MVP-1
- [x] LookupParameter (embeddings)
- [x] LSTMBuilder
- [x] AdamTrainer
- [x] Softmax, PickNegLogSoftmax
- [x] I/O (save/load)
- [x] Dict (vocabulary)

**Lines of code**: ~5000 (+ RNN infrastructure)

### Is it production-ready? (MVP-3)
- [x] All of MVP-2
- [x] GPU support
- [x] Python bindings
- [x] Documentation
- [x] Test suite
- [x] Examples
- [x] Autobatching

**Lines of code**: ~15000 (full framework)

---

## Learning Resources

### Essential Reading Order
1. `dynet.h` - ComputationGraph (100 lines to understand core)
2. `expr.h` - Expression API (understand user interface)
3. `exec.cc` - SimpleExecutionEngine::forward/backward (the meat)
4. `tensor.h` - Tensor abstraction (data model)
5. `nodes-def-macros.h` - Node patterns (implementation strategy)
6. One node file (e.g., `nodes-activations.cc`) - concrete example

### Example Code to Study
1. `examples/xor/train-xor.cc` - Simplest complete example
2. `examples/mnist/train-mnist.cc` - Vision task
3. `examples/rnnlm/rnnlm.cc` - Sequence modeling
4. `tests/test-nodes.cc` - Node testing patterns

### Advanced Topics
1. `exec.cc::BatchedExecutionEngine` - Autobatching algorithm
2. `gpu-ops.cu` - CUDA kernel patterns
3. `cudnn-ops.cu` - cuDNN integration
4. `python/_dynet.pyx` - Cython binding patterns

---

## Glossary

| Term | Meaning |
|------|---------|
| **ComputationGraph** | Container for DAG of operations |
| **Expression** | User-facing handle to graph node |
| **Node** | Single operation in graph (e.g., MatMul) |
| **VariableIndex** | Integer ID referencing node in graph |
| **Tensor** | Multidimensional array with batch support |
| **Dim** | Shape descriptor (feature dims + batch dim) |
| **Parameter** | Learnable weight matrix/vector |
| **LookupParameter** | Learnable embedding table |
| **AlignedMemoryPool** | Linear allocator with bulk freeing |
| **Device** | Computational backend (CPU or GPU) |
| **RNNBuilder** | High-level sequence model builder |
| **Trainer** | Optimizer (SGD, Adam, etc.) |
| **Shadow Parameters** | Optimizer state (momentum, variance) |
| **Autobatching** | Automatic minibatch formation |

---

## Success Metrics

### v0 Success
- XOR converges in <100 iterations
- No memory leaks (Valgrind clean)
- Numerical gradients match analytical (error < 1e-5)

### v1 Success
- PTB language model perplexity < 100
- GPU 5-10× faster than CPU
- Python API matches C++ API
- Models save/load correctly

### v2 Success
- Autobatch 2-3× speedup on variable-length sequences
- cuDNN convolution matches PyTorch speed
- Handle 100K vocabulary efficiently
- Comprehensive test coverage (>80%)

---

## Conclusion

**Core Philosophy**: Simple primitives (graph, tensor, pool) + composable builders (RNN, softmax) = flexible framework.

**Key Insight**: Dynamic graphs + linear allocation = fast prototyping without memory overhead.

**Rebuild Priority**: Memory layer → Graph → Execution → Operations → Training → Optimization

Start with XOR, build to LSTM language model, optimize to production.

Good luck rebuilding! 🚀
