# DyNet Component Dependency Map

This document provides visual dependency relationships for rebuilding DyNet.

## Component Index (Alphabetical)

Quick reference to locate any component in the codebase:

| Component | File | Classification | Rebuild Phase | Dependencies |
|-----------|------|----------------|---------------|--------------|
| AlignedMemoryPool | aligned-mem-pool.h/cc | CP | v0 | MemAllocator |
| ClassFactoredSoftmax | cfsm-builder.h/cc | PO | v2 | Expression |
| ComputationGraph | dynet.h/cc | CP | v0 | Tensor, ExecutionEngine, Device |
| Conv2D | nodes-conv2d.h/cc | OC | v2 | Node, cuDNN |
| CPUAllocator | mem.h/cc | CP | v0 | None |
| cuDNN Integration | cudnn-ops.h/cu | PO | v2 | cuDNN library |
| DeepLSTM | deep-lstm.h/cc | EO | v2 | LSTMBuilder |
| Device | devices.h/cc | CP | v0 | AlignedMemoryPool |
| Dict | dict.h/cc | DE | v1 | STL |
| Dim | dim.h/cc | CP | v0 | None |
| ExecutionEngine | exec.h/cc | CP | v0 | ComputationGraph, Node |
| Expression | expr.h/cc | CP | v0 | ComputationGraph |
| FastLSTM | fast-lstm.h/cc | PO | v2 | RNNBuilder |
| GradCheck | grad-check.h/cc | DV | v1 | ParameterCollection |
| GRUBuilder | gru.h/cc | OC | v1 | RNNBuilder |
| GPU Operations | gpu-ops.h/cu | PO | v1 | CUDA |
| HierarchicalSoftmax | hsm-builder.h/cc | PO | v2 | Expression |
| Init | init.h/cc | CP | v0 | Device, globals |
| IO System | io.h/cc | DE | v1 | ParameterCollection |
| LSTMBuilder | lstm.h/cc | OC | v1 | RNNBuilder |
| LookupParameter | model.h/cc | CP | v1 | Tensor, Device |
| MemAllocator | mem.h/cc | CP | v0 | None |
| Node (base) | dynet.h/cc | CP | v0 | Tensor |
| Nodes: Activations | nodes-activations.h/cc | CP | v0 | Node |
| Nodes: Arithmetic | nodes-arith-*.h/cc | CP | v0 | Node |
| Nodes: Convolution | nodes-conv.h/cc | OC | v2 | Node |
| Nodes: LSTM | nodes-lstm.h/cc | PO | v2 | Node |
| Nodes: Losses | nodes-losses.h/cc | CP | v0 | Node |
| Nodes: MatMul | nodes-matrixmultiply.h/cc | CP | v0 | Node |
| Nodes: Parameters | param-nodes.h/cc | CP | v0 | Node, Parameter |
| Nodes: Softmax | nodes-softmaxes.h/cc | CP | v0 | Node |
| Parameter | model.h/cc | CP | v0 | Tensor, Device |
| ParameterCollection | model.h/cc | OC | v0 | Parameter |
| ParameterInit | param-init.h/cc | DE | v1 | Tensor |
| Python Bindings | python/* | DE | v1 | All C++ |
| RNNBuilder | rnn.h/cc | OC | v1 | Expression, RNNStateMachine |
| RNNStateMachine | rnn-state-machine.h/cc | DV | v1 | None |
| SaxeInit | saxe-init.h/cc | EO | v2 | Tensor |
| ShadowParameters | shadow-params.h/cc | OC | v1 | Tensor |
| SimpleSGD | training.h/cc | CP | v0 | ParameterCollection |
| Tensor | tensor.h/cc | CP | v0 | Dim, Device, Eigen |
| Timing | timing.h | DV | v2 | STL chrono |
| Trainers (advanced) | training.h/cc | OC | v1 | ShadowParameters |
| TreeLSTM | treelstm.h/cc | EO | v2 | RNNBuilder |
| WeightDecay | weight-decay.h/cc | PO | v1 | None |

---

## Dependency Graph (Bottom-Up)

### Layer 0: Foundation (No Dependencies)
```
┌─────────────────┐
│  MemAllocator   │  (mem.h/cc)
└─────────────────┘

┌─────────────────┐
│      Dim        │  (dim.h/cc)
└─────────────────┘

┌─────────────────┐
│  RNNStateMach   │  (rnn-state-machine.h/cc)
└─────────────────┘

┌─────────────────┐
│  WeightDecay    │  (weight-decay.h/cc)
└─────────────────┘
```

### Layer 1: Memory Management
```
┌─────────────────┐
│  MemAllocator   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ AlignedMemPool  │  (aligned-mem-pool.h/cc)
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│     Device      │ ◄─── │      Dim        │
└────────┬────────┘     └─────────────────┘
         │
         ▼
```

### Layer 2: Data Structures
```
┌─────────────────┐     ┌─────────────────┐
│     Device      │     │      Eigen      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌─────────────────────────┐
         │        Tensor           │  (tensor.h/cc)
         │  + Dim (composition)    │
         └───────────┬─────────────┘
                     │
                     ▼
```

### Layer 3: Parameters
```
         ┌─────────────────────────┐
         │        Tensor           │
         └───────────┬─────────────┘
                     │
         ┌───────────┴──────────────┐
         ▼                          ▼
┌─────────────────┐       ┌─────────────────┐
│   Parameter     │       │ LookupParameter │
│  (dense param)  │       │ (sparse param)  │
└────────┬────────┘       └────────┬────────┘
         │                         │
         └──────────┬──────────────┘
                    ▼
         ┌─────────────────────────┐
         │  ParameterCollection    │  (model.h/cc)
         └─────────────────────────┘
```

### Layer 4: Computation Graph
```
┌─────────────────┐     ┌─────────────────┐
│     Tensor      │     │  Parameter      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌─────────────────────────┐
         │       Node (base)       │
         │   forward_impl()        │
         │   backward_impl()       │
         └───────────┬─────────────┘
                     │
                     ▼
         ┌─────────────────────────┐
         │   ComputationGraph      │  (dynet.h/cc)
         │   - vector<Node*>       │
         │   - topological order   │
         └───────────┬─────────────┘
                     │
                     ▼
         ┌─────────────────────────┐
         │      Expression         │  (expr.h/cc)
         │   (user-facing API)     │
         └─────────────────────────┘
```

### Layer 5: Node Implementations
```
         ┌─────────────────────────┐
         │       Node (base)       │
         └───────────┬─────────────┘
                     │
         ┌───────────┴───────────┬───────────┬─────────────┐
         ▼                       ▼           ▼             ▼
┌─────────────────┐   ┌─────────────────┐   ...   ┌─────────────────┐
│  ParameterNode  │   │   InputNode     │         │  MatrixMultiply │
└─────────────────┘   └─────────────────┘         └─────────────────┘
         ▼                       ▼                         ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  Affine         │   │   Tanh          │   │   Softmax       │
└─────────────────┘   └─────────────────┘   └─────────────────┘
         ▼                       ▼                         ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  PickNegLogSM   │   │   Sum           │   │   Conv2D        │
└─────────────────┘   └─────────────────┘   └─────────────────┘
```

### Layer 6: Execution
```
┌─────────────────────────┐
│   ComputationGraph      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   ExecutionEngine       │  (exec.h/cc)
│   (abstract)            │
└───────────┬─────────────┘
            │
    ┌───────┴────────┐
    ▼                ▼
┌─────────────┐  ┌─────────────────┐
│   Simple    │  │   Batched       │
│   Exec      │  │   Exec          │
└─────────────┘  └─────────────────┘
```

### Layer 7: RNN Builders
```
┌─────────────────┐     ┌─────────────────────┐
│   Expression    │     │  RNNStateMachine    │
└────────┬────────┘     └────────┬────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌─────────────────────────┐
         │     RNNBuilder          │  (rnn.h/cc)
         │     (abstract)          │
         └───────────┬─────────────┘
                     │
         ┌───────────┴───────────┬─────────────┐
         ▼                       ▼             ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  SimpleRNN      │   │   LSTMBuilder   │   │   GRUBuilder    │
└─────────────────┘   └─────────────────┘   └─────────────────┘
                              ▼                     ▼
                    ┌─────────────────┐   ┌─────────────────┐
                    │   FastLSTM      │   │   TreeLSTM      │
                    └─────────────────┘   └─────────────────┘
```

### Layer 8: Training
```
┌─────────────────────────┐     ┌─────────────────┐
│  ParameterCollection    │     │  WeightDecay    │
└───────────┬─────────────┘     └────────┬────────┘
            │                            │
            └────────────┬───────────────┘
                         ▼
             ┌─────────────────────────┐
             │       Trainer           │  (training.h/cc)
             │      (abstract)         │
             └───────────┬─────────────┘
                         │
         ┌───────────────┼───────────────┬────────────┐
         ▼               ▼               ▼            ▼
┌─────────────┐  ┌─────────────┐  ┌──────────┐  ┌─────────┐
│  SimpleSGD  │  │   Momentum  │  │   Adam   │  │  Adagrad│
└─────────────┘  └─────────────┘  └──────────┘  └─────────┘
                         │
                         ▼
             ┌─────────────────────────┐
             │   ShadowParameters      │  (shadow-params.h/cc)
             └─────────────────────────┘
```

### Layer 9: High-Level Utilities
```
┌─────────────────┐
│ParameterInit    │  (param-init.h/cc)
└────────┬────────┘
         │
         ▼ (uses)
┌─────────────────┐
│  Parameter      │
└─────────────────┘

┌─────────────────┐
│      Dict       │  (dict.h/cc)
└────────┬────────┘
         │
         ▼ (indexes)
┌─────────────────┐
│ LookupParameter │
└─────────────────┘

┌─────────────────┐
│   IO System     │  (io.h/cc)
└────────┬────────┘
         │
         ▼ (serializes)
┌─────────────────┐
│ParameterColl    │
└─────────────────┘
```

---

## Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER CODE                                │
│  ComputationGraph cg;                                       │
│  Expression x = input(cg, {10});                            │
│  Expression W = parameter(cg, p_W);                         │
│  Expression y = W * x;                                      │
│  Expression loss = pickneglogsoftmax(y, target);            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              EXPRESSION API (expr.h)                        │
│  - Validates graph_id                                       │
│  - Calls cg.add_function<NodeType>(...)                     │
│  - Returns Expression{&cg, node_index, graph_id}            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         COMPUTATION GRAPH (dynet.h)                         │
│  - Creates new Node                                         │
│  - Appends to nodes vector (topological order)              │
│  - Stores in nodes[VariableIndex]                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              USER CALLS .value()                            │
│  float loss_value = loss.value();                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         EXECUTION ENGINE (exec.h)                           │
│  SimpleExecutionEngine::forward(target_index)               │
│  - Iterate nodes[0..target_index] topologically             │
│  - For each node:                                           │
│    • Allocate output tensor from FXS pool                   │
│    • Call node->forward(input_tensors, output_tensor)       │
│    • Cache result in nfxs[node_index]                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              NODE IMPLEMENTATION                            │
│  MatrixMultiply::forward_impl(xs, fx)                       │
│  - xs[0] = W tensor, xs[1] = x tensor                       │
│  - fx = output tensor (pre-allocated)                       │
│  - fx.mat() = xs[0].mat() * xs[1].mat()  (Eigen)            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              TENSOR OPERATIONS (tensor.h)                   │
│  - Eigen maps over raw memory                               │
│  - SIMD-optimized operations                                │
│  - Batch-aware indexing                                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              USER CALLS backward()                          │
│  cg.backward(loss);                                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         EXECUTION ENGINE (exec.h)                           │
│  SimpleExecutionEngine::backward(target_index, full=false)  │
│  - Initialize dE/dloss = 1.0                                │
│  - Iterate nodes[target_index..0] reverse topologically     │
│  - For each node:                                           │
│    • Allocate gradient tensors from EXS pool                │
│    • Call node->backward(xs, fx, dEdf, i, dEdxi)            │
│    • ACCUMULATE gradients: dEdxi[arg] += local_gradient     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              NODE IMPLEMENTATION                            │
│  MatrixMultiply::backward_impl(xs, fx, dEdf, ai, dEdxi)     │
│  - If ai==0: dEdxi += dEdf * xs[1]^T  (gradient w.r.t. W)   │
│  - If ai==1: dEdxi += xs[0]^T * dEdf  (gradient w.r.t. x)   │
│  - NOTE: += not = (accumulation for fan-in)                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         PARAMETER GRADIENTS UPDATED                         │
│  p_W.gradient() now contains ∂loss/∂W                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              USER CALLS update()                            │
│  trainer.update();                                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              TRAINER (training.h)                           │
│  SimpleSGDTrainer::update()                                 │
│  - Clip gradients if needed                                 │
│  - For each parameter:                                      │
│    • params -= learning_rate * gradients / weight_decay     │
│  - Zero gradients for next iteration                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         NEXT ITERATION (new graph)                          │
│  cg.clear();  // or create new ComputationGraph             │
│  - Invalidates all previous Expressions                     │
│  - Frees Forward/Backward/Scratch pools (linear reset)      │
│  - Parameters persist in Parameter pool                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Memory Layout Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DEVICE MEMORY                            │
└─────────────────────────────────────────────────────────────┘
                      │
         ┌────────────┼────────────┬────────────┬────────────┐
         ▼            ▼            ▼            ▼            ▼
┌─────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     
│  Pool[0]    │ │ Pool[1]  │ │ Pool[2]  │ │ Pool[3]  │     
│  Forward    │ │ Backward │ │Parameter │ │ Scratch  │     
│  Values     │ │ Grads    │ │ Weights  │ │ Temp     │     
└──────┬──────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘     
       │              │            │            │           
       │              │            │            │           
┌──────▼──────────────▼────────────▼────────────▼──────┐
│         AlignedMemoryPool (per pool)                 │
│  ┌────────────────────────────────────────────────┐  │
│  │  InternalMemoryPool[0] (16MB)                  │  │
│  │  [ used──────►|          free          ]       │  │
│  └────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────┐  │
│  │  InternalMemoryPool[1] (16MB, lazy)            │  │
│  │  [                    unused                   ]  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
       ▲
       │ Linear allocation (bump pointer)
       │ Bulk free (resets used pointer)
```

**Key Insight**: Forward/Backward pools reset every graph; Parameter pool persists.

---

## Critical Path for Forward Pass

```
User Code
   │
   ▼
Expression::value()
   │
   ▼
ComputationGraph::incremental_forward(target)
   │
   ▼
ExecutionEngine::incremental_forward(target)
   │
   ▼
for i in [num_nodes_evaluated .. target]:
   │
   ├─► node = nodes[i]
   │
   ├─► Allocate fx from pools[FXS]  ← MEMORY ALLOCATION
   │
   ├─► Get input tensors xs from nfxs cache
   │
   ├─► node->forward(xs, fx)  ← COMPUTATION
   │      │
   │      ▼
   │   Node::forward_impl()
   │      │
   │      ▼
   │   Device dispatch (CPU vs GPU)
   │      │
   │      ▼
   │   Eigen/CUDA operations
   │
   ├─► Cache result: nfxs[i] = fx
   │
   └─► num_nodes_evaluated++
   │
   ▼
Return nfxs[target]
```

---

## Critical Path for Backward Pass

```
User Code
   │
   ▼
ComputationGraph::backward(loss, full=false)
   │
   ▼
ExecutionEngine::backward(loss, full)
   │
   ▼
Initialize: dE/dloss = 1.0
   │
   ▼
for i in [last_backward_computed .. 0]:  (reverse order)
   │
   ├─► node = nodes[i]
   │
   ├─► if node is constant && !full: skip
   │
   ├─► Get dEdf (gradient w.r.t. this node's output)
   │
   ├─► for each argument arg of node:
   │   │
   │   ├─► Allocate dEdxi from pools[EXS]  ← MEMORY ALLOCATION
   │   │
   │   ├─► node->backward(xs, fx, dEdf, arg, dEdxi)  ← COMPUTATION
   │   │      │
   │   │      ▼
   │   │   Node::backward_impl()
   │   │      │
   │   │      ▼
   │   │   Device dispatch (CPU vs GPU)
   │   │      │
   │   │      ▼
   │   │   ACCUMULATE: dEdxi += local_gradient  ← CRITICAL
   │   │
   │   └─► Cache gradient for argument node
   │
   ├─► If parameter node: update parameter.gradient
   │
   └─► last_backward_computed--
   │
   ▼
All parameter gradients populated
```

**Critical Invariant**: Gradients MUST accumulate (+=) because nodes can have multiple uses.

---

## File Organization

```
dynet/
├── Core Graph
│   ├── dynet.h/cc           (ComputationGraph, Node)
│   ├── expr.h/cc            (Expression API)
│   └── exec.h/cc            (ExecutionEngine)
│
├── Data Structures
│   ├── tensor.h/cc          (Tensor + Eigen)
│   ├── dim.h/cc             (Dimension)
│   └── index-tensor.h       (Index operations)
│
├── Memory Management
│   ├── mem.h/cc             (MemAllocator)
│   ├── aligned-mem-pool.h/cc(AlignedMemoryPool)
│   └── devices.h/cc         (Device abstraction)
│
├── Parameters
│   ├── model.h/cc           (Parameter, ParameterCollection)
│   ├── param-init.h/cc      (Initialization strategies)
│   └── shadow-params.h/cc   (Optimizer state)
│
├── Nodes (Operations)
│   ├── nodes.h              (Node registry)
│   ├── nodes-def-macros.h   (Implementation patterns)
│   ├── param-nodes.h/cc     (Input, Parameter nodes)
│   ├── nodes-activations.h/cc
│   ├── nodes-arith-*.h/cc   (4 files)
│   ├── nodes-matrixmultiply.h/cc
│   ├── nodes-softmaxes.h/cc
│   ├── nodes-losses.h/cc
│   ├── nodes-conv*.h/cc     (Convolution variants)
│   ├── nodes-lstm.h/cc      (Fused LSTM)
│   └── ... (20+ node files)
│
├── RNN Infrastructure
│   ├── rnn.h/cc             (RNNBuilder base)
│   ├── rnn-state-machine.h/cc
│   ├── lstm.h/cc            (LSTM variants)
│   ├── gru.h/cc             (GRU)
│   ├── fast-lstm.h/cc       (Optimized LSTM)
│   ├── deep-lstm.h/cc       (Highway LSTM)
│   └── treelstm.h/cc        (Tree-structured)
│
├── Training
│   ├── training.h/cc        (Trainers/Optimizers)
│   ├── weight-decay.h/cc    (Regularization)
│   └── grad-check.h/cc      (Gradient validation)
│
├── Utilities
│   ├── init.h/cc            (Framework initialization)
│   ├── io.h/cc              (Serialization)
│   ├── dict.h/cc            (Vocabulary)
│   ├── timing.h             (Profiling)
│   └── str-util.h           (String utilities)
│
├── GPU Support
│   ├── cuda.h/cc            (CUDA utilities)
│   ├── gpu-ops.h/cu         (Custom kernels)
│   ├── gpu-kernels.h        (Kernel templates)
│   └── cudnn-ops.h/cu       (cuDNN integration)
│
├── Specialized
│   ├── cfsm-builder.h/cc    (Class-factored softmax)
│   ├── hsm-builder.h/cc     (Hierarchical softmax)
│   ├── c2w.h                (Character-to-word)
│   └── pretrain.h/cc        (Pretrained embeddings)
│
└── Support
    ├── globals.h/cc         (Global state)
    ├── except.h             (Exceptions)
    ├── functors.h           (Eigen functors)
    └── tensor-eigen.h       (Eigen typedefs)
```

---

## Rebuild Checklist (Detailed)

### Phase 0: Minimal Core (v0)

#### Week 1: Memory + Data
- [ ] Day 1-2: `mem.h/cc` - CPUAllocator
- [ ] Day 2-3: `aligned-mem-pool.h/cc` - InternalMemoryPool, AlignedMemoryPool
- [ ] Day 3-4: `devices.h/cc` - Device_CPU with 4 pools
- [ ] Day 4-5: `dim.h/cc` - Dimension representation
- [ ] Day 5-7: `tensor.h/cc` - Tensor + Eigen integration

**Milestone**: Allocate tensors, perform Eigen operations

#### Week 2: Graph + Execution
- [ ] Day 1-2: `dynet.h/cc` - Node (base), ComputationGraph (minimal)
- [ ] Day 2-3: `expr.h/cc` - Expression wrapper
- [ ] Day 3-4: `exec.h/cc` - SimpleExecutionEngine (forward/backward)
- [ ] Day 4-5: `param-nodes.h/cc` - InputNode, ParameterNode
- [ ] Day 5-6: `model.h/cc` - Parameter, ParameterCollection (minimal)
- [ ] Day 6-7: Basic nodes (Sum, CwiseMultiply, MatrixMultiply, Tanh)

**Milestone**: Compute forward pass manually

#### Week 3: Training Loop
- [ ] Day 1-2: `nodes-softmaxes.h/cc` - Softmax
- [ ] Day 2-3: `nodes-pickneglogsoftmax.h/cc` - Loss
- [ ] Day 3-4: `training.h/cc` - SimpleSGDTrainer
- [ ] Day 4-5: `init.h/cc` - initialize(), cleanup()
- [ ] Day 5-6: Gradient checking (manual finite difference)
- [ ] Day 6-7: Write XOR example, debug until converges

**Milestone**: XOR trains successfully

---

### Phase 1: Production Features (v1)

#### Week 4: Extended Operations
- [ ] All arithmetic nodes (nodes-arith-*.h/cc)
- [ ] All activation nodes (nodes-activations.h/cc)
- [ ] All loss nodes (nodes-losses.h/cc)
- [ ] Select/concat nodes (nodes-select.h/cc, nodes-concat.h/cc)

**Milestone**: Build MLP for MNIST

#### Week 5: RNN Infrastructure
- [ ] `rnn-state-machine.h/cc` - State machine
- [ ] `rnn.h/cc` - RNNBuilder base, SimpleRNNBuilder
- [ ] `lstm.h/cc` - VanillaLSTMBuilder
- [ ] `gru.h/cc` - GRUBuilder

**Milestone**: Train character-level language model

#### Week 6: Sparse Parameters + I/O
- [ ] `model.h/cc` - LookupParameter, LookupParameterStorage
- [ ] `param-nodes.h/cc` - LookupNode
- [ ] `dict.h/cc` - Dictionary
- [ ] `io.h/cc` - TextFileSaver, TextFileLoader
- [ ] `param-init.h/cc` - ParameterInit variants

**Milestone**: Train word-level language model, save/load

#### Week 7: Advanced Training
- [ ] `shadow-params.h/cc` - ShadowParameters
- [ ] `training.h/cc` - MomentumSGDTrainer, AdamTrainer
- [ ] `weight-decay.h/cc` - L2 regularization
- [ ] `grad-check.h/cc` - Automated gradient checking

**Milestone**: Match PyTorch training convergence

#### Week 8: GPU Support
- [ ] `cuda.h/cc` - CUDA utilities
- [ ] `devices.h/cc` - Device_GPU
- [ ] `gpu-ops.cu` - Basic kernels (matmul, activations)
- [ ] `gpu-kernels.h` - Kernel templates
- [ ] Test CPU vs GPU parity

**Milestone**: GPU 5-10× speedup

#### Week 9: Python Bindings
- [ ] `python/_dynet.pyx` - Core bindings
- [ ] `python/setup.py` - Build system
- [ ] Test Python API matches C++

**Milestone**: Train models from Python

---

### Phase 2: Advanced Features (v2)

#### Week 10: Autobatching
- [ ] `exec.h/cc` - BatchedExecutionEngine
- [ ] `devices.h/cc` - mark/revert for checkpointing
- [ ] Benchmark autobatch speedup

**Milestone**: 2-3× speedup on variable-length sequences

#### Week 11: CNN Support
- [ ] `nodes-conv2d.h/cc` - Conv2D
- [ ] `nodes-maxpooling2d.h/cc` - MaxPooling2D
- [ ] `cudnn-ops.h/cu` - cuDNN integration
- [ ] Test on MNIST/CIFAR-10

**Milestone**: Competitive CNN performance

#### Week 12: Optimizations
- [ ] `fast-lstm.h/cc` - Diagonal LSTM
- [ ] `cfsm-builder.h/cc` - Class-factored softmax
- [ ] `hsm-builder.h/cc` - Hierarchical softmax
- [ ] `treelstm.h/cc` - Tree-structured LSTM
- [ ] Comprehensive benchmarking

**Milestone**: State-of-the-art NLP results

---

## Summary

**Total Rebuild Effort**: ~3 months (1 engineer) or ~1 month (team of 3)

**Critical Path**: Memory → Graph → Execution → Training

**Most Complex**: BatchedExecutionEngine (autobatching algorithm)

**Most Delicate**: Gradient accumulation (easy to get wrong)

**Biggest Win**: AlignedMemoryPool (linear allocation)

**Ready to rebuild!** Follow this guide, refer to `ARCHITECTURAL_ANALYSIS.md` for details, and consult the original code when stuck.
