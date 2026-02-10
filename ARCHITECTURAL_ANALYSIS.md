# DyNet Architectural Analysis: Reverse Engineering Guide

**Objective**: Extract high-signal architectural knowledge for rebuilding DyNet from scratch.

**Classification Legend**:
- **CP**: Core Primitive (must exist for anything to work)
- **OC**: Orchestration / Control
- **PO**: Performance Optimization
- **DE**: Developer Ergonomics
- **DV**: Debug / Visualization
- **EO**: Experimental / Optional

---

## CORE COMPUTATION ENGINE

### FILE: dynet/dynet.h, dynet/dynet.cc
**ROLE**: Central computation graph manager; maintains DAG of operations in topological order.

**DEPENDS ON**:
- `Tensor` (tensor.h) - node value/gradient storage
- `ExecutionEngine` (exec.h) - forward/backward execution
- `Parameter`, `LookupParameter` (model.h) - parameter nodes
- `Device` (devices.h) - memory allocation

**USED BY**:
- `Expression` (expr.h) - user-facing API wrapper
- All node implementations (nodes-*.h) - registered via add_function<T>()
- Execution engines - iterate over nodes vector

**CORE OR AUX**: **CP** - Nothing works without ComputationGraph

**INVARIANTS**:
- Nodes stored in strict topological order (parents before children)
- Only one active ComputationGraph per thread (enforced by n_hgs check)
- Node arguments always reference earlier nodes (DAG property)
- Parameter nodes tracked separately in parameter_nodes vector

**REBUILD PHASE**: **v0** - First component to implement

**WHAT BREAKS**: Entire framework collapses; no way to define computations

**NOTES FOR REBUILDING**:
- Start with simple linear graph (no branching)
- Add graph_id staleness checking early to prevent subtle bugs
- Separate parameter vs computation nodes from day 1

---

### FILE: dynet/expr.h, dynet/expr.cc
**ROLE**: Lightweight user-facing API wrapper; provides composable operations over computation graph nodes.

**DEPENDS ON**:
- `ComputationGraph` (dynet.h) - all operations delegated to pg->add_function<T>()
- Node implementations (nodes-*.h) - wrapped by factory functions

**USED BY**:
- All user code - primary interface for building models
- RNN builders (rnn.h, lstm.h, etc.) - compose sequences
- Training loops - call value(), gradient()

**CORE OR AUX**: **CP** - Core user interface

**INVARIANTS**:
- Expression validity tied to graph_id (throws std::runtime_error if stale)
- value() requires forward pass completion
- gradient() requires backward pass completion
- Gradient computation skips constant nodes unless full=true

**REBUILD PHASE**: **v0** - Implement alongside ComputationGraph

**WHAT BREAKS**: No user-friendly API; direct node manipulation required

**NOTES FOR REBUILDING**:
- Keep Expression struct minimal (pg pointer, index, graph_id)
- Factory functions (input, parameter, lookup) are the real API
- Operator overloads (*, +, -) provide ergonomics but are v1 features

---

### FILE: dynet/tensor.h, dynet/tensor.cc
**ROLE**: Memory-safe abstraction over Eigen tensors; bridges DyNet memory pools with Eigen operations.

**DEPENDS ON**:
- `Dim` (dim.h) - shape + batch dimensions
- `Device` (devices.h) - memory pool allocation
- `AlignedMemoryPool` (aligned-mem-pool.h) - backing memory
- Eigen library - tensor operations

**USED BY**:
- All node implementations - forward/backward operate on Tensors
- `Parameter`/`LookupParameter` (model.h) - value/gradient storage
- Execution engines - cache intermediate results

**CORE OR AUX**: **CP** - Data container for all computation

**INVARIANTS**:
- Tensors DO NOT own memory (pointers only; lifecycle managed by device pools)
- Batch dimension broadcast if d.bd == 1
- All dimensions, device, and pool must be set before use
- Gradient tensors parallel-structured to value tensors
- No Eigen temporary allocation (EIGEN_NO_MALLOC enforced)

**REBUILD PHASE**: **v0** - Critical data structure

**WHAT BREAKS**: No computation possible; nodes need data containers

**NOTES FOR REBUILDING**:
- Start with CPU-only, flat memory model
- Add batching in v1 (significant complexity)
- Eigen integration is powerful but optional (could use raw arrays in v0)

---

### FILE: dynet/exec.h, dynet/exec.cc
**ROLE**: Implements forward/backward passes over computation graph; dispatches node evaluations in topological order.

**DEPENDS ON**:
- `ComputationGraph` (dynet.h) - node list to iterate
- `Node` (dynet.h) - forward/backward implementations
- `Tensor` (tensor.h) - value/gradient storage

**USED BY**:
- `Expression::value()`, `Expression::gradient()` - trigger execution
- `ComputationGraph::backward()` - full backward pass

**CORE OR AUX**: **CP** - Execution orchestrator

**INVARIANTS**:
- Gradients ACCUMULATE (+=) not replace due to DAG fan-in
- Forward must complete before backward
- Invalidation clears caches for recomputation
- Constant nodes skipped in backward unless full=true

**REBUILD PHASE**: **v0** - SimpleExecutionEngine first; BatchedExecutionEngine in v2

**WHAT BREAKS**: No way to execute graphs; static definition only

**NOTES FOR REBUILDING**:
- v0: Implement SimpleExecutionEngine (straightforward topological traversal)
- v1: Add incremental execution (num_nodes_evaluated tracking)
- v2: Add BatchedExecutionEngine (autobatching requires significant complexity)
- Gradient accumulation is critical - easy to get wrong

---

### FILE: dynet/model.h, dynet/model.cc
**ROLE**: Hierarchical parameter storage; manages learnable weights with dense/sparse update support.

**DEPENDS ON**:
- `Tensor` (tensor.h) - value/gradient storage
- `Device` (devices.h) - memory allocation (DeviceMempool::PS)
- `ParameterInit` (param-init.h) - initialization strategies

**USED BY**:
- User code - create parameters via ParameterCollection
- `Expression` API - parameter(), lookup() factory functions
- Trainers (training.h) - update() modifies parameter values
- I/O system (io.h) - serialization

**CORE OR AUX**: **OC** - Auxiliary to computation, but essential for training

**INVARIANTS**:
- Parameters survive across graph resets (owned by ParameterCollection, not ComputationGraph)
- Gradients zero'd before backward pass
- LookupParameters track non_zero_grads for sparse updates
- Each parameter has updated flag and nonzero_grad status

**REBUILD PHASE**: **v0** - Need parameters for any non-trivial model

**WHAT BREAKS**: No trainable models; only static computation

**NOTES FOR REBUILDING**:
- Start with dense Parameters only (simpler)
- Add LookupParameters in v1 (sparse updates add complexity)
- Hierarchical naming is v2 feature (useful but not critical)

---

## MEMORY MANAGEMENT

### FILE: dynet/mem.h, dynet/mem.cc
**ROLE**: Device-specific memory allocation interface (CPU, GPU, shared memory).

**DEPENDS ON**:
- CUDA runtime (for GPUAllocator)
- System mmap (for SharedAllocator)

**USED BY**:
- `AlignedMemoryPool` (aligned-mem-pool.h) - uses MemAllocator for raw memory

**CORE OR AUX**: **CP** - Primitive memory layer

**INVARIANTS**:
- Each device has exactly one allocator instance
- Alignment constants hardcoded (CPU=32, GPU=256)
- Allocators are not copyable

**REBUILD PHASE**: **v0** - Implement CPUAllocator first; GPUAllocator in v1

**WHAT BREAKS**: No memory allocation; framework cannot run

**NOTES FOR REBUILDING**:
- v0: CPUAllocator with standard malloc/free (ignore alignment initially)
- v1: Add SIMD alignment (32 bytes) for CPU
- v2: Add GPUAllocator (CUDA required)

---

### FILE: dynet/aligned-mem-pool.h, dynet/aligned-mem-pool.cc
**ROLE**: Multi-pool memory manager with linear allocation and bulk freeing.

**DEPENDS ON**:
- `MemAllocator` (mem.h) - device-specific allocation

**USED BY**:
- `Device` (devices.h) - creates 4 pools (forward, backward, parameter, scratch)

**CORE OR AUX**: **CP** - Critical memory strategy

**INVARIANTS**:
- Linear allocation within pool (fast, no fragmentation)
- No individual deallocation; only bulk reset
- Lazy expansion (creates new pool when current exhausts)
- Single active pool index tracked

**REBUILD PHASE**: **v0** - Core to efficient execution

**WHAT BREAKS**: Memory fragmentation, allocation overhead, or OOM

**NOTES FOR REBUILDING**:
- v0: Single pool, linear allocator, bulk free
- v1: Add multiple pools with automatic expansion
- v2: Add checkpointing (mark/revert for autobatching)
- This design is DyNet's secret sauce for speed

---

### FILE: dynet/devices.h, dynet/devices.cc
**ROLE**: Encapsulate device resources (CPU/GPU) with four specialized memory pools.

**DEPENDS ON**:
- `AlignedMemoryPool` (aligned-mem-pool.h) - pool management
- `MemAllocator` (mem.h) - device-specific allocation
- CUDA/cuBLAS/cuDNN (for Device_GPU)

**USED BY**:
- `Tensor` (tensor.h) - allocate_tensor() from pools
- `ComputationGraph` (dynet.h) - uses default_device
- DeviceManager singleton - manages all devices

**CORE OR AUX**: **CP** - Resource manager

**INVARIANTS**:
- 4 pools per device: Forward, Backward, Parameter, Scratch
- Checkpointing via mark()/revert()
- Device destructor protected (prevents accidental deletion)

**REBUILD PHASE**: **v0** - Implement Device_CPU; Device_GPU in v1

**WHAT BREAKS**: No memory management; tensors cannot allocate

**NOTES FOR REBUILDING**:
- v0: Device_CPU with 4 pools (even if simplified)
- v1: Add Device_GPU (requires CUDA expertise)
- v2: Add mark/revert for autobatching

---

## NODE IMPLEMENTATIONS

### FILE: dynet/nodes-def-macros.h
**ROLE**: Macro-based code generation for device-agnostic node implementations.

**DEPENDS ON**:
- Nothing (pure macros)

**USED BY**:
- All node implementation files (nodes-*.h)

**CORE OR AUX**: **DE** - Developer ergonomics (code organization)

**INVARIANTS**:
- forward_impl() must NOT initialize fx (assumes scratch memory)
- backward_impl() must ACCUMULATE to dEdxi (+=, not =)
- No memory allocation in forward/backward (pools pre-allocated)
- Eigen temporaries requested via aux_storage_size()

**REBUILD PHASE**: **v1** - Can inline implementations in v0

**WHAT BREAKS**: More boilerplate; no abstraction for CPU/GPU dispatch

**NOTES FOR REBUILDING**:
- v0: Skip macros, write direct implementations
- v1: Add macro layer for code sharing
- v2: Add device dispatch templates

---

### FILE: dynet/param-nodes.h, dynet/param-nodes.cc
**ROLE**: Leaf nodes representing inputs and parameters in computation graph.

**DEPENDS ON**:
- `Parameter`, `LookupParameter` (model.h) - parameter references
- `ComputationGraph` (dynet.h) - Node base class

**USED BY**:
- `Expression` factories: parameter(), lookup(), input(), const_parameter()

**CORE OR AUX**: **CP** - Graph entry points

**INVARIANTS**:
- InputNode: no backward (external data)
- ParameterNode: backward accumulates to parameter gradient
- LookupNode: backward accumulates to specific embedding rows
- ConstParameterNode: no backward (fixed weights)

**REBUILD PHASE**: **v0** - Need these for any model

**WHAT BREAKS**: Cannot define model inputs or parameters

**NOTES FOR REBUILDING**:
- Implement InputNode and ParameterNode in v0
- Add LookupNode in v1 (sparse embeddings)
- ConstParameterNode is optimization (v2)

---

### FILE: dynet/nodes-activations.h, dynet/nodes-activations.cc
**ROLE**: Nonlinear activation functions (ReLU, Tanh, Sigmoid, etc.).

**DEPENDS ON**:
- `Node` (dynet.h) - base class
- Eigen functors - element-wise operations

**USED BY**:
- Neural network models via Expression API

**CORE OR AUX**: **CP** - Essential for neural networks

**INVARIANTS**:
- Element-wise operations (input/output same shape)
- Backward derivatives: f'(x) computed from forward values

**REBUILD PHASE**: **v0** - Implement ReLU, Tanh; others in v1

**WHAT BREAKS**: No nonlinearities; limited model expressiveness

**NOTES FOR REBUILDING**:
- v0: ReLU (trivial), Tanh (use Eigen)
- v1: Sigmoid, ELU, SELU, etc.
- v2: Exotic activations (SoftSign, Swish)

---

### FILE: dynet/nodes-arith-*.h (4 files: const, sum, unary, cwise)
**ROLE**: Arithmetic operations organized by semantic category.

**DEPENDS ON**:
- `Node` (dynet.h) - base class
- Eigen - mathematical operations

**USED BY**:
- Expression operator overloads (*, +, -, /)
- User models

**CORE OR AUX**: **CP** - Fundamental operations

**INVARIANTS**:
- Broadcasting semantics for batch dimensions
- Sum operations reduce dimensions
- Element-wise operations preserve shape

**REBUILD PHASE**: **v0** - Basic arithmetic (add, multiply); v1 full suite

**WHAT BREAKS**: Cannot perform basic math operations

**NOTES FOR REBUILDING**:
- v0: Implement CwiseMultiply, CwiseQuotient, Sum, ScalarMultiply
- v1: Add full arithmetic suite
- v2: Optimize broadcasts and reductions

---

### FILE: dynet/nodes-matrixmultiply.h, dynet/nodes-matrixmultiply.cc
**ROLE**: Matrix multiplication operations (gemm, gemv).

**DEPENDS ON**:
- `Node` (dynet.h) - base class
- Eigen (CPU) or cuBLAS (GPU) - optimized BLAS

**USED BY**:
- Affine transforms (Wx + b)
- Attention mechanisms
- Linear layers

**CORE OR AUX**: **CP** - Critical for neural networks

**INVARIANTS**:
- Dimension compatibility: (M×K) · (K×N) → (M×N)
- Batch dimension broadcasted across non-batch dims

**REBUILD PHASE**: **v0** - Essential early

**WHAT BREAKS**: Cannot implement linear layers or transformers

**NOTES FOR REBUILDING**:
- v0: Implement MatrixMultiply with Eigen
- v1: Optimize with cuBLAS for GPU
- v2: Add specialized variants (MatrixVectorMultiply)

---

### FILE: dynet/nodes-softmaxes.h, dynet/nodes-softmaxes.cc
**ROLE**: Softmax and variants (logsoftmax, sparsemax) for probability distributions.

**DEPENDS ON**:
- `Node` (dynet.h) - base class
- Eigen - exponential, logarithm operations

**USED BY**:
- Classification models (output layer)
- Attention mechanisms

**CORE OR AUX**: **CP** - Essential for classification

**INVARIANTS**:
- Numerical stability via log-sum-exp trick (subtract max before exp)
- Outputs sum to 1 (probability distribution)

**REBUILD PHASE**: **v0** - Implement Softmax early

**WHAT BREAKS**: Cannot do classification or attention

**NOTES FOR REBUILDING**:
- v0: Softmax with numerical stability
- v1: LogSoftmax (more efficient for loss computation)
- v2: Sparsemax (sparse probabilities)

---

### FILE: dynet/nodes-pickneglogsoftmax.h, dynet/nodes-pickneglogsoftmax.cc
**ROLE**: Fused operation: negative log-softmax picking for efficient loss computation.

**DEPENDS ON**:
- `Node` (dynet.h) - base class
- Softmax computation

**USED BY**:
- Classification training loops

**CORE OR AUX**: **PO** - Performance optimization (fused operation)

**INVARIANTS**:
- Computes -log(softmax(x)[i]) efficiently
- Gradient only affects selected index i

**REBUILD PHASE**: **v1** - Optimize after v0 works

**WHAT BREAKS**: Slower classification loss (but separable Softmax + PickElement + NegativeLogLoss works)

**NOTES FOR REBUILDING**:
- v0: Skip, use separate Softmax + PickElement + Log + Negate
- v1: Add fused version for efficiency
- This is a common optimization pattern in DyNet

---

### FILE: dynet/nodes-losses.h, dynet/nodes-losses.cc
**ROLE**: Loss functions (hinge, binary log loss, Poisson regression, pairwise rank loss).

**DEPENDS ON**:
- `Node` (dynet.h) - base class

**USED BY**:
- Training loops (final loss computation)

**CORE OR AUX**: **CP** - Essential for training

**INVARIANTS**:
- Scalar outputs (losses reduce to single value)
- Gradients flow back to inputs

**REBUILD PHASE**: **v0** - Implement one loss (e.g., MSE); others in v1

**WHAT BREAKS**: Cannot train models (no training signal)

**NOTES FOR REBUILDING**:
- v0: Squared error or cross-entropy
- v1: Add diverse losses for different tasks

---

### FILE: dynet/nodes-conv.h, dynet/nodes-conv2d.h, nodes-maxpooling2d.h
**ROLE**: Convolutional and pooling operations (1D and 2D variants).

**DEPENDS ON**:
- `Node` (dynet.h) - base class
- cuDNN (GPU) - optimized convolutions

**USED BY**:
- CNN models (image classification, object detection)

**CORE OR AUX**: **OC** - Essential for CNNs but not core framework

**INVARIANTS**:
- Stride, padding, filter size parameters
- Output dimensions calculated from input + hyperparameters

**REBUILD PHASE**: **v2** - Not needed for basic RNN/MLP models

**WHAT BREAKS**: Cannot implement CNNs

**NOTES FOR REBUILDING**:
- v0/v1: Skip if focusing on RNNs/MLPs
- v2: Add for computer vision support
- cuDNN integration significantly boosts performance

---

### FILE: dynet/nodes-lstm.h, dynet/nodes-lstm.cc
**ROLE**: Fused LSTM cell operations (gates, cell state, hidden state).

**DEPENDS ON**:
- `Node` (dynet.h) - base class

**USED BY**:
- LSTM builders (lstm.h)

**CORE OR AUX**: **PO** - Performance optimization (fused LSTM)

**INVARIANTS**:
- Input/forget/output gates coupled
- Cell state and hidden state computed jointly

**REBUILD PHASE**: **v2** - Can compose LSTM from basic ops in v0/v1

**WHAT BREAKS**: Slower LSTMs (but compositional version works)

**NOTES FOR REBUILDING**:
- v0: Build LSTM from separate operations
- v1: Basic fused LSTM
- v2: Optimized with cuDNN LSTMCell

---

## RNN INFRASTRUCTURE

### FILE: dynet/rnn.h, dynet/rnn.cc
**ROLE**: Abstract RNN builder interface; defines standard RNN API with state machine enforcement.

**DEPENDS ON**:
- `ComputationGraph` (dynet.h) - builds nodes
- `Expression` (expr.h) - sequence composition
- `RNNStateMachine` (rnn-state-machine.h) - API validation

**USED BY**:
- LSTM, GRU, SimpleRNN builders - concrete implementations
- User code - sequence modeling

**CORE OR AUX**: **OC** - Orchestration for sequence models

**INVARIANTS**:
- Must call new_graph() before start_new_sequence()
- Must call start_new_sequence() before add_input()
- State machine enforces valid operation sequences

**REBUILD PHASE**: **v1** - After basic graph execution works

**WHAT BREAKS**: No standard RNN API; manual graph construction required

**NOTES FOR REBUILDING**:
- v0: Skip RNN builders, manually construct recurrence
- v1: Add RNNBuilder abstraction
- v2: Add advanced features (RNNPointer, arbitrary recurrence)

---

### FILE: dynet/lstm.h, dynet/lstm.cc
**ROLE**: LSTM builder implementation with variants (vanilla, coupled, compact).

**DEPENDS ON**:
- `RNNBuilder` (rnn.h) - abstract interface
- `Parameter` (model.h) - LSTM weights
- `Expression` (expr.h) - gate computations

**USED BY**:
- User sequence models (language modeling, translation)

**CORE OR AUX**: **OC** - High-level building block

**INVARIANTS**:
- num_h0_components() = 2 * layers (cell + hidden for each layer)
- Dropout masks tied across timesteps (variational dropout)

**REBUILD PHASE**: **v1** - After RNNBuilder interface exists

**WHAT BREAKS**: No LSTM support; manual implementation required

**NOTES FOR REBUILDING**:
- v0: Skip
- v1: Implement VanillaLSTMBuilder
- v2: Add variants (coupled gates, compact representation)

---

### FILE: dynet/gru.h, dynet/gru.cc
**ROLE**: GRU builder implementation (simpler than LSTM).

**DEPENDS ON**:
- `RNNBuilder` (rnn.h) - abstract interface

**USED BY**:
- User sequence models (alternative to LSTM)

**CORE OR AUX**: **OC** - High-level building block

**INVARIANTS**:
- num_h0_components() = layers (only hidden, no cell state)

**REBUILD PHASE**: **v1** - After RNNBuilder interface exists

**WHAT BREAKS**: No GRU support

**NOTES FOR REBUILDING**:
- v0: Skip
- v1: Add after LSTM (simpler to implement)

---

### FILE: dynet/fast-lstm.h, dynet/fast-lstm.cc
**ROLE**: Optimized LSTM with diagonal weight matrices for cell connections.

**DEPENDS ON**:
- `RNNBuilder` (rnn.h) - abstract interface

**USED BY**:
- Performance-critical sequence models

**CORE OR AUX**: **PO** - Performance optimization

**INVARIANTS**:
- Reduces parameters while maintaining expressiveness

**REBUILD PHASE**: **v2** - Optimization after standard LSTM works

**WHAT BREAKS**: Slower LSTMs

**NOTES FOR REBUILDING**:
- v0/v1: Use standard LSTM
- v2: Add for parameter efficiency

---

### FILE: dynet/treelstm.h, dynet/treelstm.cc
**ROLE**: Tree-structured LSTM variants (N-ary, unidirectional, bidirectional).

**DEPENDS ON**:
- `RNNBuilder` (rnn.h) - abstract interface

**USED BY**:
- Syntax-aware NLP models (parsing, tree encoding)

**CORE OR AUX**: **EO** - Experimental / Optional

**INVARIANTS**:
- add_input(id, children_indices, x) processes nodes with specific children
- Requires pre-allocation via set_num_elements()

**REBUILD PHASE**: **v2** - Advanced feature

**WHAT BREAKS**: Cannot model tree structures

**NOTES FOR REBUILDING**:
- v0/v1: Skip entirely
- v2: Add for research applications

---

### FILE: dynet/rnn-state-machine.h, dynet/rnn-state-machine.cc
**ROLE**: Enforces valid RNN API usage through state transitions.

**DEPENDS ON**:
- Nothing (standalone state machine)

**USED BY**:
- `RNNBuilder` (rnn.h) - validates API calls

**CORE OR AUX**: **DV** - Debug / Validation

**INVARIANTS**:
- States: CREATED → GRAPH_READY → READING_INPUT
- Throws invalid_operation exceptions on violations

**REBUILD PHASE**: **v1** - Add with RNNBuilder

**WHAT BREAKS**: Runtime errors harder to debug

**NOTES FOR REBUILDING**:
- v0: Skip validation
- v1: Add for better error messages

---

## TRAINING INFRASTRUCTURE

### FILE: dynet/training.h, dynet/training.cc
**ROLE**: Optimizer implementations (SGD, Momentum, Adam, Adagrad, etc.).

**DEPENDS ON**:
- `ParameterCollection` (model.h) - parameters to update
- `ShadowParameters` (shadow-params.h) - optimizer state
- `WeightDecay` (weight-decay.h) - L2 regularization

**USED BY**:
- Training loops - update() after backward pass

**CORE OR AUX**: **CP** - Essential for training

**INVARIANTS**:
- Gradient clipping before updates
- Sparse updates for LookupParameters (if enabled)
- Shadow parameters track momentum, adaptive learning rates

**REBUILD PHASE**: **v0** - Implement SimpleSGD; others in v1

**WHAT BREAKS**: Cannot train models

**NOTES FOR REBUILDING**:
- v0: SimpleSGDTrainer only
- v1: Add Adam (most popular modern optimizer)
- v2: Add full suite (Adagrad, RMSProp, etc.)

---

### FILE: dynet/param-init.h, dynet/param-init.cc
**ROLE**: Parameter initialization strategies (Glorot, Uniform, Normal, etc.).

**DEPENDS ON**:
- `Tensor` (tensor.h) - initialize parameter values

**USED BY**:
- `ParameterCollection` (model.h) - initialize new parameters

**CORE OR AUX**: **DE** - Developer ergonomics

**INVARIANTS**:
- Initialization affects training convergence
- Glorot/Xavier preserves variance across layers

**REBUILD PHASE**: **v0** - Implement uniform initialization; others in v1

**WHAT BREAKS**: Poor initialization slows training

**NOTES FOR REBUILDING**:
- v0: Uniform random initialization
- v1: Add Glorot (critical for deep networks)
- v2: Add specialized initializers (Saxe, orthogonal)

---

### FILE: dynet/weight-decay.h, dynet/weight-decay.cc
**ROLE**: L2 weight decay (regularization) for preventing overfitting.

**DEPENDS ON**:
- Nothing (simple exponential decay computation)

**USED BY**:
- Trainers (training.h) - apply decay during updates

**CORE OR AUX**: **PO** - Performance optimization (generalization)

**INVARIANTS**:
- Exponential decay: weight *= (1 - lambda)^num_updates
- Rescaling when decay < 0.25 to prevent underflow

**REBUILD PHASE**: **v1** - After basic training works

**WHAT BREAKS**: Overfitting on small datasets

**NOTES FOR REBUILDING**:
- v0: Skip regularization
- v1: Add L2 weight decay
- v2: Add other regularization (dropout, batch norm)

---

### FILE: dynet/shadow-params.h, dynet/shadow-params.cc
**ROLE**: Auxiliary tensors for optimizer state (momentum, adaptive learning rates).

**DEPENDS ON**:
- `Tensor` (tensor.h) - state storage

**USED BY**:
- Trainers (training.h) - maintain optimizer state

**CORE OR AUX**: **OC** - Orchestration for advanced optimizers

**INVARIANTS**:
- One shadow tensor per parameter
- Sparse/dense variants for LookupParameters

**REBUILD PHASE**: **v1** - When adding advanced optimizers

**WHAT BREAKS**: Only SimpleSGD available

**NOTES FOR REBUILDING**:
- v0: Skip (SimpleSGD has no state)
- v1: Add for Momentum, Adam

---

### FILE: dynet/grad-check.h, dynet/grad-check.cc
**ROLE**: Numerical gradient verification for debugging.

**DEPENDS ON**:
- `ParameterCollection` (model.h) - parameters to check
- `Expression` (expr.h) - compute analytical gradients

**USED BY**:
- Unit tests - validate node implementations
- User debugging - verify custom nodes

**CORE OR AUX**: **DV** - Debug / Visualization

**INVARIANTS**:
- Finite difference approximation: (f(x+ε) - f(x-ε)) / (2ε)

**REBUILD PHASE**: **v1** - Critical for debugging but not core functionality

**WHAT BREAKS**: Harder to debug gradient bugs

**NOTES FOR REBUILDING**:
- v0: Skip
- v1: Add for testing node implementations

---

## I/O AND UTILITIES

### FILE: dynet/io.h, dynet/io.cc
**ROLE**: Model serialization (save/load parameters to/from files).

**DEPENDS ON**:
- `ParameterCollection` (model.h) - parameters to serialize

**USED BY**:
- Training scripts - save checkpoints
- Inference scripts - load trained models

**CORE OR AUX**: **DE** - Developer ergonomics

**INVARIANTS**:
- Text format with metadata (dimensions, gradient state)
- Preserves hierarchical structure via key prefixes

**REBUILD PHASE**: **v1** - After training works

**WHAT BREAKS**: Cannot save/load models

**NOTES FOR REBUILDING**:
- v0: Skip (retrain from scratch each time)
- v1: Add text-based serialization
- v2: Add binary format for efficiency

---

### FILE: dynet/dict.h, dynet/dict.cc
**ROLE**: Bidirectional word ↔ integer ID mapping for NLP tasks.

**DEPENDS ON**:
- Standard library (map, vector)

**USED BY**:
- NLP examples - vocabulary management
- LookupParameter indexing

**CORE OR AUX**: **DE** - Developer ergonomics

**INVARIANTS**:
- freeze() locks vocabulary
- UNK token for out-of-vocabulary words

**REBUILD PHASE**: **v1** - After LookupParameters exist

**WHAT BREAKS**: Manual vocabulary management required

**NOTES FOR REBUILDING**:
- v0: Skip (use raw integers)
- v1: Add for NLP convenience

---

### FILE: dynet/dim.h, dynet/dim.cc
**ROLE**: Tensor dimension representation with batch support.

**DEPENDS ON**:
- Nothing (simple struct)

**USED BY**:
- `Tensor` (tensor.h) - shape specification
- All nodes - dimension inference

**CORE OR AUX**: **CP** - Fundamental data structure

**INVARIANTS**:
- Up to 7 dimensions (DYNET_MAX_TENSOR_DIM)
- Separate batch dimension (bd)
- size() = batch_size() × batch_elems()

**REBUILD PHASE**: **v0** - Essential from day 1

**WHAT BREAKS**: No shape information; cannot allocate tensors

**NOTES FOR REBUILDING**:
- v0: Fixed 2D (matrix) + batch dimension
- v1: Add full multidimensional support
- v2: Optimize dimension operations

---

### FILE: dynet/init.h, dynet/init.cc
**ROLE**: Framework initialization and configuration.

**DEPENDS ON**:
- `Device` (devices.h) - device setup
- Global state - random seed, memory allocation

**USED BY**:
- User code - main() calls initialize()

**CORE OR AUX**: **CP** - Entry point

**INVARIANTS**:
- Must call initialize() before any DyNet operations
- Must call cleanup() before exit

**REBUILD PHASE**: **v0** - First function called

**WHAT BREAKS**: Framework doesn't initialize; undefined behavior

**NOTES FOR REBUILDING**:
- v0: Basic initialization (RNG seed, single CPU device)
- v1: Add command-line argument parsing
- v2: Add multi-GPU configuration

---

### FILE: dynet/timing.h
**ROLE**: Performance profiling utilities (Timer, Timing, NamedTimer).

**DEPENDS ON**:
- Standard library (chrono)

**USED BY**:
- Benchmarking code
- Performance debugging

**CORE OR AUX**: **DV** - Debug / Visualization

**INVARIANTS**:
- Millisecond precision
- Accumulative profiling with NamedTimer

**REBUILD PHASE**: **v2** - Optimization phase

**WHAT BREAKS**: Harder to profile performance

**NOTES FOR REBUILDING**:
- v0/v1: Skip
- v2: Add for optimization

---

## GPU/CUDA INFRASTRUCTURE

### FILE: dynet/cuda.h, dynet/cuda.cc
**ROLE**: CUDA error checking macros and utilities.

**DEPENDS ON**:
- CUDA runtime

**USED BY**:
- All GPU code - wrap CUDA API calls

**CORE OR AUX**: **DE** - Developer ergonomics (error handling)

**INVARIANTS**:
- Detailed error messages on CUDA failures
- Memory diagnostics on allocation failures

**REBUILD PHASE**: **v1** - When adding GPU support

**WHAT BREAKS**: Cryptic CUDA errors

**NOTES FOR REBUILDING**:
- v0: CPU-only
- v1: Add CUDA with error checking

---

### FILE: dynet/gpu-ops.h, dynet/gpu-ops.cu
**ROLE**: Custom CUDA kernels for GPU operations.

**DEPENDS ON**:
- CUDA runtime

**USED BY**:
- Node implementations - device-specific forward/backward

**CORE OR AUX**: **PO** - Performance optimization

**INVARIANTS**:
- Thread-coalesced memory access
- Dynamic grid/block sizing

**REBUILD PHASE**: **v1** - With GPU support

**WHAT BREAKS**: GPU operations unavailable

**NOTES FOR REBUILDING**:
- v0: CPU-only
- v1: Add GPU kernels for common operations

---

### FILE: dynet/gpu-kernels.h
**ROLE**: Templated kernel patterns for GPU operations.

**DEPENDS ON**:
- CUDA runtime

**USED BY**:
- `gpu-ops.cu` - kernel implementations

**CORE OR AUX**: **DE** - Developer ergonomics (code reuse)

**INVARIANTS**:
- Kernel functor pattern
- Thread-coalesced loops

**REBUILD PHASE**: **v1** - With GPU support

**WHAT BREAKS**: More boilerplate in GPU code

**NOTES FOR REBUILDING**:
- v0: CPU-only
- v1: Add templated kernels for code reuse

---

### FILE: dynet/cudnn-ops.h, dynet/cudnn-ops.cu
**ROLE**: cuDNN integration for optimized convolution and pooling.

**DEPENDS ON**:
- cuDNN library

**USED BY**:
- Convolution nodes (nodes-conv2d.h)
- Pooling nodes (nodes-maxpooling2d.h)

**CORE OR AUX**: **PO** - Performance optimization

**INVARIANTS**:
- Pre-allocated cuDNN descriptors
- Workspace management (8MB limit)
- Algorithm caching

**REBUILD PHASE**: **v2** - CNN optimization

**WHAT BREAKS**: Slower convolutions (can fall back to manual implementation)

**NOTES FOR REBUILDING**:
- v0/v1: CPU convolutions or skip CNNs
- v2: Add cuDNN for performance

---

## SPECIALIZED BUILDERS

### FILE: dynet/cfsm-builder.h, dynet/cfsm-builder.cc
**ROLE**: Class-factored softmax for large vocabulary efficiency.

**DEPENDS ON**:
- `Expression` (expr.h) - builds computation graph

**USED BY**:
- Language models with large vocabularies

**CORE OR AUX**: **PO** - Performance optimization

**INVARIANTS**:
- Two-level hierarchy: class → words within class
- Reduces complexity from O(V) to O(sqrt(V))

**REBUILD PHASE**: **v2** - Optimization for large-scale LMs

**WHAT BREAKS**: Slow softmax for large vocabularies

**NOTES FOR REBUILDING**:
- v0/v1: Use standard softmax
- v2: Add for scaling to large vocabularies

---

### FILE: dynet/hsm-builder.h, dynet/hsm-builder.cc
**ROLE**: Hierarchical softmax for efficient vocabulary scaling.

**DEPENDS ON**:
- `Expression` (expr.h) - builds computation graph

**USED BY**:
- Language models with large vocabularies

**CORE OR AUX**: **PO** - Performance optimization

**INVARIANTS**:
- Binary tree structure
- Reduces complexity from O(V) to O(log V)

**REBUILD PHASE**: **v2** - Optimization for large-scale LMs

**WHAT BREAKS**: Slow softmax for large vocabularies

**NOTES FOR REBUILDING**:
- v0/v1: Use standard softmax
- v2: Add for scaling to large vocabularies

---

### FILE: dynet/deep-lstm.h, dynet/deep-lstm.cc
**ROLE**: Deep transition LSTM (highway connections between layers).

**DEPENDS ON**:
- `LSTMBuilder` (lstm.h) - base LSTM

**USED BY**:
- Deep sequence models

**CORE OR AUX**: **EO** - Experimental / Optional

**INVARIANTS**:
- Highway connections for gradient flow in deep networks

**REBUILD PHASE**: **v2** - Research feature

**WHAT BREAKS**: Cannot implement deep transition LSTMs

**NOTES FOR REBUILDING**:
- v0/v1: Use standard multi-layer LSTM
- v2: Add for research

---

## AUXILIARY FILES

### FILE: dynet/except.h
**ROLE**: Custom exception types for DyNet-specific errors.

**DEPENDS ON**:
- Standard library (exception)

**USED BY**:
- All DyNet code - error reporting

**CORE OR AUX**: **DE** - Developer ergonomics

**REBUILD PHASE**: **v0** - Simple error handling from day 1

**WHAT BREAKS**: Generic exceptions harder to debug

---

### FILE: dynet/globals.h, dynet/globals.cc
**ROLE**: Global state management (default device, random number generator).

**DEPENDS ON**:
- `Device` (devices.h)
- Random number generator

**USED BY**:
- All DyNet code - access global state

**CORE OR AUX**: **OC** - Orchestration

**INVARIANTS**:
- Single default device per thread
- Thread-safe RNG access

**REBUILD PHASE**: **v0** - Manage global state

**WHAT BREAKS**: No default device; explicit device passing everywhere

---

### FILE: dynet/functors.h, dynet/simd-functors.h
**ROLE**: Eigen functors for custom operations.

**DEPENDS ON**:
- Eigen library

**USED BY**:
- Node implementations - custom element-wise operations

**CORE OR AUX**: **DE** - Developer ergonomics

**INVARIANTS**:
- SIMD-optimized operations

**REBUILD PHASE**: **v1** - After basic operations work

**WHAT BREAKS**: Slower custom operations

---

### FILE: dynet/tensor-eigen.h
**ROLE**: Eigen tensor type definitions and utilities.

**DEPENDS ON**:
- Eigen library

**USED BY**:
- `Tensor` (tensor.h) - Eigen integration

**CORE OR AUX**: **CP** - Type definitions

**REBUILD PHASE**: **v0** - If using Eigen

**WHAT BREAKS**: No Eigen integration (can use raw arrays)

---

### FILE: dynet/str-util.h
**ROLE**: String utilities for formatting and parsing.

**DEPENDS ON**:
- Standard library

**USED BY**:
- Debugging code - format dimensions, node info

**CORE OR AUX**: **DE** - Developer ergonomics

**REBUILD PHASE**: **v1** - Nice to have

**WHAT BREAKS**: Manual string formatting

---

### FILE: dynet/c2w.h
**ROLE**: Character-to-word composition for character-level models.

**DEPENDS ON**:
- `Expression` (expr.h)
- `RNNBuilder` (rnn.h)

**USED BY**:
- Character-level language models

**CORE OR AUX**: **EO** - Experimental / Optional

**REBUILD PHASE**: **v2** - Research feature

**WHAT BREAKS**: Cannot do character-level modeling

---

### FILE: dynet/pretrain.cc, dynet/pretrain.h
**ROLE**: Utilities for loading pretrained embeddings.

**DEPENDS ON**:
- `LookupParameter` (model.h)

**USED BY**:
- NLP models - initialize embeddings from Word2Vec, GloVe, etc.

**CORE OR AUX**: **DE** - Developer ergonomics

**REBUILD PHASE**: **v1** - After LookupParameters exist

**WHAT BREAKS**: Manual embedding initialization

---

### FILE: dynet/saxe-init.h, dynet/saxe-init.cc
**ROLE**: Saxe random orthogonal matrix initialization.

**DEPENDS ON**:
- `Tensor` (tensor.h)

**USED BY**:
- `ParameterInit` (param-init.h)

**CORE OR AUX**: **EO** - Experimental initialization

**REBUILD PHASE**: **v2** - Research feature

**WHAT BREAKS**: Cannot use Saxe initialization (Glorot works fine)

---

## PYTHON BINDINGS

### FILE: python/* (entire directory)
**ROLE**: Python API wrapping C++ core via Cython.

**DEPENDS ON**:
- Entire C++ core

**USED BY**:
- Python users - primary interface for many

**CORE OR AUX**: **DE** - Developer ergonomics (language binding)

**INVARIANTS**:
- Mirrors C++ API structure
- Adds Pythonic conveniences (operator overloading, list comprehensions)

**REBUILD PHASE**: **v1** - After C++ core stable

**WHAT BREAKS**: No Python interface; C++-only framework

**NOTES FOR REBUILDING**:
- v0: C++ only
- v1: Add Python bindings
- Critical for adoption but not core functionality

---

## TESTING INFRASTRUCTURE

### FILE: tests/* (entire directory)
**ROLE**: Unit tests and integration tests.

**DEPENDS ON**:
- DyNet core

**USED BY**:
- CI/CD - validate changes
- Developers - verify implementations

**CORE OR AUX**: **DV** - Debug / Validation

**REBUILD PHASE**: **v0** onwards - TDD approach

**WHAT BREAKS**: No automated validation

**NOTES FOR REBUILDING**:
- v0: Add basic tests (forward/backward correctness)
- v1: Add comprehensive test suite
- Critical for confidence but not runtime functionality

---

## BUILD SYSTEM

### FILE: CMakeLists.txt (root and subdirectories)
**ROLE**: Build configuration (CMake).

**DEPENDS ON**:
- CMake, compilers

**USED BY**:
- Developers - build DyNet

**CORE OR AUX**: **DE** - Developer ergonomics

**REBUILD PHASE**: **v0** - Build from day 1

**WHAT BREAKS**: Cannot compile

**NOTES FOR REBUILDING**:
- v0: Simple CMake (CPU-only, minimal dependencies)
- v1: Add CUDA, Eigen, cuDNN detection
- v2: Add installation, packaging

---

## THIRD-PARTY DEPENDENCIES

### FILE: third_party/* (entire directory)
**ROLE**: External libraries (Eigen, etc.).

**DEPENDS ON**:
- Nothing (vendored code)

**USED BY**:
- DyNet core - tensor operations

**CORE OR AUX**: **CP** - Core dependencies

**REBUILD PHASE**: **v0** - Eigen essential for tensor ops

**WHAT BREAKS**: No tensor operations (unless implementing from scratch)

**NOTES FOR REBUILDING**:
- v0: Vendor Eigen (header-only)
- v1: Add other dependencies as needed
- Could replace Eigen with custom tensor lib (major effort)

---

## DOCUMENTATION

### FILE: doc/* (entire directory)
**ROLE**: User documentation (Sphinx).

**DEPENDS ON**:
- Sphinx, RST files

**USED BY**:
- Users - learn DyNet

**CORE OR AUX**: **DE** - Developer ergonomics (documentation)

**REBUILD PHASE**: **v1** - After API stabilizes

**WHAT BREAKS**: No user documentation

**NOTES FOR REBUILDING**:
- v0: README only
- v1: Add comprehensive docs
- Essential for adoption but not functionality

---

## EXAMPLES

### FILE: examples/* (entire directory)
**ROLE**: Example applications (XOR, MNIST, language modeling, etc.).

**DEPENDS ON**:
- DyNet core

**USED BY**:
- Users - learning and templates

**CORE OR AUX**: **DE** - Developer ergonomics (examples)

**REBUILD PHASE**: **v1** - After core features work

**WHAT BREAKS**: Harder to learn DyNet

**NOTES FOR REBUILDING**:
- v0: Skip
- v1: Add 2-3 simple examples (XOR, MNIST)
- v2: Add comprehensive examples

---

## REBUILD ROADMAP SUMMARY

### **v0: Minimal Viable Framework** (Core Primitives)
**Goal**: Single-threaded, CPU-only, simple graphs

**Must Have**:
1. ✅ `ComputationGraph` - DAG storage
2. ✅ `Expression` - user API
3. ✅ `Tensor` - data container (Eigen or raw arrays)
4. ✅ `Dim` - shape representation
5. ✅ `SimpleExecutionEngine` - forward/backward
6. ✅ `ParameterCollection`, `Parameter` - trainable weights
7. ✅ `ParameterNode`, `InputNode` - leaf nodes
8. ✅ Basic arithmetic nodes (add, multiply, matmul)
9. ✅ `Softmax`, one loss function
10. ✅ `SimpleSGDTrainer` - gradient descent
11. ✅ `Device_CPU` with 4 memory pools
12. ✅ `AlignedMemoryPool` - linear allocator
13. ✅ `CPUAllocator` - raw memory
14. ✅ `initialize()`, `cleanup()` - framework setup

**Can Skip**:
- GPU support
- RNN builders
- Advanced optimizers
- Serialization
- Python bindings
- cuDNN
- Most node types

**Test**: Train XOR or simple regression

---

### **v1: Practical Framework** (Orchestration + Ergonomics)
**Goal**: Production-ready for common tasks

**Add**:
1. ✅ `LookupParameter` - sparse embeddings
2. ✅ `RNNBuilder` interface
3. ✅ `LSTMBuilder`, `GRUBuilder` - sequence models
4. ✅ Full arithmetic, activation, loss nodes
5. ✅ `AdamTrainer` - modern optimizer
6. ✅ `ShadowParameters` - optimizer state
7. ✅ I/O system - save/load models
8. ✅ `Dict` - vocabulary management
9. ✅ `ParameterInit` variants (Glorot)
10. ✅ Python bindings
11. ✅ GPU support (`Device_GPU`, cuBLAS)
12. ✅ Basic gradient checking
13. ✅ Documentation

**Test**: Train language model, sequence labeling

---

### **v2: Advanced Framework** (Optimizations + Research)
**Goal**: Competitive with PyTorch/TensorFlow

**Add**:
1. ✅ `BatchedExecutionEngine` - autobatching
2. ✅ CNN support (Conv2D, Pooling)
3. ✅ cuDNN integration
4. ✅ Fused operations (PickNegLogSoftmax, LSTM cells)
5. ✅ TreeLSTM builders
6. ✅ Hierarchical softmax, class-factored softmax
7. ✅ Full optimizer suite (Adagrad, RMSProp, etc.)
8. ✅ Weight decay, advanced regularization
9. ✅ Timing/profiling tools
10. ✅ Deep LSTM, Fast LSTM
11. ✅ Character-to-word composition
12. ✅ Comprehensive examples
13. ✅ Performance benchmarks

**Test**: State-of-the-art results on NLP benchmarks

---

## KEY ARCHITECTURAL INSIGHTS

### **1. Memory Management Philosophy**
- **Linear allocation + bulk freeing**: DyNet's core optimization
- 4 separate pools prevent fragmentation, enable batching
- Tensors don't own memory (lifecycle managed by graph/device)
- Critical for automatic differentiation efficiency

### **2. Computation Graph Design**
- **Dynamic graphs**: Rebuilt per example (unlike static TensorFlow v1)
- **Topological order**: Maintained in nodes vector
- **Expression as handle**: Lightweight wrapper, not data container
- Enables flexible control flow (if/else, loops)

### **3. Node Implementation Pattern**
- **Device dispatch**: Abstract forward/backward with CPU/GPU variants
- **Gradient accumulation**: +=, not =, due to DAG fan-in
- **No allocation in ops**: Pre-allocated pools
- Macros reduce boilerplate while maintaining clarity

### **4. RNN Builder Abstraction**
- **State machine enforcement**: Prevents API misuse
- **Expression-based composition**: Natural sequence building
- **Flexible recurrence**: Supports trees, graphs, not just chains
- Critical for NLP applications

### **5. Training Loop Separation**
- **Graph construction ≠ parameter storage**: Parameters persist across graphs
- **Sparse updates**: LookupParameters only update non-zero gradients
- **Shadow parameters**: Keep optimizer state separate from model
- Enables flexible training strategies

### **6. Extensibility Points**
- **New nodes**: Subclass Node, implement forward/backward
- **New optimizers**: Subclass Trainer, implement update_rule
- **New RNNs**: Subclass RNNBuilder
- **New initializers**: Subclass ParameterInit
- Well-defined extension APIs

---

## CRITICAL DEPENDENCIES (MUST NOT BREAK)

1. **Topological order in ComputationGraph**: Everything depends on this
2. **Gradient accumulation in backward**: Correctness of all gradients
3. **Memory pool lifecycle**: Graph invalidation frees all intermediate tensors
4. **Expression staleness checking**: Prevents use of stale graph references
5. **Device abstraction**: CPU/GPU dispatch mechanism
6. **Batch dimension separation**: Enables efficient minibatching

---

## END OF ARCHITECTURAL ANALYSIS

This document provides a blueprint for rebuilding DyNet. Follow the v0 → v1 → v2 roadmap, implementing components in dependency order. Start with core primitives (memory, tensors, graph), add orchestration (expressions, execution), then optimize (GPU, autobatching, cuDNN).
