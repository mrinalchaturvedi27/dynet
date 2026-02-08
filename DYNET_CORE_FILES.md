# DyNet Computation Graph and Autograd System: Core Files

This document lists the 5-7 files that form the backbone of DyNet's computation graph and automatic differentiation (autograd) system, ranked by conceptual importance.

## Ranking by Conceptual Importance

### 1. **dynet/dynet.h** (~835 lines)
**Conceptual Role**: The heart of DyNet's computation graph architecture.

This is the most conceptually important file as it defines:
- `ComputationGraph`: The core data structure representing the entire computational graph where nodes represent intermediate values and edges represent operations
- `Node`: The abstract base class for all operations, defining the interface for forward and backward passes
- Graph management functions and the overall framework that ties everything together
- Memory management and device handling infrastructure

**Why #1**: This file establishes the fundamental abstractions that make DyNet work. Without understanding `ComputationGraph` and `Node`, you cannot understand how DyNet builds and executes neural networks.

### 2. **dynet/expr.h** (~2,643 lines)
**Conceptual Role**: The user-facing API for building computation graphs.

This file defines:
- `Expression`: The primary user-facing abstraction that wraps nodes in the computation graph
- All high-level operations (arithmetic, losses, activations, convolutions, etc.) that users call to build networks
- The declarative interface that makes DyNet easy to use

**Why #2**: While `dynet.h` defines the internal machinery, `expr.h` provides the API that users actually interact with. It bridges the gap between user intent and graph construction. Every DyNet program uses Expressions extensively.

### 3. **dynet/exec.h** + **dynet/exec.cc** (~108 + ~1,127 lines)
**Conceptual Role**: The execution engines that perform forward and backward passes.

These files define:
- `ExecutionEngine`: Abstract base class for executing computation graphs
- `SimpleExecutionEngine`: Straightforward sequential execution
- `BatchedExecutionEngine`: Sophisticated auto-batching execution for performance
- The core logic for traversing the graph and computing gradients (autograd)

**Why #3**: This is where the "magic" of automatic differentiation happens. The execution engines orchestrate the forward pass (computing values) and backward pass (computing gradients), which is the essence of training neural networks.

### 4. **dynet/tensor.h** + **dynet/tensor.cc** (~410 + ~643 lines)
**Conceptual Role**: The fundamental data container for all computations.

These files define:
- `Tensor`: The core data structure that holds multidimensional arrays
- Memory management and device abstraction (CPU/GPU)
- Integration with Eigen for efficient linear algebra operations
- Batch handling mechanisms

**Why #4**: Every node in the computation graph operates on Tensors. This is the foundational data structure that flows through the entire system during both forward and backward passes.

### 5. **dynet/nodes-def-macros.h** (~64 lines)
**Conceptual Role**: Infrastructure for defining new node types.

This file provides:
- `DYNET_NODE_DEFINE_DEV_IMPL()`: A critical macro that standardizes how nodes are implemented
- Device dispatch mechanisms to handle CPU vs GPU execution
- Common utilities for node implementations

**Why #5**: While small, this file is architecturally crucial. It establishes the pattern that all ~30+ node implementation files follow (nodes-activations.h/cc, nodes-matrixmultiply.h/cc, etc.). Understanding this macro is key to understanding how to extend DyNet with new operations.

### 6. **dynet/param-nodes.h** + **dynet/param-nodes.cc** (~114 + ~342 lines)
**Conceptual Role**: Bridge between learnable parameters and the computation graph.

These files define:
- `ParameterNode`: Represents trainable parameters in the graph
- `InputNode`: Represents user-provided input data
- `ParameterNodeBase`: Interface for gradient accumulation
- The mechanism for connecting the parameter storage (from model.h) to the computation graph

**Why #6**: Parameters are special nodes in the graph that need gradient accumulation for training. This file defines how learnable weights connect to the autograd system, which is fundamental to neural network training.

### 7. **dynet/model.h** + **dynet/model.cc** (~762 + ~873 lines)
**Conceptual Role**: Parameter storage and management.

These files define:
- `ParameterCollection`: Container for all trainable parameters
- `Parameter` and `LookupParameter`: Individual parameter objects
- Parameter initialization and serialization
- The storage backend that `ParameterNode` references

**Why #7**: While not directly part of the graph execution, this file manages the persistent state (weights) that the computation graph operates on. Every training loop involves parameters from model.h being inserted into the graph via param-nodes.h.

## Summary

The **computational flow** through these files:

1. Users create **Expressions** (expr.h) which internally create **Nodes** (dynet.h) in a **ComputationGraph** (dynet.h)
2. Nodes operate on **Tensors** (tensor.h) and may reference **Parameters** (model.h via param-nodes.h)
3. The **ExecutionEngine** (exec.h/cc) traverses the graph to:
   - Perform forward computation (calling each Node's `forward_impl`)
   - Perform backward computation for autograd (calling each Node's `backward_impl`)
4. Node implementations follow patterns defined by **nodes-def-macros.h**

## Additional Context

While these 7 files form the core, the complete system includes:
- ~30 node implementation files (nodes-*.h/cc) that define specific operations
- Device management (devices.h/cc) for CPU/GPU coordination
- Training algorithms (training.h/cc) that use the gradients computed by autograd
- Memory management (aligned-mem-pool.h/cc) for efficient allocation

Understanding these 7 core files provides approximately 80% of the conceptual foundation needed to understand how DyNet builds, executes, and trains neural networks.
