.. _fluxions_research:

Fluxions: DyNet + mlpack + WebGPU Research Analysis
====================================================

This page links to the full research analysis document for *Fluxions* — a
differentiable-programming system that leverages DyNet's dynamic autograd engine,
mlpack's classical-ML toolkit, and (in Phase 3) WebGPU compute for cross-platform
hardware acceleration.

The full document is maintained as a Markdown file at
``doc/fluxions_dynet_mlpack_webgpu.md`` in the repository root.

.. rubric:: Sections

* **What Is Fluxions?** — scope and naming rationale
* **DyNet Capabilities** — computation graph, autobatching, memory pools, trainers
* **mlpack Capabilities** — data I/O, spatial indices, ANN framework, classical ML
* **Integration Architecture** — data-convention bridge, feature-pipeline pattern, hybrid training loop, serialisation strategy
* **Leverage Points per mlpack Layer** — concrete mapping from mlpack architectural layers to Fluxions use cases
* **Phase 3: WebGPU Acceleration** — execution engine replacement, WGSL BLAS kernels, memory alignment, implementation sub-phases
* **Risk Register** — identified risks and mitigations
* **Decision Matrix** — architectural trade-offs and recommendations
* **Global Invariants** — invariants that must survive all integration phases

.. seealso::

   :doc:`dynet_vs_pytorch`
      Comparison of DyNet and PyTorch design philosophies.

   The mlpack architecture reference at ``doc/mlpack.md`` contains
   ``NOTES FOR FLUXIONS:`` annotations throughout that informed this analysis.

   ``ARCHITECTURAL_ANALYSIS.md`` provides the file-by-file DyNet analysis.
