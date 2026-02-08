.. _dynet_vs_pytorch:

DyNet vs PyTorch: A Comprehensive Comparison
=============================================

This document provides a detailed comparison between DyNet and PyTorch, two popular deep learning frameworks. Both are powerful tools, but they have different design philosophies and strengths that make them suitable for different use cases.

Overview
--------

**DyNet (Dynamic Neural Network Toolkit)**

* Developed by Carnegie Mellon University and collaborators
* Written in C++ with Python bindings
* Designed for **dynamic networks** that change structure for each training instance
* Particularly strong for **natural language processing** tasks
* Emphasizes efficiency with **dynamic computation graphs**

**PyTorch**

* Developed by Facebook AI Research (FAIR) and community
* Written in Python with C++ backend
* General-purpose deep learning framework
* Strong in both research and production
* Large ecosystem and community support

Key Differences
---------------

1. Dynamic Computation Graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**DyNet:**

* Built from the ground up for **truly dynamic** graphs
* Each training instance can have a completely different graph structure
* Optimized for cases where structure varies significantly (e.g., parsing, tree-structured networks)
* Explicit graph construction with ``dy.renew_cg()``

**PyTorch:**

* Also supports dynamic graphs (define-by-run)
* More optimized for graphs with similar structures across batches
* Implicit graph construction during forward pass

2. Automatic Batching
^^^^^^^^^^^^^^^^^^^^^

**DyNet:**

* **Automatic batching** (``--dynet-autobatch 1``) is a unique feature
* DyNet can automatically batch operations even when structures differ
* No need to manually pad sequences or structure batches
* Particularly powerful for variable-length or tree-structured inputs

**PyTorch:**

* Manual batching required
* User must handle padding, masking, and batch dimensions
* More control but requires more boilerplate code
* Uses ``PackedSequence`` for variable-length sequences

3. API Design
^^^^^^^^^^^^^

**DyNet:**

.. code-block:: python

    import dynet as dy
    
    # Create parameter collection
    m = dy.ParameterCollection()
    W = m.add_parameters((output_dim, input_dim))
    b = m.add_parameters(output_dim)
    
    # Build computation graph
    dy.renew_cg()
    x = dy.inputVector([1.0, 2.0, 3.0])
    W_expr = dy.parameter(W)
    b_expr = dy.parameter(b)
    y = W_expr * x + b_expr

**PyTorch:**

.. code-block:: python

    import torch
    import torch.nn as nn
    
    # Define model as a class
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel(3, 5)
    x = torch.tensor([1.0, 2.0, 3.0])
    y = model(x)

4. Performance Characteristics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**DyNet:**

* Optimized for variable-structure networks
* Lower memory overhead for dynamic graphs
* Efficient for small to medium batch sizes
* Auto-batching can match or exceed manual batching in many cases

**PyTorch:**

* Highly optimized for tensor operations
* Better performance for large, uniform batches
* Extensive hardware optimizations (CUDA, cuDNN)
* Generally faster for standard architectures (CNNs, standard RNNs)

See :ref:`benchmarks` for detailed performance comparisons.

5. Ecosystem and Community
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**DyNet:**

* Focused community, particularly in NLP research
* Used in many academic NLP projects
* Smaller ecosystem of pre-trained models
* Excellent for research in dynamic architectures

**PyTorch:**

* Large, active community across all ML domains
* Extensive libraries (torchvision, torchaudio, etc.)
* Many pre-trained models available (PyTorch Hub)
* Strong industry adoption

When to Use DyNet
-----------------

DyNet is particularly well-suited for:

1. **Natural Language Processing Tasks**
   
   * Syntactic parsing (constituency, dependency)
   * Tree-structured networks
   * Graph neural networks with varying structure
   * Morphological analysis
   * Machine translation with attention

2. **Variable-Structure Networks**
   
   * When each example has a different computational graph
   * Recursive neural networks
   * Models with data-dependent control flow

3. **Research in Dynamic Architectures**
   
   * Prototyping new architectures with complex dynamics
   * When automatic batching is valuable
   * When you want to focus on model design rather than batching logic

When to Use PyTorch
-------------------

PyTorch is particularly well-suited for:

1. **Standard Deep Learning Architectures**
   
   * Convolutional Neural Networks (CNNs)
   * Standard RNNs, LSTMs, GRUs
   * Transformers (with uniform structure)
   * GANs, VAEs

2. **Production Deployment**
   
   * TorchScript for model export
   * ONNX support
   * Mobile deployment (PyTorch Mobile)
   * Large-scale training with distributed computing

3. **Transfer Learning**
   
   * Using pre-trained models (BERT, ResNet, etc.)
   * Fine-tuning on custom datasets
   * Leveraging the extensive model zoo

4. **Multi-Domain Projects**
   
   * Computer vision
   * Reinforcement learning
   * Audio processing
   * Multi-modal learning

.. _benchmarks:

Performance Benchmarks
----------------------

MNIST Classification
^^^^^^^^^^^^^^^^^^^^

The following benchmark compares DyNet and PyTorch on MNIST digit classification using identical network architectures. See the full implementation in ``examples/mnist/basic-mnist-benchmarks/``.

**Network Architecture:**

* Conv2D (32 filters, 5x5 kernel)
* MaxPool2D (2x2)
* ReLU
* Conv2D (64 filters, 5x5 kernel)
* MaxPool2D (2x2)
* ReLU
* Fully Connected (1024 units)
* Dropout (0.4)
* Fully Connected (10 units, output)

**Results (Batch size: 64, Learning rate: 0.01, 20 epochs):**

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - OS
     - Device
     - Framework
     - Speed
     - Accuracy
   * - Ubuntu 16.04
     - GeForce GTX 1080 Ti
     - PyTorch
     - ~4.49±0.11 s/epoch
     - 98.95%
   * - Ubuntu 16.04
     - GeForce GTX 1080 Ti
     - DyNet (autobatch)
     - ~8.58±0.09 s/epoch
     - 98.98%
   * - Ubuntu 16.04
     - GeForce GTX 1080 Ti
     - DyNet (minibatch)
     - ~4.13±0.13 s/epoch
     - 98.99%

**Key Observations:**

* For this standard CNN task, DyNet with manual minibatching is competitive with PyTorch
* DyNet's automatic batching has some overhead but achieves similar accuracy
* Both frameworks achieve excellent accuracy (>98.9%)
* For standard architectures with uniform batches, performance is comparable

Code Examples
-------------

Simple Feed-Forward Network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**DyNet:**

.. code-block:: python

    import dynet as dy
    import numpy as np
    
    # Create model
    m = dy.ParameterCollection()
    W1 = m.add_parameters((128, 784))
    b1 = m.add_parameters(128)
    W2 = m.add_parameters((10, 128))
    b2 = m.add_parameters(10)
    
    # Training loop
    trainer = dy.AdamTrainer(m)
    
    for epoch in range(num_epochs):
        for x_batch, y_batch in data_loader:
            dy.renew_cg()
            losses = []
            
            for x, y in zip(x_batch, y_batch):
                # Forward pass
                x_var = dy.inputVector(x)
                W1_expr = dy.parameter(W1)
                b1_expr = dy.parameter(b1)
                h = dy.tanh(W1_expr * x_var + b1_expr)
                
                W2_expr = dy.parameter(W2)
                b2_expr = dy.parameter(b2)
                y_pred = W2_expr * h + b2_expr
                
                # Compute loss
                loss = dy.pickneglogsoftmax(y_pred, y)
                losses.append(loss)
            
            # Backward pass
            batch_loss = dy.average(losses)
            batch_loss.backward()
            trainer.update()

**PyTorch:**

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # Define model
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            h = torch.tanh(self.fc1(x))
            return self.fc2(h)
    
    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(num_epochs):
        for x_batch, y_batch in data_loader:
            # Forward pass
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

Variable-Length RNN (with Auto-batching)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**DyNet (with automatic batching):**

.. code-block:: python

    import dynet as dy
    
    # Model with automatic batching
    m = dy.ParameterCollection()
    lstm = dy.LSTMBuilder(1, input_dim, hidden_dim, m)
    W = m.add_parameters((output_dim, hidden_dim))
    
    trainer = dy.AdamTrainer(m)
    
    # Enable autobatch via command line: --dynet-autobatch 1
    for epoch in range(num_epochs):
        for sentences, labels in minibatches:
            dy.renew_cg()
            losses = []
            
            # Each sentence can have different length!
            for sentence, label in zip(sentences, labels):
                # Build LSTM for this sentence
                s = lstm.initial_state()
                for word in sentence:
                    word_emb = dy.lookup(embeddings, word)
                    s = s.add_input(word_emb)
                
                # Predict
                h = s.output()
                y_pred = dy.parameter(W) * h
                loss = dy.pickneglogsoftmax(y_pred, label)
                losses.append(loss)
            
            # DyNet automatically batches compatible operations!
            batch_loss = dy.average(losses)
            batch_loss.backward()
            trainer.update()

**PyTorch (manual padding required):**

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    
    class RNNClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
            super(RNNClassifier, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, sentences, lengths):
            # Embed
            embedded = self.embedding(sentences)
            
            # Pack sequences (handle variable lengths)
            packed = pack_padded_sequence(embedded, lengths, 
                                         batch_first=True, 
                                         enforce_sorted=False)
            
            # LSTM
            packed_output, (h, c) = self.lstm(packed)
            
            # Use final hidden state
            return self.fc(h[-1])
    
    model = RNNClassifier(vocab_size, emb_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        for padded_sentences, lengths, labels in data_loader:
            # Forward pass
            outputs = model(padded_sentences, lengths)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

Migration Guide
---------------

If you're considering switching between frameworks:

From PyTorch to DyNet
^^^^^^^^^^^^^^^^^^^^^

**Consider migrating if:**

* You're working on NLP tasks with variable-length sequences
* You want to avoid manual padding and batching logic
* Your model has data-dependent structure
* You're implementing recursive or tree-structured networks

**Key changes:**

1. Replace ``nn.Module`` classes with function-based computation graphs
2. Add ``dy.renew_cg()`` at the start of each training instance
3. Convert ``dy.parameter()`` calls for each parameter use
4. Use ``dy.inputVector`` / ``dy.inputTensor`` for input data
5. Enable ``--dynet-autobatch 1`` for automatic batching

From DyNet to PyTorch
^^^^^^^^^^^^^^^^^^^^^

**Consider migrating if:**

* You need pre-trained models (BERT, ResNet, etc.)
* You want better support for production deployment
* You're working outside NLP (computer vision, RL, etc.)
* You need a larger ecosystem and community

**Key changes:**

1. Refactor computation graphs into ``nn.Module`` classes
2. Remove ``dy.renew_cg()`` calls
3. Handle batching explicitly (padding, masking)
4. Use ``torch.tensor()`` for input data
5. Restructure code to use PyTorch's object-oriented API

Conclusion
----------

Both DyNet and PyTorch are excellent frameworks with different strengths:

* **Choose DyNet** for NLP tasks with variable structures, tree-based models, or when you want automatic batching for dynamic graphs
* **Choose PyTorch** for general-purpose deep learning, production deployment, transfer learning, or when you need a large ecosystem

The best choice depends on your specific use case, team expertise, and project requirements.

Additional Resources
--------------------

**DyNet:**

* `DyNet Documentation <http://dynet.readthedocs.io/>`_
* `DyNet GitHub Repository <https://github.com/clab/dynet>`_
* `DyNet Tutorial Examples <https://github.com/clab/dynet_tutorial_examples>`_
* `Technical Report <https://arxiv.org/abs/1701.03980>`_

**PyTorch:**

* `PyTorch Documentation <https://pytorch.org/docs/>`_
* `PyTorch Tutorials <https://pytorch.org/tutorials/>`_
* `PyTorch GitHub Repository <https://github.com/pytorch/pytorch>`_
* `PyTorch Examples <https://github.com/pytorch/examples>`_

**Comparisons:**

* :ref:`minibatching` - DyNet's minibatching and autobatching documentation
* `MNIST Benchmark Examples <https://github.com/clab/dynet/tree/master/examples/mnist/basic-mnist-benchmarks>`_
