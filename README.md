# Awesome Tensor Compilers
![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/merrymercy/awesome-tensor-compilers/graphs/commit-activity)

A list of awesome compiler projects and papers for tensor computation and deep learning. 

## Contents
- [Awesome Tensor Compilers](#awesome-tensor-compilers)
  - [Contents](#contents)
  - [Open Source Projects](#open-source-projects)
  - [Papers](#papers)
    - [Survey](#survey)
    - [Compiler](#compiler)
    - [Auto-tuning and Auto-scheduling](#auto-tuning-and-auto-scheduling)
    - [Cost Model](#cost-model)
    - [CPU & GPU Optimization](#cpu-and-gpu-optimization)
    - [NPU Optimization](#npu-optimization)
    - [Graph-level Optimization](#graph-level-optimization)
    - [Dynamic Model](#dynamic-model)
    - [Graph Neural Networks](#graph-neural-networks)
    - [Distributed Computing](#distributed-computing)
    - [Quantization and Sparsification](#quantization-and-sparsification)
    - [Program Rewriting](#program-rewriting)
  - [Tutorials](#tutorials)
  - [Contribute](#contribute)

## Open Source Projects
- [TVM:  An End to End Deep Learning Compiler Stack ](https://tvm.apache.org/)
- [Halide: A Language for Fast, Portable Computation on Images and Tensors](https://halide-lang.org/)
- [TensorComprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions](https://facebookresearch.github.io/TensorComprehensions/)
- [Tiramisu: A Polyhedral Compiler for Expressing Fast and Portable Code](http://tiramisu-compiler.org/)
- [XLA: Optimizing Compiler for Machine Learning](https://www.tensorflow.org/xla)
- [MLIR: Multi-Level Intermediate Representation](https://mlir.llvm.org/)
- [Hummingbird: Compiling Trained ML Models into Tensor Computation](https://github.com/microsoft/hummingbird)
- [nnfusion: A Flexible and Efficient Deep Neural Network Compiler](https://github.com/microsoft/nnfusion)
- [nGraph: An Open Source C++ library, compiler and runtime for Deep Learning](https://www.ngraph.ai/)
- [PlaidML: A Platform for Making Deep Learning Work Everywhere](https://www.intel.com/content/www/us/en/artificial-intelligence/plaidml.html)
- [Glow: Compiler for Neural Network Hardware Accelerators](https://github.com/pytorch/glow)
- [TACO: The Tensor Algebra Compiler](http://tensor-compiler.org/)
- [TASO: The Tensor Algebra SuperOptimizer for Deep Learning](https://github.com/jiazhihao/TASO)
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://github.com/ptillet/triton)
- [DLVM: Modern Compiler Infrastructure for Deep Learning Systems](https://dlvm-team.github.io/)
- [NN-512: A compiler that generates C99 code for neural net inference](https://nn-512.com/)
- [BladeDISC: An end-to-end DynamIc Shape Compiler project for machine learning workloads.](https://github.com/alibaba/BladeDISC)

## Papers

### Survey
- [The Deep Learning Compiler: A Comprehensive Survey](https://arxiv.org/abs/2002.03794) by Mingzhen Li et al., TPDS 2020
- [An In-depth Comparison of Compilers for DeepNeural Networks on Hardware](https://ieeexplore.ieee.org/document/8782480) by Yu Xing et al., ICESS 2019

### Compiler
- [Roller: Fast and Efficient Tensor Compilation for Deep Learning](https://www.microsoft.com/en-us/research/publication/roller-fast-and-efficient-tensor-compilation-for-deep-learning/) by Hongyu Zhu et al., OSDI 2022
- [AStitch: Enabling a New Multi-dimensional Optimization Space for Memory-Intensive ML Training and Inference on Modern SIMT Architectures](https://dl.acm.org/doi/10.1145/3503222.3507723) by Zhen Zheng et al., ASPLOS 2022
- [Composable and Modular Code Generation in MLIR: A Structured and Retargetable Approach to Tensor Compiler Construction](https://arxiv.org/pdf/2202.03293.pdf) by Nicolas Vasilache et al., ArXiv 2022
- [Compiler Support for Sparse Tensor Computations in MLIR](https://arxiv.org/pdf/2202.04305.pdf) by Aart J.C. Bik et al., ArXiv 2022
- [DeepCuts: A deep learning optimization framework for versatile GPU workloads](https://pldi21.sigplan.org/details/pldi-2021-papers/13/DeepCuts-A-Deep-Learning-Optimization-Framework-for-Versatile-GPU-Workloads) by Wookeun Jung et al., PLDI 2021
- [PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections](https://www.usenix.org/conference/osdi21/presentation/wang) by Haojie Wang et al., OSDI 2021
- [MLIR: Scaling Compiler Infrastructure for Domain Specific Computation](https://research.google/pubs/pub49988/) by Chris Lattner et al., CGO 2021
- [A Tensor Compiler for Unified Machine Learning Prediction Serving](https://www.usenix.org/conference/osdi20/presentation/nakandala) by Supun Nakandala et al., OSDI 2020
- [Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks](https://www.usenix.org/conference/osdi20/presentation/ma) by Lingxiao Ma et al., OSDI 2020
- [TASO: The Tensor Algebra SuperOptimizer for Deep Learning](https://dl.acm.org/doi/abs/10.1145/3341301.3359630) by Zhihao Jia et al., SOSP 2019
- [Tiramisu: A polyhedral compiler for expressing fast and portable code](https://arxiv.org/abs/1804.10694) by Riyadh Baghdadi et al., CGO 2019
- [Triton: an intermediate language and compiler for tiled neural network computations](https://dl.acm.org/doi/abs/10.1145/3315508.3329973) by Philippe Tillet et al., MAPL 2019
- [Relay: A High-Level Compiler for Deep Learning](https://arxiv.org/abs/1904.08368) by Jared Roesch et al., arXiv 2019
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) by Tianqi Chen et al., OSDI 2018
- [Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions](https://arxiv.org/abs/1802.04730) by Nicolas Vasilache et al., arXiv 2018
- [Intel nGraph: An Intermediate Representation, Compiler, and Executor for Deep Learning](https://arxiv.org/abs/1801.08058) by Scott Cyphers et al., arXiv 2018
- [Glow: Graph Lowering Compiler Techniques for Neural Networks](https://arxiv.org/abs/1805.00907) by Nadav Rotem et al., arXiv 2018
- [DLVM: A modern compiler infrastructure for deep learning systems](https://arxiv.org/abs/1711.03016) by Richard Wei et al., arXiv 2018
- [Diesel: DSL for linear algebra and neural net computations on GPUs](https://dl.acm.org/doi/abs/10.1145/3211346.3211354) by Venmugil Elango et al., MAPL 2018
- [The Tensor Algebra Compiler](https://dl.acm.org/doi/abs/10.1145/3133901) by Fredrik Kjolstad et al., OOPSLA 2017
- [Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines](http://people.csail.mit.edu/jrk/halide-pldi13.pdf) by Jonathan Ragan-Kelley et al., PLDI 2013


### Auto-tuning and Auto-scheduling
- [A Full-Stack Search Technique for Domain Optimized Deep Learning Accelerators](https://dl.acm.org/doi/10.1145/3503222.3507767) by Dan Zhang et al., ASPLOS 2022
- [Lorien: Efficient Deep Learning Workloads Delivery](https://dl.acm.org/doi/abs/10.1145/3472883.3486973) by Cody Hao Yu et al., SoCC 2021
- [Value Learning for Throughput Optimization of Deep Neural Networks](https://proceedings.mlsys.org/paper/2021/hash/73278a4a86960eeb576a8fd4c9ec6997-Abstract.html) by Benoit Steiner et al., MLSys 2021
- [A Flexible Approach to Autotuning Multi-Pass Machine Learning Compilers](https://mangpo.net/papers/xla-autotuning-pact2021.pdf) by Phitchaya Mangpo Phothilimthana et al., PACT 2021
- [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://arxiv.org/abs/2006.06762) by Lianmin Zheng et al., OSDI 2020
- [Schedule Synthesis for Halide Pipelines on GPUs](https://dl.acm.org/doi/fullHtml/10.1145/3406117) by Sioutas Savvas et al., TACO 2020
- [FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System](https://dl.acm.org/doi/abs/10.1145/3373376.3378508) by Size Zheng et al., ASPLOS 2020
- [ProTuner: Tuning Programs with Monte Carlo Tree Search](https://arxiv.org/abs/2005.13685) by Ameer Haj-Ali et al., arXiv 2020
- [AdaTune: Adaptive tensor program compilation made efficient](https://www.microsoft.com/en-us/research/uploads/prod/2020/10/nips20adatune.pdf) by Menghao Li et al., NeurIPS 2020
- [Optimizing the Memory Hierarchy by Compositing Automatic Transformations on Computations and Data](https://www.microarch.org/micro53/papers/738300a427.pdf) by Jie Zhao et al., MICRO 2020
- [Chameleon: Adaptive Code Optimization for Expedited Deep Neural Network Compilation](https://openreview.net/forum?id=rygG4AVFvH) by Byung Hoon Ahn et al., ICLR 2020
- [A Sparse Iteration Space Transformation Framework for Sparse Tensor Algebra](http://tensor-compiler.org/senanayake-oopsla20-taco-scheduling.pdf) by Ryan Senanayake et al. OOPSLA 2020
- [Learning to Optimize Halide with Tree Search and Random Programs](https://halide-lang.org/papers/autoscheduler2019.html) by Andrew Adams et al., SIGGRAPH 2019
- [Learning to Optimize Tensor Programs](https://arxiv.org/abs/1805.08166) by Tianqi Chen et al., NeurIPS 2018
- [Automatically Scheduling Halide Image Processing Pipelines](http://graphics.cs.cmu.edu/projects/halidesched/) by Ravi Teja Mullapudi et al., SIGGRAPH 2016

### Cost Model
- [An Asymptotic Cost Model for Autoscheduling Sparse Tensor Programs](https://peterahrens.io/assets/documents/ahrens_asymptotic_2021.pdf) by Peter Ahrens et al., PLDI 2022
- [TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/a684eceee76fc522773286a895bc8436-Abstract-round1.html) by Lianmin Zheng., NeurIPS 2021
- [A Deep Learning Based Cost Model for Automatic Code Optimization](https://proceedings.mlsys.org/paper/2021/hash/3def184ad8f4755ff269862ea77393dd-Abstract.html) by Riyadh Baghdadi et al., MLSys 2021
- [A Learned Performance Model for the Tensor Processing Unit](https://arxiv.org/abs/2008.01040) by Samuel J. Kaufman et al., MLSys 2021
- [DYNATUNE: Dynamic Tensor Program Optimization in Deep Neural Network Compilation](https://openreview.net/pdf/f2330b850544ed7b0157ff0411638fd7ee8aefc0.pdf) by Minjia Zhang et al., ICLR 2021
- [MetaTune: Meta-Learning Based Cost Model for Fast and Efficient Auto-tuning Frameworks](https://arxiv.org/abs/2102.04199) by Jaehun Ryu et al., arxiv 2021

### CPU and GPU Optimization
- [Analytical characterization and design space exploration for optimization of CNNs](https://dl.acm.org/doi/abs/10.1145/3445814.3446759) by Rui Li et al., ASPLOS 2021
- [UNIT: Unifying Tensorized Instruction Compilation](https://ieeexplore.ieee.org/abstract/document/9370330) by Jian Weng et al., CGO 2021
- [PolyDL: Polyhedral Optimizations for Creation of HighPerformance DL primitives](https://arxiv.org/abs/2006.02230) by Sanket Tavarageri et al., arXiv 2020
- [Fireiron: A Data-Movement-Aware Scheduling Language for GPUs](https://dl.acm.org/doi/abs/10.1145/3410463.3414632) by Bastian Hagedorn et al., PACT 2020
- [Automatic Kernel Generation for Volta Tensor Cores](https://arxiv.org/abs/2006.12645) by Somashekaracharya G. Bhaskaracharya et al., arXiv 2020
- [Swizzle Inventor: Data Movement Synthesis for GPU Kernels](https://mangpo.net/papers/swizzle-inventor-asplos19.pdf) by Phitchaya Mangpo Phothilimthana et al., ASPLOS 2019
- [Optimizing CNN Model Inference on CPUs](https://www.usenix.org/system/files/atc19-liu-yizhi.pdf) by Yizhi Liu et al., ATC 2019
- [Analytical cache modeling and tilesize optimization for tensor contractions](https://dl.acm.org/doi/abs/10.1145/3295500.3356218) by Rui Li et al., SC 19

### NPU Optimization
- [AMOS: Enabling Automatic Mapping for Tensor Computations On Spatial Accelerators with Hardware Abstraction](https://cs.stanford.edu/~anjiang/papers/ZhengETAL22AMOS.pdf) by Size Zheng et al., ISCA 2022 
- [AKG: Automatic Kernel Generation for Neural Processing Units using Polyhedral Transformations](https://www.di.ens.fr/~zhaojie/pldi2021-paper) by Jie Zhao et al., PLDI 2021

### Graph-level Optimization
- [Equality Saturation for Tensor Graph Superoptimization](https://arxiv.org/abs/2101.01332) by Yichen Yang et al., MLSys 2021
- [IOS: An Inter-Operator Scheduler for CNN Acceleration](https://arxiv.org/abs/2011.01302) by Yaoyao Ding et al., MLSys 2021
- [Optimizing DNN Computation Graph using Graph Substitutions](http://www.vldb.org/pvldb/vol13/p2734-fang.pdf) by Jingzhi Fang et al., VLDB 2020
- [Transferable Graph Optimizers for ML Compilers](https://proceedings.neurips.cc/paper/2020/file/9f29450d2eb58feb555078bdefe28aa5-Paper.pdf) by Yanqi Zhou et al., NeurIPS 2020
- [FusionStitching: Boosting Memory IntensiveComputations for Deep Learning Workloads](https://arxiv.org/abs/2009.10924) by Zhen Zheng et al., arXiv 2020
- [Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning](https://proceedings.neurips.cc/paper/2020/file/5f0ad4db43d8723d18169b2e4817a160-Paper.pdf) by Woosuk Kwon et al., Neurips 2020

### Dynamic Model
- [DietCode: Automatic Optimization for Dynamic Tensor Programs](https://proceedings.mlsys.org/paper/2022/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf) by Bojian Zheng et al., MLSys 2022
- [The CoRa Tensor Compiler: Compilation for Ragged Tensors with Minimal Padding](https://arxiv.org/abs/2110.10221) by Pratik Fegade et al., MLSys 2022
- [Nimble: Efficiently Compiling Dynamic Neural Networks for Model Inference](https://arxiv.org/abs/2006.03031) by Haichen Shen et al., MLSys 2021
- [DISC: A Dynamic Shape Compiler for Machine Learning Workloads](https://arxiv.org/pdf/2103.05288.pdf) by Kai Zhu et al., EuroMLSys 2021
- [Cortex: A Compiler for Recursive Deep Learning Models](https://arxiv.org/abs/2011.01383) by Pratik Fegade et al., MLSys 2021

### Graph Neural Networks
- [Graphiler: Optimizing Graph Neural Networks with Message Passing Data Flow Graph](https://proceedings.mlsys.org/paper/2022/file/a87ff679a2f3e71d9181a67b7542122c-Paper.pdf) by Zhiqiang Xie et al., MLSys 2022
- [Seastar: vertex-centric programming for graph neural networks](http://www.cse.cuhk.edu.hk/~jcheng/papers/seastar_eurosys21.pdf) by Yidi Wu et al., Eurosys 2021 
- [FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems](https://arxiv.org/pdf/2008.11359.pdf) by Yuwei Hu et al., SC 2020

### Distributed Computing
- [Pathways: Asynchronous Distributed Dataflow for ML](https://proceedings.mlsys.org/paper/2022/file/98dce83da57b0395e163467c9dae521b-Paper.pdf) by Paul Barham et al., MLSys 2022
- [Synthesizing Optimal Parallelism Placement and Reduction Strategies on Hierarchical Systems for Deep Learning](https://arxiv.org/abs/2110.10548) by Ningning Xie, Tamara Norman, Diminik Grewe, Dimitrios Vytiniotis et al., MLSys 2022
- [DISTAL: The Distributed Tensor Algebra Compiler](https://arxiv.org/pdf/2203.08069.pdf) by Rohan Yadav et al., PLDI 2022
- [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://arxiv.org/abs/2201.12023) by Lianmin Zheng, Zhuohan Li, Hao Zhang et al., OSDI 2022
- [GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/abs/2105.04663) by Yuanzhong Xu et al., arXiv 2021
- [OneFlow: Redesign the Distributed Deep Learning Framework from Scratch](https://arxiv.org/abs/2110.15032) by Jinhui Yuan et al., arXiv 2021
- [Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads](https://arxiv.org/abs/2105.05720) by Abhinav Jangda et al., arXiv 2021
- [Distributed Halide](https://dl.acm.org/doi/abs/10.1145/3016078.2851157) by Tyler Denniston et al., PPoPP 2016

### Quantization and Sparsification
- [SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute](https://www.microsoft.com/en-us/research/uploads/prod/2021/08/SparGen.pdf) by Ningxin Zheng et al., OSDI 2022
- [Automatic Generation of High-Performance Quantized Machine Learning Kernels](https://www.cs.utexas.edu/~bornholt/papers/quantized-cgo20.pdf) by Meghan Cowan et al., CGO 2020

### Program Rewriting
- [Verified tensor-program optimization via high-level scheduling rewrites](https://dl.acm.org/doi/pdf/10.1145/3498717) by Amanda Liu et al., POPL 2022
- [Pure Tensor Program Rewriting via Access Patterns (Representation Pearl)](https://arxiv.org/pdf/2105.09377.pdf) by Gus Smith et al., MAPL 2021
- [Equality Saturation for Tensor Graph Superoptimization](https://proceedings.mlsys.org/paper/2021/file/65ded5353c5ee48d0b7d48c591b8f430-Paper.pdf) by Yichen Yang et al., MLSys 2021

## Tutorials
- [Dive into Deep Learning Compiler](https://tvm.d2l.ai/)


## Contribute
We encourage all contributions to this repository. Open an [issue](https://github.com/merrymercy/awesome-tensor-compilers/issues) or send a [pull request](https://github.com/merrymercy/awesome-tensor-compilers/pulls).

