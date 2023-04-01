# Awesome Tensor Compilers
![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/merrymercy/awesome-tensor-compilers/graphs/commit-activity)

A list of awesome compiler projects and papers for tensor computation and deep learning. 

## Contents
- [Open Source Projects](#open-source-projects)
- [Papers](#papers)
  - [Survey](#survey)
  - [Compiler and IR Design](#compiler-and-ir-design)
  - [Auto-tuning and Auto-scheduling](#auto-tuning-and-auto-scheduling)
  - [Cost Model](#cost-model)
  - [CPU & GPU Optimization](#cpu-and-gpu-optimization)
  - [NPU Optimization](#npu-optimization)
  - [Graph-level Optimization](#graph-level-optimization)
  - [Dynamic Model](#dynamic-model)
  - [Graph Neural Networks](#graph-neural-networks)
  - [Distributed Computing](#distributed-computing)
  - [Quantization](#quantization)
  - [Sparse](#sparse)
  - [Program Rewriting](#program-rewriting)
  - [Verification and Testing](#verification-and-testing)
- [Tutorials](#tutorials)
- [Contribute](#contribute)

## Open Source Projects
- [TVM: An End to End Machine Learning Compiler Framework](https://tvm.apache.org/)
- [MLIR: Multi-Level Intermediate Representation](https://mlir.llvm.org/)
- [XLA: Optimizing Compiler for Machine Learning](https://www.tensorflow.org/xla)
- [Halide: A Language for Fast, Portable Computation on Images and Tensors](https://halide-lang.org/)
- [Glow: Compiler for Neural Network Hardware Accelerators](https://github.com/pytorch/glow)
- [nnfusion: A Flexible and Efficient Deep Neural Network Compiler](https://github.com/microsoft/nnfusion)
- [Hummingbird: Compiling Trained ML Models into Tensor Computation](https://github.com/microsoft/hummingbird)
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://github.com/openai/triton)
- [AITemplate: A Python framework which renders neural network into high performance CUDA/HIP C++ code](https://github.com/facebookincubator/AITemplate)
- [Hidet: A Compilation-based Deep Learning Framework](https://github.com/hidet-org/hidet)
- [Tiramisu: A Polyhedral Compiler for Expressing Fast and Portable Code](http://tiramisu-compiler.org/)
- [TensorComprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions](https://facebookresearch.github.io/TensorComprehensions/)
- [PlaidML: A Platform for Making Deep Learning Work Everywhere](https://github.com/plaidml/plaidml)
- [BladeDISC: An End-to-End DynamIc Shape Compiler for Machine Learning Workloads](https://github.com/alibaba/BladeDISC)
- [TACO: The Tensor Algebra Compiler](http://tensor-compiler.org/)
- [Nebulgym: Easy-to-use Library to Accelerate AI Training](https://github.com/nebuly-ai/nebulgym)
- [Speedster: Automatically apply SOTA optimization techniques to achieve the maximum inference speed-up on your hardware](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/speedster)
- [NN-512: A Compiler That Generates C99 Code for Neural Net Inference](https://nn-512.com/)
- [DaCeML: A Data-Centric Compiler for Machine Learning](https://github.com/spcl/daceml)

## Papers

### Survey
- [The Deep Learning Compiler: A Comprehensive Survey](https://arxiv.org/abs/2002.03794) by Mingzhen Li et al., TPDS 2020
- [An In-depth Comparison of Compilers for DeepNeural Networks on Hardware](https://ieeexplore.ieee.org/document/8782480) by Yu Xing et al., ICESS 2019

### Compiler and IR Design
- [Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs](https://arxiv.org/abs/2210.09603) by Yaoyao Ding et al., ASPLOS 2023
- [TensorIR: An Abstraction for Automatic Tensorized Program Optimization](https://arxiv.org/abs/2207.04296) by Siyuan Feng, Bohan Hou et al., ASPLOS 2023
- [Exocompilation for Productive Programming of Hardware Accelerators](https://dl.acm.org/doi/abs/10.1145/3519939.3523446) by Yuka Ikarashi, Gilbert Louis Bernstein et al., PLDI 2022
- [DaCeML: A Data-Centric Compiler for Machine Learning](https://arxiv.org/abs/2110.10802) by Oliver Rausch et al., ICS 22
- [FreeTensor: A Free-Form DSL with Holistic Optimizations for Irregular Tensor Programs](https://dl.acm.org/doi/10.1145/3519939.3523448) by Shizhi Tang et al., PLDI 2022
- [Roller: Fast and Efficient Tensor Compilation for Deep Learning](https://www.usenix.org/conference/osdi22/presentation/zhu) by Hongyu Zhu et al., OSDI 2022
- [AStitch: Enabling a New Multi-dimensional Optimization Space for Memory-Intensive ML Training and Inference on Modern SIMT Architectures](https://dl.acm.org/doi/10.1145/3503222.3507723) by Zhen Zheng et al., ASPLOS 2022
- [Composable and Modular Code Generation in MLIR: A Structured and Retargetable Approach to Tensor Compiler Construction](https://arxiv.org/pdf/2202.03293.pdf) by Nicolas Vasilache et al., arXiv 2022
- [PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections](https://www.usenix.org/conference/osdi21/presentation/wang) by Haojie Wang et al., OSDI 2021
- [MLIR: Scaling Compiler Infrastructure for Domain Specific Computation](https://ieeexplore.ieee.org/document/9370308) by Chris Lattner et al., CGO 2021
- [A Tensor Compiler for Unified Machine Learning Prediction Serving](https://www.usenix.org/conference/osdi20/presentation/nakandala) by Supun Nakandala et al., OSDI 2020
- [Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks](https://www.usenix.org/conference/osdi20/presentation/ma) by Lingxiao Ma et al., OSDI 2020
- [Stateful Dataflow Multigraphs: A Data-Centric Model for Performance Portability on Heterogeneous Architectures](https://arxiv.org/abs/1902.10345) by Tal Ben-Nun et al., SC 2019
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
- [Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines](https://dl.acm.org/doi/10.1145/2491956.2462176) by Jonathan Ragan-Kelley et al., PLDI 2013


### Auto-tuning and Auto-scheduling
- [Tensor Program Optimization with Probabilistic Programs](https://arxiv.org/abs/2205.13603) by Junru Shao et al., NeurIPS 2022
- [One-shot tuner for deep learning compilers](https://dl.acm.org/doi/abs/10.1145/3497776.3517774) by Jaehun Ryu et al., CC 2022 
- [Autoscheduling for sparse tensor algebra with an asymptotic cost model](https://dl.acm.org/doi/abs/10.1145/3519939.3523442) by Peter Ahrens et al., PLDI 2022
- [Bolt: Bridging the Gap between Auto-tuners and Hardware-native Performance](https://proceedings.mlsys.org/paper/2022/hash/38b3eff8baf56627478ec76a704e9b52-Abstract.html) by Jiarong Xing et al., MLSys 2022
- [A Full-Stack Search Technique for Domain Optimized Deep Learning Accelerators](https://dl.acm.org/doi/10.1145/3503222.3507767) by Dan Zhang et al., ASPLOS 2022
- [Efficient Automatic Scheduling of Imaging and Vision Pipelines for the GPU](https://dl.acm.org/doi/10.1145/3485486) by Luke Anderson et al., OOPSLA 2021
- [Lorien: Efficient Deep Learning Workloads Delivery](https://dl.acm.org/doi/abs/10.1145/3472883.3486973) by Cody Hao Yu et al., SoCC 2021
- [Value Learning for Throughput Optimization of Deep Neural Networks](https://proceedings.mlsys.org/paper/2021/hash/73278a4a86960eeb576a8fd4c9ec6997-Abstract.html) by Benoit Steiner et al., MLSys 2021
- [A Flexible Approach to Autotuning Multi-Pass Machine Learning Compilers](https://mangpo.net/papers/xla-autotuning-pact2021.pdf) by Phitchaya Mangpo Phothilimthana et al., PACT 2021
- [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://arxiv.org/abs/2006.06762) by Lianmin Zheng et al., OSDI 2020
- [Schedule Synthesis for Halide Pipelines on GPUs](https://dl.acm.org/doi/abs/10.1145/3406117) by Sioutas Savvas et al., TACO 2020
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
- [TLP: A Deep Learning-based Cost Model for Tensor Program Tuning](https://arxiv.org/abs/2211.03578) by Yi Zhai et al., ASPLOS 2023
- [An Asymptotic Cost Model for Autoscheduling Sparse Tensor Programs](https://arxiv.org/abs/2111.14947) by Peter Ahrens et al., PLDI 2022
- [TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/a684eceee76fc522773286a895bc8436-Abstract-round1.html) by Lianmin Zheng et al., NeurIPS 2021
- [A Deep Learning Based Cost Model for Automatic Code Optimization](https://proceedings.mlsys.org/paper/2021/hash/3def184ad8f4755ff269862ea77393dd-Abstract.html) by Riyadh Baghdadi et al., MLSys 2021
- [A Learned Performance Model for the Tensor Processing Unit](https://arxiv.org/abs/2008.01040) by Samuel J. Kaufman et al., MLSys 2021
- [DYNATUNE: Dynamic Tensor Program Optimization in Deep Neural Network Compilation](https://openreview.net/forum?id=GTGb3M_KcUl) by Minjia Zhang et al., ICLR 2021
- [MetaTune: Meta-Learning Based Cost Model for Fast and Efficient Auto-tuning Frameworks](https://arxiv.org/abs/2102.04199) by Jaehun Ryu et al., arxiv 2021

### CPU and GPU Optimization
- [DeepCuts: A deep learning optimization framework for versatile GPU workloads](https://pldi21.sigplan.org/details/pldi-2021-papers/13/DeepCuts-A-Deep-Learning-Optimization-Framework-for-Versatile-GPU-Workloads) by Wookeun Jung et al., PLDI 2021
- [Analytical characterization and design space exploration for optimization of CNNs](https://dl.acm.org/doi/abs/10.1145/3445814.3446759) by Rui Li et al., ASPLOS 2021
- [UNIT: Unifying Tensorized Instruction Compilation](https://ieeexplore.ieee.org/abstract/document/9370330) by Jian Weng et al., CGO 2021
- [PolyDL: Polyhedral Optimizations for Creation of HighPerformance DL primitives](https://arxiv.org/abs/2006.02230) by Sanket Tavarageri et al., arXiv 2020
- [Fireiron: A Data-Movement-Aware Scheduling Language for GPUs](https://dl.acm.org/doi/abs/10.1145/3410463.3414632) by Bastian Hagedorn et al., PACT 2020
- [Automatic Kernel Generation for Volta Tensor Cores](https://arxiv.org/abs/2006.12645) by Somashekaracharya G. Bhaskaracharya et al., arXiv 2020
- [Swizzle Inventor: Data Movement Synthesis for GPU Kernels](https://dl.acm.org/doi/10.1145/3297858.3304059) by Phitchaya Mangpo Phothilimthana et al., ASPLOS 2019
- [Optimizing CNN Model Inference on CPUs](https://www.usenix.org/conference/atc19/presentation/liu-yizhi) by Yizhi Liu et al., ATC 2019
- [Analytical cache modeling and tilesize optimization for tensor contractions](https://dl.acm.org/doi/abs/10.1145/3295500.3356218) by Rui Li et al., SC 19

### NPU Optimization
- [Heron: Automatically Constrained High-Performance Library Generation for Deep Learning Accelerators](https://dl.acm.org/doi/10.1145/3582016.3582061) by Jun Bi et al., ASPLOS 2023
- [AMOS: Enabling Automatic Mapping for Tensor Computations On Spatial Accelerators with Hardware Abstraction](https://cs.stanford.edu/~anjiang/papers/ZhengETAL22AMOS.pdf) by Size Zheng et al., ISCA 2022
- [Towards the Co-design of Neural Networks and Accelerators](https://proceedings.mlsys.org/paper/2022/hash/31fefc0e570cb3860f2a6d4b38c6490d-Abstract.html) by Yanqi Zhou et al., MLSys 2022
- [AKG: Automatic Kernel Generation for Neural Processing Units using Polyhedral Transformations](https://www.di.ens.fr/~zhaojie/pldi2021-paper) by Jie Zhao et al., PLDI 2021

### Graph-level Optimization
- [POET: Training Neural Networks on Tiny Devices with Integrated Rematerialization and Paging](https://arxiv.org/abs/2207.07697) by Shishir G. Patil et al., ICML 2022
- [Collage: Seamless Integration of Deep Learning Backends with Automatic Placement](https://arxiv.org/abs/2111.00655) by Byungsoo Jeon et al., PACT 2022
- [Apollo: Automatic Partition-based Operator Fusion through Layer by Layer Optimization](https://proceedings.mlsys.org/paper/2022/hash/069059b7ef840f0c74a814ec9237b6ec-Abstract.html) by Jie Zhao et al., MLSys 2022
- [Equality Saturation for Tensor Graph Superoptimization](https://arxiv.org/abs/2101.01332) by Yichen Yang et al., MLSys 2021
- [IOS: An Inter-Operator Scheduler for CNN Acceleration](https://arxiv.org/abs/2011.01302) by Yaoyao Ding et al., MLSys 2021
- [Optimizing DNN Computation Graph using Graph Substitutions](https://dl.acm.org/doi/10.14778/3407790.3407857) by Jingzhi Fang et al., VLDB 2020
- [Transferable Graph Optimizers for ML Compilers](https://papers.nips.cc/paper/2020/hash/9f29450d2eb58feb555078bdefe28aa5-Abstract.html) by Yanqi Zhou et al., NeurIPS 2020
- [FusionStitching: Boosting Memory IntensiveComputations for Deep Learning Workloads](https://arxiv.org/abs/2009.10924) by Zhen Zheng et al., arXiv 2020
- [Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning](https://proceedings.neurips.cc/paper/2020/hash/5f0ad4db43d8723d18169b2e4817a160-Abstract.html) by Woosuk Kwon et al., Neurips 2020

### Dynamic Model
- [Axon: A Language for Dynamic Shapes in Deep Learning Graphs](https://arxiv.org/abs/2210.02374) by Alexander Collins et al., arXiv 2022
- [DietCode: Automatic Optimization for Dynamic Tensor Programs](https://proceedings.mlsys.org/paper/2022/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html) by Bojian Zheng et al., MLSys 2022
- [The CoRa Tensor Compiler: Compilation for Ragged Tensors with Minimal Padding](https://arxiv.org/abs/2110.10221) by Pratik Fegade et al., MLSys 2022
- [Nimble: Efficiently Compiling Dynamic Neural Networks for Model Inference](https://arxiv.org/abs/2006.03031) by Haichen Shen et al., MLSys 2021
- [DISC: A Dynamic Shape Compiler for Machine Learning Workloads](https://arxiv.org/abs/2103.05288) by Kai Zhu et al., EuroMLSys 2021
- [Cortex: A Compiler for Recursive Deep Learning Models](https://arxiv.org/abs/2011.01383) by Pratik Fegade et al., MLSys 2021

### Graph Neural Networks
- [Graphiler: Optimizing Graph Neural Networks with Message Passing Data Flow Graph](https://proceedings.mlsys.org/paper/2022/hash/a87ff679a2f3e71d9181a67b7542122c-Abstract.html) by Zhiqiang Xie et al., MLSys 2022
- [Seastar: vertex-centric programming for graph neural networks](https://dl.acm.org/doi/10.1145/3447786.3456247) by Yidi Wu et al., Eurosys 2021 
- [FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems](https://arxiv.org/abs/2008.11359) by Yuwei Hu et al., SC 2020

### Distributed Computing
- [SpDISTAL: Compiling Distributed Sparse Tensor Computations](https://arxiv.org/abs/2207.13901) by Rohan Yadav et al., SC 2022
- [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://arxiv.org/abs/2201.12023) by Lianmin Zheng, Zhuohan Li, Hao Zhang et al., OSDI 2022
- [Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization](https://www.usenix.org/conference/osdi22/presentation/unger) by Colin Unger, Zhihao Jia, et al., OSDI 2022
- [Synthesizing Optimal Parallelism Placement and Reduction Strategies on Hierarchical Systems for Deep Learning](https://arxiv.org/abs/2110.10548) by Ningning Xie, Tamara Norman, Diminik Grewe, Dimitrios Vytiniotis et al., MLSys 2022
- [DISTAL: The Distributed Tensor Algebra Compiler](https://arxiv.org/abs/2203.08069) by Rohan Yadav et al., PLDI 2022
- [GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/abs/2105.04663) by Yuanzhong Xu et al., arXiv 2021
- [Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads](https://arxiv.org/abs/2105.05720) by Abhinav Jangda et al., ASPLOS 2022
- [OneFlow: Redesign the Distributed Deep Learning Framework from Scratch](https://arxiv.org/abs/2110.15032) by Jinhui Yuan et al., arXiv 2021
- [Beyond Data and Model Parallelism for Deep Neural Networks](https://proceedings.mlsys.org/paper/2019/hash/c74d97b01eae257e44aa9d5bade97baf-Abstract.html) by Zhihao et al., MLSys 2019
- [Supporting Very Large Models using Automatic Dataflow Graph Partitioning](https://dl.acm.org/doi/10.1145/3302424.3303953) by Minjie Wang et al., EuroSys 2019
- [Distributed Halide](https://dl.acm.org/doi/abs/10.1145/3016078.2851157) by Tyler Denniston et al., PPoPP 2016

### Quantization
- [Automated Backend-Aware Post-Training Quantization](https://arxiv.org/abs/2103.14949) by Ziheng Jiang et al., arXiv 2021
- [Efficient Execution of Quantized Deep Learning Models: A Compiler Approach](https://arxiv.org/abs/2006.10226) by Animesh Jain et al., arXiv 2020
- [Automatic Generation of High-Performance Quantized Machine Learning Kernels](https://dl.acm.org/doi/10.1145/3368826.3377912) by Meghan Cowan et al., CGO 2020

### Sparse
- [The Sparse Abstract Machine](https://arxiv.org/abs/2208.14610) by Olivia Hsu et al., ASPLOS 2023
- [SparseTIR: Composable Abstractions for Sparse Compilation in Deep Learning](https://arxiv.org/abs/2207.04606) by Zihao Ye et al., ASPLOS 2023
- [WACO: Learning Workload-Aware Co-optimization of the Format and Schedule of a Sparse Tensor Program](https://dl.acm.org/doi/10.1145/3575693.3575742) by Jaeyeon Won et al., ASPLOS 2023
- [Looplets: A Language For Structured Coiteration](https://arxiv.org/abs/2209.05250) by Willow Ahrens et al., CGO 2023
- [Code Synthesis for Sparse Tensor Format Conversion and Optimization](https://www.researchgate.net/publication/367180198_Code_Synthesis_for_Sparse_Tensor_Format_Conversion_and_Optimization) by Tobi Popoola et al., CGO 2023
- [Stardust: Compiling Sparse Tensor Algebra to a Reconfigurable Dataflow Architecture](https://arxiv.org/abs/2211.03251) by Olivia Hsu et al., arXiv 2022
- [The Sparse Abstract Machine](https://arxiv.org/abs/2208.14610) by Olivia Hsu et al., arXiv 2022
- [Unified Compilation for Lossless Compression and Sparse Computing](https://dl.acm.org/doi/10.1109/CGO53902.2022.9741282) by Daniel Donenfeld et al., CGO 2022
- [SparseLNR: Accelerating Sparse Tensor Computations Using Loop Nest Restructuring](https://dl.acm.org/doi/abs/10.1145/3524059.3532386) by Adhitha Dias et al., ICS 2022
- [SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute](https://www.usenix.org/conference/osdi22/presentation/zheng-ningxin) by Ningxin Zheng et al., OSDI 2022
- [Compiler Support for Sparse Tensor Computations in MLIR](https://arxiv.org/abs/2202.04305) by Aart J.C. Bik et al., TACO 2022
- [Compilation of Sparse Array Programming Models](https://dl.acm.org/doi/abs/10.1145/3485505) by Rawn Henry and Olivia Hsu et al., OOPSLA 2021
- [A High Performance Sparse Tensor Algebra Compiler in MLIR](https://ieeexplore.ieee.org/document/9651314) by Ruiqin Tian et al., LLVM-HPC 2021
- [Dynamic Sparse Tensor Algebra Compilation](https://arxiv.org/abs/2112.01394) by Stephen Chou et al., arXiv 2021
- [Automatic Generation of Efficient Sparse Tensor Format Conversion Routines](https://dl.acm.org/doi/abs/10.1145/3385412.3385963) by Stephen Chou et al., PLDI 2020
- [TIRAMISU: A Polyhedral Compiler for Dense and Sparse Deep Learning](https://arxiv.org/abs/2005.04091) by Riyadh Baghdadi et al., arXiv 2020
- [Tensor Algebra Compilation with Workspaces](https://dl.acm.org/doi/abs/10.5555/3314872.3314894) by Fredrik Kjolstad et al., CGO 2019
- [Sparse Computation Data Dependence Simplification for Efficient Compiler-Generated Inspectors](https://dl.acm.org/doi/10.1145/3314221.3314646) by Mahdi Soltan Mohammadi et al., PLDI 2019
- [Taichi: A Language for High-Performance Computation on Spatially Sparse Data Structures](https://dl.acm.org/doi/abs/10.1145/3355089.3356506) by Yuanming Hu et al., ACM ToG 2019
- [The Sparse Polyhedral Framework: Composing Compiler-Generated Inspector-Executor Code](https://ieeexplore.ieee.org/document/8436444) by Michelle Mills Strout et al., Proceedings of the IEEE 2018
- [Format Abstraction for Sparse Tensor Algebra Compilers](https://dl.acm.org/doi/abs/10.1145/3276493) by Stephen Chou et al., OOPSLA 2018
- [ParSy: Inspection and Transformation of Sparse Matrix Computations for Parallelism](https://ieeexplore.ieee.org/document/8665791) by Kazem Cheshmi et al., SC 2018
- [Sympiler: Transforming Sparse Matrix Codes by Decoupling Symbolic Analysis](https://dl.acm.org/doi/10.1145/3126908.3126936) by Kazem Cheshmi et al., SC 2017
- [The Tensor Algebra Compiler](https://dl.acm.org/doi/abs/10.1145/3133901) by Fredrik Kjolstad et al., OOPSLA 2017
- [Next-generation Generic Programming and its Application to Sparse Matrix Computations](https://dl.acm.org/doi/abs/10.1145/335231.335240) by Nikolay Mateev et al., ICS 2000
- [A Framework for Sparse Matrix Code Synthesis from High-level Specifications](https://ieeexplore.ieee.org/document/1592771) by Nawaaz Ahmed et al., SC 2000
- [Automatic Nonzero Structure Analysis](https://epubs.siam.org/doi/10.1137/S009753979529595X) by Aart Bik et al., SIAM Journal on Computing 1999
- [SIPR: A New Framework for Generating Efficient Code for Sparse Matrix Computations](https://link.springer.com/chapter/10.1007/3-540-48319-5_14) by William Pugh et al., LCPC 1998
- [Automatic Data Structure Selection and Transformation for Sparse Matrix Computations](https://ieeexplore.ieee.org/document/485501) by Aart Bik et al., TPDS 1996
- [Compilation Techniques for Sparse Matrix Computations](https://dl.acm.org/doi/abs/10.1145/165939.166023) by Aart Bik et al., ICS 1993


### Program Rewriting
- [Verified tensor-program optimization via high-level scheduling rewrites](https://dl.acm.org/doi/10.1145/3498717) by Amanda Liu et al., POPL 2022
- [Pure Tensor Program Rewriting via Access Patterns (Representation Pearl)](https://arxiv.org/abs/2105.09377) by Gus Smith et al., MAPL 2021
- [Equality Saturation for Tensor Graph Superoptimization](https://arxiv.org/abs/2101.01332) by Yichen Yang et al., MLSys 2021

### Verification and Testing
- [NNSmith: Generating Diverse and Valid Test Cases for Deep Learning Compilers](https://dl.acm.org/doi/10.1145/3575693.3575707) by Jiawei Liu et al., ASPLOS 2023
- [Coverage-guided tensor compiler fuzzing with joint IR-pass mutation](https://dl.acm.org/doi/abs/10.1145/3527317) by Jiawei Liu et al., OOPSLA 2022
- [End-to-End Translation Validation for the Halide Language](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/2d03e3ae1106d3a2c950fcdc5eeb2c383eb24372.pdf) by Basile Clément et al., OOPSLA 2022
- [A comprehensive study of deep learning compiler bugs](https://dl.acm.org/doi/abs/10.1145/3468264.3468591) by Qingchao Shen et al., ESEC/FSE 2021
- [Verifying and Improving Halide’s Term Rewriting System with Program Synthesis](https://dl.acm.org/doi/abs/10.1145/3428234) by Julie L. Newcomb et al., OOPSLA 2020

## Tutorials
- [Machine Learning Compilation](https://mlc.ai/summer22/)
- [Dive into Deep Learning Compiler](https://tvm.d2l.ai/)

## Contribute
We encourage all contributions to this repository. Open an [issue](https://github.com/merrymercy/awesome-tensor-compilers/issues) or send a [pull request](https://github.com/merrymercy/awesome-tensor-compilers/pulls).

### Notes on the Link Format
We prefer using a link which points to a more informative page instead of a single pdf. For example, for arxiv papers, we prefer https://arxiv.org/abs/1802.04799 over https://arxiv.org/pdf/1802.04799.pdf. For USENIX papers (OSDI/ATC), we prefer https://www.usenix.org/conference/osdi18/presentation/chen over https://www.usenix.org/system/files/osdi18-chen.pdf. For ACM papers (ASPLOS/PLDI/Eurosys), we prefer https://dl.acm.org/doi/abs/10.1145/3519939.3523446 over https://dl.acm.org/doi/pdf/10.1145/3519939.3523446.
