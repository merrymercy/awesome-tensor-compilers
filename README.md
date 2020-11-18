## Contents
- [Open Source Projects](#open-source-projects)
- [Papers](#papers)
  - [Survey](#survey)
  - [Compiler](#compiler)
  - [Auto-tuning and Auto-scheduling](#auto-tuning-and-auto-scheduling)
  - [Cost Model](#cost-model)
  - [CPU Optimizaiton](#cpu-optimizaiton)
  - [GPU Optimizaiton](#gpu-optimizaiton)
  - [Graph-level Optimization](#graph-level-optimization)
- [Tutorials](#tutorials)

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

## Papers

### Survey
- [The Deep Learning Compiler: A Comprehensive Survey](https://arxiv.org/pdf/2002.03794.pdf) by Mingzhen Li et al., TPDS 2020
- [An In-depth Comparison of Compilers for DeepNeural Networks on Hardware](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8782480&casa_token=mzwyH78qqnoAAAAA:CrQHJ9e4ToeRw7hvB90cCHU3QVzPshRju---blvfOJvJwRvy0gfpvrrooayO1wGDUOh1Evw2LMI) by Yu Xing et al., ICESS 2019

### Compiler
- [A Tensor Compiler for Unified Machine Learning Prediction Serving](https://www.usenix.org/conference/osdi20/presentation/nakandala) by Supun Nakandala et al., OSDI 20
- [Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks](https://www.usenix.org/conference/osdi20/presentation/ma) by Lingxiao Ma et al., OSDI 20
- [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/abs/2002.11054) by Chris Lattner et al., arXiv 2020
- [TASO: The Tensor Algebra SuperOptimizer for Deep Learning](https://dl.acm.org/doi/pdf/10.1145/3341301.3359630?casa_token=dYBNBVyhmV0AAAAA:zD-feoFh6susJzp9mE6KKsffaV94Ec-LJxJL-GQoA_16mTjXtYL3q0Xqiuh5jdD5PAuhyHH1lPWkGQ) by Zhihao Jia et al., SOSP 2019
- [Tiramisu: A polyhedral compiler for expressing fast and portable code](https://arxiv.org/abs/1804.10694) by Riyadh Baghdadi et al., CGO 2019
- [Triton: an intermediate language and compiler for tiled neural network computations](https://dl.acm.org/doi/pdf/10.1145/3315508.3329973?casa_token=w0MaltEBfKYAAAAA:X27ScRTBiDR3WfL1VKTuU34wXJhr0r4H32JEcFe-DkmkJogCDG9dG7Tvp45sR9aB5tUKwky_hE25xg) by Philippe Tillet et al., MAPL 2019
- [Relay: A High-Level Compiler for Deep Learning](https://arxiv.org/pdf/1904.08368.pdf) by Jared Roesch et al., arXiv 2019
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) by Tianqi Chen et al., OSDI 2018
- [Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions](https://arxiv.org/abs/1802.04730) by Nicolas Vasilache et al., arXiv 2018
- [Intel nGraph: An Intermediate Representation, Compiler, and Executor for Deep Learning](https://arxiv.org/abs/1801.08058) by Scott Cyphers et al., arXiv 2018
- [Glow: Graph Lowering Compiler Techniques for Neural Networks](https://arxiv.org/abs/1805.00907) by Nadav Rotem et al., arXiv 2018
- [DLVM: A modern compiler infrastructure for deep learning systems](https://arxiv.org/pdf/1711.03016.pdf) by Richard Wei et al., arXiv 2018
- [Diesel: DSL for linear algebra and neural net computations on GPUs](https://dl.acm.org/doi/pdf/10.1145/3211346.3211354) by Venmugil Elango et al., MAPL 2018
- [The Tensor Algebra Compiler](https://dl.acm.org/doi/pdf/10.1145/3133901) by Fredrik Kjolstad et al., OOPSLA 2017
- [Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines](http://people.csail.mit.edu/jrk/halide-pldi13.pdf) by Jonathan Ragan-Kelley et al., PLDI 2013


### Auto-tuning and Auto-scheduling

- [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://arxiv.org/abs/2006.06762) by Lianmin Zheng et al., OSDI 2020
- [Schedule Synthesis for Halide Pipelines on GPUs](https://dl.acm.org/doi/fullHtml/10.1145/3406117) by Sioutas Savvas et al., TACO 2020
- [FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System](https://dl.acm.org/doi/pdf/10.1145/3373376.3378508?casa_token=2mWk5Qp3Ll8AAAAA:67phDw6-xWqKmo9A2EMXhVwl8KhHOGU_MeYc0sGiORNtNQTP_IDYmTW1gFtapsPuV48i1U5FRmRNfg) by Size Zheng et al., ASPLOS 2020
- [ProTuner: Tuning Programs with Monte Carlo Tree Search](https://arxiv.org/abs/2005.13685) by Ameer Haj-Ali et al., arXiv 2020
- [Chameleon: Adaptive Code Optimization for Expedited Deep Neural Network Compilation](https://openreview.net/forum?id=rygG4AVFvH) by Byung Hoon Ahn et al., ICLR 2020
- [Learning to Optimize Halide with Tree Search and Random Programs](https://halide-lang.org/papers/autoscheduler2019.html) by Andrew Adams et al., SIGGRAPH 2019
- [Learning to Optimize Tensor Programs](https://arxiv.org/abs/1805.08166) by Tianqi Chen et al., NeurIPS 2018
- [Automatically Scheduling Halide Image Processing Pipelines](http://graphics.cs.cmu.edu/projects/halidesched/) by Ravi Teja Mullapudi et al., SIGGRAPH 2016

### Cost Model
- [A Deep Learning Based Cost Model for Automatic Code Optimization in Tiramisu](https://www.researchgate.net/profile/Massinissa_Merouani/publication/344948008_A_Deep_Learning_Based_Cost_Model_for_Automatic_Code_Optimization_in_Tiramisu/links/5f9a79b2458515b7cfa73e8d/A-Deep-Learning-Based-Cost-Model-for-Automatic-Code-Optimization-in-Tiramisu.pdf) by Massinissa Merouani et al., Graduation Thesis 2020
- [A Learned Performance Model for the Tensor Processing Unit](https://arxiv.org/pdf/2008.01040.pdf) by Samuel J. Kaufman et al., arXiv 2020

### CPU Optimizaiton
- [PolyDL: Polyhedral Optimizations for Creation of HighPerformance DL primitives](https://arxiv.org/pdf/2006.02230.pdf) by Sanket Tavarageri et al., arXiv 2020
- [Automatic Generation of High-Performance Quantized Machine Learning Kernels](https://www.cs.utexas.edu/~bornholt/papers/quantized-cgo20.pdf) by Meghan Cowan et al., CGO 2020
- [Optimizing CNN Model Inference on CPUs](https://www.usenix.org/system/files/atc19-liu-yizhi.pdf) by Yizhi Liu et al., ATC 2019

### GPU Optimization
- [Fireiron: A Data-Movement-Aware Scheduling Language for GPUs](https://dl.acm.org/doi/pdf/10.1145/3410463.3414632?casa_token=jQw5p7cYSOAAAAAA:Re5S2oGp3_ld1L4tyjoSPoJ8H26oLaGbsM8taHXW1majFMR7so2Gl_eN-RQNU21Sfm0Cf3rnHuqAJw) by Bastian Hagedorn et al., PACT 2020
- [Automatic Kernel Generation for Volta Tensor Cores](https://arxiv.org/abs/2006.12645) by Somashekaracharya G. Bhaskaracharya et al., arXiv 2020

### Graph-level Optimization
- [Transferable Graph Optimizers for ML Compilers](https://proceedings.neurips.cc/paper/2020/file/9f29450d2eb58feb555078bdefe28aa5-Paper.pdf) by et al., NeurIPS 2020

## Tutorials
- [Dive into Deep Learning Compiler](https://tvm.d2l.ai/)
