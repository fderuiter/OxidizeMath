# Math Explorer Examples Index

This directory contains runnable, copy-pasteable examples for the Math Explorer library.

## Usage

Run any example using Cargo by passing its required domain feature flag:
`cargo run --release --package math_explorer --features <domain_feature> --example <example_name>`

## Domain Catalog

### 🤖 AI (`--features ai`)
- `bench_gaussian`: 2D Gaussian splatting blending benchmark.
- `transformer_demo`: Transformer Encoder initialization and forward pass.
- `transformer_simple`: End-to-end Transformer encoder-decoder assembly and forward pass.

### 🧮 Applied (`--features applied`)
- `battery_degradation_demo`: Li-ion battery health and cycle life modeling.
- `bench_lorahub`: LoRA module ensemble linear combination benchmark.
- `bench_mfg`: Mean-field game fixed-point solver benchmark.
- `bench_replicator`: Evolutionary game theory replicator dynamics benchmark.
- `favoritism_demo`: Satirical multi-factor model determining the favorite child.
- `grpo_demo`: Group Relative Policy Optimization advantage normalization and objective.
- `isosurface_torus`: Marching Cubes algorithm for 3D torus mesh extraction.
- `profile_isosurface`: Isosurface extraction and normal vector micro-benchmark.

### 🧬 Biology (`--features biology`)
- `bench_evolution`: Hawk-Dove evolutionary population frequency benchmark.
- `bench_generic_rd`: 1D reaction-diffusion system benchmark with Schnakenberg kinetics.
- `bench_morphogenesis`: Turing pattern generation benchmark.
- `demo_neuron_builder`: Safe Hodgkin-Huxley neuron construction via builder pattern.
- `hodgkin_huxley_demo`: Simulating an excitable neuron's membrane potential.

### 🌍 Climate (`--features climate`)
- `cera_demo`: Climate-invariant Encoding through Representation Alignment (CERA).

### 🦠 Epidemiology (`--features epidemiology`)
- `bench_stochastic`: Zero-allocation Gillespie stochastic SIR simulation benchmark.

### 🔬 Physics (`--features physics`)
- `bench_ising_custom`: Ising model spin lattice phase transition benchmark.
- `bench_lbm_custom`: D2Q9 Lattice Boltzmann Method fluid dynamics benchmark.
- `bench_music`: MUSIC super-resolution radar spectrum estimation benchmark.
- `hello_world`: Quantum Mechanics Clebsch-Gordan coefficient calculation.
- `lorenz_chaos`: Lorenz chaotic attractor simulation and trajectory divergence.
- `mmwave_setup`: Verified FMCW mmWave radiotherapy radar displacement pipeline.
- `security_repro_lbm`: Lattice Boltzmann state encapsulation and invariant verification.

### 📐 Pure Math (`--features pure_math`)
- `algebra_demo`: Finite fields and polynomial arithmetic.
- `bench_rk4_alloc`: Fourth-order Runge-Kutta ODE solver allocation benchmark.
- `glicko2_demo`: Glicko-2 competitive rating system simulation.
- `markov_demo`: Discrete-time, continuous-time, and hidden Markov chain models.
- `sports_analytics_demo`: Copula, Glicko-2, Kelly criterion, Markov, and Ornstein-Uhlenbeck analytics.
- `statistics_demo`: Glicko-2, Kelly criterion, and topological data analysis overview.
- `tensor_demo`: Computing Riemannian metrics and Christoffel symbols on manifolds.

For more details, see the inline documentation within each `.rs` file.
