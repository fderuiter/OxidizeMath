//! # Chemotactic Slime-Mold Networks (`Physarum polycephalum`)
//!
//! Implements a zero-allocation Lagrangian-Eulerian **`PhysarumEngine`** coupling
//! Structure-of-Arrays (SoA) mobile chemotactic agents with diffused, nonlinearly pruned
//! chemoattractant trail grids, nutrient beacons, and obstacle masks, alongside live
//! weighted graph skeleton extraction benchmarked against [`pure_math::pure_math::graph_theory::dijkstra::dijkstra`]
//! and static minimum spanning trees.
//!
//! ## Mathematical Formulation
//!
//! Each mobile agent $i \in \{1, \dots, N\}$ maintains continuous state $(x_i, y_i, \theta_i)$
//! and samples the Eulerian chemoattractant field $T(x, y)$ at three forward-cone sensors
//! offset by $\pm \theta_s$ at distance $d_s$ (Jones, 2010):
//!
//! $$F = T(x_i + d_s \cos\theta_i,\, y_i + d_s \sin\theta_i), \qquad
//!   F_{\pm} = T\bigl(x_i + d_s \cos(\theta_i \pm \theta_s),\, y_i + d_s \sin(\theta_i \pm \theta_s)\bigr)$$
//!
//! Agents rotate toward stronger chemoattractant gradients by $\pm\theta_r$, advance by step $v$
//! outside obstacle cells, deposit trail mass $w_{\text{dep}}$, and reverse shuttle heading upon
//! feeding at nutrient beacons. The Eulerian trail field evolves via obstacle-masked $3\times 3$
//! diffusion blended by $\alpha$ and Tero-inspired sub-threshold Hill pruning (Tero et al., 2010):
//!
//! $$T_{t+1} = (1 - \gamma)\, \mathcal{D}_\alpha(T_t) - \gamma_{\text{prune}} \frac{\mathcal{D}_\alpha(T_t)}{1 + (\mathcal{D}_\alpha(T_t) / T_c)^2}$$
//!
//! ## Quick Start
//!
//! ```rust
//! # use domain_biology::__doctest_facade as math_explorer;
//! use math_explorer::biology::physarum::{
//!     PhysarumEngine, PhysarumPresetKind, extract_physarum_benchmark,
//! };
//! use math_explorer::rng::OxidizeRng;
//!
//! let rng = OxidizeRng::new(42);
//! let mut engine = PhysarumEngine::from_preset(96, 96, 2048, PhysarumPresetKind::SteinerTriangle, rng);
//! for _ in 0..50 {
//!     engine.step();
//! }
//! let benchmark = extract_physarum_benchmark(&engine);
//! assert!(engine.telemetry().total_trail_mass > 0.0);
//! assert!(benchmark.mst_cost > 0.0);
//! ```

mod dynamics;
/// Zero-allocation SoA Lagrangian-Eulerian simulation engine.
pub mod engine;
/// Morphological skeleton thinning and Dijkstra / spanning-tree network benchmarks.
pub mod network;
/// Validated chemotaxis, diffusion, pruning, and nutrient terminal parameters.
pub mod params;
/// Canonical Steiner, Asymmetric Maze, and Tokyo-Rail spatial configurations.
pub mod presets;
mod skeleton;

#[cfg(test)]
mod tests;

/// Minimum supported grid dimension along each axis.
pub const MIN_PHYSARUM_GRID: usize = 64;
/// Maximum supported grid dimension along each axis.
pub const MAX_PHYSARUM_GRID: usize = 256;
/// Minimum supported Lagrangian chemotactic agent count.
pub const MIN_PHYSARUM_AGENTS: usize = 256;
/// Maximum supported pre-allocated Lagrangian chemotactic agent count.
pub const MAX_PHYSARUM_AGENTS: usize = 16_384;
/// Minimum number of nutrient beacon terminals for network routing benchmarks.
pub const MIN_PHYSARUM_TERMINALS: usize = 2;
/// Maximum number of nutrient beacon terminals supported per simulation.
pub const MAX_PHYSARUM_TERMINALS: usize = 16;

pub use engine::PhysarumEngine;
pub use network::{PhysarumNetworkBenchmark, PhysarumSkeletonEdge, extract_physarum_benchmark};
pub use params::{PhysarumParams, PhysarumTelemetry, PhysarumTerminal};
pub use presets::{
    MazeCorridorRegions, PhysarumPresetKind, asymmetric_maze_corridors, write_preset_obstacles,
};

// [cite:biology]
// [cite:dijkstra]
