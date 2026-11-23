#![no_std]
#![doc = include_str!("../README.md")]
// [cite:engine_symbolic]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(test)]
extern crate std;

/// Forward-mode `Dual` scalar numbers over real and complex tangents.
pub mod dual;
mod dual_ops;
/// Second-order $N$-variate Taylor jets `Jet2<N>` for exact Hessians and curvatures.
pub mod jet2;
mod jet2_ops;
/// Fixed-capacity zero-allocation scratch workspace for reverse-mode autodiff.
pub mod workspace;

/// Bounded `#![no_std]` + `alloc` e-graph equality saturation and minimum-cost extraction.
#[cfg(feature = "alloc")]
pub mod egraph;
#[cfg(feature = "alloc")]
mod egraph_rules;
/// Arena-indexed symbolic operators and extracted DAG nodes.
#[cfg(feature = "alloc")]
pub mod sym;

#[cfg(test)]
mod tests;

pub use dual::{Dual, Dual64, DualComplex64, DualScalar};
pub use jet2::{Jet2, Jet2_3D, Jet2D};
pub use workspace::{MAX_WORKSPACE_STACK, MAX_WORKSPACE_TAPE, SolverWorkspace};

#[cfg(feature = "alloc")]
pub use egraph::{EGraph, MAX_EGRAPH_NODES, MAX_REWRITE_PASSES};
#[cfg(feature = "alloc")]
pub use sym::{SymExpr, SymFunc1, SymFunc2, SymOp, canonical_f64_bits};
