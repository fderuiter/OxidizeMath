#![doc = include_str!("../README.md")]
/// Single-pass `syn` AST visitor for verification and traceability metrics.
#[cfg(feature = "ast-analysis")]
pub mod ast_visitor;
/// Discrete 2D grid boundary-condition index resolvers.
pub mod boundary;
/// Zero-allocation interior/boundary split iteration helpers for 2D grids.
pub mod iteration;
/// 3D triangle mesh primitives and OBJ, PLY, STL, and Gaussian-splat exporters.
pub mod mesh;
/// Cross-platform forward-slash path normalization utilities.
pub mod path_utils;
/// Deterministic `ChaCha8Rng` random-number generator enforcing the No-Entropy policy.
pub mod rng;
/// Unified workspace traceability engine, AST cache, and certificate schema.
pub mod traceability;
/// Asynchronous virtual file system abstraction across native and WebAssembly targets.
pub mod vfs;
/// Workspace `Cargo.toml` root discovery and member manifest parsers.
pub mod workspace;
