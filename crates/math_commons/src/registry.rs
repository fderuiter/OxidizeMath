//! Unified Math Registry
//! Centralized repository for numerical standards and grid limits.

/// High precision tolerance for critical calculations (e.g., 1e-10).
pub const TOLERANCE_HIGH: f64 = 1e-10;

/// Standard precision tolerance for general calculations (e.g., 1e-9).
pub const TOLERANCE_STANDARD: f64 = 1e-9;

/// Fast approximate tolerance for performance-critical or less sensitive calculations (e.g., 1e-6).
pub const TOLERANCE_FAST: f64 = 1e-6;

/// High precision tolerance for critical calculations in f32 (e.g., 1e-10).
pub const TOLERANCE_HIGH_F32: f32 = 1e-10;

/// Standard precision tolerance for general calculations in f32 (e.g., 1e-9).
pub const TOLERANCE_STANDARD_F32: f32 = 1e-9;

/// Fast approximate tolerance for performance-critical or less sensitive calculations in f32 (e.g., 1e-6).
pub const TOLERANCE_FAST_F32: f32 = 1e-6;

/// Common grid sizes and iteration limits.
pub const MAX_GRID_SIZE: usize = 1000;
/// Default upper bound on iterative numerical solver steps.
pub const MAX_ITERATIONS: usize = 5000;
