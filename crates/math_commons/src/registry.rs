//! Unified Math Registry
//! Centralized repository for numerical standards and grid limits.

/// High precision tolerance for critical calculations (e.g., 1e-10).
pub const TOLERANCE_HIGH: f64 = 1e-10;

/// Standard precision tolerance for general calculations (e.g., 1e-9).
pub const TOLERANCE_STANDARD: f64 = 1e-9;

/// Fast approximate tolerance for performance-critical or less sensitive calculations (e.g., 1e-6).
pub const TOLERANCE_FAST: f64 = 1e-6;

/// High precision tolerance for critical calculations in `f32` (`2.5e-7`, strictly above `f32::EPSILON`).
pub const TOLERANCE_HIGH_F32: f32 = 2.5e-7;

/// Standard precision tolerance for general calculations in `f32` (`5.0e-7`, strictly above `f32::EPSILON`).
pub const TOLERANCE_STANDARD_F32: f32 = 5.0e-7;

/// Fast approximate tolerance for performance-critical or less sensitive calculations in `f32` (`1.0e-6`).
pub const TOLERANCE_FAST_F32: f32 = 1.0e-6;

/// Common grid sizes and iteration limits.
pub const MAX_GRID_SIZE: usize = 1000;
#[allow(missing_docs)]
pub const MAX_ITERATIONS: usize = 5000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn test_f32_tolerances_exceed_machine_epsilon() {
        const { assert!(TOLERANCE_HIGH_F32 > f32::EPSILON) };
        const { assert!(TOLERANCE_STANDARD_F32 > TOLERANCE_HIGH_F32) };
        const { assert!(TOLERANCE_FAST_F32 > TOLERANCE_STANDARD_F32) };
        assert!(core::hint::black_box(TOLERANCE_HIGH_F32) > f32::EPSILON);
        assert!(core::hint::black_box(TOLERANCE_STANDARD_F32) > f32::EPSILON);
        assert!(core::hint::black_box(TOLERANCE_FAST_F32) > f32::EPSILON);
    }
}
