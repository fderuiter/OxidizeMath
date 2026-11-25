// [cite:engine_pde]
use diagnostics::CoreDiagnostic;
#[cfg(feature = "std")]
use diagnostics::{Diagnostic, Severity};
use math_commons::error::MathError;
#[cfg(feature = "std")]
use std::collections::HashMap;
use thiserror::Error;

/// Stack-allocated `#![no_std]` error type for partial and stochastic partial differential equation solvers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PdeError {
    /// Grid axis extent was zero, below stencil minimum, or overflowed `usize`.
    #[error("Invalid grid dimension on axis '{axis}': {size}")]
    InvalidGridDimension {
        /// Static axis label (`"n"`, `"width"`, `"height"`, `"depth"`, or `"grid"`).
        axis: &'static str,
        /// Rejected grid size.
        size: usize,
    },

    /// Radix-2 spectral solver received a grid extent that is zero or not a power of two.
    #[error("Fourier grid axis '{axis}' of length {size} must be a power of two")]
    NonRadix2Dimension {
        /// Static axis label.
        axis: &'static str,
        /// Non-radix-2 grid size.
        size: usize,
    },

    /// Requested cell coordinate exceeded the grid axis extent.
    #[error("Grid index {index} out of bounds on axis '{axis}' of length {len}")]
    IndexOutOfBounds {
        /// Static axis label.
        axis: &'static str,
        /// Out-of-bounds cell index.
        index: usize,
        /// Grid axis length.
        len: usize,
    },

    /// Field, spectrum, or column scratch buffer was shorter than required.
    #[error("Buffer '{buffer}' too small: required at least {required}, found {actual}")]
    BufferTooSmall {
        /// Static buffer label.
        buffer: &'static str,
        /// Minimum slice length required.
        required: usize,
        /// Actual slice length provided.
        actual: usize,
    },

    /// Physical or numerical PDE parameter (`dx`, `dt`, `mobility`, `kappa`, `nu`, `diffusivity`) was invalid.
    #[error("Invalid PDE solver parameter: {parameter}")]
    InvalidParameter {
        /// Static parameter identifier.
        parameter: &'static str,
    },

    /// Heap scratch allocation for [`crate::PdeOwnedWorkspace`] failed.
    #[error("Failed to allocate PDE workspace of capacity {capacity}")]
    WorkspaceAllocationFailed {
        /// Requested scalar capacity.
        capacity: usize,
    },

    /// Underlying workspace or primitive math error.
    #[error("{0}")]
    Math(#[from] MathError),
}

impl CoreDiagnostic for PdeError {
    fn error_type(&self) -> &'static str {
        "PdeError"
    }

    fn code(&self) -> &'static str {
        match self {
            Self::InvalidGridDimension { .. } => "PDE-001",
            Self::NonRadix2Dimension { .. } => "PDE-002",
            Self::IndexOutOfBounds { .. } => "PDE-003",
            Self::BufferTooSmall { .. } => "PDE-004",
            Self::InvalidParameter { .. } => "PDE-005",
            Self::WorkspaceAllocationFailed { .. } => "PDE-006",
            Self::Math(_) => "PDE-007",
        }
    }
}

#[cfg(feature = "std")]
impl Diagnostic for PdeError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), self.error_type().to_string());
        map.insert("code".to_string(), self.code().to_string());
        map.insert("description".to_string(), self.to_string());
        match self {
            Self::InvalidGridDimension { axis, size } | Self::NonRadix2Dimension { axis, size } => {
                map.insert("axis".to_string(), (*axis).to_string());
                map.insert("size".to_string(), size.to_string());
            }
            Self::IndexOutOfBounds { axis, index, len } => {
                map.insert("axis".to_string(), (*axis).to_string());
                map.insert("index".to_string(), index.to_string());
                map.insert("len".to_string(), len.to_string());
            }
            Self::BufferTooSmall {
                buffer,
                required,
                actual,
            } => {
                map.insert("buffer".to_string(), (*buffer).to_string());
                map.insert("required".to_string(), required.to_string());
                map.insert("actual".to_string(), actual.to_string());
            }
            Self::InvalidParameter { parameter } => {
                map.insert("parameter".to_string(), (*parameter).to_string());
            }
            Self::WorkspaceAllocationFailed { capacity } => {
                map.insert("capacity".to_string(), capacity.to_string());
            }
            Self::Math(inner) => {
                for (k, v) in inner.metadata() {
                    map.entry(k).or_insert(v);
                }
            }
        }
        map
    }
}
