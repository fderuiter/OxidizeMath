// [cite:engine_pde]
use core::fmt;
use math_commons::error::MathError;
use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};

/// Stack-allocated `#![no_std]` error type for partial and stochastic partial differential equation solvers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum PdeError {
    /// Grid axis extent was zero, below stencil minimum, or overflowed `usize`.
    InvalidGridDimension {
        /// Static axis label (`"n"`, `"width"`, `"height"`, `"depth"`, or `"grid"`).
        axis: &'static str,
        /// Rejected grid size.
        size: usize,
    },

    /// Radix-2 spectral solver received a grid extent that is zero or not a power of two.
    NonRadix2Dimension {
        /// Static axis label.
        axis: &'static str,
        /// Non-radix-2 grid size.
        size: usize,
    },

    /// Requested cell coordinate exceeded the grid axis extent.
    IndexOutOfBounds {
        /// Static axis label.
        axis: &'static str,
        /// Out-of-bounds cell index.
        index: usize,
        /// Grid axis length.
        len: usize,
    },

    /// Field, spectrum, or column scratch buffer was shorter than required.
    BufferTooSmall {
        /// Static buffer label.
        buffer: &'static str,
        /// Minimum slice length required.
        required: usize,
        /// Actual slice length provided.
        actual: usize,
    },

    /// Physical or numerical PDE parameter (`dx`, `dt`, `mobility`, `kappa`, `nu`, `diffusivity`) was invalid.
    InvalidParameter {
        /// Static parameter identifier.
        parameter: &'static str,
    },

    /// Heap scratch allocation for [`crate::PdeOwnedWorkspace`] failed.
    WorkspaceAllocationFailed {
        /// Requested scalar capacity.
        capacity: usize,
    },

    /// Underlying workspace or primitive math error.
    Math(MathError),
}

impl PdeError {
    /// Returns the static type identifier `"PdeError"`.
    #[inline]
    pub const fn error_type(&self) -> &'static str {
        "PdeError"
    }

    /// Returns the stable diagnostic error code (`PDE-001`..`PDE-007`).
    #[inline]
    pub const fn code(&self) -> &'static str {
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

impl fmt::Display for PdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGridDimension { axis, size } => {
                write!(f, "Invalid grid dimension on axis '{axis}': {size}")
            }
            Self::NonRadix2Dimension { axis, size } => {
                write!(
                    f,
                    "Fourier grid axis '{axis}' of length {size} must be a power of two"
                )
            }
            Self::IndexOutOfBounds { axis, index, len } => {
                write!(
                    f,
                    "Grid index {index} out of bounds on axis '{axis}' of length {len}"
                )
            }
            Self::BufferTooSmall {
                buffer,
                required,
                actual,
            } => write!(
                f,
                "Buffer '{buffer}' too small: required at least {required}, found {actual}"
            ),
            Self::InvalidParameter { parameter } => {
                write!(f, "Invalid PDE solver parameter: {parameter}")
            }
            Self::WorkspaceAllocationFailed { capacity } => {
                write!(f, "Failed to allocate PDE workspace of capacity {capacity}")
            }
            Self::Math(inner) => fmt::Display::fmt(inner, f),
        }
    }
}

impl core::error::Error for PdeError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Math(inner) => Some(inner),
            _ => None,
        }
    }
}

impl From<MathError> for PdeError {
    #[inline]
    fn from(err: MathError) -> Self {
        Self::Math(err)
    }
}

impl Diagnostic for PdeError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Math(inner) = self {
            inner.visit_fields(visitor);
        }
        visitor.record_str("error_type", self.error_type());
        visitor.record_str("code", self.code());
        visitor.record_display("description", self);
        match self {
            Self::InvalidGridDimension { axis, size } | Self::NonRadix2Dimension { axis, size } => {
                visitor.record_str("axis", axis);
                visitor.record_usize("size", *size);
            }
            Self::IndexOutOfBounds { axis, index, len } => {
                visitor.record_str("axis", axis);
                visitor.record_usize("index", *index);
                visitor.record_usize("len", *len);
            }
            Self::BufferTooSmall {
                buffer,
                required,
                actual,
            } => {
                visitor.record_str("buffer", buffer);
                visitor.record_usize("required", *required);
                visitor.record_usize("actual", *actual);
            }
            Self::InvalidParameter { parameter } => {
                visitor.record_str("parameter", parameter);
            }
            Self::WorkspaceAllocationFailed { capacity } => {
                visitor.record_usize("capacity", *capacity);
            }
            Self::Math(_) => {}
        }
    }
}

#[cfg(feature = "serde")]
mod pde_serde {
    use super::PdeError;
    use math_commons::error::MathError;

    const KNOWN_AXES: &[&str] = &[
        "n", "width", "height", "depth", "grid", "x", "y", "z", "axis",
    ];
    const KNOWN_BUFFERS: &[&str] = &[
        "buf",
        "col_scratch",
        "spec",
        "phi",
        "h",
        "field",
        "u",
        "out_lap",
        "laplacian",
        "buffer",
    ];
    const KNOWN_PARAMETERS: &[&str] = &[
        "dx",
        "dy",
        "dz",
        "dt",
        "mobility",
        "kappa",
        "mobility_temp",
        "nu",
        "lambda",
        "intensity",
        "diffusivity",
        "parameter",
    ];

    struct StaticAxis(&'static str);
    struct StaticBuffer(&'static str);
    struct StaticParam(&'static str);

    fn lookup(table: &[&'static str], fallback: &'static str, candidate: &str) -> &'static str {
        for &known in table {
            if known == candidate {
                return known;
            }
        }
        fallback
    }

    fn de_axis<'de, D: serde::Deserializer<'de>>(d: D) -> Result<StaticAxis, D::Error> {
        struct Visitor;
        impl serde::de::Visitor<'_> for Visitor {
            type Value = StaticAxis;
            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("a grid axis label")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(StaticAxis(lookup(KNOWN_AXES, "axis", v)))
            }
        }
        d.deserialize_str(Visitor)
    }

    fn de_buffer<'de, D: serde::Deserializer<'de>>(d: D) -> Result<StaticBuffer, D::Error> {
        struct Visitor;
        impl serde::de::Visitor<'_> for Visitor {
            type Value = StaticBuffer;
            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("a PDE buffer label")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(StaticBuffer(lookup(KNOWN_BUFFERS, "buffer", v)))
            }
        }
        d.deserialize_str(Visitor)
    }

    fn de_param<'de, D: serde::Deserializer<'de>>(d: D) -> Result<StaticParam, D::Error> {
        struct Visitor;
        impl serde::de::Visitor<'_> for Visitor {
            type Value = StaticParam;
            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("a PDE parameter identifier")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(StaticParam(lookup(KNOWN_PARAMETERS, "parameter", v)))
            }
        }
        d.deserialize_str(Visitor)
    }

    #[derive(serde::Deserialize)]
    enum PdeErrorRepr {
        InvalidGridDimension {
            #[serde(deserialize_with = "de_axis")]
            axis: StaticAxis,
            size: usize,
        },
        NonRadix2Dimension {
            #[serde(deserialize_with = "de_axis")]
            axis: StaticAxis,
            size: usize,
        },
        IndexOutOfBounds {
            #[serde(deserialize_with = "de_axis")]
            axis: StaticAxis,
            index: usize,
            len: usize,
        },
        BufferTooSmall {
            #[serde(deserialize_with = "de_buffer")]
            buffer: StaticBuffer,
            required: usize,
            actual: usize,
        },
        InvalidParameter {
            #[serde(deserialize_with = "de_param")]
            parameter: StaticParam,
        },
        WorkspaceAllocationFailed {
            capacity: usize,
        },
        Math(MathError),
    }

    impl<'de> serde::Deserialize<'de> for PdeError {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            Ok(match PdeErrorRepr::deserialize(deserializer)? {
                PdeErrorRepr::InvalidGridDimension {
                    axis: StaticAxis(axis),
                    size,
                } => Self::InvalidGridDimension { axis, size },
                PdeErrorRepr::NonRadix2Dimension {
                    axis: StaticAxis(axis),
                    size,
                } => Self::NonRadix2Dimension { axis, size },
                PdeErrorRepr::IndexOutOfBounds {
                    axis: StaticAxis(axis),
                    index,
                    len,
                } => Self::IndexOutOfBounds { axis, index, len },
                PdeErrorRepr::BufferTooSmall {
                    buffer: StaticBuffer(buffer),
                    required,
                    actual,
                } => Self::BufferTooSmall {
                    buffer,
                    required,
                    actual,
                },
                PdeErrorRepr::InvalidParameter {
                    parameter: StaticParam(parameter),
                } => Self::InvalidParameter { parameter },
                PdeErrorRepr::WorkspaceAllocationFailed { capacity } => {
                    Self::WorkspaceAllocationFailed { capacity }
                }
                PdeErrorRepr::Math(err) => Self::Math(err),
            })
        }
    }
}
