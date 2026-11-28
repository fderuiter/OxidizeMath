// [cite:engine_diffeq]
use core::fmt;
use math_commons::error::MathError;
use math_commons::math_kernel::types::Scalar;
use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use math_memory::{ArenaError, ByteCursor};

/// Stack-allocated `#![no_std]` errors returned during Kalman filter initialization and updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum KalmanError {
    /// Innovation covariance matrix $\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k$ was singular.
    MatrixInversionError,
    /// Underlying dimension mismatch or linear algebra error from `math_commons`.
    Math(MathError),
    /// Required builder field (`initial_state` or `initial_covariance`) was omitted.
    InitializationError(&'static str),
}

impl fmt::Display for KalmanError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatrixInversionError => {
                f.write_str("Failed to invert innovation covariance matrix (singular)")
            }
            Self::Math(err) => write!(f, "Mathematical error: {err}"),
            Self::InitializationError(field) => {
                write!(f, "Missing required initialization field: {field}")
            }
        }
    }
}

impl core::error::Error for KalmanError {
    #[verified_engine::verified]
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Math(err) => Some(err),
            Self::MatrixInversionError | Self::InitializationError(_) => None,
        }
    }
}

impl From<MathError> for KalmanError {
    #[inline]
    #[verified_engine::verified]
    fn from(err: MathError) -> Self {
        Self::Math(err)
    }
}

impl Diagnostic for KalmanError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Math(math_err) = self {
            math_err.visit_fields(visitor);
        }
        visitor.record_str("error_type", "KalmanError");
        visitor.record_display("description", self);
    }
}

#[cfg(feature = "serde")]
mod kalman_serde {
    use super::KalmanError;
    use math_commons::error::MathError;

    const KNOWN_FIELDS: &[&str] = &["initial_state", "initial_covariance", "field"];

    struct StaticField(&'static str);

    fn de_field<'de, D: serde::Deserializer<'de>>(d: D) -> Result<StaticField, D::Error> {
        struct Visitor;
        impl serde::de::Visitor<'_> for Visitor {
            type Value = StaticField;
            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("a Kalman builder field name")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                for &known in KNOWN_FIELDS {
                    if known == v {
                        return Ok(StaticField(known));
                    }
                }
                Ok(StaticField("field"))
            }
        }
        d.deserialize_str(Visitor)
    }

    #[derive(serde::Deserialize)]
    enum KalmanErrorRepr {
        MatrixInversionError,
        Math(MathError),
        InitializationError(#[serde(deserialize_with = "de_field")] StaticField),
    }

    impl<'de> serde::Deserialize<'de> for KalmanError {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            Ok(match KalmanErrorRepr::deserialize(deserializer)? {
                KalmanErrorRepr::MatrixInversionError => Self::MatrixInversionError,
                KalmanErrorRepr::Math(err) => Self::Math(err),
                KalmanErrorRepr::InitializationError(StaticField(field)) => {
                    Self::InitializationError(field)
                }
            })
        }
    }
}

/// Stack-allocated `#![no_std]` error type for differential and stochastic equation solvers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum DiffeqError {
    /// Physical or numerical solver parameter violated positivity or finiteness invariants.
    InvalidParameter {
        /// Static identifier of the invalid parameter.
        parameter: &'static str,
    },

    /// Integration step size `dt` was negative or non-finite.
    InvalidStepSize,

    /// Caller-supplied state, gradient, or Wiener increment slice was shorter than required.
    BufferTooSmall {
        /// Static label of the undersized slice.
        buffer: &'static str,
        /// Minimum slice length required.
        required: usize,
        /// Actual slice length provided.
        actual: usize,
    },

    /// Heap scratch allocation for an arena workspace failed.
    WorkspaceAllocationFailed {
        /// Requested scalar capacity.
        capacity: usize,
    },

    /// Underlying workspace or primitive math error.
    Math(MathError),
}

impl DiffeqError {
    /// Returns the static type identifier `"DiffeqError"`.
    #[inline]
    pub const fn error_type(&self) -> &'static str {
        "DiffeqError"
    }

    /// Returns the stable diagnostic error code (`DIFFEQ-001`..`DIFFEQ-005`).
    #[inline]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidParameter { .. } => "DIFFEQ-001",
            Self::InvalidStepSize => "DIFFEQ-002",
            Self::BufferTooSmall { .. } => "DIFFEQ-003",
            Self::WorkspaceAllocationFailed { .. } => "DIFFEQ-004",
            Self::Math(_) => "DIFFEQ-005",
        }
    }
}

impl fmt::Display for DiffeqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { parameter } => {
                write!(
                    f,
                    "Invalid differential equation solver parameter: {parameter}"
                )
            }
            Self::InvalidStepSize => {
                f.write_str("Integration step size dt must be finite and non-negative")
            }
            Self::BufferTooSmall {
                buffer,
                required,
                actual,
            } => write!(
                f,
                "Buffer '{buffer}' too small: required at least {required}, found {actual}"
            ),
            Self::WorkspaceAllocationFailed { capacity } => {
                write!(
                    f,
                    "Failed to allocate solver workspace of capacity {capacity}"
                )
            }
            Self::Math(inner) => fmt::Display::fmt(inner, f),
        }
    }
}

impl core::error::Error for DiffeqError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Math(inner) => Some(inner),
            _ => None,
        }
    }
}

impl From<MathError> for DiffeqError {
    #[inline]
    fn from(err: MathError) -> Self {
        Self::Math(err)
    }
}

impl From<ArenaError> for DiffeqError {
    #[inline]
    fn from(err: ArenaError) -> Self {
        Self::Math(MathError::from(err))
    }
}

/// Carves `count` zeroed scalar `T` elements from `cur`, mapping byte exhaustion to scalar counts.
#[inline]
#[verified_engine::verified]
pub(crate) fn carve_scalars<'a, T: Scalar>(
    cur: &mut ByteCursor<'a>,
    count: usize,
) -> Result<&'a mut [T], DiffeqError> {
    let elem_size = core::mem::size_of::<T>();
    cur.alloc_slice::<T>(count).map_err(|err| match err {
        ArenaError::InsufficientBytes { available, .. } => {
            DiffeqError::Math(MathError::WorkspaceTooSmall {
                required: count,
                available: available
                    .checked_div(elem_size)
                    .unwrap_or(0)
                    .min(count.saturating_sub(1)),
            })
        }
        other => DiffeqError::from(other),
    })
}

impl Diagnostic for DiffeqError {
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
            Self::InvalidParameter { parameter } => {
                visitor.record_str("parameter", parameter);
            }
            Self::InvalidStepSize | Self::Math(_) => {}
            Self::BufferTooSmall {
                buffer,
                required,
                actual,
            } => {
                visitor.record_str("buffer", buffer);
                visitor.record_usize("required", *required);
                visitor.record_usize("actual", *actual);
            }
            Self::WorkspaceAllocationFailed { capacity } => {
                visitor.record_usize("capacity", *capacity);
            }
        }
    }
}

#[cfg(feature = "serde")]
mod diffeq_serde {
    use super::DiffeqError;
    use math_commons::error::MathError;

    const KNOWN_PARAMETERS: &[&str] = &["mass", "friction", "temperature", "dt", "parameter"];
    const KNOWN_BUFFERS: &[&str] = &["q", "p", "state", "dw", "dz", "x", "out_drift", "buffer"];

    struct StaticParam(&'static str);
    struct StaticBuffer(&'static str);

    fn lookup(table: &[&'static str], fallback: &'static str, candidate: &str) -> &'static str {
        for &known in table {
            if known == candidate {
                return known;
            }
        }
        fallback
    }

    fn de_param<'de, D: serde::Deserializer<'de>>(d: D) -> Result<StaticParam, D::Error> {
        struct Visitor;
        impl serde::de::Visitor<'_> for Visitor {
            type Value = StaticParam;
            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("a solver parameter name")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(StaticParam(lookup(KNOWN_PARAMETERS, "parameter", v)))
            }
        }
        d.deserialize_str(Visitor)
    }

    fn de_buffer<'de, D: serde::Deserializer<'de>>(d: D) -> Result<StaticBuffer, D::Error> {
        struct Visitor;
        impl serde::de::Visitor<'_> for Visitor {
            type Value = StaticBuffer;
            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("a solver buffer name")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(StaticBuffer(lookup(KNOWN_BUFFERS, "buffer", v)))
            }
        }
        d.deserialize_str(Visitor)
    }

    #[derive(serde::Deserialize)]
    enum DiffeqErrorRepr {
        InvalidParameter {
            #[serde(deserialize_with = "de_param")]
            parameter: StaticParam,
        },
        InvalidStepSize,
        BufferTooSmall {
            #[serde(deserialize_with = "de_buffer")]
            buffer: StaticBuffer,
            required: usize,
            actual: usize,
        },
        WorkspaceAllocationFailed {
            capacity: usize,
        },
        Math(MathError),
    }

    impl<'de> serde::Deserialize<'de> for DiffeqError {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            Ok(match DiffeqErrorRepr::deserialize(deserializer)? {
                DiffeqErrorRepr::InvalidParameter {
                    parameter: StaticParam(parameter),
                } => Self::InvalidParameter { parameter },
                DiffeqErrorRepr::InvalidStepSize => Self::InvalidStepSize,
                DiffeqErrorRepr::BufferTooSmall {
                    buffer: StaticBuffer(buffer),
                    required,
                    actual,
                } => Self::BufferTooSmall {
                    buffer,
                    required,
                    actual,
                },
                DiffeqErrorRepr::WorkspaceAllocationFailed { capacity } => {
                    Self::WorkspaceAllocationFailed { capacity }
                }
                DiffeqErrorRepr::Math(err) => Self::Math(err),
            })
        }
    }
}
