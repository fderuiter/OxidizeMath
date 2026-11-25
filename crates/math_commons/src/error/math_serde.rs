// [cite:math_commons]
//! Zero-allocation `#![no_std]` `Deserialize` implementation for [`super::MathError`].

use super::MathError;
use crate::math_kernel::types::Dimension;

const KNOWN_NUMERICAL_REASONS: &[&str] = &[
    "singular",
    "numerical computation failure",
    "Failed to solve linear system",
    "Singular matrix encountered during Gaussian elimination",
    "Chisquared distribution parameter nu must be positive",
    "Eigenvalues contain NaN",
    "CSR graph exceeds maximum spin system bounds",
    "CSR row_ptr must have length n_nodes + 1 starting at 0",
    "CSR col_idx and weights lengths must match row_ptr[n_nodes]",
    "CSR row_ptr must be monotonically non-decreasing",
    "CSR column indices must be < n_nodes and weights must be finite",
    "Failed to compute linear regression slope for Lyapunov exponent",
    "Failed to initialize multivariate QMC integrator",
    "Multivariate QMC unit-cube integration failed",
    "Failed to invert (V-U) in Padé approximation",
    "Gillespie CTMC step failed",
    "Failed to create initial distribution",
    "Failed to create emission distribution",
    "Failed to create transition distribution",
    "Calibration input slices must be non-empty",
    "Clipping epsilon must be finite and within [f64::EPSILON, 0.5)",
    "Target class index out of bounds for multiclass probability row",
    "Multiclass probability row must sum to 1.0",
    "multiclass_log_loss requires at least 2 classes",
    "multiclass_log_loss requires non-empty targets",
    "Sample count * num_classes overflowed usize",
    "Gaussian CRPS mean and observation must be finite",
    "Failed to construct standard normal distribution",
    "dataset_gaussian_crps requires non-empty slices",
    "Ensemble sample at index 0 is non-finite",
    "ensemble_crps_sorted requires finite non-decreasing samples",
    "ensemble_crps_sorted requires at least one ensemble member",
    "Observation must be finite for ensemble CRPS",
    "Number of calibration bins must be at least 1",
    "Calibration logits must be non-empty",
    "Calibration logits must be finite",
    "Calibration fitting requires both positive and negative outcomes",
    "Invalid PlattConfig hyperparameters",
    "Input logit must be finite",
    "fit_multiclass requires num_classes >= 2 and non-empty targets",
    "targets.len() * num_classes overflowed usize",
    "fit_multiclass requires finite logits and in-bounds class targets",
    "predict_multiclass_into requires non-empty finite logits",
    "PAVA isotonic regression requires at least one observation",
    "PAVA requires finite scores, targets in [0, 1], and positive weights",
    "fit_pava_sorted_into requires non-decreasing sorted_scores",
    "Cannot predict with an empty PAVA block slice",
    "Prediction score must be finite",
    "IsotonicCalibrator::fit requires non-empty inputs",
    "IsotonicCalibrator::fit requires finite scores",
    "Threshold ranking inputs must be non-empty",
    "Ranking scores must be finite numbers",
    "ROC and PR curves require both positive and negative outcomes",
    "ensemble_crps_loss requires non-empty ensemble members and observations",
    "Ensemble forecast contains non-finite value",
    "Probability vector contains non-finite or out-of-[0, 1] value",
    "cross_entropy_loss requires non-empty vectors",
    "Non-finite state during Lyapunov integration",
    "Negative argument for sqrt in Li & Ma",
    "Mask grid size overflow",
    "Lattice directions must be within [-1, 1]",
    "Shepp-Logan grid dimensions overflow",
    "2D Ginzburg-Landau grid dimensions overflowed usize",
    "non-finite scalar cannot form an interval enclosure",
    "interval division by zero-containing enclosure",
    "2D upwind advection grid dimensions overflow usize",
];

const KNOWN_CONVERSION_REASONS: &[&str] = &[
    "overflow",
    "numeric conversion failure",
    "Failed to convert usize to T",
    "Failed to convert 2.0 to T",
    "Failed to convert 1e-15 to T",
    "Failed to convert 16.0 to T",
    "Adam step counter exceeded i32::MAX",
    "Failed to convert constant 2.0",
    "Failed to convert tolerance 1e-15",
    "Failed to convert constant 16.0",
    "Failed to convert dimension",
    "Failed to convert step counter to i32",
    "Failed to convert holding time",
];

struct StaticNumericalReason(&'static str);
struct StaticConversionReason(&'static str);

fn lookup_static(table: &[&'static str], fallback: &'static str, candidate: &str) -> &'static str {
    for &known in table {
        if known == candidate {
            return known;
        }
    }
    fallback
}

fn deserialize_numerical_reason<'de, D>(deserializer: D) -> Result<StaticNumericalReason, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct ReasonVisitor;

    impl serde::de::Visitor<'_> for ReasonVisitor {
        type Value = StaticNumericalReason;

        fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            formatter.write_str("a numerical error reason string")
        }

        fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(StaticNumericalReason(lookup_static(
                KNOWN_NUMERICAL_REASONS,
                "numerical computation failure",
                v,
            )))
        }
    }

    deserializer.deserialize_str(ReasonVisitor)
}

fn deserialize_conversion_reason<'de, D>(
    deserializer: D,
) -> Result<StaticConversionReason, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct ReasonVisitor;

    impl serde::de::Visitor<'_> for ReasonVisitor {
        type Value = StaticConversionReason;

        fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            formatter.write_str("a conversion error reason string")
        }

        fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(StaticConversionReason(lookup_static(
                KNOWN_CONVERSION_REASONS,
                "numeric conversion failure",
                v,
            )))
        }
    }

    deserializer.deserialize_str(ReasonVisitor)
}

#[derive(serde::Deserialize)]
enum MathErrorRepr {
    DimensionMismatch {
        expected: Dimension,
        actual: Dimension,
    },
    OutOfBounds {
        value: f64,
        min: f64,
        max: f64,
    },
    StrictlyOutOfBounds {
        value: f64,
        min: f64,
        max: f64,
    },
    InvalidPermutation {
        index: usize,
        value: usize,
        len: usize,
    },
    WorkspaceTooSmall {
        required: usize,
        available: usize,
    },
    NumericalError {
        #[serde(deserialize_with = "deserialize_numerical_reason")]
        reason: StaticNumericalReason,
    },
    ConversionError {
        #[serde(deserialize_with = "deserialize_conversion_reason")]
        reason: StaticConversionReason,
    },
}

impl From<MathErrorRepr> for MathError {
    fn from(repr: MathErrorRepr) -> Self {
        match repr {
            MathErrorRepr::DimensionMismatch { expected, actual } => {
                Self::DimensionMismatch { expected, actual }
            }
            MathErrorRepr::OutOfBounds { value, min, max } => Self::OutOfBounds { value, min, max },
            MathErrorRepr::StrictlyOutOfBounds { value, min, max } => {
                Self::StrictlyOutOfBounds { value, min, max }
            }
            MathErrorRepr::InvalidPermutation { index, value, len } => {
                Self::InvalidPermutation { index, value, len }
            }
            MathErrorRepr::WorkspaceTooSmall {
                required,
                available,
            } => Self::WorkspaceTooSmall {
                required,
                available,
            },
            MathErrorRepr::NumericalError {
                reason: StaticNumericalReason(reason),
            } => Self::NumericalError { reason },
            MathErrorRepr::ConversionError {
                reason: StaticConversionReason(reason),
            } => Self::ConversionError { reason },
        }
    }
}

impl<'de> serde::Deserialize<'de> for MathError {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        MathErrorRepr::deserialize(deserializer).map(Self::from)
    }
}
