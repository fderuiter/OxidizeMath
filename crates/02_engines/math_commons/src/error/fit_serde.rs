// [cite:math_commons]
//! Zero-allocation `#![no_std]` `Deserialize` implementation for [`super::FitError`].

use super::FitError;

const KNOWN_INVALID_DOMAIN_REASONS: &[&str] = &[
    "non-finite exponential seed parameters",
    "exponential fit requires at least 2 points with y > 0",
    "power-law fit requires at least 2 points with x > 0 and y > 0",
    "non-finite power-law seed parameters",
    "logarithmic fit requires at least 2 points with x > 0",
    "logistic fit requires positive response values",
    "sinusoidal fit requires non-degenerate x span",
    "polynomial FitModel passed to fit_nonlinear",
    "exponential log-space fit requires at least 2 points with y > 0",
    "power-law log-space fit requires at least 2 points with x > 0 and y > 0",
    "insufficient positive coordinates for logarithmic polynomial fit",
    "non-finite initial residual sum of squares",
    "cycle count in calibration series must be non-negative",
    "state-of-health capacity fraction must lie within [0.0, 1.2]",
    "calibration series must include positive cycle counts",
    "depth of discharge must be strictly positive for cycle calibration",
    "non-finite calibrated battery aging parameters",
    "non-finite calibrated residual sum of squares",
    "custom formula requires fit_custom_evaluator",
    "invalid curve-fitting domain",
];

struct StaticReason(&'static str);

fn deserialize_static_reason<'de, D>(deserializer: D) -> Result<StaticReason, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct StaticReasonVisitor;

    impl serde::de::Visitor<'_> for StaticReasonVisitor {
        type Value = StaticReason;

        fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            formatter.write_str("a curve-fitting domain reason string")
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            for &known in KNOWN_INVALID_DOMAIN_REASONS {
                if known == v {
                    return Ok(StaticReason(known));
                }
            }
            Ok(StaticReason("invalid curve-fitting domain"))
        }
    }

    deserializer.deserialize_str(StaticReasonVisitor)
}

#[derive(serde::Deserialize)]
enum FitErrorRepr {
    InsufficientPoints {
        required: usize,
        actual: usize,
    },
    UnsupportedDegree {
        degree: usize,
    },
    InvalidDomain {
        #[serde(deserialize_with = "deserialize_static_reason")]
        reason: StaticReason,
    },
    SingularMatrix,
    NonConvergence {
        iterations: usize,
    },
}

impl From<FitErrorRepr> for FitError {
    fn from(repr: FitErrorRepr) -> Self {
        match repr {
            FitErrorRepr::InsufficientPoints { required, actual } => {
                Self::InsufficientPoints { required, actual }
            }
            FitErrorRepr::UnsupportedDegree { degree } => Self::UnsupportedDegree { degree },
            FitErrorRepr::InvalidDomain {
                reason: StaticReason(reason),
            } => Self::InvalidDomain { reason },
            FitErrorRepr::SingularMatrix => Self::SingularMatrix,
            FitErrorRepr::NonConvergence { iterations } => Self::NonConvergence { iterations },
        }
    }
}

impl<'de> serde::Deserialize<'de> for FitError {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        FitErrorRepr::deserialize(deserializer).map(Self::from)
    }
}
