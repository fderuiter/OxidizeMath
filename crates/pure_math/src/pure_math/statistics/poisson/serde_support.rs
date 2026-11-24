//! Validated `serde` deserialization and round-trip tests for Poisson arrival processes.
// [cite:clinical_trials_statistics]

use super::{HomogeneousPoisson, InhomogeneousPoisson, IntensityProfile};
use serde::{Deserialize, Deserializer};

impl<'de> Deserialize<'de> for HomogeneousPoisson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawHomogeneousPoisson {
            rate: f64,
        }
        let raw = RawHomogeneousPoisson::deserialize(deserializer)?;
        Self::new(raw.rate).map_err(serde::de::Error::custom)
    }
}

impl<'de> Deserialize<'de> for InhomogeneousPoisson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawInhomogeneousPoisson {
            profile: IntensityProfile,
        }
        let raw = RawInhomogeneousPoisson::deserialize(deserializer)?;
        Self::new(raw.profile).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{HomogeneousPoisson, InhomogeneousPoisson, IntensityProfile, PoissonError};

    #[test]
    #[verified_engine::verified]
    fn test_poisson_serde_roundtrip_and_invariant_rejection() {
        let hp = HomogeneousPoisson::new(3.25).unwrap();
        let hp_json = serde_json::to_string(&hp).unwrap();
        let hp_restored: HomogeneousPoisson = serde_json::from_str(&hp_json).unwrap();
        assert_eq!(hp_restored, hp);
        assert!(serde_json::from_str::<HomogeneousPoisson>(r#"{"rate":0.0}"#).is_err());
        assert!(serde_json::from_str::<HomogeneousPoisson>(r#"{"rate":-1.5}"#).is_err());
        assert!(serde_json::from_str::<HomogeneousPoisson>(r#"{"rate":1.0,"extra":1}"#).is_err());

        let profile = IntensityProfile::StepBurst {
            base_rate: 2.0,
            burst_rate: 6.0,
            burst_start: 1.0,
            burst_end: 3.0,
        };
        let ip = InhomogeneousPoisson::new(profile).unwrap();
        let ip_json = serde_json::to_string(&ip).unwrap();
        let ip_restored: InhomogeneousPoisson = serde_json::from_str(&ip_json).unwrap();
        assert_eq!(ip_restored, ip);

        let bad_ip = r#"{"profile":{"Sinusoidal":{"base_rate":2.0,"amplitude":5.0,"angular_freq":1.0,"phase":0.0}}}"#;
        assert!(serde_json::from_str::<InhomogeneousPoisson>(bad_ip).is_err());

        let err = PoissonError::InvalidHorizon { horizon: -1.0 };
        let err_restored: PoissonError =
            serde_json::from_str(&serde_json::to_string(&err).unwrap()).unwrap();
        assert_eq!(err_restored, err);
    }
}
