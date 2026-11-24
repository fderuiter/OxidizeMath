//! Validated `serde` deserialization and round-trip tests for birth-death queueing models.
// [cite:clinical_trials_statistics]

use super::{Mm1Queue, MmcQueue};
use serde::{Deserialize, Deserializer};

impl<'de> Deserialize<'de> for Mm1Queue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawMm1Queue {
            arrival_rate: f64,
            service_rate: f64,
        }
        let raw = RawMm1Queue::deserialize(deserializer)?;
        Self::new(raw.arrival_rate, raw.service_rate).map_err(serde::de::Error::custom)
    }
}

impl<'de> Deserialize<'de> for MmcQueue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawMmcQueue {
            arrival_rate: f64,
            service_rate: f64,
            servers: usize,
        }
        let raw = RawMmcQueue::deserialize(deserializer)?;
        Self::new(raw.arrival_rate, raw.service_rate, raw.servers).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Mm1Queue, MmcQueue, QueueMetrics};

    #[test]
    #[verified_engine::verified]
    fn test_queueing_serde_roundtrip_and_stability_rejection() {
        let mm1 = Mm1Queue::new(2.0, 3.0).unwrap();
        let mm1_json = serde_json::to_string(&mm1).unwrap();
        let mm1_restored: Mm1Queue = serde_json::from_str(&mm1_json).unwrap();
        assert_eq!(mm1_restored, mm1);
        assert!(
            serde_json::from_str::<Mm1Queue>(r#"{"arrival_rate":3.0,"service_rate":3.0}"#).is_err()
        );
        assert!(
            serde_json::from_str::<Mm1Queue>(r#"{"arrival_rate":-1.0,"service_rate":3.0}"#)
                .is_err()
        );

        let mmc = MmcQueue::new(5.0, 3.0, 2).unwrap();
        let mmc_json = serde_json::to_string(&mmc).unwrap();
        let mmc_restored: MmcQueue = serde_json::from_str(&mmc_json).unwrap();
        assert_eq!(mmc_restored, mmc);
        assert!(
            serde_json::from_str::<MmcQueue>(
                r#"{"arrival_rate":5.0,"service_rate":3.0,"servers":0}"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<MmcQueue>(
                r#"{"arrival_rate":7.0,"service_rate":3.0,"servers":2}"#
            )
            .is_err()
        );

        let metrics = mmc.metrics();
        let metrics_restored: QueueMetrics =
            serde_json::from_str(&serde_json::to_string(&metrics).unwrap()).unwrap();
        assert_eq!(metrics_restored, metrics);

        let err = Mm1Queue::new(4.0, 3.0).unwrap_err();
        let err_json = serde_json::to_string(&err).unwrap();
        assert!(err_json.contains("UnstableQueue"));
    }
}
