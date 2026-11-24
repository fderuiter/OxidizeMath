// [cite:biology]

use super::{GrowthParams, KernelShells, MAX_KERNEL_RADIUS, MAX_KERNEL_RINGS, MIN_KERNEL_RADIUS};

impl<'de> serde::Deserialize<'de> for GrowthParams {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawGrowth {
            mu: f64,
            sigma: f64,
        }
        let raw = RawGrowth::deserialize(deserializer)?;
        if !raw.mu.is_finite() || !(0.001..=1.0).contains(&raw.mu) {
            return Err(serde::de::Error::custom(format!(
                "GrowthParams mu={} must be finite in 0.001..=1.0",
                raw.mu
            )));
        }
        if !raw.sigma.is_finite() || !(1e-4..=0.5).contains(&raw.sigma) {
            return Err(serde::de::Error::custom(format!(
                "GrowthParams sigma={} must be finite in 1e-4..=0.5",
                raw.sigma
            )));
        }
        Ok(Self {
            mu: raw.mu,
            sigma: raw.sigma,
        })
    }
}

impl serde::Serialize for KernelShells {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(serde::Serialize)]
        struct RawShells<'a> {
            radius: f64,
            ring_weights: &'a [f64],
        }
        RawShells {
            radius: self.radius,
            ring_weights: self.ring_weights(),
        }
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for KernelShells {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawShells {
            radius: f64,
            ring_weights: Vec<f64>,
        }
        let raw = RawShells::deserialize(deserializer)?;
        if !raw.radius.is_finite() || !(MIN_KERNEL_RADIUS..=MAX_KERNEL_RADIUS).contains(&raw.radius)
        {
            return Err(serde::de::Error::custom(format!(
                "KernelShells radius={} must be finite in {MIN_KERNEL_RADIUS}..={MAX_KERNEL_RADIUS}",
                raw.radius
            )));
        }
        if !(1..=MAX_KERNEL_RINGS).contains(&raw.ring_weights.len()) {
            return Err(serde::de::Error::custom(format!(
                "KernelShells ring_weights length={} must be in 1..={MAX_KERNEL_RINGS}",
                raw.ring_weights.len()
            )));
        }
        if raw
            .ring_weights
            .iter()
            .any(|w| !w.is_finite() || !(0.0..=1.0).contains(w))
        {
            return Err(serde::de::Error::custom(
                "KernelShells ring_weights entries must be finite in 0.0..=1.0",
            ));
        }
        if raw.ring_weights.iter().sum::<f64>() <= 1e-12 {
            return Err(serde::de::Error::custom(
                "KernelShells ring_weights must have a strictly positive sum",
            ));
        }
        Ok(Self::new(raw.radius, &raw.ring_weights))
    }
}
