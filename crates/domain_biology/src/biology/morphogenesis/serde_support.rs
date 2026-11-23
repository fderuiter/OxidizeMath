#![cfg_attr(any(), verified(opt_out = "serde serialization adapter"))]

use super::{ReactionKinetics, SpatialDiffusion, TuringSolverStrategy, TuringState, TuringSystem};

impl<const N: usize, K, D, S> serde::Serialize for TuringSystem<N, K, D, S>
where
    K: ReactionKinetics<N> + serde::Serialize,
    D: SpatialDiffusion<N> + serde::Serialize,
    S: TuringSolverStrategy<N> + serde::Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut st = serializer.serialize_struct("TuringSystem", 5)?;
        st.serialize_field("state", &self.state)?;
        st.serialize_field("diffusion_coeffs", &self.diffusion_coeffs.as_slice())?;
        st.serialize_field("kinetics", &self.kinetics)?;
        st.serialize_field("diffusion", &self.diffusion)?;
        st.serialize_field("solver", &self.solver)?;
        st.end()
    }
}

impl<'de, const N: usize, K, D, S> serde::Deserialize<'de> for TuringSystem<N, K, D, S>
where
    K: ReactionKinetics<N> + serde::Deserialize<'de>,
    D: SpatialDiffusion<N> + serde::Deserialize<'de>,
    S: TuringSolverStrategy<N> + serde::Deserialize<'de>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawTuringSystem<const N: usize, K, D, S> {
            state: TuringState<N>,
            diffusion_coeffs: Vec<f64>,
            kinetics: K,
            diffusion: D,
            solver: S,
        }
        let raw = RawTuringSystem::<N, K, D, S>::deserialize(deserializer)?;
        if raw.diffusion_coeffs.len() != N {
            return Err(serde::de::Error::custom(format!(
                "expected {N} diffusion coefficients, got {}",
                raw.diffusion_coeffs.len()
            )));
        }
        if raw
            .diffusion_coeffs
            .iter()
            .any(|c| !c.is_finite() || *c < 0.0)
        {
            return Err(serde::de::Error::custom(
                "diffusion coefficients must be finite and non-negative",
            ));
        }
        let diffusion_coeffs: [f64; N] = raw
            .diffusion_coeffs
            .try_into()
            .map_err(|_| serde::de::Error::custom("diffusion coefficients count mismatch"))?;
        let next_state = TuringState::new(raw.state.len());
        Ok(Self {
            state: raw.state,
            next_state,
            diffusion_coeffs,
            kinetics: raw.kinetics,
            diffusion: raw.diffusion,
            solver: raw.solver,
        })
    }
}
