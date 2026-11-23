use super::{
    ContinuousMarkovChain, HiddenMarkovModel, ParrondoParams, ParrondoSchedule, TimeIndex,
    TransitionTensor,
};
use nalgebra::{DMatrix, DVector, RealField};
use num_traits::ToPrimitive;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

impl<T> Serialize for ContinuousMarkovChain<T>
where
    T: RealField + Copy + ToPrimitive + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct CtmcRef<'a, U: RealField + Copy + ToPrimitive + Serialize> {
            generator: &'a DMatrix<U>,
        }
        CtmcRef {
            generator: self.generator(),
        }
        .serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for ContinuousMarkovChain<T>
where
    T: RealField + Copy + ToPrimitive + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(bound(deserialize = "U: RealField + Copy + ToPrimitive + Deserialize<'de>"))]
        struct CtmcRaw<U: RealField + Copy + ToPrimitive> {
            generator: DMatrix<U>,
        }
        let raw = CtmcRaw::<T>::deserialize(deserializer)?;
        ContinuousMarkovChain::new(raw.generator).map_err(serde::de::Error::custom)
    }
}

impl<T> Serialize for HiddenMarkovModel<T>
where
    T: RealField + Copy + ToPrimitive + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct HmmRef<'a, U: RealField + Copy + ToPrimitive + Serialize> {
            initial: &'a DVector<U>,
            transitions: &'a DMatrix<U>,
            emissions: &'a DMatrix<U>,
        }
        HmmRef {
            initial: self.initial(),
            transitions: self.transitions(),
            emissions: self.emissions(),
        }
        .serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for HiddenMarkovModel<T>
where
    T: RealField + Copy + ToPrimitive + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(bound(deserialize = "U: RealField + Copy + ToPrimitive + Deserialize<'de>"))]
        struct HmmRaw<U: RealField + Copy + ToPrimitive> {
            initial: DVector<U>,
            transitions: DMatrix<U>,
            emissions: DMatrix<U>,
        }
        let raw = HmmRaw::<T>::deserialize(deserializer)?;
        HiddenMarkovModel::new(raw.initial, raw.transitions, raw.emissions)
            .map_err(serde::de::Error::custom)
    }
}

impl<T> Serialize for TransitionTensor<T>
where
    T: RealField + Copy + ToPrimitive + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct TensorRef<'a, U: RealField + Copy + ToPrimitive + Serialize> {
            num_states: usize,
            min_time: TimeIndex<U>,
            max_time: TimeIndex<U>,
            time_slices: &'a [(TimeIndex<U>, DMatrix<U>)],
        }
        let (min_t, max_t) = self.time_range();
        let min_time = TimeIndex::new(min_t).map_err(serde::ser::Error::custom)?;
        let max_time = TimeIndex::new(max_t).map_err(serde::ser::Error::custom)?;
        TensorRef {
            num_states: self.num_states(),
            min_time,
            max_time,
            time_slices: &self.time_slices,
        }
        .serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for TransitionTensor<T>
where
    T: RealField + Copy + ToPrimitive + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(bound(deserialize = "U: RealField + Copy + ToPrimitive + Deserialize<'de>"))]
        struct TensorRaw<U: RealField + Copy + ToPrimitive> {
            num_states: usize,
            min_time: TimeIndex<U>,
            max_time: TimeIndex<U>,
            time_slices: Vec<(TimeIndex<U>, DMatrix<U>)>,
        }
        let raw = TensorRaw::<T>::deserialize(deserializer)?;
        if raw.min_time.value() > raw.max_time.value() {
            return Err(serde::de::Error::custom("min_time must be <= max_time"));
        }
        let mut tensor = TransitionTensor::new(raw.num_states, raw.min_time, raw.max_time);
        for (time, matrix) in raw.time_slices {
            tensor
                .add_time_slice(time, matrix)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(tensor)
    }
}

impl<'de> Deserialize<'de> for ParrondoParams {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ParrondoParamsRaw {
            modulus: usize,
            p_a_base: f64,
            p_b0_base: f64,
            p_b1_base: f64,
            epsilon: f64,
        }
        let raw = ParrondoParamsRaw::deserialize(deserializer)?;
        ParrondoParams::new(
            raw.modulus,
            raw.p_a_base,
            raw.p_b0_base,
            raw.p_b1_base,
            raw.epsilon,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl Serialize for ParrondoSchedule {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.as_string())
    }
}

impl<'de> Deserialize<'de> for ParrondoSchedule {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        ParrondoSchedule::parse(&raw).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        ContinuousMarkovChain, HiddenMarkovModel, MarkovChain, StateType, TimeIndex,
        TransitionTensor,
    };
    use nalgebra::{DMatrix, DVector};

    #[test]
    #[verified_engine::verified]
    fn test_markov_serde_roundtrip_and_validation() {
        let p = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.0, 1.0]);
        let chain = MarkovChain::new(p, vec![StateType::Transient, StateType::Absorbing]).unwrap();
        let json = serde_json::to_string(&chain).unwrap();
        let restored: MarkovChain<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.num_states(), 2);

        let bad_json = r#"{"transition_matrix":[0.5,0.2,0.0,1.0,2,2],"state_types":["Transient","Absorbing"]}"#;
        assert!(serde_json::from_str::<MarkovChain<f64>>(bad_json).is_err());

        let g = DMatrix::from_row_slice(2, 2, &[-2.0, 2.0, 3.0, -3.0]);
        let ctmc = ContinuousMarkovChain::new(g).unwrap();
        let ctmc_json = serde_json::to_string(&ctmc).unwrap();
        let ctmc_restored: ContinuousMarkovChain<f64> = serde_json::from_str(&ctmc_json).unwrap();
        assert_eq!(ctmc_restored.num_states(), 2);

        let hmm = HiddenMarkovModel::new(
            DVector::from_vec(vec![0.5, 0.5]),
            DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]),
            DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.1, 0.9]),
        )
        .unwrap();
        let hmm_json = serde_json::to_string(&hmm).unwrap();
        let hmm_restored: HiddenMarkovModel<f64> = serde_json::from_str(&hmm_json).unwrap();
        assert_eq!(hmm_restored.num_states(), 2);

        let mut tensor = TransitionTensor::new(
            2,
            TimeIndex::new(0.0).unwrap(),
            TimeIndex::new(24.0).unwrap(),
        );
        tensor
            .add_time_slice(
                TimeIndex::new(10.0).unwrap(),
                DMatrix::from_row_slice(2, 2, &[0.6, 0.4, 0.2, 0.8]),
            )
            .unwrap();
        let tensor_json = serde_json::to_string(&tensor).unwrap();
        let tensor_restored: TransitionTensor<f64> = serde_json::from_str(&tensor_json).unwrap();
        assert_eq!(tensor_restored.num_time_slices(), 1);
    }

    #[test]
    #[verified_engine::verified]
    fn test_parrondo_serde_roundtrip_and_validation() {
        use super::super::{ParrondoParams, ParrondoSchedule};

        let params = ParrondoParams::canonical();
        let json = serde_json::to_string(&params).unwrap();
        let restored: ParrondoParams = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, params);
        let bad_params =
            r#"{"modulus":4,"p_a_base":0.5,"p_b0_base":0.1,"p_b1_base":0.75,"epsilon":0.005}"#;
        assert!(serde_json::from_str::<ParrondoParams>(bad_params).is_err());

        let sched = ParrondoSchedule::parse("ABBB").unwrap();
        let sched_json = serde_json::to_string(&sched).unwrap();
        assert_eq!(sched_json, r#""ABBB""#);
        let sched_restored: ParrondoSchedule = serde_json::from_str(&sched_json).unwrap();
        assert_eq!(sched_restored, sched);
        assert!(serde_json::from_str::<ParrondoSchedule>(r#""ABC""#).is_err());
    }
}
