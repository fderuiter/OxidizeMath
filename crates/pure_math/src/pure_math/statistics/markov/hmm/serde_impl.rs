//! Validated `serde` serialization and deserialization for [`HiddenMarkovModel`].
// [cite:clinical_trials_statistics]

use super::model::HiddenMarkovModel;
use crate::error::MarkovError;
use crate::pure_math::statistics::markov::validation::{matrix_to_rows, rows_to_matrix};
use nalgebra::{DVector, RealField};
use num_traits::ToPrimitive;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHiddenMarkovModel<T> {
    initial: Vec<T>,
    transitions: Vec<Vec<T>>,
    emissions: Vec<Vec<T>>,
}

impl<T> Serialize for HiddenMarkovModel<T>
where
    T: RealField + Copy + ToPrimitive + Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let initial: Vec<T> = self.initial.iter().copied().collect();
        let transitions = matrix_to_rows(&self.transitions);
        let emissions = matrix_to_rows(&self.emissions);
        let mut state = serializer.serialize_struct("HiddenMarkovModel", 3)?;
        state.serialize_field("initial", &initial)?;
        state.serialize_field("transitions", &transitions)?;
        state.serialize_field("emissions", &emissions)?;
        state.end()
    }
}

impl<'de, T> Deserialize<'de> for HiddenMarkovModel<T>
where
    T: RealField + Copy + ToPrimitive + Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawHiddenMarkovModel::<T>::deserialize(deserializer)?;
        if raw.initial.is_empty() {
            return Err(serde::de::Error::custom(MarkovError::NotStochastic {
                reason: "Initial probability vector cannot be empty".to_string(),
            }));
        }
        let initial = DVector::from_vec(raw.initial);
        let transitions = rows_to_matrix(raw.transitions).map_err(serde::de::Error::custom)?;
        let emissions = rows_to_matrix(raw.emissions).map_err(serde::de::Error::custom)?;
        Self::new(initial, transitions, emissions).map_err(serde::de::Error::custom)
    }
}
