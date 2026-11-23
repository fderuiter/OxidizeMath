//! Validated `serde` serialization and deserialization for [`MarkovChain`].
// [cite:clinical_trials_statistics]

use super::{MarkovChain, StateType};
use crate::pure_math::statistics::markov::validation::{matrix_to_rows, rows_to_matrix};
use nalgebra::RealField;
use num_traits::ToPrimitive;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMarkovChain<T> {
    transition_matrix: Vec<Vec<T>>,
    state_types: Vec<StateType>,
}

impl<T> Serialize for MarkovChain<T>
where
    T: RealField + Copy + ToPrimitive + Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let rows = matrix_to_rows(self.transition_matrix());
        let mut state = serializer.serialize_struct("MarkovChain", 2)?;
        state.serialize_field("transition_matrix", &rows)?;
        state.serialize_field("state_types", &self.state_types())?;
        state.end()
    }
}

impl<'de, T> Deserialize<'de> for MarkovChain<T>
where
    T: RealField + Copy + ToPrimitive + Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawMarkovChain::<T>::deserialize(deserializer)?;
        let matrix = rows_to_matrix(raw.transition_matrix).map_err(serde::de::Error::custom)?;
        Self::new(matrix, raw.state_types).map_err(serde::de::Error::custom)
    }
}
