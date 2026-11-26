//! Validated `serde` serialization and deserialization for [`ContinuousMarkovChain`].
// [cite:clinical_trials_statistics]

use super::ContinuousMarkovChain;
use crate::pure_math::statistics::markov::validation::{matrix_to_rows, rows_to_matrix};
use nalgebra::RealField;
use num_traits::ToPrimitive;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawContinuousMarkovChain<T> {
    generator: Vec<Vec<T>>,
}

impl<T> Serialize for ContinuousMarkovChain<T>
where
    T: RealField + Copy + ToPrimitive + Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let rows = matrix_to_rows(self.generator());
        let mut state = serializer.serialize_struct("ContinuousMarkovChain", 1)?;
        state.serialize_field("generator", &rows)?;
        state.end()
    }
}

impl<'de, T> Deserialize<'de> for ContinuousMarkovChain<T>
where
    T: RealField + Copy + ToPrimitive + Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawContinuousMarkovChain::<T>::deserialize(deserializer)?;
        let generator = rows_to_matrix(raw.generator).map_err(serde::de::Error::custom)?;
        Self::new(generator).map_err(serde::de::Error::custom)
    }
}
