use super::AdamState;
use nalgebra::Scalar;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::hash::Hash;

pub(super) fn serialize<S, K, T>(
    map: &HashMap<K, AdamState<T>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    K: Serialize,
    T: Scalar + Serialize,
{
    let entries: Vec<(&K, &AdamState<T>)> = map.iter().collect();
    entries.serialize(serializer)
}

pub(super) fn deserialize<'de, D, K, T>(
    deserializer: D,
) -> Result<HashMap<K, AdamState<T>>, D::Error>
where
    D: Deserializer<'de>,
    K: Deserialize<'de> + Eq + Hash,
    T: Scalar + Deserialize<'de>,
{
    let entries = Vec::<(K, AdamState<T>)>::deserialize(deserializer)?;
    Ok(entries.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::super::{Adam, AdamState2D, Optimizer, ParamType};
    use nalgebra::{DMatrix, Vector2};

    #[test]
    #[verified_engine::verified]
    fn test_adam_and_adam_state_2d_serde_roundtrip() {
        let mut adam = Adam::<f64, (usize, ParamType)>::new(0.01).expect("adam");
        let mut param = DMatrix::from_element(2, 1, 1.0_f64);
        let grad = DMatrix::from_element(2, 1, 0.25_f64);
        adam.update_matrix((0, ParamType::Weight), &mut param, &grad)
            .expect("update");
        let json = serde_json::to_string(&adam).expect("serialize adam");
        let restored: Adam<f64, (usize, ParamType)> =
            serde_json::from_str(&json).expect("deserialize adam");
        assert_eq!(restored.states.len(), 1);

        let mut s2d = AdamState2D::<f64>::new();
        s2d.t = 3;
        s2d.m = Vector2::new(0.1, -0.2);
        let s2d_json = serde_json::to_string(&s2d).expect("serialize state2d");
        let s2d_restored: AdamState2D<f64> =
            serde_json::from_str(&s2d_json).expect("deserialize state2d");
        assert_eq!(s2d_restored, s2d);
    }
}
