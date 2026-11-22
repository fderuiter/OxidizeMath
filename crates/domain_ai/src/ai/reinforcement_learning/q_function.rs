use super::types::{Action, QFunction, State};
use std::collections::HashMap;
use std::hash::Hash;

/// A Tabular implementation of the Q-Function using a HashMap.
///
/// Stores Q-values for discrete (State, Action) pairs.
/// The default Q-value for unseen pairs is 0.0.
#[derive(Debug, Clone, PartialEq)]
pub struct TabularQFunction<S, A>
where
    S: State + Hash + Eq,
    A: Action + Hash + Eq,
{
    table: HashMap<(S, A), f64>,
}

impl<S, A> Default for TabularQFunction<S, A>
where
    S: State + Hash + Eq,
    A: Action + Hash + Eq,
{
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

impl<S, A> TabularQFunction<S, A>
where
    S: State + Hash + Eq,
    A: Action + Hash + Eq,
{
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }
}

impl<S, A> QFunction<S, A> for TabularQFunction<S, A>
where
    S: State + Hash + Eq,
    A: Action + Hash + Eq,
{
    #[verified_engine::verified]
    fn value(&self, state: &S, action: &A) -> f64 {
        // We must clone the key to search if we only have references,
        // or we need to borrow properly. HashMap `get` takes `&K`.
        // Since `K` is `(S, A)`, we can construct a reference if we could,
        // but `(S, A)` is a tuple.
        // It's easier to clone the key for lookup if S and A are cheap (Copy).
        // If not, we might incur overhead.
        // For TabularQFunction, we assume S and A are relatively small keys.
        *self
            .table
            .get(&(state.clone(), action.clone()))
            .unwrap_or(&0.0)
    }

    #[verified_engine::verified]
    fn update(&mut self, state: &S, action: &A, value: f64) {
        self.table.insert((state.clone(), action.clone()), value);
    }
}

#[cfg(feature = "serde")]
impl<S, A> serde::Serialize for TabularQFunction<S, A>
where
    S: State + Hash + Eq + serde::Serialize,
    A: Action + Hash + Eq + serde::Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(self.table.len()))?;
        for ((state, action), value) in &self.table {
            seq.serialize_element(&(state, action, value))?;
        }
        seq.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, S, A> serde::Deserialize<'de> for TabularQFunction<S, A>
where
    S: State + Hash + Eq + serde::Deserialize<'de>,
    A: Action + Hash + Eq + serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let entries = Vec::<(S, A, f64)>::deserialize(deserializer)?;
        let mut table = HashMap::with_capacity(entries.len());
        for (state, action, value) in entries {
            if !value.is_finite() {
                return Err(serde::de::Error::custom("Q-table values must be finite"));
            }
            table.insert((state, action), value);
        }
        Ok(Self { table })
    }
}
