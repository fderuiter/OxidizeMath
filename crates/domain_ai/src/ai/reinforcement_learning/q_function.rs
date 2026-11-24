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

/// Zero-copy discretized state index for fixed-capacity tabular reinforcement learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiscreteState(pub usize);

impl State for DiscreteState {}

/// Zero-copy discretized action index for fixed-capacity tabular reinforcement learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiscreteAction(pub usize);

impl Action for DiscreteAction {}

/// Preallocated fixed-array `QFunction` over `STATES` discrete states and `ACTIONS` discrete actions.
///
/// Unlike [`TabularQFunction`], lookups and Bellman updates index directly into a contiguous
/// `[[f64; ACTIONS]; STATES]` array with zero dynamic heap allocations.
#[derive(Debug, Clone)]
pub struct FixedArrayQFunction<const STATES: usize, const ACTIONS: usize> {
    table: [[f64; ACTIONS]; STATES],
}

impl<const STATES: usize, const ACTIONS: usize> Default for FixedArrayQFunction<STATES, ACTIONS> {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

impl<const STATES: usize, const ACTIONS: usize> FixedArrayQFunction<STATES, ACTIONS> {
    /// Creates a zero-initialized fixed-capacity Q-table.
    #[must_use]
    #[verified_engine::verified]
    pub const fn new() -> Self {
        Self {
            table: [[0.0; ACTIONS]; STATES],
        }
    }

    /// Resets all stored state-action Q-values to `0.0` in place.
    #[verified_engine::verified]
    pub fn reset(&mut self) {
        self.table = [[0.0; ACTIONS]; STATES];
    }

    /// Borrows the raw `[STATES][ACTIONS]` Q-value table.
    #[must_use]
    #[verified_engine::verified]
    pub const fn as_table(&self) -> &[[f64; ACTIONS]; STATES] {
        &self.table
    }

    /// Overwrites the Q-value table from a snapshot array in place.
    #[verified_engine::verified]
    pub fn copy_from_table(&mut self, src: &[[f64; ACTIONS]; STATES]) {
        self.table = *src;
    }
}

impl<const STATES: usize, const ACTIONS: usize> QFunction<DiscreteState, DiscreteAction>
    for FixedArrayQFunction<STATES, ACTIONS>
{
    #[verified_engine::verified]
    fn value(&self, state: &DiscreteState, action: &DiscreteAction) -> f64 {
        self.table
            .get(state.0)
            .and_then(|row| row.get(action.0))
            .copied()
            .unwrap_or(0.0)
    }

    #[verified_engine::verified]
    fn update(&mut self, state: &DiscreteState, action: &DiscreteAction, value: f64) {
        if let Some(slot) = self
            .table
            .get_mut(state.0)
            .and_then(|row| row.get_mut(action.0))
        {
            *slot = value;
        }
    }
}
