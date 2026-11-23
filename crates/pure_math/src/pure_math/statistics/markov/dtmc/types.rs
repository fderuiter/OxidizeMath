/// Classification of a state in a Markov chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StateType {
    /// Transient state (can leave and never return).
    Transient,
    /// Absorbing state (once entered, cannot leave).
    Absorbing,
}

#[cfg(test)]
mod tests {
    use super::StateType;

    #[test]
    #[verified_engine::verified]
    fn test_state_type_variants() {
        assert_ne!(StateType::Transient, StateType::Absorbing);
    }
}

// [cite:clinical_trials_statistics]
