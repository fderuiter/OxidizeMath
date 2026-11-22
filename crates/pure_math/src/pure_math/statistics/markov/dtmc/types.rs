/// Classification of a state in a Markov chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StateType {
    /// Transient state (can leave and never return).
    Transient,
    /// Absorbing state (once entered, cannot leave).
    Absorbing,
}

#[cfg(feature = "serde")]
mod dtmc_serde {
    use super::StateType;
    use crate::pure_math::statistics::markov::dtmc::MarkovChain;
    use nalgebra::{DMatrix, RealField};
    use num_traits::ToPrimitive;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    #[serde(bound(
        serialize = "T: RealField + Copy + ToPrimitive + Serialize",
        deserialize = "T: RealField + Copy + ToPrimitive + Deserialize<'de>"
    ))]
    struct MarkovChainRaw<T: RealField + Copy + ToPrimitive> {
        transition_matrix: DMatrix<T>,
        state_types: Vec<StateType>,
    }

    impl<T> Serialize for MarkovChain<T>
    where
        T: RealField + Copy + ToPrimitive + Serialize,
    {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            #[derive(Serialize)]
            struct MarkovChainRef<'a, U: RealField + Copy + ToPrimitive + Serialize> {
                transition_matrix: &'a DMatrix<U>,
                state_types: &'a [StateType],
            }
            MarkovChainRef {
                transition_matrix: self.transition_matrix(),
                state_types: self.state_types(),
            }
            .serialize(serializer)
        }
    }

    impl<'de, T> Deserialize<'de> for MarkovChain<T>
    where
        T: RealField + Copy + ToPrimitive + Deserialize<'de>,
    {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let raw = MarkovChainRaw::<T>::deserialize(deserializer)?;
            MarkovChain::new(raw.transition_matrix, raw.state_types)
                .map_err(serde::de::Error::custom)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StateType;

    #[test]
    #[verified_engine::verified]
    fn test_state_type_variants() {
        assert_ne!(StateType::Transient, StateType::Absorbing);
    }

    #[cfg(feature = "serde")]
    #[test]
    #[verified_engine::verified]
    fn test_markov_chain_serde_validation() {
        use crate::pure_math::statistics::markov::dtmc::MarkovChain;
        use nalgebra::DMatrix;

        let mat = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.0, 1.0]);
        let states = vec![StateType::Transient, StateType::Absorbing];
        let chain = MarkovChain::<f64>::new(mat, states.clone()).unwrap();
        let json = serde_json::to_string(&chain).unwrap();
        let decoded: MarkovChain<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.state_types(), states.as_slice());

        let bad_json = json.replace("0.7", "0.9");
        assert!(serde_json::from_str::<MarkovChain<f64>>(&bad_json).is_err());
    }
}
