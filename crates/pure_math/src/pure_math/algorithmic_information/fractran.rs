use crate::math_types::Rational;

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct FractranProgram {
    #[allow(missing_docs)]
    pub fractions: Vec<Rational>,
}

impl FractranProgram {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(fractions: Vec<Rational>) -> Self {
        Self { fractions }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct FractranState {
    #[allow(missing_docs)]
    pub state_primes: Vec<u64>,
    #[allow(missing_docs)]
    pub register_primes: Vec<u64>,
}
