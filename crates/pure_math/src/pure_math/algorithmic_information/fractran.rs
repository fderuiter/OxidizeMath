use crate::math_types::Rational;

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
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

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub struct FractranState {
    #[allow(missing_docs)]
    pub state_primes: Vec<u64>,
    #[allow(missing_docs)]
    pub register_primes: Vec<u64>,
}
