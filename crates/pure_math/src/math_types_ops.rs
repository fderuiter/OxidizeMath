//! Power and Euclidean remainder traits for [`super::Integer`].

/// Exponentiation trait for arbitrary-precision [`super::Integer`] values.
pub trait Pow<Rhs> {
    /// Result type of raising `self` to `rhs`.
    type Output;
    /// Raises `self` to `rhs`.
    fn pow(self, rhs: Rhs) -> Self::Output;
}

impl Pow<u32> for super::Integer {
    type Output = super::Integer;
    #[verified_engine::verified]
    fn pow(self, rhs: u32) -> Self::Output {
        super::Integer(self.0.pow(rhs as usize))
    }
}

/// Non-negative Euclidean remainder trait for arbitrary-precision [`super::Integer`] values.
pub trait RemRounding<Rhs> {
    /// Remainder output type.
    type Output;
    /// Computes the non-negative Euclidean remainder of `self` modulo `rhs`.
    fn rem_euc(self, rhs: Rhs) -> Self::Output;
}

impl RemRounding<super::Integer> for super::Integer {
    type Output = super::Integer;
    #[verified_engine::verified]
    fn rem_euc(self, rhs: super::Integer) -> Self::Output {
        let mut r = &self.0 % &rhs.0;
        if r < dashu::integer::IBig::from(0u8) {
            use dashu::base::Abs;
            r += rhs.0.abs();
        }
        assert!(r >= dashu::integer::IBig::from(0u8));
        super::Integer(r)
    }
}

impl RemRounding<&super::Integer> for super::Integer {
    type Output = super::Integer;
    #[verified_engine::verified]
    fn rem_euc(self, rhs: &super::Integer) -> Self::Output {
        let mut r = &self.0 % &rhs.0;
        if r < dashu::integer::IBig::from(0u8) {
            use dashu::base::Abs;
            r += rhs.0.clone().abs();
        }
        assert!(r >= dashu::integer::IBig::from(0u8));
        super::Integer(r)
    }
}
