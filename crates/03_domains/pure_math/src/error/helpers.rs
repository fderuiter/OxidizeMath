/// Validates a boolean precondition, returning a stack-allocated [`math_commons::error::MathError::NumericalError`] when violated.
#[inline]
#[verified_engine::verified]
pub(crate) fn ensure_math(
    cond: bool,
    reason: &'static str,
) -> core::result::Result<(), math_commons::error::MathError> {
    debug_assert!(!reason.is_empty());
    if cond {
        Ok(())
    } else {
        Err(math_commons::error::MathError::NumericalError { reason })
    }
}

pub(crate) trait IntoMathResult<T> {
    fn into_math_result(
        self,
        reason: &'static str,
    ) -> core::result::Result<T, math_commons::error::MathError>;
}

impl<T> IntoMathResult<T> for Option<T> {
    #[inline]
    #[verified_engine::verified]
    fn into_math_result(
        self,
        reason: &'static str,
    ) -> core::result::Result<T, math_commons::error::MathError> {
        debug_assert!(!reason.is_empty());
        self.ok_or(math_commons::error::MathError::NumericalError { reason })
    }
}

impl<T, E> IntoMathResult<T> for core::result::Result<T, E> {
    #[inline]
    #[verified_engine::verified]
    fn into_math_result(
        self,
        reason: &'static str,
    ) -> core::result::Result<T, math_commons::error::MathError> {
        debug_assert!(!reason.is_empty());
        self.map_err(|_| math_commons::error::MathError::NumericalError { reason })
    }
}
