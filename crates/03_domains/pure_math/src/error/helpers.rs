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
