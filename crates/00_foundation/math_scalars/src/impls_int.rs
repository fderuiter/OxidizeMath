// [cite:math_commons]
use crate::traits::{Ring, Scalar};

macro_rules! impl_signed_int_ring {
    ($($t:ty),* $(,)?) => {
        $(
            impl Scalar for $t {}

            impl Ring for $t {
                #[inline(always)]
                fn zero() -> Self {
                    0
                }

                #[inline(always)]
                fn one() -> Self {
                    1
                }

                #[inline(always)]
                fn is_zero(&self) -> bool {
                    *self == 0
                }
            }
        )*
    };
}

impl_signed_int_ring!(i32, i64, i128, isize);
