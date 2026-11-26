/// Macro to implement common arithmetic operations for compartmental states.
#[macro_export]
macro_rules! impl_compartmental_ops {
    ($type:ty, $($field:ident),+) => {
        impl std::ops::Add for $type {
            type Output = Self;
            #[verified_engine::verified]
            fn add(self, rhs: Self) -> Self {
                Self {
                    $($field: self.$field + rhs.$field),+
                }
            }
        }

        impl std::ops::AddAssign for $type {
            #[verified_engine::verified]
            fn add_assign(&mut self, rhs: Self) {
                $(self.$field += rhs.$field;)+
            }
        }

        impl std::ops::Mul<f64> for $type {
            type Output = Self;
            #[verified_engine::verified]
            fn mul(self, scalar: f64) -> Self {
                Self {
                    $($field: self.$field * scalar),+
                }
            }
        }

        impl std::ops::MulAssign<f64> for $type {
            #[verified_engine::verified]
            fn mul_assign(&mut self, scalar: f64) {
                $(self.$field *= scalar;)+
            }
        }

        impl pure_math::pure_math::analysis::ode::VectorOperations for $type {
            #[verified_engine::verified]
            fn scale_add(&mut self, other: &Self, scale: f64) {
                $(self.$field += other.$field * scale;)+
            }

            #[verified_engine::verified]
            fn copy_from(&mut self, other: &Self) {
                *self = *other;
            }
        }
    };
}
