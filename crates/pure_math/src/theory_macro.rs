#[macro_export]
#[allow(missing_docs)]
macro_rules! theory_verification {
    (
        module = $module_name:expr,
        constants = {
            $( $const_name:ident = $const_val:expr; )*
        },
        test = $test_body:block
    ) => {
        #[cfg(test)]
        #[allow(unused_imports, clippy::all)]
        mod theory_verification {
            use super::*;
            use approx::assert_relative_eq;

            $(
                const $const_name: f64 = $const_val;
            )*

            #[test]
            #[$crate::__macro_deps::verified_engine::verified]
            fn test_theory_verification() {
                // Check that the module is registered in the shared engine registry
                if !$crate::__macro_deps::oxidize_core::traceability::TraceabilityEngine::<$crate::__macro_deps::oxidize_core::vfs::DefaultVfs>::verify_module_registered($module_name) {
                    panic!("Traceability mismatch: module '{}' is not registered to any paper", $module_name);
                }

                $test_body
            }
        }
    };
}

#[macro_export]
#[allow(missing_docs)]
macro_rules! stochastic_signature_verification {
    (
        module = $module_name:expr,
        external_dependencies = [ $( $dep:literal ),* ],
        signatures = {
            $( $layer_name:ident => $expected_shape:expr; )*
        },
        statistical_bounds = {
            $( $stat_name:ident in [$min:expr, $max:expr]; )*
        },
        stochastic_test = { $( $stmt:stmt )* }
    ) => {
        #[cfg(test)]
        #[allow(unused_imports, clippy::all)]
        mod stochastic_verification {
            use super::*;

            #[test]
            #[$crate::__macro_deps::verified_engine::verified]
            fn test_stochastic_verification() {
                // 1. Check that the module is registered in the shared engine registry
                if !$crate::__macro_deps::oxidize_core::traceability::TraceabilityEngine::<$crate::__macro_deps::oxidize_core::vfs::DefaultVfs>::verify_module_registered($module_name) {
                    panic!("Traceability mismatch: module '{}' is not registered to any paper", $module_name);
                }

                // 2. Register/track dependencies to external C++ libraries
                $(
                    $crate::__macro_deps::verified_engine::engine::register_cpp_dependency($dep);
                )*

                // 3. Run the stochastic tests to generate variables
                $( $stmt )*

                // 4. Evaluate architecture signatures
                $(
                    let actual_shape = $layer_name.shape();
                    let expected = $expected_shape;
                    assert_eq!(actual_shape, expected, "Signature mismatch for {}", stringify!($layer_name));
                )*

                // 5. Evaluate statistical bounds
                $(
                    assert!($stat_name >= $min && $stat_name <= $max,
                        "Statistical bound failure for {}: value {} not in [{}, {}]",
                        stringify!($stat_name), $stat_name, $min, $max
                    );
                )*
            }
        }
    };
}
