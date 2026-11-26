/// Declarative verification macro `theory_verification`.
#[macro_export]
macro_rules! theory_verification {
    (
        module = $module_name:expr,
        constants = {
            $( $const_name:ident = $const_val:expr; )*
        },
        test = $test_body:block
    ) => {
        #[cfg(test)]
        #[allow(unused_imports)]
        mod theory_verification {
            use super::*;

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

/// Tier 1 (Manufactured-Solution & Grid-Refinement Convergence) verification registration macro.
#[macro_export]
macro_rules! mms_convergence_verification {
    (module = $module_name:expr, test = $test_body:block) => {
        #[cfg(test)]
        #[allow(unused_imports)]
        mod mms_convergence_verification {
            use super::*;

            #[test]
            #[$crate::__macro_deps::verified_engine::verified]
            fn test_mms_convergence_verification() {
                assert!(
                    $crate::__macro_deps::oxidize_core::traceability::TraceabilityEngine::<
                        $crate::__macro_deps::oxidize_core::vfs::DefaultVfs,
                    >::verify_module_registered($module_name),
                    "Traceability mismatch: module '{}' is not registered",
                    $module_name
                );
                $test_body
            }
        }
    };
}

/// Tier 2 (Deterministic Property & Conservation Invariant) verification registration macro.
#[macro_export]
macro_rules! property_invariant_verification {
    (module = $module_name:expr, test = $test_body:block) => {
        #[cfg(test)]
        #[allow(unused_imports)]
        mod property_invariant_verification {
            use super::*;

            #[test]
            #[$crate::__macro_deps::verified_engine::verified]
            fn test_property_invariant_verification() {
                assert!(
                    $crate::__macro_deps::oxidize_core::traceability::TraceabilityEngine::<
                        $crate::__macro_deps::oxidize_core::vfs::DefaultVfs,
                    >::verify_module_registered($module_name),
                    "Traceability mismatch: module '{}' is not registered",
                    $module_name
                );
                $test_body
            }
        }
    };
}

/// Tier 3 (High-Precision Offline Golden Fixture) verification registration macro.
#[macro_export]
macro_rules! golden_fixture_verification {
    (module = $module_name:expr, test = $test_body:block) => {
        #[cfg(test)]
        #[allow(unused_imports)]
        mod golden_fixture_verification {
            use super::*;

            #[test]
            #[$crate::__macro_deps::verified_engine::verified]
            fn test_golden_fixture_verification() {
                assert!(
                    $crate::__macro_deps::oxidize_core::traceability::TraceabilityEngine::<
                        $crate::__macro_deps::oxidize_core::vfs::DefaultVfs,
                    >::verify_module_registered($module_name),
                    "Traceability mismatch: module '{}' is not registered",
                    $module_name
                );
                $test_body
            }
        }
    };
}

/// Tier 5 (IEEE-754 Outward-Rounded Interval Enclosure) verification registration macro.
#[macro_export]
macro_rules! interval_enclosure_verification {
    (module = $module_name:expr, test = $test_body:block) => {
        #[cfg(test)]
        #[allow(unused_imports)]
        mod interval_enclosure_verification {
            use super::*;

            #[test]
            #[$crate::__macro_deps::verified_engine::verified]
            fn test_interval_enclosure_verification() {
                assert!(
                    $crate::__macro_deps::oxidize_core::traceability::TraceabilityEngine::<
                        $crate::__macro_deps::oxidize_core::vfs::DefaultVfs,
                    >::verify_module_registered($module_name),
                    "Traceability mismatch: module '{}' is not registered",
                    $module_name
                );
                $test_body
            }
        }
    };
}

/// Tier 6 (Lean 4 Mechanized Proof Certificate) verification registration macro.
#[macro_export]
macro_rules! formal_proof_verification {
    (module = $module_name:expr, proof = $proof_file:literal, test = $test_body:block) => {
        #[cfg(test)]
        #[allow(unused_imports)]
        mod formal_proof_verification {
            use super::*;

            #[test]
            #[$crate::__macro_deps::verified_engine::verified]
            fn test_formal_proof_verification() {
                assert!(!$proof_file.is_empty());
                assert!(
                    $crate::__macro_deps::oxidize_core::traceability::TraceabilityEngine::<
                        $crate::__macro_deps::oxidize_core::vfs::DefaultVfs,
                    >::verify_module_registered($module_name),
                    "Traceability mismatch: module '{}' is not registered",
                    $module_name
                );
                $test_body
            }
        }
    };
}

/// Declarative verification macro `stochastic_signature_verification`.
#[macro_export]
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
        #[allow(unused_imports)]
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
                    assert!(($min..=$max).contains(&$stat_name),
                        "Statistical bound failure for {}: value {} not in [{}, {}]",
                        stringify!($stat_name), $stat_name, $min, $max
                    );
                )*
            }
        }
    };
}
