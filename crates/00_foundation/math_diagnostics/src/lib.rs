#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]

use core::fmt;

/// Operational severity level attached to structured diagnostics and telemetry events.
///
/// # Example
/// ```rust
/// use math_diagnostics::Severity;
///
/// assert_eq!(Severity::Warning.to_string(), "WARNING");
/// assert_eq!(Severity::Fatal.to_string(), "FATAL");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Severity {
    /// Routine progress or non-actionable diagnostic note.
    Info,
    /// Recoverable numerical degradation (e.g., adaptive step clamping or fallback).
    Warning,
    /// Domain validation failure or non-convergent solver step.
    Error,
    /// Unrecoverable invariant breach or caught thread panic.
    Fatal,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => f.write_str("INFO"),
            Self::Warning => f.write_str("WARNING"),
            Self::Error => f.write_str("ERROR"),
            Self::Fatal => f.write_str("FATAL"),
        }
    }
}

/// Zero-allocation stack visitor receiving structured key-value fields from a [`Diagnostic`].
pub trait DiagnosticVisitor {
    /// Records a borrowed string attribute under `key`.
    fn record_str(&mut self, key: &str, value: &str);

    /// Records a [`fmt::Display`] value under `key` without requiring heap allocation.
    fn record_display(&mut self, key: &str, value: &dyn fmt::Display);

    /// Records an unsigned pointer-sized integer attribute under `key`.
    fn record_usize(&mut self, key: &str, value: usize) {
        self.record_display(key, &value);
    }

    /// Records a signed pointer-sized integer attribute under `key`.
    fn record_isize(&mut self, key: &str, value: isize) {
        self.record_display(key, &value);
    }

    /// Records an unsigned 32-bit integer attribute under `key`.
    fn record_u32(&mut self, key: &str, value: u32) {
        self.record_display(key, &value);
    }

    /// Records a signed 32-bit integer attribute under `key`.
    fn record_i32(&mut self, key: &str, value: i32) {
        self.record_display(key, &value);
    }

    /// Records an unsigned 64-bit integer attribute under `key`.
    fn record_u64(&mut self, key: &str, value: u64) {
        self.record_display(key, &value);
    }

    /// Records a signed 64-bit integer attribute under `key`.
    fn record_i64(&mut self, key: &str, value: i64) {
        self.record_display(key, &value);
    }

    /// Records a 32-bit floating-point attribute under `key`.
    fn record_f32(&mut self, key: &str, value: f32) {
        self.record_display(key, &value);
    }

    /// Records a 64-bit floating-point attribute under `key`.
    fn record_f64(&mut self, key: &str, value: f64) {
        self.record_display(key, &value);
    }

    /// Records a boolean attribute under `key`.
    fn record_bool(&mut self, key: &str, value: bool) {
        self.record_display(key, &value);
    }
}

#[cfg(feature = "std")]
struct HashMapVisitor {
    map: std::collections::HashMap<std::string::String, std::string::String>,
}

#[cfg(feature = "std")]
impl HashMapVisitor {
    fn insert_field(&mut self, key: &str, value: std::string::String) {
        use std::string::String;
        match key {
            "code" => {
                if self.map.contains_key("code") {
                    self.map.insert(String::from("wrapper_code"), value);
                } else {
                    self.map.insert(String::from("code"), value);
                }
            }
            "error_type" => {
                if let Some(prev) = self.map.insert(String::from("error_type"), value) {
                    self.map
                        .entry(String::from("root_error_type"))
                        .or_insert(prev);
                }
            }
            "description" => {
                if let Some(prev) = self.map.insert(String::from("description"), value) {
                    self.map
                        .entry(String::from("root_description"))
                        .or_insert(prev);
                }
            }
            _ => {
                self.map.insert(String::from(key), value);
            }
        }
    }
}

#[cfg(feature = "std")]
impl DiagnosticVisitor for HashMapVisitor {
    fn record_str(&mut self, key: &str, value: &str) {
        self.insert_field(key, std::string::ToString::to_string(value));
    }

    fn record_display(&mut self, key: &str, value: &dyn fmt::Display) {
        self.insert_field(key, std::string::ToString::to_string(value));
    }
}

/// Materializes visited [`Diagnostic`] fields into an owned [`HashMap`](std::collections::HashMap).
#[cfg(feature = "std")]
pub fn collect_diagnostic_metadata<D: Diagnostic + ?Sized>(
    diagnostic: &D,
) -> std::collections::HashMap<std::string::String, std::string::String> {
    let mut visitor = HashMapVisitor {
        map: std::collections::HashMap::new(),
    };
    diagnostic.visit_fields(&mut visitor);
    visitor.map
}

/// Zero-allocation structured diagnostic contract for `#![no_std]` and `std` error types.
pub trait Diagnostic: core::error::Error + Send + Sync + 'static {
    /// Returns the operational [`Severity`] associated with this error (defaults to [`Severity::Error`]).
    fn severity(&self) -> Severity {
        Severity::Error
    }

    /// Emits structured key-value diagnostic fields into `visitor` on the stack.
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        let _ = visitor;
    }

    /// Returns structured key-value diagnostic attributes materialized into an owned `HashMap`.
    #[cfg(feature = "std")]
    fn metadata(&self) -> std::collections::HashMap<std::string::String, std::string::String> {
        collect_diagnostic_metadata(self)
    }
}

#[cfg(all(test, feature = "std"))]
#[global_allocator]
static TEST_ALLOCATOR: verified_engine::allocator::VerifiedAllocator =
    verified_engine::allocator::VerifiedAllocator;

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Copy, Clone, PartialEq)]
    struct SampleNoStdError {
        required: usize,
        tol: f64,
    }

    impl fmt::Display for SampleNoStdError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "sample error: required={}, tol={}",
                self.required, self.tol
            )
        }
    }

    impl core::error::Error for SampleNoStdError {}

    impl Diagnostic for SampleNoStdError {
        fn severity(&self) -> Severity {
            Severity::Warning
        }

        fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
            visitor.record_str("error_type", "SampleNoStdError");
            visitor.record_display("description", self);
            visitor.record_usize("required", self.required);
            visitor.record_f64("tol", self.tol);
            visitor.record_u64("u64_val", 7);
            visitor.record_i64("i64_val", -3);
            visitor.record_u32("u32_val", 11);
            visitor.record_i32("i32_val", -5);
            visitor.record_isize("isize_val", -9);
            visitor.record_f32("f32_val", 1.5);
            visitor.record_bool("fatal", false);
        }
    }

    struct StackFieldCounter {
        str_count: usize,
        display_count: usize,
        last_usize: usize,
    }

    impl DiagnosticVisitor for StackFieldCounter {
        fn record_str(&mut self, _key: &str, _value: &str) {
            self.str_count += 1;
        }

        fn record_display(&mut self, _key: &str, _value: &dyn fmt::Display) {
            self.display_count += 1;
        }

        fn record_usize(&mut self, _key: &str, value: usize) {
            self.last_usize = value;
            self.display_count += 1;
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn visiting_diagnostic_fields_allocates_zero_bytes() {
        let err = SampleNoStdError {
            required: 32,
            tol: 1e-9,
        };
        let ((str_count, display_count, last_usize, sev), _metrics) =
            verified_engine::engine::VerifiedEngine::run_verified(|| {
                let mut visitor = StackFieldCounter {
                    str_count: 0,
                    display_count: 0,
                    last_usize: 0,
                };
                err.visit_fields(&mut visitor);
                (
                    visitor.str_count,
                    visitor.display_count,
                    visitor.last_usize,
                    err.severity(),
                )
            });
        assert_eq!(sev, Severity::Warning);
        assert_eq!(str_count, 1);
        assert_eq!(display_count, 10);
        assert_eq!(last_usize, 32);
    }

    #[test]
    fn no_std_visitor_and_severity_contracts_hold() {
        let err = SampleNoStdError {
            required: 8,
            tol: 0.25,
        };
        let mut visitor = StackFieldCounter {
            str_count: 0,
            display_count: 0,
            last_usize: 0,
        };
        err.visit_fields(&mut visitor);
        assert_eq!(err.severity(), Severity::Warning);
        assert_eq!(visitor.str_count, 1);
        assert_eq!(visitor.display_count, 10);
        assert_eq!(visitor.last_usize, 8);
        assert_ne!(Severity::Info, Severity::Fatal);
    }

    #[cfg(feature = "std")]
    #[test]
    fn std_metadata_helper_materializes_all_visited_fields() {
        let err = SampleNoStdError {
            required: 12,
            tol: 0.5,
        };
        let meta = err.metadata();
        assert_eq!(
            meta.get("error_type").map(std::string::String::as_str),
            Some("SampleNoStdError")
        );
        assert_eq!(
            meta.get("required").map(std::string::String::as_str),
            Some("12")
        );
        assert_eq!(
            meta.get("u64_val").map(std::string::String::as_str),
            Some("7")
        );
        assert_eq!(
            meta.get("i64_val").map(std::string::String::as_str),
            Some("-3")
        );
        assert_eq!(
            meta.get("u32_val").map(std::string::String::as_str),
            Some("11")
        );
        assert_eq!(
            meta.get("i32_val").map(std::string::String::as_str),
            Some("-5")
        );
        assert_eq!(
            meta.get("isize_val").map(std::string::String::as_str),
            Some("-9")
        );
        assert_eq!(
            meta.get("f32_val").map(std::string::String::as_str),
            Some("1.5")
        );
        assert_eq!(
            meta.get("fatal").map(std::string::String::as_str),
            Some("false")
        );
    }

    #[cfg(feature = "std")]
    #[derive(Debug)]
    struct LayeredDiagnostic;

    #[cfg(feature = "std")]
    impl fmt::Display for LayeredDiagnostic {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("outer wrapper description")
        }
    }

    #[cfg(feature = "std")]
    impl core::error::Error for LayeredDiagnostic {}

    #[cfg(feature = "std")]
    impl Diagnostic for LayeredDiagnostic {
        fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
            visitor.record_str("error_type", "RootError");
            visitor.record_str("code", "ROOT-001");
            visitor.record_str("description", "root cause message");
            visitor.record_str("error_type", "MiddleWrapper");
            visitor.record_str("code", "MID-002");
            visitor.record_str("description", "middle wrapper message");
            visitor.record_str("error_type", "OuterWrapper");
            visitor.record_display("description", self);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn hashmap_visitor_preserves_root_metadata_across_causal_wrappers() {
        let meta = LayeredDiagnostic.metadata();
        assert_eq!(
            meta.get("code").map(std::string::String::as_str),
            Some("ROOT-001")
        );
        assert_eq!(
            meta.get("wrapper_code").map(std::string::String::as_str),
            Some("MID-002")
        );
        assert_eq!(
            meta.get("root_error_type").map(std::string::String::as_str),
            Some("RootError")
        );
        assert_eq!(
            meta.get("error_type").map(std::string::String::as_str),
            Some("OuterWrapper")
        );
        assert_eq!(
            meta.get("root_description")
                .map(std::string::String::as_str),
            Some("root cause message")
        );
        assert_eq!(
            meta.get("description").map(std::string::String::as_str),
            Some("outer wrapper description")
        );
    }
}
