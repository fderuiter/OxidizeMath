#![doc = include_str!("../README.md")]
extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;

mod crate_paths;
mod embed_theory;
mod latex_parser;
mod semantic_integrity;
mod theory;
mod verified;

/// Emits a structured `FATAL` JSON diagnostic line to `stderr` and immediately terminates the
/// process via [`std::process::abort`].
///
/// Used by `verified_engine::allocator::VerifiedAllocator` when dynamic heap allocation is
/// attempted while post-initialization memory locks are engaged.
///
/// # Example
/// ```rust
/// use verified_engine_macros::runtime_violation;
///
/// let allocation_locked = false;
/// if allocation_locked {
///     runtime_violation!("VIOLATION: Dynamic memory allocation after initialization.");
/// }
/// assert!(!allocation_locked);
/// ```
#[proc_macro]
pub fn runtime_violation(input: TokenStream) -> TokenStream {
    let msg = syn::parse_macro_input!(input as syn::LitStr);
    let msg_str = msg.value();
    let json = format!(
        r#"{{"severity":"FATAL","message":"{}","metadata":{{}},"thread_name":null}}"#,
        msg_str
    );
    let expanded = quote! {
        {
            eprintln!("{}", #json);
            std::process::abort();
        }
    };
    TokenStream::from(expanded)
}

/// Extracts one or more `\label{...}` mathematical environments from a workspace `.tex` manuscript
/// into the annotated item's Rustdoc comments and verifies AST operator semantic integrity.
///
/// # Compile-Time Invariants
/// - Resolves the `.tex` path relative to ancestor workspace roots from `CARGO_MANIFEST_DIR` and
///   emits a `const _: &str = include_str!(...)` dependency tracker for incremental rebuilds.
/// - Rejects manual `$$ ... $$` shadow docstrings on the same item to maintain single-source
///   equation fidelity.
/// - Validates that operators required by the LaTeX formula (such as division for `\frac`) appear
///   in the Rust function body.
///
/// # Example
/// ```rust
/// use verified_engine_macros::embed_theory;
///
/// #[embed_theory("papers/mmwave_radiotherapy_setup.tex", label = "eq:range")]
/// pub fn beat_frequency(bandwidth: f64, range: f64, c: f64, sweep_time: f64) -> f64 {
///     (2.0 * bandwidth * range) / (c * sweep_time)
/// }
///
/// let fb = beat_frequency(4.0e9, 1.5, 3.0e8, 1.0e-3);
/// assert!((fb - 40_000.0).abs() < 1e-6);
/// ```
#[proc_macro_attribute]
pub fn embed_theory(attr: TokenStream, item: TokenStream) -> TokenStream {
    embed_theory::embed_theory_impl(attr.into(), item.into()).into()
}

/// Enforces NASA Power of 10 static safety invariants on functions and methods and injects
/// runtime complexity instrumentation on non-`const` bodies.
///
/// # Compile-Time Invariants
/// - **Rule 1 (No Direct Recursion)**: Rejects direct self-calls unless `allow_recursion = "reason"`
///   (for inherent delegation) or `opt_out = "reason"` is explicitly supplied.
/// - **Rule 4 (Bounded Function Size)**: Rejects function bodies exceeding 60 statements unless
///   `opt_out = "reason"` is supplied.
///
/// # Examples
/// ```rust
/// use verified_engine_macros::verified;
///
/// #[verified]
/// pub const fn kinetic_energy(mass: f64, velocity: f64) -> f64 {
///     0.5 * mass * velocity * velocity
/// }
///
/// #[verified(allow_recursion = "bounded Euclidean remainder descent")]
/// pub const fn gcd(a: u64, b: u64) -> u64 {
///     if b == 0 { a } else { gcd(b, a % b) }
/// }
///
/// assert_eq!(kinetic_energy(2.0, 3.0), 9.0);
/// assert_eq!(gcd(48, 18), 6);
/// ```
///
/// Direct recursion without an explicit justification is rejected at compile time:
///
/// ```rust,compile_fail
/// use verified_engine_macros::verified;
///
/// #[verified]
/// const fn bad_factorial(n: u64) -> u64 {
///     if n <= 1 { 1 } else { n * bad_factorial(n - 1) }
/// }
/// ```
#[proc_macro_attribute]
pub fn verified(attr: TokenStream, item: TokenStream) -> TokenStream {
    verified::verified_impl(attr.into(), item.into()).into()
}

/// Derives `scientific_metadata::theory::TheoryDescribable` for structs with named fields,
/// validating parameter bounds (`min <= max`, `step > 0.0`) and citations at macro-expansion time.
///
/// # Examples
/// ```rust
/// use scientific_metadata::theory::TheoryDescribable;
/// use verified_engine_macros::Theory;
///
/// #[derive(Theory)]
/// #[theory(
///     description = "Damped harmonic oscillator",
///     phonetic = "Damped harmonic oscillator",
///     citation = "Goldstein, Classical Mechanics (2002)"
/// )]
/// pub struct OscillatorConfig {
///     #[theory(min = 0.0, max = 50.0, step = 0.1)]
///     pub damping: f64,
/// }
///
/// let cfg = OscillatorConfig { damping: 0.5 };
/// assert_eq!(cfg.theory_description(), "Damped harmonic oscillator");
/// assert_eq!(cfg.theory_parameters()["damping"].max, 50.0);
/// ```
///
/// Inverted parameter ranges (`min > max`) fail at compile time:
///
/// ```rust,compile_fail
/// use verified_engine_macros::Theory;
///
/// #[derive(Theory)]
/// pub struct InvalidBounds {
///     #[theory(min = 10.0, max = 1.0, step = 0.5)]
///     pub ratio: f64,
/// }
/// ```
#[proc_macro_derive(Theory, attributes(theory))]
pub fn derive_theory(input: TokenStream) -> TokenStream {
    theory::derive_theory_impl(input.into()).into()
}
