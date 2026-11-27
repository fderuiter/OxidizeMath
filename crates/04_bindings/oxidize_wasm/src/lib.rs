#![doc = include_str!("../README.md")]
// [cite:engine_linalg]
// [cite:engine_diffeq]
// [cite:engine_pde]
// [cite:math_commons]

/// Symbolic expression bytecode evaluation and WebGPU `@compute` WGSL emission.
pub mod ast_eval;
/// Multivariate SDE and BAOAB Langevin pre-allocated workspace handles.
pub mod diffeq;
/// Typed [`Diagnostic`](math_diagnostics::Diagnostic) to JavaScript `Error` conversion.
pub mod error;
/// Pre-allocated dense and sparse numerical linear algebra workspace handles.
pub mod linalg;
/// Zero-copy linear-memory `Float64Array` buffer descriptors.
pub mod memory;
/// Continuum 1D/2D/3D finite-difference stencil and radix-2 FFT workspace handles.
pub mod pde;

pub use ast_eval::WasmAstKernel;
pub use diffeq::{MAX_SDE_DIM, WasmSdeWorkspace};
pub use error::{DiagnosticField, WasmSdkError, WasmSdkResult};
pub use linalg::{MAX_LINALG_DIM, WasmLinalgWorkspace};
pub use memory::{MAX_WASM_F64_BUFFER_LEN, WasmF64Buffer};
pub use pde::{MAX_PDE_CELLS, WasmPdeWorkspace};

use wasm_bindgen::prelude::*;

#[wasm_bindgen(typescript_custom_section)]
const TS_CUSTOM_TYPES: &'static str = r#"
/** Structured diagnostic error thrown by `@oxidizemath/wasm` methods on domain or capacity violations. */
export interface OxidizeDiagnosticError extends Error {
    readonly code: string;
    readonly errorType: string;
    readonly severity: "Info" | "Warning" | "Error" | "Fatal";
    readonly fields: Readonly<Record<string, string>>;
}

/** Zero-copy Float64Array descriptor into `wasmLinearMemory().buffer`. */
export interface LinearMemoryF64View {
    readonly byteOffset: number;
    readonly length: number;
}
"#;

/// Returns the active `WebAssembly.Memory` handle so JS/TS callers can construct zero-copy
/// `new Float64Array(wasmLinearMemory().buffer, byteOffset, length)` views.
#[wasm_bindgen(js_name = wasmLinearMemory)]
#[must_use]
pub fn wasm_linear_memory() -> JsValue {
    wasm_bindgen::memory()
}

/// Returns the semantic version of the `@oxidizemath/wasm` SDK.
#[wasm_bindgen(js_name = sdkVersion)]
#[must_use]
pub fn sdk_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn error_to_js_value(err: &WasmSdkError) -> JsValue {
    let js_err = js_sys::Error::new(&err.to_string());
    js_err.set_name(&err.error_type);
    let sev = match err.severity {
        math_diagnostics::Severity::Info => "Info",
        math_diagnostics::Severity::Warning => "Warning",
        math_diagnostics::Severity::Error => "Error",
        math_diagnostics::Severity::Fatal => "Fatal",
    };
    let fields_obj = js_sys::Object::new();
    for field in &err.fields {
        let _ = js_sys::Reflect::set(
            &fields_obj,
            &JsValue::from_str(&field.key),
            &JsValue::from_str(&field.value),
        );
    }
    let obj: JsValue = js_err.into();
    let _ = js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("code"),
        &JsValue::from_str(err.code),
    );
    let _ = js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("errorType"),
        &JsValue::from_str(&err.error_type),
    );
    let _ = js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("severity"),
        &JsValue::from_str(sev),
    );
    let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("fields"), &fields_obj);
    obj
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn error_to_js_value(_err: &WasmSdkError) -> JsValue {
    JsValue::UNDEFINED
}

#[cfg(test)]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;
