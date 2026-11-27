//! Headless WebAssembly (`wasm32-unknown-unknown`) tests for `oxidize_wasm`.
// [cite:engine_linalg]
// [cite:engine_diffeq]
// [cite:engine_pde]
// [cite:math_commons]

use super::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn wasm_zero_copy_float64array_linear_memory_views_and_typed_js_errors() {
    let mut buf = WasmF64Buffer::try_new(4).expect("buffer");
    buf.copy_from_slice(&[1.25, 2.5, 3.75, 5.0])
        .expect("copy into buffer");

    let mem: js_sys::WebAssembly::Memory = wasm_linear_memory()
        .dyn_into()
        .expect("WebAssembly.Memory instance");
    let array_buf = mem.buffer();
    let view = js_sys::Float64Array::new_with_byte_offset_and_length(
        &array_buf,
        buf.byte_offset() as u32,
        buf.len() as u32,
    );
    assert_eq!(view.get_index(0), 1.25);
    assert_eq!(view.get_index(3), 5.0);

    // Mutate directly through the JS Float64Array view and verify Rust sees it with zero copies!
    view.set_index(1, 42.0);
    assert_eq!(buf.as_slice()[1], 42.0);

    // Verify structured JS Error object emitted by WasmSdkError::into(JsValue).
    let mut linalg = WasmLinalgWorkspace::try_new(4).expect("linalg");
    let singular = [1.0, 0.0, 0.0, 0.0];
    let rhs = [1.0, 1.0];
    let mut sol = [0.0; 2];
    let err = linalg
        .solve_lu(&singular, &rhs, &mut sol, 2)
        .expect_err("singular");
    let js_val: JsValue = err.into();
    assert!(js_val.is_instance_of::<js_sys::Error>());
    let code = js_sys::Reflect::get(&js_val, &JsValue::from_str("code"))
        .expect("code prop")
        .as_string();
    assert_eq!(code.as_deref(), Some("E_LINALG_SINGULAR"));
}

#[wasm_bindgen_test]
fn wasm_sdk_sde_pde_and_compute_wgsl_roundtrip() {
    assert!(!sdk_version().is_empty());
    let mut sde = WasmSdeWorkspace::try_new(2, 2, 77).expect("sde");
    let mut x = [1.0, 2.0];
    sde.step_geometric_brownian(0.0, 0.01, 0.05, 0.2, false, &mut x)
        .expect("gbm step");
    assert!(x[0].is_finite() && x[1].is_finite());

    let mut pde = WasmPdeWorkspace::try_new_2d(8, 8, 1.0, 1.0).expect("pde");
    pde.seed_schnakenberg_steady_state(0.1, 0.9, 0.03);
    pde.step_schnakenberg_2d_in_place(0.005, 1.0, 18.0, 0.1, 0.9, 2)
        .expect("step 2d");
    assert_eq!(pde.grid_len(), 64);

    let ast = WasmAstKernel::try_compile_3d("sin(x) * cos(y) + sin(z)").expect("ast");
    let wgsl = ast.to_compute_wgsl(64, 2.5).expect("wgsl");
    assert!(wgsl.contains("@compute @workgroup_size(4, 4, 4)"));
}
