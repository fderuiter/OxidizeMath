//! Host and headless Chromium (`wasm32`) tests for [`super::WebGpuComputeDispatcher`].

use super::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

const MINIMAL_COMPUTE_WGSL: &str = "\
struct Params { scale: f32, _p0: f32, _p1: f32, _p2: f32 };
@group(0) @binding(0) var<uniform> u_p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(4, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < 4u) { dst[gid.x] = src[gid.x] * u_p.scale; }
}
";

#[test]
fn test_workgroup_counts_and_double_buffered_staging_slot_rotation() {
    assert_eq!(
        compute_workgroup_counts([128, 128, 128], [4, 4, 4]),
        [32, 32, 32]
    );
    assert_eq!(compute_workgroup_counts([50, 50, 1], [8, 8, 1]), [7, 7, 1]);
    assert_eq!(compute_workgroup_counts([0, 0, 0], [0, 4, 4]), [1, 1, 1]);
    assert_eq!(GPU_BUFFER_USAGE_MAP_READ, 0x0001);
    assert_eq!(GPU_BUFFER_USAGE_STORAGE, 0x0080);

    let mut dispatcher = WebGpuComputeDispatcher::new();
    assert_eq!(dispatcher.active_staging_slot(), 0);
    let uniform = [0u8; 16];
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let dispatched = dispatcher.submit_compute(WebGpuDispatchSpec {
        shader_label: "unit_test_compute",
        wgsl_source: MINIMAL_COMPUTE_WGSL,
        entry_point: "cs_main",
        workgroups: [1, 1, 1],
        uniform_bytes: &uniform,
        input_f32: &input,
        output_len: 4,
    });
    assert!(!dispatched);
    assert_eq!(dispatcher.active_staging_slot(), 1);
    assert_eq!(dispatcher.state(), WebGpuDispatcherState::Unavailable);
    let mut out = [0.0_f32; 4];
    assert!(dispatcher.poll_readback(&mut out).is_none());
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test]
fn wasm_webgpu_dispatcher_probes_navigator_gpu_and_handles_missing_adapter_cleanly() {
    let has_gpu = WebGpuComputeDispatcher::has_navigator_gpu();
    let mut dispatcher = WebGpuComputeDispatcher::new();
    if has_gpu {
        assert_eq!(dispatcher.state(), WebGpuDispatcherState::Uninitialized);
    } else {
        assert_eq!(dispatcher.state(), WebGpuDispatcherState::Unavailable);
    }
    let uniform = [0u8; 16];
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let _ = dispatcher.submit_compute(WebGpuDispatchSpec {
        shader_label: "wasm_smoke_compute",
        wgsl_source: MINIMAL_COMPUTE_WGSL,
        entry_point: "cs_main",
        workgroups: [1, 1, 1],
        uniform_bytes: &uniform,
        input_f32: &input,
        output_len: 4,
    });
    assert!(matches!(
        dispatcher.state(),
        WebGpuDispatcherState::Initializing
            | WebGpuDispatcherState::Ready
            | WebGpuDispatcherState::Unavailable
    ));
    let mut out = [0.0_f32; 4];
    let _ = dispatcher.poll_readback(&mut out);
}
