//! Host and headless Chromium (`wasm32`) tests for [`super::WebGpuVolumetricRunner`] and
//! [`super::super::compute_wgsl`].

use super::*;
use crate::framework::InteractiveTool;
use crate::tabs::quantum::volumetric_studio::compute_wgsl::{
    build_volumetric_compute_wgsl, build_volumetric_fragment_wgsl,
};
use crate::tabs::quantum::volumetric_studio::presets::PRESET_3D_TURING_GYROID_JSON;
use crate::tabs::quantum::volumetric_studio::VolumetricFieldStudioTool;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

#[test]
fn test_webgpu_compute_128_cubed_allocates_two_million_voxels_and_extracts_meshes() {
    let mut tool = VolumetricFieldStudioTool::default();
    tool.config.compute_backend = ComputeBackend::WebGpuCompute;
    tool.config.source = FieldSourceMode::TuringMorphogenesis;
    tool.config.grid_resolution = 128;
    assert!(tool.config.validate().is_ok());
    tool.refresh_zero_alloc();
    assert_eq!(tool.active_voxel_dimension(), 128);
    let obs = tool.observables();
    assert!(obs.contains(&("grid_resolution", 128.0)));
    let hi = tool
        .webgpu_runner
        .high_res_grid
        .as_ref()
        .expect("128^3 high_res_grid allocated");
    assert_eq!(hi.data().len(), 128 * 128 * 128);
    assert!(hi.data().iter().all(|v| v.is_finite()));
    assert!(!tool.engine.pos_mesh.vertices.is_empty());

    // Switching back to CpuZeroAlloc must clamp grid_resolution back to <= 48.
    tool.config.compute_backend = ComputeBackend::CpuZeroAlloc;
    tool.config.clamp_in_place();
    assert_eq!(tool.config.grid_resolution, 48);
    assert!(tool.config.validate().is_ok());
}

#[test]
fn test_compute_and_fragment_wgsl_emission_across_all_four_volumetric_modes() {
    let mut tool = VolumetricFieldStudioTool::default();
    tool.import_preset_json(PRESET_3D_TURING_GYROID_JSON)
        .expect("legacy preset without compute_backend loads cleanly");
    assert_eq!(tool.config.compute_backend, ComputeBackend::CpuZeroAlloc);

    for mode in [
        FieldSourceMode::QuantumSuperposition,
        FieldSourceMode::TuringMorphogenesis,
        FieldSourceMode::CustomImplicit,
        FieldSourceMode::CustomImplicitAst,
    ] {
        tool.config.source = mode;
        let compute_wgsl = build_volumetric_compute_wgsl(&tool.config)
            .unwrap_or_else(|e| panic!("compute wgsl failed for {mode:?}: {e}"));
        assert!(
            compute_wgsl.contains("@compute @workgroup_size(4, 4, 4)")
                && compute_wgsl
                    .contains("fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>)")
                && compute_wgsl.contains("@group(0) @binding(0)"),
            "missing @compute contract in {mode:?}: {compute_wgsl}"
        );
        let via_trait = tool.export_compute_shader_wgsl().expect("trait hook");
        assert_eq!(via_trait, compute_wgsl);

        let frag_wgsl = build_volumetric_fragment_wgsl(&tool.config).expect("fragment wgsl");
        assert!(frag_wgsl.contains("@fragment") && frag_wgsl.contains("fn fs_main("));
    }
}

#[test]
fn test_missing_navigator_gpu_clamps_to_48_cubed_cpu_and_keeps_wgsl_export_enabled() {
    let mut tool = VolumetricFieldStudioTool::default();
    tool.webgpu_runner.adapter_status = WebGpuAdapterStatus::BrowserWorkerFallback;
    tool.config.compute_backend = ComputeBackend::WebGpuCompute;
    tool.config.grid_resolution = 128;
    assert!(tool
        .webgpu_runner
        .enforce_navigator_gpu_fallback(&mut tool.config));
    assert_eq!(tool.config.compute_backend, ComputeBackend::CpuZeroAlloc);
    assert_eq!(tool.config.grid_resolution, 48);
    assert!(tool.export_capabilities().compute_shader);
    let wgsl = tool
        .export_compute_shader_wgsl()
        .expect("wgsl still exports");
    assert!(wgsl.contains("@compute @workgroup_size(4, 4, 4)"));
}

#[test]
fn test_webgpu_uniform_packing_and_interleaved_uv_staging_roundtrip() {
    use super::webgpu_dispatch::{
        deinterleave_uv_into, interleave_uv_into, pack_orbital_uniform_bytes,
        pack_turing_uniform_bytes, pack_volume_ast_uniform_bytes,
    };
    let cfg = VolumetricStudioConfig {
        compute_backend: ComputeBackend::WebGpuCompute,
        grid_resolution: 128,
        ..Default::default()
    };
    let turing_u = pack_turing_uniform_bytes(128, &cfg);
    assert_eq!(u32::from_le_bytes(turing_u[0..4].try_into().unwrap()), 128);
    assert_eq!(turing_u.len(), 32);
    let orbital_u = pack_orbital_uniform_bytes(96, 1.25, &cfg);
    assert_eq!(u32::from_le_bytes(orbital_u[0..4].try_into().unwrap()), 96);
    assert_eq!(orbital_u.len(), 32);
    let ast_u = pack_volume_ast_uniform_bytes(64);
    assert_eq!(u32::from_le_bytes(ast_u[0..4].try_into().unwrap()), 64);
    assert_eq!(ast_u.len(), 48);

    let u_in = [0.25_f32, 0.5, 0.75, 1.0];
    let v_in = [2.0_f32, 3.0, 4.0, 5.0];
    let mut staged = [0.0_f32; 8];
    interleave_uv_into(&u_in, &v_in, &mut staged);
    assert_eq!(staged, [0.25, 2.0, 0.5, 3.0, 0.75, 4.0, 1.0, 5.0]);
    let mut u_out = [0.0_f32; 4];
    let mut v_out = [0.0_f32; 4];
    deinterleave_uv_into(&staged, &mut u_out, &mut v_out);
    assert_eq!(u_out, u_in);
    assert_eq!(v_out, v_in);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test]
fn wasm_probe_webgpu_and_run_64_cubed_volumetric_step() {
    let status = probe_webgpu_adapter_status();
    assert!(matches!(
        status,
        WebGpuAdapterStatus::NavigatorGpuAvailable | WebGpuAdapterStatus::BrowserWorkerFallback
    ));
    let mut tool = VolumetricFieldStudioTool::default();
    tool.config.compute_backend = ComputeBackend::WebGpuCompute;
    tool.config.source = FieldSourceMode::TuringMorphogenesis;
    tool.config.grid_resolution = 64;
    tool.refresh_zero_alloc();
    assert_eq!(tool.active_voxel_dimension(), 64);
    assert!(tool.webgpu_runner.is_finite());
}
