//! Unit tests for 3D spatial AST WGSL validation and WebGPU `@compute` kernel emission.
// [cite:math_commons]

use super::compute_shader::{
    ComputeKernelConfig, DEFAULT_COMPUTE_WORKGROUP_3D, emit_wgsl_3d_compute_kernel,
};
use super::parser::parse;
use super::shader::ShaderEmitError;

#[test]
fn test_emit_3d_gyroid_compute_kernel_structure_and_params() {
    let expr = parse("sin(k * x) * cos(k * y) + sin(k * y) * cos(k * z) + sin(k * z) * cos(k * x)")
        .expect("valid gyroid expression");
    let cfg = ComputeKernelConfig::new_3d(128, core::f64::consts::PI).with_param("k", 1.5);
    let wgsl = emit_wgsl_3d_compute_kernel(&expr, &cfg).expect("compute WGSL should emit");

    assert_eq!(DEFAULT_COMPUTE_WORKGROUP_3D, [4, 4, 4]);
    assert!(wgsl.contains("@compute @workgroup_size(4, 4, 4)"));
    assert!(wgsl.contains("fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>)"));
    assert!(wgsl.contains("@group(0) @binding(0) var<uniform> u_vol: VolumeComputeUniforms;"));
    assert!(wgsl.contains("@group(0) @binding(1) var<storage, read> field_in: array<f32>;"));
    assert!(wgsl.contains("@group(0) @binding(2) var<storage, read_write> field_out: array<f32>;"));
    assert!(wgsl.contains("const DEFAULT_DIMS: vec3<u32> = vec3<u32>(128u, 128u, 128u);"));
    assert!(wgsl.contains("const k: f32 = 1.5;"));
    assert!(wgsl.contains("let x: f32 = pos.x; let y: f32 = pos.y; let z: f32 = pos.z;"));
}

#[test]
fn test_to_wgsl_3d_rejects_undeclared_or_reserved_variables() {
    let undeclared = parse("x^2 + y^2 + z^2 + omega").expect("valid syntax");
    let err = undeclared
        .to_wgsl_3d_with_vars(&["x", "y", "z", "t"])
        .expect_err("undeclared identifier must fail");
    assert_eq!(err, ShaderEmitError::InvalidIdentifier("omega".to_string()));

    let reserved_cfg = ComputeKernelConfig::new_3d(64, 2.5).with_param("field_out", 1.0);
    let valid_expr = parse("x + y + z").expect("valid syntax");
    let res = emit_wgsl_3d_compute_kernel(&valid_expr, &reserved_cfg);
    assert!(matches!(res, Err(ShaderEmitError::InvalidIdentifier(_))));
}

#[test]
fn test_compute_kernel_rejects_oversized_workgroup_and_invalid_bounds() {
    let expr = parse("sin(x) + cos(y) + sin(z * t)").expect("valid syntax");
    let mut bad_wg = ComputeKernelConfig::new_3d(64, 2.5);
    bad_wg.workgroup_size = [8, 8, 8]; // 512 > 256 WebGPU baseline limit
    assert!(matches!(
        emit_wgsl_3d_compute_kernel(&expr, &bad_wg),
        Err(ShaderEmitError::InvalidIdentifier(_))
    ));

    let mut bad_bounds = ComputeKernelConfig::new_3d(64, 2.5);
    bad_bounds.x_bounds = [2.0, -2.0];
    assert_eq!(
        emit_wgsl_3d_compute_kernel(&expr, &bad_bounds),
        Err(ShaderEmitError::NonFiniteLiteral)
    );
}
