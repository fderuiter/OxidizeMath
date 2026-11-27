#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Uniform packing, interleaved `vec2<f32>` staging, and `WebGpuComputeDispatcher` submission
//! helpers across all four [`FieldSourceMode`] kernels for [`super::WebGpuVolumetricRunner`].

use super::super::compute_wgsl::build_volumetric_compute_wgsl;
use super::super::state::{FieldSourceMode, QuantumObservableChoice, VolumetricStudioConfig};
use super::{WebGpuAdapterStatus, WebGpuVolumetricRunner};
use crate::async_sim::webgpu::{compute_workgroup_counts, WebGpuDispatchSpec};
use math_commons::math_kernel::types::flatten_3d_index;
use math_explorer::applied::isosurface::{Point3D, VoxelGrid};

/// Tracks how the next completed `MAP_READ` staging buffer should be unpacked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum PendingGpuReadback {
    #[default]
    ScalarField,
    TuringInterleavedUv(usize),
}

/// Packs the 32-byte `TuringParams` uniform block matching `emit_turing_3d_compute_wgsl`.
#[must_use]
pub(super) fn pack_turing_uniform_bytes(dim: usize, cfg: &VolumetricStudioConfig) -> [u8; 32] {
    let d32 = dim as u32;
    let mut out = [0u8; 32];
    out[0..4].copy_from_slice(&d32.to_le_bytes());
    out[4..8].copy_from_slice(&d32.to_le_bytes());
    out[8..12].copy_from_slice(&d32.to_le_bytes());
    out[12..16].copy_from_slice(&(cfg.turing_dt as f32).to_le_bytes());
    out[16..20].copy_from_slice(&(cfg.turing_du as f32).to_le_bytes());
    out[20..24].copy_from_slice(&(cfg.turing_dv as f32).to_le_bytes());
    out[24..28].copy_from_slice(&(cfg.turing_a as f32).to_le_bytes());
    out[28..32].copy_from_slice(&(cfg.turing_b as f32).to_le_bytes());
    out
}

/// Packs the 32-byte `OrbitalParams` uniform block matching `emit_quantum_3d_compute_wgsl`.
#[must_use]
pub(super) fn pack_orbital_uniform_bytes(
    dim: usize,
    elapsed_time: f64,
    cfg: &VolumetricStudioConfig,
) -> [u8; 32] {
    let d32 = dim as u32;
    let prob = u32::from(cfg.observable == QuantumObservableChoice::ProbabilityDensity);
    let mut out = [0u8; 32];
    out[0..4].copy_from_slice(&d32.to_le_bytes());
    out[4..8].copy_from_slice(&d32.to_le_bytes());
    out[8..12].copy_from_slice(&d32.to_le_bytes());
    out[12..16].copy_from_slice(&(elapsed_time as f32).to_le_bytes());
    out[16..20].copy_from_slice(&(cfg.physical_quantum_extent() as f32).to_le_bytes());
    out[20..24].copy_from_slice(&(cfg.superposition_mix as f32).to_le_bytes());
    out[24..28].copy_from_slice(&(cfg.relative_phase as f32).to_le_bytes());
    out[28..32].copy_from_slice(&prob.to_le_bytes());
    out
}

/// Packs the 48-byte `VolumeComputeUniforms` block matching `emit_wgsl_3d_compute_kernel`.
#[must_use]
pub(super) fn pack_volume_ast_uniform_bytes(dim: usize) -> [u8; 48] {
    let d32 = dim as u32;
    let mut out = [0u8; 48];
    out[0..4].copy_from_slice(&d32.to_le_bytes());
    out[4..8].copy_from_slice(&d32.to_le_bytes());
    out[8..12].copy_from_slice(&d32.to_le_bytes());
    out
}

/// Interleaves `u` and `v` scalar slices into `dst` (`[u0, v0, u1, v1, ...]`) without reallocating.
pub(super) fn interleave_uv_into(u: &[f32], v: &[f32], dst: &mut [f32]) {
    let n = u.len().min(v.len()).min(dst.len() / 2);
    for i in 0..n {
        dst[2 * i] = u[i];
        dst[2 * i + 1] = v[i];
    }
}

/// De-interleaves `src` (`[u0, v0, u1, v1, ...]`) back into `u` and `v` slices without reallocating.
pub(super) fn deinterleave_uv_into(src: &[f32], u: &mut [f32], v: &mut [f32]) {
    let n = u.len().min(v.len()).min(src.len() / 2);
    for i in 0..n {
        u[i] = src[2 * i];
        v[i] = src[2 * i + 1];
    }
}

impl WebGpuVolumetricRunner {
    pub(super) fn try_dispatch_turing_gpu(
        &mut self,
        cfg: &VolumetricStudioConfig,
        dim: usize,
    ) -> bool {
        if self.adapter_status != WebGpuAdapterStatus::NavigatorGpuAvailable {
            return false;
        }
        let Ok(wgsl) = build_volumetric_compute_wgsl(cfg) else {
            return false;
        };
        let total = dim * dim * dim;
        let pair_len = total * 2;
        if self.uv_staging.len() != pair_len {
            self.uv_staging.resize(pair_len, 0.0);
        }
        interleave_uv_into(&self.u_buf, &self.v_buf, &mut self.uv_staging);
        let u_bytes = pack_turing_uniform_bytes(dim, cfg);
        let d32 = dim as u32;
        let ok = self.gpu_dispatcher.submit_compute(WebGpuDispatchSpec {
            shader_label: "volumetric_3d_turing_compute",
            wgsl_source: &wgsl,
            entry_point: "cs_main",
            workgroups: compute_workgroup_counts([d32, d32, d32], [4, 4, 4]),
            uniform_bytes: &u_bytes,
            input_f32: &self.uv_staging,
            output_len: pair_len,
        });
        if ok {
            self.pending_readback = PendingGpuReadback::TuringInterleavedUv(total);
        }
        ok
    }

    pub(super) fn try_dispatch_quantum_gpu(
        &mut self,
        cfg: &VolumetricStudioConfig,
        dim: usize,
        elapsed_time: f64,
    ) -> bool {
        if self.adapter_status != WebGpuAdapterStatus::NavigatorGpuAvailable
            || self.quantum_basis_uv.len() != dim * dim * dim * 2
        {
            return false;
        }
        let Ok(wgsl) = build_volumetric_compute_wgsl(cfg) else {
            return false;
        };
        let u_bytes = pack_orbital_uniform_bytes(dim, elapsed_time, cfg);
        let d32 = dim as u32;
        let ok = self.gpu_dispatcher.submit_compute(WebGpuDispatchSpec {
            shader_label: "volumetric_3d_quantum_compute",
            wgsl_source: &wgsl,
            entry_point: "cs_main",
            workgroups: compute_workgroup_counts([d32, d32, d32], [4, 4, 4]),
            uniform_bytes: &u_bytes,
            input_f32: &self.quantum_basis_uv,
            output_len: dim * dim * dim,
        });
        if ok {
            self.pending_readback = PendingGpuReadback::ScalarField;
        }
        ok
    }

    pub(super) fn try_dispatch_formula_gpu(
        &mut self,
        cfg: &VolumetricStudioConfig,
        dim: usize,
    ) -> bool {
        if self.adapter_status != WebGpuAdapterStatus::NavigatorGpuAvailable {
            return false;
        }
        let Ok(wgsl) = build_volumetric_compute_wgsl(cfg) else {
            return false;
        };
        let label = if cfg.source == FieldSourceMode::CustomImplicit {
            "volumetric_3d_implicit_cad_compute"
        } else {
            "volumetric_3d_ast_compute"
        };
        let u_bytes = pack_volume_ast_uniform_bytes(dim);
        let d32 = dim as u32;
        let ok = self.gpu_dispatcher.submit_compute(WebGpuDispatchSpec {
            shader_label: label,
            wgsl_source: &wgsl,
            entry_point: "cs_main",
            workgroups: compute_workgroup_counts([d32, d32, d32], [4, 4, 4]),
            uniform_bytes: &u_bytes,
            input_f32: &[0.0_f32; 4],
            output_len: dim * dim * dim,
        });
        if ok {
            self.pending_readback = PendingGpuReadback::ScalarField;
        }
        ok
    }
}

pub(super) fn seed_gyroid_buffers_f32(dim: usize, a: f32, b: f32, u: &mut [f32], v: &mut [f32]) {
    let u0 = a + b;
    let v0 = b / (u0 * u0).max(1e-4);
    let scale = std::f32::consts::TAU / (dim.max(2) as f32);
    for z in 0..dim {
        let zf = (z as f32) * scale;
        for y in 0..dim {
            let yf = (y as f32) * scale;
            for x in 0..dim {
                let xf = (x as f32) * scale;
                let g = xf.sin() * yf.cos() + yf.sin() * zf.cos() + zf.sin() * xf.cos();
                let idx = flatten_3d_index(x, y, z, dim, dim);
                u[idx] = (u0 + 0.18 * g).clamp(0.05, 8.0);
                v[idx] = (v0 - 0.12 * g).clamp(0.05, 8.0);
            }
        }
    }
}

/// Copies `src` directly when dimensions match, or downsamples `src` (`<= 128^3`) into `dst` (`48^3`)
/// with zero heap allocations.
pub fn downsample_or_copy(src: &VoxelGrid, dst: &mut VoxelGrid) {
    let s_dim = src.width();
    let d_dim = dst.width();
    if s_dim == d_dim && src.data().len() == dst.data().len() {
        dst.data_mut().copy_from_slice(src.data());
        return;
    }
    if s_dim < 2 || d_dim < 2 {
        return;
    }
    let ratio = ((s_dim - 1) as f64) / ((d_dim - 1) as f64);
    let src_slice = src.data();
    let dst_slice = dst.data_mut();
    for z in 0..d_dim {
        let sz = (((z as f64) * ratio).round() as usize).min(s_dim - 1);
        for y in 0..d_dim {
            let sy = (((y as f64) * ratio).round() as usize).min(s_dim - 1);
            for x in 0..d_dim {
                let sx = (((x as f64) * ratio).round() as usize).min(s_dim - 1);
                let s_idx = flatten_3d_index(sx, sy, sz, s_dim, s_dim);
                let d_idx = flatten_3d_index(x, y, z, d_dim, d_dim);
                dst_slice[d_idx] = src_slice[s_idx];
            }
        }
    }
}

pub(super) fn step_and_origin(dim: usize, ext: [f64; 3]) -> (Point3D, Point3D) {
    let s = 2.0 / ((dim.max(2) - 1) as f64);
    let step = Point3D::new(
        (ext[0] * s) as f32,
        (ext[1] * s) as f32,
        (ext[2] * s) as f32,
    );
    let orig = Point3D::new(-(ext[0] as f32), -(ext[1] as f32), -(ext[2] as f32));
    (step, orig)
}
