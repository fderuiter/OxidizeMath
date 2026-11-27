#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! `128^3` (`2,097,152`-voxel) WebGPU compute & off-main-thread Web Worker volumetric runner
//! with `navigator.gpu` runtime capability probing and zero-allocation extraction downsampling.

use super::ast_field::PREVIEW_DRAG_GRID;
use super::compute_wgsl::ShaderPreviewKind;
use super::engine::VolumetricEngine;
use super::state::{
    FieldSourceMode, QuantumObservableChoice, VolumetricStudioConfig, MAX_VOLUMETRIC_GRID,
    MAX_WEBGPU_VOLUMETRIC_GRID, MIN_VOLUMETRIC_GRID,
};
use crate::async_sim::web_worker::{
    step_schnakenberg_3d_f32, VolumetricStepJob, VolumetricWorkerBridge,
};
use crate::web_io::PlatformInstant;
use math_commons::math_kernel::types::flatten_3d_index;
use math_explorer::applied::isosurface::{Point3D, VoxelGrid};
use math_explorer::physics::quantum::{HydrogenicSuperposition3D, OrbitalQuantumNumbers};
use serde::{Deserialize, Serialize};

/// Execution backend selected in `VolumetricStudioConfig`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeBackend {
    /// Pre-allocated CPU zero-allocation workspace (`16^3..=48^3`).
    #[default]
    CpuZeroAlloc,
    /// High-resolution WebGPU `@compute` / Web Worker volumetric pipeline (`16^3..=128^3`).
    WebGpuCompute,
}

impl ComputeBackend {
    /// Serde predicate omitting default `cpu_zero_alloc` to preserve backward-compatible preset bytes.
    #[must_use]
    pub const fn is_cpu_default(&self) -> bool {
        matches!(self, Self::CpuZeroAlloc)
    }

    /// Human-readable UI toggle label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::CpuZeroAlloc => "CPU Zero-Alloc",
            Self::WebGpuCompute => "WebGPU Compute",
        }
    }
}

/// Probed WebGPU adapter status in the active browser or native runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WebGpuAdapterStatus {
    /// Browser exposes `navigator.gpu` (`GPU` object available).
    NavigatorGpuAvailable,
    /// Browser lacks `navigator.gpu`; falls back cleanly to off-thread `VolumetricWorkerBridge`.
    BrowserWorkerFallback,
    /// Native desktop host using background `VolumetricWorkerBridge` + SIMD slice stepping.
    #[cfg(not(target_arch = "wasm32"))]
    NativeHostWorker,
}

impl WebGpuAdapterStatus {
    /// Maps a `navigator.gpu` presence boolean to the corresponding browser adapter status.
    #[must_use]
    pub const fn from_browser_probe(has_gpu: bool) -> Self {
        if has_gpu {
            Self::NavigatorGpuAvailable
        } else {
            Self::BrowserWorkerFallback
        }
    }

    /// Short HUD telemetry tag paired with [`ComputeBackend`].
    #[must_use]
    pub const fn hud_backend_label(self, backend: ComputeBackend) -> &'static str {
        match (backend, self) {
            (ComputeBackend::CpuZeroAlloc, _) => "CPU",
            (ComputeBackend::WebGpuCompute, Self::NavigatorGpuAvailable) => "WebGPU",
            (ComputeBackend::WebGpuCompute, Self::BrowserWorkerFallback) => "WebGPU/Worker",
            #[cfg(not(target_arch = "wasm32"))]
            (ComputeBackend::WebGpuCompute, Self::NativeHostWorker) => "WebGPU/Sim",
        }
    }
}

/// Probes `navigator.gpu` availability on `wasm32` without requiring unstable `web-sys` flags.
#[must_use]
pub fn probe_webgpu_adapter_status() -> WebGpuAdapterStatus {
    #[cfg(target_arch = "wasm32")]
    {
        wasm_gpu_probe::probe_navigator_gpu()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        match std::env::var("OXIDIZE_FORCE_WEBGPU_PROBE").ok().as_deref() {
            Some("gpu") => WebGpuAdapterStatus::from_browser_probe(true),
            Some("worker") => WebGpuAdapterStatus::from_browser_probe(false),
            _ => WebGpuAdapterStatus::NativeHostWorker,
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm_gpu_probe {
    use super::WebGpuAdapterStatus;
    use wasm_bindgen::{JsCast, JsValue};

    pub(super) fn probe_navigator_gpu() -> WebGpuAdapterStatus {
        let global = js_sys::global();
        let has_gpu = global
            .dyn_into::<web_sys::Window>()
            .ok()
            .and_then(|win| js_sys::Reflect::get(&win.navigator(), &JsValue::from_str("gpu")).ok())
            .is_some_and(|gpu| !gpu.is_undefined() && !gpu.is_null());
        WebGpuAdapterStatus::from_browser_probe(has_gpu)
    }
}

/// Viewport mouse navigation gesture mode selected in the top-left HUD pill.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewportNavMode {
    /// Left-drag orbits camera yaw/pitch.
    #[default]
    Orbit,
    /// Drag pans camera in screen plane.
    Pan,
    /// Drag/scroll zooms camera.
    Zoom,
}

/// Lazily allocated high-resolution (`<= 128^3`) volumetric compute & worker runner.
pub struct WebGpuVolumetricRunner {
    /// Probed WebGPU adapter status.
    pub adapter_status: WebGpuAdapterStatus,
    /// Active WGSL stage shown in the sidebar `GPU Shader Preview` drawer.
    pub shader_preview: ShaderPreviewKind,
    /// Active viewport HUD navigation pill mode.
    pub nav_mode: ViewportNavMode,
    /// Duration of the most recent volumetric step/evaluation in milliseconds.
    pub last_step_ms: f64,
    /// High-resolution cubic `VoxelGrid` (`dim^3`, up to `128^3 = 2,097,152` voxels) when active.
    pub high_res_grid: Option<VoxelGrid>,
    worker: VolumetricWorkerBridge,
    u_buf: Vec<f32>,
    v_buf: Vec<f32>,
    u_scratch: Vec<f32>,
    v_scratch: Vec<f32>,
    turing_seeded_dim: usize,
    quantum_hi: Option<HydrogenicSuperposition3D>,
    quantum_key: (u32, u32, i32, u32, u32, i32, usize, u64),
}

impl Default for WebGpuVolumetricRunner {
    fn default() -> Self {
        Self {
            adapter_status: probe_webgpu_adapter_status(),
            shader_preview: ShaderPreviewKind::ComputeWgsl,
            nav_mode: ViewportNavMode::Orbit,
            last_step_ms: 0.4,
            high_res_grid: None,
            worker: VolumetricWorkerBridge::new(),
            u_buf: Vec::new(),
            v_buf: Vec::new(),
            u_scratch: Vec::new(),
            v_scratch: Vec::new(),
            turing_seeded_dim: 0,
            quantum_hi: None,
            quantum_key: (0, 0, 0, 0, 0, 0, 0, 0),
        }
    }
}

impl WebGpuVolumetricRunner {
    /// Returns `true` when `high_res_grid` is active and holds finite values.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.high_res_grid
            .as_ref()
            .is_none_or(|g| g.data().iter().all(|v| v.is_finite()))
    }

    /// Pre-allocates `high_res_grid` and ping-pong buffers up to `max_dim^3` (`<= 128^3`).
    pub fn ensure_capacity(&mut self, max_dim: usize, half_extents: [f64; 3]) {
        let cap_dim = max_dim.clamp(MIN_VOLUMETRIC_GRID, MAX_WEBGPU_VOLUMETRIC_GRID);
        let cap_len = cap_dim * cap_dim * cap_dim;
        if self.u_buf.capacity() < cap_len {
            self.u_buf.reserve(cap_len - self.u_buf.len());
            self.v_buf.reserve(cap_len - self.v_buf.len());
            self.u_scratch.reserve(cap_len - self.u_scratch.len());
            self.v_scratch.reserve(cap_len - self.v_scratch.len());
        }
        if self
            .high_res_grid
            .as_ref()
            .is_none_or(|g| g.data().len() < cap_len)
        {
            let mut backing = Vec::with_capacity(cap_len);
            backing.resize(cap_len, 0.0_f32);
            let (step, orig) = step_and_origin(cap_dim, half_extents);
            self.high_res_grid = VoxelGrid::builder()
                .dimensions(cap_dim, cap_dim, cap_dim)
                .data(backing)
                .voxel_size(step)
                .origin(orig)
                .build()
                .ok();
        }
    }

    /// Resets seeded 3D Turing buffers and quantum phase when the user clicks Reset.
    pub fn reset(&mut self) {
        self.turing_seeded_dim = 0;
        if let Some(q) = &mut self.quantum_hi {
            q.set_superposition(0.5, 0.0);
        }
    }

    /// Polls any completed off-thread Web Worker 3D Turing step, returning `true` if new voxels arrived.
    pub fn poll_worker_into(
        &mut self,
        cfg: &VolumetricStudioConfig,
        extract_grid: &mut VoxelGrid,
    ) -> bool {
        if cfg.compute_backend != ComputeBackend::WebGpuCompute
            || cfg.source != FieldSourceMode::TuringMorphogenesis
        {
            return false;
        }
        let Some(ms) = self
            .worker
            .poll_completed_step(&mut self.u_buf, &mut self.v_buf)
        else {
            return false;
        };
        self.last_step_ms = ms.max(0.05);
        self.write_turing_to_grids(cfg, extract_grid);
        true
    }

    /// Evaluates or advances the `WebGpuCompute` field (`16^3..=128^3`) and populates `engine.voxel_grid` (`<= 48^3`).
    pub fn evaluate_and_sync(
        &mut self,
        cfg: &VolumetricStudioConfig,
        advance_time: bool,
        drag_preview: bool,
        async_playback: bool,
        engine: &mut VolumetricEngine,
    ) {
        let t0 = PlatformInstant::now();
        let target_dim = if drag_preview {
            PREVIEW_DRAG_GRID
        } else {
            cfg.grid_resolution
                .clamp(MIN_VOLUMETRIC_GRID, MAX_WEBGPU_VOLUMETRIC_GRID)
        };
        let half_ext = cfg.viewport_half_extents();
        self.ensure_capacity(target_dim, half_ext);
        let (step, orig) = step_and_origin(target_dim, half_ext);
        if let Some(hi) = &mut self.high_res_grid {
            let _ = hi.reconfigure_in_place((target_dim, target_dim, target_dim), step, orig);
        }
        let ext_dim = target_dim.min(MAX_VOLUMETRIC_GRID);
        let (ext_step, ext_orig) = step_and_origin(ext_dim, half_ext);
        let _ =
            engine
                .voxel_grid
                .reconfigure_in_place((ext_dim, ext_dim, ext_dim), ext_step, ext_orig);

        match cfg.source {
            FieldSourceMode::TuringMorphogenesis => {
                self.step_or_seed_turing(cfg, target_dim, advance_time, async_playback);
                self.write_turing_to_grids(cfg, &mut engine.voxel_grid);
            }
            FieldSourceMode::QuantumSuperposition => {
                self.eval_quantum_hi(cfg, target_dim, advance_time);
                if let Some(hi) = &self.high_res_grid {
                    downsample_or_copy(hi, &mut engine.voxel_grid);
                }
            }
            FieldSourceMode::CustomImplicit => {
                engine.implicit_eval.sync_zero_alloc(cfg);
                if let Some(hi) = &mut self.high_res_grid {
                    engine.implicit_eval.populate_grid_in_place(cfg, hi);
                    downsample_or_copy(hi, &mut engine.voxel_grid);
                }
            }
            FieldSourceMode::CustomImplicitAst => {
                if let Some(hi) = &mut self.high_res_grid {
                    engine.ast_kernel.populate_voxel_grid_in_place(hi);
                    downsample_or_copy(hi, &mut engine.voxel_grid);
                }
            }
        }
        self.last_step_ms = (t0.elapsed().as_secs_f64() * 1_000.0).max(0.05);
    }

    fn step_or_seed_turing(
        &mut self,
        cfg: &VolumetricStudioConfig,
        dim: usize,
        advance_time: bool,
        async_playback: bool,
    ) {
        let total = dim * dim * dim;
        if self.turing_seeded_dim != dim || self.u_buf.len() != total {
            self.u_buf.resize(total, 0.0);
            self.v_buf.resize(total, 0.0);
            self.u_scratch.resize(total, 0.0);
            self.v_scratch.resize(total, 0.0);
            seed_gyroid_buffers_f32(
                dim,
                cfg.turing_a as f32,
                cfg.turing_b as f32,
                &mut self.u_buf,
                &mut self.v_buf,
            );
            self.turing_seeded_dim = dim;
        }
        if !advance_time {
            return;
        }
        let job = VolumetricStepJob {
            dim,
            dt: cfg.turing_dt as f32,
            du: cfg.turing_du as f32,
            dv: cfg.turing_dv as f32,
            a: cfg.turing_a as f32,
            b: cfg.turing_b as f32,
            substeps: 1,
        };
        if async_playback {
            let _ = self
                .worker
                .submit_turing_step(job, &self.u_buf, &self.v_buf);
        } else {
            step_schnakenberg_3d_f32(
                job,
                &mut self.u_buf,
                &mut self.v_buf,
                &mut self.u_scratch,
                &mut self.v_scratch,
            );
        }
    }

    fn write_turing_to_grids(
        &mut self,
        cfg: &VolumetricStudioConfig,
        extract_grid: &mut VoxelGrid,
    ) {
        let Some(hi) = &mut self.high_res_grid else {
            return;
        };
        let u_ss = (cfg.turing_a + cfg.turing_b) as f32;
        for (dst_slot, &u_val) in hi.data_mut().iter_mut().zip(self.u_buf.iter()) {
            *dst_slot = u_val - u_ss;
        }
        downsample_or_copy(hi, extract_grid);
    }

    fn eval_quantum_hi(&mut self, cfg: &VolumetricStudioConfig, dim: usize, advance_time: bool) {
        let key = (
            cfg.n1,
            cfg.l1,
            cfg.m1,
            cfg.n2,
            cfg.l2,
            cfg.m2,
            dim,
            cfg.quantum_dt.to_bits(),
        );
        let q1 = OrbitalQuantumNumbers::clamped(cfg.n1, cfg.l1, cfg.m1);
        let q2 = OrbitalQuantumNumbers::clamped(cfg.n2, cfg.l2, cfg.m2);
        if self.quantum_hi.is_none() || self.quantum_key != key {
            let mut q = HydrogenicSuperposition3D::new(
                q1,
                q2,
                dim,
                dim,
                cfg.physical_quantum_extent(),
                cfg.quantum_dt,
            );
            q.set_superposition(cfg.superposition_mix, cfg.relative_phase);
            self.quantum_hi = Some(q);
            self.quantum_key = key;
        }
        if let Some(q) = &mut self.quantum_hi {
            if !advance_time {
                q.set_superposition(cfg.superposition_mix, cfg.relative_phase);
            } else {
                q.step_in_place();
            }
            if let Some(hi) = &mut self.high_res_grid {
                let obs = match cfg.observable {
                    QuantumObservableChoice::SignedWavefunction => cfg.observable.to_physics(),
                    QuantumObservableChoice::ProbabilityDensity => cfg.observable.to_physics(),
                };
                q.write_field_into(obs, hi.data_mut());
            }
        }
    }
}

fn seed_gyroid_buffers_f32(dim: usize, a: f32, b: f32, u: &mut [f32], v: &mut [f32]) {
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

fn step_and_origin(dim: usize, ext: [f64; 3]) -> (Point3D, Point3D) {
    let s = 2.0 / ((dim.max(2) - 1) as f64);
    let step = Point3D::new(
        (ext[0] * s) as f32,
        (ext[1] * s) as f32,
        (ext[2] * s) as f32,
    );
    let orig = Point3D::new(-(ext[0] as f32), -(ext[1] as f32), -(ext[2] as f32));
    (step, orig)
}

#[cfg(test)]
#[path = "webgpu_runner_tests.rs"]
mod tests;
