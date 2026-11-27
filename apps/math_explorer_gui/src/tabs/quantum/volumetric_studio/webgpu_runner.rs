#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! `128^3` (`2,097,152`-voxel) WebGPU compute & off-main-thread Web Worker volumetric runner
//! with `navigator.gpu` runtime capability probing and zero-allocation extraction downsampling.

#[path = "webgpu_dispatch.rs"]
mod webgpu_dispatch;

use super::ast_field::PREVIEW_DRAG_GRID;
use super::compute_wgsl::ShaderPreviewKind;
use super::engine::VolumetricEngine;
use super::state::{
    FieldSourceMode, VolumetricStudioConfig, MAX_VOLUMETRIC_GRID, MAX_WEBGPU_VOLUMETRIC_GRID,
    MIN_VOLUMETRIC_GRID,
};
use crate::async_sim::web_worker::{
    step_schnakenberg_3d_f32, VolumetricStepJob, VolumetricWorkerBridge,
};
use crate::async_sim::webgpu::WebGpuComputeDispatcher;
use crate::web_io::PlatformInstant;
use math_explorer::applied::isosurface::VoxelGrid;
use math_explorer::physics::quantum::{
    HydrogenicSuperposition3D, OrbitalQuantumNumbers, QuantumFieldObservable,
};
use serde::{Deserialize, Serialize};
pub use webgpu_dispatch::downsample_or_copy;
use webgpu_dispatch::{
    deinterleave_uv_into, interleave_uv_into, seed_gyroid_buffers_f32, step_and_origin,
    PendingGpuReadback,
};

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
        WebGpuAdapterStatus::from_browser_probe(WebGpuComputeDispatcher::has_navigator_gpu())
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
    /// Thin `web-sys` WebGPU `@compute` pipeline and double-buffered `MAP_READ` staging dispatcher.
    pub gpu_dispatcher: WebGpuComputeDispatcher,
    worker: VolumetricWorkerBridge,
    pending_readback: PendingGpuReadback,
    u_buf: Vec<f32>,
    v_buf: Vec<f32>,
    u_scratch: Vec<f32>,
    v_scratch: Vec<f32>,
    uv_staging: Vec<f32>,
    quantum_basis_uv: Vec<f32>,
    turing_seeded_dim: usize,
    quantum_hi: Option<HydrogenicSuperposition3D>,
    quantum_key: ([u32; 4], [i32; 2], usize, u64),
}

impl Default for WebGpuVolumetricRunner {
    fn default() -> Self {
        Self {
            adapter_status: probe_webgpu_adapter_status(),
            shader_preview: ShaderPreviewKind::ComputeWgsl,
            nav_mode: ViewportNavMode::Orbit,
            last_step_ms: 0.4,
            high_res_grid: None,
            gpu_dispatcher: WebGpuComputeDispatcher::new(),
            worker: VolumetricWorkerBridge::new(),
            pending_readback: PendingGpuReadback::ScalarField,
            u_buf: Vec::new(),
            v_buf: Vec::new(),
            u_scratch: Vec::new(),
            v_scratch: Vec::new(),
            uv_staging: Vec::new(),
            quantum_basis_uv: Vec::new(),
            turing_seeded_dim: 0,
            quantum_hi: None,
            quantum_key: ([0; 4], [0; 2], 0, 0),
        }
    }
}

impl WebGpuVolumetricRunner {
    /// Clamps `cfg` down to `<= 48^3` `CpuZeroAlloc` stepping when the browser lacks `navigator.gpu`.
    pub fn enforce_navigator_gpu_fallback(&self, cfg: &mut VolumetricStudioConfig) -> bool {
        if self.adapter_status == WebGpuAdapterStatus::BrowserWorkerFallback
            && cfg.compute_backend == ComputeBackend::WebGpuCompute
        {
            cfg.compute_backend = ComputeBackend::CpuZeroAlloc;
            cfg.clamp_in_place();
            return true;
        }
        false
    }

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
        if self.uv_staging.capacity() < cap_len * 2 {
            self.uv_staging.reserve(cap_len * 2 - self.uv_staging.len());
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

    /// Polls any completed WebGPU staging readback or off-thread Web Worker 3D Turing step.
    pub fn poll_worker_into(
        &mut self,
        cfg: &VolumetricStudioConfig,
        extract_grid: &mut VoxelGrid,
    ) -> bool {
        if cfg.compute_backend != ComputeBackend::WebGpuCompute {
            return false;
        }
        match self.pending_readback {
            PendingGpuReadback::TuringInterleavedUv(total) => {
                if self.uv_staging.len() < total * 2 {
                    self.uv_staging.resize(total * 2, 0.0);
                }
                if let Some(gpu_ms) = self.gpu_dispatcher.poll_readback(&mut self.uv_staging) {
                    self.last_step_ms = gpu_ms.max(0.05);
                    deinterleave_uv_into(&self.uv_staging, &mut self.u_buf, &mut self.v_buf);
                    self.write_turing_to_grids(cfg, extract_grid);
                    return true;
                }
            }
            PendingGpuReadback::ScalarField => {
                if let Some(hi) = &mut self.high_res_grid {
                    if let Some(gpu_ms) = self.gpu_dispatcher.poll_readback(hi.data_mut()) {
                        self.last_step_ms = gpu_ms.max(0.05);
                        downsample_or_copy(hi, extract_grid);
                        return true;
                    }
                }
            }
        }
        if cfg.source != FieldSourceMode::TuringMorphogenesis {
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
                if !self.try_dispatch_formula_gpu(cfg, target_dim) {
                    if let Some(hi) = &mut self.high_res_grid {
                        engine.implicit_eval.populate_grid_in_place(cfg, hi);
                        downsample_or_copy(hi, &mut engine.voxel_grid);
                    }
                }
            }
            FieldSourceMode::CustomImplicitAst => {
                if !self.try_dispatch_formula_gpu(cfg, target_dim) {
                    if let Some(hi) = &mut self.high_res_grid {
                        engine.ast_kernel.populate_voxel_grid_in_place(hi);
                        downsample_or_copy(hi, &mut engine.voxel_grid);
                    }
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
        if !advance_time || self.try_dispatch_turing_gpu(cfg, dim) {
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
            [cfg.n1, cfg.l1, cfg.n2, cfg.l2],
            [cfg.m1, cfg.m2],
            dim,
            cfg.quantum_dt.to_bits(),
        );
        let q1 = OrbitalQuantumNumbers::clamped(cfg.n1, cfg.l1, cfg.m1);
        let q2 = OrbitalQuantumNumbers::clamped(cfg.n2, cfg.l2, cfg.m2);
        if self.quantum_hi.is_none() || self.quantum_key != key {
            let ext = cfg.physical_quantum_extent();
            let mut q = HydrogenicSuperposition3D::new(q1, q2, dim, dim, ext, cfg.quantum_dt);
            let total = dim * dim * dim;
            self.u_scratch.resize(total, 0.0);
            self.v_scratch.resize(total, 0.0);
            self.quantum_basis_uv.resize(total * 2, 0.0);
            q.set_superposition(0.0, 0.0);
            q.write_field_into(
                QuantumFieldObservable::SignedRealWavefunction,
                &mut self.u_scratch,
            );
            q.set_superposition(1.0, 0.0);
            q.write_field_into(
                QuantumFieldObservable::SignedRealWavefunction,
                &mut self.v_scratch,
            );
            interleave_uv_into(&self.u_scratch, &self.v_scratch, &mut self.quantum_basis_uv);
            q.set_superposition(cfg.superposition_mix, cfg.relative_phase);
            self.quantum_hi = Some(q);
            self.quantum_key = key;
        }
        let mut elapsed = 0.0;
        if let Some(q) = &mut self.quantum_hi {
            if !advance_time {
                q.set_superposition(cfg.superposition_mix, cfg.relative_phase);
            } else {
                q.step_in_place();
            }
            elapsed = q.elapsed_time();
        }
        if self.try_dispatch_quantum_gpu(cfg, dim, elapsed) {
            return;
        }
        if let (Some(q), Some(hi)) = (&self.quantum_hi, &mut self.high_res_grid) {
            q.write_field_into(cfg.observable.to_physics(), hi.data_mut());
        }
    }
}

#[cfg(test)]
#[path = "webgpu_runner_tests.rs"]
mod tests;
