#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::kspace_phantom::{build_preset_mask_into, PhantomEngine};
use super::kspace_types::{
    KSpaceMaskPreset, MriKSpaceSharedParams, MriKSpaceToolState, PhantomPreset, ISOCHROMAT_COUNT,
    MAX_TRAJECTORY_POINTS, STUDIO_FOV_METERS,
};
use math_explorer::physics::mri::bloch::BlochSimulator;
use math_explorer::physics::mri::proton;
use math_explorer::physics::mri::reconstruction::Fft2dWorkspace;
use math_explorer::physics::mri::scanner::{accumulated_phase, k_space_coordinate};
use math_explorer::pure_math::analysis::ode::{RungeKutta4, TimeStepper};
use nalgebra::{DMatrix, Vector3};
use num_complex::Complex;
use std::collections::VecDeque;
use std::f64::consts::PI;
use std::sync::Arc;

/// Rotating-frame $\Delta B_0$ pocket amplitude in Tesla (`0.22 uT`, $\gamma \Delta B_0 \approx 58.9\,\text{rad/s}$).
pub(crate) const DELTA_B0_POCKET_TESLA: f64 = 2.2e-7;
/// Sub-voxel inhomogeneity spread fraction driving reversible $T_2^*$ dephasing.
pub(crate) const ISOCHROMAT_SPREAD_FRAC: f64 = 0.85;
/// Worker integration step `dt` in seconds (`2 ms`).
pub(crate) const BLOCH_STEP_DT: f64 = 0.002;

/// Zero-allocation 2D MRI k-space, RF pulse, and spatial Bloch worker runner.
pub struct MriKSpaceRunner {
    /// Shared configuration handle synchronized with UI controls.
    pub params: Arc<std::sync::RwLock<MriKSpaceSharedParams>>,
    pub(crate) size: usize,
    pub(crate) active_phantom: PhantomPreset,
    pub(crate) active_mask_preset: KSpaceMaskPreset,
    pub(crate) cached_sequence_key: [u64; 5],
    pub(crate) cached_phasor_key: Option<(u64, u64, [usize; 2])>,
    pub(crate) current_time: f64,
    pub(crate) echo_tau: Option<f64>,
    pub(crate) k_coord_cycles_per_m: [f64; 2],
    pub(crate) selected_grad_phase_rad: f64,
    pub(crate) phantom_engine: PhantomEngine,
    pub(crate) fft_workspace: Fft2dWorkspace,
    pub(crate) bloch_scratch: BlochSimulator,
    pub(crate) rk4_solver: RungeKutta4<Vector3<f64>>,
    pub(crate) rho: Vec<f64>,
    pub(crate) t1: Vec<f64>,
    pub(crate) t2: Vec<f64>,
    pub(crate) exp_t1: Vec<f64>,
    pub(crate) exp_t2: Vec<f64>,
    pub(crate) delta_b0: Vec<f64>,
    pub(crate) local_flip_rad: Vec<f64>,
    pub(crate) kspace_mask: Vec<f64>,
    pub(crate) kspace_spikes: Vec<Complex<f64>>,
    pub(crate) dft_mask: Vec<f64>,
    pub(crate) dft_spikes: Vec<Complex<f64>>,
    pub(crate) is_identity_kspace: bool,
    pub(crate) spins: Vec<[Vector3<f64>; ISOCHROMAT_COUNT]>,
    pub(crate) phasor_x: Vec<Complex<f64>>,
    pub(crate) phasor_y: Vec<Complex<f64>>,
    pub(crate) transverse_grid: DMatrix<Complex<f64>>,
    pub(crate) masked_kspace_rm: Vec<Complex<f64>>,
    pub(crate) recon_grid: DMatrix<Complex<f64>>,
    pub(crate) recon_mag: Vec<f64>,
    pub(crate) selected_trajectory: VecDeque<Vector3<f64>>,
}

impl MriKSpaceRunner {
    /// Constructs a pre-allocated 2D MRI k-space runner ready for zero-allocation stepping.
    pub fn new(params: Arc<std::sync::RwLock<MriKSpaceSharedParams>>) -> Self {
        let cfg = params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .config
            .clone();
        let size = cfg.grid_size;
        let total = size * size;
        let mut traj = VecDeque::with_capacity(MAX_TRAJECTORY_POINTS);
        traj.push_back(Vector3::zeros());

        let mut runner = Self {
            params,
            size,
            active_phantom: cfg.phantom,
            active_mask_preset: cfg.mask_preset,
            cached_sequence_key: [0; 5],
            cached_phasor_key: None,
            current_time: 0.0,
            echo_tau: None,
            k_coord_cycles_per_m: [0.0, 0.0],
            selected_grad_phase_rad: 0.0,
            phantom_engine: PhantomEngine::new(size, cfg.phantom),
            fft_workspace: Fft2dWorkspace::new(size, size),
            bloch_scratch: BlochSimulator::new(Vector3::zeros(), 1.0),
            rk4_solver: RungeKutta4::new(&Vector3::zeros()),
            rho: vec![0.0; total],
            t1: vec![1.0; total],
            t2: vec![0.1; total],
            exp_t1: vec![1.0; total],
            exp_t2: vec![1.0; total],
            delta_b0: vec![0.0; total],
            local_flip_rad: vec![0.0; total],
            kspace_mask: vec![1.0; total],
            kspace_spikes: vec![Complex::new(0.0, 0.0); total],
            dft_mask: vec![1.0; total],
            dft_spikes: vec![Complex::new(0.0, 0.0); total],
            is_identity_kspace: true,
            spins: vec![[Vector3::zeros(); ISOCHROMAT_COUNT]; total],
            phasor_x: vec![Complex::new(1.0, 0.0); size],
            phasor_y: vec![Complex::new(1.0, 0.0); size],
            transverse_grid: DMatrix::zeros(size, size),
            masked_kspace_rm: vec![Complex::new(0.0, 0.0); total],
            recon_grid: DMatrix::zeros(size, size),
            recon_mag: vec![0.0; total],
            selected_trajectory: traj,
        };
        runner.rebuild_from_config(&cfg);
        runner
    }

    pub(crate) fn rebuild_from_config(&mut self, cfg: &MriKSpaceToolState) {
        if self.size != cfg.grid_size {
            self.resize_buffers(cfg.grid_size);
        }
        self.active_phantom = cfg.phantom;
        self.active_mask_preset = cfg.mask_preset;
        self.phantom_engine.rebuild(self.size, cfg.phantom);
        self.phantom_engine
            .write_tissues_into(&mut self.rho, &mut self.t1, &mut self.t2);
        self.refresh_relaxation_factors();
        self.cached_phasor_key = None;
        self.delta_b0.fill(0.0);
        self.local_flip_rad.fill(0.0);
        self.kspace_spikes.fill(Complex::new(0.0, 0.0));
        build_preset_mask_into(
            cfg.mask_preset,
            self.size,
            cfg.mask_cutoff,
            cfg.radial_spokes,
            cfg.cartesian_stride,
            &mut self.kspace_mask,
        );
        for stroke in &cfg.strokes {
            if let Ok((mode, cx, cy, r)) = stroke.validate(self.size) {
                self.apply_stroke_direct(mode, cx, cy, r, false);
            }
        }
        self.sync_dft_kspace_tables();
        self.excite_to_initial_echo(cfg);
    }

    fn resize_buffers(&mut self, new_size: usize) {
        let total = new_size * new_size;
        self.size = new_size;
        self.fft_workspace.resize(new_size, new_size);
        self.rho.resize(total, 0.0);
        self.t1.resize(total, 1.0);
        self.t2.resize(total, 0.1);
        self.exp_t1.resize(total, 1.0);
        self.exp_t2.resize(total, 1.0);
        self.delta_b0.resize(total, 0.0);
        self.local_flip_rad.resize(total, 0.0);
        self.kspace_mask.resize(total, 1.0);
        self.kspace_spikes.resize(total, Complex::new(0.0, 0.0));
        self.dft_mask.resize(total, 1.0);
        self.dft_spikes.resize(total, Complex::new(0.0, 0.0));
        self.spins
            .resize(total, [Vector3::zeros(); ISOCHROMAT_COUNT]);
        self.phasor_x.resize(new_size, Complex::new(1.0, 0.0));
        self.phasor_y.resize(new_size, Complex::new(1.0, 0.0));
        self.transverse_grid = DMatrix::zeros(new_size, new_size);
        self.masked_kspace_rm.resize(total, Complex::new(0.0, 0.0));
        self.recon_grid = DMatrix::zeros(new_size, new_size);
        self.recon_mag.resize(total, 0.0);
    }

    pub(crate) fn excite_to_initial_echo(&mut self, cfg: &MriKSpaceToolState) {
        let base_flip = cfg.flip_angle_deg * (PI / 180.0);
        let tr = cfg.tr;
        let te = cfg.te;
        let gamma = proton::GYROMAGNETIC_RATIO;

        for idx in 0..(self.size * self.size) {
            let m0 = self.rho[idx];
            let t1 = self.t1[idx].max(1e-4);
            let t2 = self.t2[idx].max(1e-4);
            let mz_pre = m0 * (1.0 - (-tr / t1).exp());
            let alpha = base_flip + self.local_flip_rad[idx];
            let m_xy0 = mz_pre * alpha.sin();
            let mz0 = mz_pre * alpha.cos();
            let e_t2 = (-te / t2).exp();
            let e_t1 = (-te / t1).exp();
            let mz_te = m0 + (mz0 - m0) * e_t1;
            let b0 = self.delta_b0[idx];
            let spread = ISOCHROMAT_SPREAD_FRAC * b0.abs();

            for s in 0..ISOCHROMAT_COUNT {
                let bz = b0 + ((s as f64) - 1.0) * spread;
                let theta = gamma * bz * te;
                self.spins[idx][s] = Vector3::new(
                    m_xy0 * e_t2 * theta.sin(),
                    m_xy0 * e_t2 * theta.cos(),
                    mz_te,
                );
            }
        }
        self.current_time = te;
        self.echo_tau = None;
        self.cached_sequence_key = Self::sequence_key(cfg);
        self.selected_trajectory.clear();
        let sel_m = self.net_voxel_spin(cfg.selected_voxel);
        self.selected_trajectory.push_back(sel_m);
        self.reconstruct_in_place(
            cfg.grad_gx_integral,
            cfg.grad_gy_integral,
            cfg.selected_voxel,
        );
    }

    fn sequence_key(cfg: &MriKSpaceToolState) -> [u64; 5] {
        [
            cfg.flip_angle_deg.to_bits(),
            cfg.tr.to_bits(),
            cfg.te.to_bits(),
            cfg.mask_cutoff.to_bits(),
            ((cfg.radial_spokes as u64) << 16) | (cfg.cartesian_stride as u64),
        ]
    }

    /// Applies an instantaneous $180^\circ_y$ Hahn/CPMG spin-echo refocusing pulse to all voxel spins.
    pub fn apply_refocus_180_pulse(&mut self, gx: f64, gy: f64, selected_voxel: [usize; 2]) {
        self.echo_tau = Some(self.current_time.max(BLOCH_STEP_DT));
        for ensemble in &mut self.spins {
            for spin in ensemble {
                spin.x = -spin.x;
                spin.z = -spin.z;
            }
        }
        self.record_trajectory_point(selected_voxel);
        self.reconstruct_in_place(gx, gy, selected_voxel);
    }

    pub(crate) fn refresh_relaxation_factors(&mut self) {
        for idx in 0..self.t1.len() {
            self.exp_t1[idx] = (-BLOCH_STEP_DT / self.t1[idx].max(1e-4)).exp();
            self.exp_t2[idx] = (-BLOCH_STEP_DT / self.t2[idx].max(1e-4)).exp();
        }
    }

    /// Maps centered `(R/2, C/2)` row-major mask and spike tables into transposed DFT order `dc * n + dr`.
    pub(crate) fn sync_dft_kspace_tables(&mut self) {
        let n = self.size;
        let shift = n - (n / 2);
        let mut identity = true;
        for r in 0..n {
            let dr = (r + shift) % n;
            let src_off = r * n;
            for c in 0..n {
                let dc = (c + shift) % n;
                let m = self.kspace_mask[src_off + c];
                let sp = self.kspace_spikes[src_off + c];
                if (m - 1.0).abs() > 1e-12 || sp.re != 0.0 || sp.im != 0.0 {
                    identity = false;
                }
                let dst_idx = dc * n + dr;
                self.dft_mask[dst_idx] = m;
                self.dft_spikes[dst_idx] = sp;
            }
        }
        self.is_identity_kspace = identity;
    }

    pub(crate) fn advance_spatial_bloch(&mut self, dt: f64, selected_voxel: [usize; 2]) {
        let sel_r = selected_voxel[0].min(self.size.saturating_sub(1));
        let sel_c = selected_voxel[1].min(self.size.saturating_sub(1));
        let sel_idx = sel_r * self.size + sel_c;
        self.bloch_scratch.m0 = self.rho[sel_idx];
        self.bloch_scratch.t1 = self.t1[sel_idx].max(1e-4);
        self.bloch_scratch.t2 = self.t2[sel_idx].max(1e-4);
        self.bloch_scratch.b_field = Vector3::new(0.0, 0.0, self.delta_b0[sel_idx]);
        self.bloch_scratch.magnetization = self.spins[sel_idx][1];
        TimeStepper::step_with(&mut self.bloch_scratch, &mut self.rk4_solver, dt);

        self.bloch_scratch.advance_isochromat_grid(
            &mut self.spins,
            &self.rho,
            &self.exp_t1,
            &self.exp_t2,
            &self.delta_b0,
            ISOCHROMAT_SPREAD_FRAC,
            dt,
        );
        self.current_time += dt;
    }

    fn update_scanner_phasors(
        &mut self,
        grad_gx_integral: f64,
        grad_gy_integral: f64,
        selected_voxel: [usize; 2],
    ) {
        let key = (
            grad_gx_integral.to_bits(),
            grad_gy_integral.to_bits(),
            selected_voxel,
        );
        if self.cached_phasor_key == Some(key) {
            return;
        }
        self.cached_phasor_key = Some(key);
        let n = self.size;
        let grad_vec = Vector3::new(grad_gx_integral, grad_gy_integral, 0.0);
        let k_vec = k_space_coordinate(grad_vec);
        self.k_coord_cycles_per_m = [k_vec.x, k_vec.y];

        let inv_span = if n > 1 {
            STUDIO_FOV_METERS / ((n - 1) as f64)
        } else {
            0.0
        };
        let half_fov = 0.5 * STUDIO_FOV_METERS;
        for i in 0..n {
            let coord = -half_fov + (i as f64) * inv_span;
            let px = accumulated_phase(Vector3::new(coord, 0.0, 0.0), grad_vec);
            let py = accumulated_phase(Vector3::new(0.0, -coord, 0.0), grad_vec);
            self.phasor_x[i] = Complex::from_polar(1.0, -px);
            self.phasor_y[i] = Complex::from_polar(1.0, -py);
        }
        let sel_r = selected_voxel[0].min(n.saturating_sub(1));
        let sel_c = selected_voxel[1].min(n.saturating_sub(1));
        let sel_pos = Vector3::new(
            -half_fov + (sel_c as f64) * inv_span,
            half_fov - (sel_r as f64) * inv_span,
            0.0,
        );
        self.selected_grad_phase_rad = accumulated_phase(sel_pos, grad_vec);
    }

    pub(crate) fn reconstruct_in_place(
        &mut self,
        grad_gx_integral: f64,
        grad_gy_integral: f64,
        selected_voxel: [usize; 2],
    ) {
        self.update_scanner_phasors(grad_gx_integral, grad_gy_integral, selected_voxel);
        let zero_grad = grad_gx_integral == 0.0 && grad_gy_integral == 0.0;
        self.fill_transverse_grid(zero_grad);

        if self.is_identity_kspace {
            self.fft_workspace
                .forward_transposed_in_place(&mut self.transverse_grid, &mut self.masked_kspace_rm);
            return;
        }

        self.fft_workspace.filter_kspace_2d_into(
            &self.transverse_grid,
            &self.dft_mask,
            &self.dft_spikes,
            false,
            &mut self.masked_kspace_rm,
            &mut self.recon_grid,
        );
        for (mag, z) in self.recon_mag.iter_mut().zip(self.recon_grid.as_slice()) {
            *mag = (z.re * z.re + z.im * z.im).sqrt();
        }
    }

    fn fill_transverse_grid(&mut self, zero_grad: bool) {
        let n = self.size;
        let inv_iso = 1.0 / (ISOCHROMAT_COUNT as f64);
        if zero_grad && self.is_identity_kspace {
            for ((ens, out), mag) in self
                .spins
                .iter()
                .zip(self.transverse_grid.as_mut_slice().iter_mut())
                .zip(self.recon_mag.iter_mut())
            {
                let mx = (ens[0].x + ens[1].x + ens[2].x) * inv_iso;
                let my = (ens[0].y + ens[1].y + ens[2].y) * inv_iso;
                *out = Complex::new(mx, my);
                *mag = (mx * mx + my * my).sqrt();
            }
            return;
        }
        for ((row_spins, row_mag), (row_out, &py)) in self
            .spins
            .chunks_exact(n)
            .zip(self.recon_mag.chunks_exact_mut(n))
            .zip(
                self.transverse_grid
                    .as_mut_slice()
                    .chunks_exact_mut(n)
                    .zip(&self.phasor_y),
            )
        {
            for (((ens, mag), out), &px) in row_spins
                .iter()
                .zip(row_mag.iter_mut())
                .zip(row_out.iter_mut())
                .zip(&self.phasor_x)
            {
                let mx = (ens[0].x + ens[1].x + ens[2].x) * inv_iso;
                let my = (ens[0].y + ens[1].y + ens[2].y) * inv_iso;
                let z = if zero_grad {
                    Complex::new(mx, my)
                } else {
                    Complex::new(mx, my) * (px * py)
                };
                *out = z;
                if self.is_identity_kspace {
                    *mag = (mx * mx + my * my).sqrt();
                }
            }
        }
    }

    pub(crate) fn net_voxel_spin(&self, voxel: [usize; 2]) -> Vector3<f64> {
        let r = voxel[0].min(self.size.saturating_sub(1));
        let c = voxel[1].min(self.size.saturating_sub(1));
        let ens = &self.spins[r * self.size + c];
        (ens[0] + ens[1] + ens[2]) / (ISOCHROMAT_COUNT as f64)
    }

    pub(crate) fn record_trajectory_point(&mut self, voxel: [usize; 2]) {
        let net_m = self.net_voxel_spin(voxel);
        if self.selected_trajectory.len() >= MAX_TRAJECTORY_POINTS {
            self.selected_trajectory.pop_front();
        }
        self.selected_trajectory.push_back(net_m);
    }

    pub(crate) fn sync_zero_alloc_step_params(&mut self) -> (bool, f64, f64, [usize; 2]) {
        let guard = self.params.read().unwrap_or_else(|e| e.into_inner());
        (
            guard.config.evolve_turing,
            guard.config.grad_gx_integral,
            guard.config.grad_gy_integral,
            guard.config.selected_voxel,
        )
    }
}
