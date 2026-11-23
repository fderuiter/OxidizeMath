#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Lazy zero-allocation propagation and phase-holography evaluation cache for [`super::FourierOpticsWorkbenchTool`].

// [cite:graph_parameters_rust]

use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::num_complex::Complex;
use math_explorer::physics::optics::diffraction::{
    AngularSpectrumPropagator, SlitApertureConfig, apply_thin_lens_phase_into,
    apply_zernike_aberration_into, field_intensity_into, fill_circular_pupil_into,
    fill_opaque_disk_aperture_into, fill_ronchi_grating_into, fill_slit_aperture_into,
    fresnel_number, optical_power, talbot_length, wrapped_phase_into,
};
use math_explorer::physics::optics::holography::{
    GerchbergSaxtonSolver, HologramMetrics, HologramRegime, fill_target_amplitude_into,
};
use oxidize_core::rng::OxidizeRng;

use super::{
    AperturePreset, CARPET_STEPS, DIFF_GRID, FourierOpticsState, FourierOpticsWorkbenchTool,
    GS_GRID, OpticsDisplayMode, OpticsMode,
};

pub(crate) struct WorkbenchCache {
    pub propagator: AngularSpectrumPropagator,
    pub aperture_field: Vec<Complex<f64>>,
    pub propagated_field: Vec<Complex<f64>>,
    pub diff_intensity: Vec<f64>,
    pub diff_phase: Vec<f64>,
    pub axial_carpet: Vec<f64>,
    pub carpet_dirty_state: Option<FourierOpticsState>,
    pub gs_solver: GerchbergSaxtonSolver,
    pub gs_target_buf: Vec<f64>,
    pub gs_replay_intensity: Vec<f64>,
    pub gs_slm_u8: Vec<u8>,
    pub gs_convergence: Vec<[f64; 2]>,
    pub gs_metrics: HologramMetrics,
    pub real_scalar_metrics: HologramMetrics,
    pub evaluated_state: Option<FourierOpticsState>,
    pub power_in: f64,
    pub power_out: f64,
    pub center_intensity: f64,
    pub max_intensity: f64,
    pub fresnel_nf: f64,
    pub talbot_mm: f64,
}

impl WorkbenchCache {
    pub(crate) fn allocate(state: &FourierOpticsState) -> Self {
        let dx = state.pixel_pitch_um * 1.0e-6;
        let wl = state.wavelength_nm * 1.0e-9;
        let diff_n = DIFF_GRID * DIFF_GRID;
        let gs_n = GS_GRID * GS_GRID;
        Self {
            propagator: AngularSpectrumPropagator::new(DIFF_GRID, DIFF_GRID, dx, dx, wl),
            aperture_field: vec![Complex::new(0.0, 0.0); diff_n],
            propagated_field: vec![Complex::new(0.0, 0.0); diff_n],
            diff_intensity: vec![0.0; diff_n],
            diff_phase: vec![0.0; diff_n],
            axial_carpet: vec![0.0; DIFF_GRID * CARPET_STEPS],
            carpet_dirty_state: None,
            gs_solver: GerchbergSaxtonSolver::new(
                GS_GRID,
                GS_GRID,
                dx,
                dx,
                wl,
                HologramRegime::FourierLens,
            ),
            gs_target_buf: vec![0.0; gs_n],
            gs_replay_intensity: vec![0.0; gs_n],
            gs_slm_u8: vec![0; gs_n],
            gs_convergence: Vec::with_capacity(101),
            gs_metrics: HologramMetrics::default(),
            real_scalar_metrics: HologramMetrics::default(),
            evaluated_state: None,
            power_in: 0.0,
            power_out: 0.0,
            center_intensity: 0.0,
            max_intensity: 1.0,
            fresnel_nf: 0.0,
            talbot_mm: 0.0,
        }
    }
}

impl FourierOpticsWorkbenchTool {
    pub(crate) fn clamp_state_in_place(&mut self) {
        let s = &mut self.state;
        s.wavelength_nm = s.wavelength_nm.clamp(380.0, 1550.0);
        s.pixel_pitch_um = s.pixel_pitch_um.clamp(2.0, 25.0);
        s.propagation_mm = s.propagation_mm.clamp(-150.0, 150.0);
        s.carpet_max_mm = s.carpet_max_mm.clamp(1.0, 200.0);
        s.feature_scale_um = s.feature_scale_um.clamp(16.0, 320.0);
        s.slit_pitch_um = s.slit_pitch_um.clamp(16.0, 400.0);
        s.slit_count = s.slit_count.clamp(1, 7);
        s.duty_cycle = s.duty_cycle.clamp(0.1, 0.9);
        s.focal_length_mm = s.focal_length_mm.clamp(2.0, 120.0);
        s.zernike_n = s.zernike_n.min(6);
        let n_i32 = s.zernike_n as i32;
        s.zernike_m = s.zernike_m.clamp(-n_i32, n_i32);
        if (n_i32 - s.zernike_m.abs()) % 2 != 0 {
            s.zernike_m = if s.zernike_m >= 0 {
                (s.zernike_m - 1).max(-n_i32)
            } else {
                (s.zernike_m + 1).min(n_i32)
            };
        }
        s.zernike_rms_waves = s.zernike_rms_waves.clamp(-2.0, 2.0);
        s.gs_iterations = s.gs_iterations.clamp(1, 100);
    }

    pub(crate) fn ensure_evaluated(&self) {
        let mut guard = self.cache.borrow_mut();
        let cache = guard.get_or_insert_with(|| WorkbenchCache::allocate(&self.state));
        if cache.evaluated_state.as_ref() == Some(&self.state) {
            if self.state.mode == OpticsMode::HelmholtzDiffraction
                && self.state.display == OpticsDisplayMode::AxialCarpetXZ
                && cache.carpet_dirty_state.as_ref() != Some(&self.state)
            {
                Self::evaluate_carpet(&self.state, cache);
            }
            return;
        }
        match self.state.mode {
            OpticsMode::HelmholtzDiffraction => Self::evaluate_diffraction(&self.state, cache),
            OpticsMode::GerchbergSaxtonHolography => Self::evaluate_holography(&self.state, cache),
        }
        cache.evaluated_state = Some(self.state.clone());
    }

    fn build_aperture_field(s: &FourierOpticsState, out: &mut [Complex<f64>]) {
        let dx = s.pixel_pitch_um * 1.0e-6;
        let wl = s.wavelength_nm * 1.0e-9;
        let scale_m = s.feature_scale_um * 1.0e-6;
        match s.aperture {
            AperturePreset::OpaqueCircularDisk => {
                fill_opaque_disk_aperture_into(out, DIFF_GRID, DIFF_GRID, dx, dx, scale_m, 0.0);
            }
            AperturePreset::SingleMultiSlit => {
                fill_slit_aperture_into(
                    out,
                    DIFF_GRID,
                    DIFF_GRID,
                    dx,
                    dx,
                    SlitApertureConfig {
                        slit_count: s.slit_count,
                        slit_width: scale_m,
                        slit_pitch: s.slit_pitch_um * 1.0e-6,
                        slit_height: 0.72 * (DIFF_GRID as f64) * dx,
                    },
                );
            }
            AperturePreset::RonchiGrating => {
                fill_ronchi_grating_into(out, DIFF_GRID, DIFF_GRID, dx, scale_m, s.duty_cycle, false);
            }
            AperturePreset::ThinQuadraticLens => {
                fill_circular_pupil_into(out, DIFF_GRID, DIFF_GRID, dx, dx, scale_m);
                apply_thin_lens_phase_into(
                    out,
                    DIFF_GRID,
                    DIFF_GRID,
                    dx,
                    dx,
                    wl,
                    s.focal_length_mm * 1.0e-3,
                    scale_m,
                );
            }
        }
        if s.zernike_rms_waves.abs() > 1.0e-9 {
            let pupil_r = (0.45 * (DIFF_GRID as f64) * dx).min(scale_m.max(80.0e-6));
            apply_zernike_aberration_into(
                out,
                DIFF_GRID,
                DIFF_GRID,
                dx,
                dx,
                pupil_r,
                s.zernike_n as usize,
                s.zernike_m as isize,
                s.zernike_rms_waves,
            );
        }
    }

    fn evaluate_diffraction(s: &FourierOpticsState, cache: &mut WorkbenchCache) {
        let dx = s.pixel_pitch_um * 1.0e-6;
        let wl = s.wavelength_nm * 1.0e-9;
        let z_m = s.propagation_mm * 1.0e-3;
        let scale_m = s.feature_scale_um * 1.0e-6;
        cache.propagator.reconfigure(dx, dx, wl);
        Self::build_aperture_field(s, &mut cache.aperture_field);

        cache.power_in = optical_power(&cache.aperture_field, dx, dx);
        cache
            .propagator
            .propagate_into(&cache.aperture_field, z_m, &mut cache.propagated_field);
        cache.power_out = optical_power(&cache.propagated_field, dx, dx);
        field_intensity_into(&cache.propagated_field, &mut cache.diff_intensity);
        wrapped_phase_into(&cache.propagated_field, &mut cache.diff_phase);

        let center = flatten_2d_index(DIFF_GRID / 2, DIFF_GRID / 2, DIFF_GRID);
        cache.center_intensity = cache.diff_intensity[center];
        cache.max_intensity = cache
            .diff_intensity
            .iter()
            .copied()
            .fold(1.0e-12_f64, f64::max);
        cache.fresnel_nf = fresnel_number(scale_m, wl, z_m);
        cache.talbot_mm = talbot_length(scale_m, wl) * 1.0e3;

        if s.display == OpticsDisplayMode::AxialCarpetXZ {
            Self::evaluate_carpet(s, cache);
        }
    }

    fn evaluate_carpet(s: &FourierOpticsState, cache: &mut WorkbenchCache) {
        cache.propagator.propagate_axial_carpet_into(
            &cache.aperture_field,
            DIFF_GRID / 2,
            0.0,
            s.carpet_max_mm * 1.0e-3,
            CARPET_STEPS,
            &mut cache.axial_carpet,
        );
        cache.carpet_dirty_state = Some(s.clone());
    }

    fn evaluate_holography(s: &FourierOpticsState, cache: &mut WorkbenchCache) {
        let dx = s.pixel_pitch_um * 1.0e-6;
        let wl = s.wavelength_nm * 1.0e-9;
        let regime = if s.gs_lensless {
            HologramRegime::FresnelAngularSpectrum {
                distance_z: s.propagation_mm.abs().max(0.5) * 1.0e-3,
            }
        } else {
            HologramRegime::FourierLens
        };
        cache.gs_solver.reconfigure(dx, dx, wl, regime);
        fill_target_amplitude_into(s.gs_target, GS_GRID, GS_GRID, &mut cache.gs_target_buf);
        cache.gs_solver.set_target_amplitude(&cache.gs_target_buf);
        cache.real_scalar_metrics = cache.gs_solver.evaluate_real_scalar_baseline();

        let mut rng = OxidizeRng::from_seed(s.gs_seed);
        cache.gs_solver.seed_random_phase(&mut rng);
        cache.gs_convergence.clear();
        cache
            .gs_convergence
            .push([0.0, cache.gs_solver.metrics().target_nrmse]);
        for iter in 1..=s.gs_iterations {
            let m = cache.gs_solver.step_iterations(1);
            cache.gs_convergence.push([iter as f64, m.target_nrmse]);
        }
        cache.gs_metrics = cache.gs_solver.metrics();
        cache.gs_solver.quantize_phase_u8_into(&mut cache.gs_slm_u8);
        field_intensity_into(cache.gs_solver.replay_field(), &mut cache.gs_replay_intensity);
        cache.max_intensity = cache
            .gs_replay_intensity
            .iter()
            .copied()
            .fold(1.0e-12_f64, f64::max);
    }
}
