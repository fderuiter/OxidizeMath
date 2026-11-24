#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Domain mask assembly, physics stepping, and spectral/contour analysis refresh for [`super::ChladniWaveTankTool`].

use math_explorer::pure_math::analysis::pde::wave_2d::{BoundaryMode2D, Wave2DSolver};

use super::{eigenmodes, ir_export, presets, sculpt, ChladniWaveTankTool, TankPreset, GRID_SIZE};

pub(crate) fn max_mask_speed(mask: &[f64]) -> f64 {
    mask.iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(1.0_f64, f64::max)
}

#[derive(Default)]
pub(crate) struct WorkbenchMaskContext<'a> {
    pub refractive: Option<&'a super::refractive::RefractiveSandbox>,
    #[cfg(feature = "physics")]
    pub metamaterial: Option<&'a super::metamaterial::MetamaterialWorkbench>,
    pub sculpt_mask: &'a [u8],
}

impl WorkbenchMaskContext<'_> {
    pub(crate) fn apply_to_solver(
        &self,
        solver: &mut Wave2DSolver,
        state: &super::ChladniWaveTankState,
    ) {
        let (w, h) = (solver.width(), solver.height());
        ir_export::configure_solver_masks(solver, state, self.refractive);
        #[cfg(feature = "physics")]
        if let Some(meta) = self.metamaterial {
            meta.apply_ast_to_wave_speed_mask(&mut solver.wave_speed_mask, w, h);
        }
        let _ = (w, h);
        sculpt::replay_sculpt_mask_on_solver(self.sculpt_mask, solver);
    }
}

impl ChladniWaveTankTool {
    pub(crate) fn apply_preset(&mut self, preset: TankPreset) {
        self.sculpt_mask.fill(0);
        self.state = presets::default_state_for_preset(preset);
        self.active_mode = eigenmodes::default_mode_for_preset(preset);
        self.refractive.sync_from_state(&self.state);
        self.state_rev = self.state_rev.wrapping_add(1);
        let warmup = match preset {
            TankPreset::DoubleSlitInterference | TankPreset::AcousticLens => 24,
            TankPreset::SquareChladniPlate | TankPreset::CircularDrumhead => 6,
        };
        self.rebuild_domain_and_warmup(warmup);
    }

    pub(crate) fn rebuild_domain_and_warmup(&mut self, warmup_steps: usize) {
        let cell_spacing = if self.state.barrier_enabled
            || self.state.lens_enabled
            || self.state.boundary == BoundaryMode2D::Sponge
        {
            1.0
        } else {
            1.0 / ((GRID_SIZE - 1) as f64)
        };
        if (self.solver.dx() - cell_spacing).abs() > 1.0e-12 {
            self.solver = Wave2DSolver::new(GRID_SIZE, GRID_SIZE, cell_spacing, cell_spacing);
        } else {
            self.solver.reset_all();
        }
        self.rebuild_masks();
        eigenmodes::seed_eigenmode_field(
            &mut self.solver,
            &self.state,
            self.active_mode.0,
            self.active_mode.1,
        );
        self.sand.reseed(
            self.state.rng_seed,
            GRID_SIZE,
            GRID_SIZE,
            &self.solver.wall_mask,
        );
        if warmup_steps > 0 {
            for _ in 0..warmup_steps {
                self.advance_physics_step();
            }
            self.refresh_analysis_outputs();
        } else {
            self.analysis_dirty = true;
        }
    }

    pub(super) fn prepare_studio_ir(&mut self, total_samples: usize) {
        let ctx = WorkbenchMaskContext {
            refractive: Some(&self.refractive),
            #[cfg(feature = "physics")]
            metamaterial: Some(&self.metamaterial),
            sculpt_mask: &self.sculpt_mask,
        };
        self.ir_studio
            .prepare_ir_capture(&self.state, &ctx, total_samples);
    }

    pub(crate) fn rebuild_masks(&mut self) {
        let ctx = WorkbenchMaskContext {
            refractive: Some(&self.refractive),
            #[cfg(feature = "physics")]
            metamaterial: Some(&self.metamaterial),
            sculpt_mask: &self.sculpt_mask,
        };
        ctx.apply_to_solver(&mut self.solver, &self.state);
    }

    pub(crate) fn advance_physics_step(&mut self) {
        let start = crate::web_io::PlatformInstant::now();
        let params = self.pde_params();
        let driver = self.driver_config();
        let det_col = Some(self.state.detector_x.min(GRID_SIZE - 2));
        self.last_substeps =
            self.solver
                .step_driven(&params, &driver, self.state.time_step, det_col);
        if self.state.show_sand_particles {
            self.sand.step(
                self.solver.energy_field(),
                &self.solver.wall_mask,
                GRID_SIZE,
                GRID_SIZE,
                14.0,
                0.30,
                0.15,
            );
        }
        self.last_step_ms = start.elapsed().as_secs_f64() * 1_000.0;
    }

    pub(crate) fn refresh_analysis_outputs(&mut self) {
        self.analysis_dirty = false;
        self.latest_probe = self
            .solver
            .sample_probe(self.state.pickup_x, self.state.pickup_y);
        self.solver
            .read_detector_profile(self.state.detector_x, &mut self.detector_buf);
        self.fft.compute_spectrum(self.solver.displacement());
        self.active_peak_count = self.fft.top_peaks(&mut self.modal_peaks);
        self.dominant_peak = self.modal_peaks[0];
        self.fft
            .compute_radial_profile(&mut self.radial_spectrum, &mut self.radial_counts);
        self.update_sonification_frame();
        if self.state.show_nodal_contours {
            let field = if self.state.driver_enabled && !self.state.line_source {
                self.solver.modal_field()
            } else {
                self.solver.displacement()
            };
            self.contours.extract_zero_contours(
                field,
                &self.solver.wall_mask,
                GRID_SIZE,
                GRID_SIZE,
                1.0e-5,
            );
        }
    }
}
