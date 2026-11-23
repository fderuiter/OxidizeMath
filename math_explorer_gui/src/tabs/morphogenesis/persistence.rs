//! Runner construction, RLE/fork state persistence, and raster export helpers for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::counterfactual::MorphogenesisHeatmapMode;
use super::runner::{
    MorphogenesisKineticsState, MorphogenesisRunner, MorphogenesisSharedParams,
    DEFAULT_MORPHOGENESIS_SEED,
};
use super::state::{initialize_for_kinetics, MorphogenesisToolState};
use super::MorphogenesisTool;
use crate::async_sim::{BackgroundWorker, SculptStatePayload, SimCommand, SimWorkerState};
use crate::framework::{decode_rle_mask, InteractiveTool};
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{
    DiffusionCoeff, GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
use math_explorer::math_kernel::types::{Dimension, StepSize};
use scientific_metadata::theory::TheoryDescribable;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

impl MorphogenesisTool {
    #[cfg(test)]
    pub(crate) fn create_masked_runner(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
        width: usize,
        height: usize,
    ) -> MorphogenesisRunner {
        Self::build_masked_runner(params, None, mask_rle, width, height)
    }

    pub(crate) fn create_kinetics_runner(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        kinetics_state: &Arc<std::sync::RwLock<MorphogenesisKineticsState>>,
        mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
        width: usize,
        height: usize,
    ) -> MorphogenesisRunner {
        Self::build_masked_runner(params, Some(kinetics_state), mask_rle, width, height)
    }

    fn initial_kinetics_sync(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        kinetics_state: Option<&Arc<std::sync::RwLock<MorphogenesisKineticsState>>>,
        p: &mut MorphogenesisSharedParams,
    ) -> MorphogenesisKineticsState {
        let Some(ks_arc) = kinetics_state else {
            return MorphogenesisKineticsState {
                mode: p.kinetics,
                f: p.f,
                k: p.k,
            };
        };
        let ks = *ks_arc.read().unwrap_or_else(|e| e.into_inner());
        let def = MorphogenesisKineticsState::default();
        if ks != def
            && p.kinetics == def.mode
            && p.f.to_bits() == def.f.to_bits()
            && p.k.to_bits() == def.k.to_bits()
        {
            p.kinetics = ks.mode;
            p.f = ks.f;
            p.k = ks.k;
            if let Ok(mut wg) = params.try_write() {
                wg.kinetics = ks.mode;
                wg.f = ks.f;
                wg.k = ks.k;
            }
            ks
        } else {
            let next_ks = MorphogenesisKineticsState {
                mode: p.kinetics,
                f: p.f,
                k: p.k,
            };
            if let Ok(mut wg) = ks_arc.try_write() {
                *wg = next_ks;
            }
            next_ks
        }
    }

    fn build_masked_runner(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        kinetics_state: Option<&Arc<std::sync::RwLock<MorphogenesisKineticsState>>>,
        mask_rle: &Arc<std::sync::RwLock<Vec<[u32; 2]>>>,
        width: usize,
        height: usize,
    ) -> MorphogenesisRunner {
        let mut p = *params.read().unwrap_or_else(|e| e.into_inner());
        let synced_ks = Self::initial_kinetics_sync(params, kinetics_state, &mut p);
        let kinetics = SchnakenbergKinetics { a: p.a, b: p.b };
        let gray_scott = GrayScottKinetics::new(p.f, p.k);
        let diffusion = FiniteDifference2D::new(
            Dimension(width),
            Dimension(height),
            StepSize(1.0),
            StepSize(1.0),
        )
        .with_mask();
        let mut system = TuringSystem::new_with_kinetics(
            Dimension(width * height),
            DiffusionCoeff(p.d_u),
            DiffusionCoeff(p.d_v),
            kinetics,
            diffusion,
        );
        initialize_for_kinetics(&mut system, p.kinetics, width, height);
        let rle = mask_rle.read().unwrap_or_else(|e| e.into_inner()).clone();
        if !rle.is_empty() {
            if let Ok(bytes) = decode_rle_mask(&rle, width * height) {
                let _ = system.diffusion.mask.set_roles_slice(&bytes);
            }
        }
        let mut runner = MorphogenesisRunner {
            system,
            kinetics: p.kinetics,
            gray_scott,
            params: Arc::clone(params),
            kinetics_state: kinetics_state.map(Arc::clone),
            last_synced_ks: synced_ks,
            width,
            height,
            step_count: 0,
            rng: oxidize_core::rng::OxidizeRng::new(DEFAULT_MORPHOGENESIS_SEED),
            noise_amplitude: 0.0,
            twin: None,
        };
        runner.enforce_mask_concentrations();
        runner
    }

    pub(crate) fn apply_loaded_dimensions_and_state(
        &mut self,
        next_w: usize,
        next_h: usize,
        prev_had_mask: bool,
        target_state: SimWorkerState,
        next_pending: bool,
    ) {
        if next_w != self.width || next_h != self.height {
            self.width = next_w;
            self.height = next_h;
            self.local_runner = Self::create_kinetics_runner(
                &self.params,
                &self.kinetics_state,
                &self.mask_rle,
                self.width,
                self.height,
            );
            self.branch_engine.clear();
            let runner = Self::create_kinetics_runner(
                &self.params,
                &self.kinetics_state,
                &self.mask_rle,
                self.width,
                self.height,
            );
            self.worker = BackgroundWorker::new(runner);
            self.stepped_pixels = None;
            self.step_runner = None;
            self.dirty = true;
            self.invalidate_scalar_field_cache();
        } else {
            self.reset();
            let rle_mask = self
                .mask_rle
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            if prev_had_mask || !rle_mask.is_empty() {
                let _ = self
                    .worker
                    .send_command(SimCommand::SyncSculptState(SculptStatePayload {
                        rle_mask,
                        sparse_potential: Vec::new(),
                    }));
            }
        }
        if target_state == SimWorkerState::Running {
            self.pending_initial_start = false;
            if !self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Start);
            }
        } else {
            self.pending_initial_start = next_pending;
            if self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Pause);
            }
        }
    }

    pub(crate) fn compute_state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let (mode, f, k) = self.effective_kinetics_triplet();
        {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            mode.hash(&mut h);
            p.a.to_bits().hash(&mut h);
            p.b.to_bits().hash(&mut h);
            f.to_bits().hash(&mut h);
            k.to_bits().hash(&mut h);
            p.d_u.to_bits().hash(&mut h);
            p.d_v.to_bits().hash(&mut h);
            p.dt.to_bits().hash(&mut h);
            p.auto_dt.hash(&mut h);
            p.heatmap_mode.hash(&mut h);
        }
        self.width.hash(&mut h);
        self.height.hash(&mut h);
        self.is_forked().hash(&mut h);
        self.local_runner.step_count.hash(&mut h);
        self.mask_rle
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .hash(&mut h);
        std::mem::discriminant(&self.worker.state()).hash(&mut h);
        self.pending_initial_start.hash(&mut h);
        self.atlas.is_visible().hash(&mut h);
        self.atlas.axis_keys().hash(&mut h);
        h.finish().max(1)
    }

    pub(crate) fn serialize_morpho_state(&self) -> Option<String> {
        let (kinetics_mode, f, k) = self.effective_kinetics_triplet();
        let p = *self.params.read().unwrap_or_else(|e| e.into_inner());
        let fork = self
            .local_runner
            .twin
            .as_ref()
            .map(|t| t.to_descriptor(p.heatmap_mode));
        let mask_rle = self
            .mask_rle
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let atlas_axes = if self.atlas.has_default_persisted_state() {
            super::cartographer::default_morpho_atlas_axes()
        } else {
            let (xk, yk) = self.atlas.axis_keys();
            [xk.to_string(), yk.to_string()]
        };
        serde_json::to_string(&MorphogenesisToolState {
            kinetics_mode,
            a: p.a,
            b: p.b,
            f,
            k,
            d_u: p.d_u,
            d_v: p.d_v,
            dt: p.dt,
            auto_dt: p.auto_dt,
            width: self.width,
            height: self.height,
            state: Some(self.worker.state()),
            paused: None,
            pending_initial_start: self.pending_initial_start,
            fork,
            mask_rle,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes,
        })
        .ok()
    }

    fn apply_loaded_kinetics_and_params(&mut self, parsed: &MorphogenesisToolState) {
        self.theory.mode = parsed.kinetics_mode;
        {
            let mut ks = self
                .kinetics_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            ks.mode = parsed.kinetics_mode;
            ks.f = parsed.f;
            ks.k = parsed.k;
        }
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.kinetics = parsed.kinetics_mode;
        p.a = parsed.a;
        p.b = parsed.b;
        p.f = parsed.f;
        p.k = parsed.k;
        p.d_u = parsed.d_u;
        p.d_v = parsed.d_v;
        p.dt = parsed.dt;
        p.auto_dt = parsed.auto_dt;
        p.heatmap_mode = parsed
            .fork
            .map_or(MorphogenesisHeatmapMode::ActiveUb, |fk| fk.heatmap_mode);
    }

    pub(crate) fn deserialize_morpho_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: MorphogenesisToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Morphogenesis state: {e}"))?;
        parsed.validate()?;
        self.sync_atlas_options_for_mode(parsed.kinetics_mode);
        let use_mode_default = parsed.kinetics_mode
            == crate::presets::MorphogenesisKineticsMode::GrayScott
            && super::cartographer::is_default_morpho_atlas_axes(&parsed.atlas_axes);
        if !use_mode_default {
            self.atlas
                .set_axis_keys(&parsed.atlas_axes[0], &parsed.atlas_axes[1])?;
        }
        self.atlas.set_visible(parsed.atlas_visible);
        let (target_state, next_pending) = parsed.resolve_target_state(self.pending_initial_start);
        self.apply_loaded_kinetics_and_params(&parsed);
        let prev_had_mask = !self
            .mask_rle
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty();
        *self.mask_rle.write().unwrap_or_else(|e| e.into_inner()) = parsed.mask_rle;
        scientific_metadata::citation_registry::CitationRegistry::register(
            self.name().to_string(),
            self.theory.theory_citation(),
        );
        if let Some(fork_desc) = parsed.fork {
            self.apply_loaded_fork_state(
                parsed.width,
                parsed.height,
                fork_desc,
                target_state,
                next_pending,
            )?;
        } else {
            self.apply_loaded_dimensions_and_state(
                parsed.width,
                parsed.height,
                prev_had_mask,
                target_state,
                next_pending,
            );
        }
        Ok(())
    }
}
