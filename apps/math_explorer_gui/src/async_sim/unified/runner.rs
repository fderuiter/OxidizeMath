//! Background [`SimulationRunner`] adapter for [`super::UnifiedModel`].

use super::{CompiledFormulaPayload, UnifiedModel};
use crate::async_sim::{
    CheckpointParams, SculptStatePayload, SimCommand, SimulationRunner, StateSnapshot,
    TimeTravelCheckpoint,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Background [`SimulationRunner`] adapter driving a [`UnifiedModel`] with shared parameters,
/// formula revisions, persisted [`SculptStatePayload`] masks, and optional time-travel checkpoint hooks.
pub struct UnifiedSimRunner<M: UnifiedModel> {
    model: M,
    params: Arc<RwLock<HashMap<String, f64>>>,
    sculpt_state: Arc<RwLock<SculptStatePayload>>,
    formula: Arc<RwLock<Option<CompiledFormulaPayload>>>,
    applied_formula_rev: Option<u64>,
    obstacle_rle: Arc<RwLock<Option<String>>>,
    steps_per_frame: usize,
}

impl<M: UnifiedModel> UnifiedSimRunner<M> {
    /// Creates a new [`UnifiedSimRunner`] initialized from `params` with empty sculpt and formula state.
    pub fn new(params: Arc<RwLock<HashMap<String, f64>>>) -> Self {
        Self::with_shared(
            params,
            Arc::new(RwLock::new(SculptStatePayload::default())),
            Arc::new(RwLock::new(None)),
        )
    }

    /// Creates a runner sharing both `params` and an optional [`CompiledFormulaPayload`] slot.
    pub fn with_formula(
        params: Arc<RwLock<HashMap<String, f64>>>,
        formula: Arc<RwLock<Option<CompiledFormulaPayload>>>,
    ) -> Self {
        Self::with_shared(
            params,
            Arc::new(RwLock::new(SculptStatePayload::default())),
            formula,
        )
    }

    /// Creates a runner sharing `params`, `sculpt_state`, and an optional [`CompiledFormulaPayload`] slot.
    pub fn with_shared(
        params: Arc<RwLock<HashMap<String, f64>>>,
        sculpt_state: Arc<RwLock<SculptStatePayload>>,
        formula: Arc<RwLock<Option<CompiledFormulaPayload>>>,
    ) -> Self {
        Self::with_shared_and_obstacles(params, sculpt_state, formula, Arc::new(RwLock::new(None)))
    }

    /// Creates a runner sharing `params`, `sculpt_state`, `formula`, and an optional RLE obstacle mask.
    pub fn with_shared_and_obstacles(
        params: Arc<RwLock<HashMap<String, f64>>>,
        sculpt_state: Arc<RwLock<SculptStatePayload>>,
        formula: Arc<RwLock<Option<CompiledFormulaPayload>>>,
        obstacle_rle: Arc<RwLock<Option<String>>>,
    ) -> Self {
        let initial_params = params.read().unwrap_or_else(|e| e.into_inner()).clone();
        let mut model = M::new(&initial_params);
        if let Some(ref rle) = *obstacle_rle.read().unwrap_or_else(|e| e.into_inner()) {
            model.apply_obstacle_rle(rle);
        }
        let mut runner = Self {
            model,
            params,
            sculpt_state,
            formula,
            applied_formula_rev: None,
            obstacle_rle,
            steps_per_frame: 5,
        };
        runner.sync_formula(&initial_params);
        runner.apply_sculpt_if_present(&initial_params);
        runner
    }

    fn apply_sculpt_fields(
        model: &mut M,
        sculpt_state: &RwLock<SculptStatePayload>,
        current_params: &HashMap<String, f64>,
    ) {
        if M::supports_spatial_brush() {
            let sculpt = sculpt_state.read().unwrap_or_else(|e| e.into_inner());
            if !sculpt.is_empty() {
                model.apply_sculpt_payload(&sculpt, current_params);
            }
        }
    }

    fn apply_sculpt_if_present(&mut self, current_params: &HashMap<String, f64>) {
        Self::apply_sculpt_fields(&mut self.model, &self.sculpt_state, current_params);
    }

    fn sync_sculpt_fields(
        model: &M,
        sculpt_state: &RwLock<SculptStatePayload>,
        obstacle_rle: &RwLock<Option<String>>,
        current_params: &HashMap<String, f64>,
    ) {
        if M::supports_spatial_brush()
            && obstacle_rle
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .is_none()
        {
            let exported = model.export_sculpt_payload(current_params);
            let mut guard = sculpt_state.write().unwrap_or_else(|e| e.into_inner());
            *guard = exported;
        }
    }

    fn sync_formula_fields(
        model: &mut M,
        sculpt_state: &RwLock<SculptStatePayload>,
        formula: &RwLock<Option<CompiledFormulaPayload>>,
        applied_formula_rev: &mut Option<u64>,
        current_params: &HashMap<String, f64>,
    ) {
        let next_rev = {
            let guard = formula.read().unwrap_or_else(|e| e.into_inner());
            let rev = guard.as_ref().map(|p| p.revision);
            if rev == *applied_formula_rev {
                return;
            }
            model.apply_compiled_formula(guard.as_ref(), current_params);
            rev
        };
        *applied_formula_rev = next_rev;
        if M::supports_spatial_brush() {
            let sculpt = sculpt_state.read().unwrap_or_else(|e| e.into_inner());
            if !sculpt.is_empty() {
                model.apply_sculpt_payload(&sculpt, current_params);
            }
        }
    }

    fn sync_formula(&mut self, current_params: &HashMap<String, f64>) {
        Self::sync_formula_fields(
            &mut self.model,
            &self.sculpt_state,
            &self.formula,
            &mut self.applied_formula_rev,
            current_params,
        );
    }

    fn reset_runner(&mut self) {
        let current_params = self.params.read().unwrap_or_else(|e| e.into_inner());
        let saved_rle = self
            .obstacle_rle
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if let Some(ref rle) = saved_rle {
            let mut rebuilt = M::new(&current_params);
            rebuilt.apply_obstacle_rle(rle);
            self.model = rebuilt;
        } else {
            let had_sculpt = !self
                .sculpt_state
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty();
            if had_sculpt {
                Self::apply_sculpt_fields(&mut self.model, &self.sculpt_state, &current_params);
            }
            self.model.reset_model(&current_params);
            if had_sculpt {
                Self::sync_sculpt_fields(
                    &self.model,
                    &self.sculpt_state,
                    &self.obstacle_rle,
                    &current_params,
                );
            }
        }
        self.applied_formula_rev = None;
        Self::sync_formula_fields(
            &mut self.model,
            &self.sculpt_state,
            &self.formula,
            &mut self.applied_formula_rev,
            &current_params,
        );
    }

    fn sync_parameters_runner(&mut self) {
        let current_params = self.params.read().unwrap_or_else(|e| e.into_inner());
        let had_sculpt = !self
            .sculpt_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty();
        Self::sync_formula_fields(
            &mut self.model,
            &self.sculpt_state,
            &self.formula,
            &mut self.applied_formula_rev,
            &current_params,
        );
        self.model.sync_parameters(&current_params);
        if had_sculpt {
            Self::sync_sculpt_fields(
                &self.model,
                &self.sculpt_state,
                &self.obstacle_rle,
                &current_params,
            );
        }
    }
}

impl<M: UnifiedModel> SimulationRunner for UnifiedSimRunner<M> {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::SetSpeed(speed) => self.steps_per_frame = speed,
            SimCommand::Reset => self.reset_runner(),
            SimCommand::SyncSculptState(payload) => {
                let current_params = self.params.read().unwrap_or_else(|e| e.into_inner());
                Self::sync_formula_fields(
                    &mut self.model,
                    &self.sculpt_state,
                    &self.formula,
                    &mut self.applied_formula_rev,
                    &current_params,
                );
                self.model.apply_sculpt_payload(&payload, &current_params);
                let mut guard = self.sculpt_state.write().unwrap_or_else(|e| e.into_inner());
                *guard = payload;
            }
            SimCommand::Custom(ref action) if action == super::SYNC_PARAMETERS_COMMAND => {
                self.sync_parameters_runner();
            }
            other => {
                let current_params = self.params.read().unwrap_or_else(|e| e.into_inner());
                Self::sync_formula_fields(
                    &mut self.model,
                    &self.sculpt_state,
                    &self.formula,
                    &mut self.applied_formula_rev,
                    &current_params,
                );
                self.model.process_command(other, &current_params);
                Self::sync_sculpt_fields(
                    &self.model,
                    &self.sculpt_state,
                    &self.obstacle_rle,
                    &current_params,
                );
            }
        }
    }

    fn step(&mut self) {
        let guard = self.params.read().unwrap_or_else(|e| e.into_inner());
        Self::sync_formula_fields(
            &mut self.model,
            &self.sculpt_state,
            &self.formula,
            &mut self.applied_formula_rev,
            &guard,
        );
        self.model.step(&guard);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        self.model.get_snapshot()
    }

    fn get_steps_per_frame(&self) -> usize {
        self.steps_per_frame
    }

    fn is_finite_state(&self) -> bool {
        self.model.is_finite_state()
    }

    fn diagnostic_report(&self) -> Option<crate::framework::recovery::DiagnosticReport> {
        self.model.diagnostic_report().cloned()
    }

    fn checkpoint_stride(&self) -> Option<usize> {
        self.model.checkpoint_stride()
    }

    fn current_step(&self) -> usize {
        self.model.current_step()
    }

    fn capture_checkpoint(&self) -> Option<TimeTravelCheckpoint> {
        let guard = self.params.read().unwrap_or_else(|e| e.into_inner());
        self.model.capture_checkpoint(&guard)
    }

    fn capture_active_params(&self) -> Option<CheckpointParams> {
        let guard = self.params.read().unwrap_or_else(|e| e.into_inner());
        self.model.capture_active_params(&guard)
    }

    fn active_params_match(&self, last: &CheckpointParams) -> bool {
        let guard = self.params.read().unwrap_or_else(|e| e.into_inner());
        self.model.active_params_match(&guard, last)
    }

    fn restore_checkpoint(&mut self, checkpoint: &TimeTravelCheckpoint) {
        self.model.restore_checkpoint(checkpoint);
    }

    fn rebuild_history(&mut self, front: &[TimeTravelCheckpoint], back: &[TimeTravelCheckpoint]) {
        self.model.rebuild_history(front, back);
    }

    fn replay_step(&mut self, params: &CheckpointParams) {
        self.model.replay_step(params);
    }
}
