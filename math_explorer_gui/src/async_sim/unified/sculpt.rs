//! Compact UI-side sculpt state and background worker runner for [`super::UnifiedSimTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CompiledFormulaPayload, UnifiedModel};
use crate::async_sim::{SculptStatePayload, SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::MAX_RLE_MASK_CELLS;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

impl SculptStatePayload {
    /// Returns `true` when neither RLE 2D masks nor sparse 1D potentials are populated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rle_mask.is_empty() && self.sparse_potential.is_empty()
    }

    /// Clears all persisted sculpt deltas.
    pub fn clear(&mut self) {
        self.rle_mask.clear();
        self.sparse_potential.clear();
    }

    /// Hashes sculpt state into `hasher` so `InteractiveTool::state_revision` tracks brush edits.
    pub fn hash_into(&self, hasher: &mut impl Hasher) {
        self.rle_mask.hash(hasher);
        for &(idx, val) in &self.sparse_potential {
            idx.hash(hasher);
            val.to_bits().hash(hasher);
        }
    }
}

pub(super) fn validate_rle_mask_entries(entries: &[[u32; 2]]) -> Result<(), String> {
    let total_cells: usize = entries.iter().try_fold(0_usize, |acc, &[v, len]| {
        if v > u32::from(u8::MAX) {
            return Err("RLE value out of u8 range".to_string());
        }
        acc.checked_add(len as usize)
            .ok_or_else(|| "RLE length overflow".to_string())
    })?;
    if total_cells > MAX_RLE_MASK_CELLS {
        return Err(format!(
            "rle_mask cell count {total_cells} exceeds {MAX_RLE_MASK_CELLS}"
        ));
    }
    Ok(())
}

pub(super) fn validate_sparse_potential_entries(
    entries: &[(u16, f64)],
    model_name: &str,
) -> Result<(), String> {
    if entries.iter().any(|&(_, v)| !v.is_finite()) {
        return Err(format!("Non-finite sparse_potential value in {model_name}"));
    }
    Ok(())
}

/// Background simulation runner bridging [`UnifiedModel`] with shared parameters, sculpt masks,
/// custom 1D series, and compiled AST formula bytecode.
pub struct UnifiedSimRunner<M: UnifiedModel> {
    model: M,
    params: Arc<RwLock<HashMap<String, f64>>>,
    sculpt_state: Arc<RwLock<SculptStatePayload>>,
    custom_series: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    formula: Arc<RwLock<Option<CompiledFormulaPayload>>>,
    applied_formula_rev: Option<u64>,
    steps_per_frame: usize,
}

impl<M: UnifiedModel> UnifiedSimRunner<M> {
    /// Creates a runner sharing `params` with empty sculpt, series, and formula slots.
    pub fn new(params: Arc<RwLock<HashMap<String, f64>>>) -> Self {
        Self::with_all_shared(
            params,
            Arc::new(RwLock::new(SculptStatePayload::default())),
            Arc::new(RwLock::new(HashMap::new())),
            Arc::new(RwLock::new(None)),
        )
    }

    /// Creates a runner sharing `params` and `custom_series`.
    pub fn new_with_series(
        params: Arc<RwLock<HashMap<String, f64>>>,
        custom_series: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    ) -> Self {
        Self::with_all_shared(
            params,
            Arc::new(RwLock::new(SculptStatePayload::default())),
            custom_series,
            Arc::new(RwLock::new(None)),
        )
    }

    /// Creates a runner sharing `params` and an optional [`CompiledFormulaPayload`] slot.
    pub fn with_formula(
        params: Arc<RwLock<HashMap<String, f64>>>,
        formula: Arc<RwLock<Option<CompiledFormulaPayload>>>,
    ) -> Self {
        Self::with_all_shared(
            params,
            Arc::new(RwLock::new(SculptStatePayload::default())),
            Arc::new(RwLock::new(HashMap::new())),
            formula,
        )
    }

    /// Creates a runner sharing `params`, `custom_series`, and an optional [`CompiledFormulaPayload`] slot.
    pub fn with_series_and_formula(
        params: Arc<RwLock<HashMap<String, f64>>>,
        custom_series: Arc<RwLock<HashMap<String, Vec<f64>>>>,
        formula: Arc<RwLock<Option<CompiledFormulaPayload>>>,
    ) -> Self {
        Self::with_all_shared(
            params,
            Arc::new(RwLock::new(SculptStatePayload::default())),
            custom_series,
            formula,
        )
    }

    /// Creates a runner sharing `params`, `sculpt_state`, and an optional [`CompiledFormulaPayload`] slot.
    pub fn with_shared(
        params: Arc<RwLock<HashMap<String, f64>>>,
        sculpt_state: Arc<RwLock<SculptStatePayload>>,
        formula: Arc<RwLock<Option<CompiledFormulaPayload>>>,
    ) -> Self {
        Self::with_all_shared(
            params,
            sculpt_state,
            Arc::new(RwLock::new(HashMap::new())),
            formula,
        )
    }

    /// Creates a runner sharing `params`, `sculpt_state`, `custom_series`, and `formula`.
    pub fn with_all_shared(
        params: Arc<RwLock<HashMap<String, f64>>>,
        sculpt_state: Arc<RwLock<SculptStatePayload>>,
        custom_series: Arc<RwLock<HashMap<String, Vec<f64>>>>,
        formula: Arc<RwLock<Option<CompiledFormulaPayload>>>,
    ) -> Self {
        let initial_params = params.read().unwrap_or_else(|e| e.into_inner()).clone();
        let mut runner = Self {
            model: M::new(&initial_params),
            params,
            sculpt_state,
            custom_series,
            formula,
            applied_formula_rev: None,
            steps_per_frame: 5,
        };
        runner.sync_formula(&initial_params);
        runner.apply_sculpt_if_present(&initial_params);
        runner.apply_series_if_present(&initial_params);
        runner
    }

    fn apply_sculpt_if_present(&mut self, current_params: &HashMap<String, f64>) {
        if M::supports_spatial_brush() {
            let sculpt = self.sculpt_state.read().unwrap_or_else(|e| e.into_inner());
            if !sculpt.is_empty() {
                self.model.apply_sculpt_payload(&sculpt, current_params);
            }
        }
    }

    fn apply_series_if_present(&mut self, current_params: &HashMap<String, f64>) {
        let series = self.custom_series.read().unwrap_or_else(|e| e.into_inner());
        if !series.is_empty() {
            self.model.apply_custom_series(&series, current_params);
        }
    }

    fn sync_formula(&mut self, current_params: &HashMap<String, f64>) {
        let payload_opt = self
            .formula
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let next_rev = payload_opt.as_ref().map(|p| p.revision);
        if next_rev != self.applied_formula_rev {
            self.model
                .apply_compiled_formula(payload_opt.as_ref(), current_params);
            self.applied_formula_rev = next_rev;
            self.apply_sculpt_if_present(current_params);
            self.apply_series_if_present(current_params);
        }
    }

    fn sync_sculpt_from_model(&self, current_params: &HashMap<String, f64>) {
        if M::supports_spatial_brush() {
            let exported = self.model.export_sculpt_payload(current_params);
            let mut guard = self.sculpt_state.write().unwrap_or_else(|e| e.into_inner());
            *guard = exported;
        }
    }
}

impl<M: UnifiedModel> SimulationRunner for UnifiedSimRunner<M> {
    fn process_command(&mut self, cmd: SimCommand) {
        let current_params = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        match cmd {
            SimCommand::SetSpeed(speed) => self.steps_per_frame = speed,
            SimCommand::Reset => {
                let had_sculpt = !self
                    .sculpt_state
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_empty();
                if had_sculpt {
                    self.apply_sculpt_if_present(&current_params);
                }
                self.model.reset_model(&current_params);
                self.applied_formula_rev = None;
                self.sync_formula(&current_params);
                self.apply_series_if_present(&current_params);
                if had_sculpt {
                    self.sync_sculpt_from_model(&current_params);
                }
            }
            SimCommand::SyncSculptState(payload) => {
                self.sync_formula(&current_params);
                self.model.apply_sculpt_payload(&payload, &current_params);
                let mut guard = self.sculpt_state.write().unwrap_or_else(|e| e.into_inner());
                *guard = payload;
            }
            other => {
                self.sync_formula(&current_params);
                self.model.process_command(other, &current_params);
                self.sync_sculpt_from_model(&current_params);
            }
        }
    }

    fn step(&mut self) {
        let current_params = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        self.sync_formula(&current_params);
        self.model.step(&current_params);
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
}
