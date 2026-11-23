//! Compact UI-side sculpt state and JSON preset/URL persistence helpers for [`super::UnifiedSimTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CompiledFormulaPayload, UnifiedModel, UnifiedSimTool};
use crate::async_sim::{SculptStatePayload, SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::MAX_RLE_MASK_CELLS;
use std::collections::{BTreeMap, HashMap};
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

impl<M: UnifiedModel> UnifiedSimTool<M> {
    pub(super) fn serialize_unified_state(&self) -> Option<String> {
        let params = self.params.read().unwrap_or_else(|e| e.into_inner());
        let sculpt = self.sculpt_state.read().unwrap_or_else(|e| e.into_inner());

        let mut map = serde_json::Map::new();
        let sorted: BTreeMap<&String, &f64> = params.iter().collect();
        for (k, v) in sorted {
            if let Some(num) = serde_json::Number::from_f64(*v) {
                map.insert(k.clone(), serde_json::Value::Number(num));
            }
        }

        if !sculpt.rle_mask.is_empty() {
            if let Ok(val) = serde_json::to_value(&sculpt.rle_mask) {
                map.insert("rle_mask".to_string(), val);
            }
        }
        if !sculpt.sparse_potential.is_empty() {
            if let Ok(val) = serde_json::to_value(&sculpt.sparse_potential) {
                map.insert("sparse_potential".to_string(), val);
            }
        }

        serde_json::to_string(&map).ok()
    }

    fn parse_unified_state_entry(
        &self,
        key: String,
        val: serde_json::Value,
        parsed_params: &mut HashMap<String, f64>,
        parsed_rle: &mut Vec<[u32; 2]>,
        parsed_sparse: &mut Vec<(u16, f64)>,
    ) -> Result<(), String> {
        if key == "rle_mask" && M::supports_spatial_brush() && M::is_raster_model() {
            *parsed_rle = serde_json::from_value(val)
                .map_err(|e| format!("Invalid rle_mask in {}: {e}", M::name()))?;
            validate_rle_mask_entries(parsed_rle)?;
            return Ok(());
        }
        if key == "sparse_potential" && M::supports_spatial_brush() && !M::is_raster_model() {
            *parsed_sparse = serde_json::from_value(val)
                .map_err(|e| format!("Invalid sparse_potential in {}: {e}", M::name()))?;
            validate_sparse_potential_entries(parsed_sparse, M::name())?;
            return Ok(());
        }
        if !self.param_metadata.iter().any(|meta| meta.0 == key) {
            return Err(format!("Unknown parameter '{}' for {}", key, M::name()));
        }
        let Some(number) = val.as_f64().filter(|n| n.is_finite()) else {
            return Err(format!(
                "Non-finite or non-numeric value for parameter '{}' in {}",
                key,
                M::name()
            ));
        };
        parsed_params.insert(key, number);
        Ok(())
    }

    pub(super) fn deserialize_unified_state(&mut self, state: &str) -> Result<(), String> {
        let raw_map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(state)
            .map_err(|e| format!("Invalid simulation parameter state for {}: {e}", M::name()))?;

        let mut parsed_params: HashMap<String, f64> = HashMap::new();
        let mut parsed_rle: Vec<[u32; 2]> = Vec::new();
        let mut parsed_sparse: Vec<(u16, f64)> = Vec::new();

        for (key, val) in raw_map {
            self.parse_unified_state_entry(
                key,
                val,
                &mut parsed_params,
                &mut parsed_rle,
                &mut parsed_sparse,
            )?;
        }

        if !self.param_metadata.is_empty()
            && parsed_params.is_empty()
            && parsed_rle.is_empty()
            && parsed_sparse.is_empty()
        {
            return Err(format!(
                "Empty parameter configuration for {}; expected parameters: {}",
                M::name(),
                self.param_metadata
                    .iter()
                    .map(|m| m.0.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        self.apply_parsed_atlas_state(None)?;
        {
            let mut guard = self.params.write().unwrap_or_else(|e| e.into_inner());
            for (key, value) in parsed_params {
                guard.insert(key, value);
            }
        }
        self.reset_formula_bar_to_default();
        {
            let mut s_guard = self.sculpt_state.write().unwrap_or_else(|e| e.into_inner());
            s_guard.rle_mask = parsed_rle;
            s_guard.sparse_potential = parsed_sparse;
        }
        self.texture = None;
        self.respawn_preserving_sculpt(None);
        Ok(())
    }
}

/// Background worker runner driving a [`UnifiedModel`] with synchronized parameters, formulas, and sculpt payloads.
pub struct UnifiedSimRunner<M: UnifiedModel> {
    model: M,
    params: Arc<RwLock<HashMap<String, f64>>>,
    sculpt_state: Arc<RwLock<SculptStatePayload>>,
    formula: Arc<RwLock<Option<CompiledFormulaPayload>>>,
    applied_formula_rev: Option<u64>,
    steps_per_frame: usize,
}

impl<M: UnifiedModel> UnifiedSimRunner<M> {
    /// Creates a new [`UnifiedSimRunner`] bound to `params` with empty default sculpt and formula slots.
    pub fn new(params: Arc<RwLock<HashMap<String, f64>>>) -> Self {
        Self::with_shared(
            params,
            Arc::new(RwLock::new(SculptStatePayload::default())),
            Arc::new(RwLock::new(None)),
        )
    }

    /// Creates a runner sharing `params`, `sculpt_state`, and an optional [`CompiledFormulaPayload`] slot.
    pub fn with_shared(
        params: Arc<RwLock<HashMap<String, f64>>>,
        sculpt_state: Arc<RwLock<SculptStatePayload>>,
        formula: Arc<RwLock<Option<CompiledFormulaPayload>>>,
    ) -> Self {
        let initial_params = params.read().unwrap_or_else(|e| e.into_inner()).clone();
        let mut runner = Self {
            model: M::new(&initial_params),
            params,
            sculpt_state,
            formula,
            applied_formula_rev: None,
            steps_per_frame: 5,
        };
        runner.sync_formula(&initial_params);
        runner.apply_sculpt_if_present(&initial_params);
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
