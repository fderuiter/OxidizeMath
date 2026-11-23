#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "hodgkin_huxley/atlas.rs"]
pub mod atlas;

#[cfg(test)]
pub use atlas::HodgkinHuxleyRegime;
pub use atlas::{HodgkinHuxleyCartographer, HH_ATLAS_AXIS_PAIRS};

use crate::async_sim::atlas::RegimeCartographer;
use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::{CheckpointParams, CheckpointPayload, StateSnapshot, TimeTravelCheckpoint};
use math_explorer::biology::neuroscience::{
    HodgkinHuxleyModel, HodgkinHuxleyParameters, HodgkinHuxleyState,
};
use math_explorer::pure_math::analysis::ode::{Euler, SolverExt};
use oxidize_core::rng::OxidizeRng;
use scientific_metadata::theory::ParameterConstraint;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const HH_MAX_HISTORY: usize = 10_000;

#[allow(missing_docs)]
pub struct HodgkinHuxleyUnified {
    state: HodgkinHuxleyState,
    params: HodgkinHuxleyParameters,
    history: VecDeque<[f64; 2]>,
    time: f64,
    i_ext: f64,
    dt: f64,
    step_count: usize,
    rng: OxidizeRng,
    prev_v: f64,
    spiked_since_snapshot: AtomicBool,
}

impl HodgkinHuxleyUnified {
    fn apply_scalar_step(&mut self, g_na: f64, g_k: f64, g_l: f64, i_ext: f64, dt: f64) {
        self.params.g_na = g_na.max(0.0);
        self.params.g_k = g_k.max(0.0);
        self.params.g_l = g_l.max(0.0);
        self.i_ext = i_ext;
        self.dt = dt;
        let prev_v = self.state.v;
        let mut solver = Euler::new(&self.state);
        let model = HodgkinHuxleyModel::new(self.params.clone(), self.i_ext);
        let next = solver.solve(&model, 0.0, &self.state, self.dt);
        self.state = HodgkinHuxleyState {
            v: next.v,
            n: next.n.clamp(0.0, 1.0),
            m: next.m.clamp(0.0, 1.0),
            h: next.h.clamp(0.0, 1.0),
        };
        if prev_v <= 0.0 && self.state.v > 0.0 {
            self.spiked_since_snapshot.store(true, Ordering::Relaxed);
        }
        self.prev_v = prev_v;
        self.time += self.dt;
        self.step_count += 1;
        if self.history.len() >= HH_MAX_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back([self.time, self.state.v]);
    }
}

impl crate::framework::recovery::FiniteStateGuard for HodgkinHuxleyUnified {}

impl UnifiedModel for HodgkinHuxleyUnified {
    fn domain() -> &'static str {
        "neuroscience"
    }

    fn new(params: &HashMap<String, f64>) -> Self {
        let g_na = *params.get("g_na").unwrap_or(&120.0);
        let g_k = *params.get("g_k").unwrap_or(&36.0);
        let g_l = *params.get("g_l").unwrap_or(&0.3);
        let i_ext = *params.get("i_ext").unwrap_or(&10.0);

        let hh_params = HodgkinHuxleyParameters {
            g_na: g_na.max(0.0),
            g_k: g_k.max(0.0),
            g_l: g_l.max(0.0),
            ..Default::default()
        };

        Self {
            state: HodgkinHuxleyState {
                v: -65.0,
                n: 0.32,
                m: 0.05,
                h: 0.6,
            },
            params: hh_params,
            history: VecDeque::with_capacity(HH_MAX_HISTORY),
            time: 0.0,
            i_ext,
            dt: 0.01,
            step_count: 0,
            rng: OxidizeRng::new(0x4848_5F4E_4555_524F),
            prev_v: -65.0,
            spiked_since_snapshot: AtomicBool::new(false),
        }
    }

    fn step(&mut self, params: &HashMap<String, f64>) {
        let new_g_na = *params.get("g_na").unwrap_or(&120.0);
        let new_g_k = *params.get("g_k").unwrap_or(&36.0);
        let new_g_l = *params.get("g_l").unwrap_or(&0.3);
        let new_i_ext = *params.get("i_ext").unwrap_or(&10.0);
        self.apply_scalar_step(new_g_na, new_g_k, new_g_l, new_i_ext, self.dt);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let mut custom_data = Vec::with_capacity(self.history.len() * 2 + 2);
        for &[t, v] in &self.history {
            custom_data.push(t);
            custom_data.push(v);
        }
        custom_data.push(self.time);
        custom_data.push(self.state.v);

        let spiked = self.spiked_since_snapshot.swap(false, Ordering::Relaxed);
        let ref_prev_v = if spiked { -5.0 } else { self.prev_v };
        let ref_cur_v = if spiked {
            self.state.v.max(10.0)
        } else {
            self.state.v
        };
        let sonification = crate::audio::mappings::hodgkin_huxley_frame(
            ref_cur_v,
            ref_prev_v,
            self.state.m,
            self.state.h,
            self.state.n,
            self.i_ext,
        );

        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data,
            structured_data: None,
            sonification,
        }
    }

    fn sonification_tool_id() -> Option<&'static str> {
        Some("hodgkin_huxley")
    }

    fn parameters() -> HashMap<String, ParameterConstraint> {
        let mut map = HashMap::new();
        map.insert(
            "g_na".to_string(),
            ParameterConstraint {
                min: 0.0,
                max: 200.0,
                step: 1.0,
            },
        );
        map.insert(
            "g_k".to_string(),
            ParameterConstraint {
                min: 0.0,
                max: 100.0,
                step: 1.0,
            },
        );
        map.insert(
            "g_l".to_string(),
            ParameterConstraint {
                min: 0.0,
                max: 5.0,
                step: 0.1,
            },
        );
        map.insert(
            "i_ext".to_string(),
            ParameterConstraint {
                min: 0.0,
                max: 50.0,
                step: 1.0,
            },
        );
        map
    }

    fn parameter_label(name: &str) -> String {
        match name {
            "g_na" => "Sodium Conductance (mS/cm²)".to_string(),
            "g_k" => "Potassium Conductance (mS/cm²)".to_string(),
            "g_l" => "Leak Conductance (mS/cm²)".to_string(),
            "i_ext" => "External Stimulus Current (µA/cm²)".to_string(),
            other => other.to_string(),
        }
    }

    fn name() -> &'static str {
        "Hodgkin-Huxley Model"
    }

    fn evaluate_observable(params: &HashMap<String, f64>) -> Option<(&'static str, f64)> {
        let g_na = params.get("g_na").copied().unwrap_or(120.0).max(0.0);
        let g_k = params.get("g_k").copied().unwrap_or(36.0).max(0.0);
        let g_l = params.get("g_l").copied().unwrap_or(0.3).max(0.0);
        let i_ext = params.get("i_ext").copied().unwrap_or(10.0);

        let hh_params = HodgkinHuxleyParameters {
            g_na,
            g_k,
            g_l,
            ..Default::default()
        };

        let mut state = HodgkinHuxleyState {
            v: -65.0,
            n: 0.32,
            m: 0.05,
            h: 0.6,
        };
        let mut solver = Euler::new(&state);
        let model = HodgkinHuxleyModel::new(hh_params, i_ext);
        let dt = 0.01;
        let mut peak_v = state.v;
        for _ in 0..600 {
            let next = solver.solve(&model, 0.0, &state, dt);
            state = HodgkinHuxleyState {
                v: next.v,
                n: next.n.clamp(0.0, 1.0),
                m: next.m.clamp(0.0, 1.0),
                h: next.h.clamp(0.0, 1.0),
            };
            peak_v = peak_v.max(state.v);
        }

        Some(("Peak Membrane Potential (mV)", peak_v))
    }

    fn checkpoint_stride(&self) -> Option<usize> {
        Some(1)
    }

    fn current_step(&self) -> usize {
        self.step_count
    }

    fn capture_active_params(&self, params: &HashMap<String, f64>) -> Option<CheckpointParams> {
        let g_na = *params.get("g_na").unwrap_or(&120.0);
        let g_k = *params.get("g_k").unwrap_or(&36.0);
        let g_l = *params.get("g_l").unwrap_or(&0.3);
        let i_ext = *params.get("i_ext").unwrap_or(&10.0);
        Some(CheckpointParams::scalars(&[g_na, g_k, g_l, i_ext, self.dt]))
    }

    fn capture_checkpoint(&self, params: &HashMap<String, f64>) -> Option<TimeTravelCheckpoint> {
        let cp_params = self.capture_active_params(params)?;
        Some(TimeTravelCheckpoint {
            step: self.step_count,
            rng: self.rng.clone(),
            params: cp_params,
            state: CheckpointPayload::scalars(&[
                self.state.v,
                self.state.n,
                self.state.m,
                self.state.h,
                self.time,
            ]),
        })
    }

    fn restore_checkpoint(&mut self, checkpoint: &TimeTravelCheckpoint) {
        if let Some(p) = checkpoint.params.as_scalars() {
            if p.len() >= 5 {
                self.params.g_na = p[0].max(0.0);
                self.params.g_k = p[1].max(0.0);
                self.params.g_l = p[2].max(0.0);
                self.i_ext = p[3];
                self.dt = p[4];
            }
        }
        if let Some(s) = checkpoint.state.as_scalars() {
            if s.len() >= 5 {
                self.state = HodgkinHuxleyState {
                    v: s[0],
                    n: s[1].clamp(0.0, 1.0),
                    m: s[2].clamp(0.0, 1.0),
                    h: s[3].clamp(0.0, 1.0),
                };
                self.time = s[4];
            }
        }
        self.prev_v = self.state.v;
        self.spiked_since_snapshot.store(false, Ordering::Relaxed);
        self.step_count = checkpoint.step;
        self.rng = checkpoint.rng.clone();
    }

    fn rebuild_history(&mut self, front: &[TimeTravelCheckpoint], back: &[TimeTravelCheckpoint]) {
        self.history.clear();
        let total = front
            .iter()
            .chain(back.iter())
            .filter(|cp| cp.step >= 1)
            .count();
        let skip = total.saturating_sub(HH_MAX_HISTORY);
        for cp in front
            .iter()
            .chain(back.iter())
            .filter(|cp| cp.step >= 1)
            .skip(skip)
        {
            if let Some(s) = cp.state.as_scalars() {
                if s.len() >= 5 {
                    self.history.push_back([s[4], s[0]]);
                }
            }
        }
    }

    fn replay_step(&mut self, params: &CheckpointParams) {
        if let Some(p) = params.as_scalars() {
            if p.len() >= 5 {
                self.apply_scalar_step(p[0], p[1], p[2], p[3], p[4]);
            }
        } else if let Some(map) = params.as_named_map() {
            self.step(map);
        }
    }

    fn atlas_axis_pairs() -> &'static [(&'static str, &'static str)] {
        HH_ATLAS_AXIS_PAIRS
    }

    fn build_atlas_cartographer(
        params: &HashMap<String, f64>,
        axis_keys: (&'static str, &'static str),
    ) -> Option<Arc<dyn RegimeCartographer>> {
        Some(Arc::new(HodgkinHuxleyCartographer::from_params(
            params, axis_keys,
        )))
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "HodgkinHuxleyTool",
        display_name: "Hodgkin-Huxley Model",
        domain: "neuroscience",
        tags: &[],
        build: || Box::new(UnifiedSimTool::<HodgkinHuxleyUnified>::new()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for HodgkinHuxleyUnified {
    fn theory_description(&self) -> String {
        HodgkinHuxleyModel::new(HodgkinHuxleyParameters::default(), 0.0).theory_description()
    }

    fn phonetic_description(&self) -> String {
        HodgkinHuxleyModel::new(HodgkinHuxleyParameters::default(), 0.0).phonetic_description()
    }

    fn theory_citation(&self) -> String {
        HodgkinHuxleyModel::new(HodgkinHuxleyParameters::default(), 0.0).theory_citation()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        HodgkinHuxleyModel::new(HodgkinHuxleyParameters::default(), 0.0).available_descriptions()
    }

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        Self::parameters()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Hodgkin-Huxley Action Potential",
    "domain": "neuroscience",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "g_na": 120.0,
    "g_k": 36.0,
    "g_l": 0.3,
    "i_ext": 10.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "hodgkin_huxley_starter",
        title: "Hodgkin-Huxley Action Potential",
        description: "C_m·dV/dt = -g_Na·m³h(V - E_Na) - g_K·n⁴(V - E_K) - g_L(V - E_L) + I: Squid axon membrane spikes.",
        category: "Computational Neuroscience",
        icon: "🧠",
        domain: "neuroscience",
        tool_name: Some("HodgkinHuxleyTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
