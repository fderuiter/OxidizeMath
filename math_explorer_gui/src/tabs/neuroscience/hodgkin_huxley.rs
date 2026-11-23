#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "hodgkin_huxley/atlas.rs"]
pub mod atlas;

use crate::async_sim::atlas::RegimeCartographer;
use crate::async_sim::unified::{UnifiedCheckpointSlot, UnifiedModel, UnifiedSimTool};
use crate::async_sim::StateSnapshot;
#[cfg(test)]
pub use atlas::HodgkinHuxleyRegime;
pub use atlas::{HodgkinHuxleyCartographer, HH_ATLAS_AXIS_PAIRS};
use math_explorer::biology::neuroscience::{
    HodgkinHuxleyModel, HodgkinHuxleyNeuron, HodgkinHuxleyParameters,
};
use scientific_metadata::theory::ParameterConstraint;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

#[allow(missing_docs)]
pub struct HodgkinHuxleyUnified {
    neuron: HodgkinHuxleyNeuron,
    params: HodgkinHuxleyParameters,
    history: VecDeque<[f64; 2]>,
    time: f64,
    i_ext: f64,
    dt: f64,
    prev_v: f64,
    spiked_since_snapshot: std::sync::atomic::AtomicBool,
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
            g_na,
            g_k,
            g_l,
            ..Default::default()
        };

        Self {
            neuron: HodgkinHuxleyNeuron::builder()
                .with_params(hh_params.clone())
                .build()
                .unwrap_or(HodgkinHuxleyNeuron::new(-65.0)),
            params: hh_params,
            history: VecDeque::new(),
            time: 0.0,
            i_ext,
            dt: 0.01,
            prev_v: -65.0,
            spiked_since_snapshot: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn step(&mut self, params: &HashMap<String, f64>) {
        let new_g_na = *params.get("g_na").unwrap_or(&120.0);
        let new_g_k = *params.get("g_k").unwrap_or(&36.0);
        let new_g_l = *params.get("g_l").unwrap_or(&0.3);
        let new_i_ext = *params.get("i_ext").unwrap_or(&10.0);

        let params_changed = self.params.g_na != new_g_na
            || self.params.g_k != new_g_k
            || self.params.g_l != new_g_l;

        self.params.g_na = new_g_na;
        self.params.g_k = new_g_k;
        self.params.g_l = new_g_l;
        self.i_ext = new_i_ext;

        if params_changed {
            let builder = HodgkinHuxleyNeuron::builder()
                .with_initial_v(self.neuron.v())
                .with_n(self.neuron.n())
                .with_m(self.neuron.m())
                .with_h(self.neuron.h())
                .with_params(self.params.clone());

            if let Ok(new_neuron) = builder.build() {
                self.neuron = new_neuron;
            }
        }

        let prev_v = self.neuron.v();
        self.neuron.update(self.dt, self.i_ext);
        let cur_v = self.neuron.v();
        if prev_v <= 0.0 && cur_v > 0.0 {
            self.spiked_since_snapshot
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        self.prev_v = prev_v;
        self.time += self.dt;
        self.history.push_back([self.time, cur_v]);
        while self.history.len() > 10_000 {
            self.history.pop_front();
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let mut custom_data = Vec::with_capacity(self.history.len() * 2 + 2);
        for &[t, v] in &self.history {
            custom_data.push(t);
            custom_data.push(v);
        }
        custom_data.push(self.time);
        custom_data.push(self.neuron.v());

        let spiked = self
            .spiked_since_snapshot
            .swap(false, std::sync::atomic::Ordering::Relaxed);
        let ref_prev_v = if spiked { -5.0 } else { self.prev_v };
        let ref_cur_v = if spiked {
            self.neuron.v().max(10.0)
        } else {
            self.neuron.v()
        };
        let sonification = crate::audio::mappings::hodgkin_huxley_frame(
            ref_cur_v,
            ref_prev_v,
            self.neuron.m(),
            self.neuron.h(),
            self.neuron.n(),
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

        let mut neuron = HodgkinHuxleyNeuron::builder()
            .with_params(hh_params)
            .build()
            .ok()?;

        let dt = 0.01;
        let mut peak_v = neuron.v();
        for _ in 0..600 {
            neuron.update(dt, i_ext);
            peak_v = peak_v.max(neuron.v());
        }

        Some(("Peak Membrane Potential (mV)", peak_v))
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

    fn supports_timeline_branch() -> bool {
        true
    }

    fn checkpoint_stride() -> usize {
        5
    }

    fn time_step_dt(&self, _params: &HashMap<String, f64>) -> f64 {
        self.dt
    }

    fn time_unit() -> &'static str {
        "ms"
    }

    fn write_checkpoint(&self, slot: &mut UnifiedCheckpointSlot) {
        slot.scalars[0] = self.time;
        slot.scalars[1] = self.neuron.v();
        slot.scalars[2] = self.neuron.n();
        slot.scalars[3] = self.neuron.m();
        slot.scalars[4] = self.neuron.h();
        slot.scalars[5] = self.prev_v;
        slot.scalars[6] = self.i_ext;
    }

    fn restore_checkpoint(&mut self, slot: &UnifiedCheckpointSlot, params: &HashMap<String, f64>) {
        self.time = slot.scalars[0];
        self.prev_v = slot.scalars[5];
        self.i_ext = params.get("i_ext").copied().unwrap_or(slot.scalars[6]);
        self.params.g_na = params.get("g_na").copied().unwrap_or(self.params.g_na);
        self.params.g_k = params.get("g_k").copied().unwrap_or(self.params.g_k);
        self.params.g_l = params.get("g_l").copied().unwrap_or(self.params.g_l);
        if let Ok(restored) = HodgkinHuxleyNeuron::builder()
            .with_initial_v(slot.scalars[1])
            .with_n(slot.scalars[2].clamp(0.0, 1.0))
            .with_m(slot.scalars[3].clamp(0.0, 1.0))
            .with_h(slot.scalars[4].clamp(0.0, 1.0))
            .with_params(self.params.clone())
            .build()
        {
            self.neuron = restored;
        }
    }

    fn truncate_history_to_step(&mut self, _step: usize, time: f64) {
        let eps = self.dt * 0.25;
        while let Some(&[t, _]) = self.history.back() {
            if t > time + eps {
                self.history.pop_back();
            } else {
                break;
            }
        }
    }

    fn fork_shadow_twin(&self) -> Option<Self> {
        let cloned_neuron = HodgkinHuxleyNeuron::builder()
            .with_initial_v(self.neuron.v())
            .with_n(self.neuron.n().clamp(0.0, 1.0))
            .with_m(self.neuron.m().clamp(0.0, 1.0))
            .with_h(self.neuron.h().clamp(0.0, 1.0))
            .with_params(self.params.clone())
            .build()
            .ok()?;
        Some(Self {
            neuron: cloned_neuron,
            params: self.params.clone(),
            history: self.history.clone(),
            time: self.time,
            i_ext: self.i_ext,
            dt: self.dt,
            prev_v: self.prev_v,
            spiked_since_snapshot: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn scalar_field_len(&self) -> usize {
        4
    }

    fn read_scalar_field(&self, out: &mut [f64]) -> usize {
        if out.len() < 4 {
            return 0;
        }
        out[0] = self.neuron.v();
        out[1] = self.neuron.n();
        out[2] = self.neuron.m();
        out[3] = self.neuron.h();
        4
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
