#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer,
};
use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::StateSnapshot;
use eframe::egui;
use math_explorer::biology::neuroscience::{
    HodgkinHuxleyModel, HodgkinHuxleyNeuron, HodgkinHuxleyParameters, HodgkinHuxleyState,
};
use math_explorer::pure_math::analysis::ode::OdeSystem;
use scientific_metadata::theory::ParameterConstraint;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Supported `(x, y)` parameter axis pairs for [`HodgkinHuxleyCartographer`].
pub const HH_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[("i_ext", "g_na"), ("i_ext", "g_k")];

const HH_ATLAS_DT: f64 = 0.02;
const HH_ATLAS_STEPS: usize = 1800;

/// Discrete action-potential excitability regimes of the Hodgkin-Huxley membrane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HodgkinHuxleyRegime {
    /// Sub-threshold resting equilibrium without action-potential threshold crossings (`0` spikes).
    Quiescent = 0,
    /// Phasic onset or depolarization-blocked transient firing (`1` spike).
    SingleSpike = 1,
    /// Sustained repetitive action-potential limit cycle (`>= 2` spikes).
    TonicFiring = 2,
}

impl HodgkinHuxleyRegime {
    /// Converts a raw `u8` regime code into [`Self`].
    #[inline]
    #[must_use]
    pub const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::Quiescent,
            1 => Self::SingleSpike,
            _ => Self::TonicFiring,
        }
    }

    /// Classifies an action-potential count over the integration window.
    #[inline]
    #[must_use]
    pub const fn from_spike_count(spikes: u32) -> Self {
        match spikes {
            0 => Self::Quiescent,
            1 => Self::SingleSpike,
            _ => Self::TonicFiring,
        }
    }

    /// Human-readable excitability regime label.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Quiescent => "Quiescent",
            Self::SingleSpike => "Single Spike",
            Self::TonicFiring => "Tonic Firing",
        }
    }
}

/// Zero-allocation 2D excitability cartographer for [`HodgkinHuxleyUnified`].
#[derive(Clone)]
pub struct HodgkinHuxleyCartographer {
    axes: AtlasAxisPair,
    template_params: HodgkinHuxleyParameters,
    base_i_ext: f64,
}

impl HodgkinHuxleyCartographer {
    /// Constructs a [`HodgkinHuxleyCartographer`] preallocating its gating-kinetics handle once.
    #[must_use]
    pub fn from_params(
        params: &HashMap<String, f64>,
        axis_keys: (&'static str, &'static str),
    ) -> Self {
        let theory = HodgkinHuxleyUnified::parameters();
        let spec_for = |key: &'static str| match key {
            "g_na" => AtlasAxisSpec::from_theory(
                &theory,
                "g_na",
                "Sodium Conductance g_Na (mS/cm²)",
                ParameterConstraint {
                    min: 0.0,
                    max: 200.0,
                    step: 1.0,
                },
            ),
            "g_k" => AtlasAxisSpec::from_theory(
                &theory,
                "g_k",
                "Potassium Conductance g_K (mS/cm²)",
                ParameterConstraint {
                    min: 0.0,
                    max: 100.0,
                    step: 1.0,
                },
            ),
            _ => AtlasAxisSpec::from_theory(
                &theory,
                "i_ext",
                "Stimulus Current I_ext (µA/cm²)",
                ParameterConstraint {
                    min: 0.0,
                    max: 50.0,
                    step: 1.0,
                },
            ),
        };
        let template_params = HodgkinHuxleyParameters {
            g_na: params.get("g_na").copied().unwrap_or(120.0).max(0.0),
            g_k: params.get("g_k").copied().unwrap_or(36.0).max(0.0),
            g_l: params.get("g_l").copied().unwrap_or(0.3).max(0.0),
            ..Default::default()
        };
        Self {
            axes: AtlasAxisPair {
                x: spec_for(axis_keys.0),
                y: spec_for(axis_keys.1),
            },
            template_params,
            base_i_ext: params.get("i_ext").copied().unwrap_or(10.0),
        }
    }
}

impl RegimeCartographer for HodgkinHuxleyCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let mut hh_params = self.template_params.clone();
        let mut i_ext = self.base_i_ext;
        for (key, val) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match key {
                "i_ext" => i_ext = val,
                "g_na" => hh_params.g_na = val.max(0.0),
                "g_k" => hh_params.g_k = val.max(0.0),
                "g_l" => hh_params.g_l = val.max(0.0),
                _ => {}
            }
        }
        let model = HodgkinHuxleyModel::new(hh_params, i_ext);
        let mut state = HodgkinHuxleyState {
            v: -65.0,
            n: 0.32,
            m: 0.05,
            h: 0.6,
        };
        let mut peak_v = state.v;
        let mut spikes = 0u32;
        let mut armed = true;

        for _ in 0..HH_ATLAS_STEPS {
            let d = model.derivative(0.0, &state);
            state.v += HH_ATLAS_DT * d.v;
            state.n = (state.n + HH_ATLAS_DT * d.n).clamp(0.0, 1.0);
            state.m = (state.m + HH_ATLAS_DT * d.m).clamp(0.0, 1.0);
            state.h = (state.h + HH_ATLAS_DT * d.h).clamp(0.0, 1.0);
            if !state.v.is_finite() {
                break;
            }
            peak_v = peak_v.max(state.v);
            if armed && state.v >= 0.0 {
                spikes += 1;
                armed = false;
            } else if !armed && state.v < -20.0 {
                armed = true;
            }
        }

        let regime = HodgkinHuxleyRegime::from_spike_count(spikes);
        AtlasCell {
            regime: regime as u8,
            primary: f64::from(spikes),
            secondary: peak_v,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = match HodgkinHuxleyRegime::from_u8(cell.regime) {
            HodgkinHuxleyRegime::Quiescent => ((cell.secondary + 65.0) / 65.0).clamp(0.0, 1.0),
            HodgkinHuxleyRegime::SingleSpike => ((cell.secondary + 20.0) / 70.0).clamp(0.25, 1.0),
            HodgkinHuxleyRegime::TonicFiring => (cell.primary / 6.0).clamp(0.2, 1.0),
        };
        atlas_regime_color(cell.regime, 3, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        HodgkinHuxleyRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        "Spike Count (36 ms)"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        format!(
            "{} = {:.2}, {} = {:.2}\nRegime: {}\nSpikes: {:.0}  |  Peak Voltage: {:.1} mV",
            self.axes.x.key,
            p1,
            self.axes.y.key,
            p2,
            self.regime_label(cell.regime),
            cell.primary,
            cell.secondary
        )
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        [
            HodgkinHuxleyRegime::Quiescent,
            HodgkinHuxleyRegime::SingleSpike,
            HodgkinHuxleyRegime::TonicFiring,
        ]
        .into_iter()
        .map(|r| (r.label(), atlas_regime_color(r as u8, 3, 0.7, high_contrast)))
        .collect()
    }
}

#[allow(missing_docs)]
pub struct HodgkinHuxleyUnified {
    neuron: HodgkinHuxleyNeuron,
    params: HodgkinHuxleyParameters,
    history: VecDeque<[f64; 2]>,
    time: f64,
    i_ext: f64,
    dt: f64,
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

        self.neuron.update(self.dt, self.i_ext);
        self.time += self.dt;
        self.history.push_back([self.time, self.neuron.v()]);
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

        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data,
            structured_data: None,
        }
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
