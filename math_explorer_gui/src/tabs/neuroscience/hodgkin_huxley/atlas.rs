#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::HodgkinHuxleyUnified;
use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer,
};
use crate::async_sim::unified::UnifiedModel;
use eframe::egui;
use math_explorer::biology::neuroscience::{
    HodgkinHuxleyModel, HodgkinHuxleyParameters, HodgkinHuxleyState,
};
use math_explorer::pure_math::analysis::ode::OdeSystem;
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;

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
                    default: Some(120.0),
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
                    default: Some(36.0),
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
                    default: Some(10.0),
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
        .map(|r| {
            (
                r.label(),
                atlas_regime_color(r as u8, 3, 0.7, high_contrast),
            )
        })
        .collect()
    }
}
