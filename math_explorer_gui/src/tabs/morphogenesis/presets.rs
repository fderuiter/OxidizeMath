#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::presets::{
    deserialize_preset_envelope, serialize_preset_envelope, MorphogenesisPresetConfig,
};

pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = self.state;
        let x = x ^ x >> 18;
        let rot = (x >> 27) as u32;
        let val = (x as u32).rotate_right(rot);
        (val as f64) / (u32::MAX as f64)
    }

    pub fn range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }
}

impl MorphogenesisTab {
    /// Domain category string used for preset envelope validation and filenames.
    pub fn domain(&self) -> &'static str {
        "biology"
    }

    /// Exports active simulation parameters into [`MorphogenesisPresetConfig`].
    pub fn export_config(&self) -> MorphogenesisPresetConfig {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        MorphogenesisPresetConfig {
            a: p.a,
            b: p.b,
            d_u: p.d_u,
            d_v: p.d_v,
            dt: p.dt,
            width: self.width,
            height: self.height,
        }
    }

    /// Applies imported [`MorphogenesisPresetConfig`] parameters to the simulation state.
    pub fn apply_config(&mut self, config: MorphogenesisPresetConfig) -> Result<(), String> {
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.a = config.a;
            p.b = config.b;
            p.d_u = config.d_u;
            p.d_v = config.d_v;
            p.dt = config.dt;
        }
        if config.width != self.width || config.height != self.height {
            self.width = config.width;
            self.height = config.height;
            let was_running = self.worker.is_running();
            let runner = Self::create_runner(&self.params, self.width, self.height);
            self.worker = BackgroundWorker::new(runner);
            if was_running {
                let _ = self.worker.send_command(SimCommand::Start);
            }
            self.dirty = true;
        } else {
            let _ = self.worker.send_command(SimCommand::Reset);
            self.dirty = true;
        }
        Ok(())
    }

    /// Serializes active simulation parameters to a formatted [`crate::presets::PresetWrapper`] JSON string.
    pub fn to_json_string(&self) -> Result<String, String> {
        serialize_preset_envelope(
            "Turing Morphogenesis Simulation",
            self.domain(),
            self.export_config(),
        )
    }

    /// Validates a [`crate::presets::PresetWrapper`] JSON string and applies its configuration.
    #[allow(clippy::wrong_self_convention)]
    pub fn from_json_str(&mut self, json_str: &str) -> Result<(), String> {
        let config: MorphogenesisPresetConfig =
            deserialize_preset_envelope(json_str, self.domain(), self.name())?;
        self.apply_config(config)
    }

    pub(crate) fn apply_starter_card(
        &mut self,
        card_title: &str,
        preset_json: Option<&str>,
    ) -> Result<(), String> {
        let Some(json_str) = preset_json else {
            return Ok(());
        };
        match self.from_json_str(json_str) {
            Ok(()) => {
                self.dialog_state.toast_error = None;
                Ok(())
            }
            Err(err) => {
                let msg = format!("Failed to launch '{}': {}", card_title, err);
                self.dialog_state.record_import_error(msg.clone());
                Err(msg)
            }
        }
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "turing_spots",
        title: "Turing Spots",
        description: "∂u/∂t = D_u∇²u + a - u + u²v: Activator-inhibitor reaction-diffusion generating spot morphology.",
        category: "Reaction-Diffusion",
        icon: "🐆",
        domain: "biology",
        tool_name: None,
        preset_generator: || crate::starter_gallery::PRESET_TURING_SPOTS,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "turing_stripes",
        title: "Turing Stripes",
        description: "∂v/∂t = D_v∇²v + b - u²v: Differential diffusion instability yielding labyrinthine stripe patterns.",
        category: "Morphogenesis",
        icon: "🦓",
        domain: "biology",
        tool_name: None,
        preset_generator: || crate::starter_gallery::PRESET_TURING_STRIPES,
    }
}
