#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::presets::{InteractivePreset, MorphogenesisPresetConfig};

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

impl InteractivePreset for MorphogenesisTab {
    type Config = MorphogenesisPresetConfig;

    fn domain(&self) -> &'static str {
        "biology"
    }

    fn title(&self) -> String {
        "Turing Morphogenesis Simulation".to_string()
    }

    fn export_config(&self) -> Self::Config {
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

    fn apply_config(&mut self, config: Self::Config) -> Result<(), String> {
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
}

impl MorphogenesisTab {
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
