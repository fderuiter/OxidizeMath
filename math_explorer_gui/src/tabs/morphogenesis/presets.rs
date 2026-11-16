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
        let p = self.params.read().unwrap();
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
            let mut p = self.params.write().unwrap();
            p.a = config.a;
            p.b = config.b;
            p.d_u = config.d_u;
            p.d_v = config.d_v;
            p.dt = config.dt;
        }
        if config.width != self.width || config.height != self.height {
            self.width = config.width;
            self.height = config.height;
            let kinetics = SchnakenbergKinetics {
                a: config.a,
                b: config.b,
            };
            let diffusion = FiniteDifference2D::new(
                math_explorer::math_kernel::types::Dimension(self.width),
                math_explorer::math_kernel::types::Dimension(self.height),
                math_explorer::math_kernel::types::StepSize(1.0),
                math_explorer::math_kernel::types::StepSize(1.0),
            );
            let mut system = TuringSystem::new_with_kinetics(
                math_explorer::math_kernel::types::Dimension(self.width * self.height),
                math_explorer::biology::morphogenesis::DiffusionCoeff(config.d_u),
                math_explorer::biology::morphogenesis::DiffusionCoeff(config.d_v),
                kinetics,
                diffusion,
            );
            Self::initialize_system(&mut system, self.width, self.height);
            let runner = MorphogenesisRunner {
                system,
                params: Arc::clone(&self.params),
                width: self.width,
                height: self.height,
            };
            self.controller = BackgroundWorker::new(runner);
            if !self.paused {
                let _ = self.controller.send_command(SimCommand::Start);
            }
            self.dirty = true;
        } else {
            let _ = self.controller.send_command(SimCommand::Reset);
            self.dirty = true;
        }
        Ok(())
    }
}
