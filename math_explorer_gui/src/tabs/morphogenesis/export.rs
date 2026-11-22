//! Cold-path scalar-field, raster, and state serialization helpers for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::tool::{
    MorphogenesisBoundaryMode, MorphogenesisKineticsMode, MorphogenesisTool, MorphogenesisToolState,
};
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState, SimulationRunner};
use crate::framework::InteractiveTool;
use eframe::egui::ColorImage;
use math_commons::math_kernel::marching_squares::ScalarField2D;

impl MorphogenesisTool {
    pub(super) fn build_export_raster_image(&self) -> Option<ColorImage> {
        if let Some((w, h, ref px)) = self.stepped_pixels {
            if w > 0 && h > 0 && px.len() == w * h {
                return Some(ColorImage {
                    size: [w, h],
                    pixels: px.clone(),
                    ..Default::default()
                });
            }
        }
        if !self.dirty && self.pixels.len() == self.width * self.height && !self.pixels.is_empty() {
            return Some(ColorImage {
                size: [self.width, self.height],
                pixels: self.pixels.clone(),
                ..Default::default()
            });
        }
        let snapshot = self.worker.latest_snapshot()?;
        let px = snapshot.pixels.read().unwrap_or_else(|e| e.into_inner());
        if snapshot.width == 0
            || snapshot.height == 0
            || px.len() != snapshot.width * snapshot.height
        {
            return None;
        }
        Some(ColorImage {
            size: [snapshot.width, snapshot.height],
            pixels: px.clone(),
            ..Default::default()
        })
    }

    pub(super) fn build_export_scalar_field(&self) -> Option<ScalarField2D> {
        let periodic = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .boundary
            == MorphogenesisBoundaryMode::Periodic;
        if let Some((w, h, ref values)) = self.stepped_scalar {
            if w > 0 && h > 0 && values.len() == w * h {
                return Some(ScalarField2D {
                    width: w,
                    height: h,
                    values: values.clone(),
                    periodic,
                    default_levels: 8,
                });
            }
        }
        if let Some(snapshot) = self.worker.latest_snapshot() {
            if let Some(values) = snapshot
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<Vec<f64>>())
                .filter(|v| v.len() == snapshot.width * snapshot.height && snapshot.width > 0)
            {
                return Some(ScalarField2D {
                    width: snapshot.width,
                    height: snapshot.height,
                    values: values.clone(),
                    periodic,
                    default_levels: 8,
                });
            }
        }
        let runner = Self::create_runner(&self.params, self.width, self.height);
        Some(ScalarField2D {
            width: self.width,
            height: self.height,
            values: runner.active_concentration_slice().to_vec(),
            periodic,
            default_levels: 8,
        })
    }

    pub(super) fn step_simulation_offscreen(&mut self) {
        let runner = self
            .step_runner
            .get_or_insert_with(|| Self::create_runner(&self.params, self.width, self.height));
        for _ in 0..5 {
            runner.step();
        }
        let snap = runner.get_snapshot();
        let px = snap.pixels.read().unwrap_or_else(|e| e.into_inner());
        self.stepped_pixels = Some((snap.width, snap.height, px.clone()));
        if let Some(vals) = snap
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<Vec<f64>>())
        {
            self.stepped_scalar = Some((snap.width, snap.height, vals.clone()));
        }
    }

    pub(super) fn snapshot_tool_state(&self) -> MorphogenesisToolState {
        let (a, b, d_u, d_v, dt, kinetics, boundary, feed_rate, kill_rate) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (
                p.a,
                p.b,
                p.d_u,
                p.d_v,
                p.dt,
                p.kinetics,
                p.boundary,
                p.feed_rate,
                p.kill_rate,
            )
        };
        MorphogenesisToolState {
            a,
            b,
            d_u,
            d_v,
            dt,
            width: self.width,
            height: self.height,
            kinetics,
            boundary,
            feed_rate,
            kill_rate,
            state: Some(self.worker.state()),
            paused: None,
            pending_initial_start: self.pending_initial_start,
        }
    }

    pub(super) fn apply_parsed_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: MorphogenesisToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Morphogenesis state: {e}"))?;
        parsed.validate()?;
        let target_state = if let Some(s) = parsed.state {
            s
        } else if parsed.paused.unwrap_or(false) {
            SimWorkerState::Ready
        } else {
            SimWorkerState::Running
        };
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.a = parsed.a;
            p.b = parsed.b;
            p.d_u = parsed.d_u;
            p.d_v = parsed.d_v;
            p.dt = parsed.dt;
            p.kinetics = parsed.kinetics;
            p.boundary = parsed.boundary;
            p.feed_rate = parsed.feed_rate;
            p.kill_rate = parsed.kill_rate;
        }
        if parsed.width != self.width || parsed.height != self.height {
            self.width = parsed.width;
            self.height = parsed.height;
            let runner = Self::create_runner(&self.params, self.width, self.height);
            self.worker = BackgroundWorker::new(runner);
            self.stepped_pixels = None;
            self.stepped_scalar = None;
            self.step_runner = None;
            self.dirty = true;
        } else {
            self.reset();
        }
        if target_state == SimWorkerState::Running {
            self.pending_initial_start = false;
            if !self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Start);
            }
        } else {
            self.pending_initial_start = parsed.pending_initial_start;
            if self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Pause);
            }
        }
        Ok(())
    }

    pub(crate) fn apply_named_preset(&mut self, a: f64, b: f64, d_u: f64, d_v: f64, dt: f64) {
        let target_state = if self.worker.is_running() {
            SimWorkerState::Running
        } else {
            SimWorkerState::Ready
        };
        let boundary = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .boundary;
        let preset_state = MorphogenesisToolState {
            a,
            b,
            d_u,
            d_v,
            dt,
            width: self.width,
            height: self.height,
            kinetics: MorphogenesisKineticsMode::Schnakenberg,
            boundary,
            feed_rate: 0.037,
            kill_rate: 0.060,
            state: Some(target_state),
            paused: None,
            pending_initial_start: false,
        };
        if let Ok(json) = serde_json::to_string(&preset_state) {
            let _ = self.try_load_state(&json);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morphogenesis_gray_scott_periodic_state_and_scalar_field() {
        let mut tool = MorphogenesisTool::default();
        let json = r#"{"a":0.1,"b":0.9,"d_u":1.0,"d_v":40.0,"dt":0.05,"width":64,"height":64,"kinetics":"GrayScott","boundary":"Periodic","feed_rate":0.035,"kill_rate":0.065,"paused":true}"#;
        tool.try_load_state(json)
            .expect("valid Gray-Scott periodic state");
        for _ in 0..8 {
            tool.step_simulation();
        }
        let field = tool.export_scalar_field().expect("scalar field exported");
        assert_eq!((field.width, field.height), (64, 64));
        assert!(field.periodic);
        assert!(field.values.iter().all(|v| v.is_finite()));
        let saved = tool.save_state().expect("serialized state");
        assert!(saved.contains("GrayScott") && saved.contains("Periodic"));
    }
}
