//! Cold-path scalar-field and raster export helpers for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::MorphogenesisBoundaryMode;
use super::tool::MorphogenesisTool;
use crate::async_sim::SimulationRunner;
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
            .kinetics_state
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
        let runner = Self::create_kinetics_runner(
            &self.params,
            &self.kinetics_state,
            self.width,
            self.height,
        );
        Some(ScalarField2D {
            width: self.width,
            height: self.height,
            values: runner.active_concentration_slice().to_vec(),
            periodic,
            default_levels: 8,
        })
    }

    pub(super) fn step_simulation_offscreen(&mut self) {
        let runner = self.step_runner.get_or_insert_with(|| {
            Self::create_kinetics_runner(
                &self.params,
                &self.kinetics_state,
                self.width,
                self.height,
            )
        });
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::InteractiveTool;

    #[test]
    fn test_morphogenesis_gray_scott_periodic_state_and_scalar_field() {
        let mut tool = MorphogenesisTool::default();
        let json = r#"{"a":0.1,"b":0.9,"d_u":0.16,"d_v":0.08,"dt":1.0,"width":64,"height":64,"kinetics_mode":"GrayScott","boundary":"Periodic","f":0.0367,"k":0.0649,"paused":true}"#;
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
