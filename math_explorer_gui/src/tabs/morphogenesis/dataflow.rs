//! `PaneDataflowBus` 2D scalar field publishing for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::MorphogenesisTool;
use super::runner::MorphoSnapshotData;
use crate::app::dataflow_bus::{ScalarField2DPayload, publish_pane_scalar_field};
use crate::async_sim::StateSnapshot;
use eframe::egui;

const DEFAULT_FIELD_DOMAIN: [f64; 2] = [-2.0, 2.0];

/// Tracks downsampled concentration grids published by `MorphogenesisTool` onto `PaneDataflowBus`.
#[derive(Default, Clone)]
pub(crate) struct MorphoFieldStreamState {
    revision: u64,
    latest: Option<ScalarField2DPayload>,
}

impl MorphoFieldStreamState {
    pub(crate) fn record_snapshot(&mut self, snapshot: &StateSnapshot) {
        let Some(data) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<MorphoSnapshotData>())
        else {
            return;
        };
        let next_rev = self.revision.wrapping_add(1).max(1);
        if let Some(payload) = ScalarField2DPayload::from_grid_downsampled(
            next_rev,
            snapshot.width,
            snapshot.height,
            &data.u_data,
            DEFAULT_FIELD_DOMAIN,
            DEFAULT_FIELD_DOMAIN,
        ) {
            self.revision = next_rev;
            self.latest = Some(payload);
        }
    }

    pub(crate) fn invalidate(&mut self) {
        self.latest = None;
    }
}

impl MorphogenesisTool {
    pub(crate) fn ensure_seeded_scalar_field(&mut self) {
        if self.field_stream.latest.is_some() {
            return;
        }
        if let Some(snap) = self.worker.latest_snapshot() {
            self.field_stream.record_snapshot(snap);
            if self.field_stream.latest.is_some() {
                return;
            }
        }
        let runner = Self::create_kinetics_runner(
            &self.params,
            &self.kinetics_state,
            &self.mask_rle,
            self.width,
            self.height,
        );
        let snap = crate::async_sim::SimulationRunner::get_snapshot(&runner);
        self.field_stream.record_snapshot(&snap);
    }

    pub(crate) fn publish_bus_scalar_field(&mut self, ctx: &egui::Context) {
        self.ensure_seeded_scalar_field();
        if let Some(ref payload) = self.field_stream.latest {
            publish_pane_scalar_field(ctx, payload.clone());
        }
    }

    pub(crate) fn invalidate_scalar_field_cache(&mut self) {
        self.field_stream.invalidate();
    }

    pub(crate) fn advance_morpho_step(&mut self) {
        if self.is_forked() {
            self.step_lockstep(5);
            let snap = crate::async_sim::SimulationRunner::get_snapshot(&self.local_runner);
            self.field_stream.record_snapshot(&snap);
            return;
        }
        let runner = self.step_runner.get_or_insert_with(|| {
            Self::create_kinetics_runner(
                &self.params,
                &self.kinetics_state,
                &self.mask_rle,
                self.width,
                self.height,
            )
        });
        for _ in 0..5 {
            crate::async_sim::SimulationRunner::step(runner);
        }
        let snap = crate::async_sim::SimulationRunner::get_snapshot(runner);
        self.field_stream.record_snapshot(&snap);
        let px = snap.pixels.read().unwrap_or_else(|e| e.into_inner());
        self.stepped_pixels = Some((snap.width, snap.height, px.clone()));
    }

    pub(crate) fn export_morpho_raster_image(&self) -> Option<egui::ColorImage> {
        if self.is_forked() {
            return Some(self.render_current_heatmap_image());
        }
        if let Some((w, h, ref px)) = self.stepped_pixels
            && w > 0
            && h > 0
            && px.len() == w * h
        {
            return Some(egui::ColorImage {
                size: [w, h],
                pixels: px.clone(),
                ..Default::default()
            });
        }
        if !self.dirty && self.pixels.len() == self.width * self.height && !self.pixels.is_empty() {
            return Some(egui::ColorImage {
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
        Some(egui::ColorImage {
            size: [snapshot.width, snapshot.height],
            pixels: px.clone(),
            ..Default::default()
        })
    }
}
