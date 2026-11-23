//! `PaneDataflowBus` 2D scalar field publishing for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::MorphoSnapshotData;
use super::MorphogenesisTool;
use crate::app::dataflow_bus::{publish_pane_scalar_field, ScalarField2DPayload};
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
}
