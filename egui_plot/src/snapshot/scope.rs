//! Scoped plot persistence synchronization between framework containers and [`crate::Plot::show`].

use std::collections::BTreeMap;

use egui::{Context, Id, Vec2b};

use crate::{AccessState, PlotBounds, PlotMemory};

use super::PlotOverlaySnapshot;

#[derive(Clone, Copy)]
struct ActivePlotScope {
    scope_id: u64,
    generation: u64,
}

#[derive(Clone, Default)]
struct ScopePlotStore {
    seeded_generation: Option<u64>,
    snapshots: BTreeMap<String, PlotOverlaySnapshot>,
    applied: BTreeMap<String, (u64, Id)>,
}

fn active_scope_stack_id() -> Id {
    Id::new("egui_plot_active_persistence_scope_stack")
}

fn scope_store_id(scope_id: u64) -> Id {
    Id::new(("egui_plot_persistence_scope_store", scope_id))
}

/// Enters a scoped plot persistence region for `scope_id`, seeding `snapshots` when `generation` advances.
pub fn begin_plot_persistence_scope(
    ctx: &Context,
    scope_id: u64,
    generation: u64,
    seed_snapshots: &BTreeMap<String, PlotOverlaySnapshot>,
) {
    ctx.data_mut(|d| {
        let stack: &mut Vec<ActivePlotScope> = d.get_temp_mut_or_default(active_scope_stack_id());
        stack.push(ActivePlotScope {
            scope_id,
            generation,
        });
        let store: &mut ScopePlotStore = d.get_temp_mut_or_default(scope_store_id(scope_id));
        if store.seeded_generation != Some(generation) {
            store.snapshots = seed_snapshots.clone();
            store.seeded_generation = Some(generation);
            store.applied.clear();
        }
    });
}

/// Exits the scoped plot persistence region for `scope_id` and returns updated non-default plot snapshots.
#[must_use]
pub fn end_plot_persistence_scope(
    ctx: &Context,
    scope_id: u64,
) -> BTreeMap<String, PlotOverlaySnapshot> {
    ctx.data_mut(|d| {
        let stack: &mut Vec<ActivePlotScope> = d.get_temp_mut_or_default(active_scope_stack_id());
        if let Some(pos) = stack.iter().rposition(|s| s.scope_id == scope_id) {
            stack.remove(pos);
        }
        let store: &mut ScopePlotStore = d.get_temp_mut_or_default(scope_store_id(scope_id));
        store.snapshots.clone()
    })
}

pub(crate) struct BeforeShowSyncParams<'a> {
    pub plot_id: Id,
    pub plot_key: &'a str,
    pub had_existing_mem: bool,
    pub default_auto_bounds: Vec2b,
    pub min_auto_bounds: PlotBounds,
    pub allow_bounds_persist: bool,
    pub allow_2d_analysis: bool,
}

pub(crate) fn sync_plot_before_show(
    ctx: &Context,
    params: &BeforeShowSyncParams<'_>,
    mem: &mut PlotMemory,
) {
    let active = ctx.data_mut(|d| {
        d.get_temp_mut_or_default::<Vec<ActivePlotScope>>(active_scope_stack_id())
            .last()
            .copied()
    });
    let Some(active) = active else {
        return;
    };

    let snap_to_apply = ctx.data_mut(|d| {
        let store: &mut ScopePlotStore = d.get_temp_mut_or_default(scope_store_id(active.scope_id));
        if params.allow_2d_analysis
            && !store.snapshots.contains_key(params.plot_key)
            && let Some(wildcard) = store.snapshots.remove("*")
        {
            store.snapshots.insert(params.plot_key.to_owned(), wildcard);
            store.applied.remove(params.plot_key);
        }
        let already = store.applied.get(params.plot_key).copied()
            == Some((active.generation, params.plot_id))
            && params.had_existing_mem;
        if already {
            None
        } else {
            store.applied.insert(
                params.plot_key.to_owned(),
                (active.generation, params.plot_id),
            );
            Some(
                store
                    .snapshots
                    .get(params.plot_key)
                    .cloned()
                    .unwrap_or_default(),
            )
        }
    });

    if let Some(snap) = snap_to_apply {
        apply_snapshot_to_memory_and_access(ctx, params, mem, &snap);
    }
}

fn apply_snapshot_to_memory_and_access(
    ctx: &Context,
    params: &BeforeShowSyncParams<'_>,
    mem: &mut PlotMemory,
    snap: &PlotOverlaySnapshot,
) {
    mem.hidden_items = snap.hidden_items.iter().map(|id| id.0).collect();
    mem.imported_datasets = snap.imported_datasets.clone();
    mem.callouts = if params.allow_2d_analysis {
        snap.callouts.clone()
    } else {
        Vec::new()
    };
    mem.log_x = snap.log_x;
    mem.log_y = snap.log_y;
    mem.grid_density = snap.grid_density.clamp(4.0, 100.0);
    mem.show_x_labels = snap.show_x_labels;
    mem.show_y_labels = snap.show_y_labels;
    mem.clamp_grid = snap.clamp_grid;
    mem.show_grid = Vec2b::new(snap.show_grid[0], snap.show_grid[1]);
    mem.transform.set_log_axis(Vec2b::new(mem.log_x, mem.log_y));

    if params.allow_bounds_persist {
        if let Some(saved_bounds) = snap.bounds.filter(|b| b.is_valid() && b.is_finite()) {
            let ab = snap.auto_bounds.unwrap_or([false, false]);
            mem.auto_bounds = Vec2b::new(ab[0], ab[1]);
            mem.set_bounds(saved_bounds);
        } else {
            mem.auto_bounds = params.default_auto_bounds;
            mem.set_bounds(params.min_auto_bounds);
        }
    }

    ctx.data_mut(|d| {
        let mut access = d
            .get_temp::<AccessState>(params.plot_id)
            .unwrap_or_default();
        access.show_table = snap.show_table;
        access.fit_model = if params.allow_2d_analysis {
            snap.fit_model
        } else {
            None
        };
        access.fit_log_mode = snap.fit_log_mode;
        access.show_fit_residuals = snap.show_fit_residuals;
        access.fit_dataset_idx = snap.fit_dataset_idx;
        access.show_fit_overlay = snap.show_fit_overlay;
        access.custom_fit_formula = snap.custom_fit_formula.clone();
        access.custom_fit_seeds_text = snap.custom_fit_seeds_text.clone();
        access.custom_fit_error = None;
        access.cached_fit = None;
        access.cached_pois = None;
        access.log_x = snap.log_x;
        access.log_y = snap.log_y;
        access.grid_density = mem.grid_density;
        access.show_x_labels = snap.show_x_labels;
        access.show_y_labels = snap.show_y_labels;
        access.clamp_grid = snap.clamp_grid;
        d.insert_temp(params.plot_id, access);
    });
}

pub(crate) struct AfterShowSyncParams<'a> {
    pub plot_key: &'a str,
    pub default_auto_bounds: Vec2b,
    pub min_auto_bounds: PlotBounds,
    pub has_data_aspect: bool,
    pub allow_bounds_persist: bool,
    pub allow_2d_analysis: bool,
    pub scene_3d_mode: bool,
}

pub(crate) fn sync_plot_after_show(
    ctx: &Context,
    params: &AfterShowSyncParams<'_>,
    mem: &PlotMemory,
    access: &AccessState,
) {
    let active = ctx.data_mut(|d| {
        d.get_temp_mut_or_default::<Vec<ActivePlotScope>>(active_scope_stack_id())
            .last()
            .copied()
    });
    let Some(active) = active else {
        return;
    };

    let mut snap = PlotOverlaySnapshot::capture(
        mem,
        access,
        params.default_auto_bounds,
        params.min_auto_bounds,
        params.has_data_aspect,
        params.allow_bounds_persist,
        params.allow_2d_analysis,
    );
    if params.scene_3d_mode {
        snap.log_x = false;
        snap.log_y = false;
        snap.show_grid = [true, true];
    }
    ctx.data_mut(|d| {
        let store: &mut ScopePlotStore = d.get_temp_mut_or_default(scope_store_id(active.scope_id));
        if snap.is_default() {
            store.snapshots.remove(params.plot_key);
        } else {
            store.snapshots.insert(params.plot_key.to_owned(), snap);
        }
    });
}
