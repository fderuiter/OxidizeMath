#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SimulationFramework;
use eframe::egui;

#[derive(Default, Clone, Copy)]
pub(crate) struct FrameworkSyncState {
    pub(super) theory_portal: Option<bool>,
    pub(super) starter_gallery: Option<bool>,
    pub(super) datasets_owner_tool: Option<&'static str>,
    pub(super) last_modal_scope_id: Option<egui::Id>,
    pub(super) theory_portal_trigger: Option<egui::Id>,
    pub(super) default_theory_toggle: Option<egui::Id>,
}

pub(crate) fn record_default_theory_portal_toggle(ctx: &egui::Context, toggle_id: egui::Id) {
    ctx.data_mut(|d| {
        if let Some(inst_key) =
            d.get_temp::<egui::Id>(egui::Id::new("CURRENT_THEORY_PORTAL_INST_ID"))
        {
            d.insert_temp(inst_key.with("default_toggle_id"), toggle_id);
        }
    });
}

pub(crate) fn record_theory_portal_toggle(
    ctx: &egui::Context,
    trigger_id: egui::Id,
    next_open: bool,
) {
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new("SHOW_THEORY_PORTAL"), next_open);
        if let Some(inst_key) =
            d.get_temp::<egui::Id>(egui::Id::new("CURRENT_THEORY_PORTAL_INST_ID"))
        {
            d.insert_temp(inst_key, next_open);
            if next_open {
                d.insert_temp(inst_key.with("trigger_id"), trigger_id);
            }
        }
    });
    ctx.request_repaint();
}

#[derive(Clone, Copy)]
pub(crate) struct UntrackedUrlDigestCache {
    pub(super) tool_name: &'static str,
    pub(super) checked_untracked_rev: u64,
    pub(super) url_digest: u64,
}

#[derive(Default)]
pub(crate) struct FrameworkRevisionCache {
    pub(super) framework_revision: u64,
    pub(super) untracked_tool_revision: u64,
    pub(super) untracked_url_digest: std::cell::Cell<Option<UntrackedUrlDigestCache>>,
    pub(super) active_tool_json:
        std::cell::RefCell<Option<(&'static str, u64, std::sync::Arc<str>)>>,
    pub(super) last_snapshot_rev: Option<(&'static str, u64)>,
    pub(super) last_snapshot_source_datasets: Vec<(String, egui_plot::SharedPoints)>,
    pub(super) last_plot_capture_hash: Option<(&'static str, u64)>,
}

pub(super) fn next_framework_id() -> u64 {
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn resolve_scoped_overlay(
    self_val: bool,
    last_synced: Option<bool>,
    inst_val: Option<bool>,
    domain_val: Option<bool>,
    global_val: Option<bool>,
    is_same_owner: bool,
    has_any_owner: bool,
) -> bool {
    if last_synced.is_some() && last_synced != Some(self_val) {
        return self_val;
    }
    if last_synced.is_none() {
        if self_val {
            return true;
        }
        return inst_val.unwrap_or_else(|| {
            if !has_any_owner {
                domain_val.or(global_val).unwrap_or(self_val)
            } else {
                self_val
            }
        });
    }
    if let Some(v) = inst_val.filter(|&v| Some(v) != last_synced) {
        return v;
    }
    if is_same_owner {
        if let Some(v) = domain_val.filter(|&v| Some(v) != last_synced) {
            return v;
        }
        if let Some(v) = global_val.filter(|&v| Some(v) != last_synced) {
            return v;
        }
    }
    self_val
}

impl SimulationFramework {
    pub(super) fn sync_overlays_before_frame(&mut self, ctx: &egui::Context, id_source: &str) {
        let owner_id = egui::Id::new("FRAMEWORK_OVERLAY_OWNER");
        let inst_portal = egui::Id::new(("SHOW_THEORY_PORTAL_INST", id_source, self.instance_id));
        let inst_gallery =
            egui::Id::new(("SHOW_STARTER_GALLERY_INST", id_source, self.instance_id));
        let dom_portal = egui::Id::new("SHOW_THEORY_PORTAL").with(id_source);
        let dom_gallery = egui::Id::new("SHOW_STARTER_GALLERY").with(id_source);
        let global_portal = egui::Id::new("SHOW_THEORY_PORTAL");
        let global_gallery = egui::Id::new("SHOW_STARTER_GALLERY");

        let (portal, gallery) = ctx.data(|d| {
            let owner = d.get_temp::<(String, u64)>(owner_id);
            let same = owner
                .as_ref()
                .is_some_and(|(src, id)| src == id_source && *id == self.instance_id);
            let any = owner.is_some();
            let p = resolve_scoped_overlay(
                self.show_theory_portal,
                self.sync_state.theory_portal,
                d.get_temp::<bool>(inst_portal),
                d.get_temp::<bool>(dom_portal),
                d.get_temp::<bool>(global_portal),
                same,
                any,
            );
            let g = resolve_scoped_overlay(
                self.show_starter_gallery,
                self.sync_state.starter_gallery,
                d.get_temp::<bool>(inst_gallery),
                d.get_temp::<bool>(dom_gallery),
                d.get_temp::<bool>(global_gallery),
                same,
                any,
            );
            (p, g)
        });

        let prev_portal = self.sync_state.theory_portal.unwrap_or(false);
        self.show_theory_portal = portal;
        self.show_starter_gallery = gallery;
        if portal && !prev_portal {
            self.capture_theory_portal_trigger(ctx, id_source);
        }
        self.write_overlay_keys(ctx, id_source);
    }

    pub(super) fn capture_theory_portal_trigger(&mut self, ctx: &egui::Context, id_source: &str) {
        let inst_portal = egui::Id::new(("SHOW_THEORY_PORTAL_INST", id_source, self.instance_id));
        let (recorded_trigger, default_toggle) = ctx.data(|d| {
            (
                d.get_temp::<egui::Id>(inst_portal.with("trigger_id")),
                d.get_temp::<egui::Id>(inst_portal.with("default_toggle_id")),
            )
        });
        if let Some(def_id) = default_toggle {
            self.sync_state.default_theory_toggle = Some(def_id);
        }
        if let Some(trig_id) = recorded_trigger {
            self.sync_state.theory_portal_trigger = Some(trig_id);
        } else if let Some(focused) = ctx.memory(|m| m.focused()) {
            self.sync_state.theory_portal_trigger = Some(focused);
        }
    }

    pub(super) fn refresh_default_theory_toggle(&mut self, ctx: &egui::Context, id_source: &str) {
        let inst_portal = egui::Id::new(("SHOW_THEORY_PORTAL_INST", id_source, self.instance_id));
        let (recorded_trigger, default_toggle) = ctx.data_mut(|d| {
            let key = inst_portal.with("trigger_id");
            let trig = d.get_temp::<egui::Id>(key);
            d.remove::<egui::Id>(key);
            let def = d.get_temp::<egui::Id>(inst_portal.with("default_toggle_id"));
            (trig, def)
        });
        if let Some(def_id) = default_toggle {
            self.sync_state.default_theory_toggle = Some(def_id);
        }
        if let Some(trig_id) = recorded_trigger {
            self.sync_state.theory_portal_trigger = Some(trig_id);
        }
    }

    pub(super) fn close_theory_portal_and_restore_focus(
        &mut self,
        ctx: &egui::Context,
        id_source: &str,
    ) {
        self.show_theory_portal = false;
        self.sync_state.theory_portal = Some(false);
        self.write_overlay_keys(ctx, id_source);
        let restore_id = self
            .sync_state
            .theory_portal_trigger
            .take()
            .or(self.sync_state.default_theory_toggle);
        if let Some(target_id) = restore_id {
            ctx.memory_mut(|m| m.request_focus(target_id));
        }
        crate::accessibility::announce_live_status(ctx, "Theory Context Portal closed");
        ctx.request_repaint();
    }

    pub(super) fn sync_overlays_after_frame(&mut self, ctx: &egui::Context, id_source: &str) {
        let global_portal = egui::Id::new("SHOW_THEORY_PORTAL");
        let global_gallery = egui::Id::new("SHOW_STARTER_GALLERY");
        if let Some(p) = ctx.data(|d| d.get_temp::<bool>(global_portal)) {
            self.show_theory_portal = p;
        }
        if let Some(g) = ctx.data(|d| d.get_temp::<bool>(global_gallery)) {
            self.show_starter_gallery = g;
        }
        self.write_overlay_keys(ctx, id_source);
        self.sync_state.theory_portal = Some(self.show_theory_portal);
        self.sync_state.starter_gallery = Some(self.show_starter_gallery);
        self.sync_counterfactual_snapshot();
    }

    /// Synchronizes `branch_engine` with the active tool's latest counterfactual divergence snapshot.
    pub fn sync_counterfactual_snapshot(&mut self) {
        let snap = self
            .active_tool
            .as_ref()
            .and_then(|tool| tool.counterfactual_snapshot());
        self.branch_engine.sync_snapshot(snap);
    }

    /// Dispatches a `CounterfactualBranchEngine` control action to the active tool and refreshes `branch_engine`.
    pub fn dispatch_branch_action(&mut self, action: super::BranchControlAction) -> bool {
        if let super::BranchControlAction::SetPerturbation(delta) = action {
            self.branch_engine.set_effective_delta(delta);
        }
        let applied = self
            .active_tool
            .as_mut()
            .is_some_and(|tool| tool.apply_counterfactual_action(action));
        self.sync_counterfactual_snapshot();
        applied
    }

    pub(crate) fn sync_starter_gallery_state(&mut self, ctx: &egui::Context, id_source: &str) {
        self.sync_state.starter_gallery = Some(self.show_starter_gallery);
        let owner_id = egui::Id::new("FRAMEWORK_OVERLAY_OWNER");
        let inst_gallery =
            egui::Id::new(("SHOW_STARTER_GALLERY_INST", id_source, self.instance_id));
        let dom_gallery = egui::Id::new("SHOW_STARTER_GALLERY").with(id_source);
        ctx.data_mut(|d| {
            d.insert_temp(inst_gallery, self.show_starter_gallery);
            let is_owner = d
                .get_temp::<(String, u64)>(owner_id)
                .is_some_and(|(src, id)| src == id_source && id == self.instance_id);
            if is_owner {
                d.insert_temp(dom_gallery, self.show_starter_gallery);
                d.insert_temp(
                    egui::Id::new("SHOW_STARTER_GALLERY"),
                    self.show_starter_gallery,
                );
            }
        });
    }

    fn write_overlay_keys(&self, ctx: &egui::Context, id_source: &str) {
        let owner_id = egui::Id::new("FRAMEWORK_OVERLAY_OWNER");
        let inst_portal = egui::Id::new(("SHOW_THEORY_PORTAL_INST", id_source, self.instance_id));
        let inst_gallery =
            egui::Id::new(("SHOW_STARTER_GALLERY_INST", id_source, self.instance_id));
        let dom_portal = egui::Id::new("SHOW_THEORY_PORTAL").with(id_source);
        let dom_gallery = egui::Id::new("SHOW_STARTER_GALLERY").with(id_source);
        ctx.data_mut(|d| {
            d.insert_temp(owner_id, (id_source.to_string(), self.instance_id));
            d.insert_temp(inst_portal, self.show_theory_portal);
            d.insert_temp(dom_portal, self.show_theory_portal);
            d.insert_temp(egui::Id::new("SHOW_THEORY_PORTAL"), self.show_theory_portal);
            d.insert_temp(inst_gallery, self.show_starter_gallery);
            d.insert_temp(dom_gallery, self.show_starter_gallery);
            d.insert_temp(
                egui::Id::new("SHOW_STARTER_GALLERY"),
                self.show_starter_gallery,
            );
        });
    }

    /// Constructs a [`SimulationFramework`] populated with registered tools for `domain`.
    pub fn new(domain: &str) -> Self {
        let available_tools = sorted_domain_tools(domain);
        let mut framework = Self {
            available_tools,
            active_tool: None,
            selected_tool_index: None,
            input_mode: super::InputMode::Mouse,
            show_theory_portal: false,
            show_starter_gallery: false,
            saved_tool_states: std::collections::HashMap::new(),
            saved_plot_states: std::collections::BTreeMap::new(),
            plot_sync_generation: 1,
            search_query: String::new(),
            last_announced_count: None,
            last_announced_query: String::new(),
            show_export_modal: false,
            show_code_export_modal: false,
            code_export_tab: super::CodeExportTab::default(),
            export_options: egui_plot::export::ExportOptions::default(),
            export_bg_dark: true,
            export_gif_fps: 15,
            export_gif_frames: 20,
            export_status_message: None,
            export_status_severity: super::ExportStatusSeverity::Success,
            domain: domain.to_string(),
            preset_dialog: crate::presets::PresetDialogState::default(),
            export_queue: super::ExportQueue::default(),
            gif_job: None,
            last_valid_snapshot: None,
            baseline_tool_state: None,
            current_valid_plot_datasets: Vec::new(),
            current_valid_plot_datasets_3d: Vec::new(),
            is_diverged: false,
            diagnostic_report: None,
            step_counter: 0,
            branch_engine: super::CounterfactualBranchEngine::default(),
            start_time: crate::web_io::PlatformInstant::now(),
            challenge_hud: super::ChallengeHudState::default(),
            active_challenge: None,
            challenge_was_passed: false,
            instance_id: next_framework_id(),
            sync_state: FrameworkSyncState::default(),
            rev_cache: FrameworkRevisionCache::default(),
            atlas_host: super::atlas_host::FrameworkAtlasHost::default(),
        };
        if !framework.available_tools.is_empty() {
            framework.select_tool(0);
        }
        framework
    }
}

fn sorted_domain_tools(domain: &str) -> Vec<&'static super::ToolMetadata> {
    let mut available_tools: Vec<&'static super::ToolMetadata> =
        inventory::iter::<super::ToolMetadata>
            .into_iter()
            .filter(|t| t.domain == domain)
            .collect();
    available_tools.sort_by_key(|t| (default_tool_priority(domain, t.id), t.display_name, t.id));
    available_tools
}

fn default_tool_priority(domain: &str, id: &str) -> u8 {
    match (domain, id) {
        ("quantum", "WaveSimulator")
        | ("solid_state", "IsingModelTool")
        | ("medical", "DoseCalculationTool")
        | ("relativity", "GeodesicPlaygroundTool")
        | ("optics", "CausticsRayBenchTool")
        | ("elliptic_curves", "WeierstrassPlaygroundTool")
        | ("clinical_trials", "PharmacokineticsTool")
        | ("epidemiology", "SirTool")
        | ("ai", "LossLandscapeTool") => 0,
        _ => 1,
    }
}
