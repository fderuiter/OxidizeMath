#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SimulationFramework;
use eframe::egui;

#[derive(Default, Clone, Copy)]
pub(crate) struct FrameworkSyncState {
    pub(super) theory_portal: Option<bool>,
    pub(super) starter_gallery: Option<bool>,
    pub(super) datasets_owner_tool: Option<&'static str>,
    pub(super) last_modal_scope_id: Option<egui::Id>,
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

        self.show_theory_portal = portal;
        self.show_starter_gallery = gallery;
        self.write_overlay_keys(ctx, id_source);
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
        let mut available_tools: Vec<&'static super::ToolMetadata> =
            inventory::iter::<super::ToolMetadata>
                .into_iter()
                .filter(|t| t.domain == domain)
                .collect();

        available_tools
            .sort_by_key(|t| (default_tool_priority(domain, t.id), t.display_name, t.id));

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
            export_options: egui_plot::export::ExportOptions::default(),
            export_bg_dark: true,
            export_gif_fps: 15,
            export_gif_frames: 20,
            export_status_message: None,
            export_status_severity: super::ExportStatusSeverity::Success,
            domain: domain.to_string(),
            preset_dialog: crate::presets::PresetDialogState::default(),
            export_queue: super::ExportQueue::default(),
            last_valid_snapshot: None,
            baseline_tool_state: None,
            current_valid_plot_datasets: Vec::new(),
            current_valid_plot_datasets_3d: Vec::new(),
            is_diverged: false,
            diagnostic_report: None,
            step_counter: 0,
            start_time: crate::web_io::PlatformInstant::now(),
            challenge_hud: super::ChallengeHudState::default(),
            active_challenge: None,
            challenge_was_passed: false,
            instance_id: next_framework_id(),
            sync_state: FrameworkSyncState::default(),
            rev_cache: super::FrameworkRevisionCache::default(),
        };

        if !framework.available_tools.is_empty() {
            framework.select_tool(0);
        }

        framework
    }
}

fn default_tool_priority(domain: &str, id: &str) -> u8 {
    match (domain, id) {
        ("quantum", "WaveSimulator")
        | ("solid_state", "IsingModelTool")
        | ("medical", "DoseCalculationTool")
        | ("relativity", "GeodesicPlaygroundTool")
        | ("elliptic_curves", "WeierstrassPlaygroundTool")
        | ("clinical_trials", "PharmacokineticsTool")
        | ("epidemiology", "SirTool")
        | ("ai", "LossLandscapeTool") => 0,
        _ => 1,
    }
}
