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
}
