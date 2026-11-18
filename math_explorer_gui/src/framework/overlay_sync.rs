#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SimulationFramework;
use eframe::egui;

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
                self.overlay_sync.0,
                d.get_temp::<bool>(inst_portal),
                d.get_temp::<bool>(dom_portal),
                d.get_temp::<bool>(global_portal),
                same,
                any,
            );
            let g = resolve_scoped_overlay(
                self.show_starter_gallery,
                self.overlay_sync.1,
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
        self.overlay_sync.0 = Some(self.show_theory_portal);
        self.overlay_sync.1 = Some(self.show_starter_gallery);
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
