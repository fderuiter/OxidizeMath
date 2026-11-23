//! Global starter catalog routing, cold-boot lifecycle guards, and pane launch helpers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MathExplorerApp, SplitLayoutMode, ViewportLayout};
use crate::starter_gallery::{
    ChallengeCard, GallerySelection, GoalChallengeMetadata, PresetStoryMetadata, StarterCard,
};
use eframe::egui;

impl MathExplorerApp {
    /// Finds the `tabs` index matching a `ToolMetadata` or `StarterPresetMetadata` domain slug.
    #[must_use]
    pub fn find_tab_index_for_domain(&self, domain_slug: &str) -> Option<usize> {
        self.tabs.iter().position(|tab| {
            crate::url_state::matches_tab_slug(domain_slug, tab.url_slug(), tab.name())
        })
    }

    /// Closes any open per-pane `SimulationFramework` starter gallery flags.
    pub(super) fn close_pane_starter_galleries(&mut self) {
        for tab in &mut self.tabs {
            if let Some(fw) = tab.framework_mut() {
                fw.show_starter_gallery = false;
            }
        }
        for pane_idx in 1..4 {
            if let Some(slot) = self.pane_slots.slot_mut(pane_idx) {
                if let Some((_, ref mut tab)) = slot.active {
                    if let Some(fw) = tab.framework_mut() {
                        fw.show_starter_gallery = false;
                    }
                }
            }
        }
    }

    /// Opens the global Starter Template Gallery across `All Domains` and announces it.
    pub fn open_explore_catalog(&mut self, ctx: &egui::Context) {
        self.command_palette.close();
        self.close_pane_starter_galleries();
        self.starter_gallery.open_all(ctx);
        self.show_starter_gallery = true;
        self.last_synced_show_starter_gallery = true;
        ctx.request_repaint();
    }

    fn sync_starter_gallery_flags(&mut self, ctx: &egui::Context) {
        if self.show_starter_gallery != self.last_synced_show_starter_gallery {
            if self.show_starter_gallery {
                self.starter_gallery.open_all(ctx);
            } else {
                self.starter_gallery.close_silently();
            }
            self.last_synced_show_starter_gallery = self.show_starter_gallery;
        }
    }

    fn update_synced_gallery_bool(&mut self) {
        self.show_starter_gallery =
            self.starter_gallery.is_open && !self.starter_gallery.is_cold_boot;
        self.last_synced_show_starter_gallery = self.show_starter_gallery;
    }

    /// Dismisses the cold-boot starter gallery if workspace state was altered away from defaults.
    pub(super) fn sync_cold_boot_starter_gallery(&mut self) {
        if !self.starter_gallery.is_cold_boot {
            return;
        }
        if self.selected_tab != 0
            || self.viewport_layout != ViewportLayout::Single
            || self.split_layout_mode != SplitLayoutMode::Single
            || self.command_palette.is_open
            || self.show_help_menu
            || self.show_diagnostics_panel
        {
            self.starter_gallery.dismiss_cold_boot();
            self.update_synced_gallery_bool();
        }
    }

    /// Registers and checks `Ctrl+Shift+G` / `Cmd+Shift+G` to toggle the global `Explore Catalog`.
    pub(super) fn poll_explore_catalog_shortcut(&mut self, ctx: &egui::Context) {
        let base_mods = if cfg!(target_os = "macos") {
            egui::Modifiers::MAC_CMD
        } else {
            egui::Modifiers::CTRL
        };
        let shift_mods = base_mods | egui::Modifiers::SHIFT;
        let triggered = egui_plot::commands::CommandRegistryData::register_and_check(
            ctx,
            "Explore Catalog",
            "Open global starter template gallery across all simulation domains",
            egui_plot::commands::CommandTrigger::Shortcut(shift_mods, egui::Key::G),
            true,
            "Global",
            None,
            None,
        );
        if triggered {
            if self.starter_gallery.is_open {
                self.starter_gallery.close_silently();
                self.update_synced_gallery_bool();
                crate::accessibility::announce_live_status(ctx, "Starter template gallery closed");
            } else {
                self.open_explore_catalog(ctx);
            }
        }
    }

    /// Renders the root `GlobalStarterGallery` modal and routes any selected item to `focused_pane`.
    pub(super) fn render_global_starter_gallery(&mut self, ctx: &egui::Context) {
        self.sync_starter_gallery_flags(ctx);
        self.sync_cold_boot_starter_gallery();
        if self.command_palette.is_open && self.starter_gallery.is_open {
            self.starter_gallery.close_silently();
            self.update_synced_gallery_bool();
            return;
        }
        let target_pane = self.active_focused_pane();
        let selection = self.starter_gallery.show(ctx);
        self.update_synced_gallery_bool();
        if let Some(sel) = selection {
            self.dispatch_gallery_selection_to_pane(ctx, target_pane, sel);
        }
    }

    /// Routes a root-hoisted per-pane `StarterGallery` selection to `pane_idx`.
    pub(crate) fn apply_deferred_starter_gallery_selection(
        &mut self,
        ctx: &egui::Context,
        pane_idx: usize,
        id_source: &str,
        is_open: bool,
        selected: Option<GallerySelection>,
    ) {
        let Some(selection) = selected else {
            if let Some(tab) = self.pane_tab_mut(pane_idx) {
                tab.apply_starter_gallery_selection(ctx, id_source, is_open, None);
            }
            return;
        };
        if let Some(tab) = self.pane_tab_mut(pane_idx) {
            tab.apply_starter_gallery_selection(ctx, id_source, false, None);
        }
        self.dispatch_gallery_selection_to_pane(ctx, pane_idx, selection);
    }

    fn dispatch_gallery_selection_to_pane(
        &mut self,
        ctx: &egui::Context,
        pane_idx: usize,
        selection: GallerySelection,
    ) {
        match selection {
            GallerySelection::Starter(card) => {
                let _ = self.launch_starter_card_in_pane(ctx, pane_idx, &card);
            }
            GallerySelection::Story(story) => {
                let _ = self.launch_story_in_pane(ctx, pane_idx, story);
            }
            GallerySelection::Challenge(ch) => {
                let _ = self.launch_challenge_in_pane(ctx, pane_idx, &ch);
            }
            GallerySelection::GoalChallenge(goal) => {
                let _ = self.launch_goal_challenge_in_pane(ctx, pane_idx, goal);
            }
        }
    }

    fn switch_pane_to_domain(
        &mut self,
        pane_idx: usize,
        domain: &str,
        kind_label: &str,
    ) -> Result<usize, String> {
        self.starter_gallery.close_silently();
        self.update_synced_gallery_bool();
        let tab_idx = self.find_tab_index_for_domain(domain).ok_or_else(|| {
            format!("No simulation module registered for {kind_label} domain '{domain}'")
        })?;
        let pane_count = self.viewport_layout.pane_count();
        let target_pane = if pane_idx < pane_count {
            pane_idx
        } else {
            self.active_focused_pane()
        };
        self.focus_pane(target_pane);
        if target_pane == 0 && self.pane_tabs.is_empty() {
            self.selected_tab = tab_idx;
        } else {
            self.set_pane_tab(target_pane, tab_idx);
        }
        Ok(target_pane)
    }

    fn finalize_pane_launch(&mut self, ctx: &egui::Context, target_pane: usize) {
        if let Some(slot) = self.pane_slots.slot_mut(target_pane) {
            slot.snapshot_active();
        }
        self.clear_active_pane_caches();
        ctx.request_repaint();
    }

    /// Routes `card` to `focused_pane`, switching domains, activating the tool, and loading its preset.
    pub fn launch_starter_card_in_focused_pane(
        &mut self,
        ctx: &egui::Context,
        card: &StarterCard,
    ) -> Result<(), String> {
        let target_pane = self.active_focused_pane();
        self.launch_starter_card_in_pane(ctx, target_pane, card)
    }

    /// Routes `card` to `pane_idx`, switching to the target `ExplorerTab` and importing the preset JSON.
    pub fn launch_starter_card_in_pane(
        &mut self,
        ctx: &egui::Context,
        pane_idx: usize,
        card: &StarterCard,
    ) -> Result<(), String> {
        let target_pane = self.switch_pane_to_domain(pane_idx, card.domain, "starter preset")?;
        let res = {
            let tab = self
                .pane_tab_mut(target_pane)
                .ok_or_else(|| format!("Target pane {target_pane} is unavailable"))?;
            tab.launch_starter_card(card).map(|()| tab.name())
        };
        self.finalize_pane_launch(ctx, target_pane);

        match res {
            Ok(tab_name) => {
                let msg = format!("Launched starter preset {} in {}", card.title, tab_name);
                crate::accessibility::announce_live_status(ctx, msg);
                Ok(())
            }
            Err(err) => {
                let msg = format!("Failed to launch starter preset {}: {}", card.title, err);
                crate::accessibility::announce_live_assertive_status(ctx, &msg);
                Err(err)
            }
        }
    }

    /// Routes a multi-step guided `PresetStoryMetadata` tour to `pane_idx` at step 0.
    pub fn launch_story_in_pane(
        &mut self,
        ctx: &egui::Context,
        pane_idx: usize,
        story: &'static PresetStoryMetadata,
    ) -> Result<(), String> {
        let target_pane = self.switch_pane_to_domain(pane_idx, story.domain, "guided story")?;
        let res = {
            let tab = self
                .pane_tab_mut(target_pane)
                .ok_or_else(|| format!("Target pane {target_pane} is unavailable"))?;
            let tab_name = tab.name();
            let out = {
                let fw = tab
                    .framework_mut()
                    .ok_or_else(|| format!("Tab '{tab_name}' does not support guided stories"))?;
                let res = fw.launch_story(story, 0);
                fw.flush_hud_announcements(ctx);
                res
            };
            if out.is_ok() {
                tab.sync_after_framework_launch();
            }
            out
        };
        self.finalize_pane_launch(ctx, target_pane);
        res
    }

    /// Routes an inverse `ChallengeCard` to `pane_idx`, switching to the target `ExplorerTab` and activating its baseline.
    pub fn launch_challenge_in_pane(
        &mut self,
        ctx: &egui::Context,
        pane_idx: usize,
        ch: &ChallengeCard,
    ) -> Result<(), String> {
        let target_pane = self.switch_pane_to_domain(pane_idx, ch.domain, "challenge")?;
        let res = {
            let tab = self
                .pane_tab_mut(target_pane)
                .ok_or_else(|| format!("Target pane {target_pane} is unavailable"))?;
            let tab_name = tab.name();
            let out = {
                let fw = tab.framework_mut().ok_or_else(|| {
                    format!("Tab '{tab_name}' does not support inverse challenges")
                })?;
                fw.launch_challenge_by_id(ch.id, Some(ctx))
            };
            if out.is_ok() {
                tab.sync_after_framework_launch();
            }
            out
        };
        self.finalize_pane_launch(ctx, target_pane);
        res
    }

    /// Routes a playable AST `GoalChallengeMetadata` to `pane_idx`, switching tabs and arming the HUD.
    pub fn launch_goal_challenge_in_pane(
        &mut self,
        ctx: &egui::Context,
        pane_idx: usize,
        goal: &'static GoalChallengeMetadata,
    ) -> Result<(), String> {
        let target_pane = self.switch_pane_to_domain(pane_idx, goal.domain, "goal challenge")?;
        let res = {
            let tab = self
                .pane_tab_mut(target_pane)
                .ok_or_else(|| format!("Target pane {target_pane} is unavailable"))?;
            let tab_name = tab.name();
            let out = {
                let fw = tab
                    .framework_mut()
                    .ok_or_else(|| format!("Tab '{tab_name}' does not support goal challenges"))?;
                let res = fw.launch_challenge(goal);
                fw.flush_hud_announcements(ctx);
                res
            };
            if out.is_ok() {
                tab.sync_after_framework_launch();
            }
            out
        };
        self.finalize_pane_launch(ctx, target_pane);
        res
    }
}
