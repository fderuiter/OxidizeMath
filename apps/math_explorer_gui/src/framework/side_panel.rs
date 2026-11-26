#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::{announce_status, AccessibleHoverText};
use eframe::egui;

/// Maximum fraction of an enclosing pane's width a single [`PaneSidePanel`] may occupy.
pub const MAX_PANE_SIDE_PANEL_WIDTH_RATIO: f32 = 0.35;

const COLLAPSED_RAIL_WIDTH: f32 = 28.0;
const DEFAULT_EXPANDED_WIDTH: f32 = 280.0;
const DEFAULT_MIN_WIDTH: f32 = 96.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Left,
    Right,
}

/// Snapshot of a rendered [`PaneSidePanel`] recorded for layout and scroll verification.
#[derive(Clone, Copy, Debug)]
pub struct PaneSidePanelRecord {
    /// Scoped panel identifier passed to [`PaneSidePanel::left`] or [`PaneSidePanel::right`].
    pub id: egui::Id,
    /// Outer panel rectangle allocated inside the parent `Ui`.
    pub outer_rect: egui::Rect,
    /// Enclosing pane width used to compute the 35% cap.
    pub pane_width: f32,
    /// Whether the panel rendered in its collapsed rail state.
    pub collapsed: bool,
    /// Inner viewport height of the vertical [`egui::ScrollArea`] (`0.0` when collapsed).
    pub scroll_viewport_height: f32,
    /// Full scrollable content height inside the vertical [`egui::ScrollArea`].
    pub scroll_content_height: f32,
}

fn collapse_key(id: egui::Id) -> egui::Id {
    id.with("pane_side_panel_collapsed")
}

fn records_key() -> egui::Id {
    egui::Id::new("PANE_SIDE_PANEL_FRAME_RECORDS")
}

fn enclosing_pane_width(parent_ui: &egui::Ui) -> f32 {
    parent_ui
        .ctx()
        .data(|d| {
            d.get_temp::<egui::Rect>(egui::Id::new(crate::app::viewport::CURRENT_PANE_RECT_KEY))
        })
        .map(|r| r.width())
        .unwrap_or_else(|| parent_ui.max_rect().width())
        .max(48.0)
}

fn compute_outer_max_width(parent_ui: &egui::Ui, pane_width: f32) -> f32 {
    let avail_w = parent_ui.available_width().max(16.0);
    let min_canvas_w = (pane_width * 0.18).clamp(24.0, 72.0);
    // Subtract 1.0 px so pixel rounding in `SidePanel::show_inside` never exceeds 35%.
    let cap_by_pane = (pane_width * MAX_PANE_SIDE_PANEL_WIDTH_RATIO - 1.0).max(18.0);
    let cap_by_avail = (avail_w - min_canvas_w).max(16.0);
    cap_by_pane.min(cap_by_avail)
}

fn record_panel_metrics(ctx: &egui::Context, record: PaneSidePanelRecord) {
    ctx.data_mut(|d| {
        let mut list = d
            .get_temp::<Vec<PaneSidePanelRecord>>(records_key())
            .unwrap_or_default();
        if let Some(existing) = list.iter_mut().find(|r| r.id == record.id) {
            *existing = record;
        } else {
            list.push(record);
        }
        d.insert_temp(records_key(), list);
    });
}

/// Clears recorded [`PaneSidePanelRecord`] entries for the current test/frame inspection pass.
pub fn clear_pane_side_panel_records(ctx: &egui::Context) {
    ctx.data_mut(|d| d.remove::<Vec<PaneSidePanelRecord>>(records_key()));
}

/// Returns [`PaneSidePanelRecord`] snapshots recorded since the last clear.
pub fn pane_side_panel_records(ctx: &egui::Context) -> Vec<PaneSidePanelRecord> {
    ctx.data(|d| {
        d.get_temp::<Vec<PaneSidePanelRecord>>(records_key())
            .unwrap_or_default()
    })
}

/// Pane-scoped wrapper around [`egui::SidePanel`] providing vertical scrolling, a collapse toggle,
/// and a responsive `35%` enclosing-pane width cap that preserves positive central canvas width.
pub struct PaneSidePanel {
    id: egui::Id,
    side: Side,
    resizable: bool,
    default_width: Option<f32>,
    min_width: Option<f32>,
}

impl PaneSidePanel {
    /// Creates a left-docked pane side panel with a scoped `id`.
    pub fn left(id: impl Into<egui::Id>) -> Self {
        Self {
            id: id.into(),
            side: Side::Left,
            resizable: true,
            default_width: None,
            min_width: None,
        }
    }

    /// Creates a right-docked pane side panel with a scoped `id`.
    pub fn right(id: impl Into<egui::Id>) -> Self {
        Self {
            id: id.into(),
            side: Side::Right,
            resizable: true,
            default_width: None,
            min_width: None,
        }
    }

    /// Configures whether the expanded side panel can be resized by dragging its edge.
    pub fn resizable(mut self, resizable: bool) -> Self {
        self.resizable = resizable;
        self
    }

    /// Sets the preferred expanded width (clamped to `35%` of the enclosing pane width).
    pub fn default_width(mut self, default_width: f32) -> Self {
        self.default_width = Some(default_width);
        self
    }

    /// Sets the minimum expanded width (clamped to `35%` of the enclosing pane width).
    pub fn min_width(mut self, min_width: f32) -> Self {
        self.min_width = Some(min_width);
        self
    }

    /// Returns `true` when the panel identified by `id` is currently collapsed.
    pub fn is_collapsed(ctx: &egui::Context, id: egui::Id) -> bool {
        ctx.data_mut(|d| d.get_persisted::<bool>(collapse_key(id)).unwrap_or(false))
    }

    /// Sets the collapsed state of the panel identified by `id`.
    pub fn set_collapsed(ctx: &egui::Context, id: egui::Id, collapsed: bool) {
        ctx.data_mut(|d| d.insert_persisted(collapse_key(id), collapsed));
    }

    /// Renders the side panel inside `parent_ui`.
    pub fn show_inside<R>(
        self,
        parent_ui: &mut egui::Ui,
        add_contents: impl FnOnce(&mut egui::Ui) -> R,
    ) -> egui::InnerResponse<Option<R>> {
        let ctx = parent_ui.ctx().clone();
        let pane_width = enclosing_pane_width(parent_ui);
        let outer_max = compute_outer_max_width(parent_ui, pane_width);
        if Self::is_collapsed(&ctx, self.id) {
            self.show_collapsed_rail(parent_ui, &ctx, pane_width, outer_max)
        } else {
            self.show_expanded_panel(parent_ui, &ctx, pane_width, outer_max, add_contents)
        }
    }

    fn show_collapsed_rail<R>(
        self,
        parent_ui: &mut egui::Ui,
        ctx: &egui::Context,
        pane_width: f32,
        outer_max: f32,
    ) -> egui::InnerResponse<Option<R>> {
        let rail_id = self.id.with("collapsed_rail");
        let rail_w = COLLAPSED_RAIL_WIDTH.min(outer_max);
        let base = match self.side {
            Side::Left => egui::SidePanel::left(rail_id),
            Side::Right => egui::SidePanel::right(rail_id),
        };
        let inner = base
            .resizable(false)
            .exact_width(rail_w)
            .show_inside(parent_ui, |ui| {
                render_collapsed_toggle(ui, ctx, self.id, self.side);
                None
            });
        record_panel_metrics(
            ctx,
            PaneSidePanelRecord {
                id: self.id,
                outer_rect: inner.response.rect,
                pane_width,
                collapsed: true,
                scroll_viewport_height: 0.0,
                scroll_content_height: 0.0,
            },
        );
        inner
    }

    fn show_expanded_panel<R>(
        self,
        parent_ui: &mut egui::Ui,
        ctx: &egui::Context,
        pane_width: f32,
        outer_max: f32,
        add_contents: impl FnOnce(&mut egui::Ui) -> R,
    ) -> egui::InnerResponse<Option<R>> {
        let min_outer = self.min_width.unwrap_or(DEFAULT_MIN_WIDTH).min(outer_max);
        let default_outer = self
            .default_width
            .unwrap_or(DEFAULT_EXPANDED_WIDTH)
            .max(min_outer)
            .min(outer_max);
        let base = match self.side {
            Side::Left => egui::SidePanel::left(self.id),
            Side::Right => egui::SidePanel::right(self.id),
        };
        let mut viewport_h = 0.0;
        let mut content_h = 0.0;
        let inner = base
            .resizable(self.resizable)
            .default_width(default_outer)
            .width_range(min_outer..=outer_max)
            .show_inside(parent_ui, |ui| {
                let (out, vh, ch) =
                    render_expanded_contents(ui, ctx, self.id, self.side, add_contents);
                viewport_h = vh;
                content_h = ch;
                Some(out)
            });
        record_panel_metrics(
            ctx,
            PaneSidePanelRecord {
                id: self.id,
                outer_rect: inner.response.rect,
                pane_width,
                collapsed: false,
                scroll_viewport_height: viewport_h,
                scroll_content_height: content_h,
            },
        );
        inner
    }
}

fn render_collapsed_toggle(ui: &mut egui::Ui, ctx: &egui::Context, id: egui::Id, side: Side) {
    let bounded = ui.max_rect();
    let inner_w = bounded.width().max(1.0);
    let mut child_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(bounded)
            .layout(egui::Layout::top_down(egui::Align::Center)),
    );
    child_ui.set_clip_rect(bounded.intersect(ui.clip_rect()));
    let (glyph, label, status) = match side {
        Side::Left => ("»", "Expand left side panel", "Left side panel expanded"),
        Side::Right => ("«", "Expand right side panel", "Right side panel expanded"),
    };
    if child_ui
        .add(egui::Button::new(glyph).small())
        .accessible_hover_text(label)
        .clicked()
    {
        PaneSidePanel::set_collapsed(ctx, id, false);
        announce_status(status);
        ctx.request_repaint();
    }
    let used_h = (child_ui.min_rect().max.y - bounded.min.y).clamp(0.0, bounded.height());
    ui.advance_cursor_after_rect(egui::Rect::from_min_size(
        bounded.min,
        egui::vec2(inner_w, used_h),
    ));
}

fn seeded_vertical_scroll_state() -> egui::scroll_area::State {
    static SEEDED: std::sync::OnceLock<egui::scroll_area::State> = std::sync::OnceLock::new();
    *SEEDED.get_or_init(|| {
        let dummy_ctx = egui::Context::default();
        let mut captured = egui::scroll_area::State::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(120.0, 40.0),
            )),
            ..Default::default()
        };
        let _ = dummy_ctx.run(input, |c| {
            egui::CentralPanel::default().show(c, |ui| {
                let out = egui::ScrollArea::vertical()
                    .max_height(10.0)
                    .show(ui, |ui| {
                        ui.add_space(100.0);
                    });
                captured = out.state;
            });
        });
        captured.offset = egui::Vec2::ZERO;
        captured
    })
}

fn prime_compact_scroll_state(
    ui: &egui::Ui,
    ctx: &egui::Context,
    scroll_salt: (egui::Id, &str),
    height: f32,
) {
    if height >= 760.0 {
        return;
    }
    let scroll_id = ui.make_persistent_id(egui::Id::new(scroll_salt));
    if egui::scroll_area::State::load(ctx, scroll_id).is_some() {
        return;
    }
    seeded_vertical_scroll_state().store(ctx, scroll_id);
    let _ = ctx.animate_bool_with_time(scroll_id.with("v"), true, 0.0);
}

fn render_expanded_contents<R>(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    id: egui::Id,
    side: Side,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> (R, f32, f32) {
    let bounded = ui.max_rect();
    let inner_w = bounded.width().max(1.0);
    let mut child_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(bounded)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    child_ui.set_clip_rect(bounded.intersect(ui.clip_rect()));
    if inner_w < 190.0 {
        let spacing = child_ui.spacing_mut();
        spacing.slider_width = spacing.slider_width.min((inner_w * 0.45).max(16.0));
        spacing.combo_width = spacing.combo_width.min((inner_w * 0.72).max(24.0));
        spacing.interact_size.x = spacing.interact_size.x.min(inner_w);
    }

    render_collapse_header(&mut child_ui, ctx, id, side);
    let scroll_salt = (id, "pane_side_scroll");
    prime_compact_scroll_state(&child_ui, ctx, scroll_salt, child_ui.available_height());
    let scroll_out = egui::ScrollArea::vertical()
        .id_salt(scroll_salt)
        .auto_shrink([false, false])
        .show(&mut child_ui, add_contents);
    if scroll_out.content_size.y > scroll_out.inner_rect.height() {
        let _ = ctx.animate_bool_with_time(scroll_out.id.with("v"), true, 0.0);
    }

    let used_h = (child_ui.min_rect().max.y - bounded.min.y).clamp(0.0, bounded.height());
    ui.advance_cursor_after_rect(egui::Rect::from_min_size(
        bounded.min,
        egui::vec2(inner_w, used_h),
    ));
    (
        scroll_out.inner,
        scroll_out.inner_rect.height(),
        scroll_out.content_size.y,
    )
}

fn render_collapse_header(ui: &mut egui::Ui, ctx: &egui::Context, id: egui::Id, side: Side) {
    let (layout, glyph, label, status) = match side {
        Side::Left => (
            egui::Layout::right_to_left(egui::Align::Center),
            "«",
            "Collapse left side panel",
            "Left side panel collapsed",
        ),
        Side::Right => (
            egui::Layout::left_to_right(egui::Align::Center),
            "»",
            "Collapse right side panel",
            "Right side panel collapsed",
        ),
    };
    let row_size = egui::vec2(ui.available_width().max(1.0), ui.spacing().interact_size.y);
    ui.allocate_ui_with_layout(row_size, layout, |ui| {
        if ui
            .add(egui::Button::new(glyph).small())
            .accessible_hover_text(label)
            .clicked()
        {
            PaneSidePanel::set_collapsed(ctx, id, true);
            announce_status(status);
            ctx.request_repaint();
        }
    });
}
