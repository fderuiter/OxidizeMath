use std::string::String;

use egui::{
    Align, Color32, Direction, Frame, Id, Layout, Rect, Response, Sense, Shadow, Shape, TextStyle,
    Ui, Widget, WidgetInfo, WidgetType, epaint::CircleShape, pos2, vec2,
};

use super::items::PlotItem;

/// Where to place the plot legend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum Corner {
    LeftTop,
    RightTop,
    LeftBottom,
    RightBottom,
}

impl Corner {
    pub fn all() -> impl Iterator<Item = Self> {
        [
            Self::LeftTop,
            Self::RightTop,
            Self::LeftBottom,
            Self::RightBottom,
        ]
        .iter()
        .copied()
    }
}

/// How to handle multiple conflicting color for a legend item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum ColorConflictHandling {
    PickFirst,
    PickLast,
    RemoveColor,
}

/// How to group legend entries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum LegendGrouping {
    /// Items with the same name share a single legend entry (default).
    #[default]
    ByName,

    /// Each item gets its own legend entry, keyed by its unique [`Id`].
    ById,
}

/// The configuration for a plot legend.
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Legend {
    pub text_style: TextStyle,
    pub background_alpha: f32,
    pub position: Corner,
    pub title: Option<String>,

    follow_insertion_order: bool,
    grouping: LegendGrouping,
    color_conflict_handling: ColorConflictHandling,

    /// Used for overriding the `hidden_items` set in [`LegendWidget`].
    hidden_items: Option<ahash::HashSet<Id>>,
}

impl Default for Legend {
    fn default() -> Self {
        Self {
            text_style: TextStyle::Body,
            background_alpha: 0.75,
            position: Corner::RightTop,
            title: None,
            follow_insertion_order: false,
            grouping: LegendGrouping::default(),
            color_conflict_handling: ColorConflictHandling::RemoveColor,
            hidden_items: None,
        }
    }
}

impl Legend {
    /// Which text style to use for the legend. Default: `TextStyle::Body`.
    #[inline]
    pub fn text_style(mut self, style: TextStyle) -> Self {
        self.text_style = style;
        self
    }

    /// The alpha of the legend background. Default: `0.75`.
    #[inline]
    pub fn background_alpha(mut self, alpha: f32) -> Self {
        self.background_alpha = alpha;
        self
    }

    /// In which corner to place the legend. Default: `Corner::RightTop`.
    #[inline]
    pub fn position(mut self, corner: Corner) -> Self {
        self.position = corner;
        self
    }

    /// Set the title of the legend. Default: `None`.
    #[inline]
    pub fn title(mut self, title: &str) -> Self {
        self.title = Some(title.to_owned());
        self
    }

    /// Specifies hidden items in the legend configuration to override the existing ones. This
    /// allows the legend traces' visibility to be controlled from the application code.
    #[inline]
    pub fn hidden_items<I>(mut self, hidden_items: I) -> Self
    where
        I: IntoIterator<Item = Id>,
    {
        self.hidden_items = Some(hidden_items.into_iter().collect());
        self
    }

    /// Specifies if the legend item order should be the inserted order.
    /// Default: `false`.
    /// If `true`, the order of the legend items will be the same as the order as they were added.
    /// By default it will be sorted alphabetically.
    #[inline]
    pub fn follow_insertion_order(mut self, follow: bool) -> Self {
        self.follow_insertion_order = follow;
        self
    }

    /// Specifies how to handle conflicting colors for an item.
    #[inline]
    pub fn color_conflict_handling(
        mut self,
        color_conflict_handling: ColorConflictHandling,
    ) -> Self {
        self.color_conflict_handling = color_conflict_handling;
        self
    }

    /// Specifies how legend entries are grouped. Default: [`LegendGrouping::ByName`].
    ///
    /// With [`LegendGrouping::ByName`], items sharing the same name are
    /// merged into a single legend entry. With [`LegendGrouping::ById`],
    /// each item gets its own entry keyed by its unique [`Id`].
    #[inline]
    pub fn grouping(mut self, grouping: LegendGrouping) -> Self {
        self.grouping = grouping;
        self
    }
}

#[derive(Clone)]
struct LegendEntry {
    id: Id,
    name: String,
    color: Color32,
    checked: bool,
    hovered: bool,
}

impl LegendEntry {
    fn new(id: Id, name: String, color: Color32, checked: bool) -> Self {
        Self {
            id,
            name,
            color,
            checked,
            hovered: false,
        }
    }

    fn ui(&self, ui: &mut Ui, text_style: &TextStyle, min_row_height: f32) -> Response {
        let job = crate::latex::parse_latex_to_layout_job(
            &self.name,
            text_style.clone(),
            ui,
            ui.visuals().text_color(),
        );
        let font_height = ui.text_style_height(text_style).max(12.0);
        let icon_size = font_height;
        let icon_spacing = icon_size / 5.0;
        let max_text_width = (ui.available_width() - icon_size - icon_spacing - 4.0).max(32.0);
        let galley = egui::WidgetText::LayoutJob(std::sync::Arc::new(job)).into_galley(
            ui,
            Some(egui::TextWrapMode::Wrap),
            max_text_width,
            text_style.clone(),
        );

        let desired_size = vec2(
            (icon_size + icon_spacing + galley.size().x).max(24.0),
            galley.size().y.max(min_row_height),
        );
        let (rect, response) = ui.allocate_exact_size(desired_size, Sense::click());

        response.widget_info(|| {
            WidgetInfo::selected(
                WidgetType::Checkbox,
                ui.is_enabled(),
                self.checked,
                galley.text(),
            )
        });

        let visuals = ui.style().interact(&response);
        let label_on_the_left = ui.layout().horizontal_placement() == Align::RIGHT;
        self.paint_icon(ui, rect, icon_size, label_on_the_left, visuals);

        let text_x = if label_on_the_left {
            rect.right() - icon_size - icon_spacing - galley.size().x
        } else {
            rect.left() + icon_size + icon_spacing
        };
        let text_pos = pos2(text_x, rect.center().y - 0.5 * galley.size().y);
        ui.painter().galley(text_pos, galley, visuals.text_color());

        if response.has_focus() {
            let ring = crate::a11y::focus_ring_stroke(ui.visuals(), ui.visuals().extreme_bg_color);
            ui.painter()
                .rect_stroke(rect.expand(1.5), 2.0, ring, egui::StrokeKind::Outside);
        }

        response
    }

    fn paint_icon(
        &self,
        ui: &Ui,
        rect: Rect,
        icon_size: f32,
        label_on_the_left: bool,
        visuals: &egui::style::WidgetVisuals,
    ) {
        let icon_x = if label_on_the_left {
            rect.right() - icon_size / 2.0
        } else {
            rect.left() + icon_size / 2.0
        };
        let center = pos2(icon_x, rect.center().y);
        let painter = ui.painter();
        painter.add(CircleShape {
            center,
            radius: icon_size * 0.35,
            fill: visuals.bg_fill,
            stroke: visuals.bg_stroke,
        });
        if self.checked {
            let fill = if self.color == Color32::TRANSPARENT {
                ui.visuals().noninteractive().fg_stroke.color
            } else {
                self.color
            };
            painter.add(Shape::circle_filled(center, icon_size * 0.25, fill));
        }
    }
}

#[derive(Clone)]
pub(super) struct LegendWidget {
    plot_id: Id,
    rect: Rect,
    entries: Vec<LegendEntry>,
    config: Legend,
    rendered_rect: Option<Rect>,
}

impl LegendWidget {
    /// Create a new legend from items, the names of items that are hidden and the style of the
    /// text. Returns `None` if the legend has no entries.
    pub(super) fn try_new<'a>(
        plot_id: Id,
        rect: Rect,
        config: Legend,
        items: &[Box<dyn PlotItem + 'a>],
        hidden_items: &ahash::HashSet<Id>, // Existing hidden items in the plot memory.
    ) -> Option<Self> {
        // If `config.hidden_items` is not `None`, it is used.
        let hidden_items = config.hidden_items.as_ref().unwrap_or(hidden_items);

        // Collect the legend entries. With `ByName` grouping, items sharing the
        // same name are merged into a single checkbox. With `ById` grouping,
        // items sharing the same `Id` are merged instead. When colors conflict
        // within a merged entry, `color_conflict_handling` decides which color
        // to show.
        let mut entries: Vec<LegendEntry> = Vec::new();
        let mut seen: ahash::HashMap<Id, usize> = ahash::HashMap::default();
        for item in items.iter().filter(|item| !item.name().is_empty()) {
            let dedup_key = match config.grouping {
                LegendGrouping::ByName => Id::new(item.name()),
                LegendGrouping::ById => item.id(),
            };

            if let Some(&idx) = seen.get(&dedup_key) {
                let entry = &mut entries[idx];
                if entry.color != item.color() {
                    match config.color_conflict_handling {
                        ColorConflictHandling::PickFirst => (),
                        ColorConflictHandling::PickLast => entry.color = item.color(),
                        ColorConflictHandling::RemoveColor => {
                            entry.color = Color32::TRANSPARENT;
                        }
                    }
                }
            } else {
                seen.insert(dedup_key, entries.len());
                let color = item.color();
                let checked = !hidden_items.contains(&item.id());
                entries.push(LegendEntry::new(
                    item.id(),
                    item.name().to_owned(),
                    color,
                    checked,
                ));
            }
        }

        if !config.follow_insertion_order {
            entries.sort_by(|a, b| a.name.cmp(&b.name));
        }

        (!entries.is_empty()).then_some(Self {
            plot_id,
            rect,
            entries,
            config,
            rendered_rect: None,
        })
    }

    // Get the names of the hidden items.
    pub fn hidden_items(&self) -> ahash::HashSet<Id> {
        self.entries
            .iter()
            .filter_map(|entry| (!entry.checked).then_some(entry.id))
            .collect()
    }

    // Get the name of the hovered items.
    pub fn hovered_item(&self) -> Option<Id> {
        self.entries
            .iter()
            .find_map(|entry| entry.hovered.then_some(entry.id))
    }

    /// Corner placement configured for this legend.
    pub(crate) const fn position(&self) -> Corner {
        self.config.position
    }

    /// Bounding rectangle occupied by the rendered legend frame, once drawn.
    pub(crate) const fn rendered_rect(&self) -> Option<Rect> {
        self.rendered_rect
    }
}

impl Widget for &mut LegendWidget {
    fn ui(self, ui: &mut Ui) -> Response {
        let LegendWidget {
            plot_id,
            rect,
            entries,
            config,
            rendered_rect,
        } = self;
        let target_plot_id = *plot_id;

        let main_dir = match config.position {
            Corner::LeftTop | Corner::RightTop => Direction::TopDown,
            Corner::LeftBottom | Corner::RightBottom => Direction::BottomUp,
        };
        let cross_align = match config.position {
            Corner::LeftTop | Corner::LeftBottom => Align::LEFT,
            Corner::RightTop | Corner::RightBottom => Align::RIGHT,
        };
        let visible_h = rect.intersect(ui.clip_rect()).height();
        let full_targets_fit = visible_h >= (entries.len() as f32).mul_add(24.0, 16.0);
        let (outer_pad, inner_v_pad, row_spacing, min_row_h) = if full_targets_fit {
            (4.0, 4.0, 0.0, 24.0)
        } else {
            (2.0, 2.0, 1.0, 0.0)
        };
        let layout = Layout::from_main_dir_and_cross_align(main_dir, cross_align);
        let mut legend_ui = ui.new_child(
            egui::UiBuilder::new()
                .id_salt(("plot_legend", target_plot_id))
                .max_rect(rect.shrink(outer_pad))
                .layout(layout),
        );
        legend_ui
            .scope(|ui| {
                ui.spacing_mut().item_spacing.y = row_spacing;
                let background_frame = Frame {
                    inner_margin: vec2(8.0, inner_v_pad).into(),
                    corner_radius: ui.style().visuals.window_corner_radius,
                    shadow: Shadow::NONE,
                    fill: ui.style().visuals.extreme_bg_color,
                    stroke: ui.style().visuals.window_stroke(),
                    ..Default::default()
                }
                .multiply_with_opacity(config.background_alpha);
                let frame_resp = background_frame.show(ui, |ui| {
                    render_legend_frame_contents(
                        ui,
                        main_dir,
                        config,
                        entries,
                        target_plot_id,
                        min_row_h,
                    )
                });
                *rendered_rect = Some(frame_resp.response.rect);
                frame_resp.inner
            })
            .inner
    }
}

fn render_legend_frame_contents(
    ui: &mut Ui,
    main_dir: Direction,
    config: &Legend,
    entries: &mut [LegendEntry],
    target_plot_id: Id,
    min_row_h: f32,
) -> Response {
    if main_dir == Direction::TopDown
        && let Some(title) = &config.title
    {
        ui.heading(title);
    }
    let mut focus_on_item = None;
    let response_union = entries
        .iter_mut()
        .map(|entry| {
            let response = entry.ui(ui, &config.text_style, min_row_h);
            let isolate_triggered = check_legend_isolate_trigger(ui, &response);
            if isolate_triggered {
                focus_on_item = Some(entry.id);
            }
            handle_interaction_on_legend_item(ui.ctx(), &response, entry, isolate_triggered);
            response
        })
        .reduce(|r1, r2| r1.union(r2))
        .expect("No entries in the legend");

    if main_dir == Direction::BottomUp
        && let Some(title) = &config.title
    {
        ui.heading(title);
    }
    if let Some(focus_on_item) = focus_on_item {
        handle_focus_on_legend_item(ui.ctx(), &focus_on_item, entries);
    }
    response_union.context_menu(|ui| {
        if ui.button("Import CSV").clicked() {
            crate::export::trigger_import_csv_for_plot(ui.ctx(), target_plot_id);
            ui.close();
        }
    });
    response_union
}

fn check_legend_isolate_trigger(ui: &Ui, response: &Response) -> bool {
    let alt_enter_trigger =
        crate::commands::CommandTrigger::Shortcut(egui::Modifiers::ALT, egui::Key::Enter);
    crate::commands::CommandRegistryData::register(
        ui.ctx(),
        "Isolate Plot",
        "Alt-click or Alt+Enter to isolate or restore plot series",
        alt_enter_trigger.clone(),
        false,
        "Plot Legend",
    );
    let alt_enter = response.has_focus()
        && crate::commands::CommandRegistryData::register_and_check(
            ui.ctx(),
            "Isolate Plot",
            "Alt-click or Alt+Enter to isolate or restore plot series",
            alt_enter_trigger,
            false,
            "Plot Legend",
            Some(ui),
            Some(response),
        );
    let alt_click = crate::commands::CommandRegistryData::register_and_check(
        ui.ctx(),
        "Isolate Plot",
        "Alt-click or Alt+Enter to isolate or restore plot series",
        crate::commands::CommandTrigger::AltClick,
        true,
        "Plot Legend",
        Some(ui),
        Some(response),
    );
    alt_enter || alt_click
}

/// Handle per-entry interactions (keyboard Space/Enter, primary click, and focus highlight).
fn handle_interaction_on_legend_item(
    ctx: &egui::Context,
    response: &Response,
    entry: &mut LegendEntry,
    isolate_triggered: bool,
) {
    entry.hovered = response.hovered() || response.has_focus();
    if !isolate_triggered && response.clicked() {
        entry.checked = !entry.checked;
        let state_word = if entry.checked { "shown" } else { "hidden" };
        crate::emit_plot_live_message(ctx, format!("Series {} {state_word}", entry.name), false);
    }
}

/// Handle alt-click or Alt+Enter isolation/restoration across all legend entries.
fn handle_focus_on_legend_item(
    ctx: &egui::Context,
    clicked_entry: &Id,
    entries: &mut [LegendEntry],
) {
    let is_focus_item_only_visible = entries
        .iter()
        .all(|entry| !entry.checked || (clicked_entry == &entry.id));

    let mut isolated_name = String::new();
    for entry in entries.iter_mut() {
        if clicked_entry == &entry.id {
            isolated_name.clone_from(&entry.name);
        }
        entry.checked = is_focus_item_only_visible || clicked_entry == &entry.id;
    }

    let msg = if is_focus_item_only_visible {
        "All series shown".to_owned()
    } else {
        format!("Isolated series {isolated_name}")
    };
    crate::emit_plot_live_message(ctx, msg, false);
}
