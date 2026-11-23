#![doc = include_str!("../README.md")]
#![cfg_attr(feature = "document-features", doc = document_features::document_features!())]
//!

#![allow(missing_docs)]
#![allow(
    clippy::too_many_lines,
    clippy::cognitive_complexity,
    clippy::too_many_arguments
)]

mod a11y;
mod axis;
pub mod commands;
pub mod csv_import;
mod data_table;
pub mod export;
pub mod export_modal;
pub mod export_primitives;
mod file_io;
mod fit;
mod fit_export;
mod fit_overlay;
mod format;
mod items;
mod latex;
mod legend;
mod memory;
mod modal_lifecycle;
mod plot_overlays;
mod plot_ui;
pub mod poi;
mod poi_interaction;
pub mod sidecar_3d;
pub mod snapshot;
mod transform;

use std::{cmp::Ordering, ops::RangeInclusive, sync::Arc};

use crate::a11y::AccessibleTooltipExt as _;
use ahash::HashMap;
use egui::{
    Align2, Color32, CursorIcon, Id, Layout, NumExt as _, PointerButton, Pos2, Rangef, Rect,
    Response, Sense, Shape, Stroke, TextStyle, Ui, Vec2, Vec2b, WidgetText, epaint, remap_clamp,
    vec2,
};
use emath::Float as _;

pub use crate::{
    axis::{Axis, AxisHints, HPlacement, Placement, VPlacement},
    export::{
        AspectRatioOption, ExportOptions, ImportedDataset, MAX_EXPORT_DIMENSION,
        color32_image_to_png, color32_images_to_gif, copy_image_to_clipboard,
        copy_image_to_clipboard_with_callback, copy_text_to_clipboard,
        copy_text_to_clipboard_with_callback, export_csv, export_csv_3d, export_custom_gif,
        export_custom_image, export_custom_png, export_custom_svg, export_json, export_json_3d,
        parse_csv, save_file_dialog, trigger_import_csv_for_plot,
    },
    export_modal::{
        CallbackExportSource, ExportDataSource, ExportModalAction, ExportModalBindings,
        ExportModalConfig, ExportModalState, clipboard_status_id, show_export_modal,
    },
    export_primitives::{
        ExportPrimitive, append_last_plot_primitives, clear_last_plot_primitives,
        export_primitives_svg, export_raster_gif, export_raster_svg, get_last_plot_primitives,
        render_primitives_image,
    },
    fit::{
        CustomEvaluatorHandle, CustomFitCallback, CustomFitEvaluator, FitError, FitResult, FitType,
        MAX_FIT_PARAMS, fit_custom_evaluator, fit_model_with_axes,
    },
    fit_overlay::{
        fit_custom_with_registered_callback, has_custom_fit_callback, register_custom_fit_callback,
    },
    items::{
        Arrows, Bar, BarChart, BoxElem, BoxPlot, BoxSpread, Callout, ClosestElem, HLine, Line,
        LineStyle, MarkerShape, Orientation, PlotConfig, PlotGeometry, PlotImage, PlotItem,
        PlotItemBase, PlotPoint, PlotPoints, Points, Polygon, SavedCallout, Text, VLine,
    },
    latex::{parse_latex_to_layout_job, parse_prose_with_inline_math, render_math_label},
    legend::{ColorConflictHandling, Corner, Legend, LegendGrouping},
    memory::{CanvasCallout, PlotMemory},
    modal_lifecycle::{ModalFrameContext, ModalLifecycle},
    plot_ui::PlotUi,
    poi::{CriticalPoint, CriticalPointKind, MAX_CRITICAL_POINTS, detect_critical_points},
    sidecar_3d::{
        Plot3DSeries, Plot3DSeriesSidecar, append_last_plot_datasets_3d,
        clear_last_plot_datasets_3d, format_world_coord, get_last_plot_datasets_3d,
    },
    snapshot::{
        OrderedId, PlotOverlaySnapshot, begin_plot_persistence_scope, end_plot_persistence_scope,
    },
    transform::{PlotBounds, PlotTransform},
};

#[doc(hidden)]
pub use self::{
    export_primitives::{
        materialized_export_primitives_call_count, reset_materialized_export_primitives_call_count,
    },
    fit::{fit_model_with_axes_call_count, reset_fit_model_with_axes_call_count},
    poi::{detect_critical_points_call_count, reset_detect_critical_points_call_count},
};

use axis::AxisWidget;
use items::{horizontal_line, rulers_color, vertical_line};
use legend::LegendWidget;

/// Callback signature invoked whenever `egui_plot` dispatches a polite or assertive live announcement.
pub type PlotLiveRegionAnnouncer = fn(&egui::Context, &str, bool);

static PLOT_LIVE_ANNOUNCER: std::sync::RwLock<Option<PlotLiveRegionAnnouncer>> =
    std::sync::RwLock::new(None);

/// Registers an application-level live-region announcer callback for `egui_plot` announcements.
pub fn set_live_region_announcer(announcer: PlotLiveRegionAnnouncer) {
    if let Ok(mut slot) = PLOT_LIVE_ANNOUNCER.write() {
        *slot = Some(announcer);
    }
}

/// Dispatches a polite (`assertive = false`) or assertive (`assertive = true`) plot live message.
pub fn emit_plot_live_message(ctx: &egui::Context, message: String, assertive: bool) {
    let announcer = PLOT_LIVE_ANNOUNCER.read().ok().and_then(|slot| *slot);
    if let Some(callback) = announcer {
        callback(ctx, &message, assertive);
    } else {
        let key = if assertive {
            "aria_live_assertive_message"
        } else {
            "aria_live_message"
        };
        ctx.data_mut(|d| d.insert_temp(Id::new(key), message));
    }
}

type LabelFormatterFn<'a> = dyn Fn(&str, &PlotPoint) -> String + 'a;
pub type LabelFormatter<'a> = Option<Box<LabelFormatterFn<'a>>>;

type GridSpacerFn<'a> = dyn Fn(GridInput) -> Vec<GridMark> + 'a;
type GridSpacer<'a> = Box<GridSpacerFn<'a>>;

type CoordinatesFormatterFn<'a> = dyn Fn(&PlotPoint, &PlotBounds) -> String + 'a;

/// Specifies the coordinates formatting when passed to [`Plot::coordinates_formatter`].
pub struct CoordinatesFormatter<'a> {
    function: Box<CoordinatesFormatterFn<'a>>,
}

impl<'a> CoordinatesFormatter<'a> {
    /// Create a new formatter based on the pointer coordinate and the plot bounds.
    pub fn new(function: impl Fn(&PlotPoint, &PlotBounds) -> String + 'a) -> Self {
        Self {
            function: Box::new(function),
        }
    }

    /// Show a fixed number of decimal places.
    pub fn with_decimals(num_decimals: usize) -> Self {
        Self {
            function: Box::new(move |value, _| {
                format!("x: {:.d$}\ny: {:.d$}", value.x, value.y, d = num_decimals)
            }),
        }
    }

    fn format(&self, value: &PlotPoint, bounds: &PlotBounds) -> String {
        (self.function)(value, bounds)
    }
}

impl Default for CoordinatesFormatter<'_> {
    fn default() -> Self {
        Self::with_decimals(3)
    }
}

// ----------------------------------------------------------------------------

/// Indicates a vertical or horizontal cursor line in plot coordinates.
#[derive(Copy, Clone, PartialEq)]
pub enum Cursor {
    Horizontal { y: f64 },
    Vertical { x: f64 },
}

/// Contains the cursors drawn for a plot widget in a single frame.
#[derive(PartialEq, Clone)]
struct PlotFrameCursors {
    id: Id,
    cursors: Vec<Cursor>,
}

#[derive(Default, Clone)]
struct CursorLinkGroups(HashMap<Id, Vec<PlotFrameCursors>>);

/// Workspace-wide ambient synchronization flags for multi-pane 2D comparison plots.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmbientPlotLinkConfig {
    /// Synchronizes X and Y coordinate bounds across matching interactive 2D plots.
    pub link_axes: bool,
    /// Synchronizes hover crosshair cursors across matching interactive 2D plots.
    pub link_cursors: bool,
}

#[derive(Default, Clone, Copy)]
struct AmbientPlotLinkState {
    config: AmbientPlotLinkConfig,
    axis_epoch: u64,
}

/// Publishes or clears the ambient multi-pane plot synchronization configuration on `ctx`.
pub fn set_ambient_plot_link_config(ctx: &egui::Context, config: AmbientPlotLinkConfig) {
    ctx.data_mut(|d| {
        let state: &mut AmbientPlotLinkState =
            d.get_temp_mut_or_default(Id::new("AMBIENT_PLOT_LINK_STATE"));
        if config.link_axes && !state.config.link_axes {
            state.axis_epoch = state.axis_epoch.wrapping_add(1);
        }
        state.config = config;
    });
}

/// Reads the current ambient multi-pane plot synchronization configuration from `ctx`.
#[must_use]
pub fn ambient_plot_link_config(ctx: &egui::Context) -> AmbientPlotLinkConfig {
    ctx.data(|d| {
        d.get_temp::<AmbientPlotLinkState>(Id::new("AMBIENT_PLOT_LINK_STATE"))
            .unwrap_or_default()
            .config
    })
}

#[derive(PartialEq, Clone)]
struct PlotFrameAutoBounds {
    id: Id,
    bounds: PlotBounds,
}

#[derive(Clone)]
struct LinkedBounds {
    bounds: PlotBounds,
    auto_bounds: Vec2b,
    auto_frames: Vec<PlotFrameAutoBounds>,
}

impl Default for LinkedBounds {
    fn default() -> Self {
        Self {
            bounds: PlotBounds::NOTHING,
            auto_bounds: Vec2b::new(true, true),
            auto_frames: Vec::new(),
        }
    }
}

#[derive(Default, Clone)]
struct BoundsLinkGroups(HashMap<Id, LinkedBounds>);

// ----------------------------------------------------------------------------

/// What [`Plot::show`] returns.
pub struct PlotResponse<R> {
    /// What the user closure returned.
    pub inner: R,

    /// The response of the plot.
    pub response: Response,

    /// The transform between screen coordinates and plot coordinates.
    pub transform: PlotTransform,

    /// The id of a currently hovered item if any.
    ///
    /// This is `None` if either no item was hovered.
    /// A plot item can be hovered either by hovering its representation in the plot (line, marker, etc.)
    /// or by hovering the item in the legend.
    pub hovered_plot_item: Option<Id>,
}

// ----------------------------------------------------------------------------

/// Converts a plot identifier into its hashed [`egui::Id`] and an optional humanized modal title.
///
/// String identifiers (`&str`, `String`, `&String`, [`std::borrow::Cow`]) are humanized into
/// readable titles (for example, `"sir_plot"` becomes `Some("SIR Plot")`), whereas raw
/// [`egui::Id`] callers retain `None` unless overridden via [`Plot::title`].
pub trait PlotIdSource: Sized {
    /// Resolve the hashed plot identifier, deterministic snapshot key, and optional humanized title.
    fn into_plot_id_key_and_title(self) -> (Id, String, Option<String>);

    /// Resolve the hashed plot identifier and optional humanized display title.
    fn into_plot_id_and_title(self) -> (Id, Option<String>) {
        let (id, _key, title) = self.into_plot_id_key_and_title();
        (id, title)
    }
}

/// Humanizes a snake_case, kebab-case, or plain-text plot label into a readable title.
#[must_use]
pub fn humanize_plot_label(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let words: Vec<String> = trimmed
        .split(|c: char| c == '_' || c == '-' || c.is_whitespace())
        .filter(|tok| !tok.is_empty())
        .map(humanize_plot_word)
        .collect();
    (!words.is_empty()).then(|| words.join(" "))
}

fn humanize_plot_word(token: &str) -> String {
    match token.to_ascii_lowercase().as_str() {
        "ode" => return "ODE".to_owned(),
        "pde" => return "PDE".to_owned(),
        "sir" => return "SIR".to_owned(),
        "isi" => return "ISI".to_owned(),
        "mri" => return "MRI".to_owned(),
        "co2" => return "CO2".to_owned(),
        "ai" => return "AI".to_owned(),
        "fft" => return "FFT".to_owned(),
        "rk4" => return "RK4".to_owned(),
        "2d" => return "2D".to_owned(),
        "3d" => return "3D".to_owned(),
        "csv" => return "CSV".to_owned(),
        "tsv" => return "TSV".to_owned(),
        "png" => return "PNG".to_owned(),
        "svg" => return "SVG".to_owned(),
        _ => {}
    }
    if token.chars().any(char::is_uppercase) {
        return token.to_owned();
    }
    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    let mut out = first.to_uppercase().to_string();
    out.push_str(chars.as_str());
    out
}

fn format_modal_title(base: &str, plot_title: Option<&str>) -> String {
    match plot_title.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => format!("{base} - {t}"),
        None => base.to_owned(),
    }
}

impl PlotIdSource for &str {
    fn into_plot_id_key_and_title(self) -> (Id, String, Option<String>) {
        (Id::new(self), self.to_owned(), humanize_plot_label(self))
    }
}

impl PlotIdSource for String {
    fn into_plot_id_key_and_title(self) -> (Id, String, Option<String>) {
        let title = humanize_plot_label(&self);
        let id = Id::new(&self);
        (id, self, title)
    }
}

impl PlotIdSource for &String {
    fn into_plot_id_key_and_title(self) -> (Id, String, Option<String>) {
        (Id::new(self), self.clone(), humanize_plot_label(self))
    }
}

impl PlotIdSource for std::borrow::Cow<'_, str> {
    fn into_plot_id_key_and_title(self) -> (Id, String, Option<String>) {
        let title = humanize_plot_label(self.as_ref());
        let key = self.into_owned();
        let id = Id::new(&key);
        (id, key, title)
    }
}

impl PlotIdSource for &std::borrow::Cow<'_, str> {
    fn into_plot_id_key_and_title(self) -> (Id, String, Option<String>) {
        let raw = self.as_ref();
        (Id::new(raw), raw.to_owned(), humanize_plot_label(raw))
    }
}

impl PlotIdSource for Id {
    fn into_plot_id_key_and_title(self) -> (Id, String, Option<String>) {
        (Id::new(self), format!("id:{:016x}", self.value()), None)
    }
}

impl PlotIdSource for &Id {
    fn into_plot_id_key_and_title(self) -> (Id, String, Option<String>) {
        (Id::new(self), format!("id:{:016x}", self.value()), None)
    }
}

/// A 2D plot, e.g. a graph of a function.
///
/// [`Plot`] supports multiple lines and points.
///
/// ```
/// # egui::__run_test_ui(|ui| {
/// use egui_plot::{Line, Plot, PlotPoints};
///
/// let sin: PlotPoints = (0..1000).map(|i| {
///     let x = i as f64 * 0.01;
///     [x, x.sin()]
/// }).collect();
/// let line = Line::new("sin", sin);
/// Plot::new("my_plot").view_aspect(2.0).show(ui, |plot_ui| plot_ui.line(line));
/// # });
/// ```
pub struct Plot<'a> {
    id_source: Id,
    plot_key: String,
    id: Option<Id>,
    title: Option<String>,

    center_axis: Vec2b,
    allow_zoom: Vec2b,
    allow_drag: Vec2b,
    allow_axis_zoom_drag: Vec2b,
    allow_scroll: Vec2b,
    allow_double_click_reset: bool,
    allow_boxed_zoom: bool,
    default_auto_bounds: Vec2b,
    min_auto_bounds: PlotBounds,
    margin_fraction: Vec2,
    boxed_zoom_pointer_button: PointerButton,
    linked_axes: Option<(Id, Vec2b)>,
    linked_cursors: Option<(Id, Vec2b)>,

    min_size: Vec2,
    width: Option<f32>,
    height: Option<f32>,
    data_aspect: Option<f32>,
    view_aspect: Option<f32>,
    invert_x: bool,
    invert_y: bool,

    reset: bool,

    show_x: bool,
    show_y: bool,
    label_formatter: LabelFormatter<'a>,
    coordinates_formatter: Option<(Corner, CoordinatesFormatter<'a>)>,
    x_axes: Vec<AxisHints<'a>>, // default x axes
    y_axes: Vec<AxisHints<'a>>, // default y axes
    legend_config: Option<Legend>,
    cursor_color: Option<Color32>,
    cursor_snap_distance: Option<f32>,
    show_background: bool,
    show_axes: Vec2b,

    show_grid: Vec2b,
    grid_spacing: Rangef,
    grid_spacers: [GridSpacer<'a>; 2],
    clamp_grid: bool,

    sense: Sense,
    show_navigation_controls: bool,
    show_zoom_buttons: bool,
    show_toolbar: bool,
    show_settings_button: bool,
    allow_2d_analysis: bool,
    allow_csv_import: bool,
}

impl<'a> Plot<'a> {
    /// Give a unique id for each plot within the same [`Ui`].
    pub fn new(id_source: impl PlotIdSource) -> Self {
        let (id_source, plot_key, title) = id_source.into_plot_id_key_and_title();
        Self {
            id_source,
            plot_key,
            id: None,
            title,

            center_axis: false.into(),
            allow_zoom: true.into(),
            allow_drag: true.into(),
            allow_axis_zoom_drag: true.into(),
            allow_scroll: true.into(),
            allow_double_click_reset: true,
            allow_boxed_zoom: true,
            default_auto_bounds: true.into(),
            min_auto_bounds: PlotBounds::NOTHING,
            margin_fraction: Vec2::splat(0.05),
            boxed_zoom_pointer_button: PointerButton::Secondary,
            linked_axes: None,
            linked_cursors: None,

            min_size: Vec2::splat(64.0),
            width: None,
            height: None,
            data_aspect: None,
            view_aspect: None,
            invert_x: false,
            invert_y: false,

            reset: false,

            show_x: true,
            show_y: true,
            label_formatter: None,
            coordinates_formatter: None,
            x_axes: vec![AxisHints::new(Axis::X)],
            y_axes: vec![AxisHints::new(Axis::Y)],
            legend_config: None,
            cursor_color: None,
            cursor_snap_distance: Some(16.0),
            show_background: true,
            show_axes: true.into(),

            show_grid: true.into(),
            grid_spacing: Rangef::new(8.0, 300.0),
            grid_spacers: [log_grid_spacer(10), log_grid_spacer(10)],
            clamp_grid: false,

            sense: egui::Sense::click_and_drag(),
            show_navigation_controls: true,
            show_zoom_buttons: true,
            show_toolbar: true,
            show_settings_button: true,
            allow_2d_analysis: true,
            allow_csv_import: true,
        }
    }

    /// Set an explicit (global) id for the plot.
    ///
    /// This will override the id set by [`Self::new`].
    ///
    /// This is the same `Id` that can be used for [`PlotMemory::load`].
    #[inline]
    pub fn id(mut self, id: Id) -> Self {
        self.id = Some(id);
        self
    }

    /// Override the deterministic key used when persisting scoped [`PlotOverlaySnapshot`] state.
    #[inline]
    pub fn plot_key(mut self, key: impl Into<String>) -> Self {
        self.plot_key = key.into();
        self
    }

    /// Set an explicit human-readable display title used in plot modals and accessibility labels.
    ///
    /// Passing an empty or whitespace-only string clears the title to `None`.
    #[inline]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        let raw = title.into();
        let trimmed = raw.trim();
        self.title = (!trimmed.is_empty()).then(|| trimmed.to_owned());
        self
    }

    /// width / height ratio of the data.
    /// For instance, it can be useful to set this to `1.0` for when the two axes show the same
    /// unit.
    /// By default the plot window's aspect ratio is used.
    #[inline]
    pub fn data_aspect(mut self, data_aspect: f32) -> Self {
        self.data_aspect = Some(data_aspect);
        self
    }

    /// width / height ratio of the plot region.
    /// By default no fixed aspect ratio is set (and width/height will fill the ui it is in).
    #[inline]
    pub fn view_aspect(mut self, view_aspect: f32) -> Self {
        self.view_aspect = Some(view_aspect);
        self
    }

    /// Set whether to invert the x-axis (i.e. positive values go to the left).
    /// By default the x-axis is not inverted (i.e. positive values go to the right).
    #[inline]
    pub fn invert_x(mut self, invert: bool) -> Self {
        self.invert_x = invert;
        self
    }

    /// Set whether to invert the y-axis (i.e. positive values go down).
    /// By default the y-axis is not inverted (i.e. positive values go up).
    #[inline]
    pub fn invert_y(mut self, invert: bool) -> Self {
        self.invert_y = invert;
        self
    }

    /// Width of plot. By default a plot will fill the ui it is in.
    /// If you set [`Self::view_aspect`], the width can be calculated from the height.
    #[inline]
    pub fn width(mut self, width: f32) -> Self {
        self.min_size.x = width;
        self.width = Some(width);
        self
    }

    /// Height of plot. By default a plot will fill the ui it is in.
    /// If you set [`Self::view_aspect`], the height can be calculated from the width.
    #[inline]
    pub fn height(mut self, height: f32) -> Self {
        self.min_size.y = height;
        self.height = Some(height);
        self
    }

    /// Minimum size of the plot view.
    #[inline]
    pub fn min_size(mut self, min_size: Vec2) -> Self {
        self.min_size = min_size;
        self
    }

    /// Show the x-value (e.g. when hovering). Default: `true`.
    #[inline]
    pub fn show_x(mut self, show_x: bool) -> Self {
        self.show_x = show_x;
        self
    }

    /// Show the y-value (e.g. when hovering). Default: `true`.
    #[inline]
    pub fn show_y(mut self, show_y: bool) -> Self {
        self.show_y = show_y;
        self
    }

    /// Always keep the X-axis centered. Default: `false`.
    #[inline]
    pub fn center_x_axis(mut self, on: bool) -> Self {
        self.center_axis.x = on;
        self
    }

    /// Always keep the Y-axis centered. Default: `false`.
    #[inline]
    pub fn center_y_axis(mut self, on: bool) -> Self {
        self.center_axis.y = on;
        self
    }

    /// Whether to allow zooming in the plot. Default: `true`.
    ///
    /// Note: Allowing zoom in one axis but not the other may lead to unexpected results if used in combination with `data_aspect`.
    #[inline]
    pub fn allow_zoom<T>(mut self, on: T) -> Self
    where
        T: Into<Vec2b>,
    {
        self.allow_zoom = on.into();
        self
    }

    /// Whether to allow scrolling in the plot. Default: `true`.
    #[inline]
    pub fn allow_scroll<T>(mut self, on: T) -> Self
    where
        T: Into<Vec2b>,
    {
        self.allow_scroll = on.into();
        self
    }

    /// Whether to allow double clicking to reset the view.
    /// Default: `true`.
    #[inline]
    pub fn allow_double_click_reset(mut self, on: bool) -> Self {
        self.allow_double_click_reset = on;
        self
    }

    /// Set the side margin as a fraction of the plot size. Only used for auto bounds.
    ///
    /// For instance, a value of `0.1` will add 10% space on both sides.
    #[inline]
    pub fn set_margin_fraction(mut self, margin_fraction: Vec2) -> Self {
        self.margin_fraction = margin_fraction;
        self
    }

    /// Whether to allow zooming in the plot by dragging out a box with the secondary mouse button.
    ///
    /// Default: `true`.
    #[inline]
    pub fn allow_boxed_zoom(mut self, on: bool) -> Self {
        self.allow_boxed_zoom = on;
        self
    }

    /// Config the button pointer to use for boxed zooming. Default: [`Secondary`](PointerButton::Secondary)
    #[inline]
    pub fn boxed_zoom_pointer_button(mut self, boxed_zoom_pointer_button: PointerButton) -> Self {
        self.boxed_zoom_pointer_button = boxed_zoom_pointer_button;
        self
    }

    /// Whether to allow dragging in the plot to move the bounds. Default: `true`.
    #[inline]
    pub fn allow_drag<T>(mut self, on: T) -> Self
    where
        T: Into<Vec2b>,
    {
        self.allow_drag = on.into();
        self
    }

    /// Whether to allow dragging in the axis areas to zoom the plot. Default: `true`.
    #[inline]
    pub fn allow_axis_zoom_drag<T>(mut self, on: T) -> Self
    where
        T: Into<Vec2b>,
    {
        self.allow_axis_zoom_drag = on.into();
        self
    }

    /// Provide a function to customize the on-hover label for the x and y axis
    ///
    /// ```
    /// # egui::__run_test_ui(|ui| {
    /// use egui_plot::{Line, Plot, PlotPoints};
    /// let sin: PlotPoints = (0..1000).map(|i| {
    ///     let x = i as f64 * 0.01;
    ///     [x, x.sin()]
    /// }).collect();
    /// let line = Line::new("sin", sin);
    /// Plot::new("my_plot").view_aspect(2.0)
    /// .label_formatter(|name, value| {
    ///     if !name.is_empty() {
    ///         format!("{}: {:.*}%", name, 1, value.y)
    ///     } else {
    ///         "".to_owned()
    ///     }
    /// })
    /// .show(ui, |plot_ui| plot_ui.line(line));
    /// # });
    /// ```
    pub fn label_formatter(
        mut self,
        label_formatter: impl Fn(&str, &PlotPoint) -> String + 'a,
    ) -> Self {
        self.label_formatter = Some(Box::new(label_formatter));
        self
    }

    /// Show the pointer coordinates in the plot.
    pub fn coordinates_formatter(
        mut self,
        position: Corner,
        formatter: CoordinatesFormatter<'a>,
    ) -> Self {
        self.coordinates_formatter = Some((position, formatter));
        self
    }

    /// Configure how the grid in the background is spaced apart along the X axis.
    ///
    /// Default is a log-10 grid, i.e. every plot unit is divided into 10 other units.
    ///
    /// The function has this signature:
    /// ```text
    /// fn step_sizes(input: GridInput) -> Vec<GridMark>;
    /// ```
    ///
    /// This function should return all marks along the visible range of the X axis.
    /// `step_size` also determines how thick/faint each line is drawn.
    /// For example, if x = 80..=230 is visible and you want big marks at steps of
    /// 100 and small ones at 25, you can return:
    /// ```no_run
    /// # use egui_plot::GridMark;
    /// vec![
    ///    // 100s
    ///    GridMark { value: 100.0, step_size: 100.0 },
    ///    GridMark { value: 200.0, step_size: 100.0 },
    ///
    ///    // 25s
    ///    GridMark { value: 125.0, step_size: 25.0 },
    ///    GridMark { value: 150.0, step_size: 25.0 },
    ///    GridMark { value: 175.0, step_size: 25.0 },
    ///    GridMark { value: 225.0, step_size: 25.0 },
    /// ];
    /// # ()
    /// ```
    ///
    /// There are helpers for common cases, see [`log_grid_spacer`] and [`uniform_grid_spacer`].
    #[inline]
    pub fn x_grid_spacer(mut self, spacer: impl Fn(GridInput) -> Vec<GridMark> + 'a) -> Self {
        self.grid_spacers[0] = Box::new(spacer);
        self
    }

    /// Default is a log-10 grid, i.e. every plot unit is divided into 10 other units.
    ///
    /// See [`Self::x_grid_spacer`] for explanation.
    #[inline]
    pub fn y_grid_spacer(mut self, spacer: impl Fn(GridInput) -> Vec<GridMark> + 'a) -> Self {
        self.grid_spacers[1] = Box::new(spacer);
        self
    }

    /// Set when the grid starts showing.
    ///
    /// When grid lines are closer than the given minimum, they will be hidden.
    /// When they get further apart they will fade in, until the reaches the given maximum,
    /// at which point they are fully opaque.
    #[inline]
    pub fn grid_spacing(mut self, grid_spacing: impl Into<Rangef>) -> Self {
        self.grid_spacing = grid_spacing.into();
        self
    }

    /// Clamp the grid to only be visible at the range of data where we have values.
    ///
    /// Default: `false`.
    #[inline]
    pub fn clamp_grid(mut self, clamp_grid: bool) -> Self {
        self.clamp_grid = clamp_grid;
        self
    }

    /// Set the sense for the plot rect.
    ///
    /// Default: `Sense::click_and_drag()`.
    #[inline]
    pub fn sense(mut self, sense: Sense) -> Self {
        self.sense = sense;
        self
    }

    /// Overwrite the starting and reset bounds used for the x axis.
    /// Set the `default_auto_bounds` of the x axis to `false`.
    ///
    /// Panics in debug builds if `min >= max`.
    #[inline]
    pub fn default_x_bounds(mut self, min: f64, max: f64) -> Self {
        debug_assert!(
            min < max,
            "`min` must be less than `max` in `default_x_bounds`"
        );
        self.default_auto_bounds.x = false;
        self.min_auto_bounds.min[0] = min;
        self.min_auto_bounds.max[0] = max;
        self
    }

    /// Overwrite the starting and reset bounds used for the y axis.
    /// Set the `default_auto_bounds` of the y axis to `false`.
    ///
    /// Panics in debug builds if `min >= max`.
    #[inline]
    pub fn default_y_bounds(mut self, min: f64, max: f64) -> Self {
        debug_assert!(
            min < max,
            "`min` must be less than `max` in `default_y_bounds`"
        );
        self.default_auto_bounds.y = false;
        self.min_auto_bounds.min[1] = min;
        self.min_auto_bounds.max[1] = max;
        self
    }

    /// Expand bounds to include the given x value.
    /// For instance, to always show the y axis, call `plot.include_x(0.0)`.
    #[inline]
    pub fn include_x(mut self, x: impl Into<f64>) -> Self {
        self.min_auto_bounds.extend_with_x(x.into());
        self
    }

    /// Expand bounds to include the given y value.
    /// For instance, to always show the x axis, call `plot.include_y(0.0)`.
    #[inline]
    pub fn include_y(mut self, y: impl Into<f64>) -> Self {
        self.min_auto_bounds.extend_with_y(y.into());
        self
    }

    /// Set whether the bounds should be automatically set based on data by default.
    ///
    /// This is enabled by default.
    #[inline]
    pub fn auto_bounds(mut self, auto_bounds: impl Into<Vec2b>) -> Self {
        self.default_auto_bounds = auto_bounds.into();
        self
    }

    /// Show a legend including all named items.
    #[inline]
    pub fn legend(mut self, legend: Legend) -> Self {
        self.legend_config = Some(legend);
        self
    }

    /// Whether or not to show the background [`Rect`].
    ///
    /// Can be useful to disable if the plot is overlaid over existing content.
    /// Default: `true`.
    #[inline]
    pub fn show_background(mut self, show: bool) -> Self {
        self.show_background = show;
        self
    }

    /// Show axis labels and grid tick values on the side of the plot.
    ///
    /// Default: `true`.
    #[inline]
    pub fn show_axes(mut self, show: impl Into<Vec2b>) -> Self {
        self.show_axes = show.into();
        self
    }

    /// Show a grid overlay on the plot.
    ///
    /// Default: `true`.
    #[inline]
    pub fn show_grid(mut self, show: impl Into<Vec2b>) -> Self {
        self.show_grid = show.into();
        self
    }

    /// Whether to show the navigation controls (zoom in, zoom out, reset bounds) overlay on the plot canvas.
    ///
    /// When enabled, renders a compact vertical button column below the settings wrench icon
    /// providing single-click zoom in (+25%), zoom out (-20%), and bounds reset to dataset extent.
    ///
    /// Default: `true`.
    #[inline]
    pub fn show_navigation_controls(mut self, show: bool) -> Self {
        self.show_navigation_controls = show;
        self.show_zoom_buttons = show;
        self
    }

    /// Whether to show the floating zoom and reset buttons overlay on the plot canvas.
    ///
    /// Alias for [`Self::show_navigation_controls`]. Default: `true`.
    #[inline]
    pub fn show_zoom_buttons(mut self, show: bool) -> Self {
        self.show_navigation_controls = show;
        self.show_zoom_buttons = show;
        self
    }

    /// Whether to show the floating zoom and reset toolbar overlay.
    ///
    /// Alias for [`Self::show_navigation_controls`]. Default: `true`.
    #[inline]
    pub fn show_zoom_overlay(mut self, show: bool) -> Self {
        self.show_navigation_controls = show;
        self.show_zoom_buttons = show;
        self
    }

    /// Whether to show the collapsible plot action header bar docked above the plot canvas.
    ///
    /// Default: `true`.
    #[inline]
    pub fn show_toolbar(mut self, show: bool) -> Self {
        self.show_toolbar = show;
        self
    }

    /// Whether to show the floating grid and axis settings wrench button in the top-right corner.
    ///
    /// Default: `true`.
    #[inline]
    pub fn show_settings_button(mut self, show: bool) -> Self {
        self.show_settings_button = show;
        self
    }

    /// Add this plot to an axis link group so that this plot will share the bounds with other plots in the
    /// same group. A plot cannot belong to more than one axis group.
    #[inline]
    pub fn link_axis(mut self, group_id: impl Into<Id>, link: impl Into<Vec2b>) -> Self {
        self.linked_axes = Some((group_id.into(), link.into()));
        self
    }

    /// Add this plot to a cursor link group so that this plot will share the cursor position with other plots
    /// in the same group. A plot cannot belong to more than one cursor group.
    #[inline]
    pub fn link_cursor(mut self, group_id: impl Into<Id>, link: impl Into<Vec2b>) -> Self {
        self.linked_cursors = Some((group_id.into(), link.into()));
        self
    }

    /// Resets the plot.
    #[inline]
    pub fn reset(mut self) -> Self {
        self.reset = true;
        self
    }

    /// Set the x axis label of the main X-axis.
    ///
    /// Default: no label.
    #[inline]
    pub fn x_axis_label(mut self, label: impl Into<WidgetText>) -> Self {
        if let Some(main) = self.x_axes.first_mut() {
            main.label = label.into();
        }
        self
    }

    /// Set the y axis label of the main Y-axis.
    ///
    /// Default: no label.
    #[inline]
    pub fn y_axis_label(mut self, label: impl Into<WidgetText>) -> Self {
        if let Some(main) = self.y_axes.first_mut() {
            main.label = label.into();
        }
        self
    }

    /// Set the position of the main X-axis.
    #[inline]
    pub fn x_axis_position(mut self, placement: axis::VPlacement) -> Self {
        if let Some(main) = self.x_axes.first_mut() {
            main.placement = placement.into();
        }
        self
    }

    /// Set the position of the main Y-axis.
    #[inline]
    pub fn y_axis_position(mut self, placement: axis::HPlacement) -> Self {
        if let Some(main) = self.y_axes.first_mut() {
            main.placement = placement.into();
        }
        self
    }

    /// Specify custom formatter for ticks on the main X-axis.
    ///
    /// Arguments of `fmt`:
    /// * the grid mark to format
    /// * currently shown range on this axis.
    pub fn x_axis_formatter(
        mut self,
        fmt: impl Fn(GridMark, &RangeInclusive<f64>) -> String + 'a,
    ) -> Self {
        if let Some(main) = self.x_axes.first_mut() {
            main.formatter = Arc::new(fmt);
        }
        self
    }

    /// Specify custom formatter for ticks on the main Y-axis.
    ///
    /// Arguments of `fmt`:
    /// * the grid mark to format
    /// * currently shown range on this axis.
    pub fn y_axis_formatter(
        mut self,
        fmt: impl Fn(GridMark, &RangeInclusive<f64>) -> String + 'a,
    ) -> Self {
        if let Some(main) = self.y_axes.first_mut() {
            main.formatter = Arc::new(fmt);
        }
        self
    }

    /// Set the minimum width of the main y-axis, in ui points.
    ///
    /// The width will automatically expand if any tickmark text is wider than this.
    #[inline]
    pub fn y_axis_min_width(mut self, min_width: f32) -> Self {
        if let Some(main) = self.y_axes.first_mut() {
            main.min_thickness = min_width;
        }
        self
    }

    /// Set custom configuration for X-axis
    ///
    /// More than one axis may be specified. The first specified axis is considered the main axis.
    #[inline]
    pub fn custom_x_axes(mut self, hints: Vec<AxisHints<'a>>) -> Self {
        self.x_axes = hints;
        self
    }

    /// Set custom configuration for left Y-axis
    ///
    /// More than one axis may be specified. The first specified axis is considered the main axis.
    #[inline]
    pub fn custom_y_axes(mut self, hints: Vec<AxisHints<'a>>) -> Self {
        self.y_axes = hints;
        self
    }

    /// Set custom cursor color.
    ///
    /// You may set the color to [`Color32::TRANSPARENT`] to hide the cursors.
    #[inline]
    pub fn cursor_color(mut self, color: Color32) -> Self {
        self.cursor_color = Some(color);
        self
    }

    /// Distance in points within which pointer snaps to plot curve points.
    /// Default is `16.0` points.
    #[inline]
    pub fn cursor_snap_distance(mut self, distance: f32) -> Self {
        self.cursor_snap_distance = Some(distance);
        self
    }

    /// Enable or disable 2D analytical overlays (critical-point detection, 2D curve hover
    /// snapping, detached 2D canvas callouts, and curve-fitting menus).
    ///
    /// Set to `false` when rendering projected 3D scenes onto a 2D [`Plot`]. Default: `true`.
    #[inline]
    pub fn allow_2d_analysis(mut self, allow: bool) -> Self {
        self.allow_2d_analysis = allow;
        self
    }

    /// Enable or disable 2D CSV/TSV dataset overlays, toolbar `Import CSV` / `Overlays` controls,
    /// `Ctrl+Shift+I`, clipboard paste, and file drag-and-drop.
    ///
    /// Set to `false` on locked 3D viewports such as `Plot3DViewport`. Default: `true`.
    #[inline]
    pub fn allow_csv_import(mut self, allow: bool) -> Self {
        self.allow_csv_import = allow;
        self
    }

    /// Interact with and add items to the plot and finally draw it.
    pub fn show<'b, R>(
        self,
        ui: &mut Ui,
        build_fn: impl FnOnce(&mut PlotUi<'b>) -> R + 'a,
    ) -> PlotResponse<R> {
        self.show_dyn(ui, Box::new(build_fn))
    }

    fn show_dyn<'b, R>(
        self,
        ui: &mut Ui,
        build_fn: Box<dyn FnOnce(&mut PlotUi<'b>) -> R + 'a>,
    ) -> PlotResponse<R> {
        let Self {
            id_source,
            plot_key,
            id,
            title,
            center_axis,
            allow_zoom,
            allow_drag,
            allow_axis_zoom_drag,
            allow_scroll,
            allow_double_click_reset,
            allow_boxed_zoom,
            boxed_zoom_pointer_button,
            default_auto_bounds,
            min_auto_bounds,
            margin_fraction,
            width,
            height,
            mut min_size,
            data_aspect,
            view_aspect,
            invert_x,
            invert_y,
            mut show_x,
            mut show_y,
            label_formatter,
            coordinates_formatter,
            x_axes,
            y_axes,
            legend_config,
            cursor_color,
            cursor_snap_distance,
            reset,
            show_background,
            show_axes,
            show_grid,
            grid_spacing,
            mut linked_axes,
            mut linked_cursors,

            clamp_grid,
            grid_spacers,
            sense,
            show_navigation_controls,
            show_zoom_buttons: _,
            show_toolbar,
            show_settings_button,
            allow_2d_analysis,
            allow_csv_import,
        } = self;

        resolve_ambient_plot_links(
            ui.ctx(),
            id_source,
            show_axes,
            allow_drag,
            allow_zoom,
            allow_scroll,
            allow_boxed_zoom,
            &mut linked_axes,
            &mut linked_cursors,
        );

        // Disable interaction if ui is disabled.
        let allow_zoom = allow_zoom.and(ui.is_enabled());
        let allow_drag = allow_drag.and(ui.is_enabled());
        let allow_scroll = allow_scroll.and(ui.is_enabled());

        // Determine position of widget.
        let pos = ui.available_rect_before_wrap().min;
        // Minimum values for screen protection
        min_size.x = min_size.x.at_least(1.0);
        min_size.y = min_size.y.at_least(1.0);

        // Determine size of widget.
        let size = {
            let width = width
                .unwrap_or_else(|| {
                    if let (Some(height), Some(aspect)) = (height, view_aspect) {
                        height * aspect
                    } else {
                        let avail_w = ui.available_size_before_wrap().x;
                        let clip_w = ui.clip_rect().max.x - pos.x;
                        if clip_w.is_finite() && clip_w >= min_size.x {
                            avail_w.min(clip_w)
                        } else {
                            avail_w
                        }
                    }
                })
                .at_least(min_size.x);

            let height = height
                .unwrap_or_else(|| {
                    if let Some(aspect) = view_aspect {
                        width / aspect
                    } else {
                        ui.available_size_before_wrap().y
                    }
                })
                .at_least(min_size.y);
            vec2(width, height)
        };

        // Determine complete rect of widget.
        let complete_rect = Rect {
            min: pos,
            max: pos + size,
        };

        let plot_id = id.unwrap_or_else(|| ui.make_persistent_id(id_source));
        let loaded_mem = if reset {
            if let Some((name, _)) = linked_axes.as_ref() {
                ui.data_mut(|data| {
                    let link_groups: &mut BoundsLinkGroups = data.get_temp_mut_or_default(Id::NULL);
                    link_groups.0.remove(name);
                });
            }
            None
        } else {
            PlotMemory::load(ui.ctx(), plot_id)
        };
        let had_existing_mem = loaded_mem.is_some();

        let toolbar_collapsed = loaded_mem.as_ref().is_some_and(|m| m.toolbar_collapsed);
        let header_h = if show_toolbar && !toolbar_collapsed && complete_rect.height() > 48.0 {
            30.0_f32
        } else {
            0.0_f32
        };
        let header_rect =
            Rect::from_min_size(complete_rect.min, vec2(complete_rect.width(), header_h));
        let body_rect = Rect::from_min_max(
            egui::pos2(complete_rect.min.x, complete_rect.min.y + header_h),
            complete_rect.max,
        );

        let ([x_axis_widgets, y_axis_widgets], plot_rect) = axis_widgets(
            loaded_mem.as_ref(),
            show_axes,
            body_rect,
            [&x_axes, &y_axes],
        );

        // Allocate the plot window.
        let response = ui.allocate_rect(plot_rect, sense | egui::Sense::focusable_noninteractive());

        let x_axis_responses = x_axis_widgets
            .iter()
            .map(|widget| {
                let axis_response = ui.allocate_rect(widget.rect, Sense::drag());
                if allow_axis_zoom_drag.x {
                    axis_response.on_hover_cursor(CursorIcon::ResizeHorizontal)
                } else {
                    axis_response
                }
            })
            .collect::<Vec<_>>();

        let y_axis_responses = y_axis_widgets
            .iter()
            .map(|widget| {
                let axis_response = ui.allocate_rect(widget.rect, Sense::drag());

                if allow_axis_zoom_drag.y {
                    axis_response.on_hover_cursor(CursorIcon::ResizeVertical)
                } else {
                    axis_response
                }
            })
            .collect::<Vec<_>>();

        // Load or initialize the memory.
        ui.ctx().check_for_id_clash(plot_id, plot_rect, "Plot");

        let mut mem = loaded_mem.unwrap_or_else(|| PlotMemory {
            auto_bounds: default_auto_bounds,
            hovered_legend_item: None,
            hidden_items: Default::default(),
            transform: PlotTransform::new_with_invert_axis(
                plot_rect,
                min_auto_bounds,
                center_axis,
                Vec2b::new(invert_x, invert_y),
            ),
            last_click_pos_for_zoom: None,
            x_axis_thickness: Default::default(),
            y_axis_thickness: Default::default(),
            imported_datasets: Default::default(),
            callouts: Default::default(),
            log_x: false,
            log_y: false,
            grid_density: 8.0,
            show_x_labels: true,
            show_y_labels: true,
            show_hover: true,
            toolbar_collapsed: false,
            clamp_grid: false,
            show_grid: Vec2b::new(true, true),
            show_settings_modal: false,
            last_right_click_coord: None,
        });

        let allow_bounds_persist = allow_drag.any()
            || allow_zoom.any()
            || allow_axis_zoom_drag.any()
            || allow_scroll.any()
            || allow_boxed_zoom;
        snapshot::sync_plot_before_show(
            ui.ctx(),
            &snapshot::BeforeShowSyncParams {
                plot_id,
                plot_key: &plot_key,
                had_existing_mem,
                default_auto_bounds,
                min_auto_bounds,
                allow_bounds_persist,
                allow_2d_analysis,
            },
            &mut mem,
        );

        // Drain any newly imported CSV datasets for this plot
        let newly_imported = export::drain_imported_datasets(plot_id);
        if allow_csv_import {
            if !newly_imported.is_empty() {
                plot_overlays::merge_imported_datasets(&mut mem.imported_datasets, newly_imported);
            }
        } else {
            mem.imported_datasets.clear();
        }

        let last_plot_transform = mem.transform;

        // Call the plot build function.
        let mut plot_ui = PlotUi {
            ctx: ui.ctx().clone(),
            items: Vec::new(),
            next_auto_color_idx: 0,
            last_plot_transform,
            last_auto_bounds: mem.auto_bounds,
            response,
            bounds_modifications: Vec::new(),
            plot_id,
            sidecar_3d: None,
        };

        let inner = build_fn(&mut plot_ui);

        let effective_allow_csv_import = allow_csv_import && plot_ui.sidecar_3d.is_none();
        if effective_allow_csv_import {
            plot_overlays::render_imported_overlays(&mut plot_ui, &mem.imported_datasets);
        } else {
            mem.imported_datasets.clear();
        }

        // Drain any remaining canvas callouts not consumed by a custom CalloutManager
        let newly_added_callouts = drain_pending_callouts(plot_id);
        let allow_2d_analysis = allow_2d_analysis && plot_ui.sidecar_3d.is_none();
        if allow_2d_analysis {
            if !newly_added_callouts.is_empty() {
                mem.callouts
                    .extend(newly_added_callouts.into_iter().map(SavedCallout::from));
            }
            // Render persisted user callouts automatically
            for saved in &mem.callouts {
                plot_ui.callout(saved.to_callout());
            }
        } else {
            mem.callouts.clear();
        }
        let PlotUi {
            mut items,
            mut response,
            last_plot_transform,
            bounds_modifications,
            sidecar_3d,
            ..
        } = plot_ui;

        let persisted_point_tooltip = if allow_2d_analysis || sidecar_3d.is_some() {
            a11y::preprocess_point_tooltip_frame(
                &response,
                response.hovered() || response.has_focus(),
            )
        } else {
            None
        };

        // Background
        let plot_border_stroke = if response.has_focus() {
            a11y::focus_ring_stroke(ui.visuals(), ui.visuals().extreme_bg_color)
        } else {
            ui.visuals().widgets.noninteractive.bg_stroke
        };
        if show_background {
            ui.painter()
                .with_clip_rect(plot_rect)
                .add(epaint::RectShape::new(
                    plot_rect,
                    2,
                    ui.visuals().extreme_bg_color,
                    plot_border_stroke,
                    egui::StrokeKind::Inside,
                ));
        }

        // Don't show hover cursor when hovering over legend unless Plot holds keyboard focus.
        if mem.hovered_legend_item.is_some() && !response.has_focus() {
            show_x = false;
            show_y = false;
        }

        // --- Bound computation ---
        let mut bounds = *last_plot_transform.bounds();

        // Find the cursors from other plots we need to draw
        let draw_cursors: Vec<Cursor> = if let Some((id, _)) = linked_cursors.as_ref() {
            ui.data_mut(|data| {
                let frames: &mut CursorLinkGroups = data.get_temp_mut_or_default(Id::NULL);
                let cursors = frames.0.entry(*id).or_default();

                // Look for our previous frame
                let index = cursors
                    .iter()
                    .enumerate()
                    .find(|(_, frame)| frame.id == plot_id)
                    .map(|(i, _)| i);

                // Remove our previous frame and all older frames as these are no longer displayed. This avoids
                // unbounded growth, as we add an entry each time we draw a plot.
                index.map(|index| cursors.drain(0..=index));

                // Gather all cursors of the remaining frames. This will be all the cursors of the
                // other plots in the group. We want to draw these in the current plot too.
                cursors
                    .iter()
                    .flat_map(|frame| frame.cursors.iter().copied())
                    .collect()
            })
        } else {
            Vec::new()
        };

        // Transfer the bounds from a link group.
        if let Some((id, axes)) = linked_axes.as_ref() {
            ui.data_mut(|data| {
                let link_groups: &mut BoundsLinkGroups = data.get_temp_mut_or_default(Id::NULL);
                if let Some(linked_bounds) = link_groups.0.get(id).filter(|lb| lb.bounds.is_valid())
                {
                    if axes.x {
                        bounds.set_x(&linked_bounds.bounds);
                        mem.auto_bounds.x = linked_bounds.auto_bounds.x;
                    }
                    if axes.y {
                        bounds.set_y(&linked_bounds.bounds);
                        mem.auto_bounds.y = linked_bounds.auto_bounds.y;
                    }
                }
            });
        }

        // Allow double-clicking or navigation button reset to reset to the initial bounds.
        let reset_view_requested = ui.ctx().data_mut(|d| {
            d.get_temp::<bool>(plot_id.with("reset_view"))
                .unwrap_or(false)
        });
        if (allow_double_click_reset && response.double_clicked()) || reset_view_requested {
            if reset_view_requested {
                ui.ctx().data_mut(|d| {
                    d.insert_temp(plot_id.with("reset_view"), false);
                });
            }
            mem.auto_bounds = true.into();
        }

        if last_plot_transform.log_axis().x != mem.log_x {
            mem.auto_bounds.x = true;
        }
        if last_plot_transform.log_axis().y != mem.log_y {
            mem.auto_bounds.y = true;
        }

        let any_dynamic_modifications = !bounds_modifications.is_empty();
        // Apply bounds modifications.
        for modification in bounds_modifications {
            match modification {
                BoundsModification::SetX(range) => {
                    bounds.min[0] = *range.start();
                    bounds.max[0] = *range.end();
                    mem.auto_bounds.x = false;
                }
                BoundsModification::SetY(range) => {
                    bounds.min[1] = *range.start();
                    bounds.max[1] = *range.end();
                    mem.auto_bounds.y = false;
                }
                BoundsModification::Translate(delta) => {
                    let delta = (delta.x as f64, delta.y as f64);
                    bounds.translate(delta);
                    mem.auto_bounds = false.into();
                }
                BoundsModification::AutoBounds(new_auto_bounds) => {
                    mem.auto_bounds = new_auto_bounds;
                }
                BoundsModification::Zoom(zoom_factor, center) => {
                    bounds.zoom_scaled(zoom_factor, center, Vec2b::new(mem.log_x, mem.log_y));
                    mem.auto_bounds = false.into();
                }
            }
        }

        let log_axes = Vec2b::new(mem.log_x, mem.log_y);
        apply_plot_bounds_with_scale(
            &mut bounds,
            &min_auto_bounds,
            default_auto_bounds,
            mem.auto_bounds,
            any_dynamic_modifications,
            &items,
            &mem.hidden_items,
            margin_fraction,
            log_axes,
        );
        let linked_union_auto = linked_axes.as_ref().map(|(link_id, link_axes)| {
            let local_auto = compute_auto_content_bounds(
                bounds,
                &min_auto_bounds,
                default_auto_bounds,
                &items,
                &mem.hidden_items,
                margin_fraction,
                log_axes,
            );
            sync_linked_auto_bounds_envelope(
                ui,
                *link_id,
                *link_axes,
                plot_id,
                local_auto,
                mem.auto_bounds,
                &mut bounds,
            )
        });

        let effective_center_axis =
            Vec2b::new(center_axis.x && !mem.log_x, center_axis.y && !mem.log_y);
        mem.transform = PlotTransform::new_with_invert_axis(
            plot_rect,
            bounds,
            effective_center_axis,
            Vec2b::new(invert_x, invert_y),
        );
        mem.transform.set_log_axis(log_axes);

        // Enforce aspect ratio
        if let Some(data_aspect) = data_aspect {
            if let Some((_, linked_axes)) = &linked_axes {
                let change_x = linked_axes.y && !linked_axes.x;
                mem.transform.set_aspect_by_changing_axis(
                    data_aspect as f64,
                    if change_x { Axis::X } else { Axis::Y },
                );
            } else if default_auto_bounds.any() {
                mem.transform.set_aspect_by_expanding(data_aspect as f64);
            } else {
                mem.transform
                    .set_aspect_by_changing_axis(data_aspect as f64, Axis::Y);
            }
        }

        if response.secondary_clicked() || response.clicked_by(PointerButton::Secondary) {
            let click_pos = response
                .interact_pointer_pos()
                .or_else(|| response.hover_pos());
            if let Some(pos) = click_pos {
                let pt = mem.transform.value_from_position(pos);
                mem.last_right_click_coord = Some([pt.x, pt.y]);
            }
        }

        let is_dragging_callout = ui.ctx().data(|d| {
            d.get_temp::<bool>(plot_id.with("dragging_callout"))
                .unwrap_or(false)
        });

        let overlay_buttons_rect = right_controls_rect(
            *last_plot_transform.frame(),
            show_settings_button,
            show_navigation_controls,
        )
        .unwrap_or(Rect::NOTHING);
        let drag_started_on_overlay = ui.input(|i| {
            i.pointer
                .press_origin()
                .is_some_and(|pos| overlay_buttons_rect.contains(pos))
        });

        // Dragging
        if allow_drag.any()
            && !is_dragging_callout
            && !drag_started_on_overlay
            && response.dragged_by(PointerButton::Primary)
        {
            response = response.on_hover_cursor(CursorIcon::Grabbing);
            let mut delta = -response.drag_delta();
            if !allow_drag.x {
                delta.x = 0.0;
            }
            if !allow_drag.y {
                delta.y = 0.0;
            }
            mem.transform
                .translate_bounds((delta.x as f64, delta.y as f64));
            mem.auto_bounds = mem.auto_bounds.and(!allow_drag);
        }

        // Drag axes to zoom:
        for d in 0..2 {
            if allow_axis_zoom_drag[d]
                && let Some(axis_response) = (if d == 0 {
                    &x_axis_responses
                } else {
                    &y_axis_responses
                })
                .iter()
                .find(|r| r.dragged_by(PointerButton::Primary))
                && let Some(drag_start_pos) = ui.input(|i| i.pointer.press_origin())
            {
                let delta = axis_response.drag_delta();

                let axis_zoom = 1.0 + (0.02 * delta[d]).clamp(-1.0, 1.0);

                let zoom = if data_aspect.is_some() {
                    // Zoom both axes equally to maintain aspect ratio:
                    Vec2::splat(axis_zoom)
                } else {
                    let mut zoom = Vec2::splat(1.0);
                    zoom[d] = axis_zoom;
                    zoom
                };

                if zoom != Vec2::splat(1.0) {
                    let mut zoom_center = plot_rect.center();
                    zoom_center[d] = drag_start_pos[d];
                    mem.transform.zoom(zoom, zoom_center);
                    mem.auto_bounds = false.into();
                }
            }
        }

        // Zooming
        let mut boxed_zoom_rect = None;
        if allow_boxed_zoom {
            // Save last click to allow boxed zooming
            if response.drag_started() && response.dragged_by(boxed_zoom_pointer_button) {
                // it would be best for egui that input has a memory of the last click pos because it's a common pattern
                mem.last_click_pos_for_zoom = response.hover_pos();
            }
            let box_start_pos = mem.last_click_pos_for_zoom;
            let box_end_pos = response.hover_pos();
            if let (Some(box_start_pos), Some(box_end_pos)) = (box_start_pos, box_end_pos) {
                // while dragging prepare a Shape and draw it later on top of the plot
                if response.dragged_by(boxed_zoom_pointer_button) {
                    response = response.on_hover_cursor(CursorIcon::ZoomIn);
                    let rect = epaint::Rect::from_two_pos(box_start_pos, box_end_pos);
                    boxed_zoom_rect = Some((
                        epaint::RectShape::stroke(
                            rect,
                            0.0,
                            epaint::Stroke::new(4_f32, Color32::DARK_BLUE),
                            egui::StrokeKind::Middle,
                        ), // Outer stroke
                        epaint::RectShape::stroke(
                            rect,
                            0.0,
                            epaint::Stroke::new(2_f32, Color32::WHITE),
                            egui::StrokeKind::Middle,
                        ), // Inner stroke
                    ));
                }
                // when the click is release perform the zoom
                if response.drag_stopped() {
                    let box_start_pos = mem.transform.value_from_position(box_start_pos);
                    let box_end_pos = mem.transform.value_from_position(box_end_pos);
                    let new_bounds = PlotBounds {
                        min: [
                            box_start_pos.x.min(box_end_pos.x),
                            box_start_pos.y.min(box_end_pos.y),
                        ],
                        max: [
                            box_start_pos.x.max(box_end_pos.x),
                            box_start_pos.y.max(box_end_pos.y),
                        ],
                    };
                    if new_bounds.is_valid() {
                        mem.transform.set_bounds(new_bounds);
                        mem.auto_bounds = false.into();
                    }
                    // reset the boxed zoom state
                    mem.last_click_pos_for_zoom = None;
                }
            }
        }

        // Note: we catch zoom/pan if the response contains the pointer, even if it isn't hovered.
        // For instance: The user is painting another interactive widget on top of the plot
        // but they still want to be able to pan/zoom the plot.
        if let (true, Some(hover_pos)) = (
            response.contains_pointer(),
            ui.input(|i| i.pointer.hover_pos()),
        ) {
            if allow_zoom.any() {
                let mut zoom_factor = if data_aspect.is_some() {
                    Vec2::splat(ui.input(|i| i.zoom_delta()))
                } else {
                    ui.input(|i| i.zoom_delta_2d())
                };
                if !allow_zoom.x {
                    zoom_factor.x = 1.0;
                }
                if !allow_zoom.y {
                    zoom_factor.y = 1.0;
                }
                if zoom_factor != Vec2::splat(1.0) {
                    mem.transform.zoom(zoom_factor, hover_pos);
                    mem.auto_bounds = mem.auto_bounds.and(!allow_zoom);
                }
            }
            if allow_scroll.any() {
                let mut scroll_delta = ui.input(|i| i.smooth_scroll_delta);
                if !allow_scroll.x {
                    scroll_delta.x = 0.0;
                }
                if !allow_scroll.y {
                    scroll_delta.y = 0.0;
                }
                if scroll_delta != Vec2::ZERO {
                    mem.transform
                        .translate_bounds((-scroll_delta.x as f64, -scroll_delta.y as f64));
                    mem.auto_bounds = false.into();
                }
            }
        }

        // --- transform initialized

        // Apply dynamic grid settings from PlotMemory
        let active_grid_spacing = Rangef::new(mem.grid_density, grid_spacing.max);
        let active_show_x = show_x && mem.show_hover;
        let active_show_y = show_y && mem.show_hover;
        let active_show_grid = Vec2b::new(
            show_grid.x && mem.show_grid.x,
            show_grid.y && mem.show_grid.y,
        );
        let active_clamp_grid = clamp_grid || mem.clamp_grid;

        let [default_x_spacer, default_y_spacer] = grid_spacers;
        let active_x_spacer: GridSpacer<'a> = if mem.log_x {
            logarithmic_grid_spacer()
        } else {
            default_x_spacer
        };
        let active_y_spacer: GridSpacer<'a> = if mem.log_y {
            logarithmic_grid_spacer()
        } else {
            default_y_spacer
        };
        let active_grid_spacers = [active_x_spacer, active_y_spacer];

        // Add legend widgets to plot
        let bounds = mem.transform.bounds();
        let x_axis_range = bounds.range_x();
        let x_steps = Arc::new({
            let input = GridInput {
                bounds: (bounds.min[0], bounds.max[0]),
                base_step_size: mem.transform.dvalue_dpos()[0].abs()
                    * active_grid_spacing.min as f64,
            };
            (active_grid_spacers[0])(input)
        });
        let y_axis_range = bounds.range_y();
        let y_steps = Arc::new({
            let input = GridInput {
                bounds: (bounds.min[1], bounds.max[1]),
                base_step_size: mem.transform.dvalue_dpos()[1].abs()
                    * active_grid_spacing.min as f64,
            };
            (active_grid_spacers[1])(input)
        });
        for (i, mut widget) in x_axis_widgets.into_iter().enumerate() {
            widget.range = x_axis_range.clone();
            widget.transform = Some(mem.transform);
            widget.steps = x_steps.clone();
            let (_response, thickness) = widget.ui(ui, Axis::X);
            mem.x_axis_thickness.insert(i, thickness);
        }
        for (i, mut widget) in y_axis_widgets.into_iter().enumerate() {
            widget.range = y_axis_range.clone();
            widget.transform = Some(mem.transform);
            widget.steps = y_steps.clone();
            let (_response, thickness) = widget.ui(ui, Axis::Y);
            mem.y_axis_thickness.insert(i, thickness);
        }

        // Initialize values from functions.
        for item in &mut items {
            item.initialize(mem.transform.bounds().range_x());
        }

        let mut auto_content_bounds = compute_auto_content_bounds(
            *bounds,
            &min_auto_bounds,
            default_auto_bounds,
            &items,
            &mem.hidden_items,
            margin_fraction,
            log_axes,
        );
        if let (Some((_, link_axes)), Some(union_auto)) = (&linked_axes, linked_union_auto) {
            if link_axes.x && union_auto.is_valid_x() {
                auto_content_bounds.set_x(&union_auto);
            }
            if link_axes.y && union_auto.is_valid_y() {
                auto_content_bounds.set_y(&union_auto);
            }
        }

        let mut access_state = Self::load_frame_access_state(ui, plot_id, &mem);
        if !allow_2d_analysis {
            access_state.show_add_callout_modal = false;
            access_state.show_edit_callout_modal = false;
            access_state.focused_poi = None;
        }
        let export_datasets_3d: Vec<(String, Vec<[f64; 3]>)> = sidecar_3d
            .as_ref()
            .map_or_else(Vec::new, Plot3DSeriesSidecar::export_datasets_3d);
        append_last_plot_datasets_3d(&export_datasets_3d);

        let base_items_hash = hash_visible_plot_items(&items, &mem.hidden_items);
        let fit_status = if !allow_2d_analysis || access_state.fit_model.is_none() {
            access_state.last_fit_error_announced = None;
            fit_overlay::FitOverlayStatus::Inactive
        } else if !access_state.show_fit_overlay {
            fit_overlay::FitOverlayStatus::Hidden
        } else {
            let base_ds = Arc::clone(get_or_populate_visible_base_datasets(
                &items,
                &mem.hidden_items,
                base_items_hash,
                &mut access_state.cached_base_datasets,
            ));
            fit_overlay::compute_and_inject_fit_items(
                ui,
                &mut items,
                &base_ds,
                &mut access_state,
                &mem.transform,
                mem.log_x,
                mem.log_y,
            )
        };

        let right_ctrl_rect =
            right_controls_rect(plot_rect, show_settings_button, show_navigation_controls);
        let legend = legend_config.and_then(|config| {
            let safe_rect = legend_safe_rect(plot_rect, config.position, right_ctrl_rect);
            LegendWidget::try_new(plot_id, safe_rect, config, &items, &mem.hidden_items)
        });
        items.retain(|item| !mem.hidden_items.contains(&item.id()));
        if let Some(item_id) = &mem.hovered_legend_item {
            items
                .iter_mut()
                .filter(|entry| &entry.id() == item_id)
                .for_each(|entry| entry.highlight());
        }
        items.sort_by_key(|item| item.highlighted());

        let items_hash = if fit_status.did_inject_items() || mem.hovered_legend_item.is_some() {
            hash_plot_items(&items)
        } else {
            base_items_hash
        };
        let critical_points: Arc<[poi::CriticalPoint]> = if !allow_2d_analysis {
            Arc::from(&[][..])
        } else {
            let bounds_val = *mem.transform.bounds();
            let poi_key = PoiCacheKey {
                items_hash,
                bounds_bits: [
                    bounds_val.min[0].to_bits(),
                    bounds_val.min[1].to_bits(),
                    bounds_val.max[0].to_bits(),
                    bounds_val.max[1].to_bits(),
                ],
                log_x: mem.log_x,
                log_y: mem.log_y,
            };
            if let Some((cached_key, cached_pts)) = &access_state.cached_pois
                && *cached_key == poi_key
            {
                Arc::clone(cached_pts)
            } else {
                let detected: Arc<[poi::CriticalPoint]> =
                    Arc::from(poi::detect_critical_points(&items, &mem.transform));
                access_state.cached_pois = Some((poi_key, Arc::clone(&detected)));
                detected
            }
        };
        if response.has_focus() {
            let arrow_pressed = ui.input(|i| {
                !i.key_pressed(egui::Key::Tab)
                    && (i.key_pressed(egui::Key::ArrowLeft)
                        || i.key_pressed(egui::Key::ArrowRight)
                        || i.key_pressed(egui::Key::ArrowUp)
                        || i.key_pressed(egui::Key::ArrowDown))
            });
            ui.memory_mut(|m| {
                m.set_focus_lock_filter(
                    response.id,
                    egui::EventFilter {
                        tab: false,
                        horizontal_arrows: true,
                        vertical_arrows: true,
                        escape: false,
                    },
                );
                if arrow_pressed {
                    m.move_focus(egui::FocusDirection::None);
                }
            });
            if response.gained_focus() {
                ui.ctx().request_repaint();
            }
        }

        let viewport_key_cfg = ViewportKeyboardConfig {
            plot_id,
            allow_zoom,
            allow_pan: allow_drag.or(allow_scroll),
            data_aspect,
            linked_axes: &linked_axes,
            auto_content_bounds,
            default_auto_bounds,
        };
        Self::register_and_handle_viewport_keys(ui, &response, &mut mem, &viewport_key_cfg);
        register_plot_dataset_commands_from_items(
            ui.ctx(),
            &items,
            sidecar_3d.as_ref(),
            access_state.focused_item,
        );
        if !critical_points.is_empty() {
            commands::CommandRegistryData::register_poi_navigation(ui.ctx());
        }
        let lazy_accessible_datasets: Option<BaseDatasetList> =
            (response.has_focus() || access_state.show_table).then(|| {
                if let Some(ref sidecar) = sidecar_3d {
                    let ds = sidecar.accessible_datasets_2d();
                    if !ds.is_empty() {
                        return ds;
                    }
                }
                collect_accessible_datasets(&items)
            });
        if response.has_focus() {
            if allow_2d_analysis {
                poi_interaction::handle_focused_plot_poi_keys(
                    ui,
                    &critical_points,
                    &mut access_state,
                );
            }
            if let Some(ref ds) = lazy_accessible_datasets {
                handle_focused_plot_dataset_keys(ui, ds, sidecar_3d.as_ref(), &mut access_state);
            }
        } else if access_state
            .focused_poi
            .is_some_and(|idx| idx >= critical_points.len())
        {
            access_state.focused_poi = None;
        }
        let focused_point = if response.has_focus() && access_state.focused_poi.is_none() {
            lazy_accessible_datasets.as_ref().and_then(|ds| {
                resolve_focused_plot_point(&items, ds, sidecar_3d.as_ref(), &mut access_state)
            })
        } else {
            None
        };

        let prepared = PreparedPlot {
            plot_area_response: &response,
            items,
            show_x: active_show_x,
            show_y: active_show_y,
            allow_2d_analysis,
            label_formatter,
            coordinates_formatter,
            show_grid: active_show_grid,
            grid_spacing: active_grid_spacing,
            transform: mem.transform,
            draw_cursor_x: linked_cursors.as_ref().is_some_and(|group| group.1.x),
            draw_cursor_y: linked_cursors.as_ref().is_some_and(|group| group.1.y),
            draw_cursors,
            cursor_color,
            cursor_snap_distance,
            grid_spacers: active_grid_spacers,
            clamp_grid: active_clamp_grid,
            plot_id,
            focused_point,
            focused_poi: (allow_2d_analysis && response.has_focus())
                .then_some(access_state.focused_poi)
                .flatten(),
            persisted_tooltip: persisted_point_tooltip,
            sidecar_3d: sidecar_3d.as_ref(),
            critical_points: Arc::clone(&critical_points),
            show_settings_button,
            right_controls_rect: right_ctrl_rect,
        };

        let PreparedUiOutput {
            mut items,
            cursors: plot_cursors,
            mut hovered_plot_item,
            hovered_poi_idx,
        } = prepared.ui(ui, &response);
        if response.has_focus() {
            let focus_stroke = a11y::focus_ring_stroke(ui.visuals(), ui.visuals().extreme_bg_color);
            ui.painter()
                .with_clip_rect(plot_rect)
                .add(epaint::RectShape::stroke(
                    plot_rect,
                    2,
                    focus_stroke,
                    egui::StrokeKind::Inside,
                ));
        }

        if let Some(boxed_zoom_rect) = boxed_zoom_rect {
            ui.painter()
                .with_clip_rect(plot_rect)
                .add(boxed_zoom_rect.0);
            ui.painter()
                .with_clip_rect(plot_rect)
                .add(boxed_zoom_rect.1);
        }

        let mut legend_placement = None;
        if let Some(mut legend) = legend {
            ui.add(&mut legend);
            legend_placement = legend
                .rendered_rect()
                .map(|lrect| (legend.position(), lrect));
            mem.hidden_items = legend.hidden_items();
            mem.hovered_legend_item = legend.hovered_item();

            if let Some(item_id) = &mem.hovered_legend_item {
                hovered_plot_item.get_or_insert(*item_id);
            }
        }

        if let Some((id, _)) = linked_cursors.as_ref() {
            // Push the frame we just drew to the list of frames
            let had_peers_with_new_cursor = ui.data_mut(|data| {
                let frames: &mut CursorLinkGroups = data.get_temp_mut_or_default(Id::NULL);
                let cursors = frames.0.entry(*id).or_default();
                let had_peers = !cursors.is_empty() && !plot_cursors.is_empty();
                cursors.push(PlotFrameCursors {
                    id: plot_id,
                    cursors: plot_cursors,
                });
                had_peers
            });
            if had_peers_with_new_cursor {
                ui.ctx().request_repaint();
            }
        }

        if let Some((id, _)) = linked_axes.as_ref() {
            // Save the linked bounds while preserving per-plot auto-bounds envelopes.
            let changed = ui.data_mut(|data| {
                let link_groups: &mut BoundsLinkGroups = data.get_temp_mut_or_default(Id::NULL);
                let entry = link_groups.0.entry(*id).or_default();
                let new_bounds = *mem.transform.bounds();
                let needs_sync = entry.auto_frames.len() > 1
                    && (entry.bounds != new_bounds || entry.auto_bounds != mem.auto_bounds);
                entry.bounds = new_bounds;
                entry.auto_bounds = mem.auto_bounds;
                needs_sync
            });
            if changed {
                ui.ctx().request_repaint();
            }
        }

        let transform = mem.transform;

        // --- ACCESSIBILITY LOGIC INJECTED ---
        data_table::set_data_table_allow_csv_import(ui.ctx(), plot_id, effective_allow_csv_import);
        if let Some(status) = export::drain_import_status(plot_id)
            && effective_allow_csv_import
        {
            access_state.import_toast = Some(status);
        }
        plot_overlays::handle_plot_csv_inputs(
            ui,
            &response,
            plot_rect,
            plot_id,
            effective_allow_csv_import,
            &mut mem,
            &mut access_state,
        );

        if ui.ctx().data_mut(|d| {
            d.get_temp::<bool>(plot_id.with("toggle_settings"))
                .unwrap_or(false)
        }) {
            access_state.show_settings_modal = !access_state.show_settings_modal;
            ui.ctx()
                .data_mut(|d| d.insert_temp(plot_id.with("toggle_settings"), false));
        }

        let has_fit_candidates =
            allow_2d_analysis && items.iter().any(|it| item_point_count(&**it) >= 2);
        if allow_2d_analysis {
            let fit_layout = fit_overlay::FitOverlayLayout {
                plot_rect,
                legend_placement,
                right_controls_rect: right_ctrl_rect,
                dataset_idx: access_state.fit_dataset_idx,
            };
            fit_overlay::paint_fit_status_overlay(ui, &fit_layout, &fit_status, &mut access_state);
        }

        let focus_response = &response;
        let logical_item_count = items
            .iter()
            .filter(|it| !it.is_companion_visual_only())
            .count();
        let series_count = sidecar_3d
            .as_ref()
            .filter(|s| !s.series().is_empty())
            .map_or(logical_item_count, |s| s.series().len());
        let has_any_export_points =
            !export_datasets_3d.is_empty() || items.iter().any(|it| it.has_any_points());
        let has_table_content = has_any_export_points
            || !critical_points.is_empty()
            || (effective_allow_csv_import && items.is_empty());
        let plot_a11y_label = match title.as_deref() {
            Some(t) => format!("Interactive Plot ({t}), {series_count} series"),
            None => format!("Interactive Plot, {series_count} series"),
        };
        let plot_enabled = ui.is_enabled();
        focus_response.widget_info({
            let label = plot_a11y_label.clone();
            move || egui::WidgetInfo::labeled(egui::WidgetType::Other, plot_enabled, label.clone())
        });
        ui.ctx()
            .accesskit_node_builder(focus_response.id, |builder| {
                builder.set_role(egui::accesskit::Role::GraphicsDocument);
                builder.set_label(plot_a11y_label);
            });

        if header_h > 0.0 {
            let toolbar_cfg = plot_overlays::PlotToolbarConfig {
                plot_id,
                header_rect,
                allow_csv_import: effective_allow_csv_import,
                allow_2d_analysis,
                has_table_content,
                has_fit_candidates,
                base_items_hash,
                items: &items,
                export_datasets_3d: &export_datasets_3d,
                fit_status: &fit_status,
                viewport_sync: plot_overlays::OverlayViewportSync {
                    data_aspect,
                    linked_axes: &linked_axes,
                    default_auto_bounds,
                },
            };
            plot_overlays::render_plot_top_toolbar(ui, &toolbar_cfg, &mut mem, &mut access_state);
        }

        if show_navigation_controls {
            let zoom_config = FloatingZoomConfig {
                plot_id,
                allow_zoom,
                data_aspect,
                linked_axes: &linked_axes,
                auto_content_bounds,
                default_auto_bounds,
                show_settings_button,
            };
            Self::render_floating_zoom_toolbar(ui, &mut mem, &zoom_config);
        }

        {
            let table_window_title = format_modal_title("Data Table", title.as_deref());
            if let Some(ref sidecar) = sidecar_3d {
                data_table::show_data_table_modal_3d(
                    ui.ctx(),
                    plot_id,
                    &table_window_title,
                    &mut access_state.show_table,
                    &mut access_state.data_table,
                    sidecar.series(),
                );
            } else {
                let table_datasets_storage;
                let table_datasets: &[(String, Vec<[f64; 2]>, Color32)] =
                    if let Some(ref ds) = lazy_accessible_datasets {
                        ds.as_slice()
                    } else if access_state.show_table {
                        table_datasets_storage = collect_accessible_datasets(&items);
                        table_datasets_storage.as_slice()
                    } else {
                        &[]
                    };
                data_table::show_data_table_modal_with_pois(
                    ui.ctx(),
                    plot_id,
                    &table_window_title,
                    &mut access_state.show_table,
                    &mut access_state.data_table,
                    table_datasets,
                    &critical_points,
                );
            }
        }

        {
            access_state.export_modal.options.log_x = mem.log_x;
            access_state.export_modal.options.log_y = mem.log_y;
            let export_window_title = format_modal_title("Export High-Res Plot", title.as_deref());
            let export_config = export_modal::ExportModalConfig {
                window_title: &export_window_title,
                id: plot_id,
                png_filename: "plot.png",
                svg_filename: "plot.svg",
                gif_filename: "plot.gif",
            };
            let mut export_source = export_modal::CallbackExportSource {
                datasets_fn: || collect_export_datasets(&items),
                sample_frames_fn: |num_frames| {
                    let ds = collect_export_datasets(&items);
                    export::sample_progressive_frames(&ds, num_frames)
                },
            };
            export_modal::show_export_modal(
                ui.ctx(),
                access_state.export_modal.bindings(),
                &export_config,
                &mut export_source,
            );
        }

        ModalLifecycle::new(plot_id.with("plot_settings_modal"))
            .title(format_modal_title("Plot Settings 🔧", title.as_deref()))
            .default_width(320.0)
            .show(
                ui.ctx(),
                &mut access_state.show_settings_modal,
                |ui, frame_ctx| {
                    ui.label(egui::RichText::new("Grid & Axis Settings").strong());
                    ui.separator();

                    ui.label(
                        egui::RichText::new("Axis Scale Transformations")
                            .small()
                            .weak(),
                    );
                    ui.checkbox(&mut access_state.log_x, "Logarithmic X-Axis");
                    ui.checkbox(&mut access_state.log_y, "Logarithmic Y-Axis");

                    ui.separator();
                    ui.label(egui::RichText::new("Grid Line Density").small().weak());
                    ui.horizontal(|ui| {
                        ui.label("Min Line Spacing (px):");
                        ui.add(egui::Slider::new(
                            &mut access_state.grid_density,
                            4.0..=100.0,
                        ));
                    });

                    ui.separator();
                    ui.label(
                        egui::RichText::new("Axis Labels & Visibility")
                            .small()
                            .weak(),
                    );
                    ui.checkbox(&mut access_state.show_x_labels, "Show X-Axis Labels");
                    ui.checkbox(&mut access_state.show_y_labels, "Show Y-Axis Labels");
                    ui.checkbox(&mut access_state.show_hover, "Show Hover Readouts");
                    ui.checkbox(&mut mem.show_grid.x, "Show X Grid Lines");
                    ui.checkbox(&mut mem.show_grid.y, "Show Y Grid Lines");
                    if show_toolbar {
                        let mut show_bar = !mem.toolbar_collapsed;
                        if ui.checkbox(&mut show_bar, "Show Plot Toolbar").changed() {
                            mem.toolbar_collapsed = !show_bar;
                            ui.ctx().request_repaint();
                        }
                    }

                    ui.separator();
                    ui.label(egui::RichText::new("Grid Bounds").small().weak());
                    ui.checkbox(&mut access_state.clamp_grid, "Clamp Grid to Data Range");

                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Reset Defaults").clicked() {
                            access_state.log_x = false;
                            access_state.log_y = false;
                            access_state.grid_density = 8.0;
                            access_state.show_x_labels = true;
                            access_state.show_y_labels = true;
                            access_state.show_hover = true;
                            access_state.clamp_grid = false;
                            mem.show_grid = Vec2b::new(true, true);
                            mem.toolbar_collapsed = false;
                        }
                        if ui.button("Close").clicked() {
                            frame_ctx.close();
                        }
                    });
                },
            );

        {
            let mut add_confirmed = false;
            ModalLifecycle::new(plot_id.with("add_callout_modal"))
                .title(format_modal_title("Add Callout", title.as_deref()))
                .default_width(320.0)
                .show(
                    ui.ctx(),
                    &mut access_state.show_add_callout_modal,
                    |ui, frame_ctx| {
                        ui.label(egui::RichText::new("New Callout Annotation").strong());
                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Target Coordinates:");
                            let x_str = format::format_readout_value(
                                access_state.new_callout_point[0],
                                4,
                                4,
                            );
                            let y_str = format::format_readout_value(
                                access_state.new_callout_point[1],
                                4,
                                4,
                            );
                            ui.monospace(format!("X: {x_str}, Y: {y_str}"));
                        });

                        ui.add_space(4.0);
                        ui.label("Label Text (max 128 chars):");
                        let text_edit_res = ui.add(
                            egui::TextEdit::singleline(&mut access_state.new_callout_text)
                                .char_limit(128)
                                .hint_text("e.g. Equilibrium / Critical Point"),
                        );
                        if frame_ctx.just_opened {
                            text_edit_res.request_focus();
                        } else if text_edit_res.lost_focus()
                            && ui.ctx().input(|i| i.key_pressed(egui::Key::Enter))
                        {
                            add_confirmed = true;
                        }

                        if access_state.new_callout_text.chars().count() > 128 {
                            access_state.new_callout_text =
                                access_state.new_callout_text.chars().take(128).collect();
                        }

                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("Screen Pixel Offset").small().weak());
                        ui.horizontal(|ui| {
                            ui.label("X Offset (dx):");
                            ui.add(
                                egui::DragValue::new(&mut access_state.new_callout_offset[0])
                                    .speed(1.0),
                            );
                            ui.label("px");
                        });
                        ui.horizontal(|ui| {
                            ui.label("Y Offset (dy):");
                            ui.add(
                                egui::DragValue::new(&mut access_state.new_callout_offset[1])
                                    .speed(1.0),
                            );
                            ui.label("px");
                        });

                        ui.separator();
                        ui.horizontal(|ui| {
                            if ui.button("Add Callout").clicked() {
                                add_confirmed = true;
                            }
                            if ui.button("Cancel").clicked() {
                                frame_ctx.close();
                            }
                        });
                        if add_confirmed {
                            frame_ctx.close();
                        }
                    },
                );

            if add_confirmed {
                let label = if access_state.new_callout_text.trim().is_empty() {
                    let x_str =
                        format::format_readout_value(access_state.new_callout_point[0], 2, 4);
                    let y_str =
                        format::format_readout_value(access_state.new_callout_point[1], 2, 4);
                    format!("({x_str}, {y_str})")
                } else {
                    access_state.new_callout_text.clone()
                };
                push_pending_callout(
                    plot_id,
                    CanvasCallout {
                        target_point: access_state.new_callout_point,
                        label_offset: access_state.new_callout_offset,
                        text: label,
                    },
                );
                ui.ctx().request_repaint();
            }
        }

        {
            let valid_callout_idx = access_state
                .editing_callout_idx
                .filter(|&idx| idx < mem.callouts.len());
            if access_state.show_edit_callout_modal && valid_callout_idx.is_none() {
                access_state.show_edit_callout_modal = false;
            }
            let mut delete_requested = false;
            let mut update_confirmed = false;
            ModalLifecycle::new(plot_id.with("edit_callout_modal"))
                .title(format_modal_title("Edit Callout", title.as_deref()))
                .default_width(320.0)
                .show(
                    ui.ctx(),
                    &mut access_state.show_edit_callout_modal,
                    |ui, frame_ctx| {
                        let Some(idx) = valid_callout_idx else {
                            frame_ctx.close();
                            return;
                        };
                        ui.label(egui::RichText::new("Edit Callout Annotation").strong());
                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Target Coordinates:");
                            let x_str =
                                format::format_readout_value(mem.callouts[idx].point[0], 4, 4);
                            let y_str =
                                format::format_readout_value(mem.callouts[idx].point[1], 4, 4);
                            ui.monospace(format!("X: {x_str}, Y: {y_str}"));
                        });

                        ui.add_space(4.0);
                        ui.label("Label Text (max 128 chars):");
                        let text_edit_res = ui.add(
                            egui::TextEdit::singleline(&mut access_state.edit_callout_text)
                                .char_limit(128),
                        );
                        if frame_ctx.just_opened {
                            text_edit_res.request_focus();
                        }

                        if access_state.edit_callout_text.chars().count() > 128 {
                            access_state.edit_callout_text =
                                access_state.edit_callout_text.chars().take(128).collect();
                        }

                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("Screen Pixel Offset").small().weak());
                        ui.horizontal(|ui| {
                            ui.label("X Offset (dx):");
                            ui.add(
                                egui::DragValue::new(&mut access_state.edit_callout_offset[0])
                                    .speed(1.0),
                            );
                            ui.label("px");
                        });
                        ui.horizontal(|ui| {
                            ui.label("Y Offset (dy):");
                            ui.add(
                                egui::DragValue::new(&mut access_state.edit_callout_offset[1])
                                    .speed(1.0),
                            );
                            ui.label("px");
                        });

                        ui.separator();
                        ui.horizontal(|ui| {
                            if ui.button("Update").clicked() {
                                update_confirmed = true;
                            }
                            if ui.button("Remove").clicked() {
                                delete_requested = true;
                            }
                            if ui.button("Cancel").clicked() {
                                frame_ctx.close();
                            }
                        });
                        if update_confirmed || delete_requested {
                            frame_ctx.close();
                        }
                    },
                );

            if let Some(idx) = valid_callout_idx {
                if update_confirmed {
                    let new_label = if access_state.edit_callout_text.trim().is_empty() {
                        let x_str = format::format_readout_value(mem.callouts[idx].point[0], 2, 4);
                        let y_str = format::format_readout_value(mem.callouts[idx].point[1], 2, 4);
                        format!("({x_str}, {y_str})")
                    } else {
                        access_state.edit_callout_text.clone()
                    };
                    mem.callouts[idx].label = new_label;
                    mem.callouts[idx].offset = access_state.edit_callout_offset;
                }
                if delete_requested {
                    mem.callouts.remove(idx);
                }
            }
        }

        mem.show_settings_modal = access_state.show_settings_modal;
        if mem.log_x != access_state.log_x {
            mem.log_x = access_state.log_x;
            mem.auto_bounds.x = true;
            ui.ctx().request_repaint();
        }
        if mem.log_y != access_state.log_y {
            mem.log_y = access_state.log_y;
            mem.auto_bounds.y = true;
            ui.ctx().request_repaint();
        }
        mem.grid_density = access_state.grid_density;
        if mem.show_x_labels != access_state.show_x_labels {
            mem.show_x_labels = access_state.show_x_labels;
            mem.x_axis_thickness.clear();
            ui.ctx().request_repaint();
        }
        if mem.show_y_labels != access_state.show_y_labels {
            mem.show_y_labels = access_state.show_y_labels;
            mem.y_axis_thickness.clear();
            ui.ctx().request_repaint();
        }
        mem.show_hover = access_state.show_hover;
        mem.clamp_grid = access_state.clamp_grid;

        // Track pointer position on secondary click (right-click) for callout placement
        if response.secondary_clicked()
            || (response.hovered()
                && ui
                    .ctx()
                    .input(|i| i.pointer.button_clicked(egui::PointerButton::Secondary)))
        {
            let click_pos = response
                .interact_pointer_pos()
                .or_else(|| ui.ctx().input(|i| i.pointer.interact_pos()))
                .or_else(|| ui.ctx().input(|i| i.pointer.latest_pos()))
                .unwrap_or_else(|| transform.position_from_point(&transform.bounds().center()));
            let plot_pt = transform.value_from_position(click_pos);
            access_state.last_context_plot_pos = Some([plot_pt.x, plot_pt.y]);
        }

        // Callout badge interaction (clicking callout badge to edit / remove)
        if allow_2d_analysis
            && response.clicked()
            && let Some(click_pos) = response
                .interact_pointer_pos()
                .or_else(|| ui.ctx().input(|i| i.pointer.interact_pos()))
        {
            let mut clicked_idx = None;
            for (idx, callout) in mem.callouts.iter().enumerate().rev() {
                let galley = ui.painter().layout_no_wrap(
                    if callout.label.is_empty() {
                        let x_str = format::format_readout_value(callout.point[0], 2, 4);
                        let y_str = format::format_readout_value(callout.point[1], 2, 4);
                        format!("({x_str}, {y_str})")
                    } else {
                        callout.label.clone()
                    },
                    egui::TextStyle::Small.resolve(ui.style()),
                    egui::Color32::WHITE,
                );
                let padding = egui::Vec2::new(6.0, 4.0);
                let frame_size = galley.size() + 2.0 * padding;
                let target_pos = transform
                    .position_from_point(&PlotPoint::new(callout.point[0], callout.point[1]));
                let anchor_pos = target_pos + egui::Vec2::new(callout.offset[0], callout.offset[1]);
                let frame_rect = egui::Align2::CENTER_CENTER.anchor_size(anchor_pos, frame_size);
                if frame_rect.expand(4.0).contains(click_pos) {
                    clicked_idx = Some(idx);
                    break;
                }
            }
            if let Some(idx) = clicked_idx {
                access_state.editing_callout_idx = Some(idx);
                access_state.edit_callout_text = mem.callouts[idx].label.clone();
                access_state.edit_callout_offset = mem.callouts[idx].offset;
                access_state.show_edit_callout_modal = true;
            } else {
                let snap_px =
                    cursor_snap_distance.unwrap_or_else(|| ui.style().interaction.interact_radius);
                let _ = poi_interaction::try_pin_clicked_poi(
                    ui,
                    click_pos,
                    hovered_poi_idx,
                    &critical_points,
                    &transform,
                    snap_px,
                    &mut mem,
                );
            }
        }

        // Hover cursor over callout badges or critical point markers
        if allow_2d_analysis {
            if hovered_poi_idx.is_some() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            } else if let Some(hover_pos) = response.hover_pos() {
                for callout in &mem.callouts {
                    let galley = ui.painter().layout_no_wrap(
                        if callout.label.is_empty() {
                            let x_str = format::format_readout_value(callout.point[0], 2, 4);
                            let y_str = format::format_readout_value(callout.point[1], 2, 4);
                            format!("({x_str}, {y_str})")
                        } else {
                            callout.label.clone()
                        },
                        egui::TextStyle::Small.resolve(ui.style()),
                        egui::Color32::WHITE,
                    );
                    let padding = egui::Vec2::new(6.0, 4.0);
                    let frame_size = galley.size() + 2.0 * padding;
                    let target_pos = transform
                        .position_from_point(&PlotPoint::new(callout.point[0], callout.point[1]));
                    let anchor_pos =
                        target_pos + egui::Vec2::new(callout.offset[0], callout.offset[1]);
                    let frame_rect =
                        egui::Align2::CENTER_CENTER.anchor_size(anchor_pos, frame_size);
                    if frame_rect.expand(4.0).contains(hover_pos) {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        break;
                    }
                }
            }
        }

        let response = if allow_2d_analysis && (active_show_x || active_show_y) {
            response.on_hover_cursor(CursorIcon::Crosshair)
        } else {
            response
        };

        response.context_menu(|ui| {
            if allow_2d_analysis {
                if ui.button("Add Callout").clicked() {
                    let default_pt = access_state
                        .last_context_plot_pos
                        .or(mem.last_right_click_coord)
                        .unwrap_or_else(|| {
                            [transform.bounds().center().x, transform.bounds().center().y]
                        });
                    access_state.new_callout_point = default_pt;
                    access_state.new_callout_text = String::new();
                    access_state.new_callout_offset = [30.0, -30.0];
                    access_state.show_add_callout_modal = true;
                    ui.close();
                }

                if !mem.callouts.is_empty() {
                    if ui.button("Clear All Callouts").clicked() {
                        mem.callouts.clear();
                        ui.close();
                    }
                    ui.menu_button("Delete Callout", |ui| {
                        let mut to_remove = None;
                        for (i, c) in mem.callouts.iter().enumerate() {
                            let label = if c.label.is_empty() {
                                format!("Callout #{} ({:.2}, {:.2})", i + 1, c.point[0], c.point[1])
                            } else {
                                format!("{}: ({:.2}, {:.2})", c.label, c.point[0], c.point[1])
                            };
                            if ui.button(label).clicked() {
                                to_remove = Some(i);
                                ui.close();
                            }
                        }
                        if let Some(i) = to_remove {
                            mem.callouts.remove(i);
                        }
                    });
                }

                ui.separator();
            }

            if effective_allow_csv_import {
                if ui.button("Import CSV").clicked() {
                    export::trigger_import_csv_for_plot(ui.ctx(), plot_id);
                    ui.close();
                }
                if !mem.imported_datasets.is_empty() {
                    let sync = plot_overlays::OverlayViewportSync {
                        data_aspect,
                        linked_axes: &linked_axes,
                        default_auto_bounds,
                    };
                    ui.menu_button("Overlays", |ui| {
                        plot_overlays::render_overlays_menu_contents(
                            ui,
                            &mut mem,
                            &mut access_state,
                            &sync,
                            true,
                        );
                    });
                }
            }
            if ui.button("Save PNG").clicked() {
                access_state.export_modal.open = true;
                ui.close();
            }
            if ui.button("Export CSV").clicked() {
                if !export_datasets_3d.is_empty() {
                    export::export_csv_3d(&export_datasets_3d);
                } else {
                    let export_datasets = collect_export_datasets(&items);
                    export::export_csv(&export_datasets);
                }
                ui.close();
            }
            if ui.button("Export JSON").clicked() {
                if !export_datasets_3d.is_empty() {
                    export::export_json_3d(&export_datasets_3d);
                } else {
                    let export_datasets = collect_export_datasets(&items);
                    export::export_json(&export_datasets);
                }
                ui.close();
            }
            let mut table_btn = a11y::focusable_action_button(
                ui,
                has_table_content,
                egui::Button::new("Table View"),
                Some("Table View"),
            );
            if !has_table_content {
                table_btn =
                    table_btn.accessible_tooltip("No tabular data points available for this plot");
            }
            if has_table_content && table_btn.clicked() {
                access_state.show_table = !access_state.show_table;
                ui.close();
            }
            if show_toolbar {
                let bar_toggle_label = if mem.toolbar_collapsed {
                    "Show Toolbar"
                } else {
                    "Hide Toolbar"
                };
                if ui.button(bar_toggle_label).clicked() {
                    mem.toolbar_collapsed = !mem.toolbar_collapsed;
                    ui.ctx().request_repaint();
                    ui.close();
                }
            }
            if ui.button("🔧 Plot Settings").clicked() {
                access_state.show_settings_modal = true;
                mem.show_settings_modal = true;
                ui.close();
            }

            if allow_2d_analysis {
                ui.separator();

                ui.menu_button("Fit Curve", |ui| {
                    let base_ds = Arc::clone(get_or_populate_base_datasets(
                        &items,
                        base_items_hash,
                        &mut access_state.cached_base_datasets,
                    ));
                    fit_overlay::render_fit_curve_menu(
                        ui,
                        &base_ds,
                        &mut access_state,
                        true,
                        &fit_status,
                    );
                });
            }
        });

        let mut deferred_datasets = Vec::with_capacity(items.len());
        let mut deferred_primitives = Vec::with_capacity(items.len());
        if let Some(ref sidecar) = sidecar_3d
            && !sidecar.series().is_empty()
        {
            for s in sidecar.series() {
                if !s.projected.is_empty() {
                    deferred_datasets.push(DeferredDatasetEntry::Owned {
                        name: s.name.clone(),
                        points: s.projected.clone(),
                    });
                }
            }
            let mut ignored_datasets = Vec::new();
            for item in &mut items {
                item.append_deferred_export(&mut ignored_datasets, &mut deferred_primitives);
            }
        } else {
            let mut ignored_datasets = Vec::new();
            for item in &mut items {
                if item.is_companion_visual_only() {
                    item.append_deferred_export(&mut ignored_datasets, &mut deferred_primitives);
                } else {
                    item.append_deferred_export(&mut deferred_datasets, &mut deferred_primitives);
                }
            }
        }
        append_deferred_plot_datasets(deferred_datasets, items_hash);
        export_primitives::append_deferred_plot_primitives(deferred_primitives);

        snapshot::sync_plot_after_show(
            ui.ctx(),
            &snapshot::AfterShowSyncParams {
                plot_key: &plot_key,
                default_auto_bounds,
                min_auto_bounds,
                has_data_aspect: data_aspect.is_some(),
                allow_bounds_persist: allow_bounds_persist && sidecar_3d.is_none(),
                allow_2d_analysis,
            },
            &mem,
            &access_state,
        );
        ui.data_mut(|d| d.insert_temp(plot_id, access_state));
        // --- END ACCESSIBILITY LOGIC ---
        mem.store(ui.ctx(), plot_id);

        for event in ui.input(|i| i.raw.events.clone()) {
            if let egui::Event::Screenshot { image, .. } = event
                && let Ok(png_bytes) = export::color32_image_to_png(&image)
            {
                export::save_file_dialog("plot.png", &png_bytes, "PNG Image", &["png"]);
            }
        }

        ui.advance_cursor_after_rect(complete_rect);

        PlotResponse {
            inner,
            response,
            transform,
            hovered_plot_item,
        }
    }

    fn load_frame_access_state(ui: &mut Ui, plot_id: Id, mem: &PlotMemory) -> AccessState {
        let mut access_state = ui
            .data_mut(|d| d.get_temp::<AccessState>(plot_id))
            .unwrap_or_else(|| AccessState {
                show_settings_modal: mem.show_settings_modal,
                log_x: mem.log_x,
                log_y: mem.log_y,
                grid_density: mem.grid_density,
                show_x_labels: mem.show_x_labels,
                show_y_labels: mem.show_y_labels,
                show_hover: mem.show_hover,
                clamp_grid: mem.clamp_grid,
                ..Default::default()
            });
        access_state.log_x = mem.log_x;
        access_state.log_y = mem.log_y;
        access_state.show_x_labels = mem.show_x_labels;
        access_state.show_y_labels = mem.show_y_labels;
        access_state.show_hover = mem.show_hover;
        access_state
    }

    fn register_and_handle_viewport_keys(
        ui: &mut Ui,
        response: &Response,
        mem: &mut PlotMemory,
        cfg: &ViewportKeyboardConfig<'_>,
    ) {
        let zoom_axes = cfg.allow_zoom.and(ui.is_enabled());
        let pan_axes = cfg.allow_pan.and(ui.is_enabled());
        let reset_enabled = ui.is_enabled();
        register_plot_viewport_commands(ui.ctx(), zoom_axes.any(), pan_axes, reset_enabled);

        if !response.has_focus()
            || ui.input(|i| i.modifiers.ctrl || i.modifiers.command || i.modifiers.alt)
        {
            return;
        }

        if pan_axes.any() && try_pan_focused_plot(ui, mem, cfg, pan_axes) {
            return;
        }

        let zoom_cfg = FloatingZoomConfig {
            plot_id: cfg.plot_id,
            allow_zoom: cfg.allow_zoom,
            data_aspect: cfg.data_aspect,
            linked_axes: cfg.linked_axes,
            auto_content_bounds: cfg.auto_content_bounds,
            default_auto_bounds: cfg.default_auto_bounds,
            show_settings_button: true,
        };
        if zoom_axes.any() {
            let plus = check_plot_key(ui, "Zoom In", "Zoom into the plot canvas", egui::Key::Plus);
            let eq = check_plot_key(
                ui,
                "Zoom In",
                "Zoom into the plot canvas",
                egui::Key::Equals,
            );
            if plus || eq {
                apply_zoom_step(ui, mem, &zoom_cfg, zoom_axes, 1.25);
                emit_plot_live_message(ui.ctx(), "Plot zoomed in".to_owned(), false);
            }
            if check_plot_key(
                ui,
                "Zoom Out",
                "Zoom out from the plot canvas",
                egui::Key::Minus,
            ) {
                apply_zoom_step(ui, mem, &zoom_cfg, zoom_axes, 0.8);
                emit_plot_live_message(ui.ctx(), "Plot zoomed out".to_owned(), false);
            }
        }
        if reset_enabled
            && check_plot_key(
                ui,
                "Reset Bounds",
                "Restore default plot bounds to fit all data",
                egui::Key::Home,
            )
        {
            apply_bounds_reset(ui, mem, &zoom_cfg);
            emit_plot_live_message(ui.ctx(), "Plot bounds reset".to_owned(), false);
        }
    }

    fn render_floating_zoom_toolbar(
        ui: &mut Ui,
        mem: &mut PlotMemory,
        config: &FloatingZoomConfig<'_>,
    ) {
        let frame = *mem.transform.frame();
        if frame.width() < 30.0 || frame.height() < 30.0 {
            return;
        }

        let btn_size = egui::vec2(24.0, 24.0);
        let top_offset = if config.show_settings_button {
            34.0
        } else {
            6.0
        };
        let nav_rect = Rect::from_min_size(
            frame.right_top() + egui::vec2(-30.0, top_offset),
            egui::vec2(24.0, 80.0),
        );
        let mut zoom_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(nav_rect)
                .layout(egui::Layout::top_down(egui::Align::Center)),
        );
        zoom_ui.set_clip_rect(frame.intersect(ui.clip_rect()));
        zoom_ui.spacing_mut().item_spacing = egui::vec2(0.0, 4.0);

        let effective_allow_zoom = config.allow_zoom.and(ui.is_enabled());
        let zoom_enabled = effective_allow_zoom.any();
        let reset_enabled = ui.is_enabled();

        let (mut zoom_in, mut zoom_out, mut reset) =
            render_zoom_toolbar_buttons(&mut zoom_ui, btn_size, zoom_enabled, reset_enabled);
        let (sig_in, sig_out, sig_reset) = drain_headless_zoom_signals(ui.ctx(), config.plot_id);
        zoom_in |= sig_in;
        zoom_out |= sig_out;
        reset |= sig_reset;

        if zoom_in && zoom_enabled {
            apply_zoom_step(ui, mem, config, effective_allow_zoom, 1.25);
        }
        if zoom_out && zoom_enabled {
            apply_zoom_step(ui, mem, config, effective_allow_zoom, 0.8);
        }
        if reset && reset_enabled {
            apply_bounds_reset(ui, mem, config);
        }
    }
}

#[derive(Clone)]
struct FocusedPlotPoint {
    item_id: Id,
    series_name: String,
    point: PlotPoint,
    world_point: Option<[f64; 3]>,
    color: Color32,
}

struct ViewportKeyboardConfig<'a> {
    plot_id: Id,
    allow_zoom: Vec2b,
    allow_pan: Vec2b,
    data_aspect: Option<f32>,
    linked_axes: &'a Option<(Id, Vec2b)>,
    auto_content_bounds: PlotBounds,
    default_auto_bounds: Vec2b,
}

struct FloatingZoomConfig<'a> {
    plot_id: Id,
    allow_zoom: Vec2b,
    data_aspect: Option<f32>,
    linked_axes: &'a Option<(Id, Vec2b)>,
    auto_content_bounds: PlotBounds,
    default_auto_bounds: Vec2b,
    show_settings_button: bool,
}

fn right_controls_rect(
    plot_rect: Rect,
    show_settings_button: bool,
    show_navigation_controls: bool,
) -> Option<Rect> {
    if plot_rect.width() < 30.0 || plot_rect.height() < 30.0 {
        return None;
    }
    match (show_settings_button, show_navigation_controls) {
        (false, false) => None,
        (true, false) => Some(Rect::from_min_size(
            plot_rect.right_top() + vec2(-30.0, 6.0),
            vec2(24.0, 24.0),
        )),
        (false, true) => Some(Rect::from_min_size(
            plot_rect.right_top() + vec2(-30.0, 6.0),
            vec2(24.0, 80.0),
        )),
        (true, true) => Some(Rect::from_min_size(
            plot_rect.right_top() + vec2(-30.0, 6.0),
            vec2(24.0, 108.0),
        )),
    }
}

fn legend_safe_rect(plot_rect: Rect, corner: Corner, right_controls: Option<Rect>) -> Rect {
    let mut safe = plot_rect;
    let Some(ctrl) = right_controls else {
        return safe;
    };
    let left_of_ctrl = (ctrl.min.x - 4.0).max(safe.min.x + 16.0);
    match corner {
        Corner::RightTop | Corner::LeftTop | Corner::LeftBottom => {
            safe.max.x = safe.max.x.min(left_of_ctrl);
        }
        Corner::RightBottom => {
            if safe.height() > ctrl.height() + 24.0 {
                safe.min.y = safe.min.y.max(ctrl.max.y + 4.0);
            } else {
                safe.max.x = safe.max.x.min(left_of_ctrl);
            }
        }
    }
    safe
}

fn render_zoom_toolbar_buttons(
    zoom_ui: &mut Ui,
    btn_size: Vec2,
    zoom_enabled: bool,
    reset_enabled: bool,
) -> (bool, bool, bool) {
    let in_btn = egui::Button::new(egui::RichText::new("+").size(14.0).strong())
        .min_size(btn_size)
        .corner_radius(4.0);
    let zoom_in = a11y::focusable_action_button(zoom_ui, zoom_enabled, in_btn, Some("Zoom in"))
        .accessible_tooltip("Zoom in (Scroll up / +)\nZooms into the plot canvas")
        .clicked()
        && zoom_enabled;

    let out_btn = egui::Button::new(egui::RichText::new("-").size(14.0).strong())
        .min_size(btn_size)
        .corner_radius(4.0);
    let zoom_out = a11y::focusable_action_button(zoom_ui, zoom_enabled, out_btn, Some("Zoom out"))
        .accessible_tooltip("Zoom out (Scroll down / -)\nZooms out from the plot canvas")
        .clicked()
        && zoom_enabled;

    let reset_btn = egui::Button::new(egui::RichText::new("🏠").size(12.0))
        .min_size(btn_size)
        .corner_radius(4.0);
    let reset =
        a11y::focusable_action_button(zoom_ui, reset_enabled, reset_btn, Some("Reset plot bounds"))
            .accessible_tooltip(
                "Reset bounds (Home / Double-click)\nRestores default plot bounds to fit all data",
            )
            .clicked()
            && reset_enabled;
    (zoom_in, zoom_out, reset)
}

fn drain_headless_zoom_signals(ctx: &egui::Context, plot_id: Id) -> (bool, bool, bool) {
    ctx.data_mut(|d| {
        let take = |d: &mut egui::util::IdTypeMap, suffix: &'static str| -> bool {
            let key = plot_id.with(suffix);
            let val = d.get_temp::<bool>(key).unwrap_or(false);
            if val {
                d.insert_temp(key, false);
            }
            val
        };
        (
            take(d, "zoom_in"),
            take(d, "zoom_out"),
            take(d, "reset_view"),
        )
    })
}

pub(crate) fn apply_aspect_and_sync_linked(
    ui: &mut Ui,
    mem: &mut PlotMemory,
    data_aspect: Option<f32>,
    linked_axes: &Option<(Id, Vec2b)>,
    default_auto_bounds: Vec2b,
) {
    if let Some(aspect) = data_aspect {
        if let Some((_, linked)) = linked_axes {
            let change_x = linked.y && !linked.x;
            mem.transform.set_aspect_by_changing_axis(
                f64::from(aspect),
                if change_x { Axis::X } else { Axis::Y },
            );
        } else if default_auto_bounds.any() {
            mem.transform.set_aspect_by_expanding(f64::from(aspect));
        } else {
            mem.transform
                .set_aspect_by_changing_axis(f64::from(aspect), Axis::Y);
        }
    }
    if let Some((id, _)) = linked_axes {
        ui.data_mut(|data| {
            let link_groups: &mut BoundsLinkGroups = data.get_temp_mut_or_default(Id::NULL);
            let entry = link_groups.0.entry(*id).or_default();
            entry.bounds = *mem.transform.bounds();
            entry.auto_bounds = mem.auto_bounds;
        });
    }
    ui.ctx().request_repaint();
}

fn apply_zoom_step(
    ui: &mut Ui,
    mem: &mut PlotMemory,
    config: &FloatingZoomConfig<'_>,
    effective_allow_zoom: Vec2b,
    scalar: f32,
) {
    let mut factor = egui::Vec2::splat(scalar);
    if !effective_allow_zoom.x {
        factor.x = 1.0;
    }
    if !effective_allow_zoom.y {
        factor.y = 1.0;
    }
    let center = mem.transform.frame().center();
    mem.transform.zoom(factor, center);
    mem.auto_bounds = mem.auto_bounds.and(!config.allow_zoom);
    apply_aspect_and_sync_linked(
        ui,
        mem,
        config.data_aspect,
        config.linked_axes,
        config.default_auto_bounds,
    );
}

fn apply_bounds_reset(ui: &mut Ui, mem: &mut PlotMemory, config: &FloatingZoomConfig<'_>) {
    mem.auto_bounds = true.into();
    mem.transform.set_bounds(config.auto_content_bounds);
    apply_aspect_and_sync_linked(
        ui,
        mem,
        config.data_aspect,
        config.linked_axes,
        config.default_auto_bounds,
    );
}

fn register_plot_viewport_commands(
    ctx: &egui::Context,
    zoom_enabled: bool,
    pan_axes: Vec2b,
    reset_enabled: bool,
) {
    use crate::commands::{CommandRegistryData, CommandTrigger};
    if zoom_enabled {
        for key in [egui::Key::Plus, egui::Key::Equals] {
            CommandRegistryData::register(
                ctx,
                "Zoom In",
                "Zoom into the plot canvas",
                CommandTrigger::Key(key),
                false,
                "Plot Viewport",
            );
        }
        CommandRegistryData::register(
            ctx,
            "Zoom Out",
            "Zoom out from the plot canvas",
            CommandTrigger::Key(egui::Key::Minus),
            false,
            "Plot Viewport",
        );
    }
    if reset_enabled {
        CommandRegistryData::register(
            ctx,
            "Reset Bounds",
            "Restore default plot bounds to fit all data",
            CommandTrigger::Key(egui::Key::Home),
            false,
            "Plot Viewport",
        );
    }
    register_plot_pan_commands(ctx, pan_axes);
}

fn register_plot_pan_commands(ctx: &egui::Context, pan_axes: Vec2b) {
    use crate::commands::{CommandRegistryData, CommandTrigger};
    if pan_axes.x {
        for (name, desc, key) in [
            ("Pan Left", "Pan plot viewport left", egui::Key::ArrowLeft),
            (
                "Pan Right",
                "Pan plot viewport right",
                egui::Key::ArrowRight,
            ),
        ] {
            CommandRegistryData::register(
                ctx,
                name,
                desc,
                CommandTrigger::Shortcut(egui::Modifiers::SHIFT, key),
                false,
                "Plot Viewport",
            );
        }
    }
    if pan_axes.y {
        for (name, desc, key) in [
            ("Pan Up", "Pan plot viewport up", egui::Key::ArrowUp),
            ("Pan Down", "Pan plot viewport down", egui::Key::ArrowDown),
        ] {
            CommandRegistryData::register(
                ctx,
                name,
                desc,
                CommandTrigger::Shortcut(egui::Modifiers::SHIFT, key),
                false,
                "Plot Viewport",
            );
        }
    }
}

fn check_plot_key(ui: &Ui, name: &str, desc: &str, key: egui::Key) -> bool {
    crate::commands::CommandRegistryData::register_and_check(
        ui.ctx(),
        name,
        desc,
        crate::commands::CommandTrigger::Key(key),
        false,
        "Plot Viewport",
        Some(ui),
        None,
    )
}

fn try_pan_focused_plot(
    ui: &mut Ui,
    mem: &mut PlotMemory,
    cfg: &ViewportKeyboardConfig<'_>,
    pan_axes: Vec2b,
) -> bool {
    use crate::commands::{CommandRegistryData, CommandTrigger};
    let frame = *mem.transform.frame();
    let step_x = f64::from(frame.width().max(10.0) * 0.1);
    let step_y = f64::from(frame.height().max(10.0) * 0.1);
    let mut delta = (0.0_f64, 0.0_f64);
    let mut direction_label = "";

    let candidates = [
        (
            pan_axes.x,
            "Pan Left",
            "Pan plot viewport left",
            egui::Key::ArrowLeft,
            (-step_x, 0.0),
            "left",
        ),
        (
            pan_axes.x,
            "Pan Right",
            "Pan plot viewport right",
            egui::Key::ArrowRight,
            (step_x, 0.0),
            "right",
        ),
        (
            pan_axes.y,
            "Pan Up",
            "Pan plot viewport up",
            egui::Key::ArrowUp,
            (0.0, -step_y),
            "up",
        ),
        (
            pan_axes.y,
            "Pan Down",
            "Pan plot viewport down",
            egui::Key::ArrowDown,
            (0.0, step_y),
            "down",
        ),
    ];
    for (allowed, name, desc, key, step, dir) in candidates {
        if allowed
            && CommandRegistryData::register_and_check(
                ui.ctx(),
                name,
                desc,
                CommandTrigger::Shortcut(egui::Modifiers::SHIFT, key),
                false,
                "Plot Viewport",
                Some(ui),
                None,
            )
        {
            delta.0 += step.0;
            delta.1 += step.1;
            direction_label = dir;
        }
    }
    if delta == (0.0, 0.0) {
        return false;
    }
    mem.transform.translate_bounds(delta);
    mem.auto_bounds = mem.auto_bounds.and(!pan_axes);
    apply_aspect_and_sync_linked(
        ui,
        mem,
        cfg.data_aspect,
        cfg.linked_axes,
        cfg.default_auto_bounds,
    );
    emit_plot_live_message(ui.ctx(), format!("Plot panned {direction_label}"), false);
    true
}

fn hash_plot_items(items: &[Box<dyn PlotItem + '_>]) -> u64 {
    use std::hash::{Hash as _, Hasher as _};
    let mut hasher = ahash::AHasher::default();
    items.len().hash(&mut hasher);
    for item in items {
        item.item_content_hash().hash(&mut hasher);
    }
    hasher.finish()
}

fn hash_visible_plot_items(
    items: &[Box<dyn PlotItem + '_>],
    hidden_items: &ahash::HashSet<Id>,
) -> u64 {
    use std::hash::{Hash as _, Hasher as _};
    let mut hasher = ahash::AHasher::default();
    let visible_count = items
        .iter()
        .filter(|it| !it.is_companion_visual_only() && !hidden_items.contains(&it.id()))
        .count();
    visible_count.hash(&mut hasher);
    for item in items {
        if !item.is_companion_visual_only() && !hidden_items.contains(&item.id()) {
            item.item_content_hash().hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn item_point_count(item: &dyn PlotItem) -> usize {
    if let Some(segs) = item.line_segments() {
        return segs.iter().map(Vec::len).sum();
    }
    match item.geometry() {
        PlotGeometry::Points(pts) => pts.len(),
        PlotGeometry::None | PlotGeometry::Rects => item.extract_points().len(),
    }
}

type BaseDatasetList = Vec<fit_overlay::BaseDatasetTuple>;
type BaseDatasetCache = Option<(u64, Arc<BaseDatasetList>)>;

fn get_or_populate_visible_base_datasets<'c>(
    items: &[Box<dyn PlotItem + '_>],
    hidden_items: &ahash::HashSet<Id>,
    base_items_hash: u64,
    cache: &'c mut BaseDatasetCache,
) -> &'c Arc<BaseDatasetList> {
    let needs_refresh = cache
        .as_ref()
        .is_none_or(|(cached_hash, _)| *cached_hash != base_items_hash);
    if needs_refresh {
        let built: BaseDatasetList = items
            .iter()
            .filter(|it| !it.is_companion_visual_only() && !hidden_items.contains(&it.id()))
            .map(|it| (it.name().to_owned(), it.extract_points(), it.color()))
            .collect();
        *cache = Some((base_items_hash, Arc::new(built)));
    }
    &cache.as_ref().expect("populated base datasets").1
}

pub(crate) fn get_or_populate_base_datasets<'c>(
    items: &[Box<dyn PlotItem + '_>],
    base_items_hash: u64,
    cache: &'c mut BaseDatasetCache,
) -> &'c Arc<BaseDatasetList> {
    let needs_refresh = cache
        .as_ref()
        .is_none_or(|(cached_hash, _)| *cached_hash != base_items_hash);
    if needs_refresh {
        let built: BaseDatasetList = items
            .iter()
            .filter(|it| !it.is_companion_visual_only())
            .map(|it| (it.name().to_owned(), it.extract_points(), it.color()))
            .collect();
        *cache = Some((base_items_hash, Arc::new(built)));
    }
    &cache.as_ref().expect("populated base datasets").1
}

fn collect_accessible_datasets(items: &[Box<dyn PlotItem + '_>]) -> BaseDatasetList {
    items
        .iter()
        .filter(|item| !item.is_companion_visual_only())
        .map(|item| (item.name().to_owned(), item.extract_points(), item.color()))
        .collect()
}

pub(crate) fn collect_export_datasets(
    items: &[Box<dyn PlotItem + '_>],
) -> Vec<(String, Vec<[f64; 2]>)> {
    items
        .iter()
        .filter(|item| !item.is_companion_visual_only())
        .filter_map(|item| {
            let pts = item.extract_points();
            (!pts.is_empty()).then(|| (item.name().to_owned(), pts))
        })
        .collect()
}

fn register_plot_dataset_commands_from_items(
    ctx: &egui::Context,
    items: &[Box<dyn PlotItem + '_>],
    sidecar_3d: Option<&Plot3DSeriesSidecar>,
    focused_item: usize,
) {
    use crate::commands::{CommandRegistryData, CommandTrigger};
    let has_series = sidecar_3d.filter(|s| !s.series().is_empty()).is_some() || !items.is_empty();
    if !has_series {
        return;
    }
    CommandRegistryData::register(
        ctx,
        "Next Dataset",
        "Navigate to the next dataset",
        CommandTrigger::Key(egui::Key::ArrowDown),
        false,
        "Plot Accessibility",
    );
    CommandRegistryData::register(
        ctx,
        "Previous Dataset",
        "Navigate to the previous dataset",
        CommandTrigger::Key(egui::Key::ArrowUp),
        false,
        "Plot Accessibility",
    );
    let focused_has_points = if let Some(sidecar) = sidecar_3d.filter(|s| !s.series().is_empty()) {
        sidecar
            .series()
            .get(focused_item)
            .or_else(|| sidecar.series().first())
            .is_some_and(|s| !s.projected.is_empty())
    } else {
        items
            .get(focused_item)
            .or_else(|| items.first())
            .is_some_and(|it| it.has_any_points())
    };
    if focused_has_points {
        CommandRegistryData::register(
            ctx,
            "Next Point",
            "Navigate to the next point in the dataset",
            CommandTrigger::Key(egui::Key::ArrowRight),
            false,
            "Plot Accessibility",
        );
        CommandRegistryData::register(
            ctx,
            "Previous Point",
            "Navigate to the previous point in the dataset",
            CommandTrigger::Key(egui::Key::ArrowLeft),
            false,
            "Plot Accessibility",
        );
    }
}

fn handle_focused_plot_dataset_keys(
    ui: &Ui,
    accessible_datasets: &[(String, Vec<[f64; 2]>, Color32)],
    sidecar_3d: Option<&Plot3DSeriesSidecar>,
    access_state: &mut AccessState,
) {
    use crate::commands::{CommandRegistryData, CommandTrigger};
    let num_items = accessible_datasets.len();
    if num_items == 0
        || ui.input(|i| {
            i.modifiers.shift || i.modifiers.alt || i.modifiers.ctrl || i.modifiers.command
        })
    {
        return;
    }

    let mut changed = false;
    if CommandRegistryData::register_and_check(
        ui.ctx(),
        "Next Dataset",
        "Navigate to the next dataset",
        CommandTrigger::Key(egui::Key::ArrowDown),
        false,
        "Plot Accessibility",
        Some(ui),
        None,
    ) && access_state.focused_item + 1 < num_items
    {
        access_state.focused_item += 1;
        access_state.focused_point = 0;
        changed = true;
    }
    if CommandRegistryData::register_and_check(
        ui.ctx(),
        "Previous Dataset",
        "Navigate to the previous dataset",
        CommandTrigger::Key(egui::Key::ArrowUp),
        false,
        "Plot Accessibility",
        Some(ui),
        None,
    ) && access_state.focused_item > 0
    {
        access_state.focused_item -= 1;
        access_state.focused_point = 0;
        changed = true;
    }
    if let Some((_, pts, _)) = accessible_datasets.get(access_state.focused_item)
        && !pts.is_empty()
    {
        changed |= step_focused_point_index(ui, pts.len(), &mut access_state.focused_point);
    }
    if changed {
        access_state.focused_poi = None;
        announce_focused_dataset_point(
            ui.ctx(),
            accessible_datasets,
            sidecar_3d,
            access_state.focused_item,
            access_state.focused_point,
        );
    }
}

fn announce_focused_dataset_point(
    ctx: &egui::Context,
    accessible_datasets: &[(String, Vec<[f64; 2]>, Color32)],
    sidecar_3d: Option<&Plot3DSeriesSidecar>,
    focused_item: usize,
    focused_point: usize,
) {
    if let Some((name, w)) = sidecar_3d
        .and_then(|s| s.series().get(focused_item))
        .and_then(|s| s.world.get(focused_point).map(|w| (&s.name, *w)))
    {
        let x_str = format_world_coord(w[0]);
        let y_str = format_world_coord(w[1]);
        let z_str = format_world_coord(w[2]);
        emit_plot_live_message(
            ctx,
            format!("Dataset {name}, Point X: {x_str}, Y: {y_str}, Z: {z_str}"),
            false,
        );
    } else if let Some((name, pts, _)) = accessible_datasets.get(focused_item)
        && let Some(p) = pts.get(focused_point)
    {
        let x_str = format::format_readout_value(p[0], 4, 4);
        let y_str = format::format_readout_value(p[1], 4, 4);
        emit_plot_live_message(
            ctx,
            format!("Dataset {name}, Point X: {x_str}, Y: {y_str}"),
            false,
        );
    }
}

fn step_focused_point_index(ui: &Ui, num_pts: usize, focused_point: &mut usize) -> bool {
    use crate::commands::{CommandRegistryData, CommandTrigger};
    let mut changed = false;
    if CommandRegistryData::register_and_check(
        ui.ctx(),
        "Next Point",
        "Navigate to the next point in the dataset",
        CommandTrigger::Key(egui::Key::ArrowRight),
        false,
        "Plot Accessibility",
        Some(ui),
        None,
    ) && *focused_point + 1 < num_pts
    {
        *focused_point += 1;
        changed = true;
    }
    if CommandRegistryData::register_and_check(
        ui.ctx(),
        "Previous Point",
        "Navigate to the previous point in the dataset",
        CommandTrigger::Key(egui::Key::ArrowLeft),
        false,
        "Plot Accessibility",
        Some(ui),
        None,
    ) && *focused_point > 0
    {
        *focused_point -= 1;
        changed = true;
    }
    changed
}

fn resolve_focused_plot_point<'a>(
    items: &[Box<dyn PlotItem + 'a>],
    accessible_datasets: &[(String, Vec<[f64; 2]>, Color32)],
    sidecar_3d: Option<&Plot3DSeriesSidecar>,
    access_state: &mut AccessState,
) -> Option<FocusedPlotPoint> {
    if accessible_datasets.is_empty() {
        return None;
    }
    if access_state.focused_item >= accessible_datasets.len() {
        access_state.focused_item = 0;
    }
    let (_, pts, _) = accessible_datasets.get(access_state.focused_item)?;
    if pts.is_empty() {
        let fallback_idx = accessible_datasets
            .iter()
            .position(|(_, p, _)| !p.is_empty())?;
        access_state.focused_item = fallback_idx;
        access_state.focused_point = 0;
    }
    let (name, pts, color) = accessible_datasets.get(access_state.focused_item)?;
    if access_state.focused_point >= pts.len() {
        access_state.focused_point = pts.len().saturating_sub(1);
    }
    let pt = *pts.get(access_state.focused_point)?;
    let world_point = sidecar_3d
        .and_then(|s| s.series().get(access_state.focused_item))
        .and_then(|s| s.world.get(access_state.focused_point).copied());
    let item_id = items
        .get(access_state.focused_item)
        .map_or_else(|| Id::new(name.as_str()), |it| it.id());
    Some(FocusedPlotPoint {
        item_id,
        series_name: name.clone(),
        point: PlotPoint::new(pt[0], pt[1]),
        world_point,
        color: *color,
    })
}

/// Returns the rect left after adding axes.
fn axis_widgets<'a>(
    mem: Option<&PlotMemory>,
    show_axes: impl Into<Vec2b>,
    complete_rect: Rect,
    [x_axes, y_axes]: [&'a [AxisHints<'a>]; 2],
) -> ([Vec<AxisWidget<'a>>; 2], Rect) {
    // Next we want to create this layout.
    // Indices are only examples.
    //
    //  left                     right
    //  +---+---------x----------+   +
    //  |   |      X-axis 3      |
    //  |   +--------------------+    top
    //  |   |      X-axis 2      |
    //  +-+-+--------------------+-+-+
    //  |y|y|                    |y|y|
    //  |-|-|                    |-|-|
    //  |A|A|                    |A|A|
    // y|x|x|    Plot Window     |x|x|
    //  |i|i|                    |i|i|
    //  |s|s|                    |s|s|
    //  |1|0|                    |2|3|
    //  +-+-+--------------------+-+-+
    //      |      X-axis 0      |   |
    //      +--------------------+   | bottom
    //      |      X-axis 1      |   |
    //  +   +--------------------+---+
    //
    let show_axes = show_axes.into();

    let mut x_axis_widgets = Vec::<AxisWidget<'_>>::new();
    let mut y_axis_widgets = Vec::<AxisWidget<'_>>::new();

    // Will shrink as we add more axes.
    let mut rect_left = complete_rect;
    let show_x_tick_labels = mem.is_none_or(|m| m.show_x_labels);
    let show_y_tick_labels = mem.is_none_or(|m| m.show_y_labels);

    if show_axes.x {
        // We will fix this later, once we know how much space the y axes take up.
        let initial_x_range = complete_rect.x_range();

        for (i, cfg) in x_axes.iter().enumerate().rev() {
            let mut height = if show_x_tick_labels || !cfg.label.is_empty() {
                let mut h = cfg.min_thickness;
                if let Some(mem) = mem {
                    h = h.max(mem.x_axis_thickness.get(&i).copied().unwrap_or_default());
                }
                h
            } else {
                0.0
            };
            height = height.min((rect_left.height() - 1.0).max(0.0));

            let rect = match VPlacement::from(cfg.placement) {
                VPlacement::Bottom => {
                    let bottom = rect_left.bottom();
                    *rect_left.bottom_mut() -= height;
                    let top = rect_left.bottom();
                    Rect::from_x_y_ranges(initial_x_range, top..=bottom)
                }
                VPlacement::Top => {
                    let top = rect_left.top();
                    *rect_left.top_mut() += height;
                    let bottom = rect_left.top();
                    Rect::from_x_y_ranges(initial_x_range, top..=bottom)
                }
            };
            let mut widget = AxisWidget::new(cfg.clone(), rect);
            widget.show_tick_labels = show_x_tick_labels;
            x_axis_widgets.push(widget);
        }
    }
    if show_axes.y {
        // We know this, since we've already allocated space for the x axes.
        let plot_y_range = rect_left.y_range();

        for (i, cfg) in y_axes.iter().enumerate().rev() {
            let mut width = if show_y_tick_labels || !cfg.label.is_empty() {
                let mut w = cfg.min_thickness;
                if let Some(mem) = mem {
                    w = w.max(mem.y_axis_thickness.get(&i).copied().unwrap_or_default());
                }
                w
            } else {
                0.0
            };
            width = width.min((rect_left.width() - 1.0).max(0.0));

            let rect = match HPlacement::from(cfg.placement) {
                HPlacement::Left => {
                    let left = rect_left.left();
                    *rect_left.left_mut() += width;
                    let right = rect_left.left();
                    Rect::from_x_y_ranges(left..=right, plot_y_range)
                }
                HPlacement::Right => {
                    let right = rect_left.right();
                    *rect_left.right_mut() -= width;
                    let left = rect_left.right();
                    Rect::from_x_y_ranges(left..=right, plot_y_range)
                }
            };
            let mut widget = AxisWidget::new(cfg.clone(), rect);
            widget.show_tick_labels = show_y_tick_labels;
            y_axis_widgets.push(widget);
        }
    }

    // The loops iterated through {x,y}_axes in reverse order, so we have to reverse the
    // {x,y}_axis_widgets vec as well. Otherwise, the indices are messed up and the plot memory
    // (mem.{x,y}_axis_thickness) will access the wrong axis given an index.
    x_axis_widgets.reverse();
    y_axis_widgets.reverse();

    let mut plot_rect = rect_left;

    // If too little space, remove axis widgets
    if plot_rect.width() <= 0.0 || plot_rect.height() <= 0.0 {
        y_axis_widgets.clear();
        x_axis_widgets.clear();
        plot_rect = complete_rect;
    }

    // Now that we know the final x_range of the plot_rect,
    // assign it to the x_axis_widgets (they are currently too wide):
    for widget in &mut x_axis_widgets {
        widget.rect = Rect::from_x_y_ranges(plot_rect.x_range(), widget.rect.y_range());
    }

    ([x_axis_widgets, y_axis_widgets], plot_rect)
}

/// User-requested modifications to the plot bounds. We collect them in the plot build function to later apply
/// them at the right time, as other modifications need to happen first.
enum BoundsModification {
    SetX(RangeInclusive<f64>),
    SetY(RangeInclusive<f64>),
    Translate(Vec2),
    AutoBounds(Vec2b),
    Zoom(Vec2, PlotPoint),
}

// ----------------------------------------------------------------------------
// Grid

/// Input for "grid spacer" functions.
///
/// See [`Plot::x_grid_spacer()`] and [`Plot::y_grid_spacer()`].
pub struct GridInput {
    /// Min/max of the visible data range (the values at the two edges of the plot,
    /// for the current axis).
    pub bounds: (f64, f64),

    /// Recommended (but not required) lower-bound on the step size returned by custom grid spacers.
    ///
    /// Computed as the ratio between the diagram's bounds (in plot coordinates) and the viewport
    /// (in frame/window coordinates), scaled up to represent the minimal possible step.
    ///
    /// Always positive.
    pub base_step_size: f64,
}

/// One mark (horizontal or vertical line) in the background grid of a plot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridMark {
    /// X or Y value in the plot.
    pub value: f64,

    /// The (approximate) distance to the next value of same thickness.
    ///
    /// Determines how thick the grid line is painted. It's not important that `step_size`
    /// matches the difference between two `value`s precisely, but rather that grid marks of
    /// same thickness have same `step_size`. For example, months can have a different number
    /// of days, but consistently using a `step_size` of 30 days is a valid approximation.
    pub step_size: f64,
}

/// Recursively splits the grid into `base` subdivisions (e.g. 100, 10, 1).
///
/// The logarithmic base, expressing how many times each grid unit is subdivided.
/// 10 is a typical value, others are possible though.
pub fn log_grid_spacer(log_base: i64) -> GridSpacer<'static> {
    let log_base = log_base as f64;
    let step_sizes = move |input: GridInput| -> Vec<GridMark> {
        // handle degenerate cases
        if input.base_step_size.abs() < f64::EPSILON {
            return Vec::new();
        }

        // The distance between two of the thinnest grid lines is "rounded" up
        // to the next-bigger power of base
        let smallest_visible_unit = next_power(input.base_step_size, log_base);

        let step_sizes = [
            smallest_visible_unit,
            smallest_visible_unit * log_base,
            smallest_visible_unit * log_base * log_base,
        ];

        generate_marks(step_sizes, input.bounds)
    };

    Box::new(step_sizes)
}

/// Grid spacer for logarithmic scale axes.
///
/// Handles non-positive bounds by clamping minimum bound to positive values.
pub fn logarithmic_grid_spacer() -> GridSpacer<'static> {
    let step_sizes = move |input: GridInput| -> Vec<GridMark> {
        let (min_b, max_b) = input.bounds;
        // Edge case handling: fallback non-positive data bounds to positive minimum bounds
        let min_pos = if min_b <= 0.0 { 1e-6 } else { min_b };
        let max_pos = if max_b <= min_pos {
            min_pos + 10.0
        } else {
            max_b
        };

        let min_log = min_pos.log10().floor() as i32;
        let max_log = max_pos.log10().ceil() as i32;

        let mut marks = Vec::new();
        let start = min_log.clamp(-300, 300);
        let end = max_log.clamp(-300, 300);

        for exp in start..=end {
            let base_val = 10.0_f64.powi(exp);
            if base_val >= min_pos && base_val <= max_pos {
                marks.push(GridMark {
                    value: base_val,
                    step_size: base_val,
                });
            }
            for m in 2..10 {
                let sub_val = m as f64 * base_val;
                if sub_val >= min_pos && sub_val <= max_pos {
                    marks.push(GridMark {
                        value: sub_val,
                        step_size: base_val,
                    });
                }
            }
        }
        if marks.is_empty() && max_pos > min_pos {
            return log_grid_spacer(10)(GridInput {
                bounds: (min_pos, max_pos),
                base_step_size: (max_pos - min_pos) / 10.0,
            });
        }
        marks
    };

    Box::new(step_sizes)
}

fn collect_positive_axis_bounds(
    axis: usize,
    min_auto_bounds: &PlotBounds,
    items: &[Box<dyn PlotItem + '_>],
    hidden_items: &ahash::HashSet<Id>,
) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut consider = |v: f64| {
        if v.is_finite() && v > 0.0 {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    };
    consider(min_auto_bounds.min[axis]);
    consider(min_auto_bounds.max[axis]);
    for item in items.iter().filter(|it| !hidden_items.contains(&it.id())) {
        match item.geometry() {
            PlotGeometry::Points(pts) => {
                for pt in pts {
                    if pt.x.is_finite() && pt.y.is_finite() {
                        consider(if axis == 0 { pt.x } else { pt.y });
                    }
                }
            }
            PlotGeometry::None | PlotGeometry::Rects => {
                let ib = item.bounds();
                consider(ib.min[axis]);
                consider(ib.max[axis]);
                for pt in item.extract_points() {
                    if pt[0].is_finite() && pt[1].is_finite() {
                        consider(pt[axis]);
                    }
                }
            }
        }
    }
    (lo.is_finite() && hi.is_finite() && lo > 0.0 && hi >= lo).then_some((lo, hi))
}

fn apply_single_log_axis_bounds(
    bounds: &mut PlotBounds,
    axis: usize,
    is_auto: bool,
    positive_bounds: Option<(f64, f64)>,
    margin_frac: f64,
) {
    if is_auto {
        let (mut lo, mut hi) = positive_bounds.unwrap_or((1e-6, 10.0));
        if (hi - lo).abs() <= f64::EPSILON * lo.max(1.0) {
            let half_decade = 10.0_f64.sqrt();
            lo = (lo / half_decade).max(1e-300);
            hi = (hi * half_decade).min(1e300);
        }
        bounds.min[axis] = lo;
        bounds.max[axis] = hi;
        bounds.add_log_margin_axis(axis, margin_frac);
    } else {
        if !bounds.min[axis].is_finite() || bounds.min[axis] <= 0.0 {
            bounds.min[axis] = positive_bounds
                .map(|(lo, _)| lo)
                .filter(|&lo| !bounds.max[axis].is_finite() || lo < bounds.max[axis])
                .unwrap_or(1e-6);
        }
        if !bounds.max[axis].is_finite() || bounds.max[axis] <= bounds.min[axis] {
            bounds.max[axis] = positive_bounds
                .map(|(_, hi)| hi)
                .filter(|&hi| hi > bounds.min[axis])
                .unwrap_or_else(|| (bounds.min[axis] * 10.0).min(1e300));
        }
    }
}

#[expect(clippy::too_many_arguments)]
fn apply_plot_bounds_with_scale(
    bounds: &mut PlotBounds,
    min_auto_bounds: &PlotBounds,
    default_auto_bounds: Vec2b,
    auto_bounds: Vec2b,
    any_dynamic_modifications: bool,
    items: &[Box<dyn PlotItem + '_>],
    hidden_items: &ahash::HashSet<Id>,
    margin_fraction: Vec2,
    log_axes: Vec2b,
) {
    if !log_axes.x && ((!default_auto_bounds.x && !any_dynamic_modifications) || auto_bounds.x) {
        bounds.set_x(min_auto_bounds);
    }
    if !log_axes.y && ((!default_auto_bounds.y && !any_dynamic_modifications) || auto_bounds.y) {
        bounds.set_y(min_auto_bounds);
    }
    let auto_x = auto_bounds.x && (!min_auto_bounds.is_valid_x() || default_auto_bounds.x);
    let auto_y = auto_bounds.y && (!min_auto_bounds.is_valid_y() || default_auto_bounds.y);
    if (!log_axes.x && auto_x) || (!log_axes.y && auto_y) {
        for item in items.iter().filter(|it| !hidden_items.contains(&it.id())) {
            let item_bounds = item.bounds();
            if !log_axes.x && auto_x {
                bounds.merge_x(&item_bounds);
            }
            if !log_axes.y && auto_y {
                bounds.merge_y(&item_bounds);
            }
        }
        if !log_axes.x && auto_x {
            bounds.add_relative_margin_x(margin_fraction);
        }
        if !log_axes.y && auto_y {
            bounds.add_relative_margin_y(margin_fraction);
        }
    }
    if log_axes.x {
        let pos = collect_positive_axis_bounds(0, min_auto_bounds, items, hidden_items);
        let is_auto = auto_bounds.x || (!default_auto_bounds.x && !any_dynamic_modifications);
        apply_single_log_axis_bounds(bounds, 0, is_auto, pos, margin_fraction.x as f64);
    }
    if log_axes.y {
        let pos = collect_positive_axis_bounds(1, min_auto_bounds, items, hidden_items);
        let is_auto = auto_bounds.y || (!default_auto_bounds.y && !any_dynamic_modifications);
        apply_single_log_axis_bounds(bounds, 1, is_auto, pos, margin_fraction.y as f64);
    }
}

fn resolve_ambient_plot_links(
    ctx: &egui::Context,
    id_source: Id,
    show_axes: Vec2b,
    allow_drag: Vec2b,
    allow_zoom: Vec2b,
    allow_scroll: Vec2b,
    allow_boxed_zoom: bool,
    linked_axes: &mut Option<(Id, Vec2b)>,
    linked_cursors: &mut Option<(Id, Vec2b)>,
) {
    let is_interactive_2d = (show_axes.x || show_axes.y)
        && (allow_drag.any() || allow_zoom.any() || allow_scroll.any() || allow_boxed_zoom);
    if !is_interactive_2d {
        return;
    }
    let state = ctx.data(|d| {
        d.get_temp::<AmbientPlotLinkState>(Id::new("AMBIENT_PLOT_LINK_STATE"))
            .unwrap_or_default()
    });
    let axes_mask = Vec2b::new(show_axes.x, show_axes.y);
    if state.config.link_axes && linked_axes.is_none() {
        *linked_axes = Some((
            id_source.with(("ambient_axis_link", state.axis_epoch)),
            axes_mask,
        ));
    }
    if state.config.link_cursors && linked_cursors.is_none() {
        *linked_cursors = Some((id_source.with("ambient_cursor_link"), axes_mask));
    }
}

fn sync_linked_auto_bounds_envelope(
    ui: &mut Ui,
    link_id: Id,
    link_axes: Vec2b,
    plot_id: Id,
    local_auto: PlotBounds,
    auto_bounds: Vec2b,
    bounds: &mut PlotBounds,
) -> PlotBounds {
    let union_auto = ui.data_mut(|data| {
        let link_groups: &mut BoundsLinkGroups = data.get_temp_mut_or_default(Id::NULL);
        let entry = link_groups.0.entry(link_id).or_default();
        if let Some(idx) = entry.auto_frames.iter().position(|f| f.id == plot_id) {
            entry.auto_frames.drain(0..=idx);
        }
        entry.auto_frames.push(PlotFrameAutoBounds {
            id: plot_id,
            bounds: local_auto,
        });
        let mut merged = PlotBounds::NOTHING;
        for frame in &entry.auto_frames {
            merged.merge_x(&frame.bounds);
            merged.merge_y(&frame.bounds);
        }
        merged
    });
    if auto_bounds.x && link_axes.x && union_auto.is_valid_x() {
        bounds.set_x(&union_auto);
    }
    if auto_bounds.y && link_axes.y && union_auto.is_valid_y() {
        bounds.set_y(&union_auto);
    }
    union_auto
}

fn compute_auto_content_bounds(
    current_bounds: PlotBounds,
    min_auto_bounds: &PlotBounds,
    default_auto_bounds: Vec2b,
    items: &[Box<dyn PlotItem + '_>],
    hidden_items: &ahash::HashSet<Id>,
    margin_fraction: Vec2,
    log_axes: Vec2b,
) -> PlotBounds {
    let mut out = current_bounds;
    if !log_axes.x {
        if default_auto_bounds.x || min_auto_bounds.is_valid_x() {
            out.set_x(min_auto_bounds);
        }
        for item in items.iter().filter(|it| !hidden_items.contains(&it.id())) {
            out.merge_x(&item.bounds());
        }
        if !out.is_valid_x() {
            out.set_x(&PlotBounds::NOTHING);
        }
        out.add_relative_margin_x(margin_fraction);
    } else {
        let pos = collect_positive_axis_bounds(0, min_auto_bounds, items, hidden_items);
        apply_single_log_axis_bounds(&mut out, 0, true, pos, margin_fraction.x as f64);
    }
    if !log_axes.y {
        if default_auto_bounds.y || min_auto_bounds.is_valid_y() {
            out.set_y(min_auto_bounds);
        }
        for item in items.iter().filter(|it| !hidden_items.contains(&it.id())) {
            out.merge_y(&item.bounds());
        }
        if !out.is_valid_y() {
            out.set_y(&PlotBounds::NOTHING);
        }
        out.add_relative_margin_y(margin_fraction);
    } else {
        let pos = collect_positive_axis_bounds(1, min_auto_bounds, items, hidden_items);
        apply_single_log_axis_bounds(&mut out, 1, true, pos, margin_fraction.y as f64);
    }
    out
}

/// Splits the grid into uniform-sized spacings (e.g. 100, 25, 1).
///
/// This function should return 3 positive step sizes, designating where the lines in the grid are drawn.
/// Lines are thicker for larger step sizes. Ordering of returned value is irrelevant.
///
/// Why only 3 step sizes? Three is the number of different line thicknesses that egui typically uses in the grid.
/// Ideally, those 3 are not hardcoded values, but depend on the visible range (accessible through `GridInput`).
pub fn uniform_grid_spacer<'a>(spacer: impl Fn(GridInput) -> [f64; 3] + 'a) -> GridSpacer<'a> {
    let get_marks = move |input: GridInput| -> Vec<GridMark> {
        let bounds = input.bounds;
        let step_sizes = spacer(input);
        generate_marks(step_sizes, bounds)
    };

    Box::new(get_marks)
}

// ----------------------------------------------------------------------------

struct PreparedPlot<'a, 'b> {
    /// The response of the whole plot area
    plot_area_response: &'a Response,
    items: Vec<Box<dyn PlotItem + 'b>>,
    show_x: bool,
    show_y: bool,
    allow_2d_analysis: bool,
    label_formatter: LabelFormatter<'a>,
    coordinates_formatter: Option<(Corner, CoordinatesFormatter<'a>)>,
    // axis_formatters: [AxisFormatter; 2],
    transform: PlotTransform,
    show_grid: Vec2b,
    grid_spacing: Rangef,
    grid_spacers: [GridSpacer<'a>; 2],
    draw_cursor_x: bool,
    draw_cursor_y: bool,
    draw_cursors: Vec<Cursor>,
    cursor_color: Option<Color32>,
    cursor_snap_distance: Option<f32>,

    clamp_grid: bool,
    plot_id: Id,
    focused_point: Option<FocusedPlotPoint>,
    focused_poi: Option<usize>,
    persisted_tooltip: Option<a11y::PointTooltipState>,
    sidecar_3d: Option<&'a Plot3DSeriesSidecar>,
    critical_points: Arc<[poi::CriticalPoint]>,
    show_settings_button: bool,
    right_controls_rect: Option<Rect>,
}

struct PreparedUiOutput<'b> {
    items: Vec<Box<dyn PlotItem + 'b>>,
    cursors: Vec<Cursor>,
    hovered_plot_item: Option<Id>,
    hovered_poi_idx: Option<usize>,
}

impl<'b> PreparedPlot<'_, 'b> {
    fn ui(mut self, ui: &mut Ui, response: &Response) -> PreparedUiOutput<'b> {
        let mut axes_shapes = Vec::new();
        if self.show_grid.x {
            self.paint_grid(ui, &mut axes_shapes, Axis::X, self.grid_spacing);
        }
        if self.show_grid.y {
            self.paint_grid(ui, &mut axes_shapes, Axis::Y, self.grid_spacing);
        }
        axes_shapes.sort_by(|(_, s1), (_, s2)| s1.total_cmp(s2));
        let mut shapes: Vec<Shape> = axes_shapes.into_iter().map(|(shape, _)| shape).collect();

        let transform = &self.transform;
        let mut plot_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(*transform.frame())
                .layout(Layout::default()),
        );
        plot_ui.set_clip_rect(transform.frame().intersect(ui.clip_rect()));
        for item in &self.items {
            item.shapes(&plot_ui, transform, &mut shapes);
        }
        if self.allow_2d_analysis {
            poi_interaction::paint_critical_point_markers(
                ui,
                transform,
                &self.critical_points,
                &mut shapes,
            );
        }

        let hover_pos = response.hover_pos();
        let (cursors, hovered_item_id, hovered_point, hovered_poi_idx) =
            if !self.allow_2d_analysis && self.sidecar_3d.is_none() {
                if let Some(target) = self.focused_point.as_ref() {
                    let (c, id, pt) = self.render_focused_point(ui, target, &mut shapes);
                    (c, id, pt, None)
                } else {
                    (Vec::new(), None, None, None)
                }
            } else if let Some(state) = self.persisted_tooltip.as_ref() {
                let (c, id, pt) = self.render_persisted_tooltip(state);
                (c, id, pt, None)
            } else if let Some(pointer) = hover_pos {
                self.hover(ui, pointer, &mut shapes)
            } else if let Some(poi_idx) = self.focused_poi
                && let Some(poi) = self.critical_points.get(poi_idx)
            {
                let (c, pt) = poi_interaction::render_active_poi(
                    ui,
                    self.plot_area_response,
                    transform,
                    self.show_x,
                    self.show_y,
                    poi,
                    true,
                    &mut shapes,
                );
                (c, Some(poi.item_id), Some(pt), None)
            } else if let Some(target) = self.focused_point.as_ref() {
                let (c, id, pt) = self.render_focused_point(ui, target, &mut shapes);
                (c, id, pt, None)
            } else {
                (Vec::new(), None, None, None)
            };

        self.append_cursor_lines(ui, &cursors, &mut shapes);

        let painter = ui.painter().with_clip_rect(*transform.frame());
        painter.extend(shapes);

        let active_coord = hover_pos
            .map(|p| hovered_point.unwrap_or_else(|| transform.value_from_position(p)))
            .or(hovered_point);
        self.paint_corner_coordinates(ui, &painter, active_coord);
        if self.show_settings_button
            && self.transform.frame().width() >= 30.0
            && self.transform.frame().height() >= 30.0
        {
            self.render_wrench_button(ui);
        }

        let items = std::mem::take(&mut self.items);
        PreparedUiOutput {
            items,
            cursors,
            hovered_plot_item: hovered_item_id,
            hovered_poi_idx,
        }
    }

    fn render_persisted_tooltip(
        &self,
        state: &a11y::PointTooltipState,
    ) -> (Vec<Cursor>, Option<Id>, Option<PlotPoint>) {
        let mut cursors = Vec::new();
        if !state.suppress_rulers && self.sidecar_3d.is_none() {
            if self.show_x {
                cursors.push(Cursor::Vertical { x: state.point[0] });
            }
            if self.show_y {
                cursors.push(Cursor::Horizontal { y: state.point[1] });
            }
        }
        a11y::show_snapped_point_tooltip(
            self.plot_area_response,
            a11y::SnappedTooltipArgs {
                anchor_pos: state.anchor_pos,
                point: state.point,
                world_point: state.world_point,
                suppress_rulers: state.suppress_rulers || self.sidecar_3d.is_some(),
                title: &state.title,
                body_lines: &state.body_lines,
                announce_on_open: false,
            },
        );
        (
            cursors,
            None,
            Some(PlotPoint::new(state.point[0], state.point[1])),
        )
    }

    fn render_focused_point(
        &self,
        ui: &Ui,
        target: &FocusedPlotPoint,
        shapes: &mut Vec<Shape>,
    ) -> (Vec<Cursor>, Option<Id>, Option<PlotPoint>) {
        if !self.show_x && !self.show_y && self.sidecar_3d.is_none() {
            return (Vec::new(), Some(target.item_id), Some(target.point));
        }
        let screen_pos = self.transform.position_from_point(&target.point);
        let marker_fill = if target.color == Color32::TRANSPARENT {
            ui.visuals().selection.stroke.color
        } else {
            target.color
        };
        let outline = a11y::focus_ring_stroke(ui.visuals(), ui.visuals().extreme_bg_color);
        shapes.push(Shape::circle_filled(screen_pos, 5.0, marker_fill));
        shapes.push(Shape::circle_stroke(screen_pos, 7.0, outline));

        let plot_cfg = items::PlotConfig {
            ui,
            transform: &self.transform,
            show_x: self.show_x,
            show_y: self.show_y,
            sidecar_3d: self.sidecar_3d,
            world_point: target.world_point,
        };
        let mut cursors = Vec::new();
        items::rulers_and_tooltip_at_value(
            self.plot_area_response,
            target.point,
            &target.series_name,
            &plot_cfg,
            &mut cursors,
            &self.label_formatter,
        );
        (cursors, Some(target.item_id), Some(target.point))
    }

    fn append_cursor_lines(&self, ui: &Ui, cursors: &[Cursor], shapes: &mut Vec<Shape>) {
        let line_color = self.cursor_color.unwrap_or_else(|| rulers_color(ui));
        for (slice, always) in [(self.draw_cursors.as_slice(), false), (cursors, true)] {
            for &cursor in slice {
                match cursor {
                    Cursor::Horizontal { y } if self.draw_cursor_y || always => {
                        shapes.push(horizontal_line(
                            self.transform.position_from_point(&PlotPoint::new(0.0, y)),
                            &self.transform,
                            line_color,
                        ));
                    }
                    Cursor::Vertical { x } if self.draw_cursor_x || always => {
                        shapes.push(vertical_line(
                            self.transform.position_from_point(&PlotPoint::new(x, 0.0)),
                            &self.transform,
                            line_color,
                        ));
                    }
                    _ => {}
                }
            }
        }
    }

    fn paint_corner_coordinates(
        &self,
        ui: &Ui,
        painter: &egui::Painter,
        active_coord: Option<PlotPoint>,
    ) {
        let (Some((corner, formatter)), Some(coordinate)) =
            (self.coordinates_formatter.as_ref(), active_coord)
        else {
            return;
        };
        let font_id = TextStyle::Monospace.resolve(ui.style());
        let text = formatter.format(&coordinate, self.transform.bounds());
        let safe_frame =
            legend_safe_rect(*self.transform.frame(), *corner, self.right_controls_rect);
        let padded_frame = safe_frame.shrink(4.0);
        let (anchor, position) = match corner {
            Corner::LeftTop => (Align2::LEFT_TOP, padded_frame.left_top()),
            Corner::RightTop => (Align2::RIGHT_TOP, padded_frame.right_top()),
            Corner::LeftBottom => (Align2::LEFT_BOTTOM, padded_frame.left_bottom()),
            Corner::RightBottom => (Align2::RIGHT_BOTTOM, padded_frame.right_bottom()),
        };
        let text_color = ui.visuals().text_color();
        let galley = painter.layout_no_wrap(text, font_id, text_color);
        let rect = anchor.anchor_size(position, galley.size());
        painter.rect_filled(
            rect.expand(4.0),
            ui.style().visuals.window_corner_radius,
            ui.style().visuals.extreme_bg_color.gamma_multiply(0.75),
        );
        painter.galley(rect.min, galley, text_color);
    }

    fn render_wrench_button(&self, ui: &mut Ui) {
        let wrench_size = Vec2::splat(24.0);
        let wrench_rect = Rect::from_min_size(
            self.transform.frame().right_top() + vec2(-30.0, 6.0),
            wrench_size,
        );
        let mut wrench_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(wrench_rect)
                .layout(Layout::centered_and_justified(egui::Direction::LeftToRight)),
        );
        wrench_ui.set_clip_rect(self.transform.frame().intersect(ui.clip_rect()));
        if a11y::focusable_action_button(
            &mut wrench_ui,
            ui.is_enabled(),
            egui::Button::new("🔧").corner_radius(4.0),
            Some("Plot grid and axis settings"),
        )
        .accessible_tooltip("Plot Grid & Axis Settings")
        .clicked()
        {
            ui.ctx().data_mut(|d| {
                d.insert_temp(self.plot_id.with("toggle_settings"), true);
            });
        }
    }

    fn paint_grid(&self, ui: &Ui, shapes: &mut Vec<(Shape, f32)>, axis: Axis, fade_range: Rangef) {
        let Self {
            transform,
            // axis_formatters,
            grid_spacers,
            clamp_grid,
            ..
        } = self;

        let iaxis = usize::from(axis);

        // Where on the cross-dimension to show the label values
        let bounds = transform.bounds();
        let value_cross = 0.0_f64.clamp(bounds.min[1 - iaxis], bounds.max[1 - iaxis]);

        let input = GridInput {
            bounds: (bounds.min[iaxis], bounds.max[iaxis]),
            base_step_size: transform.dvalue_dpos()[iaxis].abs() * fade_range.min as f64,
        };
        let steps = (grid_spacers[iaxis])(input);

        let clamp_range = clamp_grid.then(|| {
            let mut tight_bounds = PlotBounds::NOTHING;
            for item in &self.items {
                let item_bounds = item.bounds();
                tight_bounds.merge_x(&item_bounds);
                tight_bounds.merge_y(&item_bounds);
            }
            if transform.log_axis().x {
                tight_bounds.min[0] = tight_bounds.min[0].max(transform.bounds().min[0]);
                tight_bounds.max[0] = tight_bounds.max[0].max(tight_bounds.min[0]);
            }
            if transform.log_axis().y {
                tight_bounds.min[1] = tight_bounds.min[1].max(transform.bounds().min[1]);
                tight_bounds.max[1] = tight_bounds.max[1].max(tight_bounds.min[1]);
            }
            tight_bounds
        });

        for step in steps {
            let value_main = step.value;

            if let Some(clamp_range) = clamp_range {
                match axis {
                    Axis::X => {
                        if !clamp_range.range_x().contains(&value_main) {
                            continue;
                        }
                    }
                    Axis::Y => {
                        if !clamp_range.range_y().contains(&value_main) {
                            continue;
                        }
                    }
                }
            }

            let value = match axis {
                Axis::X => PlotPoint::new(value_main, value_cross),
                Axis::Y => PlotPoint::new(value_cross, value_main),
            };

            let pos_in_gui = transform.position_from_point(&value);
            let spacing_in_points = transform.step_spacing_in_points(axis, step);

            if spacing_in_points <= fade_range.min {
                continue; // Too close together
            }

            let line_strength = remap_clamp(spacing_in_points, fade_range, 0.0..=1.0);

            let line_color = color_from_strength(ui, line_strength);

            let mut p0 = pos_in_gui;
            let mut p1 = pos_in_gui;
            p0[1 - iaxis] = transform.frame().min[1 - iaxis];
            p1[1 - iaxis] = transform.frame().max[1 - iaxis];

            if let Some(clamp_range) = clamp_range {
                match axis {
                    Axis::X => {
                        p0.y = transform.position_from_point_y(clamp_range.min[1]);
                        p1.y = transform.position_from_point_y(clamp_range.max[1]);
                    }
                    Axis::Y => {
                        p0.x = transform.position_from_point_x(clamp_range.min[0]);
                        p1.x = transform.position_from_point_x(clamp_range.max[0]);
                    }
                }
            }

            shapes.push((
                Shape::line_segment([p0, p1], Stroke::new(1.0_f32, line_color)),
                line_strength,
            ));
        }
    }

    fn hover(
        &self,
        ui: &Ui,
        pointer: Pos2,
        shapes: &mut Vec<Shape>,
    ) -> (Vec<Cursor>, Option<Id>, Option<PlotPoint>, Option<usize>) {
        if !self.show_x && !self.show_y && self.sidecar_3d.is_none() {
            return (Vec::new(), None, None, None);
        }
        let snap_distance = self
            .cursor_snap_distance
            .unwrap_or_else(|| ui.style().interaction.interact_radius);
        let snap_dist_sq = snap_distance.powi(2);

        if let Some((poi_idx, _)) = poi_interaction::find_closest_poi(
            &self.critical_points,
            pointer,
            &self.transform,
            snap_dist_sq,
        ) {
            let poi = &self.critical_points[poi_idx];
            let (cursors, pt) = poi_interaction::render_active_poi(
                ui,
                self.plot_area_response,
                &self.transform,
                self.show_x,
                self.show_y,
                poi,
                false,
                shapes,
            );
            return (cursors, Some(poi.item_id), Some(pt), Some(poi_idx));
        }

        let plot = items::PlotConfig {
            ui,
            transform: &self.transform,
            show_x: self.show_x,
            show_y: self.show_y,
            sidecar_3d: self.sidecar_3d,
            world_point: None,
        };
        let mut cursors = Vec::new();
        let (hovered_id, hovered_point) =
            self.hover_closest_item(pointer, snap_dist_sq, &plot, shapes, &mut cursors);
        (cursors, hovered_id, hovered_point, None)
    }

    fn hover_closest_item(
        &self,
        pointer: Pos2,
        snap_dist_sq: f32,
        plot: &items::PlotConfig<'_>,
        shapes: &mut Vec<Shape>,
        cursors: &mut Vec<Cursor>,
    ) -> (Option<Id>, Option<PlotPoint>) {
        enum HoverHit<'a> {
            Continuous(&'a dyn PlotItem, PlotPoint),
            Discrete(&'a dyn PlotItem, ClosestElem),
        }
        let best = self
            .items
            .iter()
            .filter(|entry| entry.allow_hover())
            .filter_map(|entry| {
                let item = &**entry;
                if self.sidecar_3d.is_none()
                    && let Some((pt, dist_sq)) =
                        item.closest_continuous_point(pointer, &self.transform)
                {
                    Some((dist_sq, HoverHit::Continuous(item, pt)))
                } else {
                    item.find_closest(pointer, &self.transform)
                        .map(|elem| (elem.dist_sq, HoverHit::Discrete(item, elem)))
                }
            })
            .min_by_key(|(d, _)| d.ord())
            .filter(|(d, _)| *d <= snap_dist_sq);

        match best.map(|(_, hit)| hit) {
            Some(HoverHit::Continuous(item, value)) => {
                let pt = items::render_snapped_hover_point(
                    item,
                    self.plot_area_response,
                    value,
                    shapes,
                    cursors,
                    plot,
                    &self.label_formatter,
                );
                (Some(item.id()), Some(pt))
            }
            Some(HoverHit::Discrete(item, elem)) => {
                let pt = item.on_hover(
                    self.plot_area_response,
                    elem,
                    shapes,
                    cursors,
                    plot,
                    &self.label_formatter,
                );
                (Some(item.id()), pt)
            }
            None => {
                if self.sidecar_3d.is_none() {
                    let value = self.transform.value_from_position(pointer);
                    items::rulers_and_tooltip_at_value(
                        self.plot_area_response,
                        value,
                        "",
                        plot,
                        cursors,
                        &self.label_formatter,
                    );
                }
                (None, None)
            }
        }
    }
}

/// Returns next bigger power in given base
/// e.g.
/// ```text
/// next_power(0.01, 10.0) == 0.1
/// next_power(0.02, 10.0) == 0.1
/// next_power(0.2,  10.0) == 1.0
/// ```
fn next_power(value: f64, base: f64) -> f64 {
    debug_assert_ne!(value, 0.0, "Bad input"); // can be negative (typical for Y axis)
    let abs_value = value.abs();
    let mut exp = abs_value.log(base).ceil() as i32;
    if base.powi(exp) <= abs_value {
        exp += 1;
    }
    base.powi(exp)
}

#[test]
fn test_next_power() {
    assert_eq!(next_power(0.01, 10.0), 0.1);
    assert_eq!(next_power(0.02, 10.0), 0.1);
    assert_eq!(next_power(0.2, 10.0), 1.0);
}

/// Fill in all values between [min, max] which are a multiple of `step_size`
fn generate_marks(step_sizes: [f64; 3], bounds: (f64, f64)) -> Vec<GridMark> {
    let mut steps = vec![];
    fill_marks_between(&mut steps, step_sizes[0], bounds);
    fill_marks_between(&mut steps, step_sizes[1], bounds);
    fill_marks_between(&mut steps, step_sizes[2], bounds);

    // Remove duplicates:
    // This can happen because we have overlapping steps, e.g.:
    // step_size[0] =   10  =>  [-10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    // step_size[1] =  100  =>  [     0,                                     100          ]
    // step_size[2] = 1000  =>  [     0                                                   ]

    steps.sort_by(|a, b| cmp_f64(a.value, b.value));

    let min_step = step_sizes.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let eps = 0.1 * min_step; // avoid putting two ticks too closely together

    let mut deduplicated: Vec<GridMark> = Vec::with_capacity(steps.len());
    for step in steps {
        if let Some(last) = deduplicated.last_mut()
            && (last.value - step.value).abs() < eps
        {
            // Keep the one with the largest step size
            if last.step_size < step.step_size {
                *last = step;
            }
            continue;
        }
        deduplicated.push(step);
    }

    deduplicated
}

#[test]
fn test_generate_marks() {
    fn approx_eq(a: &GridMark, b: &GridMark) -> bool {
        (a.value - b.value).abs() < 1e-10 && a.step_size == b.step_size
    }

    let gm = |value, step_size| GridMark { value, step_size };

    let marks = generate_marks([0.01, 0.1, 1.0], (2.855, 3.015));
    let expected = vec![
        gm(2.86, 0.01),
        gm(2.87, 0.01),
        gm(2.88, 0.01),
        gm(2.89, 0.01),
        gm(2.90, 0.1),
        gm(2.91, 0.01),
        gm(2.92, 0.01),
        gm(2.93, 0.01),
        gm(2.94, 0.01),
        gm(2.95, 0.01),
        gm(2.96, 0.01),
        gm(2.97, 0.01),
        gm(2.98, 0.01),
        gm(2.99, 0.01),
        gm(3.00, 1.),
        gm(3.01, 0.01),
    ];

    let mut problem = if marks.len() == expected.len() {
        None
    } else {
        Some(format!(
            "Different lengths: got {}, expected {}",
            marks.len(),
            expected.len()
        ))
    };

    for (i, (a, b)) in marks.iter().zip(&expected).enumerate() {
        if !approx_eq(a, b) {
            problem = Some(format!("Mismatch at index {i}: {a:?} != {b:?}"));
            break;
        }
    }

    if let Some(problem) = problem {
        panic!("Test failed: {problem}. Got: {marks:#?}, expected: {expected:#?}");
    }
}

fn cmp_f64(a: f64, b: f64) -> Ordering {
    match a.partial_cmp(&b) {
        Some(ord) => ord,
        None => a.is_nan().cmp(&b.is_nan()),
    }
}

/// Fill in all values between [min, max] which are a multiple of `step_size`
fn fill_marks_between(out: &mut Vec<GridMark>, step_size: f64, (min, max): (f64, f64)) {
    debug_assert!(min <= max, "Bad plot bounds: min: {min}, max: {max}");
    let first = (min / step_size).ceil() as i64;
    let last = (max / step_size).ceil() as i64;

    let marks_iter = (first..last).map(|i| {
        let value = (i as f64) * step_size;
        GridMark { value, step_size }
    });
    out.extend(marks_iter);
}

/// Helper for formatting a number so that we always show at least a few decimals,
/// unless it is an integer, in which case we never show any decimals.
pub fn format_number(number: f64, num_decimals: usize) -> String {
    let is_integral = number as i64 as f64 == number;
    if is_integral {
        // perfect integer - show it as such:
        format!("{number:.0}")
    } else {
        // make sure we tell the user it is not an integer by always showing a decimal or two:
        format!("{:.*}", num_decimals.at_least(1), number)
    }
}

/// Determine a color from a 0-1 strength value.
pub fn color_from_strength(ui: &Ui, strength: f32) -> Color32 {
    let base_color = ui.visuals().text_color();
    base_color.gamma_multiply(strength.sqrt())
}

#[doc(hidden)]
#[derive(Clone)]
pub enum DeferredDatasetEntry {
    SharedPoints {
        name: String,
        points: Arc<Vec<PlotPoint>>,
    },
    SharedSegments {
        name: String,
        segments: Arc<Vec<Vec<PlotPoint>>>,
    },
    SharedArrows {
        name: String,
        origins: Arc<Vec<PlotPoint>>,
        tips: Arc<Vec<PlotPoint>>,
    },
    Owned {
        name: String,
        points: Vec<[f64; 2]>,
    },
}

impl DeferredDatasetEntry {
    fn materialize(&self) -> (String, Vec<[f64; 2]>) {
        match self {
            Self::SharedPoints { name, points } => {
                (name.clone(), points.iter().map(|p| [p.x, p.y]).collect())
            }
            Self::SharedSegments { name, segments } => {
                let total: usize = segments.iter().map(Vec::len).sum();
                let mut out = Vec::with_capacity(total);
                for seg in segments.iter() {
                    out.extend(seg.iter().map(|p| [p.x, p.y]));
                }
                (name.clone(), out)
            }
            Self::SharedArrows {
                name,
                origins,
                tips,
            } => {
                let mut out = Vec::with_capacity(origins.len().min(tips.len()) * 2);
                for (origin, tip) in origins.iter().zip(tips.iter()) {
                    out.push([origin.x, origin.y]);
                    out.push([tip.x, tip.y]);
                }
                (name.clone(), out)
            }
            Self::Owned { name, points } => (name.clone(), points.clone()),
        }
    }

    fn check_finite(&self) -> Result<(), (&str, char, usize, f64)> {
        match self {
            Self::SharedPoints { name, points } => {
                for (idx, pt) in points.iter().enumerate() {
                    check_xy_finite(name.as_str(), idx, pt.x, pt.y)?;
                }
                Ok(())
            }
            Self::SharedSegments { name, segments } => {
                let mut flat_idx = 0usize;
                for seg in segments.iter() {
                    for pt in seg {
                        check_xy_finite(name.as_str(), flat_idx, pt.x, pt.y)?;
                        flat_idx += 1;
                    }
                }
                Ok(())
            }
            Self::SharedArrows {
                name,
                origins,
                tips,
            } => {
                let mut flat_idx = 0usize;
                for (origin, tip) in origins.iter().zip(tips.iter()) {
                    for pt in [origin, tip] {
                        check_xy_finite(name.as_str(), flat_idx, pt.x, pt.y)?;
                        flat_idx += 1;
                    }
                }
                Ok(())
            }
            Self::Owned { name, points } => {
                for (idx, pt) in points.iter().enumerate() {
                    check_xy_finite(name.as_str(), idx, pt[0], pt[1])?;
                }
                Ok(())
            }
        }
    }
}

fn check_xy_finite(name: &str, idx: usize, x: f64, y: f64) -> Result<(), (&str, char, usize, f64)> {
    if !(x > f64::NEG_INFINITY && x < f64::INFINITY && y > f64::NEG_INFINITY && y < f64::INFINITY) {
        if !y.is_finite() {
            Err((name, 'y', idx, y))
        } else {
            Err((name, 'x', idx, x))
        }
    } else {
        Ok(())
    }
}

thread_local! {
    static LAST_PLOT_DATASETS: std::cell::RefCell<(Vec<DeferredDatasetEntry>, u64)> =
        const { std::cell::RefCell::new((Vec::new(), 0)) };
}

/// Clears thread-local record of plot datasets from the current frame.
pub fn clear_last_plot_datasets() {
    LAST_PLOT_DATASETS.with(|d| {
        let mut guard = d.borrow_mut();
        guard.0.clear();
        guard.1 = 0;
    });
    clear_last_plot_datasets_3d();
    clear_last_plot_primitives();
}

/// Appends datasets captured from an `egui_plot::Plot` widget.
pub fn append_last_plot_datasets(datasets: &[(String, Vec<[f64; 2]>)]) {
    if datasets.is_empty() {
        return;
    }
    LAST_PLOT_DATASETS.with(|d| {
        use std::hash::{Hash as _, Hasher as _};
        let mut guard = d.borrow_mut();
        let mut hasher = ahash::AHasher::default();
        guard.1.hash(&mut hasher);
        for (name, pts) in datasets {
            name.hash(&mut hasher);
            pts.len().hash(&mut hasher);
            if let Some(first) = pts.first() {
                first[0].to_bits().hash(&mut hasher);
                first[1].to_bits().hash(&mut hasher);
            }
            if let Some(last) = pts.last() {
                last[0].to_bits().hash(&mut hasher);
                last[1].to_bits().hash(&mut hasher);
            }
            guard.0.push(DeferredDatasetEntry::Owned {
                name: name.clone(),
                points: pts.clone(),
            });
        }
        guard.1 = hasher.finish().max(1);
    });
}

pub(crate) fn append_deferred_plot_datasets(entries: Vec<DeferredDatasetEntry>, hash_mix: u64) {
    if entries.is_empty() {
        return;
    }
    LAST_PLOT_DATASETS.with(|d| {
        use std::hash::{Hash as _, Hasher as _};
        let mut guard = d.borrow_mut();
        let mut hasher = ahash::AHasher::default();
        guard.1.hash(&mut hasher);
        hash_mix.hash(&mut hasher);
        entries.len().hash(&mut hasher);
        guard.0.extend(entries);
        guard.1 = hasher.finish().max(1);
    });
}

/// Returns a non-zero fingerprint when datasets were captured in the active frame (`0` if empty).
pub fn last_plot_capture_hash() -> u64 {
    LAST_PLOT_DATASETS.with(|d| d.borrow().1)
}

/// Checks captured plot datasets for non-finite coordinates without cloning point buffers.
///
/// Returns `Ok(true)` if at least one dataset was captured and all points are finite,
/// `Ok(false)` if no datasets were captured, or `Err((series_name, axis, index, bad_value))` on `NaN`/`Inf`.
pub fn check_last_plot_datasets_finite() -> Result<bool, (String, char, usize, f64)> {
    LAST_PLOT_DATASETS.with(|d| {
        let guard = d.borrow();
        if guard.0.is_empty() {
            return Ok(false);
        }
        for entry in &guard.0 {
            if let Err((name, axis, idx, bad)) = entry.check_finite() {
                return Err((name.to_owned(), axis, idx, bad));
            }
        }
        Ok(true)
    })
}

/// Retrieves all plot datasets captured across `egui_plot::Plot` widgets rendered in the active session.
pub fn get_last_plot_datasets() -> Vec<(String, Vec<[f64; 2]>)> {
    LAST_PLOT_DATASETS.with(|d| {
        let guard = d.borrow();
        guard
            .0
            .iter()
            .map(DeferredDatasetEntry::materialize)
            .collect()
    })
}

static PENDING_CANVAS_CALLOUTS: std::sync::Mutex<
    Option<std::collections::HashMap<Id, Vec<CanvasCallout>>>,
> = std::sync::Mutex::new(None);

/// Queue a callout created from canvas interactions for a specific plot.
pub fn push_pending_callout(plot_id: Id, callout: CanvasCallout) {
    let mut guard = PENDING_CANVAS_CALLOUTS.lock().unwrap();
    let map = guard.get_or_insert_with(std::collections::HashMap::new);
    map.entry(plot_id).or_default().push(callout);
}

/// Drains any pending canvas callouts created for a specific plot.
pub fn drain_pending_callouts(plot_id: Id) -> Vec<CanvasCallout> {
    let mut guard = PENDING_CANVAS_CALLOUTS.lock().unwrap();
    if let Some(map) = guard.as_mut() {
        map.remove(&plot_id).unwrap_or_default()
    } else {
        Vec::new()
    }
}

#[test]
fn test_logarithmic_grid_spacer_positive_and_non_positive_bounds() {
    let spacer = logarithmic_grid_spacer();
    let input_pos = GridInput {
        bounds: (1.0, 100.0),
        base_step_size: 1.0,
    };
    let marks_pos = (spacer)(input_pos);
    assert!(
        !marks_pos.is_empty(),
        "Log spacer should return marks for positive bounds"
    );
    assert!(marks_pos.iter().any(|m| (m.value - 1.0).abs() < 1e-5));
    assert!(marks_pos.iter().any(|m| (m.value - 10.0).abs() < 1e-5));
    assert!(marks_pos.iter().any(|m| (m.value - 100.0).abs() < 1e-5));

    // Test non-positive data bounds handling
    let spacer_nonpos = logarithmic_grid_spacer();
    let input_nonpos = GridInput {
        bounds: (-10.0, 50.0),
        base_step_size: 1.0,
    };
    let marks_nonpos = (spacer_nonpos)(input_nonpos);
    assert!(
        !marks_nonpos.is_empty(),
        "Log spacer should handle non-positive bounds by falling back to positive minimum bounds"
    );
    assert!(
        marks_nonpos.iter().all(|m| m.value > 0.0),
        "All generated log marks must be positive"
    );
}

#[test]
fn test_plot_memory_grid_settings_persistence() {
    let ctx = egui::Context::default();
    let id = Id::new("test_plot_memory_id");

    let mem = PlotMemory {
        auto_bounds: Vec2b::new(true, true),
        hovered_legend_item: None,
        hidden_items: Default::default(),
        transform: PlotTransform::new_with_invert_axis(
            Rect::from_min_size(Pos2::ZERO, vec2(100.0, 100.0)),
            PlotBounds::NOTHING,
            Vec2b::new(false, false),
            Vec2b::new(false, false),
        ),
        last_click_pos_for_zoom: None,
        x_axis_thickness: Default::default(),
        y_axis_thickness: Default::default(),
        imported_datasets: Default::default(),
        callouts: vec![SavedCallout::new([1.0, 2.0], "Test Callout", [10.0, -10.0])],
        log_x: true,
        log_y: true,
        grid_density: 16.0,
        show_x_labels: false,
        show_y_labels: true,
        show_hover: true,
        toolbar_collapsed: false,
        clamp_grid: true,
        show_grid: Vec2b::new(true, false),
        show_settings_modal: true,
        last_right_click_coord: None,
    };

    mem.clone().store(&ctx, id);

    let loaded = PlotMemory::load(&ctx, id).expect("PlotMemory should load from context");
    assert!(loaded.log_x);
    assert!(loaded.log_y);
    assert_eq!(loaded.grid_density, 16.0);
    assert!(!loaded.show_x_labels);
    assert!(loaded.show_y_labels);
    assert!(loaded.clamp_grid);
    assert_eq!(loaded.show_grid, Vec2b::new(true, false));
    assert!(loaded.show_settings_modal);
    assert_eq!(loaded.callouts.len(), 1);
    assert_eq!(loaded.callouts[0].label, "Test Callout");
    assert_eq!(loaded.callouts[0].point, [1.0, 2.0]);
}

#[test]
fn test_canvas_callout_queue_and_drain() {
    let id = Id::new("test_callout_plot_id");
    let callout = CanvasCallout {
        target_point: [2.5, 4.0],
        label_offset: [15.0, -20.0],
        text: "Test Point".to_string(),
    };
    push_pending_callout(id, callout.clone());
    let drained = drain_pending_callouts(id);
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].target_point, [2.5, 4.0]);
    assert_eq!(drained[0].label_offset, [15.0, -20.0]);
    assert_eq!(drained[0].text, "Test Point");

    // Second drain should be empty
    assert!(drain_pending_callouts(id).is_empty());
}

#[test]
fn generate_plot_settings_modal_screenshot() {
    let datasets = vec![(
        "Log-Scale Signal".to_string(),
        vec![[1.0, 10.0], [10.0, 100.0], [100.0, 1000.0]],
    )];
    let options = crate::export::ExportOptions::default();
    if let Ok(png_bytes) = crate::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/plot_grid_settings_modal.png", png_bytes);
    }
}

#[test]
fn test_plot_settings_modal_toggle_and_rendering() {
    let ctx = egui::Context::default();
    let id = Id::new("test_plot_modal_rendering");

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            Plot::new(id).id(id).show(ui, |_plot_ui| {});
        });
    });

    // Toggle settings modal via context storage signal
    ctx.data_mut(|d| d.insert_temp(id.with("toggle_settings"), true));

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            Plot::new(id).id(id).show(ui, |_plot_ui| {});
        });
    });

    let mem = PlotMemory::load(&ctx, id).expect("PlotMemory should exist");
    assert!(
        mem.show_settings_modal,
        "Settings modal should be active after toggle"
    );
}

#[cfg(test)]
mod cursor_snap_tests {
    use super::*;

    #[test]
    fn test_cursor_snap_distance_builder() {
        let default_plot = Plot::new("default_plot");
        assert_eq!(default_plot.cursor_snap_distance, Some(16.0));

        let custom_plot = Plot::new("custom_plot").cursor_snap_distance(32.0);
        assert_eq!(custom_plot.cursor_snap_distance, Some(32.0));
    }

    #[test]
    fn test_cursor_snap_distance_hover() {
        egui::__run_test_ui(|ui| {
            let pts = vec![[0.0, 0.0], [10.0, 10.0]];
            let line = Line::new("Series 1", PlotPoints::new(pts));

            let response = Plot::new("snap_test")
                .cursor_snap_distance(16.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(line);
                });

            assert!(response.response.rect.width() > 0.0);
            assert_eq!(response.hovered_plot_item, None);
        });
    }

    #[test]
    fn test_cursor_snap_custom_threshold() {
        egui::__run_test_ui(|ui| {
            let pts = vec![[5.0, 5.0]];
            let points = Points::new("Points Series", PlotPoints::new(pts));

            let res_tight = Plot::new("tight_snap")
                .cursor_snap_distance(2.0)
                .show(ui, |plot_ui| {
                    plot_ui.points(points);
                });

            assert_eq!(res_tight.hovered_plot_item, None);
            assert!(res_tight.transform.bounds().is_valid());
        });
    }

    #[test]
    fn test_overlapping_series_selection() {
        egui::__run_test_ui(|ui| {
            let s1 = vec![[0.0, 0.0], [10.0, 10.0]];
            let s2 = vec![[0.0, 1.0], [10.0, 11.0]];

            let line1 = Line::new("Series Alpha", PlotPoints::new(s1));
            let line2 = Line::new("Series Beta", PlotPoints::new(s2));

            let response = Plot::new("overlap_test").show(ui, |plot_ui| {
                plot_ui.line(line1);
                plot_ui.line(line2);
            });

            assert!(response.response.rect.width() > 0.0);
            assert_eq!(
                response.transform.frame().width(),
                response.response.rect.width()
            );
        });
    }

    #[test]
    fn test_generate_snap_tooltip_verification_screenshot() {
        let sample_pts: Vec<[f64; 2]> = (0..50)
            .map(|i| {
                let x = i as f64 * 0.2;
                let y = (x * 0.8).sin() * 2.0;
                [x, y]
            })
            .collect();

        let datasets = vec![("Sine Wave Curve".to_string(), sample_pts)];
        let options = ExportOptions {
            width: 800,
            height: 450,
            line_thickness: 2.0,
            ..Default::default()
        };

        let png_res = export_custom_png(&datasets, &options);
        assert!(png_res.is_ok());
        if let Ok(png_bytes) = png_res {
            let _ = std::fs::write("/tmp/snap_tooltip_verification.png", png_bytes);
        }
    }

    #[test]
    fn test_coordinates_formatter_snapped_to_curve_point() {
        let target_curve_point = PlotPoint::new(3.0, 7.0);

        // Obtain screen position of the target point
        let ctx = egui::Context::default();
        let mut curve_point_screen_pos = Pos2::ZERO;
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0))),
                ..Default::default()
            },
            |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let pts = vec![[0.0, 0.0], [3.0, 7.0], [10.0, 10.0]];
                    let line = Line::new("Series 1", PlotPoints::new(pts));
                    let res =
                        Plot::new("snap_fmt_test")
                            .cursor_snap_distance(20.0)
                            .show(ui, |plot_ui| {
                                plot_ui.line(line);
                            });
                    curve_point_screen_pos = res.transform.position_from_point(&target_curve_point);
                });
            },
        );

        let run_plot_at_pointer =
            |pointer_pos: Pos2| -> (Option<PlotPoint>, Option<Id>, PlotPoint) {
                let ctx = egui::Context::default();
                let pts = vec![[0.0, 0.0], [3.0, 7.0], [10.0, 10.0]];

                // Frame 1: initial layout
                let _ = ctx.run(
                    egui::RawInput {
                        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0))),
                        ..Default::default()
                    },
                    |ctx| {
                        egui::CentralPanel::default().show(ctx, |ui| {
                            let line = Line::new("Series 1", PlotPoints::new(pts.clone()));
                            let _ = Plot::new("snap_fmt_test").cursor_snap_distance(20.0).show(
                                ui,
                                |plot_ui| {
                                    plot_ui.line(line);
                                },
                            );
                        });
                    },
                );

                // Frame 2: with mouse pointer moved to destination
                let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
                let cap_clone = captured.clone();
                let mut hovered_item = None;
                let mut raw_val = PlotPoint::new(0.0, 0.0);

                let _ = ctx.run(
                    egui::RawInput {
                        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0))),
                        events: vec![egui::Event::PointerMoved(pointer_pos)],
                        ..Default::default()
                    },
                    |ctx| {
                        egui::CentralPanel::default().show(ctx, |ui| {
                            let line = Line::new("Series 1", PlotPoints::new(pts.clone()));
                            let res = Plot::new("snap_fmt_test")
                                .cursor_snap_distance(20.0)
                                .coordinates_formatter(
                                    Corner::LeftTop,
                                    CoordinatesFormatter::new({
                                        let cap = cap_clone.clone();
                                        move |point, _| {
                                            *cap.lock().unwrap() = Some(*point);
                                            format!("x: {:.2}, y: {:.2}", point.x, point.y)
                                        }
                                    }),
                                )
                                .show(ui, |plot_ui| {
                                    plot_ui.line(line);
                                });
                            hovered_item = res.hovered_plot_item;
                            raw_val = res.transform.value_from_position(pointer_pos);
                        });
                    },
                );

                let fmt_point = *captured.lock().unwrap();
                (fmt_point, hovered_item, raw_val)
            };

        // 1. Hover within snap distance (5 pixels away)
        let near_pos = curve_point_screen_pos + vec2(4.0, -3.0);
        assert!(near_pos.distance(curve_point_screen_pos) <= 20.0);
        let (snapped_opt, hovered_id, _) = run_plot_at_pointer(near_pos);
        let line_id = Id::new("Series 1");
        assert_eq!(hovered_id, Some(line_id));
        let snapped_coord =
            snapped_opt.expect("coordinates formatter should be called when hovering");
        assert_eq!(snapped_coord.x, target_curve_point.x);
        assert_eq!(snapped_coord.y, target_curve_point.y);

        // 2. Hover outside snap distance (50 pixels away): fallback to raw pointer coordinates
        let far_pos = curve_point_screen_pos + vec2(50.0, 50.0);
        assert!(far_pos.distance(curve_point_screen_pos) > 20.0);
        let (unsnapped_opt, far_hovered_id, far_raw_point) = run_plot_at_pointer(far_pos);
        assert_eq!(far_hovered_id, None);
        let unsnapped_coord = unsnapped_opt
            .expect("coordinates formatter should be called when hovering outside snap");
        assert_ne!(unsnapped_coord.x, target_curve_point.x);
        assert_ne!(unsnapped_coord.y, target_curve_point.y);
        assert!((unsnapped_coord.x - far_raw_point.x).abs() < 1e-6);
        assert!((unsnapped_coord.y - far_raw_point.y).abs() < 1e-6);
    }
}

#[cfg(test)]
mod callout_context_menu_tests {
    use super::*;

    #[test]
    fn test_callout_auto_dispatch_and_rendering() {
        let ctx = egui::Context::default();
        let id = Id::new("test_callout_auto_dispatch");

        // Seed PlotMemory with a saved callout
        let mem = PlotMemory {
            auto_bounds: Vec2b::new(true, true),
            hovered_legend_item: None,
            hidden_items: Default::default(),
            transform: PlotTransform::new_with_invert_axis(
                Rect::from_min_size(Pos2::ZERO, vec2(200.0, 200.0)),
                PlotBounds::from_min_max([-10.0, -10.0], [10.0, 10.0]),
                Vec2b::new(false, false),
                Vec2b::new(false, false),
            ),
            last_click_pos_for_zoom: None,
            x_axis_thickness: Default::default(),
            y_axis_thickness: Default::default(),
            imported_datasets: Default::default(),
            callouts: vec![SavedCallout::new(
                [2.0, 3.0],
                "Equilibrium Point",
                [25.0, -25.0],
            )],
            log_x: false,
            log_y: false,
            grid_density: 8.0,
            show_x_labels: true,
            show_y_labels: true,
            show_hover: true,
            toolbar_collapsed: false,
            clamp_grid: false,
            show_grid: Vec2b::new(true, true),
            show_settings_modal: false,
            last_right_click_coord: None,
        };
        mem.store(&ctx, id);

        // Run the plot - the saved callout should be automatically rendered via plot_ui.callout
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let _res = Plot::new(id).id(id).show(ui, |plot_ui| {
                    plot_ui.line(Line::new(
                        "Curve",
                        PlotPoints::new(vec![[0.0, 0.0], [5.0, 5.0]]),
                    ));
                });
            });
        });

        let loaded = PlotMemory::load(&ctx, id).expect("PlotMemory should exist");
        assert_eq!(loaded.callouts.len(), 1);
        assert_eq!(loaded.callouts[0].label, "Equilibrium Point");
        assert_eq!(loaded.callouts[0].point, [2.0, 3.0]);
        assert_eq!(loaded.callouts[0].offset, [25.0, -25.0]);
    }

    #[test]
    fn test_callout_deletion_and_clear_all() {
        let ctx = egui::Context::default();
        let id = Id::new("test_callout_clear");

        let mem = PlotMemory {
            auto_bounds: Vec2b::new(true, true),
            hovered_legend_item: None,
            hidden_items: Default::default(),
            transform: PlotTransform::new_with_invert_axis(
                Rect::from_min_size(Pos2::ZERO, vec2(200.0, 200.0)),
                PlotBounds::from_min_max([-10.0, -10.0], [10.0, 10.0]),
                Vec2b::new(false, false),
                Vec2b::new(false, false),
            ),
            last_click_pos_for_zoom: None,
            x_axis_thickness: Default::default(),
            y_axis_thickness: Default::default(),
            imported_datasets: Default::default(),
            callouts: vec![
                SavedCallout::new([1.0, 1.0], "Callout 1", [20.0, -20.0]),
                SavedCallout::new([2.0, 4.0], "Callout 2", [20.0, -20.0]),
            ],
            log_x: false,
            log_y: false,
            grid_density: 8.0,
            show_x_labels: true,
            show_y_labels: true,
            show_hover: true,
            toolbar_collapsed: false,
            clamp_grid: false,
            show_grid: Vec2b::new(true, true),
            show_settings_modal: false,
            last_right_click_coord: None,
        };
        mem.store(&ctx, id);

        // Remove one callout
        let mut loaded = PlotMemory::load(&ctx, id).unwrap();
        assert_eq!(loaded.callouts.len(), 2);
        loaded.callouts.remove(0);
        assert_eq!(loaded.callouts.len(), 1);
        assert_eq!(loaded.callouts[0].label, "Callout 2");
        loaded.store(&ctx, id);

        // Clear all callouts
        let mut loaded2 = PlotMemory::load(&ctx, id).unwrap();
        loaded2.callouts.clear();
        assert!(loaded2.callouts.is_empty());
        loaded2.store(&ctx, id);

        let final_loaded = PlotMemory::load(&ctx, id).unwrap();
        assert!(final_loaded.callouts.is_empty());
    }

    #[test]
    fn test_callout_coordinate_conversion_and_pan_zoom() {
        let rect = Rect::from_min_size(Pos2::new(100.0, 100.0), vec2(400.0, 300.0));
        let bounds = PlotBounds::from_min_max([0.0, 0.0], [10.0, 10.0]);
        let transform = PlotTransform::new(rect, bounds, false);

        // Canvas point at center
        let center_canvas = rect.center();
        let plot_pt = transform.value_from_position(center_canvas);
        assert!((plot_pt.x - 5.0).abs() < 1e-4);
        assert!((plot_pt.y - 5.0).abs() < 1e-4);

        // Saved callout at (5.0, 5.0) with offset (30.0, -30.0)
        let saved = SavedCallout::new([plot_pt.x, plot_pt.y], "Center Annotation", [30.0, -30.0]);
        let callout = saved.to_callout();
        assert_eq!(callout.target_point, PlotPoint::new(5.0, 5.0));
        assert_eq!(callout.label_offset, vec2(30.0, -30.0));

        // Pan/zoom transform: zoomed in by 2x
        let zoomed_bounds = PlotBounds::from_min_max([2.5, 2.5], [7.5, 7.5]);
        let zoomed_transform = PlotTransform::new(rect, zoomed_bounds, false);

        // The target position on screen should remain at rect center
        let zoomed_pos = zoomed_transform.position_from_point(&callout.target_point);
        assert!((zoomed_pos.x - center_canvas.x).abs() < 1e-4);
        assert!((zoomed_pos.y - center_canvas.y).abs() < 1e-4);
    }

    #[test]
    fn test_generate_callout_plot_verification_screenshot() {
        // Render ODE trajectory curve and generate verification artifact
        let ode_pts: Vec<[f64; 2]> = (0..100)
            .map(|i| {
                let t = i as f64 * 0.1;
                let y = (-0.2 * t).exp() * (2.0 * t).cos();
                [t, y]
            })
            .collect();

        let datasets = vec![("ODE Trajectory y(t)".to_string(), ode_pts)];
        let options = ExportOptions {
            width: 800,
            height: 450,
            line_thickness: 2.5,
            ..Default::default()
        };

        let png_res = export_custom_png(&datasets, &options);
        assert!(png_res.is_ok());
        if let Ok(png_bytes) = png_res {
            let _ = std::fs::write("/tmp/callout_context_menu_verification.png", png_bytes);
        }
    }
}

#[cfg(test)]
mod floating_zoom_toolbar_tests {
    use super::*;

    #[test]
    fn test_plot_builder_zoom_toolbar_toggle() {
        let plot = Plot::new("test_builder");
        assert!(
            plot.show_navigation_controls,
            "Navigation controls should be enabled by default"
        );
        assert!(
            plot.show_zoom_buttons,
            "Zoom buttons alias should be enabled by default"
        );

        let plot_off = plot.show_navigation_controls(false);
        assert!(
            !plot_off.show_navigation_controls,
            "show_navigation_controls(false) should disable toolbar"
        );
        assert!(
            !plot_off.show_zoom_buttons,
            "show_navigation_controls(false) should also disable alias"
        );

        let plot_on = plot_off.show_navigation_controls(true);
        assert!(
            plot_on.show_navigation_controls,
            "show_navigation_controls(true) should re-enable toolbar"
        );

        let plot_alias = plot_on.show_zoom_overlay(false);
        assert!(
            !plot_alias.show_navigation_controls,
            "show_zoom_overlay(false) alias should disable toolbar"
        );

        let plot_alias2 = plot_alias.show_zoom_buttons(true);
        assert!(
            plot_alias2.show_navigation_controls,
            "show_zoom_buttons(true) alias should re-enable toolbar"
        );
    }

    #[test]
    fn test_zoom_in_twenty_percent_magnification() {
        let ctx = egui::Context::default();
        let id = Id::new("test_zoom_in");

        // Frame 1: Establish initial plot with fixed default bounds
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id)
                    .id(id)
                    .default_x_bounds(-10.0, 10.0)
                    .default_y_bounds(-5.0, 5.0)
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                    });
            });
        });

        let mem1 = PlotMemory::load(&ctx, id).expect("PlotMemory should exist");
        let initial_bounds = *mem1.transform.bounds();
        let initial_width = initial_bounds.width();
        let initial_height = initial_bounds.height();
        assert!((initial_width - 20.0).abs() < 1e-4);
        assert!((initial_height - 10.0).abs() < 1e-4);

        // Frame 2: Trigger Zoom In signal
        ctx.data_mut(|d| d.insert_temp(id.with("zoom_in"), true));

        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id).id(id).show(ui, |plot_ui| {
                    plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                });
            });
        });

        let mem2 = PlotMemory::load(&ctx, id).expect("PlotMemory should exist");
        let zoomed_bounds = *mem2.transform.bounds();
        let zoomed_width = zoomed_bounds.width();
        let zoomed_height = zoomed_bounds.height();

        // 20% visible range reduction means range is divided by 1.25 (i.e. multiplied by 0.8)
        let expected_width = initial_width / 1.25;
        let expected_height = initial_height / 1.25;
        assert!(
            (zoomed_width - expected_width).abs() < 1e-4,
            "Zoom in width {zoomed_width} should match expected {expected_width} (20% visible range reduction)"
        );
        assert!(
            (zoomed_height - expected_height).abs() < 1e-4,
            "Zoom in height {zoomed_height} should match expected {expected_height}"
        );
        // Center should remain unchanged around origin
        assert!(zoomed_bounds.center().x.abs() < 1e-4);
        assert!(zoomed_bounds.center().y.abs() < 1e-4);
        // Auto bounds should be deactivated after zoom
        assert!(!mem2.auto_bounds.x);
        assert!(!mem2.auto_bounds.y);
    }

    #[test]
    fn test_zoom_out_twenty_percent_magnification() {
        let ctx = egui::Context::default();
        let id = Id::new("test_zoom_out");

        // Frame 1: Establish initial plot
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id)
                    .id(id)
                    .default_x_bounds(-10.0, 10.0)
                    .default_y_bounds(-5.0, 5.0)
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                    });
            });
        });

        let mem1 = PlotMemory::load(&ctx, id).expect("PlotMemory should exist");
        let initial_bounds = *mem1.transform.bounds();
        let initial_width = initial_bounds.width();
        let initial_height = initial_bounds.height();

        // Frame 2: Trigger Zoom Out signal
        ctx.data_mut(|d| d.insert_temp(id.with("zoom_out"), true));

        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id).id(id).show(ui, |plot_ui| {
                    plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                });
            });
        });

        let mem2 = PlotMemory::load(&ctx, id).expect("PlotMemory should exist");
        let zoomed_bounds = *mem2.transform.bounds();
        let zoomed_width = zoomed_bounds.width();
        let zoomed_height = zoomed_bounds.height();

        // 20% magnification decrease means factor 0.8: visible range divided by 0.8 = 25.0
        let expected_width = initial_width / 0.8;
        let expected_height = initial_height / 0.8;
        assert!(
            (zoomed_width - expected_width).abs() < 1e-4,
            "Zoom out width {zoomed_width} should match expected {expected_width} (20% magnification decrease)"
        );
        assert!(
            (zoomed_height - expected_height).abs() < 1e-4,
            "Zoom out height {zoomed_height} should match expected {expected_height}"
        );
        assert!(zoomed_bounds.center().x.abs() < 1e-4);
        assert!(zoomed_bounds.center().y.abs() < 1e-4);
        assert!(!mem2.auto_bounds.x);
        assert!(!mem2.auto_bounds.y);
    }

    #[test]
    fn test_home_reset_button_matches_double_click() {
        let ctx = egui::Context::default();
        let id_home = Id::new("test_home_reset");

        let dataset = vec![[0.0, 0.0], [50.0, 25.0], [100.0, 50.0]];

        // Frame 1: Initial auto bounds
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id_home).id(id_home).show(ui, |plot_ui| {
                    plot_ui.line(Line::new("line", dataset.clone()));
                });
            });
        });

        let initial_bounds = *PlotMemory::load(&ctx, id_home).unwrap().transform.bounds();

        // Frame 2: Zoom in to change bounds away from initial
        ctx.data_mut(|d| {
            d.insert_temp(id_home.with("zoom_in"), true);
        });

        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id_home).id(id_home).show(ui, |plot_ui| {
                    plot_ui.line(Line::new("line", dataset.clone()));
                });
            });
        });

        let zoomed_bounds = *PlotMemory::load(&ctx, id_home).unwrap().transform.bounds();
        assert!(zoomed_bounds.width() < initial_bounds.width());

        // Frame 3: Reset home via toolbar signal
        ctx.data_mut(|d| {
            d.insert_temp(id_home.with("reset_view"), true);
        });

        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id_home).id(id_home).show(ui, |plot_ui| {
                    plot_ui.line(Line::new("line", dataset.clone()));
                });
            });
        });

        let final_home = *PlotMemory::load(&ctx, id_home).unwrap().transform.bounds();

        assert!(
            (final_home.min[0] - initial_bounds.min[0]).abs() < 1e-4,
            "Home button reset should match initial auto data bounds"
        );
        assert!(
            (final_home.max[0] - initial_bounds.max[0]).abs() < 1e-4,
            "Home button reset should match initial auto data bounds"
        );
        assert!(
            (final_home.min[1] - initial_bounds.min[1]).abs() < 1e-4,
            "Home button reset should match initial auto data bounds"
        );
        assert!(
            (final_home.max[1] - initial_bounds.max[1]).abs() < 1e-4,
            "Home button reset should match initial auto data bounds"
        );

        let mem_after = PlotMemory::load(&ctx, id_home).unwrap();
        assert!(mem_after.auto_bounds.x);
        assert!(mem_after.auto_bounds.y);
    }

    #[test]
    fn test_zoom_locked_axes() {
        let ctx = egui::Context::default();
        let id = Id::new("test_locked_axes");

        // Lock Y axis: only X can zoom
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id)
                    .id(id)
                    .default_x_bounds(-10.0, 10.0)
                    .default_y_bounds(-5.0, 5.0)
                    .allow_zoom(Vec2b::new(true, false))
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                    });
            });
        });

        let mem1 = PlotMemory::load(&ctx, id).unwrap();
        let initial_bounds = *mem1.transform.bounds();

        // Zoom In
        ctx.data_mut(|d| d.insert_temp(id.with("zoom_in"), true));
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id)
                    .id(id)
                    .allow_zoom(Vec2b::new(true, false))
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                    });
            });
        });

        let mem2 = PlotMemory::load(&ctx, id).unwrap();
        let zoomed_bounds = *mem2.transform.bounds();

        // X width zoomed by 1.25
        assert!((zoomed_bounds.width() - initial_bounds.width() / 1.25).abs() < 1e-4);
        // Y height must remain identical since Y axis is locked
        assert!(
            (zoomed_bounds.height() - initial_bounds.height()).abs() < 1e-4,
            "Locked Y axis bounds must not change on zoom"
        );

        // Lock both axes: neither X nor Y can zoom
        ctx.data_mut(|d| d.insert_temp(id.with("zoom_in"), true));
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id)
                    .id(id)
                    .allow_zoom(Vec2b::new(false, false))
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                    });
            });
        });

        let mem3 = PlotMemory::load(&ctx, id).unwrap();
        let fully_locked_bounds = *mem3.transform.bounds();
        assert!((fully_locked_bounds.width() - zoomed_bounds.width()).abs() < 1e-4);
        assert!((fully_locked_bounds.height() - zoomed_bounds.height()).abs() < 1e-4);
    }

    #[test]
    fn test_zoom_disabled_state() {
        let ctx = egui::Context::default();
        let id = Id::new("test_disabled_zoom");

        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id)
                    .id(id)
                    .default_x_bounds(-10.0, 10.0)
                    .default_y_bounds(-5.0, 5.0)
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                    });
            });
        });

        let baseline = *PlotMemory::load(&ctx, id).unwrap().transform.bounds();

        // Frame with ui disabled
        ctx.data_mut(|d| d.insert_temp(id.with("zoom_in"), true));
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.disable();
                Plot::new(id).id(id).show(ui, |plot_ui| {
                    plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                });
            });
        });

        let after_disabled = *PlotMemory::load(&ctx, id).unwrap().transform.bounds();
        assert!(
            (after_disabled.width() - baseline.width()).abs() < 1e-4,
            "Zoom buttons must respect disabled ui state"
        );
    }

    #[test]
    fn test_plot_ui_zoom_around_center_and_coordinates_translation() {
        let ctx = egui::Context::default();
        let id = Id::new("test_plot_ui_zoom");

        let mut pt_screen = Pos2::ZERO;
        let mut pt_plot_initial = PlotPoint::new(0.0, 0.0);
        let mut pt_plot_zoomed = PlotPoint::new(0.0, 0.0);

        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id)
                    .id(id)
                    .default_x_bounds(-10.0, 10.0)
                    .default_y_bounds(-10.0, 10.0)
                    .show(ui, |plot_ui| {
                        let center_pos = plot_ui.transform().frame().center();
                        pt_screen = center_pos;
                        pt_plot_initial = plot_ui.plot_from_screen(center_pos);
                        // Translate center through plot_ui
                        let screen_back = plot_ui.screen_from_plot(pt_plot_initial);
                        assert!((screen_back.x - center_pos.x).abs() < 1e-4);
                        assert!((screen_back.y - center_pos.y).abs() < 1e-4);

                        // Programmatic zoom bounds around center using PlotUi API
                        plot_ui.zoom_bounds_around_center(Vec2::splat(1.2));
                    });
            });
        });

        assert!((pt_plot_initial.x).abs() < 1e-4);
        assert!((pt_plot_initial.y).abs() < 1e-4);

        // Frame 2: Check zoomed coordinate translation
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id).id(id).show(ui, |plot_ui| {
                    pt_plot_zoomed = plot_ui.plot_from_screen(pt_screen);
                    let bounds = plot_ui.plot_bounds();
                    assert!((bounds.width() - 20.0 / 1.2).abs() < 1e-4);
                });
            });
        });

        // The center point remains invariant under zoom around center
        assert!((pt_plot_zoomed.x).abs() < 1e-4);
        assert!((pt_plot_zoomed.y).abs() < 1e-4);
    }

    #[test]
    fn test_generate_floating_canvas_zoom_toolbar_screenshot() {
        let pts: Vec<[f64; 2]> = (0..100)
            .map(|i| {
                let x = i as f64 * 0.1;
                let y = (x * 1.5).sin() * (-0.1 * x).exp();
                [x, y]
            })
            .collect();
        let datasets = vec![("Damped Waveform".to_string(), pts)];
        let options = ExportOptions {
            width: 800,
            height: 480,
            line_thickness: 2.0,
            ..Default::default()
        };

        if let Ok(mut image) = export::export_custom_image(&datasets, &options) {
            let w = image.size[0];
            let h = image.size[1];
            // Overlay 4 toolbar buttons in top-right corner: Wrench, +, -, Home
            let btn_w = 24;
            let btn_h = 24;
            let x_start = w.saturating_sub(30);
            let y_offsets = [6, 34, 62, 90];
            let btn_bg = Color32::from_rgb(45, 50, 60);
            let btn_border = Color32::from_rgb(90, 100, 120);
            let icon_color = Color32::from_rgb(220, 225, 235);

            for (btn_idx, &y_start) in y_offsets.iter().enumerate() {
                for dy in 0..btn_h {
                    for dx in 0..btn_w {
                        let px = x_start + dx;
                        let py = y_start + dy;
                        if px < w && py < h {
                            let is_border =
                                dx == 0 || dx == btn_w - 1 || dy == 0 || dy == btn_h - 1;
                            let mut col = if is_border { btn_border } else { btn_bg };
                            // Simple crisp icon rendering
                            match btn_idx {
                                0 => {
                                    // Wrench icon shape
                                    if ((8..=15).contains(&dx) && (8..=15).contains(&dy))
                                        || (dx == dy && (10..=18).contains(&dx))
                                    {
                                        col = icon_color;
                                    }
                                }
                                1 => {
                                    // Plus icon: +
                                    if ((6..=17).contains(&dx) && (dy == 11 || dy == 12))
                                        || ((6..=17).contains(&dy) && (dx == 11 || dx == 12))
                                    {
                                        col = icon_color;
                                    }
                                }
                                2 => {
                                    // Minus icon: -
                                    if (6..=17).contains(&dx) && (dy == 11 || dy == 12) {
                                        col = icon_color;
                                    }
                                }
                                3 => {
                                    // Home icon: 🏠 (roof + house body)
                                    let mid_x = 11;
                                    let dist_x = (dx as isize - mid_x).abs();
                                    let roof_y = 6 + dist_x as usize;
                                    if (dy == roof_y && dy <= 12)
                                        || ((12..=18).contains(&dy) && (6..=16).contains(&dx))
                                    {
                                        col = icon_color;
                                    }
                                }
                                _ => {}
                            }
                            image.pixels[py * w + px] = col;
                        }
                    }
                }
            }

            if let Ok(png_bytes) = export::color32_image_to_png(&image) {
                let _ = std::fs::write("/tmp/floating_canvas_zoom_toolbar.png", png_bytes);
            }
        }
    }

    #[test]
    fn test_pointer_click_on_zoom_buttons() {
        let ctx = egui::Context::default();
        let id = Id::new("test_pointer_click");

        // Frame 1: Initial plot
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id)
                    .id(id)
                    .default_x_bounds(-10.0, 10.0)
                    .default_y_bounds(-5.0, 5.0)
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                    });
            });
        });

        let mem1 = PlotMemory::load(&ctx, id).unwrap();
        let frame = *mem1.transform.frame();
        let nav_rect = Rect::from_min_size(
            frame.right_top() + egui::vec2(-30.0, 34.0),
            egui::vec2(24.0, 80.0),
        );
        let plus_btn_pos = nav_rect.min + vec2(12.0, 12.0);

        // Frame 2: Pointer click on plus button
        let mut raw = egui::RawInput::default();
        raw.events.push(egui::Event::PointerButton {
            pos: plus_btn_pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::default(),
        });
        raw.events.push(egui::Event::PointerButton {
            pos: plus_btn_pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::default(),
        });

        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id).id(id).show(ui, |plot_ui| {
                    plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                });
            });
        });

        let mem2 = PlotMemory::load(&ctx, id).unwrap();
        let zoomed_bounds = *mem2.transform.bounds();
        let expected_w = 20.0 / 1.25;
        assert!(
            (zoomed_bounds.width() - expected_w).abs() < 1e-3,
            "Pointer click on plus button should zoom in by 20% (got {}, expected {})",
            zoomed_bounds.width(),
            expected_w
        );
    }

    #[test]
    fn test_canvas_wheel_scrolling_no_regression() {
        let ctx = egui::Context::default();
        let id = Id::new("test_wheel_scroll");

        // Frame 1: Establish initial plot
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id)
                    .id(id)
                    .default_x_bounds(-10.0, 10.0)
                    .default_y_bounds(-5.0, 5.0)
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                    });
            });
        });

        let mem1 = PlotMemory::load(&ctx, id).unwrap();
        let center_pos = mem1.transform.frame().center();

        // Frame 2: Wheel zoom event inside canvas (zoom_delta = 1.5)
        let mut raw = egui::RawInput::default();
        raw.events.push(egui::Event::PointerMoved(center_pos));
        raw.events.push(egui::Event::Zoom(1.5));

        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id).id(id).show(ui, |plot_ui| {
                    plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                });
            });
        });

        let mem2 = PlotMemory::load(&ctx, id).unwrap();
        let wheeled_bounds = *mem2.transform.bounds();
        // Width should be scaled by wheel zoom
        assert!(wheeled_bounds.width() < 20.0, "Wheel zoom should zoom in");
    }

    #[test]
    fn test_zoom_aspect_ratio_preservation() {
        let ctx = egui::Context::default();
        let id = Id::new("test_aspect_ratio");

        // Set data_aspect(2.0) and run initial frames so font metrics and axis margins settle
        let show_plot = |ctx: &egui::Context| {
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    Plot::new(id)
                        .id(id)
                        .data_aspect(2.0)
                        .default_x_bounds(-20.0, 20.0)
                        .default_y_bounds(-10.0, 10.0)
                        .show(ui, |plot_ui| {
                            plot_ui.line(Line::new("dummy", vec![[-20.0, -10.0], [20.0, 10.0]]));
                        });
                });
            });
        };
        show_plot(&ctx);
        show_plot(&ctx);

        let mem1 = PlotMemory::load(&ctx, id).unwrap();
        let initial_aspect = mem1.transform.bounds().width() / mem1.transform.bounds().height();

        // Zoom In
        ctx.data_mut(|d| d.insert_temp(id.with("zoom_in"), true));
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id).id(id).data_aspect(2.0).show(ui, |plot_ui| {
                    plot_ui.line(Line::new("dummy", vec![[-20.0, -10.0], [20.0, 10.0]]));
                });
            });
        });

        let mem2 = PlotMemory::load(&ctx, id).unwrap();
        let zoomed_aspect = mem2.transform.bounds().width() / mem2.transform.bounds().height();
        assert!(
            (zoomed_aspect - initial_aspect).abs() < 1e-4,
            "Aspect ratio must be preserved after zoom (initial: {}, zoomed: {})",
            initial_aspect,
            zoomed_aspect
        );
    }

    #[test]
    fn test_navigation_controls_hidden_keeps_wrench() {
        let ctx = egui::Context::default();
        let id = Id::new("test_nav_controls_hidden");

        // Render with show_navigation_controls(false)
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id)
                    .id(id)
                    .show_navigation_controls(false)
                    .default_x_bounds(-10.0, 10.0)
                    .default_y_bounds(-5.0, 5.0)
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                    });
            });
        });

        let mem = PlotMemory::load(&ctx, id).unwrap();
        let frame = *mem.transform.frame();
        // Wrench position: frame.right_top() + vec2(-30.0, 6.0)
        let wrench_pos = frame.right_top() + vec2(-18.0, 18.0);

        // Click wrench
        let mut raw = egui::RawInput::default();
        raw.events.push(egui::Event::PointerButton {
            pos: wrench_pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::default(),
        });
        raw.events.push(egui::Event::PointerButton {
            pos: wrench_pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::default(),
        });

        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id)
                    .id(id)
                    .show_navigation_controls(false)
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                    });
            });
        });

        // Wrench should still trigger toggle_settings and open modal even when nav controls are hidden
        let mem_after = PlotMemory::load(&ctx, id).expect("PlotMemory should exist");
        assert!(
            mem_after.show_settings_modal,
            "Wrench button must still be present and functional when show_navigation_controls is false"
        );
    }

    #[test]
    fn test_overlay_buttons_drag_isolation() {
        let ctx = egui::Context::default();
        let id = Id::new("test_drag_isolation");

        // Initial frame
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id)
                    .id(id)
                    .default_x_bounds(-10.0, 10.0)
                    .default_y_bounds(-5.0, 5.0)
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                    });
            });
        });

        let mem1 = PlotMemory::load(&ctx, id).unwrap();
        let initial_bounds = *mem1.transform.bounds();
        let frame = *mem1.transform.frame();

        // Drag starting on overlay button (plus button)
        let plus_btn_pos = frame.right_top() + vec2(-18.0, 46.0);
        let mut raw = egui::RawInput::default();
        raw.events.push(egui::Event::PointerMoved(plus_btn_pos));
        raw.events.push(egui::Event::PointerButton {
            pos: plus_btn_pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::default(),
        });
        // Move pointer 50 pixels while holding
        raw.events
            .push(egui::Event::PointerMoved(plus_btn_pos + vec2(50.0, 50.0)));

        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id).id(id).show(ui, |plot_ui| {
                    plot_ui.line(Line::new("dummy", vec![[-10.0, -5.0], [10.0, 5.0]]));
                });
            });
        });

        let mem2 = PlotMemory::load(&ctx, id).unwrap();
        let bounds_after_drag = *mem2.transform.bounds();

        // Plot should not have panned
        assert!(
            (bounds_after_drag.center().x - initial_bounds.center().x).abs() < 1e-4,
            "Interacting with overlay buttons must not capture drag events on the canvas"
        );
        assert!(
            (bounds_after_drag.center().y - initial_bounds.center().y).abs() < 1e-4,
            "Interacting with overlay buttons must not capture drag events on the canvas"
        );
    }
}

#[cfg(test)]
mod log_space_projection_tests {
    use super::*;

    fn collect_text_strings(shapes: &[epaint::ClippedShape], out: &mut Vec<String>) {
        for clipped in shapes {
            collect_shape_texts(&clipped.shape, out);
        }
    }

    fn collect_shape_texts(shape: &Shape, out: &mut Vec<String>) {
        match shape {
            Shape::Text(t) => out.push(t.galley.text().to_owned()),
            Shape::Vec(v) => {
                for s in v {
                    collect_shape_texts(s, out);
                }
            }
            _ => {}
        }
    }

    fn count_line_segments(shape: &Shape) -> usize {
        match shape {
            Shape::LineSegment { .. } => 1,
            Shape::Vec(v) => v.iter().map(count_line_segments).sum(),
            _ => 0,
        }
    }

    #[test]
    fn test_log_transform_equal_decade_spacing_and_round_trip() {
        let frame = Rect::from_min_size(Pos2::new(50.0, 20.0), vec2(600.0, 400.0));
        let bounds = PlotBounds::from_min_max([1.0, 0.01], [1000.0, 10_000.0]);
        let mut tf = PlotTransform::new(frame, bounds, Vec2b::FALSE);
        tf.set_log_axis(Vec2b::TRUE);

        let x1 = tf.position_from_point_x(1.0);
        let x10 = tf.position_from_point_x(10.0);
        let x100 = tf.position_from_point_x(100.0);
        let x1000 = tf.position_from_point_x(1000.0);
        assert!(((x10 - x1) - 200.0).abs() < 1e-3);
        assert!(((x100 - x10) - 200.0).abs() < 1e-3);
        assert!(((x1000 - x100) - 200.0).abs() < 1e-3);

        let y_a = tf.position_from_point_y(0.01);
        let y_b = tf.position_from_point_y(0.1);
        let y_c = tf.position_from_point_y(1.0);
        assert!(((y_a - y_b) - (y_b - y_c)).abs() < 1e-3);

        let pt = PlotPoint::new(31.622_776_6, 250.0);
        let back = tf.value_from_position(tf.position_from_point(&pt));
        assert!((back.x - pt.x).abs() / pt.x < 1e-5);
        assert!((back.y - pt.y).abs() / pt.y < 1e-5);
    }

    #[test]
    fn test_log_axes_retain_major_decade_ticks_and_grid_lines() {
        let ctx = egui::Context::default();
        let id = Id::new("log_decade_ticks_test");
        let raw = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(900.0, 650.0))),
            ..Default::default()
        };
        let _ = ctx.run(raw.clone(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(id).id(id).show(ui, |plot_ui| {
                    plot_ui.line(Line::new("span", vec![[0.001, 0.001], [1000.0, 1000.0]]));
                });
            });
        });
        let mut mem = PlotMemory::load(&ctx, id).unwrap();
        mem.log_x = true;
        mem.log_y = true;
        mem.store(&ctx, id);

        let mut out = egui::FullOutput::default();
        for _ in 0..2 {
            out = ctx.run(raw.clone(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    Plot::new(id).id(id).show(ui, |plot_ui| {
                        plot_ui.line(Line::new("span", vec![[0.001, 0.001], [1000.0, 1000.0]]));
                    });
                });
            });
        }
        let mut texts = Vec::new();
        collect_text_strings(&out.shapes, &mut texts);
        for expected in ["0.001", "0.01", "0.1", "1", "10", "100"] {
            let count = texts.iter().filter(|t| t.as_str() == expected).count();
            assert!(
                count >= 2,
                "Expected decade label '{expected}' on both X and Y axes, got {count} in {texts:?}"
            );
        }
        let seg_count: usize = out
            .shapes
            .iter()
            .map(|c| count_line_segments(&c.shape))
            .sum();
        assert!(
            seg_count >= 14,
            "Expected log grid lines on both axes, got {seg_count}"
        );
    }

    #[test]
    fn test_log_line_splitting_points_filtering_and_hover_snap() {
        let frame = Rect::from_min_size(Pos2::ZERO, vec2(400.0, 300.0));
        let bounds = PlotBounds::from_min_max([1.0, 1.0], [1000.0, 1000.0]);
        let mut tf = PlotTransform::new(frame, bounds, Vec2b::FALSE);
        tf.set_log_axis(Vec2b::TRUE);

        egui::__run_test_ui(|ui| {
            let mut line_shapes = Vec::new();
            let line = Line::new(
                "split",
                vec![
                    [1.0, 10.0],
                    [10.0, 50.0],
                    [0.0, 20.0],
                    [-5.0, 20.0],
                    [100.0, 200.0],
                    [500.0, 800.0],
                ],
            );
            line.shapes(ui, &tf, &mut line_shapes);
            let polylines = line_shapes
                .iter()
                .filter(|s| matches!(s, Shape::Path(_)))
                .count();
            assert_eq!(
                polylines, 2,
                "Line should split into 2 paths across non-positive gap"
            );

            let mut pt_shapes = Vec::new();
            let pts = Points::new(
                "pts",
                vec![[0.0, 10.0], [-1.0, 5.0], [10.0, 0.0], [10.0, 100.0]],
            )
            .radius(3.0_f32);
            pts.shapes(ui, &tf, &mut pt_shapes);
            let circles = pt_shapes
                .iter()
                .filter(|s| matches!(s, Shape::Circle(_)))
                .count();
            assert_eq!(
                circles, 1,
                "Only strictly positive points should emit markers"
            );

            let closest = pts.find_closest(tf.position_from_point(&PlotPoint::new(1.0, 10.0)), &tf);
            assert_eq!(
                closest.unwrap().index,
                3,
                "Hover snap must skip non-positive points"
            );
        });
    }

    #[test]
    fn test_log_bounds_include_zero_center_axis_and_toggle_reset() {
        let ctx = egui::Context::default();
        let id = Id::new("capacity_fade_log_bounds");
        let series = vec![[0.0, 1.0], [10.0, 0.95], [100.0, 0.85], [1000.0, 0.72]];
        let run_frame = |ctx: &egui::Context| {
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    Plot::new(id)
                        .id(id)
                        .include_y(0.0)
                        .include_y(1.0)
                        .center_x_axis(true)
                        .center_y_axis(true)
                        .show(ui, |plot_ui| {
                            plot_ui.line(Line::new("fade", series.clone()));
                        });
                });
            });
        };
        run_frame(&ctx);
        let mut mem = PlotMemory::load(&ctx, id).unwrap();
        assert!(
            mem.transform.bounds().min()[0] < 0.0,
            "Linear center_x_axis has negative min"
        );

        mem.log_x = true;
        mem.log_y = true;
        mem.auto_bounds = Vec2b::FALSE;
        mem.store(&ctx, id);
        run_frame(&ctx);

        let log_mem = PlotMemory::load(&ctx, id).unwrap();
        let b = log_mem.transform.bounds();
        assert!(
            b.min()[0] > 5.0 && b.min()[0] <= 10.0,
            "Log X should frame positive x>=10, got {:?}",
            b
        );
        assert!(
            b.min()[1] > 0.65 && b.min()[1] <= 0.72,
            "Log Y should ignore include_y(0.0), got {:?}",
            b
        );

        let mut back = log_mem;
        back.log_x = false;
        back.log_y = false;
        back.store(&ctx, id);
        run_frame(&ctx);
        let lin_mem = PlotMemory::load(&ctx, id).unwrap();
        assert!(lin_mem.auto_bounds.x && lin_mem.auto_bounds.y);
        assert!(lin_mem.transform.bounds().min()[0] < 0.0);
    }

    #[test]
    fn test_log_decade_pan_zoom_and_exponent_clamping() {
        let frame = Rect::from_min_size(Pos2::ZERO, vec2(300.0, 300.0));
        let bounds = PlotBounds::from_min_max([1.0, 1.0], [1000.0, 1000.0]);
        let mut tf = PlotTransform::new(frame, bounds, Vec2b::FALSE);
        tf.set_log_axis(Vec2b::TRUE);

        tf.translate_bounds((100.0, 0.0));
        assert!((tf.bounds().min()[0] - 10.0).abs() < 1e-6);
        assert!((tf.bounds().max()[0] - 10_000.0).abs() < 1e-3);

        tf.zoom(vec2(2.0, 2.0), frame.center());
        let span_decades = tf.bounds().max()[0].log10() - tf.bounds().min()[0].log10();
        assert!((span_decades - 1.5).abs() < 1e-5);

        for _ in 0..40 {
            tf.zoom(vec2(0.1, 0.1), frame.center());
            tf.translate_bounds((1e6, -1e6));
        }
        assert!(tf.bounds().is_valid());
        assert!(tf.bounds().min()[0] >= 1e-300 && tf.bounds().max()[0] <= 1e300);
        assert!(tf.bounds().min()[1] >= 1e-300 && tf.bounds().max()[1] <= 1e300);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PoiCacheKey {
    pub(crate) items_hash: u64,
    pub(crate) bounds_bits: [u64; 4],
    pub(crate) log_x: bool,
    pub(crate) log_y: bool,
}

#[derive(Clone)]
pub(crate) struct AccessState {
    pub(crate) show_table: bool,
    pub(crate) data_table: data_table::DataTableState,
    pub(crate) export_modal: export_modal::ExportModalState,
    pub(crate) focused_item: usize,
    pub(crate) focused_point: usize,
    pub(crate) focused_poi: Option<usize>,
    pub(crate) fit_model: Option<FitType>,
    pub(crate) fit_log_mode: bool,
    pub(crate) show_fit_residuals: bool,
    pub(crate) fit_dataset_idx: usize,
    pub(crate) show_fit_overlay: bool,
    pub(crate) custom_fit_formula: String,
    pub(crate) custom_fit_seeds_text: String,
    pub(crate) custom_fit_error: Option<String>,
    pub(crate) cached_fit: Option<(fit_overlay::FitCacheKey, Option<FitResult>)>,
    pub(crate) cached_pois: Option<(PoiCacheKey, Arc<[poi::CriticalPoint]>)>,
    pub(crate) cached_base_datasets: BaseDatasetCache,
    pub(crate) show_settings_modal: bool,
    pub(crate) log_x: bool,
    pub(crate) log_y: bool,
    pub(crate) grid_density: f32,
    pub(crate) show_x_labels: bool,
    pub(crate) show_y_labels: bool,
    pub(crate) show_hover: bool,
    pub(crate) clamp_grid: bool,
    pub(crate) show_add_callout_modal: bool,
    pub(crate) new_callout_point: [f64; 2],
    pub(crate) new_callout_text: String,
    pub(crate) new_callout_offset: [f32; 2],
    pub(crate) last_context_plot_pos: Option<[f64; 2]>,
    pub(crate) show_edit_callout_modal: bool,
    pub(crate) editing_callout_idx: Option<usize>,
    pub(crate) edit_callout_text: String,
    pub(crate) edit_callout_offset: [f32; 2],
    pub(crate) import_toast: Option<(String, bool)>,
    pub(crate) last_fit_error_announced: Option<(FitType, usize, String)>,
}

impl Default for AccessState {
    fn default() -> Self {
        Self {
            show_table: false,
            data_table: data_table::DataTableState::default(),
            export_modal: export_modal::ExportModalState::default(),
            focused_item: 0,
            focused_point: 0,
            focused_poi: None,
            fit_model: None,
            fit_log_mode: false,
            show_fit_residuals: false,
            fit_dataset_idx: 0,
            show_fit_overlay: true,
            custom_fit_formula: "a * exp(-b * x) + c".to_owned(),
            custom_fit_seeds_text: String::new(),
            custom_fit_error: None,
            cached_fit: None,
            cached_pois: None,
            cached_base_datasets: None,
            show_settings_modal: false,
            log_x: false,
            log_y: false,
            grid_density: 8.0,
            show_x_labels: true,
            show_y_labels: true,
            show_hover: true,
            clamp_grid: false,
            show_add_callout_modal: false,
            new_callout_point: [0.0, 0.0],
            new_callout_text: String::new(),
            new_callout_offset: [30.0, -30.0],
            last_context_plot_pos: None,
            show_edit_callout_modal: false,
            editing_callout_idx: None,
            edit_callout_text: String::new(),
            edit_callout_offset: [30.0, -30.0],
            import_toast: None,
            last_fit_error_announced: None,
        }
    }
}

#[cfg(test)]
mod ambient_link_tests;
#[cfg(test)]
mod docked_header_and_fit_tests;
#[cfg(test)]
mod modal_lifecycle_tests;
#[cfg(test)]
mod snapshot_tests;
