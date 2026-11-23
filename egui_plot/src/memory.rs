use std::collections::BTreeMap;

use egui::{Context, Id, Pos2, Vec2b};

use crate::{PlotBounds, PlotTransform};

/// Information about the plot that has to persist between frames.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone)]
pub struct PlotMemory {
    /// Indicates if the plot uses automatic bounds.
    ///
    /// This is set to `false` whenever the user modifies
    /// the bounds, for example by moving or zooming.
    pub auto_bounds: Vec2b,

    /// Hovered legend item if any.
    pub hovered_legend_item: Option<Id>,

    /// Which items _not_ to show?
    pub hidden_items: ahash::HashSet<Id>,

    /// The transform from last frame.
    pub(crate) transform: PlotTransform,

    /// Allows to remember the first click position when performing a boxed zoom
    pub(crate) last_click_pos_for_zoom: Option<Pos2>,

    /// The thickness of each of the axes the previous frame.
    ///
    /// This is used in the next frame to make the axes thicker
    /// in order to fit the labels, if necessary.
    pub(crate) x_axis_thickness: BTreeMap<usize, f32>,
    pub(crate) y_axis_thickness: BTreeMap<usize, f32>,

    /// Persisted imported datasets overlaid on the plot.
    pub imported_datasets: Vec<crate::export::ImportedDataset>,

    /// Persisted callout annotations overlaid on the plot.
    #[cfg_attr(feature = "serde", serde(default))]
    pub callouts: Vec<crate::SavedCallout>,

    /// Toggle logarithmic scale transformation on X-axis.
    #[cfg_attr(feature = "serde", serde(default))]
    pub log_x: bool,

    /// Toggle logarithmic scale transformation on Y-axis.
    #[cfg_attr(feature = "serde", serde(default))]
    pub log_y: bool,

    /// Minimum grid line spacing in pixels (grid density).
    #[cfg_attr(feature = "serde", serde(default = "default_grid_density"))]
    pub grid_density: f32,

    /// Visibility of X-axis labels.
    #[cfg_attr(feature = "serde", serde(default = "default_true"))]
    pub show_x_labels: bool,

    /// Visibility of Y-axis labels.
    #[cfg_attr(feature = "serde", serde(default = "default_true"))]
    pub show_y_labels: bool,

    /// Visibility of hover crosshairs, point tooltips, and keyboard point focus popups.
    #[cfg_attr(feature = "serde", serde(default = "default_true"))]
    pub show_hover: bool,

    /// Whether the docked plot action header bar is collapsed.
    #[cfg_attr(feature = "serde", serde(default))]
    pub toolbar_collapsed: bool,

    /// Clamp grid lines to actual data range.
    #[cfg_attr(feature = "serde", serde(default))]
    pub clamp_grid: bool,

    /// Grid line visibility per axis.
    #[cfg_attr(feature = "serde", serde(default = "default_show_grid"))]
    pub show_grid: Vec2b,

    /// Toggle visibility of plot settings modal window.
    #[cfg_attr(feature = "serde", serde(default))]
    pub show_settings_modal: bool,

    /// Last right-click position captured in plot coordinates.
    pub last_right_click_coord: Option<[f64; 2]>,
}

/// Persisted canvas callout annotation data.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct CanvasCallout {
    pub target_point: [f64; 2],
    pub label_offset: [f32; 2],
    pub text: String,
}

impl From<CanvasCallout> for crate::SavedCallout {
    fn from(c: CanvasCallout) -> Self {
        Self::new(c.target_point, c.text, c.label_offset)
    }
}

#[cfg(feature = "serde")]
#[inline]
fn default_grid_density() -> f32 {
    8.0
}

#[cfg(feature = "serde")]
#[inline]
fn default_true() -> bool {
    true
}

#[cfg(feature = "serde")]
#[inline]
fn default_show_grid() -> Vec2b {
    Vec2b::new(true, true)
}

impl PlotMemory {
    #[inline]
    pub fn transform(&self) -> PlotTransform {
        self.transform
    }

    #[inline]
    pub fn set_transform(&mut self, t: PlotTransform) {
        self.transform = t;
    }

    /// Plot-space bounds.
    #[inline]
    pub fn bounds(&self) -> &PlotBounds {
        self.transform.bounds()
    }

    /// Plot-space bounds.
    #[inline]
    pub fn set_bounds(&mut self, bounds: PlotBounds) {
        self.transform.set_bounds(bounds);
    }
}

#[cfg(feature = "serde")]
impl PlotMemory {
    pub fn load(ctx: &Context, id: Id) -> Option<Self> {
        ctx.data_mut(|d| d.get_persisted(id))
    }

    pub fn store(self, ctx: &Context, id: Id) {
        ctx.data_mut(|d| d.insert_persisted(id, self));
    }
}

#[cfg(not(feature = "serde"))]
impl PlotMemory {
    pub fn load(ctx: &Context, id: Id) -> Option<Self> {
        ctx.data_mut(|d| d.get_temp(id))
    }

    pub fn store(self, ctx: &Context, id: Id) {
        ctx.data_mut(|d| d.insert_temp(id, self));
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use egui::{Id, Rect, Vec2b, pos2};

    use super::{CanvasCallout, PlotMemory};
    use crate::{CriticalPoint, CriticalPointKind, PlotBounds, PlotTransform, SavedCallout};

    #[test]
    fn critical_point_and_plot_memory_roundtrip_json() {
        let cp = CriticalPoint {
            kind: CriticalPointKind::Intersection,
            point: [1.25, -0.5],
            primary_series: "f(x)".to_owned(),
            secondary_series: Some("g(x)".to_owned()),
            item_id: Id::new("series_fx"),
        };
        let cp_json = serde_json::to_string(&cp).expect("serialize CriticalPoint");
        let cp_decoded: CriticalPoint =
            serde_json::from_str(&cp_json).expect("deserialize CriticalPoint");
        assert_eq!(cp_decoded, cp);

        let callout = CanvasCallout {
            target_point: [2.0, 3.0],
            label_offset: [12.0, -8.0],
            text: "Peak".to_owned(),
        };
        let memory = PlotMemory {
            auto_bounds: Vec2b::new(true, false),
            hovered_legend_item: Some(Id::new("legend_item")),
            hidden_items: std::iter::once(Id::new("hidden_series")).collect(),
            transform: PlotTransform::new(
                Rect::from_min_max(pos2(0.0, 0.0), pos2(200.0, 100.0)),
                PlotBounds::from_min_max([-5.0, -2.0], [5.0, 10.0]),
                true,
            ),
            last_click_pos_for_zoom: None,
            x_axis_thickness: [(0, 18.0)].into_iter().collect(),
            y_axis_thickness: [(0, 24.0)].into_iter().collect(),
            imported_datasets: Vec::new(),
            callouts: vec![SavedCallout::from(callout)],
            log_x: false,
            log_y: true,
            grid_density: 10.0,
            show_x_labels: true,
            show_y_labels: false,
            show_hover: false,
            toolbar_collapsed: true,
            clamp_grid: true,
            show_grid: Vec2b::new(true, false),
            show_settings_modal: false,
            last_right_click_coord: Some([1.5, 2.5]),
        };
        let mem_json = serde_json::to_string(&memory).expect("serialize PlotMemory");
        let mem_decoded: PlotMemory =
            serde_json::from_str(&mem_json).expect("deserialize PlotMemory");
        assert_eq!(mem_decoded.auto_bounds, memory.auto_bounds);
        assert_eq!(mem_decoded.bounds(), memory.bounds());
        assert_eq!(mem_decoded.callouts, memory.callouts);
        assert_eq!(mem_decoded.log_y, memory.log_y);
        assert_eq!(mem_decoded.show_y_labels, memory.show_y_labels);
        assert_eq!(mem_decoded.show_hover, memory.show_hover);
        assert_eq!(mem_decoded.toolbar_collapsed, memory.toolbar_collapsed);
        assert_eq!(
            mem_decoded.last_right_click_coord,
            memory.last_right_click_coord
        );
    }
}
