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

    /// Clamp grid lines to actual data range.
    #[cfg_attr(feature = "serde", serde(default))]
    pub clamp_grid: bool,

    /// Grid line visibility per axis.
    #[cfg_attr(feature = "serde", serde(default = "default_show_grid"))]
    pub show_grid: Vec2b,

    /// Toggle visibility of plot settings modal window.
    #[cfg_attr(feature = "serde", serde(default))]
    pub show_settings_modal: bool,
}

#[allow(dead_code)]
#[inline]
fn default_grid_density() -> f32 {
    8.0
}

#[allow(dead_code)]
#[inline]
fn default_true() -> bool {
    true
}

#[allow(dead_code)]
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
