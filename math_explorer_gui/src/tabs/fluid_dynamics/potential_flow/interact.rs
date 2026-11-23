//! Direct singularity handle rendering, hit-testing, click-to-select/place,
//! and continuous drag-handle steering for [`super::PotentialFlowTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    ElementType, PotentialFlowTool, SerializedFlowElement, FLOW_POS_MAX, FLOW_POS_MIN,
    FLOW_STRENGTH_MAX, FLOW_STRENGTH_MIN,
};
use crate::accessibility::announce_live_status;
use crate::app::SemanticPalette;
use crate::framework::{BrushMode, BrushStroke, InteractionContext, MAX_JSON_ARRAY_LEN};
use eframe::egui::Color32;
use egui_plot::{MarkerShape, PlotUi, Points};
use math_commons::math_kernel::colormap::ensure_dark_canvas_contrast;

fn contrast_safe_color(base: Color32, dark_mode: bool) -> Color32 {
    if dark_mode {
        let rgb = ensure_dark_canvas_contrast([base.r(), base.g(), base.b()]);
        Color32::from_rgb(rgb[0], rgb[1], rgb[2])
    } else {
        base
    }
}

fn point_to_segment_dist_sq(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len_sq = dx * dx + dy * dy;
    if len_sq <= 1e-12 {
        return (p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2);
    }
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq).clamp(0.0, 1.0);
    let proj_x = a[0] + t * dx;
    let proj_y = a[1] + t * dy;
    (p[0] - proj_x).powi(2) + (p[1] - proj_y).powi(2)
}

impl PotentialFlowTool {
    /// Renders high-contrast interactive singularity markers (`Source`, `Sink`, `Vortex`, `Doublet`).
    pub(super) fn render_singularity_handles(
        &self,
        plot_ui: &mut PlotUi<'_>,
        palette: &SemanticPalette,
        dark_mode: bool,
    ) {
        let specs = [
            (
                ElementType::Source,
                "Sources (+q)",
                MarkerShape::Circle,
                6.5_f32,
                contrast_safe_color(palette.success, dark_mode),
            ),
            (
                ElementType::Sink,
                "Sinks (-q)",
                MarkerShape::Square,
                6.5_f32,
                contrast_safe_color(palette.error, dark_mode),
            ),
            (
                ElementType::Vortex,
                "Vortices (Γ)",
                MarkerShape::Plus,
                7.5_f32,
                contrast_safe_color(palette.accent, dark_mode),
            ),
            (
                ElementType::Doublet,
                "Doublets (κ)",
                MarkerShape::Up,
                7.0_f32,
                contrast_safe_color(palette.info, dark_mode),
            ),
        ];

        for (etype, label, shape, radius, color) in specs {
            let pts: Vec<[f64; 2]> = self
                .elements
                .iter()
                .filter(|el| el.element_type == etype)
                .map(|el| [el.x, el.y])
                .collect();
            if !pts.is_empty() {
                plot_ui.points(Points::new(label, pts).shape(shape).radius(radius).color(color));
            }
        }
    }

    /// Finds the closest non-`Uniform` singularity handle within `capture_radius` of `pt`.
    #[must_use]
    pub(crate) fn find_nearest_singularity(&self, pt: [f64; 2], capture_radius: f64) -> Option<usize> {
        let r_sq = capture_radius.max(0.5).powi(2);
        let mut best: Option<(usize, f64)> = None;
        for (idx, el) in self.elements.iter().enumerate() {
            if el.element_type == ElementType::Uniform {
                continue;
            }
            let d_sq = (el.x - pt[0]).powi(2) + (el.y - pt[1]).powi(2);
            if d_sq <= r_sq && best.is_none_or(|(_, best_d)| d_sq <= best_d) {
                best = Some((idx, d_sq));
            }
        }
        best.map(|(idx, _)| idx)
    }

    pub(crate) fn sync_sidebar_from_element(&mut self, idx: usize) {
        let Some(el) = self.elements.get(idx) else {
            return;
        };
        if el.element_type == ElementType::Uniform {
            return;
        }
        self.selected_element_type = el.element_type;
        self.new_element_strength = el.strength;
        self.new_element_x = el.x;
        self.new_element_y = el.y;
    }

    fn resolve_brush_element_type(&self, mode: BrushMode) -> ElementType {
        match mode {
            BrushMode::FlowSource => {
                if self.new_element_strength < 0.0 {
                    ElementType::Sink
                } else {
                    ElementType::Source
                }
            }
            BrushMode::FlowSink => ElementType::Sink,
            BrushMode::FlowVortex => ElementType::Vortex,
            BrushMode::FlowDoublet => ElementType::Doublet,
            _ => match self.selected_element_type {
                ElementType::Uniform => ElementType::Source,
                other => other,
            },
        }
    }

    /// Handles pointer click selection of existing singularities (including while in `Pan` mode).
    pub(crate) fn handle_flow_click(&mut self, ic: &InteractionContext<'_>) {
        let Some(pos) = ic.pointer_pos else {
            return;
        };
        let pt = [f64::from(pos.x), f64::from(pos.y)];
        if let Some(idx) = self.find_nearest_singularity(pt, self.brush_kit.radius.max(0.75)) {
            self.sync_sidebar_from_element(idx);
            let el = &self.elements[idx];
            announce_live_status(
                &ic.response.ctx,
                format!(
                    "Selected {} at ({:.1}, {:.1})",
                    el.element_type.label(),
                    el.x,
                    el.y
                ),
            );
        }
    }

    /// Handles continuous singularity dragging when `active_drag_element` is locked.
    pub(crate) fn handle_flow_drag(&mut self, ic: &InteractionContext<'_>) {
        if ic.brush_stroke.is_some() {
            return;
        }
        let Some(idx) = self
            .active_drag_element
            .filter(|&i| i < self.elements.len() && self.elements[i].element_type != ElementType::Uniform)
        else {
            return;
        };
        let Some(pos) = ic.pointer_pos else {
            return;
        };
        let px = f64::from(pos.x).clamp(FLOW_POS_MIN, FLOW_POS_MAX);
        let py = f64::from(pos.y).clamp(FLOW_POS_MIN, FLOW_POS_MAX);
        if (self.elements[idx].x - px).abs() > 1e-9 || (self.elements[idx].y - py).abs() > 1e-9 {
            self.elements[idx].x = px;
            self.elements[idx].y = py;
            self.sync_sidebar_from_element(idx);
            self.rebuild_field();
        }
    }

    /// Applies a [`BrushStroke`] from pointer drag or keyboard stamp, grabbing and steering
    /// existing `Source`/`Sink`/`Vortex`/`Doublet` handles or placing new ones within `[-50, 50]`.
    pub(crate) fn apply_flow_brush_context(&mut self, ic: &InteractionContext<'_>) {
        let Some(ref stroke) = ic.brush_stroke else {
            return;
        };
        self.apply_flow_stroke_with_drag(stroke, ic.is_dragging);
    }

    pub(crate) fn apply_flow_stroke_with_drag(&mut self, stroke: &BrushStroke, is_dragging: bool) {
        let px = stroke.end[0].clamp(FLOW_POS_MIN, FLOW_POS_MAX);
        let py = stroke.end[1].clamp(FLOW_POS_MIN, FLOW_POS_MAX);
        let r_sq = stroke.radius.max(0.5).powi(2);

        if stroke.mode == BrushMode::Erase {
            self.active_drag_element = None;
            let before = self.elements.len();
            self.elements.retain(|el| {
                el.element_type == ElementType::Uniform
                    || point_to_segment_dist_sq([el.x, el.y], stroke.start, [px, py]) > r_sq
            });
            if self.elements.len() != before {
                self.rebuild_field();
            }
            return;
        }

        let hit_idx = self
            .active_drag_element
            .filter(|&i| i < self.elements.len() && self.elements[i].element_type != ElementType::Uniform)
            .or_else(|| self.find_nearest_singularity(stroke.start, stroke.radius));

        if let Some(idx) = hit_idx {
            self.elements[idx].x = px;
            self.elements[idx].y = py;
            self.sync_sidebar_from_element(idx);
            if is_dragging {
                self.active_drag_element = Some(idx);
            }
            self.rebuild_field();
            return;
        }

        if self.elements.len() < MAX_JSON_ARRAY_LEN {
            let target_type = self.resolve_brush_element_type(stroke.mode);
            let strength = self
                .new_element_strength
                .abs()
                .max(1.0)
                .clamp(FLOW_STRENGTH_MIN, FLOW_STRENGTH_MAX);
            self.elements.push(SerializedFlowElement {
                element_type: target_type,
                strength,
                x: px,
                y: py,
                angle: 0.0,
            });
            let new_idx = self.elements.len() - 1;
            self.sync_sidebar_from_element(new_idx);
            if is_dragging {
                self.active_drag_element = Some(new_idx);
            }
            self.rebuild_field();
        }
    }
}
