//! Canvas critical point marker rendering, hover snapping, callout pinning, and keyboard cycling.

use egui::{Color32, Pos2, Response, Shape, Stroke, Ui};
use emath::Float as _;

use crate::{
    AccessState, Cursor, a11y,
    commands::CommandRegistryData,
    emit_plot_live_message,
    items::{PlotPoint, SavedCallout},
    memory::PlotMemory,
    poi::CriticalPoint,
    transform::PlotTransform,
};

/// Paints subtle high-contrast Desmos-style gray dots on all visible critical points.
pub(crate) fn paint_critical_point_markers(
    ui: &Ui,
    transform: &PlotTransform,
    critical_points: &[CriticalPoint],
    shapes: &mut Vec<Shape>,
) {
    if critical_points.is_empty() {
        return;
    }
    let (fill, ring) = if ui.visuals().dark_mode {
        (
            Color32::from_rgb(148, 154, 166),
            Color32::from_rgb(28, 30, 36),
        )
    } else {
        (
            Color32::from_rgb(108, 114, 126),
            Color32::from_rgb(248, 249, 252),
        )
    };
    for poi in critical_points {
        let pt = PlotPoint::new(poi.point[0], poi.point[1]);
        if !transform.is_point_valid_for_scale(&pt) {
            continue;
        }
        let pos = transform.position_from_point(&pt);
        if pos.is_finite() {
            shapes.push(Shape::circle_filled(pos, 3.6_f32, fill));
            shapes.push(Shape::circle_stroke(
                pos,
                3.6_f32,
                Stroke::new(1.2_f32, ring),
            ));
        }
    }
}

/// Finds the nearest critical point marker within `snap_dist_sq` screen pixels squared.
pub(crate) fn find_closest_poi(
    critical_points: &[CriticalPoint],
    pointer: Pos2,
    transform: &PlotTransform,
    snap_dist_sq: f32,
) -> Option<(usize, Pos2)> {
    critical_points
        .iter()
        .enumerate()
        .filter_map(|(idx, poi)| {
            let pt = PlotPoint::new(poi.point[0], poi.point[1]);
            if !transform.is_point_valid_for_scale(&pt) {
                return None;
            }
            let pos = transform.position_from_point(&pt);
            let dist_sq = pointer.distance_sq(pos);
            (pos.is_finite() && dist_sq <= snap_dist_sq).then_some((idx, pos, dist_sq))
        })
        .min_by_key(|&(_, _, d)| d.ord())
        .map(|(idx, pos, _)| (idx, pos))
}

/// Renders an active hovered or keyboard-focused critical point marker, crosshairs, and tooltip.
pub(crate) fn render_active_poi(
    ui: &Ui,
    plot_area_response: &Response,
    transform: &PlotTransform,
    show_x: bool,
    show_y: bool,
    poi: &CriticalPoint,
    keyboard_focused: bool,
    shapes: &mut Vec<Shape>,
) -> (Vec<Cursor>, PlotPoint) {
    let plot_pt = PlotPoint::new(poi.point[0], poi.point[1]);
    let screen_pos = transform.position_from_point(&plot_pt);
    let fill = if ui.visuals().dark_mode {
        Color32::from_rgb(185, 192, 205)
    } else {
        Color32::from_rgb(82, 88, 100)
    };
    let outline = if keyboard_focused {
        a11y::focus_ring_stroke(ui.visuals(), ui.visuals().extreme_bg_color)
    } else {
        Stroke::new(2.0_f32, ui.visuals().strong_text_color())
    };
    shapes.push(Shape::circle_filled(screen_pos, 5.2_f32, fill));
    shapes.push(Shape::circle_stroke(screen_pos, 7.2_f32, outline));

    let mut cursors = Vec::new();
    if show_x {
        cursors.push(Cursor::Vertical { x: poi.point[0] });
    }
    if show_y {
        cursors.push(Cursor::Horizontal { y: poi.point[1] });
    }
    a11y::show_critical_point_tooltip(
        plot_area_response,
        screen_pos,
        poi,
        !keyboard_focused,
        !keyboard_focused,
    );
    (cursors, plot_pt)
}

/// Handles `Alt+ArrowLeft` and `Alt+ArrowRight` critical point cycling on a focused plot.
pub(crate) fn handle_focused_plot_poi_keys(
    ui: &Ui,
    critical_points: &[CriticalPoint],
    access_state: &mut AccessState,
) {
    let total = critical_points.len();
    if total == 0 {
        access_state.focused_poi = None;
        return;
    }
    let Some(step) = CommandRegistryData::check_poi_navigation_step(ui.ctx(), ui) else {
        if access_state.focused_poi.is_some_and(|idx| idx >= total) {
            access_state.focused_poi = Some(total - 1);
        }
        return;
    };
    let next_idx = match access_state.focused_poi {
        Some(cur) if cur < total => ((cur as isize + step).rem_euclid(total as isize)) as usize,
        _ if step >= 0 => 0,
        _ => total - 1,
    };
    access_state.focused_poi = Some(next_idx);
    let poi = &critical_points[next_idx];
    let msg = a11y::format_critical_point_live_summary(poi, next_idx, total);
    emit_plot_live_message(ui.ctx(), msg, false);
}

/// Pins a persistent [`SavedCallout`] into [`PlotMemory::callouts`] when clicking a critical point.
pub(crate) fn try_pin_clicked_poi(
    ui: &Ui,
    click_pos: Pos2,
    hovered_poi_idx: Option<usize>,
    critical_points: &[CriticalPoint],
    transform: &PlotTransform,
    snap_distance: f32,
    mem: &mut PlotMemory,
) -> bool {
    let idx = hovered_poi_idx.or_else(|| {
        find_closest_poi(critical_points, click_pos, transform, snap_distance.powi(2))
            .map(|(i, _)| i)
    });
    let Some(poi) = idx.and_then(|i| critical_points.get(i)) else {
        return false;
    };
    let already_pinned = mem.callouts.iter().any(|c| {
        (c.point[0] - poi.point[0]).abs() <= 1e-9 && (c.point[1] - poi.point[1]).abs() <= 1e-9
    });
    if !already_pinned {
        let label = poi.callout_label();
        mem.callouts.push(SavedCallout {
            point: poi.point,
            label: label.clone(),
            offset: [30.0, -30.0],
        });
        emit_plot_live_message(ui.ctx(), format!("Pinned callout: {label}"), false);
        ui.ctx().request_repaint();
    }
    true
}
