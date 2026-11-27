#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! 3D wireframe and racer population helpers for [`super::LossLandscapeCache`].

use crate::app::SemanticPalette;
use crate::camera::Scene3DBuilder;
use eframe::egui::Stroke;
use egui_plot::{LineStyle, MarkerShape};
use nalgebra::Point3;

use super::super::surfaces::{lerp_color, SurfaceGrid};
use super::super::LossLandscapeTool;

pub(super) fn populate_3d_wireframe(
    scene: &mut Scene3DBuilder,
    grid: &SurfaceGrid,
    palette: &SemanticPalette,
) {
    let num_rows = grid.rows.len();
    let num_cols = grid.rows.first().map_or(0, Vec::len);
    if num_rows == 0 || num_cols == 0 {
        return;
    }
    let mut all_render = Vec::with_capacity(num_rows * num_cols);
    let mut all_world = Vec::with_capacity(num_rows * num_cols);
    for row in &grid.rows {
        let mut r_pts = Vec::with_capacity(num_cols);
        let mut w_pts = Vec::with_capacity(num_cols);
        let mut h_sum = 0.0;
        for &[w, b, loss] in row {
            let rp = Point3::from(grid.world_point(w, b, loss, 0.0));
            all_render.push(rp);
            all_world.push([w, b, loss]);
            r_pts.push(rp);
            w_pts.push([w, b, loss]);
            h_sum += grid.normalized_height(loss);
        }
        let color = lerp_color(palette.axis_z, palette.axis_x, h_sum / (num_cols as f64));
        scene.add_polyline_with_world(
            "Surface Mesh",
            &r_pts,
            &w_pts,
            Stroke::new(1.1_f32, color),
            LineStyle::Solid,
            false,
        );
    }
    populate_3d_columns(scene, grid, num_rows, num_cols, palette);
    scene.record_series_with_world("Loss Surface", &all_render, &all_world, palette.axis_z);
}

fn populate_3d_columns(
    scene: &mut Scene3DBuilder,
    grid: &SurfaceGrid,
    num_rows: usize,
    num_cols: usize,
    palette: &SemanticPalette,
) {
    let mut r_pts = Vec::with_capacity(num_rows);
    let mut w_pts = Vec::with_capacity(num_rows);
    for col in 0..num_cols {
        r_pts.clear();
        w_pts.clear();
        let mut h_sum = 0.0;
        for row in &grid.rows {
            let [w, b, loss] = row[col];
            r_pts.push(Point3::from(grid.world_point(w, b, loss, 0.0)));
            w_pts.push([w, b, loss]);
            h_sum += grid.normalized_height(loss);
        }
        let color = lerp_color(palette.axis_z, palette.axis_x, h_sum / (num_rows as f64));
        scene.add_polyline_with_world(
            "Surface Mesh",
            &r_pts,
            &w_pts,
            Stroke::new(1.1_f32, color),
            LineStyle::Solid,
            false,
        );
    }
}

pub(super) fn populate_3d_overlays(
    tool: &LossLandscapeTool,
    scene: &mut Scene3DBuilder,
    grid: &SurfaceGrid,
    palette: &SemanticPalette,
) {
    let eval = tool.surface_evaluator();
    let minima = eval.global_minima();
    if !minima.is_empty() {
        let mut m_render = Vec::with_capacity(minima.len());
        let mut m_world = Vec::with_capacity(minima.len());
        for &[mw, mb] in minima {
            let mloss = eval.evaluate(mw, mb);
            m_render.push(Point3::from(grid.world_point(mw, mb, mloss, 0.03)));
            m_world.push([mw, mb, mloss]);
        }
        scene.add_points_with_world(
            "Global Minimum",
            (&m_render, &m_world),
            5.5_f32,
            palette.success,
            MarkerShape::Diamond,
            true,
        );
    }
    populate_3d_racers(tool, scene, grid, palette);
    if let (Some(a), Some(c)) = (tool.slingshot_anchor, tool.slingshot_current) {
        let (la, lc) = (eval.evaluate(a[0], a[1]), eval.evaluate(c[0], c[1]));
        let pa = Point3::from(grid.world_point(a[0], a[1], la, 0.05));
        let pc = Point3::from(grid.world_point(c[0], c[1], lc, 0.05));
        scene.add_styled_segment_with_world(
            "Slingshot Vector",
            ([pa, pc], [[a[0], a[1], la], [c[0], c[1], lc]]),
            Stroke::new(2.5_f32, palette.warning),
            LineStyle::Solid,
            true,
            true,
        );
    }
}

fn populate_3d_racers(
    tool: &LossLandscapeTool,
    scene: &mut Scene3DBuilder,
    grid: &SurfaceGrid,
    palette: &SemanticPalette,
) {
    let eval = tool.surface_evaluator();
    for (idx, racer) in tool.arena.racers.iter().enumerate() {
        if !tool.arena_config.enabled_racers[idx] || !racer.kind.is_available_in_build() {
            continue;
        }
        let color = racer.kind.color(palette);
        let has_trail = racer.trail.len() >= 2;
        if has_trail {
            let mut r_pts = Vec::with_capacity(racer.trail.len());
            let mut w_pts = Vec::with_capacity(racer.trail.len());
            for [w, b] in racer.trail.iter() {
                let l = eval.evaluate(w, b);
                r_pts.push(Point3::from(grid.world_point(w, b, l, 0.035)));
                w_pts.push([w, b, l]);
            }
            scene.add_polyline_with_world(
                racer.kind.label(),
                &r_pts,
                &w_pts,
                Stroke::new(2.2_f32, color),
                LineStyle::Solid,
                true,
            );
        }
        let (qx, qy, pot) = (racer.phase.q.x, racer.phase.q.y, racer.potential_energy);
        let head_r = [Point3::from(grid.world_point(qx, qy, pot, 0.045))];
        let head_w = [[qx, qy, pot]];
        scene.add_points_with_world(
            racer.kind.label(),
            (&head_r, &head_w),
            5.0_f32,
            color,
            MarkerShape::Circle,
            !has_trail,
        );
    }
}
