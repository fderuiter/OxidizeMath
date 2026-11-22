#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{HyperbolicGeometryTool, LabMode};
use crate::app::{SemanticPalette, palette::DARK_PALETTE};
use eframe::egui;
use egui_plot::ExportPrimitive;
use math_explorer::pure_math::hyperbolic::{
    GeodesicSegment, HorocycleDisk, MAX_BILLIARD_SEGMENTS, TiledPolygon,
    cayley_complex_to_half_plane_clamped, sample_disk_geodesic_into,
};
use num_complex::Complex64;
use std::f64::consts::TAU;

const ARC_SAMPLES: usize = 11;

impl HyperbolicGeometryTool {
    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = Vec::new();
        datasets.push(("Poincare Disk Boundary".to_string(), unit_circle_points(96)));

        match self.state.mode {
            LabMode::TilingExplorer => self.append_tiling_datasets(&mut datasets),
            LabMode::ChaoticBilliards => self.append_billiard_datasets(&mut datasets),
        }

        datasets
    }

    fn append_tiling_datasets(&self, datasets: &mut Vec<(String, Vec<[f64; 2]>)>) {
        if let Some(central) = self.transformed_tiles.first() {
            if central.vertex_count >= 3 {
                datasets.push((
                    "Central Hyperbolic Tile (Disk)".to_string(),
                    sample_polygon_disk_points(central),
                ));
                datasets.push((
                    "Central Hyperbolic Tile (Upper Half-Plane)".to_string(),
                    sample_polygon_half_plane_points(central),
                ));
            }
        }

        if self.state.show_horocycles {
            let base = self.state.disk_isometry.apply_complex(Complex64::new(0.0, 1.0));
            if let Ok(horo) = HorocycleDisk::from_radius(base, self.state.horocycle_radius) {
                let mut pts = vec![[0.0; 2]; 65];
                horo.sample_into(&mut pts);
                datasets.push(("Tangent Horocycle (Disk)".to_string(), pts));
            }
        }
    }

    fn append_billiard_datasets(&self, datasets: &mut Vec<(String, Vec<[f64; 2]>)>) {
        let mut sep = vec![[0.0; 2]; MAX_BILLIARD_SEGMENTS];
        let n_sep = self.billiards.copy_separation_series_into(&mut sep);
        sep.truncate(n_sep);
        if !sep.is_empty() {
            datasets.push(("Log Separation ln(delta_t / delta_0)".to_string(), sep));
        }

        let mut hits = vec![0u8; MAX_BILLIARD_SEGMENTS];
        let n_hits = self.billiards.copy_edge_hits_into(&mut hits);
        if n_hits > 0 {
            let hit_pts: Vec<[f64; 2]> = hits[..n_hits]
                .iter()
                .enumerate()
                .map(|(idx, &code)| [idx as f64, f64::from(code)])
                .collect();
            datasets.push(("Symbolic Edge Hit Codes".to_string(), hit_pts));
        }

        let mut p_segs = vec![GeodesicSegment::default(); MAX_BILLIARD_SEGMENTS];
        let n_p = self.billiards.copy_primary_segments_into(&mut p_segs);
        if n_p > 0 {
            datasets.push((
                "Primary Geodesic Ray (Disk)".to_string(),
                sample_segments_polyline(&p_segs[..n_p.min(256)], self.state.disk_isometry),
            ));
        }

        let mut t_segs = vec![GeodesicSegment::default(); MAX_BILLIARD_SEGMENTS];
        let n_t = self.billiards.copy_twin_segments_into(&mut t_segs);
        if n_t > 0 {
            datasets.push((
                "Twin Geodesic Ray (Disk)".to_string(),
                sample_segments_polyline(&t_segs[..n_t.min(256)], self.state.disk_isometry),
            ));
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = &DARK_PALETTE;
        let mut prims = Vec::new();
        prims.push(ExportPrimitive::Polyline {
            name: "Poincare Disk Boundary".to_string(),
            points: unit_circle_points(128),
            color: palette.accent,
            stroke_width: 2.0,
            closed: true,
            fill: Some(palette.surface_fill),
        });

        match self.state.mode {
            LabMode::TilingExplorer => self.append_tiling_primitives(&mut prims, palette),
            LabMode::ChaoticBilliards => self.append_billiard_primitives(&mut prims, palette),
        }
        prims
    }

    fn append_tiling_primitives(
        &self,
        prims: &mut Vec<ExportPrimitive>,
        palette: &SemanticPalette,
    ) {
        let count = self.transformed_tile_count.min(self.transformed_tiles.len());
        for (idx, tile) in self.transformed_tiles[..count].iter().enumerate() {
            if tile.vertex_count < 3 {
                continue;
            }
            let (fill, stroke) = export_tile_colors(tile.depth, idx, palette);
            prims.push(ExportPrimitive::Polyline {
                name: format!("Tile {idx}"),
                points: sample_polygon_disk_points(tile),
                color: stroke,
                stroke_width: 1.15,
                closed: true,
                fill: Some(fill),
            });
        }
        if self.state.show_horocycles {
            let base = self.state.disk_isometry.apply_complex(Complex64::new(0.0, 1.0));
            if let Ok(horo) = HorocycleDisk::from_radius(base, self.state.horocycle_radius) {
                let mut pts = vec![[0.0; 2]; 72];
                horo.sample_into(&mut pts);
                prims.push(ExportPrimitive::Polyline {
                    name: "Tangent Horocycle".to_string(),
                    points: pts,
                    color: palette.warning,
                    stroke_width: 1.8,
                    closed: true,
                    fill: None,
                });
            }
        }
    }

    fn append_billiard_primitives(
        &self,
        prims: &mut Vec<ExportPrimitive>,
        palette: &SemanticPalette,
    ) {
        let iso = self.state.disk_isometry;
        let verts = self.billiards.table().vertices();
        let mut table_pts = Vec::with_capacity(verts.len() * ARC_SAMPLES);
        let mut arc = [[0.0; 2]; ARC_SAMPLES];
        for i in 0..verts.len() {
            let v0 = iso.apply_complex(verts[i]);
            let v1 = iso.apply_complex(verts[(i + 1) % verts.len()]);
            sample_disk_geodesic_into(v0, v1, &mut arc);
            table_pts.extend_from_slice(&arc[..ARC_SAMPLES - 1]);
        }
        prims.push(ExportPrimitive::Polyline {
            name: "Billiard Table".to_string(),
            points: table_pts,
            color: palette.info,
            stroke_width: 2.2,
            closed: true,
            fill: Some(palette.surface_fill.gamma_multiply(0.85)),
        });

        let mut p_segs = vec![GeodesicSegment::default(); MAX_BILLIARD_SEGMENTS];
        let n_p = self.billiards.copy_primary_segments_into(&mut p_segs);
        if n_p > 0 {
            prims.push(ExportPrimitive::Polyline {
                name: "Primary Geodesic Ray".to_string(),
                points: sample_segments_polyline(&p_segs[..n_p.min(320)], iso),
                color: palette.series_color(2),
                stroke_width: 1.6,
                closed: false,
                fill: None,
            });
        }
        let mut t_segs = vec![GeodesicSegment::default(); MAX_BILLIARD_SEGMENTS];
        let n_t = self.billiards.copy_twin_segments_into(&mut t_segs);
        if n_t > 0 {
            prims.push(ExportPrimitive::Polyline {
                name: "Perturbed Twin Ray".to_string(),
                points: sample_segments_polyline(&t_segs[..n_t.min(320)], iso),
                color: palette.series_color(1),
                stroke_width: 1.4,
                closed: false,
                fill: None,
            });
        }
    }
}

pub(super) fn unit_circle_points(samples: usize) -> Vec<[f64; 2]> {
    let n = samples.max(8);
    (0..=n)
        .map(|i| {
            let ang = TAU * (i as f64) / (n as f64);
            [ang.cos(), ang.sin()]
        })
        .collect()
}

pub(super) fn sample_polygon_disk_points(poly: &TiledPolygon) -> Vec<[f64; 2]> {
    let n = poly.vertex_count;
    let mut out = Vec::with_capacity(n * ARC_SAMPLES);
    let mut arc = [[0.0; 2]; ARC_SAMPLES];
    for i in 0..n {
        let z1 = poly.vertices[i];
        let z2 = poly.vertices[(i + 1) % n];
        sample_disk_geodesic_into(z1, z2, &mut arc);
        out.extend_from_slice(&arc[..ARC_SAMPLES - 1]);
    }
    out
}

fn sample_polygon_half_plane_points(poly: &TiledPolygon) -> Vec<[f64; 2]> {
    sample_polygon_disk_points(poly)
        .into_iter()
        .map(|[x, y]| {
            let w = cayley_complex_to_half_plane_clamped(Complex64::new(x, y));
            [w.re, w.im]
        })
        .collect()
}

fn sample_segments_polyline(
    segs: &[GeodesicSegment],
    iso: math_explorer::pure_math::hyperbolic::Psu11,
) -> Vec<[f64; 2]> {
    let mut pts = Vec::with_capacity(segs.len() * ARC_SAMPLES);
    let mut arc = [[0.0; 2]; ARC_SAMPLES];
    for (idx, seg) in segs.iter().enumerate() {
        let z1 = iso.apply(seg.start).as_complex();
        let z2 = iso.apply(seg.end).as_complex();
        sample_disk_geodesic_into(z1, z2, &mut arc);
        let slice = if idx + 1 == segs.len() {
            &arc[..]
        } else {
            &arc[..ARC_SAMPLES - 1]
        };
        pts.extend_from_slice(slice);
    }
    pts
}

pub(super) fn export_tile_colors(
    depth: u8,
    idx: usize,
    palette: &SemanticPalette,
) -> (egui::Color32, egui::Color32) {
    let parity = (usize::from(depth) + idx).is_multiple_of(2);
    let fill = match depth {
        0 => palette.series_color(0).gamma_multiply(0.72),
        1 if parity => palette.series_color(2).gamma_multiply(0.62),
        1 => palette.series_color(4).gamma_multiply(0.62),
        2 if parity => palette.series_color(1).gamma_multiply(0.55),
        2 => palette.series_color(0).gamma_multiply(0.52),
        _ if parity => palette.series_color(4).gamma_multiply(0.45),
        _ => palette.series_color(2).gamma_multiply(0.45),
    };
    let stroke = palette.node_stroke.gamma_multiply(0.78);
    (fill, stroke)
}
