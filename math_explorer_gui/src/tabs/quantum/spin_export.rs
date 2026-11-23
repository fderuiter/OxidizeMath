//! Cold-path watertight Bloch sphere + 3D spin arrow builder for [`super::SpinVisualizer`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SpinVisualizer;
use oxidize_core::mesh::{
    append_colored_part, build_arrow, build_uv_sphere, extrude_polyline_tube, ColoredMesh, Mesh,
    Point3D, VertexColor,
};
use std::f32::consts::TAU;

fn build_great_circle_ring(axis: usize) -> Result<Mesh, String> {
    let samples = 48_usize;
    let mut ring = Vec::with_capacity(samples);
    for i in 0..samples {
        let theta = TAU * (i as f32) / (samples as f32);
        let (sin_t, cos_t) = theta.sin_cos();
        let pt = match axis {
            0 => Point3D::new(cos_t, sin_t, 0.0),
            1 => Point3D::new(0.0, cos_t, sin_t),
            _ => Point3D::new(cos_t, 0.0, sin_t),
        };
        ring.push(pt);
    }
    extrude_polyline_tube(&ring, 0.022, 12, true).map_err(|e| e.to_string())
}

impl SpinVisualizer {
    pub(super) fn build_bloch_colored_mesh_3d(&self) -> Result<ColoredMesh, String> {
        let ex = self.sx.expectation_value(&self.psi).re;
        let ey = self.sy.expectation_value(&self.psi).re;
        let ez = self.sz.expectation_value(&self.psi).re;
        if !ex.is_finite() || !ey.is_finite() || !ez.is_finite() {
            return Err("Quantum spin expectation values are non-finite.".to_string());
        }

        let mut combined = ColoredMesh::default();

        let core = build_uv_sphere(Point3D::new(0.0, 0.0, 0.0), 0.94, 24, 16)
            .map_err(|e| e.to_string())?;
        append_colored_part(&mut combined, &core, VertexColor::new(62, 86, 122));

        let ring_color = VertexColor::new(148, 168, 196);
        for axis in 0..3 {
            let ring = build_great_circle_ring(axis)?;
            append_colored_part(&mut combined, &ring, ring_color);
        }

        let axis_specs = [
            (Point3D::new(1.22, 0.0, 0.0), VertexColor::new(220, 82, 74)),
            (Point3D::new(0.0, 1.22, 0.0), VertexColor::new(78, 194, 118)),
            (Point3D::new(0.0, 0.0, 1.22), VertexColor::new(82, 148, 236)),
        ];
        for (tip, color) in axis_specs {
            let axis_arrow = build_arrow(Point3D::new(0.0, 0.0, 0.0), tip, 0.018, 0.048, 0.12, 10)
                .map_err(|e| e.to_string())?;
            append_colored_part(&mut combined, &axis_arrow, color);
        }

        let tip = Point3D::new(ex as f32, ey as f32, ez as f32) * 1.12;
        let spin_color = VertexColor::new(248, 192, 58);
        if tip.magnitude() > 0.05 {
            let spin_arrow = build_arrow(Point3D::new(0.0, 0.0, 0.0), tip, 0.042, 0.105, 0.22, 16)
                .map_err(|e| e.to_string())?;
            append_colored_part(&mut combined, &spin_arrow, spin_color);
        } else {
            let mixed_marker = build_uv_sphere(Point3D::new(0.0, 0.0, 0.0), 0.12, 14, 10)
                .map_err(|e| e.to_string())?;
            append_colored_part(&mut combined, &mixed_marker, spin_color);
        }

        Ok(combined)
    }
}
