//! Cold-path watertight 3D Bravais unit-cell mesh builder for [`super::CrystalViewer`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::CrystalViewer;
#[cfg(feature = "physics")]
use math_explorer::physics::solid_state::lattice::CrystalSystem;
use nalgebra::Vector3;
use oxidize_core::mesh::{
    append_colored_part, build_cylinder, build_uv_sphere, ColoredMesh, Point3D, VertexColor,
};

impl CrystalViewer {
    pub(super) fn build_crystal_colored_mesh(&self) -> Result<ColoredMesh, String> {
        if !self.lattice_constant.is_finite() || self.lattice_constant <= 0.0 {
            return Err("Lattice constant must be finite and positive.".to_string());
        }
        let unit_cell = self.system.generate(self.lattice_constant);
        let [v1, v2, v3] = unit_cell.lattice_vectors;
        let center_offset = (v1 + v2 + v3) * 0.5;
        let to_p3 = |v: Vector3<f64>| -> Point3D {
            let c = v - center_offset;
            Point3D::new(c.x as f32, c.y as f32, c.z as f32)
        };

        let a = self.lattice_constant as f32;
        let atom_radius = (0.14 * a).max(0.05);
        let corner_radius = (0.10 * a).max(0.04);
        let strut_radius = (0.03 * a).max(0.015);
        let mut combined = ColoredMesh::default();

        let basis_color = VertexColor::new(235, 92, 74);
        for &pos in &unit_cell.atomic_positions {
            let sphere =
                build_uv_sphere(to_p3(pos), atom_radius, 16, 12).map_err(|e| e.to_string())?;
            append_colored_part(&mut combined, &sphere, basis_color);
        }

        let corners = [
            Vector3::zeros(),
            v1,
            v1 + v2,
            v2,
            v3,
            v1 + v3,
            v1 + v2 + v3,
            v2 + v3,
        ];
        let periodic_color = VertexColor::new(88, 176, 236);
        for corner in &corners {
            let overlaps_basis = unit_cell
                .atomic_positions
                .iter()
                .any(|p| (p - corner).norm() < 1e-5);
            if !overlaps_basis {
                let sphere = build_uv_sphere(to_p3(*corner), corner_radius, 14, 10)
                    .map_err(|e| e.to_string())?;
                append_colored_part(&mut combined, &sphere, periodic_color);
            }
        }
        let corner_pts = corners.map(to_p3);
        Self::append_unit_cell_struts(&mut combined, &corner_pts, strut_radius)?;
        Ok(combined)
    }

    fn append_unit_cell_struts(
        combined: &mut ColoredMesh,
        corners: &[Point3D; 8],
        strut_radius: f32,
    ) -> Result<(), String> {
        let strut_color = VertexColor::new(142, 156, 178);
        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        for (i0, i1) in EDGES {
            let cyl = build_cylinder(corners[i0], corners[i1], strut_radius, 12)
                .map_err(|e| e.to_string())?;
            append_colored_part(combined, &cyl, strut_color);
        }
        Ok(())
    }
}
