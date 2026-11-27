//! Decorative 3D reference axes, labels, and floor grids for [`super::Scene3DBuilder`].

use eframe::egui;
use egui_plot::LineStyle;
use nalgebra::Point3;

use super::scene_batch::LabelPrimitive3D;
use super::Scene3DBuilder;

impl Scene3DBuilder {
    /// Adds projected 3D X, Y, and Z coordinate reference axes with endpoint labels.
    pub fn add_axes(&mut self, length: f64, dark_mode: bool) {
        let palette = if dark_mode {
            &crate::app::palette::DARK_PALETTE
        } else {
            &crate::app::palette::LIGHT_PALETTE
        };
        let axes = [
            ("X Axis", "X", Point3::new(length, 0.0, 0.0), palette.axis_x),
            ("Y Axis", "Y", Point3::new(0.0, length, 0.0), palette.axis_y),
            ("Z Axis", "Z", Point3::new(0.0, 0.0, length), palette.axis_z),
        ];
        let origin = Point3::new(
            self.origin_offset[0],
            self.origin_offset[1],
            self.origin_offset[2],
        );
        for (axis_name, short, dir, color) in axes {
            let tip = Point3::new(origin.x + dir.x, origin.y + dir.y, origin.z + dir.z);
            self.add_styled_segment(
                axis_name,
                [origin, tip],
                egui::Stroke::new(1.75_f32, color),
                LineStyle::Solid,
                true,
                false,
            );
            let label_tip = Point3::new(
                origin.x + dir.x * 1.08,
                origin.y + dir.y * 1.08,
                origin.z + dir.z * 1.08,
            );
            let (proj, _, _) = self.project_world(label_tip);
            self.labels.push(LabelPrimitive3D {
                proj,
                text: short.to_owned(),
                color,
            });
        }
        self.mark_dirty();
    }

    /// Adds a projected 3D text label anchored at world coordinate `pos`.
    pub fn add_label(&mut self, pos: Point3<f64>, text: impl Into<String>, color: egui::Color32) {
        let (proj, _, _) = self.project_world(pos);
        self.labels.push(LabelPrimitive3D {
            proj,
            text: text.into(),
            color,
        });
        self.mark_dirty();
    }

    /// Adds a decorative projected `xOy` floor grid at `z_plane` sorted alongside scene geometry.
    pub fn add_floor_grid(
        &mut self,
        half_extent: f64,
        divisions: usize,
        z_plane: f64,
        stroke: egui::Stroke,
    ) {
        let divs = divisions.max(1);
        let step = (2.0 * half_extent) / (divs as f64);
        let ox = self.origin_offset[0];
        let oy = self.origin_offset[1];
        for i in 0..=divs {
            let offset = -half_extent + (i as f64) * step;
            self.add_styled_segment(
                "",
                [
                    Point3::new(ox - half_extent, oy + offset, z_plane),
                    Point3::new(ox + half_extent, oy + offset, z_plane),
                ],
                stroke,
                LineStyle::Solid,
                false,
                false,
            );
            self.add_styled_segment(
                "",
                [
                    Point3::new(ox + offset, oy - half_extent, z_plane),
                    Point3::new(ox + offset, oy + half_extent, z_plane),
                ],
                stroke,
                LineStyle::Solid,
                false,
                false,
            );
        }
    }
}
