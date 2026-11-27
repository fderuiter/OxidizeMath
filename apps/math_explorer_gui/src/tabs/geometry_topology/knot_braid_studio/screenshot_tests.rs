//! Headless software rasterizer verification generating PNG previews of `KnotBraidStudioTool`.

use super::presets::{
    BORROMEAN_RINGS_PRESET_JSON, CHIRAL_TREFOIL_PRESET_JSON, FIGURE_EIGHT_ISOTOPY_PRESET_JSON,
};
use super::KnotBraidStudioTool;
use crate::app::theme::ThemeMode;
use crate::framework::InteractiveTool;
use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, TextureId};
use math_commons::math_kernel::types::flatten_2d_index;
use std::collections::HashMap;

fn apply_textures_delta(atlas: &mut HashMap<TextureId, ColorImage>, delta: &egui::TexturesDelta) {
    for (id, img_delta) in &delta.set {
        let egui::ImageData::Color(patch) = &img_delta.image;
        if let Some([ox, oy]) = img_delta.pos {
            if let Some(dst) = atlas.get_mut(id) {
                let dst_w = dst.width();
                for py in 0..patch.height() {
                    for px in 0..patch.width() {
                        if ox + px < dst_w && oy + py < dst.height() {
                            let src_i = flatten_2d_index(px, py, patch.width());
                            let dst_i = flatten_2d_index(ox + px, oy + py, dst_w);
                            dst.pixels[dst_i] = patch.pixels[src_i];
                        }
                    }
                }
            }
        } else {
            atlas.insert(*id, (**patch).clone());
        }
    }
}

fn sample_tex(tex: Option<&ColorImage>, uv: Pos2) -> Color32 {
    let Some(img) = tex else {
        return Color32::WHITE;
    };
    if img.width() == 0 || img.height() == 0 {
        return Color32::WHITE;
    }
    let tx = ((uv.x.clamp(0.0, 1.0) * img.width() as f32) as usize).min(img.width() - 1);
    let ty = ((uv.y.clamp(0.0, 1.0) * img.height() as f32) as usize).min(img.height() - 1);
    img.pixels[flatten_2d_index(tx, ty, img.width())]
}

fn blend_premultiplied(dst: Color32, v_col: Color32, t_col: Color32) -> Color32 {
    let modulate = |a: u8, b: u8| ((u16::from(a) * u16::from(b) + 127) / 255) as u8;
    let sr = modulate(v_col.r(), t_col.r());
    let sg = modulate(v_col.g(), t_col.g());
    let sb = modulate(v_col.b(), t_col.b());
    let sa = modulate(v_col.a(), t_col.a());
    let inv = 255_u16.saturating_sub(u16::from(sa));
    let over = |s: u8, d: u8| (u16::from(s) + (u16::from(d) * inv + 127) / 255).min(255) as u8;
    Color32::from_rgba_premultiplied(over(sr, dst.r()), over(sg, dst.g()), over(sb, dst.b()), 255)
}

fn rasterize_triangle(
    fb: &mut ColorImage,
    clip: Rect,
    verts: [egui::epaint::Vertex; 3],
    tex: Option<&ColorImage>,
) {
    let [v0, v1, v2] = verts;
    let edge = |a: Pos2, b: Pos2, p: Pos2| (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
    let area = edge(v0.pos, v1.pos, v2.pos);
    if area.abs() < 1e-4 {
        return;
    }
    let min_x = v0
        .pos
        .x
        .min(v1.pos.x)
        .min(v2.pos.x)
        .max(clip.min.x)
        .floor()
        .max(0.0) as usize;
    let max_x = v0.pos.x.max(v1.pos.x).max(v2.pos.x).min(clip.max.x).ceil() as usize;
    let min_y = v0
        .pos
        .y
        .min(v1.pos.y)
        .min(v2.pos.y)
        .max(clip.min.y)
        .floor()
        .max(0.0) as usize;
    let max_y = v0.pos.y.max(v1.pos.y).max(v2.pos.y).min(clip.max.y).ceil() as usize;
    let w = fb.width();
    let h = fb.height();

    for py in min_y..max_y.min(h) {
        for px in min_x..max_x.min(w) {
            let p = Pos2::new(px as f32 + 0.5, py as f32 + 0.5);
            let w0 = edge(v1.pos, v2.pos, p) / area;
            let w1 = edge(v2.pos, v0.pos, p) / area;
            let w2 = 1.0 - w0 - w1;
            if w0 >= -1e-4 && w1 >= -1e-4 && w2 >= -1e-4 {
                let uv = Pos2::new(
                    w0 * v0.uv.x + w1 * v1.uv.x + w2 * v2.uv.x,
                    w0 * v0.uv.y + w1 * v1.uv.y + w2 * v2.uv.y,
                );
                let interp = |c0: u8, c1: u8, c2: u8| {
                    (w0 * f32::from(c0) + w1 * f32::from(c1) + w2 * f32::from(c2)).clamp(0.0, 255.0)
                        as u8
                };
                let v_col = Color32::from_rgba_premultiplied(
                    interp(v0.color.r(), v1.color.r(), v2.color.r()),
                    interp(v0.color.g(), v1.color.g(), v2.color.g()),
                    interp(v0.color.b(), v1.color.b(), v2.color.b()),
                    interp(v0.color.a(), v1.color.a(), v2.color.a()),
                );
                let idx = flatten_2d_index(px, py, w);
                fb.pixels[idx] = blend_premultiplied(fb.pixels[idx], v_col, sample_tex(tex, uv));
            }
        }
    }
}

fn render_studio_preset_png(preset_json: &str, out_path: &str) {
    let (w, h) = (1320_usize, 820_usize);
    let mut tool = KnotBraidStudioTool::default();
    tool.import_preset_json(preset_json).expect("preset json");

    let ctx = egui::Context::default();
    ThemeMode::Dark.apply_to_context(&ctx);
    let mut atlas: HashMap<TextureId, ColorImage> = HashMap::new();
    let screen = Rect::from_min_size(Pos2::ZERO, egui::vec2(w as f32, h as f32));

    // Warm-up layout pass followed by final frame capture.
    let mut clipped = Vec::new();
    for _ in 0..2 {
        let raw = egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        };
        let out = ctx.run(raw, |c| {
            egui::CentralPanel::default().show(c, |ui| tool.show_inside(ui));
        });
        apply_textures_delta(&mut atlas, &out.textures_delta);
        clipped = ctx.tessellate(out.shapes, 1.0);
    }

    let mut fb = ColorImage::new([w, h], vec![Color32::from_rgb(16, 18, 24); w * h]);
    for prim in &clipped {
        if let egui::epaint::Primitive::Mesh(mesh) = &prim.primitive {
            let tex = atlas.get(&mesh.texture_id);
            for &[i0, i1, i2] in mesh.indices.as_chunks::<3>().0 {
                let tri = [
                    mesh.vertices[i0 as usize],
                    mesh.vertices[i1 as usize],
                    mesh.vertices[i2 as usize],
                ];
                rasterize_triangle(&mut fb, prim.clip_rect, tri, tex);
            }
        }
    }
    let png = egui_plot::export::color32_image_to_png(&fb).expect("encode studio preview PNG");
    assert!(png.len() > 4096);
    let candidate = std::path::Path::new(out_path);
    let target = if candidate.parent().is_some_and(std::path::Path::is_dir) {
        candidate.to_path_buf()
    } else {
        std::env::temp_dir().join(
            candidate
                .file_name()
                .expect("screenshot path must include file name"),
        )
    };
    std::fs::write(target, png).expect("write studio preview PNG");
}

#[test]
fn generate_knot_braid_studio_verification_screenshots() {
    render_studio_preset_png(
        CHIRAL_TREFOIL_PRESET_JSON,
        "/tmp/knot_studio_chiral_trefoil.png",
    );
    render_studio_preset_png(
        FIGURE_EIGHT_ISOTOPY_PRESET_JSON,
        "/tmp/knot_studio_figure_eight_reidemeister.png",
    );
    render_studio_preset_png(
        BORROMEAN_RINGS_PRESET_JSON,
        "/tmp/knot_studio_borromean_rings.png",
    );
}
