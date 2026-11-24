#[cfg(not(target_arch = "wasm32"))]
use super::{PhysarumPreset, PhysarumTool};
#[cfg(not(target_arch = "wasm32"))]
use crate::framework::InteractiveTool;
#[cfg(not(target_arch = "wasm32"))]
use math_commons::math_kernel::types::flatten_2d_index;

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn write_verification_screenshots() {
    let specs = [
        (
            PhysarumPreset::SteinerTriangle,
            340usize,
            "/tmp/physarum_steiner_y_junction.png",
        ),
        (
            PhysarumPreset::AsymmetricMaze,
            400usize,
            "/tmp/physarum_asymmetric_maze.png",
        ),
        (
            PhysarumPreset::TokyoRail,
            240usize,
            "/tmp/physarum_tokyo_rail.png",
        ),
    ];
    for (preset, steps, path) in specs {
        let mut tool = PhysarumTool::default();
        tool.apply_preset(preset);
        for _ in 0..steps {
            tool.step_once();
        }
        tool.refresh_benchmark_and_record();
        tool.refresh_pixels_in_place();
        save_annotated_tool_png(&tool, 4, path);
        if preset == PhysarumPreset::SteinerTriangle {
            if let Ok(plot_png) = egui_plot::export::export_custom_png(
                &tool.export_datasets(),
                &egui_plot::export::ExportOptions::default(),
            ) {
                let _ = std::fs::write("/tmp/physarum_benchmark_telemetry.png", plot_png);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn save_annotated_tool_png(tool: &PhysarumTool, scale: usize, out_path: &str) {
    let w = tool.engine.width();
    let h = tool.engine.height();
    let sw = w * scale;
    let sh = h * scale;
    let mut rgba = vec![0u8; sw * sh * 4];
    for y in 0..sh {
        let gy = y / scale;
        for x in 0..sw {
            let gx = x / scale;
            let c = tool.pixels[flatten_2d_index(gx, gy, w)];
            let base = flatten_2d_index(x, y, sw) * 4;
            rgba[base] = c.r();
            rgba[base + 1] = c.g();
            rgba[base + 2] = c.b();
            rgba[base + 3] = 255;
        }
    }
    for &(p0, p1) in &tool.latest_benchmark.mst_segments {
        draw_line_rgba(&mut rgba, sw, sh, p0, p1, scale, [132, 140, 152, 220]);
    }
    for edge in &tool.latest_benchmark.edges {
        draw_line_rgba(
            &mut rgba,
            sw,
            sh,
            edge.p0,
            edge.p1,
            scale,
            [74, 214, 122, 255],
        );
    }
    for &junc in &tool.latest_benchmark.steiner_junctions {
        draw_disc_rgba(&mut rgba, sw, sh, junc, scale, 6, [245, 194, 66, 255]);
    }
    for term in tool.engine.terminals() {
        draw_disc_rgba(
            &mut rgba,
            sw,
            sh,
            [term.x, term.y],
            scale,
            6,
            [250, 204, 56, 255],
        );
    }
    let mut buf = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut buf, sw as u32, sh as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        if let Ok(mut writer) = enc.write_header() {
            let _ = writer.write_image_data(&rgba);
        }
    }
    let _ = std::fs::write(out_path, buf);
}

#[cfg(not(target_arch = "wasm32"))]
fn draw_line_rgba(
    rgba: &mut [u8],
    sw: usize,
    sh: usize,
    p0: [f64; 2],
    p1: [f64; 2],
    scale: usize,
    col: [u8; 4],
) {
    let x0 = (p0[0] + 0.5) * (scale as f64);
    let y0 = (p0[1] + 0.5) * (scale as f64);
    let x1 = (p1[0] + 0.5) * (scale as f64);
    let y1 = (p1[1] + 0.5) * (scale as f64);
    let steps = ((x1 - x0).hypot(y1 - y0).ceil() as usize).max(1);
    for s in 0..=steps {
        let t = (s as f64) / (steps as f64);
        let px = (x0 + t * (x1 - x0)).round() as isize;
        let py = (y0 + t * (y1 - y0)).round() as isize;
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                let nx = px + dx;
                let ny = py + dy;
                if nx >= 0 && nx < sw as isize && ny >= 0 && ny < sh as isize {
                    let base = flatten_2d_index(nx as usize, ny as usize, sw) * 4;
                    rgba[base..base + 4].copy_from_slice(&col);
                }
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn draw_disc_rgba(
    rgba: &mut [u8],
    sw: usize,
    sh: usize,
    center: [f64; 2],
    scale: usize,
    rad_px: isize,
    col: [u8; 4],
) {
    let cx = ((center[0] + 0.5) * (scale as f64)).round() as isize;
    let cy = ((center[1] + 0.5) * (scale as f64)).round() as isize;
    for dy in -rad_px..=rad_px {
        for dx in -rad_px..=rad_px {
            let d2 = dx * dx + dy * dy;
            if d2 <= rad_px * rad_px {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx >= 0 && nx < sw as isize && ny >= 0 && ny < sh as isize {
                    let base = flatten_2d_index(nx as usize, ny as usize, sw) * 4;
                    rgba[base..base + 4].copy_from_slice(&col);
                }
            }
        }
    }
}
