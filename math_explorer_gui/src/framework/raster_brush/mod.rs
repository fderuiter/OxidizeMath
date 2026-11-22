#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod bresenham;
pub mod canvas;
pub mod rle;
pub mod state;
pub mod stencils;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

pub use bresenham::{
    bresenham_line, cell_screen_rect, compute_aspect_locked_rect, map_pointer_to_cell,
};
pub use canvas::RasterBrushCanvas;
pub use rle::{
    decode_bool_grid, decode_u8_grid, encode_bool_grid, encode_u8_grid, MAX_RLE_CELLS,
    MAX_RLE_ENCODED_LEN, MAX_RLE_GRID_DIM,
};
pub use state::{RasterBrushState, RasterBrushStroke};
pub use stencils::{generate_brush_cells, BrushStencil, SymmetryMode};

/// Renders compact brush radius, preset stencil, and symmetry controls inside a side panel.
pub fn draw_brush_palette_controls(
    ui: &mut eframe::egui::Ui,
    id_prefix: &str,
    state: &mut RasterBrushState,
    max_radius: usize,
) {
    ui.add(eframe::egui::Slider::new(&mut state.radius, 0..=max_radius).text("Brush Radius"));
    eframe::egui::ComboBox::from_id_salt(format!("{id_prefix}_stencil"))
        .selected_text(state.stencil.label())
        .show_ui(ui, |ui| {
            for candidate in BrushStencil::ALL {
                if ui
                    .selectable_value(&mut state.stencil, candidate, candidate.label())
                    .changed()
                {
                    state.line_anchor = None;
                }
            }
        });
    ui.small("Stencil Shape");
    eframe::egui::ComboBox::from_id_salt(format!("{id_prefix}_symmetry"))
        .selected_text(state.symmetry.label())
        .show_ui(ui, |ui| {
            for sym in SymmetryMode::ALL {
                ui.selectable_value(&mut state.symmetry, sym, sym.label());
            }
        });
    ui.small("Symmetry");
}
