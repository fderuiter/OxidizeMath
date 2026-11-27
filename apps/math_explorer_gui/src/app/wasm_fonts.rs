//! Compacted TrueType font installation for WebAssembly builds (`target_arch = "wasm32"`).

use eframe::egui::{self, FontData, FontDefinitions, FontFamily, FontTweak};
use std::sync::Arc;

const HACK_COMPACT: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/wasm_fonts/Hack-Regular.ttf"));
const NOTO_EMOJI_COMPACT: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/wasm_fonts/NotoEmoji-Regular.ttf"
));
const UBUNTU_LIGHT_COMPACT: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/wasm_fonts/Ubuntu-Light.ttf"));
const EMOJI_ICON_COMPACT: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/wasm_fonts/emoji-icon-font.ttf"));

/// Builds [`FontDefinitions`] matching `epaint`'s default family hierarchy and tweaks using
/// build-time compacted TrueType subsets generated into `OUT_DIR/wasm_fonts/`.
#[must_use]
pub(super) fn compact_font_definitions() -> FontDefinitions {
    let mut defs = FontDefinitions::empty();
    defs.font_data.insert(
        "Hack".to_owned(),
        Arc::new(FontData::from_static(HACK_COMPACT)),
    );
    defs.font_data.insert(
        "NotoEmoji-Regular".to_owned(),
        Arc::new(FontData::from_static(NOTO_EMOJI_COMPACT).tweak(FontTweak {
            scale: 0.81,
            ..Default::default()
        })),
    );
    defs.font_data.insert(
        "Ubuntu-Light".to_owned(),
        Arc::new(FontData::from_static(UBUNTU_LIGHT_COMPACT)),
    );
    defs.font_data.insert(
        "emoji-icon-font".to_owned(),
        Arc::new(FontData::from_static(EMOJI_ICON_COMPACT).tweak(FontTweak {
            scale: 0.90,
            ..Default::default()
        })),
    );
    defs.families.insert(
        FontFamily::Monospace,
        vec![
            "Hack".to_owned(),
            "Ubuntu-Light".to_owned(),
            "NotoEmoji-Regular".to_owned(),
            "emoji-icon-font".to_owned(),
        ],
    );
    defs.families.insert(
        FontFamily::Proportional,
        vec![
            "Ubuntu-Light".to_owned(),
            "NotoEmoji-Regular".to_owned(),
            "emoji-icon-font".to_owned(),
        ],
    );
    defs
}

/// Installs compacted TrueType fonts on `ctx` before Frame 1 on `wasm32`.
pub(super) fn install_compact_fonts(ctx: &egui::Context) {
    ctx.set_fonts(compact_font_definitions());
}

#[cfg(test)]
mod tests {
    use super::install_compact_fonts;
    use eframe::egui;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn wasm_compact_fonts_layout_ascii_math_greek_and_ui_emojis() {
        let ctx = egui::Context::default();
        install_compact_fonts(&ctx);
        let output = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.label("Math Explorer: αβγπΩ ±√∞∫≈≠≤≥ ◻ 🦀🔬📊🌐");
                ui.monospace("f(x) = ∑ xᵢ² → ∞");
            });
        });
        assert!(!output.shapes.is_empty());
        assert!(!output.textures_delta.set.is_empty());
    }
}
