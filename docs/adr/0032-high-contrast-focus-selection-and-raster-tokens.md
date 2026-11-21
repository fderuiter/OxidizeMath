# ADR 0032: High-Contrast Focus, Selection, and Dark-Canvas Raster Tokens

## Status
Accepted (extends [ADR 0010](0010-theme-state-management-and-menu-toggles.md))

## Context
Although [ADR 0010](0010-theme-state-management-and-menu-toggles.md) introduced `ThemeMode::HighContrast` for low-vision and keyboard users, several token and raster paths still fell below WCAG 2.2 minimum contrast (`4.5:1` for text under SC 1.4.3 and `3.0:1` for non-text UI components, focus indicators, and graphical states under SC 1.4.11):
1. **Selection and Noninteractive Strokes**: In [`math_explorer_gui/src/app/theme.rs`](../../math_explorer_gui/src/app/theme.rs), `ThemeMode::HighContrast::visuals()` set `visuals.selection.stroke` to `Stroke::new(1.0, Color32::BLACK)` against `#000000` panels (`1.00:1`) and `rgb(0, 90, 158)` selection fills (`2.95:1`), while keeping `noninteractive.bg_stroke` at default dark gray (`2.26:1`).
2. **Active Domain Buttons and Progress Bars**: [`math_explorer_gui/src/app/domain_bar.rs`](../../math_explorer_gui/src/app/domain_bar.rs) colored active `AcademicDomain` labels with `selection.stroke.color` without applying `selection.bg_fill`, and [`math_explorer_gui/src/sensitivity.rs`](../../math_explorer_gui/src/sensitivity.rs) filled `egui::ProgressBar` bars with bright foreground status tokens under light `selection.stroke.color` labels.
3. **Dark-Canvas Colormaps and Raster Widgets**: In [`crates/math_commons/src/math_kernel/colormap.rs`](../../crates/math_commons/src/math_kernel/colormap.rs), `heatmap_color(0.0)` (`[0, 0, 255]`, `2.44:1` on `#000000`) and `cyclic_cosine_palette` troughs emitted pixels below `3.0:1` on `BLACK` High Contrast panels across medical dose, Turing morphogenesis, and escape-time fractal tools, while Ising spin lattices and Ulam prime spirals used hardcoded RGB constants without `1px` canvas borders.

## Decision
1. **High-Contrast Visual Tokens & Split-Pane Borders**:
   - Set `visuals.selection.stroke` to `Stroke::new(2.0, Color32::WHITE)` (`7.10:1` on `rgb(0, 90, 158)`, `21.00:1` on `BLACK`) and `visuals.widgets.noninteractive.bg_stroke` / `window_stroke` to `HIGH_CONTRAST_PALETTE.graph_edge` (`rgb(196, 196, 196)`, `12.04:1` on `BLACK`), preserving `extreme_bg_color = BLACK` and `selection.bg_fill = rgb(0, 90, 158)` so `ThemeMode::from_visuals` continues to roundtrip cleanly in split panes.
   - Honor `selection.stroke.width` and `noninteractive.bg_stroke` in [`math_explorer_gui/src/app/viewport/focus.rs`](../../math_explorer_gui/src/app/viewport/focus.rs), style active domain buttons with `selection.bg_fill` + `selection.stroke.color`, and provide theme-aware `SemanticPalette::progress_bar_fill` tokens in [`math_explorer_gui/src/app/palette.rs`](../../math_explorer_gui/src/app/palette.rs).
2. **Dark-Canvas Colormap Helpers in `math_commons`**:
   - Add `ensure_dark_canvas_contrast`, `heatmap_color_high_contrast`, and `cyclic_cosine_palette_high_contrast` in [`crates/math_commons/src/math_kernel/colormap.rs`](../../crates/math_commons/src/math_kernel/colormap.rs), lifting samples with relative luminance `L < 0.11` in linear light so every High Contrast pixel exceeds `3.0:1` against `#000000` while keeping `heatmap_color`, `cyclic_cosine_palette`, and `mapped_gradient` unchanged for `Dark` and `Light` callers.
3. **Semantic Raster Mapping and Lint Enforcement**:
   - Route Ising spin states and Ulam prime spiral pixels through `SemanticPalette`, draw `1px` `noninteractive.bg_stroke` borders around dark raster `egui::Image` canvases, remove the `prime_spiral.rs` lint exemption, and audit literal `Color32::from_rgb` calls in [`math_explorer_gui/src/app/theme_palette_tests.rs`](../../math_explorer_gui/src/app/theme_palette_tests.rs).

## Consequences
- **Positive**: Keyboard focus rings, active domain pills, `ProgressBar` text (at `0.0`, `0.5`, and `1.0` fill), and raster visualizations satisfy WCAG 2.2 `4.5:1` text and `3.0:1` graphical contrast across `ThemeMode::HighContrast`.
- **Neutral**: `ThemeMode::Dark` and `ThemeMode::Light` preserve their default `egui::Visuals` tokens and existing `heatmap_color` / `cyclic_cosine_palette` outputs.
