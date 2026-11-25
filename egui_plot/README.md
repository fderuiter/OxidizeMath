Simple plotting library for [`egui`](https://github.com/emilk/egui).

Check out [`Plot`] for how to get started.

[**Looking for maintainer!**](https://github.com/emilk/egui/issues/4705)

## Feature flags

# egui_plot

[<img alt="github" src="https://img.shields.io/badge/github-emilk/egui_plot-8da0cb?logo=github" height="20">](https://github.com/emilk/egui_plot)
[![Latest version](https://img.shields.io/crates/v/egui_plot.svg)](https://crates.io/crates/egui_plot)
[![Documentation](https://docs.rs/egui_plot/badge.svg)](https://docs.rs/egui_plot)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache](https://img.shields.io/badge/license-Apache-blue.svg)
[![Discord](https://img.shields.io/discord/900275882684477440?label=egui%20discord)](https://discord.gg/JFcEma9bJq)

Immediate mode plotting for [`egui`](https://github.com/emilk/egui).

![egui_plot_white](https://github.com/user-attachments/assets/b29acf5e-ccbf-4cb7-b03b-7e258fa5db16)

[Try the web demo](https://emilk.github.io/egui_plot/)

## Testing

- Locally: `cargo run -p demo`
- Web: `(cd demo && trunk serve)`

## Export Capabilities

`egui_plot` includes cold-path vector and procedural raster encoders in `egui_plot::export` and `egui_plot::export_primitives`:

- **Vector SVG (`ExportPrimitive::ContourPath`)**: Serializes stitched open and closed `<path d="M ... L ... Z"/>` curves grouped by `data-iso-level` alongside polylines, bars, box plots, and points.
- **Seamless 1:1 Tiles (`ExportOptions::seamless_tile`)**: Bypasses figure border padding to output exact unpadded grid dimensions for tileable textures.
- **16-Bit Heightmap & PBR Normal-Map PNGs**: `encode_heightmap_png_16bit` writes monotonic big-endian 16-bit grayscale PNGs (`0..=65535`), and `encode_normal_map_png` writes central-difference tangent-space RGB normal maps with optional toroidal wrap-around.

## History

This crate was originally hosted at <https://github.com/emilk/egui> but was extracted into its own repository on 2024-07-15.
