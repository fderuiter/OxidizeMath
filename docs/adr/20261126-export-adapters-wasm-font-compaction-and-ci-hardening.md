# ADR 20261126: Typed Export Adapters, Build-Time WASM TrueType Compaction, and CI Pipeline Hardening

## Status

Accepted

## Context

Expanding dataset and figure export parity across all 111 interactive `MathExplorerApp` tools (`108` figure-and-dataset tools plus `3` `dataset_only()` analytical calculators) alongside the five-tier workspace migration surfaced three cross-cutting engineering constraints in continuous integration:

1. **WebAssembly Bundle Footprint & Cold-Boot Preload Parity**: Adding typed table and primitive export adapters across 17 simulation tools increased `math_explorer_gui_bg.wasm` by ~600 KB `gzip -9`, driven alongside `1.41 MB` raw (`772 KB` `gzip -9`) of embedded `epaint_default_fonts` TrueType tables (`NotoEmoji-Regular.ttf`, `Ubuntu-Light.ttf`, `emoji-icon-font.ttf`, and `Hack-Regular.ttf`) containing thousands of unreferenced glyphs, TrueType bytecode hinting programs, and unused OpenType layout tables—while cold-boot service-worker preload invariants forbid adding extra HTTP font requests.
2. **WASM Test Font Isolation**: Disabling `eframe` `default_fonts` on `wasm32` production builds requires `egui_plot` `[dev-dependencies]` to opt into `egui/default_fonts` explicitly so standalone `wasm32` unit and modal tests retain deterministic glyph metrics.
3. **Least-Privilege & Deterministic GitHub Actions Hardening**: `.github/workflows/ci.yml` previously used broad `permissions: read-all` at the root level, ran unpinned `bnjbvr/cargo-machete` (querying GitHub's unauthenticated `releases/latest` endpoint and interpolating `${{ inputs.args }}` in `run:`), interpolated `${{ matrix.feature }}` and `${{ matrix.target }}` directly inside `run:` scripts, and left `cargo-bins/cargo-binstall` unpinned in `.github/actions/ci-tool-cache/action.yml`.

## Decision

1. **Typed Table & Primitive Export Adapters**: `math_explorer_gui::framework::export_tables` introduces `ExportCell` and `ExportTable` (with deterministic stride capping at `65,536` rows and CWE-1236 CSV/JSON escaping), wired across 3 `dataset_only()` calculators, 8 2D raster simulations, and 6 `ExportPrimitive` visual simulations.
2. **Deterministic Build-Time WASM TrueType Compaction**: Disable `default_fonts` on `eframe` for `wasm32` production builds (preserving `default_fonts` on native desktop builds and `wasm32` `dev-dependencies`, including `egui_plot` `[dev-dependencies]`) and generate compacted TrueType subsets in `apps/math_explorer_gui/build_stage.rs` into `$OUT_DIR/wasm_fonts/`. Non-essential SFNT hinting/layout tables (`post` v2.0 -> v3.0, `kern`, `fpgm`, `prep`, `cvt `, `GPOS`, `GSUB`, `GDEF`, `hdmx`, `VDMX`, `LTSH`, `DSIG`, `meta`) are stripped while preserving `cmap`, `glyf`, `loca`, `head`, `hhea`, `hmtx`, `maxp`, `name`, and `OS/2` with recomputed SFNT checksums, and `MathExplorerApp::new` installs the compacted fonts before Frame 1.
3. **Least-Privilege & Deterministic GitHub Actions Hardening**: Narrow root `permissions` in `.github/workflows/ci.yml` from `read-all` to `contents: read`, pin `cargo-binstall` (`1.17.9`) and `cargo-machete` (`0.9.1`) through `./.github/actions/ci-tool-cache`, and bind `${{ matrix.feature }}` and `${{ matrix.target }}` via `env:` rather than interpolating directly in `run:`.

## Consequences

- **Positive**: `math_explorer_gui_bg.wasm` compresses to ~`3.26 MB` (`brotli -9`) and ~`3.96 MB` (`gzip -9`)—comfortably below the `5 MB` (`5,242,880` byte) workspace ceiling—while preserving cold-boot zero-extra-request preload parity and full typography across ASCII, Greek, mathematical operators, UI icons, and workspace emojis.
- **Positive**: All CI jobs (`pre-flight`, `build-and-test` across Linux/Windows/macOS/WASM, `feature-matrix`, and `package`) execute deterministically with pinned binary caches and least-privilege `GITHUB_TOKEN` scopes.
- **Trade-off**: Custom OpenType GPOS/GSUB ligatures and TrueType bytecode instructions are omitted on `wasm32` in favor of `ab_glyph`'s anti-aliased outline rasterizer.
