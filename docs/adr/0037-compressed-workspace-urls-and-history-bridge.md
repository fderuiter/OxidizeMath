# ADR 0037: DEFLATE-Compressed Workspace Share URLs and Browser History Bridge

## Status

Accepted

## Context

`math_explorer_gui::url_state` previously encoded only a single simulation pane via `#tab=<slug>&tool=<id>&state=<base64url>` with a 4,096-byte raw JSON cap, unconditionally collapsed multi-pane viewports (`SplitHorizontal`, `SplitVertical`, `QuadGrid`) back to `ViewportLayout::Single` during startup hydration, lacked `popstate` / `hashchange` listeners for same-document browser navigation, and persisted only raw `AppState` JSON (omitting `undo_stack` and `redo_stack`) to `eframe::Storage` (`eframe::APP_KEY`). As a consequence:

1. Multi-pane comparative workspaces and rich single-pane tools (`GraphEditorTool`, `PotentialFlowTool`, `FavoritismTool`, `ReplicatorDynamicsTool`, `NeuralNetworkVizTool`) whose presets exceed 4,096 bytes could not sync or share via URLs.
2. Self-synced `#tab=...` fragments overwrote `localStorage` multi-pane layouts and wiped workspace undo stacks on browser reloads and `#webgl_reload_button` recovery restarts.
3. Non-share document anchors (`#overview`, `#utm_source=docs`) and invalid share URLs reset active multi-pane layouts, while in-tab Back/Forward or hash edits were ignored.

## Decision

1. **Version 2 DEFLATE `#ws=<base64url>` Workspace Codec (`url_state::workspace_codec`):**
   - Depend directly on pure-Rust `flate2` (`1.1.11`, `default-features = false`, `features = ["rust_backend"]`, already transitive in `Cargo.lock`) to compress `ViewportLayout`, clamped split ratios (`split_ratio`, `quad_split_ratios`), `focused_pane`, `ComparisonSyncConfig`, and `1..=4` `WirePaneShareEntry` descriptors.
   - Enforce a strict `32,768`-byte decompression ceiling (`DeflateDecoder::take(32_769)`) prior to JSON parsing (zip-bomb defense) and retain the `8,192`-byte `MAX_URL_FRAGMENT_BYTES` address-bar limit.
   - Preserve `#tab=<slug>&tool=<id>&state=<base64url>` for single-pane workspaces whose raw preset JSON fits within `4,096` bytes, automatically upgrading multi-pane workspaces and `4 KiB < state <= 32 KiB` single-pane states to `#ws=<base64url>`, and clearing `location.hash` via `replace_browser_url_fragment("")` (with `.mws` export guidance on `Copy Share Link`) when states exceed URL budgets.

2. **Self-Sync `History.state` Tagging, `WorkspaceEnvelope` Storage, and Atomic Hydration:**
   - Persist compact `WorkspaceEnvelope` JSON (including `history_undo_stack`, `history_redo_stack`, and `callout_manager`) to `eframe::Storage` (`eframe::APP_KEY`) and tag self-synced `History.replaceState` calls with `{ "__oxidize_self_sync": true, "fragment": hash_part }`.
   - Preserve the `localStorage` `WorkspaceEnvelope` and undo history across reloads and WebGL recovery when `History.state` matches the self-synced fragment, while pushing the prior `AppState` onto `undo_stack` before applying external share URLs.
   - Return `Ok(None)` for non-share anchors without warnings, and pre-build all candidate `ExplorerTab` instances before mutating `MathExplorerApp` so invalid `#tab=...` or `#ws=...` URLs fail atomically while preserving active multi-pane layouts.

3. **WHATWG `popstate` and `hashchange` Window Listeners (`url_state::history_bridge`):**
   - Register CSP-safe `Closure::wrap` listeners on `web_sys::Window` for `popstate` and `hashchange`, deduplicating paired browser dispatches, suppressing self-synced echoes, requesting an immediate `egui::Context::request_repaint()`, and recording workspace undo entries on external in-tab navigations.

## Consequences

- **Positive:** All 8 multi-pane focus configurations and large-state tools up to 32 KiB round-trip losslessly via compact `#ws=<base64url>` links (`<= 8 KiB`), browser reloads and `#webgl_reload_button` recovery preserve multi-pane workspaces and undo stacks, and Back/Forward history navigation integrates cleanly with `Ctrl+Z` / `Cmd+Z` undo.
- **Positive:** Zero new crates are added to `Cargo.lock`, `.wasm.gz` / `.wasm.br` bundle budgets remain well below the 3 MiB ceiling, and all legacy `#tab=...` URLs remain 100% backward compatible.
- **Neutral:** Global `Copy Share Link` (`File -> Copy Share Link` / `Ctrl+Shift+L` / `Cmd+Shift+L`) copies full workspace URLs, while per-pane action-bar `Copy Share Link` buttons copy single-pane URLs (`#tab=...` or 1-pane `#ws=...`).
