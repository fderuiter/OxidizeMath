# 10. Theme State Management and Menu Toggles

## Status
Accepted

## Context
`MathExplorerApp` previously rendered exclusively in egui's default dark mode without providing user controls or application state fields to toggle themes. Users requiring high contrast for accessibility or light mode for classroom projection and printing had no mechanism to adjust visual themes.

## Decision
1. Introduced a `ThemeMode` enum (`Dark`, `Light`, `HighContrast`) with custom egui `Visuals` generation logic for each mode.
2. Added a `theme: ThemeMode` state field to `MathExplorerApp` and `AppState` with `serde` persistence support.
3. Updated `MathExplorerApp::update()` to call `ctx.set_visuals()` matching `ThemeMode` when theme selection changes or when initial frame state is established.
4. Added an `Appearance` submenu with options for `Dark`, `Light`, and `HighContrast` in native macOS `muda` menus and fallback egui menu bars.
5. Consolidated duplicate test modules and added comprehensive unit tests for theme transitions and visual state synchronization.

## Consequences
- Users can toggle between Dark, Light, and High Contrast visual themes dynamically.
- Theme preferences are saved to application storage and restored across application restarts.
- Visuals are synchronized seamlessly on frame updates without unnecessary re-allocations.
