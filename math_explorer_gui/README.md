# Math Explorer GUI

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Framework](https://img.shields.io/badge/framework-egui-blue)

The graphical front-end for the `math_explorer` library, built using `eframe` (egui framework).

##  Install

Ensure you have the Rust toolchain installed. This crate is part of the `math_explorer` workspace.

```bash
git clone https://github.com/fderuiter/OxidizeMath.git
cd OxidizeMath
```

##  Usage

### Desktop (Native)

To run the interactive application natively:

```bash
cargo run --release --package math_explorer_gui
```

### Web (WASM)

To compile and run the application in your browser (requires Trunk):

```bash
# 1. Install Trunk and WASM target
cargo install trunk
rustup target add wasm32-unknown-unknown

# 2. Serve the application locally
cd math_explorer_gui
trunk serve
```

##  Config (Architecture)

The GUI follows a modular architecture based on the `ExplorerTab` trait. The main application `MathExplorerApp` manages a list of these tabs and handles navigation.

```mermaid
classDiagram
    class MathExplorerApp {
        -tabs: Vec~Box~dyn ExplorerTab~~
        -selected_tab: usize
        +update(ctx, frame)
    }

    class ExplorerTab {
        <<interface>>
        +name() String
        +show(ctx, frame)
    }

    class MriTab {
        -simulator: BlochSimulator
        +show()
    }

    class MorphogenesisTab {
        -system: TuringSystem
        +show()
    }

    MathExplorerApp o-- ExplorerTab
    ExplorerTab <|.. MriTab
    ExplorerTab <|.. MorphogenesisTab
```

### Key Components

*   **`MathExplorerApp`**: The root component implementing `eframe::App`. It renders the top navigation bar and delegates the main content area to the active tab.
*   **`ExplorerTab`**: The trait that all visualization modules must implement. It provides a standard interface for the app to interact with different simulations.
*   **`math_explorer` Dependency**: All heavy mathematical logic (physics, biology, etc.) resides in the core `math_explorer` crate. The GUI crate should focus **only** on visualization and input handling.

##  Contributing

Follow these steps to add a new visualization module (e.g., for `fluid_dynamics`).

### 1. Create the Module
You can create a single file `src/tabs/fluid_dynamics.rs` or a directory `src/tabs/fluid_dynamics/mod.rs` for complex modules.

**Example `src/tabs/fluid_dynamics.rs`:**

```text
use crate::tabs::ExplorerTab;
use eframe::egui;

pub struct FluidDynamicsTab {
    // State goes here
    velocity: f64,
}

impl Default for FluidDynamicsTab {
    fn default() -> Self {
        Self { velocity: 0.0 }
    }
}

impl ExplorerTab for FluidDynamicsTab {
    fn name(&self) -> &'static str {
        "Fluid Dynamics"
    }

    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Fluid Simulation");
            ui.add(egui::Slider::new(&mut self.velocity, 0.0..=100.0).text("Velocity"));
        });
    }
}
```

### 2. Register the Module
In `src/tabs/mod.rs`, add your module:

```text
pub mod fluid_dynamics;
```

### 3. Add to App
In `src/app.rs` (or wherever the tabs are initialized, likely in `MathExplorerApp::default`), add your tab to the list:

```text
Box::new(FluidDynamicsTab::default()),
```

### Design Principles

1.  **Separation of Concerns**: Keep simulation logic in `math_explorer`. If you need a new simulation model, add it to the core library first, then visualize it here.
2.  **Immediate Mode**: `egui` is an immediate mode GUI. This means the `show()` function is called every frame. Avoid heavy computations directly in `show()`. Use a separate thread or perform small updates per frame.
3.  **No Heavy Dependencies**: We try to keep the GUI crate lightweight. Avoid adding heavy dependencies like `rand` if a simple internal implementation or a dependency from `math_explorer` suffices.
