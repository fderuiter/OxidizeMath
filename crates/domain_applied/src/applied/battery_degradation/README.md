# Battery Degradation Modeling

![Status](https://img.shields.io/badge/status-stable-brightgreen)
![Domain](https://img.shields.io/badge/domain-applied-blue)

A framework to estimate battery cycle life and capacity fade for Li-ion batteries based on Depth-of-Discharge (DoD).

> **"Code explains HOW; Docs explain WHY."**

---

## Install

This module is part of the `math_explorer` core library. To use it, include the library in your `Cargo.toml`.

```toml
[dependencies]
math_explorer = { path = "path/to/math_explorer" }
```

---

## Usage

You can see this model in action by running the standalone, fully executable example:

```bash
cargo run --release --package math_explorer --example battery_degradation_demo
```

*(See `math_explorer/examples/battery_degradation_demo.rs` for implementation details)*

### Quick Start

The degradation is modeled using a Power Law fit to experimental data:
$$ N_{70}(d) = \alpha \cdot d^\beta $$

Where:
- $d$ is the Depth of Discharge (0-100%).
- $N_{70}$ is the number of equivalent full cycles until the battery reaches 70% capacity.
- $\alpha, \beta$ are empirical constants (Standard Li-ion: $\alpha \approx 1.019 \times 10^5, \beta \approx -1.26$).

This implies that **shallower discharges drastically increase cycle life**.

```rust
use domain_applied::applied::battery_degradation::{PowerLawModel, DepthOfDischarge, Cycles, Capacity};

fn main() {
    // 1. Initialize the standard model
    let model = PowerLawModel::standard();

    // 2. Define a scenario: 80% to 20% charge window = 60% DoD
    let dod = DepthOfDischarge::new_clamped(60.0);

    // 3. Estimate Life Expectancy (Cycles to 70% SOH)
    let life_cycles = model.n70(dod);
    println!("Expected Life: {:.0} cycles", life_cycles/ Cycles::new_clamped(1.0));

    // 4. Predict Capacity after 1000 cycles
    let current_cycles = Cycles::new_clamped(1000.0);
    let remaining_capacity = model.capacity(current_cycles, dod);
    println!("Capacity after 1000 cycles: {:.1}%", remaining_capacity/ Capacity::new_clamped(1.0) * 100.0);
}
```

---

## Config

- [`model`](crate::applied::battery_degradation::model): Core logic including the `PowerLawModel` struct.
- [`types`](crate::applied::battery_degradation::types): Type-safe wrappers for `Capacity`, `Cycles`, and `DepthOfDischarge`.

---

## Contributing
See the [Project Contributing Guide](../../../../../CONTRIBUTING.md) for details on adding new architectures, models, or algorithms.
