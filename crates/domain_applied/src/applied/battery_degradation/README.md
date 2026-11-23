# Battery Degradation Modeling

![Status](https://img.shields.io/badge/status-stable-brightgreen)
![Domain](https://img.shields.io/badge/domain-applied-blue)

A zero-allocation framework for Li-ion battery cycle-life projection, coupled electrochemical aging, stack-allocated Levenberg-Marquardt CSV calibration, and multi-segment duty warranty analysis.

> **"Code explains HOW; Docs explain WHY."**

---

## Install

This module is part of the `math_explorer` core library. To use it, include the library in your `Cargo.toml`.

```toml
[dependencies]
math_explorer = { path = "path/to/math_explorer", features = ["applied"] }
```

---

## Usage

You can see these models in action by running the standalone, fully executable example:

```bash
cargo run --release --package math_explorer --features applied --example battery_degradation_demo
```

*(See `math_explorer/examples/battery_degradation_demo.rs` for implementation details)*

### Quick Start: Coupled Electrochemical Aging & Duty Profiles

`ElectrochemicalAgingModel` couples Depth-of-Discharge stress $d^\beta$, stretched exponent $z \in [0.1, 2.0]$, Arrhenius thermal acceleration $\exp\left(\frac{E_a}{R}\left(\frac{1}{T_{\text{ref}}} - \frac{1}{T}\right)\right)$ normalized at $25\,^\circ\text{C}$, C-rate mechanical stress $I_c^{\gamma_c}$, and parabolic $\sqrt{t}$ SEI calendar fade across [`ChemistryPreset`](crate::applied::battery_degradation::ChemistryPreset) profiles (`Nmc811`, `Lfp`, `Nca`, `Lto`), while preserving [`PowerLawModel`](crate::applied::battery_degradation::PowerLawModel) for isothermal DoD baselines:

```rust
# use domain_applied::__doctest_facade as math_explorer;
use math_explorer::applied::battery_degradation::{
    Capacity, ChemistryPreset, Cycles, DepthOfDischarge, DutySegment, ElectrochemicalAgingModel,
    PowerLawModel, estimate_duty_lifetime,
};

fn main() {
    let dod = DepthOfDischarge::new_clamped(60.0);

    // 1. Isothermal Power-Law baseline
    let baseline = PowerLawModel::standard();
    assert!(baseline.n70(dod).as_f64() > 500.0);

    // 2. Coupled NMC811 aging at 25 °C vs 45 °C
    let nmc_25c = ElectrochemicalAgingModel::from_preset(ChemistryPreset::Nmc811);
    let nmc_45c = nmc_25c.with_temperature_c(45.0).with_c_rate(1.5);
    assert!(nmc_45c.n70(dod).as_f64() < nmc_25c.n70(dod).as_f64());

    // 3. Multi-segment duty lifetime (weekday commute + fast-charge + idle storage)
    let duty = [
        DutySegment::commute(),
        DutySegment::fast_charge(),
        DutySegment::storage(1.0, 22.0),
    ];
    let summary = estimate_duty_lifetime(&nmc_25c, &duty, Capacity::new_clamped(0.80));
    assert!(summary.years_to_80 > 0.0 && summary.years_to_70 > summary.years_to_80);
    let _ = Cycles::new_clamped(summary.cycles_to_80);
}
```

---

## Config

- [`model`](crate::applied::battery_degradation::model): `PowerLawModel`, `ChemistryPreset`, and `ElectrochemicalAgingModel`.
- [`calibration`](crate::applied::battery_degradation::calibration): Stack-allocated Levenberg-Marquardt `(cycles, capacity)` fitter.
- [`duty`](crate::applied::battery_degradation::duty): Multi-segment duty profile evaluation and `80%` / `70%` SOH warranty solvers.
- [`types`](crate::applied::battery_degradation::types): Type-safe wrappers for `Capacity`, `Cycles`, and `DepthOfDischarge`.

---

## Contributing
See the [Project Contributing Guide](../../../../../CONTRIBUTING.md) for details on adding new architectures, models, or algorithms.
