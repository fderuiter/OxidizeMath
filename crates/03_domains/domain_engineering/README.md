# Domain Engineering

![Maturity: Validated](https://img.shields.io/badge/maturity-Validated-2f81f7.svg)

RF/mmWave radar signal processing, coupled electrochemical battery degradation, CNC tool-life prognostics, and serial communication throughput solvers.

> **Maturity Tier:** `Validated` — deterministic numerical and analytical engineering solvers backed by canonical mathematical specifications (`docs/math_specs/domain_engineering.md`), zero-allocation hot paths, and `Certified` radar and battery aging verification suites.

## 1. Overview & Architecture

The `domain_engineering` crate consolidates Tier 3 electrical, mechanical, electrochemical, and RF signal-processing models into a dedicated peer domain crate depending strictly on Tier 0–2 engines and `pure_math`.

### Module Structure

- **`radar`**: Millimeter-wave Frequency-Modulated Continuous-Wave (FMCW) radar signal processing:
  - `FmcwConfig`: Beat-frequency range inversion, range resolution $\Delta R = c / (2B)$, Doppler radial velocity, and sub-millimeter phase displacement.
  - `czt`: Bluestein FFT Chirp Z-Transform (`chirp_z_transform`, `SpatialCztConfig`) for zoom spectral estimation.
  - `mimo` & `super_resolution`: Uniform linear array `Beamformer` steering and eigenspace `MusicEstimator` super-resolution range estimation.
  - `clutter`, `phase`, `geometry`, `surface`, & `tracking`: CA-CFAR noise estimation, `EllipticalFilter` static clutter rejection, `PhaseUnwrapper`, sensor-to-target transforms, SVD `BiQuadraticSurface` fitting, and 1D `ConstantVelocityModel` Kalman `TrackingFilter`.
- **`battery_degradation`**: Rechargeable cell capacity-fade and warranty lifetime modeling:
  - `PowerLawModel` & `ElectrochemicalAgingModel`: Wöhler DoD stress, stretched cycling exponent $z$, Arrhenius thermal acceleration, C-rate stress, and $\sqrt{t}$ SEI calendar fade across `ChemistryPreset` (`Nmc811`, `Lfp`, `Nca`, `Lto`).
  - `calibration` & `duty`: Zero-allocation Levenberg-Marquardt `(cycles, capacity)` curve fitting and multi-segment duty-cycle warranty solvers (`estimate_duty_lifetime`, `evaluate_duty_point`).
- **`cnc`**: Predictive machining maintenance via Taylor's tool-life equation $T = (C / V)^{1/n}$ (`calculate_taylor_tool_life`).
- **`uart`**: Asynchronous serial frame efficiency and effective bit-rate calculation (`calculate_uart_throughput`).

---

## 2. Developer Instructions

### Evaluating Radar, Battery Aging, and Hardware Solvers

```rust
# use domain_engineering::__doctest_facade as math_explorer;
use math_explorer::engineering::battery_degradation::{
    ChemistryPreset, DepthOfDischarge, ElectrochemicalAgingModel,
};
use math_explorer::engineering::radar::FmcwConfig;
use math_explorer::engineering::{
    BaudRate, CuttingSpeed, DataBits, MaterialConstant, ToolExponent, TotalBits,
    calculate_taylor_tool_life, calculate_uart_throughput,
};

// 1. 60 GHz FMCW radar range resolution (4 GHz sweep -> ~3.75 cm)
let radar = FmcwConfig::iwr6843_default();
assert!((radar.range_resolution() - 0.0375).abs() < 1e-4);

// 2. Coupled LFP battery cycle life at 60% DoD
let lfp = ElectrochemicalAgingModel::from_preset(ChemistryPreset::Lfp);
assert!(lfp.n70(DepthOfDischarge::new_clamped(60.0)).as_f64() > 2000.0);

// 3. Taylor CNC tool life & 8N1 UART effective throughput
let life = calculate_taylor_tool_life(
    CuttingSpeed::new(50.0).unwrap(),
    MaterialConstant::new(100.0).unwrap(),
    ToolExponent::new(0.25).unwrap(),
);
assert!((life.value() - 16.0).abs() < 1e-12);

let bps = calculate_uart_throughput(
    BaudRate::new(115_200),
    DataBits::new(8),
    TotalBits::new(10).unwrap(),
);
assert_eq!(bps, 92_160.0);
```

---

## 3. Ecosystem Integration

`domain_engineering` is re-exported through `oxidizemath::engineering` and `math_explorer::engineering` under the `engineering` Cargo feature, while maintaining facade compatibility for clinical radar gating and battery state-of-health tools in `math_explorer_gui`.
