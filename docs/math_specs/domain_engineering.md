# Canonical Mathematical Specification: `domain_engineering`

Canonical governing equations, bibliography references, stability bounds, and validity domains for [`crates/domain_engineering`](../../crates/03_domains/domain_engineering/README.md), linked to [`papers/references.bib`](../../papers/references.bib) and workspace manuscripts.

## 1. Millimeter-Wave FMCW Radar, Bluestein Chirp Z-Transform, MIMO/MUSIC & Kalman Tracking (`radar`)

### Governing Equations

Linear frequency-modulated continuous-wave (`FmcwConfig`) radar relates beat frequency $\hat{f}_{\text{FFT}}$ and carrier phase shift $\Delta\phi$ to round-trip target range $R$ and sub-millimeter displacement $\Delta R$, while Bluestein fast-convolution (`chirp_z_transform`) evaluates narrowband zoom spectra over $K$ uniform bins:

$$
R = \frac{c\,T_c\,\hat{f}_{\text{FFT}}}{2 B}, \qquad \Delta R = \frac{\lambda\,\Delta\phi}{4\pi}, \qquad X_k = \sum_{n=0}^{N-1} x_n \exp\!\left(-j 2\pi \frac{f_0 + k\,\Delta f}{f_s} n\right)
$$

Uniform linear array steering (`Beamformer`), eigenspace subspace projection (`MusicEstimator`, $P_{\text{MUSIC}}(R) = \|\mathbf{a}^H(R)\mathbf{E}_n\|_2^{-2}$), sliding-window static clutter subtraction (`EllipticalFilter`), $2\pi$ phase continuation (`PhaseUnwrapper`), SVD least-squares torso fitting (`BiQuadraticSurface`, $z = c_0 + c_1 x + c_2 y + c_3 x y + c_4 x^2 + c_5 y^2$), and 1D `ConstantVelocityModel` discrete Kalman filtering (`TrackingFilter`) reconstruct smooth respiratory motion trajectories.

### Citations

- BibTeX keys: [`kammler2007first`](../../papers/references.bib#kammler2007first), [`parseval1806memoire`](../../papers/references.bib#parseval1806memoire), [`nalgebra`](../../papers/references.bib#nalgebra).
- Manuscript: [`papers/mmwave_radiotherapy_setup.tex`](../../papers/mmwave_radiotherapy_setup.tex).

### Stability

Power-of-two zero-padded FFT convolution (`conv_len >= N + K - 1`) eliminates circular aliasing in $O((N + K)\log(N + K))$ operations (with exact phasor summation for $N \le 4$ or $K \le 4$); `BiQuadraticSurface::fit` uses Golub-Kahan SVD least squares to handle rank-deficient point clouds gracefully; `MusicEstimator` validates subspace dimensions (`num_sources < subspace_dim <= n_samples`) via typed `RadarError`; `TrackingFilter` delegates covariance updates to `pure_math::pure_math::analysis::kalman::KalmanFilter`.

### Domain of Validity

Requires sweep bandwidth $B > 0$, chirp duration $T_c > 0$, carrier frequency $f_c > 0$, sampling rate $f_s > 0$ satisfying Nyquist bounds ($|f_0| + K|\Delta f| \le f_s / 2$ for unaliased real targets), at least $6$ non-collinear points for `BiQuadraticSurface::fit`, and finite time step $\Delta t > 0$ for `ConstantVelocityModel`.

```rust
use math_explorer::engineering::radar::FmcwConfig;

let cfg = FmcwConfig::iwr6843_default();
let slope = cfg.bandwidth / cfg.chirp_time;
let fb = FmcwConfig::beat_frequency(slope, 1.5);
assert!((cfg.range_from_beat_frequency(fb) - 1.5).abs() < 1e-12);
```

## 2. Coupled Electrochemical & Power-Law Battery Degradation (`battery_degradation`)

### Governing Equations

Empirical Wöhler cycle aging (`PowerLawModel`, `DegradationModel`) and coupled solid-electrolyte interphase (SEI) + Tafel lithium-plating degradation (`ElectrochemicalAgingModel`) evolve remaining state of health $\mathrm{SOH} \in [0, 1]$ across equivalent full cycles $N$, depth of discharge $D$, absolute temperature $T$, and charge C-rate $I_c$:

$$
Q_{\text{rem}}^{\text{power}}(N, D) = \operatorname{clamp}_{[0, 1]}\!\left(1 - k \left(\frac{D}{100}\right)^\alpha N^\beta\right), \qquad Q_{\text{loss}}^{\text{echem}} = k_{\text{SEI}}\exp\!\left(-\frac{E_{a,\text{SEI}}}{R_g}\Bigl(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\Bigr)\right)\sqrt{t} + k_{\text{pl}}\exp(\alpha_{\text{pl}} I_c)\exp\!\left(\frac{E_{a,\text{pl}}}{R_g}\Bigl(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\Bigr)\right) N
$$

Damped Levenberg-Marquardt calibration (`calibrate_power_law_from_csv`) fits $(k, \alpha, \beta)$ against experimental cycling traces, and Palmgren-Miner cumulative damage (`DutySchedule`, `evaluate_duty_lifetime`) integrates multi-segment thermal/C-rate mission profiles to end-of-life ($\mathrm{SOH} \le \mathrm{SOH}_{\text{EOL}}$).

### Citations

- BibTeX keys: [`pushpa2022eisenstein`](../../papers/references.bib#pushpa2022eisenstein) (DOI: `10.1007/s40065-022-00360-6`), [`boisvert1997`](../../papers/references.bib#boisvert1997).
- Literature: Bloom et al. (2001), *J. Power Sources* 101(2), 238–247, DOI: `10.1016/S0378-7753(01)00783-2`; Broussely et al. (2005), DOI: `10.1016/j.jpowsour.2005.03.172`.

### Stability

Remaining capacity is clamped to physical $[0, 1]$ bounds (`StateOfHealth`), Arrhenius exponents guard against non-positive Kelvin temperatures via typed `BatteryError`, and zero-allocation Levenberg-Marquardt damping bounds parameter step norms during nonlinear CSV calibration.

### Domain of Validity

Requires non-negative cycle count $N \ge 0$, depth of discharge $D \in [0, 100]\%$, strictly positive temperature $T > 0\,\text{K}$, non-negative C-rates $I_c \ge 0$, and non-empty monotonically ordered cycle telemetry for calibration.

```rust
use math_explorer::engineering::battery_degradation::{
    Cycles, DegradationModel, DepthOfDischarge, PowerLawModel,
};

let model = PowerLawModel::standard();
let soh = f64::from(model.capacity(
    Cycles::new_clamped(100.0),
    DepthOfDischarge::new_clamped(80.0),
));
assert!(soh > 0.7 && soh < 1.0);
```

## 3. CNC Taylor Tool-Life Prognostics & Asynchronous UART Throughput (`cnc`, `uart`)

### Governing Equations

Taylor's machining wear law (`calculate_taylor_tool_life`) relates cutting velocity $V$ ($\text{m/min}$), tool-workpiece constant $C$, and Taylor exponent $n$ to expected tool life $T$ ($\text{min}$), while asynchronous serial framing (`calculate_uart_throughput`) maps symbol rate $B$ ($\text{baud}$), payload bits $N_{\text{data}}$, and total frame bits $N_{\text{total}}$ to effective payload bitrate $R_{\text{eff}}$:

$$
V\,T^n = C \iff T = \left(\frac{C}{V}\right)^{1/n}, \qquad R_{\text{eff}} = B \cdot \frac{N_{\text{data}}}{N_{\text{total}}}
$$

### Citations

- Literature: Taylor (1907), *On the Art of Cutting Metals*, *Trans. ASME* 28, 31–350; TIA/EIA-232-F asynchronous serial framing standard.

### Stability

Newtype constructors (`CuttingSpeed::new`, `MaterialConstant::new`, `ToolExponent::new`, `TotalBits::new`) reject non-positive cutting speeds ($V \le 0$), negative material constants ($C < 0$), zero exponents ($n = 0$), and zero-bit frames ($N_{\text{total}} = 0$) via typed `HardwareError::InvalidParameter`, eliminating division-by-zero and `NaN` fractional-power singularities.

### Domain of Validity

Requires strictly positive cutting speed $V > 0$, non-negative material constant $C \ge 0$, non-zero tool exponent $n \ne 0$ (typically $n \in [0.08, 0.5]$ across HSS, carbide, and ceramic tooling), and frame sizes satisfying $1 \le N_{\text{data}} \le N_{\text{total}}$ (e.g., $8/10$ for standard `8N1`).

```rust
use math_explorer::engineering::{
    BaudRate, CuttingSpeed, DataBits, MaterialConstant, ToolExponent, TotalBits,
    calculate_taylor_tool_life, calculate_uart_throughput,
};

let life = calculate_taylor_tool_life(
    CuttingSpeed::new(40.0).unwrap(),
    MaterialConstant::new(120.0).unwrap(),
    ToolExponent::new(0.25).unwrap(),
);
assert!((life.value() - 81.0).abs() < 1e-12);

let bps = calculate_uart_throughput(
    BaudRate::new(115_200),
    DataBits::new(8),
    TotalBits::new(10).unwrap(),
);
assert!((bps - 92_160.0).abs() < 1e-9);
```
