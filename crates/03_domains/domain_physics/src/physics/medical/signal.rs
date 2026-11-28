/// Models a Dirac Delta Composite Function for beam pulses.
///
/// $$ g(x) = \sum_{n=0}^{(t_{off} - t_{on})/\Delta t} \delta(x - x_n) $$
///
/// In practice, discrete modeling represents this as a sequence of pulses.
/// This function returns the number of pulses in the window.
///
/// # Arguments
///
/// * `t_on` ($t_{on}$) - Start time.
/// * `t_off` ($t_{off}$) - End time.
/// * `delta_t` ($\Delta t$) - Pulse interval.
///
/// # Returns
///
/// * `usize` - Number of pulses.
#[verified_engine::verified]
pub fn dirac_pulse_count(t_on: f64, t_off: f64, delta_t: f64) -> usize {
    if delta_t <= 0.0 || t_off < t_on {
        return 0;
    }
    ((t_off - t_on) / delta_t).floor() as usize + 1
}

/// Calculates the Signal-Front Delay.
///
/// $$ t_{\text{delay}} = \lim_{\omega \to \infty} (\frac{\phi(\omega)}{\omega}) $$
///
/// Represents the delay of the wavefront (signal front) in a dispersive medium.
///
/// # Arguments
///
/// * `phase_slope_high_freq` - The limit of $\phi(\omega) / \omega$ as $\omega \to \infty$.
///
/// # Returns
///
/// * `f64` - The delay time.
#[verified_engine::verified]
pub fn signal_front_delay(phase_slope_high_freq: f64) -> f64 {
    phase_slope_high_freq
}
