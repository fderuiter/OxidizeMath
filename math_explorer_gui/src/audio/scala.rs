//! Zero-dependency Huygens-Fokker Scala `.scl` scale and `.kbm` keyboard-mapping serializers.
//!
//! Converts prime harmonic factorizations ([`just_intonation_rational`]) and 2D Chladni / PDE
//! modal frequency ratios into standard ASCII `.scl` and `.kbm` files compatible with
//! `tuning-library` synthesizers including Surge XT and Vital.

// theory_verification!

use super::mappings::number_theory::{just_intonation_rational, PRIME_CHORD_ROOT_HZ};

/// Default reference MIDI key (`57` = `A3`) paired with [`PRIME_CHORD_ROOT_HZ`] (`220.0 Hz`).
pub const SCALA_DEFAULT_ROOT_MIDI: u8 = 57;

/// Maximum number of modal or harmonic degrees emitted into a single `.scl` scale.
pub const MAX_SCALA_DEGREES: usize = 32;

/// Single pitch entry above `1/1` in a Huygens-Fokker `.scl` scale definition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalaPitch {
    /// Exact positive rational ratio `num/den` (`num > den >= 1`), serialized without `.`.
    Ratio(u64, u64),
    /// Logarithmic interval in cents (`> 0.0`), serialized with fixed decimal `.`.
    Cents(f64),
}

impl ScalaPitch {
    /// Evaluates this pitch entry in logarithmic cents (`1200 * log2(ratio)`).
    #[must_use]
    pub fn cents(self) -> f64 {
        match self {
            Self::Ratio(num, den) => {
                let d = den.max(1) as f64;
                let n = num.max(1) as f64;
                (1_200.0 * (n / d).log2()).max(0.01)
            }
            Self::Cents(c) => {
                if c.is_finite() {
                    c.clamp(0.01, 9_600.0)
                } else {
                    1_200.0
                }
            }
        }
    }

    /// Formats the pitch line per the Scala specification (no `.` for ratios, `.` for cents).
    #[must_use]
    pub fn format_scl_token(self) -> String {
        match self {
            Self::Ratio(num, den) => {
                let d = den.max(1);
                let n = num.max(d + 1);
                format!("{n}/{d}")
            }
            Self::Cents(_) => format!("{:.5}", self.cents()),
        }
    }
}

/// Complete Huygens-Fokker microtonal scale and root-key mapping model.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalaScale {
    /// Sanitized stem name used in `! <stem>.scl` and `! <stem>.kbm` headers.
    pub stem: String,
    /// Human-readable single-line scale description required on `.scl` line 1.
    pub description: String,
    /// Strictly ascending pitch entries above `1/1`, ending at the formal period (`2/1` or cents).
    pub pitches: Vec<ScalaPitch>,
}

impl ScalaScale {
    /// Constructs a just-intonation `.scl` scale from prime-power pairs `(prime, exponent)`.
    #[must_use]
    pub fn from_prime_factors(stem: &str, description: &str, factors: &[(u64, u32)]) -> Self {
        let mut ratios: Vec<(u64, u64)> = Vec::with_capacity(factors.len() + 1);
        for &(prime, exp) in factors {
            if prime <= 1 || exp == 0 {
                continue;
            }
            let (num, den) = just_intonation_rational(prime);
            if num > den && (num, den) != (2, 1) && !ratios.contains(&(num, den)) {
                ratios.push((num, den));
            }
        }
        ratios.sort_by(|&(n1, d1), &(n2, d2)| {
            (n1 as u128 * d2 as u128).cmp(&(n2 as u128 * d1 as u128))
        });
        ratios.truncate(MAX_SCALA_DEGREES - 1);

        let mut pitches: Vec<ScalaPitch> = ratios
            .into_iter()
            .map(|(n, d)| ScalaPitch::Ratio(n, d))
            .collect();
        pitches.push(ScalaPitch::Ratio(2, 1));

        Self {
            stem: sanitize_scala_stem(stem),
            description: sanitize_scala_line(
                description,
                "OxidizeMath Just Intonation Prime Scale",
            ),
            pitches,
        }
    }

    /// Constructs an inharmonic or modal `.scl` scale from frequency ratios (`f_n / f_1 > 1.0`).
    #[must_use]
    pub fn from_modal_ratios(stem: &str, description: &str, ratios: &[f64]) -> Self {
        let mut cents_list: Vec<f64> = Vec::with_capacity(ratios.len() + 1);
        for &r in ratios {
            if r.is_finite() && r > 1.000_5 {
                let cents = (1_200.0 * r.log2()).clamp(1.0, 4_800.0);
                cents_list.push(cents);
            }
        }
        cents_list.sort_by(|a, b| a.total_cmp(b));

        let mut deduped: Vec<ScalaPitch> = Vec::with_capacity(MAX_SCALA_DEGREES);
        for c in cents_list {
            if deduped.len() + 1 >= MAX_SCALA_DEGREES {
                break;
            }
            if deduped
                .last()
                .is_none_or(|prev| (c - prev.cents()).abs() >= 0.5)
            {
                deduped.push(ScalaPitch::Cents(c));
            }
        }

        let max_cents = deduped.last().map_or(0.0, |p| p.cents());
        if deduped.is_empty() || max_cents < 1_199.5 {
            deduped.push(ScalaPitch::Cents(1_200.0));
        }

        Self {
            stem: sanitize_scala_stem(stem),
            description: sanitize_scala_line(description, "OxidizeMath Modal Resonance Scale"),
            pitches: deduped,
        }
    }

    /// Serializes this scale to a Huygens-Fokker `.scl` ASCII document.
    #[must_use]
    pub fn to_scl_string(&self) -> String {
        let count = self.pitches.len();
        let mut out = String::with_capacity(160 + count * 20);
        out.push_str(&format!("! {}.scl\n", self.stem));
        out.push_str("! Generated by OxidizeMath\n");
        out.push_str(&self.description);
        out.push('\n');
        out.push_str(&format!(" {count}\n!\n"));
        for pitch in &self.pitches {
            out.push(' ');
            out.push_str(&pitch.format_scl_token());
            out.push('\n');
        }
        out
    }

    /// Serializes a matching `.kbm` root-key mapping compatible with Surge XT and Vital.
    #[must_use]
    pub fn to_kbm_string(&self, root_midi_note: u8, root_freq_hz: f64) -> String {
        let degree_count = self.pitches.len().max(1);
        let root_note = root_midi_note.min(127);
        let ref_hz = if root_freq_hz.is_finite() {
            root_freq_hz.clamp(20.0, 8_000.0)
        } else {
            f64::from(PRIME_CHORD_ROOT_HZ)
        };

        let mut out = String::with_capacity(320 + degree_count * 6);
        out.push_str(&format!("! {}.kbm\n", self.stem));
        out.push_str("! Huygens-Fokker keyboard mapping for Surge XT and Vital\n");
        out.push_str(&format!("! Size of map:\n{degree_count}\n"));
        out.push_str("! First MIDI note number to retune:\n0\n");
        out.push_str("! Last MIDI note number to retune:\n127\n");
        out.push_str(&format!(
            "! Middle note where first entry of mapping is mapped to:\n{root_note}\n"
        ));
        out.push_str(&format!(
            "! Reference note for which frequency is given:\n{root_note}\n"
        ));
        out.push_str(&format!(
            "! Frequency to tune the above note to (Hz):\n{ref_hz:.6}\n"
        ));
        out.push_str(&format!(
            "! Scale degree to consider as formal octave:\n{degree_count}\n"
        ));
        out.push_str("! Mapping:\n");
        for deg in 0..degree_count {
            out.push_str(&format!("{deg}\n"));
        }
        out
    }

    /// Convenience helper returning `(scl_string, kbm_string)` tuned to `A3 = 220.0 Hz` (`MIDI 57`).
    #[must_use]
    pub fn to_scl_and_kbm_default(&self) -> (String, String) {
        (
            self.to_scl_string(),
            self.to_kbm_string(SCALA_DEFAULT_ROOT_MIDI, f64::from(PRIME_CHORD_ROOT_HZ)),
        )
    }
}

fn sanitize_scala_stem(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .take(48)
        .collect();
    let trimmed = cleaned.trim_matches('_');
    if trimmed.is_empty() {
        "oxidizemath_scale".to_string()
    } else {
        trimmed.to_string()
    }
}

fn sanitize_scala_line(raw: &str, fallback: &str) -> String {
    let cleaned: String = raw
        .chars()
        .filter(|c| c.is_ascii() && !c.is_ascii_control() && *c != '!')
        .take(96)
        .collect();
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        fallback.to_string()
    } else {
        trimmed.to_string()
    }
}
