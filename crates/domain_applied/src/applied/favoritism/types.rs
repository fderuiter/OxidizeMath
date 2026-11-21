use nalgebra::DVector;

/// Time and proximity related parameters.
///
/// These parameters define the temporal scope and spatial decay of the relationship.
#[derive(Debug, Clone)]
pub struct TimeParams {
    /// Time period over which proximity is integrated (in days).
    ///
    /// Default is 365.0 (1 year).
    pub t: f64,
    /// Initial distance/proximity factor (e.g., kilometers).
    ///
    /// A lower value implies closer physical proximity.
    pub x_0: f64,
}

impl Default for TimeParams {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            t: 365.0,
            x_0: 20.0,
        }
    }
}

/// Parameters related to gifts given.
///
/// Defines the monetary and sentimental value of gifts, which are
/// key drivers in the corruption of parental affection.
#[derive(Debug, Clone)]
pub struct GiftParams {
    /// Emotional value of gifts (Arbitrary 0-10 scale).
    ///
    /// Represents how "thoughtful" the gifts are.
    pub g_emotional: f64,
    /// Practical value of gifts (Arbitrary 0-10 scale).
    ///
    /// Represents the utility or monetary worth.
    pub g_practical: f64,
}

impl Default for GiftParams {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            g_emotional: 5.0,
            g_practical: 2.0,
        }
    }
}

/// Parameters related to contact frequency and decay.
///
/// Models how quickly parental love fades when you don't call your mother.
#[derive(Debug, Clone)]
pub struct ContactParams {
    /// Initial contact frequency factor (calls/visits per month).
    pub f_initial: f64,
    /// Decay constant for lack of contact.
    ///
    /// Determines how fast the 'memory' of the last contact fades.
    pub decay_constant: f64,
    /// Time since the last contact occurred (in days).
    pub time_since_last_contact: f64,
}

impl Default for ContactParams {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            f_initial: 7.0,
            decay_constant: 0.05,
            time_since_last_contact: 7.0,
        }
    }
}

/// Parameters related to personality and success traits.
///
/// Parents often favor the child who reflects best on them socially.
#[derive(Debug, Clone)]
pub struct PersonalityParams {
    /// Intelligence score (0-10 scale).
    pub intelligence: f64,
    /// Emotional sensitivity score (0-10 scale).
    pub emotional_sensitivity: f64,
    /// Wealth score (0-10 scale, or log-wealth).
    pub wealth: f64,
    /// Talent score (0-10 scale).
    pub talent: f64,
    /// Weight for intelligence importance.
    pub w_i: f64,
    /// Weight for emotional sensitivity importance.
    pub w_es: f64,
    /// Weight for wealth importance.
    pub w_w: f64,
    /// Weight for talent importance.
    pub w_t: f64,
}

impl Default for PersonalityParams {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            intelligence: 7.0,
            emotional_sensitivity: 6.0,
            wealth: 9.0,
            talent: 8.0,
            w_i: 1.2,
            w_es: 1.5,
            w_w: 1.1,
            w_t: 1.3,
        }
    }
}

/// Parameters related to social behavior and life events.
#[derive(Debug, Clone)]
pub struct SocialParams {
    /// Weight assigned based on birth order (e.g., oldest=1.2, middle=0.9).
    pub birth_order_weight: f64,
    /// Score for major life events shared (weddings, grandkids).
    pub major_life_events: f64,
    /// Whether the child helped during a crisis (Boolean multiplier).
    pub helped_during_crisis: bool,
    /// Whether the child is active on social media (Visibility multiplier).
    pub active_on_social_media: bool,
}

impl Default for SocialParams {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            birth_order_weight: 1.2,
            major_life_events: 3.0,
            helped_during_crisis: true,
            active_on_social_media: true,
        }
    }
}

/// Parameters related to compliments given.
///
/// Flattery is effective.
#[derive(Debug, Clone)]
pub struct ComplimentParams {
    /// Vector of compliment intensity scores.
    pub compliments: DVector<f64>,
    /// Weights corresponding to each compliment type (e.g., appearance vs cooking).
    pub compliment_weights: DVector<f64>,
}

impl Default for ComplimentParams {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            compliments: DVector::from_vec(vec![10.0, 5.0, 8.0]),
            compliment_weights: DVector::from_vec(vec![1.0, 0.5, 0.75]),
        }
    }
}

/// Parameters related to family context.
#[derive(Debug, Clone)]
pub struct FamilyParams {
    /// Distances of siblings (used for the denominator in the formula).
    ///
    /// Used to normalize your proximity against your competition.
    pub sibling_distances: Vec<f64>,
}

impl Default for FamilyParams {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            sibling_distances: vec![100.0, 50.0, 10.0],
        }
    }
}

/// Input parameters for the favoritism calculation.
///
/// This struct aggregates all factors required to compute the
/// satirical favoritism score.
///
/// # Example
///
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::favoritism::FavoritismInputs;
///
/// let mut inputs = FavoritismInputs::default();
/// inputs.personality.wealth = 10.0; // The "Golden Child" strategy
/// ```
#[derive(Debug, Clone, Default)]
pub struct FavoritismInputs {
    /// Time and proximity settings.
    pub time: TimeParams,
    /// Gift giving parameters.
    pub gifts: GiftParams,
    /// Contact and decay parameters.
    pub contact: ContactParams,
    /// Personality and success traits.
    pub personality: PersonalityParams,
    /// Social behavior and life events.
    pub social: SocialParams,
    /// Compliments and praise.
    pub compliments: ComplimentParams,
    /// Family context (siblings).
    pub family: FamilyParams,
}
