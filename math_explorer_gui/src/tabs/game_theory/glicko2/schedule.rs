//! Pairing generation and period simulation helpers for [`super::Glicko2TournamentTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{match_preset, Glicko2TournamentTool, ScheduledPairingRow};
use crate::presets::{Glicko2PairingMode, Glicko2PeriodPreset, MAX_TOURNAMENT_PERIODS};
use math_explorer::pure_math::statistics::glicko2::{
    round_robin_schedule, simulate_period_matches, swiss_pairings, GlickoPlayer, Pairing,
    PeriodMatch,
};
use oxidize_core::rng::OxidizeRng;
use std::collections::HashSet;

impl Glicko2TournamentTool {
    pub(crate) fn refresh_next_pairings(&mut self) {
        let active: Vec<usize> = self
            .competitors
            .iter()
            .enumerate()
            .filter_map(|(i, c)| (!c.inactive).then_some(i))
            .collect();
        if active.len() < 2 {
            self.next_pairings.clear();
            self.next_bye_player = active.first().copied();
            return;
        }
        match self.pairing_mode {
            Glicko2PairingMode::Swiss => self.refresh_swiss_pairings(&active),
            Glicko2PairingMode::RoundRobin => self.refresh_round_robin_pairings(&active),
        }
    }

    fn refresh_swiss_pairings(&mut self, active: &[usize]) {
        let mut inv = vec![None; self.competitors.len()];
        for (li, &gi) in active.iter().enumerate() {
            inv[gi] = Some(li);
        }
        let players: Vec<GlickoPlayer> = active
            .iter()
            .map(|&i| self.competitors[i].current)
            .collect();
        let points: Vec<f64> = active.iter().map(|&i| self.competitors[i].points).collect();
        let mut prior = Vec::new();
        for m in self.completed_periods.iter().flat_map(|p| &p.matches) {
            if let (Some(la), Some(lb)) = (inv[m.player_a], inv[m.player_b]) {
                if let Ok(pm) = PeriodMatch::new(la, lb, m.score_a) {
                    prior.push(pm);
                }
            }
        }
        if let Ok(plan) = swiss_pairings(&players, &points, &prior) {
            self.next_pairings = plan
                .pairings
                .into_iter()
                .map(|p| ScheduledPairingRow {
                    pairing: Pairing {
                        player_a: active[p.player_a],
                        player_b: active[p.player_b],
                    },
                    manual_score_a: 0.5,
                })
                .collect();
            self.next_bye_player = plan.bye_player.map(|b| active[b]);
        }
    }

    fn refresh_round_robin_pairings(&mut self, active: &[usize]) {
        if let Ok(schedule) = round_robin_schedule(active.len()) {
            let round = &schedule[self.completed_periods.len() % schedule.len()];
            let mut matched = HashSet::with_capacity(round.len() * 2);
            self.next_pairings = round
                .iter()
                .map(|p| {
                    matched.insert(p.player_a);
                    matched.insert(p.player_b);
                    ScheduledPairingRow {
                        pairing: Pairing {
                            player_a: active[p.player_a],
                            player_b: active[p.player_b],
                        },
                        manual_score_a: 0.5,
                    }
                })
                .collect();
            self.next_bye_player = (0..active.len())
                .find(|li| !matched.contains(li))
                .map(|li| active[li]);
        }
    }

    pub fn simulate_next_period(&mut self) -> Result<(), String> {
        if self.completed_periods.len() >= MAX_TOURNAMENT_PERIODS {
            return Err(format!(
                "Maximum period cap ({MAX_TOURNAMENT_PERIODS}) reached"
            ));
        }
        let pairings: Vec<Pairing> = self.next_pairings.iter().map(|r| r.pairing).collect();
        let roster: Vec<GlickoPlayer> = self.competitors.iter().map(|c| c.current).collect();
        let seed = self.rng_seed.wrapping_add(
            (self.completed_periods.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        let mut rng = OxidizeRng::new(seed);
        let matches = simulate_period_matches(&roster, &pairings, self.draw_probability, &mut rng)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|m| match_preset(m.player_a, m.player_b, m.score_a))
            .collect();
        self.completed_periods.push(Glicko2PeriodPreset {
            matches,
            bye_player: self.next_bye_player,
        });
        self.rebuild_trajectory()
    }

    pub fn commit_manual_period(&mut self) -> Result<(), String> {
        if self.completed_periods.len() >= MAX_TOURNAMENT_PERIODS {
            return Err(format!(
                "Maximum period cap ({MAX_TOURNAMENT_PERIODS}) reached"
            ));
        }
        let matches = self
            .next_pairings
            .iter()
            .map(|r| {
                match_preset(
                    r.pairing.player_a,
                    r.pairing.player_b,
                    r.manual_score_a.clamp(0.0, 1.0),
                )
            })
            .collect();
        self.completed_periods.push(Glicko2PeriodPreset {
            matches,
            bye_player: self.next_bye_player,
        });
        self.rebuild_trajectory()
    }

    pub fn simulate_to_target_rounds(&mut self) -> Result<(), String> {
        while self.completed_periods.len() < self.target_rounds {
            self.simulate_next_period()?;
        }
        Ok(())
    }

    pub fn reset_periods(&mut self) {
        self.completed_periods.clear();
        let _ = self.rebuild_trajectory();
    }
}
