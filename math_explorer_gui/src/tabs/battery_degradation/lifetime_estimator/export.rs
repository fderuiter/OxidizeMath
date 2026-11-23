//! Yearly SOH warranty `.csv` schedule formatter (`80%` and `70%` milestones)
//! for [`super::LifetimeEstimatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:battery_degradation]

use super::LifetimeEstimatorTool;
use crate::tabs::battery_degradation::cell_card::sanitize_csv_text_cell;
use math_explorer::applied::battery_degradation::{
    DAYS_PER_YEAR, DutyPointBreakdown, evaluate_duty_point,
};

const MAX_EXPORT_YEARS: usize = 50;

struct ScheduleRow {
    days: f64,
    milestone: String,
}

fn push_csv_row(
    out: &mut String,
    chemistry: &str,
    days: f64,
    pt: &DutyPointBreakdown,
    milestone: &str,
) {
    let year = days / DAYS_PER_YEAR;
    let safe_chem = sanitize_csv_text_cell(chemistry);
    let safe_milestone = sanitize_csv_text_cell(milestone);
    out.push_str(&format!(
        "{safe_chem},{year:.3},{days:.1},{:.1},{:.3},{:.3},{:.3},{safe_milestone}\n",
        pt.cumulative_cycles,
        pt.soh * 100.0,
        pt.cycling_only_soh * 100.0,
        pt.calendar_sei_loss * 100.0,
    ));
}

impl LifetimeEstimatorTool {
    /// Generates a formula-safe yearly `.csv` SOH schedule including exact `80%`, `70%`,
    /// and target warranty milestone rows.
    #[must_use]
    pub fn build_warranty_schedule_csv(&self) -> String {
        let model = self.active_model();
        let segs = self.effective_duty_segments();
        let summary = self.lifetime_summary();
        let max_years = (summary.years_to_70.ceil() as usize + 1).clamp(2, MAX_EXPORT_YEARS);

        let mut rows = Vec::with_capacity(max_years + 4);
        for yr in 0..=max_years {
            rows.push(ScheduleRow {
                days: (yr as f64) * DAYS_PER_YEAR,
                milestone: format!("Year {yr}"),
            });
        }
        rows.push(ScheduleRow {
            days: summary.days_to_80,
            milestone: "80% First-Life Warranty".to_string(),
        });
        rows.push(ScheduleRow {
            days: summary.days_to_70,
            milestone: "70% End-of-Life Warranty".to_string(),
        });
        if (self.target_capacity - 80.0).abs() > 1e-6 && (self.target_capacity - 70.0).abs() > 1e-6
        {
            rows.push(ScheduleRow {
                days: summary.days_to_target,
                milestone: format!("{:.1}% Target EOL", self.target_capacity),
            });
        }

        rows.sort_by(|a, b| a.days.total_cmp(&b.days));

        let mut csv = String::from(
            "chemistry,year,days,cumulative_cycles,coupled_soh_pct,cycling_only_soh_pct,calendar_sei_loss_pct,milestone\n",
        );
        let chem = self.chemistry.label();
        for row in rows {
            let pt = evaluate_duty_point(&model, &segs, row.days);
            push_csv_row(&mut csv, chem, row.days, &pt, &row.milestone);
        }
        csv
    }
}
