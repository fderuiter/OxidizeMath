#!/usr/bin/env Rscript
# Offline R Cross-Validation Script for OxidizeMath Golden Fixtures (Tier 3)
# Uses base R `besselJ`, `besselY`, `gamma`, `beta`, and Sheppard's identity
# with zero external CRAN packages required.

verify_r_bessel <- function() {
  xs <- c(0.25, 1.5, 5.0, 15.0, 25.0, 40.0, 50.0, 100.0)
  for (n in c(0, 1, 2, 5)) {
    for (x in xs) {
      val <- besselJ(x, n)
      cat(sprintf("bessel_j(nu=%d, x=%.2f) = %.17g\n", n, x, val))
    }
  }
  for (n in c(0, 1, 2)) {
    for (x in c(0.5, 2.0, 10.0, 25.0, 40.0, 50.0, 100.0)) {
      val <- besselY(x, n)
      cat(sprintf("bessel_y(nu=%d, x=%.2f) = %.17g\n", n, x, val))
    }
  }
}

verify_r_bessel()
