# Reference fits from R's `segmented` package (Muggeo 2003, 2008, 2020).
#
# Usage:
#   Rscript r_segmented.R export <out_datasets.csv>
#   Rscript r_segmented.R run <in_cases.csv> <out_results.csv>
#
# Requires only base R + CRAN package `segmented`.

args <- commandArgs(trailingOnly = TRUE)
suppressMessages(library(segmented))

cmd <- args[1]

if (cmd == "export") {
  out_path <- args[2]
  data(stagnant)
  data(globTempAnom)
  data(plant)

  rows <- list(
    data.frame(dataset = "stagnant", t = stagnant$x, y = stagnant$y, stringsAsFactors = FALSE),
    data.frame(dataset = "globTempAnom", t = as.numeric(globTempAnom$Year), y = globTempAnom$Anomaly, stringsAsFactors = FALSE)
  )
  for (g in levels(plant$group)) {
    sub_df <- plant[plant$group == g, ]
    rows[[length(rows) + 1]] <- data.frame(
      dataset = paste0("plant_", g),
      t = as.numeric(sub_df$time),
      y = as.numeric(sub_df$y),
      stringsAsFactors = FALSE
    )
  }
  all_df <- do.call(rbind, rows)
  write.csv(all_df, out_path, row.names = FALSE)
  quit(status = 0)
}

if (cmd != "run") {
  stop(paste("Unknown command:", cmd))
}

in_path <- args[2]
out_path <- args[3]
d <- read.csv(in_path, stringsAsFactors = FALSE)

fmt_vec <- function(v) {
  if (is.null(v) || length(v) == 0 || all(is.na(v))) return("")
  paste(sprintf("%.10g", as.numeric(v)), collapse = ";")
}

run_case <- function(z) {
  cid <- z$case_id[1]
  mode <- z$mode[1]
  k_fixed <- as.integer(z$k_fixed[1])
  k_max <- as.integer(z$k_max[1])
  ord <- order(z$t)
  t_vec <- as.numeric(z$t[ord])
  y_vec <- as.numeric(z$y[ord])
  df_case <- data.frame(t = t_vec, y = y_vec)

  lm0 <- lm(y ~ t, data = df_case)
  ctrl <- seg.control(n.boot = 20, seed = 42, display = FALSE)

  converged <- TRUE
  n_bp <- 0L
  bps <- numeric(0)
  bp_se <- numeric(0)
  bp_lo <- numeric(0)
  bp_hi <- numeric(0)
  slopes <- as.numeric(coef(lm0)[2])
  bic_val <- as.numeric(BIC(lm0))
  rss_val <- sum(residuals(lm0)^2)
  score_n_bp <- NA_integer_
  davies_p <- NA_real_

  if (mode == "fixed") {
    if (k_fixed > 0L) {
      fit <- tryCatch(
        suppressWarnings(segmented(lm0, seg.Z = ~t, npsi = k_fixed, control = ctrl)),
        error = function(e) NULL
      )
      if (!is.null(fit) && inherits(fit, "segmented") && !is.null(fit$psi) && is.matrix(fit$psi)) {
        n_bp <- nrow(fit$psi)
        ord_psi <- order(fit$psi[, "Est."])
        bps <- fit$psi[ord_psi, "Est."]
        bp_se <- fit$psi[ord_psi, "St.Err"]
        ci_mat <- tryCatch(confint(fit), error = function(e) NULL)
        if (!is.null(ci_mat) && is.matrix(ci_mat)) {
          bp_lo <- ci_mat[ord_psi, 2]
          bp_hi <- ci_mat[ord_psi, 3]
        }
        sl_mat <- tryCatch(slope(fit)$t, error = function(e) NULL)
        if (!is.null(sl_mat) && is.matrix(sl_mat)) {
          slopes <- sl_mat[, "Est."]
        }
        bic_val <- as.numeric(BIC(fit))
        rss_val <- sum(residuals(fit)^2)
      } else {
        converged <- FALSE
        n_bp <- NA_integer_
        slopes <- numeric(0)
        bic_val <- NA_real_
        rss_val <- NA_real_
      }
    }
  } else if (mode == "select") {
    # Davies test for any breakpoint
    dt <- tryCatch(suppressWarnings(davies.test(lm0, seg.Z = ~t, k = 20)), error = function(e) NULL)
    if (!is.null(dt)) davies_p <- as.numeric(dt$p.value)

    # BIC-based model selection via selgmented
    sel <- tryCatch(
      suppressWarnings(selgmented(lm0, seg.Z = ~t, Kmax = k_max, type = "bic", msg = FALSE, control = ctrl)),
      error = function(e) NULL
    )
    if (!is.null(sel) && !is.null(sel$selection.psi$npsi)) {
      n_bp <- as.integer(sel$selection.psi$npsi)
      if (n_bp > 0L && inherits(sel, "segmented") && is.matrix(sel$psi)) {
        ord_psi <- order(sel$psi[, "Est."])
        bps <- sel$psi[ord_psi, "Est."]
        bp_se <- sel$psi[ord_psi, "St.Err"]
        ci_mat <- tryCatch(confint(sel), error = function(e) NULL)
        if (!is.null(ci_mat) && is.matrix(ci_mat)) {
          bp_lo <- ci_mat[ord_psi, 2]
          bp_hi <- ci_mat[ord_psi, 3]
        }
        sl_mat <- tryCatch(slope(sel)$t, error = function(e) NULL)
        if (!is.null(sl_mat) && is.matrix(sl_mat)) {
          slopes <- sl_mat[, "Est."]
        }
        bic_val <- as.numeric(BIC(sel))
        rss_val <- sum(residuals(sel)^2)
      }
    } else {
      converged <- FALSE
      n_bp <- NA_integer_
    }

    # Score-based selection (supported for Kmax <= 2 in segmented 2.0-3)
    if (k_max <= 2L) {
      sel_sc <- tryCatch(
        suppressWarnings(selgmented(lm0, seg.Z = ~t, Kmax = k_max, type = "score", msg = FALSE, control = ctrl)),
        error = function(e) NULL
      )
      if (!is.null(sel_sc) && !is.null(sel_sc$selection.psi$npsi)) {
        score_n_bp <- as.integer(sel_sc$selection.psi$npsi)
      }
    }
  } else {
    stop(paste("Unknown mode:", mode))
  }

  data.frame(
    case_id = cid,
    mode = mode,
    converged = converged,
    n_bp = n_bp,
    bps = fmt_vec(bps),
    bp_se = fmt_vec(bp_se),
    bp_ci_lo = fmt_vec(bp_lo),
    bp_ci_hi = fmt_vec(bp_hi),
    slopes = fmt_vec(slopes),
    bic = bic_val,
    rss = rss_val,
    score_n_bp = score_n_bp,
    davies_p = davies_p,
    r_version = R.version.string,
    segmented_version = as.character(packageVersion("segmented")),
    stringsAsFactors = FALSE
  )
}

case_ids <- unique(d$case_id)
res_list <- lapply(case_ids, function(cid) run_case(d[d$case_id == cid, ]))
out_df <- do.call(rbind, res_list)
write.csv(out_df, out_path, row.names = FALSE)
