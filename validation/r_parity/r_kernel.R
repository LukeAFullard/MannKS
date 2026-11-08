# Runs LWP-TRENDS v2502 GetKendal / GetInterObservationSlopes on pre-built vectors.
# Usage: Rscript r_kernel.R <lwp_script> <input.csv> <output.csv>
# input columns: case_id, V1, Censored, CenType, Season, Year, NewDate (YYYY-MM-DD)
args <- commandArgs(trailingOnly = TRUE)
suppressMessages(suppressWarnings(source(args[1])))
d <- read.csv(args[2], stringsAsFactors = FALSE)
d$NewDate <- as.Date(d$NewDate)
d$Censored <- as.logical(d$Censored)
out <- do.call(rbind, lapply(split(d, d$case_id), function(x) {
  x <- x[order(x$NewDate), ]
  k <- suppressWarnings(GetKendal(x))
  sl <- suppressWarnings(GetInterObservationSlopes(x))
  data.frame(case_id = x$case_id[1], n = k$nTimeIncr, S = k$S, VarS = k$vars, D = k$D,
             Z = ifelse(is.null(k$Z), NA, k$Z), p = ifelse(is.null(k$p), NA, k$p),
             sen = median(sl$Slopes), nslopes = nrow(sl))
}))
write.csv(out, args[3], row.names = FALSE)
