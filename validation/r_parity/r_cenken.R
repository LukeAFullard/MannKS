# NADA::cenken on prepared vectors.  Usage: Rscript r_cenken.R <unused> <in.csv> <out.csv>
args <- commandArgs(trailingOnly = TRUE)
suppressMessages(library(NADA))
d <- read.csv(args[2])
out <- do.call(rbind, lapply(split(d, d$case_id), function(z) {
  k <- cenken(z$y, as.logical(z$cen), z$x)
  data.frame(case_id = z$case_id[1], slope = k$slope, intercept = k$intercept, tau = k$tau, p = k$p)
}))
write.csv(out, args[3], row.names = FALSE)
