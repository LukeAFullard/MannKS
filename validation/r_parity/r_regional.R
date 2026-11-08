# Per-site MannKendall + getTAU.  Usage: Rscript r_regional.R <lwp> <obs.csv> <out_prefix>
args <- commandArgs(trailingOnly = TRUE)
suppressMessages(suppressWarnings(source(args[1])))
obs <- read.csv(args[2], stringsAsFactors = FALSE)
obs$myDate <- as.Date(obs$myDate); obs$analyte <- "TP"; obs$Value <- as.character(obs$Value)
obs <- RemoveAlphaDetect(obs, "Value")
obs <- GetMoreDateInfo(obs, FindDateShifts = FALSE)
obs$TimeIncr <- obs$Month; obs$Season <- obs$Month
x <- do.call(rbind, lapply(split(obs, obs$siteID), function(s) {
  k <- suppressWarnings(MannKendall(s))
  data.frame(siteID = s$siteID[1], analyte = "TP", S = k$S, C = k$C, TimeIncr = "Monthly")
}))
res <- suppressWarnings(capture.output(tau <- getTAU(x, obs)))
write.csv(x, paste0(args[3], "_sites.csv"), row.names = FALSE)
write.csv(tau$SummaryResults, paste0(args[3], "_tau.csv"), row.names = FALSE)
