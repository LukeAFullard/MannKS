# SeasonalityTest on parity cases. Usage: Rscript r_seasonality.R <lwp> <data.csv> <out.csv> <case ids comma-sep>
args <- commandArgs(trailingOnly = TRUE)
suppressMessages(suppressWarnings(source(args[1])))
d <- read.csv(args[2], colClasses = c("character", "character", "character"))
ids <- strsplit(args[4], ",")[[1]]
out <- do.call(rbind, lapply(ids, function(id) {
  x <- d[d$case_id == id, ]; x$myDate <- as.Date(x$myDate); x$Value[x$Value == "NA"] <- NA
  x <- suppressWarnings(RemoveAlphaDetect(x, "Value")); x <- GetMoreDateInfo(x, FindDateShifts = FALSE)
  x$TimeIncr <- x$Month; x$Season <- x$Month
  s <- suppressWarnings(SeasonalityTest(x))
  data.frame(case_id = id, KW = s$KWstat, p = s$pvalue, note = s$SeasNote)
}))
write.csv(out, args[3], row.names = FALSE)
