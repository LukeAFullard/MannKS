# Wrapper-level LWP-TRENDS v2502 runs.
# Usage: Rscript r_wrapper.R <lwp_script> <data.csv> <out.csv> <cfg.csv>
#  data.csv: case_id, myDate (YYYY-MM-DD), Value (string, may contain < or >, "NA")
#  cfg.csv : case_id, seasonal, timeincr, season, use_mid, shift
args <- commandArgs(trailingOnly = TRUE)
suppressMessages(suppressWarnings(source(args[1])))
d <- read.csv(args[2], colClasses = c("character", "character", "character"))
cfg <- read.csv(args[4], stringsAsFactors = FALSE)
g <- function(df, nm) if (!is.null(df[[nm]])) as.character(df[[nm]][1]) else NA
run_one <- function(cc) {
  x <- d[d$case_id == cc$case_id, ]
  x$myDate <- as.Date(x$myDate)
  x$Value[x$Value == "NA"] <- NA
  x <- suppressWarnings(RemoveAlphaDetect(x, ColToUse = "Value"))
  x <- GetMoreDateInfo(x, FindDateShifts = as.logical(cc$shift))
  x$TimeIncr <- x[[cc$timeincr]]
  x$Season <- x[[cc$season]]
  um <- as.logical(cc$use_mid)
  o <- list(case_id = cc$case_id)
  if (!as.logical(cc$seasonal)) {
    A1 <- tryCatch(suppressWarnings(MannKendall(x, UseMidObs = um)), error = function(e) NULL)
    A2 <- tryCatch(suppressWarnings(SenSlope(x, UseMidObs = um)), error = function(e) NULL)
  } else {
    A1 <- tryCatch(suppressWarnings(SeasonalKendall(x, UseMidObs = um)), error = function(e) NULL)
    A2 <- tryCatch(suppressWarnings(SeasonalSenSlope(x, UseMidObs = um)), error = function(e) NULL)
  }
  for (nm in c("nObs", "nTimeIncr", "S", "VarS", "D", "tau", "Z", "p", "C", "Cd", "AnalysisNote", "prop.censored"))
    o[[paste0("mk_", nm)]] <- if (is.null(A1)) "ERROR" else g(A1, nm)
  for (nm in c("Median", "AnnualSenSlope", "Sen_Lci", "Sen_Uci", "AnalysisNote", "Sen_Probability", "Percent.annual.change"))
    o[[paste0("sen_", nm)]] <- if (is.null(A2)) "ERROR" else g(A2, nm)
  if (!is.null(A1) && !is.na(suppressWarnings(as.numeric(g(A1, "C"))))) {
    o$dir <- as.character(GetTrendDirectionandClass(A1)$TrendDirection)
    o$confcat_direction <- as.character(AssignConfCat(as.numeric(g(A1, "C")), CatType = "Direction"))
    o$confcat_decrease <- as.character(AssignConfCat(data.frame(Cd = as.numeric(g(A1, "Cd"))), CatType = "Decrease"))
  } else { o$dir <- NA; o$confcat_direction <- NA; o$confcat_decrease <- NA }
  as.data.frame(o, stringsAsFactors = FALSE)
}
out <- do.call(rbind, lapply(seq_len(nrow(cfg)), function(i) run_one(cfg[i, ])))
write.csv(out, args[3], row.names = FALSE)
