# ==============================================================================
# Genuine USGS EGRET Quantile-Kendall reference for MannKS validation.
#
# The functions between the BEGIN/END markers are copied VERBATIM from the
# EGRET package vignette "Daily Streamflow Trend Analysis"
#   https://github.com/DOI-USGS/EGRET/blob/main/vignettes/streamflow_trend.Rmd
#   https://doi-usgs.github.io/EGRET/articles/streamflow_trend.html
# (Hirsch & De Cicco, USGS; EGRET is released to the public domain, CC0).
# They are the code behind EGRET's plotQuantileKendall(). Nothing in them has
# been altered, so the numbers produced here are EGRET's, computed with the
# EGRET (eList, setPA, DecYear, MonthSeq), rkt and zyp R packages.
#
# run_egret_qk() below is only a thin driver: it builds an EGRET eList from a
# plain (date, discharge) table and calls makeSortQ() + trendSortQ().
# ==============================================================================
suppressPackageStartupMessages({
  library(EGRET)
  library(rkt)
  library(zyp)
})

# ----------------------------- BEGIN EGRET VERBATIM ---------------------------
makeSortQ <- function(eList){
  localINFO <- getInfo(eList)
  localDaily <- getDaily(eList)
  paStart <- localINFO$paStart
  paLong <- localINFO$paLong
  # determine the maximum number of days to put in the array
  numDays <- length(localDaily$DecYear)
  monthSeqFirst <- localDaily$MonthSeq[1]
  monthSeqLast <- localDaily$MonthSeq[numDays]
  # creating a data frame (called startEndSeq) of the MonthSeq values that go into each year
  Starts <- seq(paStart, monthSeqLast, 12)
  Ends <- Starts + paLong - 1
  startEndSeq <- data.frame(Starts, Ends)
  # trim this list of Starts and Ends to fit the period of record
  startEndSeq <- subset(startEndSeq, Ends >= monthSeqFirst & Starts <= monthSeqLast)
  numYearsRaw <- length(startEndSeq$Ends)
  # set up some vectors to keep track of years
  good <- rep(0, numYearsRaw)
  numDays <- rep(0, numYearsRaw)
  midDecYear <- rep(0, numYearsRaw)
  Qraw <- matrix(nrow = 366, ncol = numYearsRaw)
  for(i in 1: numYearsRaw) {
    startSeq <- startEndSeq$Starts[i]
    endSeq <- startEndSeq$Ends[i]
    startJulian <- getFirstJulian(startSeq)
    # startJulian is the first julian day of the first month in the year being processed
    # endJulian is the first julian day of the month right after the last month in the year being processed
    endJulian <- getFirstJulian(endSeq + 1)
    fullDuration <- endJulian - startJulian
    yearDaily <- localDaily[localDaily$MonthSeq >= startSeq & (localDaily$MonthSeq <= endSeq), ]
    nDays <- length(yearDaily$Q)
    if(nDays == fullDuration) {
      good[i] <- 1
      numDays[i] <- nDays
      midDecYear[i] <- (yearDaily$DecYear[1] + yearDaily$DecYear[nDays]) / 2
      Qraw[1:nDays,i] <- yearDaily$Q
    }	else {
      numDays[i] <- NA
      midDecYear[i] <- NA
    }
  }
  # now we compress the matrix down to equal number of values in each column
  j <- 0
  numGoodYears <- sum(good)
  dayCounts <- ifelse(good==1, numDays, NA)
  lowDays <- min(dayCounts, na.rm = TRUE)
  highDays <- max(dayCounts, na.rm = TRUE)
  dimYears <- numGoodYears
  dimDays <- lowDays
  sortQ <- matrix(nrow = dimDays, ncol = dimYears)
  time <- rep(0,dimYears)
  for (i in 1:numYearsRaw){
    if(good[i]==1) {
      j <- j + 1
      numD <- numDays[i]
      x <- sort(Qraw[1:numD, i])
      # separate odd numbers from even numbers of days
      if(numD == lowDays) {
        sortQ[1:dimDays,j] <- x
      } else {
        sortQ[1:dimDays,j] <- if(odd(numD)) leapOdd(x) else leapEven(x)
      }
      time[j] <- midDecYear[i]
    } 
  }
  
  sortQList = list(sortQ,time)
  
  return(sortQList)			
}

trendSortQ <- function(Qsort, time){
  # note requires packages zyp and rkt
  nFreq <- dim(Qsort)[1]
  nYears <- length(time)
  results <- as.data.frame(matrix(ncol=9,nrow=nFreq))
  colnames(results) <- c("slopeLog","slopePct","pValue","pValueAdj","tau","rho1","rho2","freq","z")
  for(iRank in 1:nFreq){
    mkOut <- rkt::rkt(time,log(Qsort[iRank,]))
    results$slopeLog[iRank] <- mkOut$B
    results$slopePct[iRank] <- 100 * (exp(mkOut$B) - 1)
    results$pValue[iRank] <- mkOut$sl
    outZYP <- zyp.zhang(log(Qsort[iRank,]),time)
    results$pValueAdj[iRank] <- outZYP[6]
    results$tau[iRank] <- mkOut$tau
    # I don't actually use this information in the current outputs, but the code is there 
    # if one wanted to look at the serial correlation structure of the flow series 		
    serial <- acf(log(Qsort[iRank,]), lag.max = 2, plot = FALSE)
    results$rho1[iRank] <- serial$acf[2]
    results$rho2[iRank] <- serial$acf[3]
    frequency <- iRank / (nFreq + 1)
    results$freq[iRank] <- frequency
    results$z[iRank] <- qnorm(frequency)	
  }
  return(results)
}

getFirstJulian <- function(monthSeq){
  year <- 1850 + trunc((monthSeq - 1) / 12)
  month <- monthSeq - 12 * (trunc((monthSeq-1)/12))
  charMonth <- ifelse(month<10, paste0("0",as.character(month)), as.character(month))
  theDate <- paste0(year,"-",charMonth,"-01")
  Julian1 <- as.numeric(julian(as.Date(theDate),origin=as.Date("1850-01-01")))
  return(Julian1)
}

leapOdd <- function(x){
  n <- length(x)
  m <- n - 1
  mid <- (n + 1) / 2
  mid1 <- mid + 1
  midMinus <- mid - 1
  y <- rep(NA, m)
  y[1:midMinus] <- x[1:midMinus]
  y[mid:m] <- x[mid1:n]
  return(y)}

leapEven <- function(x){
  n <- length(x)
  m <- n - 1
  mid <- n / 2
  y <- rep(NA, m)
  mid1 <- mid + 1
  mid2 <- mid + 2
  midMinus <- mid - 1
  y[1:midMinus] <- x[1:midMinus]
  y[mid] <- (x[mid] + x[mid1]) / 2
  y[mid1:m] <- x[mid2 : n]
  return(y)
}

odd <- function(x) {(!(x %% 2) == 0)}
# ------------------------------ END EGRET VERBATIM ----------------------------

# Thin driver (not EGRET code): build an eList and run the two EGRET functions.
#   dates : character vector "YYYY-MM-DD" (daily)
#   q     : discharge, already in m3/s (qConvert = 1, no unit conversion)
#   paStart, paLong : EGRET period of analysis (start month, length in months)
# Returns list(sortQ = dimDays x nYears matrix, time = midDecYear,
#              results = trendSortQ data frame, daily = Date/DecYear/MonthSeq,
#              versions = package versions used).
run_egret_qk <- function(dates, q, paStart = 4, paLong = 12) {
  raw <- data.frame(dateTime = as.character(dates), value = as.numeric(q),
                    code = "", stringsAsFactors = FALSE)
  Daily <- populateDaily(raw, qConvert = 1, verbose = FALSE)
  INFO <- data.frame(shortName = "MannKS parity series", staAbbrev = "mannks",
                     paStart = paStart, paLong = paLong, stringsAsFactors = FALSE)
  eList <- as.egret(INFO, Daily, NA, NA)
  eList <- setPA(eList, paStart = paStart, paLong = paLong)
  v <- makeSortQ(eList)
  res <- trendSortQ(v[[1]], v[[2]])
  res$rank <- seq_len(nrow(res))
  # rkt's S and Var(S) for completeness (not kept by trendSortQ)
  res$S <- NA_real_; res$varS <- NA_real_
  for (i in seq_len(nrow(res))) {
    mk <- rkt::rkt(v[[2]], log(v[[1]][i, ]))
    res$S[i] <- mk$S; res$varS[i] <- mk$varS
  }
  list(sortQ = v[[1]], time = v[[2]], results = res,
       daily = Daily[, c("Date", "DecYear", "MonthSeq")],
       versions = c(R = as.character(getRversion()),
                    EGRET = as.character(packageVersion("EGRET")),
                    rkt = as.character(packageVersion("rkt")),
                    zyp = as.character(packageVersion("zyp"))))
}
