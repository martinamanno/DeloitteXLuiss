library(gtrendsR)
library(tidyverse)
library(lubridate)
install.packages("CausalImpact")
library(CausalImpact)

#Estimating Google trends of sephora
get_daily_gtrend <- function(keyword = c('sephora'), geo = 'IT', from = '2018-10-15', to = '2020-02-13') {
  if (ymd(to) >= floor_date(Sys.Date(), 'month')) {
    to <- floor_date(ymd(to), 'month') - days(1)
    
    if (to < from) {
      stop("Specifying \'to\' date in the current month is not allowed")
    }
  }
  
  aggregated_data <- gtrends(keyword = keyword, geo = geo, time = paste(from, to))
  if(is.null(aggregated_data$interest_over_time)) {
    print('There is no data in Google Trends!')
    return()
  }
  
  mult_m <- aggregated_data$interest_over_time %>%
    mutate(hits = as.integer(ifelse(hits == '<1', '0', hits))) %>%
    group_by(month = floor_date(date, 'month'), keyword) %>%
    summarise(hits = sum(hits)) %>%
    ungroup() %>%
    mutate(ym = format(month, '%Y-%m'),
           mult = hits / max(hits)) %>%
    select(month, ym, keyword, mult) %>%
    as_tibble()
  
  pm <- tibble(s = seq(ymd(from), ymd(to), by = 'month'), 
               e = seq(ymd(from), ymd(to), by = 'month') + months(1) - days(1))
  
  raw_trends_m <- tibble()
  
  for (i in seq(1, nrow(pm), 1)) {
    curr <- gtrends(keyword, geo = geo, time = paste(pm$s[i], pm$e[i]))
    if(is.null(curr$interest_over_time)) next
    print(paste('for', pm$s[i], pm$e[i], 'retrieved', count(curr$interest_over_time), 'days of data (all keywords)'))
    raw_trends_m <- rbind(raw_trends_m,
                          curr$interest_over_time)
  }
  
  trend_m <- raw_trends_m %>%
    select(date, keyword, hits) %>%
    mutate(ym = format(date, '%Y-%m'),
           hits = as.integer(ifelse(hits == '<1', '0', hits))) %>%
    as_tibble()
  
  trend_res <- trend_m %>%
    left_join(mult_m) %>%
    mutate(sephora_hits = hits * mult) %>%
    select(date, keyword, sephora_hits) %>%
    as_tibble() %>%
    mutate(date = as.Date(date))
  
  return(trend_res)
}

G_daily_S<-get_daily_gtrend(keyword = c('sephora'), geo = 'IT', from = '2018-10-15', to = '2020-02-13')


#Estimating Google trends of Douglas
get_daily_gtrend <- function(keyword = c('douglas'), geo = 'IT', from = '2018-10-15', to = '2020-02-13') {
  if (ymd(to) >= floor_date(Sys.Date(), 'month')) {
    to <- floor_date(ymd(to), 'month') - days(1)
    
    if (to < from) {
      stop("Specifying \'to\' date in the current month is not allowed")
    }
  }
  
  aggregated_data <- gtrends(keyword = keyword, geo = geo, time = paste(from, to))
  if(is.null(aggregated_data$interest_over_time)) {
    print('There is no data in Google Trends!')
    return()
  }
  
  mult_m <- aggregated_data$interest_over_time %>%
    mutate(hits = as.integer(ifelse(hits == '<1', '0', hits))) %>%
    group_by(month = floor_date(date, 'month'), keyword) %>%
    summarise(hits = sum(hits)) %>%
    ungroup() %>%
    mutate(ym = format(month, '%Y-%m'),
           mult = hits / max(hits)) %>%
    select(month, ym, keyword, mult) %>%
    as_tibble()
  
  pm <- tibble(s = seq(ymd(from), ymd(to), by = 'month'), 
               e = seq(ymd(from), ymd(to), by = 'month') + months(1) - days(1))
  
  raw_trends_m <- tibble()
  
  for (i in seq(1, nrow(pm), 1)) {
    curr <- gtrends(keyword, geo = geo, time = paste(pm$s[i], pm$e[i]))
    if(is.null(curr$interest_over_time)) next
    print(paste('for', pm$s[i], pm$e[i], 'retrieved', count(curr$interest_over_time), 'days of data (all keywords)'))
    raw_trends_m <- rbind(raw_trends_m,
                          curr$interest_over_time)
  }
  
  trend_m <- raw_trends_m %>%
    select(date, keyword, hits) %>%
    mutate(ym = format(date, '%Y-%m'),
           hits = as.integer(ifelse(hits == '<1', '0', hits))) %>%
    as_tibble()
  
  trend_res <- trend_m %>%
    left_join(mult_m) %>%
    mutate(douglas_hits = hits * mult) %>%
    select(date, keyword, douglas_hits) %>%
    as_tibble() %>%
    mutate(date = as.Date(date))
  
  return(trend_res)
}

G_daily_D<-get_daily_gtrend(keyword = c('douglas'), geo = 'IT', from = '2018-10-15', to = '2020-02-13')

#Combining Google Trends for both Sephora and Douglas
Gdaily_DS <- cbind(G_daily_D, G_daily_S)
Gdaily_DS<- Gdaily_DS[-488, ]
Gdaily_DS$keyword <- NULL
Gdaily_DS$keyword <- NULL

#Retrieve total sales and visits
df <- read.csv('/Users/carloardito/Deloitte/DeloitteR/Totals_R.csv', sep=",")
df$X <- NULL
df$tot_onpurc <- NULL

#Combining sales,visits with g trens from D and S
dfmaster <- cbind(df, Gdaily_DS)
dfmaster$date <- NULL
dfmaster$date <- NULL
dfmaster$keyword <- NULL
dfmaster$keyword <- NULL

#retrieve the Google query about the products
dfquery <- read.csv('/Users/carloardito/Deloitte/DeloitteR/dataset.csv', sep=",")
dfquery$day <- NULL
dfquery$convrate <- NULL
dfquery$avspend <- NULL
dfquery$visits <- NULL
dfquery$online_purchases <- NULL

#Casual Impact on Sales,Visits,D,S and Google query
df5 <- cbind(dfmaster, dfquery)
pre.period <- c(1,414) #fino a inizio campagna
post.period <- c(415,487)
dfmaster$day <- NULL

impact <- CausalImpact(df5, pre.period, post.period)
plot(impact)
summary(impact, "report")

#Casual Impact on total sales only
dfmaster$tot_visits <- NULL
impact <- CausalImpact(dfmaster, pre.period, post.period)
plot(impact)
summary(impact, "report")

#Casual Impact on total visits only
dfmaster$tot_sales <- NULL
impact <- CausalImpact(dfmaster, pre.period, post.period)
plot(impact)
summary(impact, "report")

#Casual Impact on total visits and Google query
dfmaster <- cbind(dfmaster, dfquery)
impact <- CausalImpact(dfmaster, pre.period, post.period)
plot(impact)
summary(impact, "report")

