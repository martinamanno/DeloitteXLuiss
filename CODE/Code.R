# Import required libraries
library(MASS)
library(tidyverse)
library(ggplot2)
library(data.table)
library(ggplot2)
require(pROC)
require(class)
require(magrittr)
library(dplyr)
library(skimr)
require(pscl)
require(boot)
library(DAAG)
library(gtrendsR)
library(lubridate)
library(CausalImpact)


# Open datasets
df <- read.csv('/Users/martinamanno/Desktop/LUISS/CORSI da seguire/DATA SCIENCE IN ACTION/DELOITTE/progetto 2/FINAL/DATASET/dataset_final.csv', sep=",")
df1 <- read.csv('/Users/martinamanno/Desktop/LUISS/CORSI da seguire/DATA SCIENCE IN ACTION/DELOITTE/progetto 2/FINAL/DATASET/df_dummy.csv', sep =",")

# Visualize the variable that will be the y of the model
hist(df$sales)
df_final =  cbind(df, df1)


# Linear Regression Model
model4 <- lm(sales ~ online_purchases+visits + Gsearchprod + Si.no + day_of_week_Friday +day_of_week_Monday + day_of_week_Saturday+ day_of_week_Sunday + day_of_week_Thursday + day_of_week_Tuesday + day_of_week_Wednesday  + month_December + month_February + month_January + month_July + month_June +month_March + month_May + month_November + month_October+ month_September + month_April + month_August, data =df_final)
summary(model4)
model_perf4 <- step(model4)
summary(model_perf4)
model_perf <- lm(sales ~  visits +  Si.no +  month_December + day_of_week_Sunday , data =df_final)
summary(model_perf)


# Causal Impact Algorithm

# Get daily data from Google trend to create a control group for applying the algorithm
# Sephora
get_daily_gtrend1 <- function(keyword = c('sephora'), geo = 'IT', from = '2018-10-15', to = '2020-02-13') { 
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
    mutate(est_hits = hits * mult) %>%
    select(date, keyword, est_hits) %>%
    as_tibble() %>%
    mutate(date = as.Date(date))
  
  return(trend_res)
}

Gdaily1 <- get_daily_gtrend1(keyword = c('sephora'), geo = 'IT', from = '2018-10-15', to = '2020-02-13') 


# Douglas
get_daily_gtrend2 <- function(keyword = c('douglas'), geo = 'IT', from = '2018-10-15', to = '2020-02-13') { 
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
    mutate(est_hits = hits * mult) %>%
    select(date, keyword, est_hits) %>%
    as_tibble() %>%
    mutate(date = as.Date(date))
  
  return(trend_res)
}

Gdaily2<-get_daily_gtrend2(keyword = c('douglas'), geo = 'IT', from = '2018-10-15', to = '2020-02-13') 

Gdaily_completo <- cbind(Gdaily2, Gdaily)
Gdaily_completo<- Gdaily_completo[-488, ]
Gdaily_completo$date <- NULL
Gdaily_completo$keyword <- NULL
Gdaily_completo$keyword <- NULL
colnames(Gdaily_completo)[3] <- "Sephora"
colnames(Gdaily_completo)[1] <- "Douglas"


# Prepare the dataset with only the variables of interest
df$convrate <- NULL
df$avspend <- NULL
df$online_purchases <- NULL
dfmod<- cbind(df, Gdaily_completo)
dfmod$date <- NULL


# Split the dataset in period: pre and post ad campaign
pre.period <- c(1,414) #fino a inizio campagna
post.period <- c(415,487)
dfmod$day <- NULL


impact <- CausalImpact(dfmod, pre.period, post.period)
plot(impact)
summary(impact, "report")

