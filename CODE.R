#install.packages("devtools")*
library("devtools")
#install.packages("bsts")
#install.packages("xts")
library("bsts")
library("xts")
#install.packages("DBI")
library("DBI")
#install.packages("Rcpp")
library("Rcpp")
#install.packages("magrittr")
library("magrittr")
#install.packages("tibble")
library("tibble")
#install.packages("gtable")
library("gtable")
#install.packages("scales")
library("scales")
devtools::install_github("google/CausalImpact", force = TRUE)
#install.packages("CausalImpact")
library(CausalImpact)
library(dplyr)

df <- read.csv('/Users/martinamanno/Desktop/LUISS/CORSI da seguire/DATA SCIENCE IN ACTION/DELOITTE/progetto 2/Totals_R.csv', sep=",")
df$X <- NULL
#df$day <- NULL
df$tot_visits <- NULL
df$tot_onpurc <- NULL

pre.period <- c(1,414) #fino a inizio campagna
post.period <- c(415,487)
impact <- CausalImpact(df1, pre.period, post.period)
plot(impact)
summary(impact, "report")


'''
dfg <- read.csv('/Users/martinamanno/Desktop/LUISS/CORSI da seguire/DATA SCIENCE IN ACTION/DELOITTE/progetto 2/Gsearchdata.csv', sep=",")
dfg$city <- NULL
dfg$day <- NULL
df1 <- cbind(df, dfg)
df1$day <- NULL
df1$Gsearchprod <- NULL
'''

install.packages("gtrendsR")
devtools::install_github('PMassicotte/gtrendsR')
library(gtrendsR)

#gtrends(keyword=c("Douglas","Sephora"), geo='IT',time= c("all", time= c("all", "2018-10-1 2018-10-31","2018-11-1 2018-11-30","2018-12-1 2018-12-31", "2019-01-1 2019-01-31","2019-02-1 2019-02-28","2019-03-1 2019-03-31","2019-04-1 2019-04-30","2019-05-1 2019-05-31", "2019-06-1 2019-06-30","2019-07-1 2019-07-31", "2019-08-1 2019-08-31", "2019-09-1 2019-09-30","2019-10-1 2019-10-31", "2019-11-1 2019-11-30","2019-12-1 2019-12-31","2020-01-1 2020-01-31", "2020-02-01 2020-02-28"), gprop ="web", hl = "it-IT")
a<-head(gtrends("Douglas"), geo='ITA',time= c("all", time= c("all", "2018-10-1 2020-02-28"), gprop ="web", hl = "it-IT")$interest_over_time)
capture.output(a$interest_over_time, file = "a.csv")
a= a$interest_over_time
a<-a[a[["date"]] >= "2018-10-14", ]
a<-a[a[["date"]] <= "2020-02-16", ]

a$keyword <- NULL
a$geo <- NULL
a$time <- NULL
a$gprop <- NULL
a$category <- NULL


c<- head(gtrends("Sephora"), geo='ITA',time= c("all", time= c("all", "2018-10-1 2020-02-28"), gprop ="web", hl = "it-IT")$interest_over_time)
capture.output(c$interest_over_time, file = "a.csv")
c= c$interest_over_time
c<-c[c[["date"]] >= "2018-10-14", ]
c<-c[c[["date"]] <= "2020-02-16", ]

c$keyword <- NULL
c$geo <- NULL
c$time <- NULL
c$gprop <- NULL
c$category <- NULL


z<- head(gtrends("Zalando"), geo='ITA',time= c("all", time= c("all", "2018-10-1 2020-02-28"), gprop ="web", hl = "it-IT")$interest_over_time)
capture.output(z$interest_over_time, file = "a.csv")
z= z$interest_over_time
z<-z[z[["date"]] >= "2018-10-14", ]
z<-z[z[["date"]] <= "2020-02-16", ]

z$keyword <- NULL
z$geo <- NULL
z$time <- NULL
z$gprop <- NULL
z$category <- NULL

h <-head(gtrends("Chanel"), geo='ITA',time= c("all", time= c("all", "2018-10-1 2020-02-28"), gprop ="web", hl = "it-IT")$interest_over_time)
capture.output(h$interest_over_time, file = "a.csv")
h= h$interest_over_time
h<-h[h[["date"]] >= "2018-10-14", ]
h<-h[h[["date"]] <= "2020-02-16", ]

h$keyword <- NULL
h$geo <- NULL
h$time <- NULL
h$gprop <- NULL
h$category <- NULL

'''
x <- df$day
as.numeric(x)

Week <- as.Date(cut(df$day, "week"))

aggregate(Frequency ~ Week, df, sum)

weekno <- as.numeric(df$day - df$day[1]) %/% 7
Week <- DF$Date[match(weekno, weekno)]
aggregate(Frequency ~ Week, DF, sum)
'''


df$date<- df$day
df$day <- NULL

df$seven_day_index <- c(0, rep(1:(nrow(df)-1)%/%7))
b<-df%>%
  group_by(seven_day_index) %>% 
  summarise(value = mean(tot_sales))

df_new <- cbind(a, b,c,z, h)
df_new$seven_day_index <- NULL


## Causal Impact modello 1
pre.period <- c(1,60) #fino a inizio campagna
post.period <- c(61,70)
df_new$date <- NULL

impact <- CausalImpact(df_new, pre.period, post.period)
plot(impact)
summary(impact, "report")



#Causal Impact modello 2
df_new <- df_new[1:60, ]
pre.period <- c(1,30)
post.period <- c(31,60)
df_new$seven_day_index <- NULL
df_new$date <- NULL
impact <- CausalImpact(df_new, pre.period, post.period)
plot(impact)
summary(impact, "report")



#Causal Impact modello df originale
dforiginal <- read.csv('/Users/martinamanno/Desktop/LUISS/CORSI da seguire/DATA SCIENCE IN ACTION/DELOITTE/progetto 2/Totals_R.csv', sep=",")
dforiginal$X <- NULL
pre.period <- c(1,414)
post.period <- c(415, 487)
dforiginal$day <- NULL
impact <- CausalImpact(dforiginal, pre.period, post.period)
plot(impact)
summary(impact, "report")


#Causal Impact modello su visite

df_visits1 <- dforiginal['tot_visits']
df_visits2 <- dforiginal['day']
df_visits <- cbind(df_visits1, df_visits2)

df_visits$seven_day_index <- c(0, rep(1:(nrow(df)-1)%/%7))

df_visits<-df_visits%>%
  group_by(seven_day_index) %>% 
  summarise(value_ = mean(tot_visits))

df_visits <- cbind(df_new, df_visits)

df_visits$seven_day_index <- NULL
df_visits$value <- NULL
df_visits$date <- NULL
pre.period <- c(1,60) #fino a inizio campagna
post.period <- c(61,70)

names(df_visits)[names(df_visits) == "hits"] <- "v1"
df_visits <- df_visits %>% 
  rename(hits1 = 1, 
         hits2=2,
         hits3 =3,
         hits4 = 4)

impact <- CausalImpact(df_visits, pre.period, post.period)
plot(impact)
summary(impact, "report")

