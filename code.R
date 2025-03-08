rm(list = objects())

library(mgcv)
library(yarrr)
library(magrittr)
library(forecast)
library(tidyverse)
library(qgam)
library(xgboost)
library(viking)
library(foreach)
library(doParallel)
library(data.table)
library(opera)
source("score.R")

data0 <- read_delim("data/Data0.csv", delim = ",")
data1 <- read_delim("data/Data1.csv", delim = ",")

# Formatting data1 so column names correspond to data0.
data1 <- data1[, -c(36, 37)]
data1 <- data1 %>% mutate(Net_demand = lead(Net_demand.1, n = 1))
data1$Net_demand[nrow(data1)] <- data1$Net_demand[nrow(data1) - 1]
data1 <- data1 %>% mutate(Load = lead(Load.1, n = 1))
data1$Load[nrow(data1)] <- data1$Load[nrow(data1) - 1]
data1 <- data1 %>% mutate(Solar_power = lead(Solar_power.1, n = 1))
data1$Solar_power[nrow(data1)] <- data1$Solar_power[nrow(data1) - 1]
data1 <- data1 %>% mutate(Wind_power = lead(Wind_power.1, n = 1))
data1$Wind_power[nrow(data1)] <- data1$Wind_power[nrow(data1) - 1]
data1 <- data1[, names(data0)]

data0$Time <- as.numeric(data0$Date)
data1$Time <- as.numeric(data1$Date)

# For prediction on data1.
sel_a <- seq_len(nrow(data0))
full_data <- rbind(data0, data1)

pinball_loss(data1$Net_demand, final_pred, quant = .8,
             output.vect = FALSE)

rmse(data1$Net_demand, final_pred)
plot(data1$Net_demand, type = "l")
lines(final_pred, col = "red")

final_res <- data1$Net_demand - final_pred
plot(data1$Date, final_res,
     type = "l",
     xlab = "Time",
     ylab = "Residuals")
abline(h = 0,
       col = "red",
       lwd = 2)

acf(final_res)

hist(final_res,
     probability = TRUE,
     breaks = 30,
     col = "lightblue",
     ylab = "Residuals",
     main = "Histogram of the residuals")
lines(density(final_res),
      col = "red",
      lwd = 2)
legend("topleft",
       legend = "Density",
       col = "red",
       lty = 1)

qqnorm(final_res)
qqline(final_res, col = "red", lwd = 2)

shapiro.test(final_res)
