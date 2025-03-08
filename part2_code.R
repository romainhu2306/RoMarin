# CLEAR ENVIRONMENT
rm(list = objects())

# LOAD LIBRARIES
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

# SCORE FUNCTIONS
source("score.R")

# DATA
data0 <- read_delim("data/Data0.csv", delim = ",")
data1 <- read_delim("data/Data1.csv", delim = ",")

# MERGING DATA1 AND DATA0
# Remove usage and Id columns.
data1 <- data1[, -c(36, 37)]

# Use lagged features to reconstruct missing features.
data1 <- data1 %>% mutate(Net_demand = lead(Net_demand.1, n = 1))
data1$Net_demand[nrow(data1)] <- data1$Net_demand[nrow(data1) - 1]

data1 <- data1 %>% mutate(Load = lead(Load.1, n = 1))
data1$Load[nrow(data1)] <- data1$Load[nrow(data1) - 1]

data1 <- data1 %>% mutate(Solar_power = lead(Solar_power.1, n = 1))
data1$Solar_power[nrow(data1)] <- data1$Solar_power[nrow(data1) - 1]

data1 <- data1 %>% mutate(Wind_power = lead(Wind_power.1, n = 1))
data1$Wind_power[nrow(data1)] <- data1$Wind_power[nrow(data1) - 1]

# Reordering column names.
data1 <- data1[, names(data0)]

# Create full dataset for online learning.
full_data <- rbind(data0, data1)

# For identifying data0 in the full dataset.
data0_idx <- seq_len(nrow(data0))


hypotrochoid <- function(t, r1, r2, r3, w1, w2, w3, d1, d2) {
  (r1 - r2) * cos(w1 * t) +
    d1 * cos((r1 - r2) / r2 * w2 * t) +
    d2 * cos((r2 - r3) / r3 * w3 * t)
}

sequence <- seq(0, 30, by = .01)
graph <- hypotrochoid(sequence,
                      r1 = 10,
                      r2 = 9,
                      r3 = 1,
                      w1 = 1,
                      w2 = 1,
                      w3 = 1,
                      d1 = 1,
                      d2 = 1)
plot(sequence, graph, type = "l", lwd = 2)





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
