rm(list = objects())

library(mgcv)
library(yarrr)
library(magrittr)
library(forecast)
library(tidyverse)
library(qgam)
library(xgboost)
library(viking)
source("score.R")

data0 <- read_delim("data/Data0.csv", delim = ",")
data1 <- read_delim("data/Data1.csv", delim = ",")

data0$Time <- as.numeric(data0$Date)
data1$Time <- as.numeric(data1$Date)

sel_a <- which(data0$Year <= 2021)
sel_b <- which(data0$Year > 2021)
train <- data0[sel_a, ]
test <- data0[sel_b, ]

# qgam model.
qgam_eq <- Net_demand ~ s(as.numeric(Date), k = 3, bs = "cr") +
  s(toy, k = 30, bs = "cc") + s(Temp, k = 10, bs = "cr") +
  s(Load.1, bs = "cr", by =  as.factor(WeekDays)) +
  s(Load.7, bs = "cr")  + as.factor(WeekDays) + as.factor(BH) +
  s(Net_demand.1, bs = "cr") + s(Net_demand.7, bs = "cr")

qgam <- qgam(qgam_eq, data = train, qu = .8)

# Boosting : xgboost is trained on the residuals from qgam.
qgam_pred <- predict(qgam, type = "terms")
qgam_res <- train$Net_demand - qgam_pred
dtrain <- xgb.DMatrix(data = as.matrix(train[, -1]), label = qgam_res)
xgb_params <- list(objective = "reg:squarederror", eta = .1, max_depth = 3)
xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100)

# Online learning : apply kalman filtering to the boosted prediction.
qgam_pred <- predict(qgam, newdata = data0, type = "terms")
xgb_pred <- predict(xgb_model, newdata = as.matrix(data0[, -1]), type = "terms")
kalman_input <- qgam_pred + xgb_pred
kalman_target <- data0$Net_demand

for (j in seq_len(ncol(kalman_input))){
  kalman_input[, j] <- (kalman_input[, j] - mean(kalman_input[, j])) /
    sd(kalman_input[, j])
}

kalman_input <- cbind(kalman_input, 1)
input_dim <- ncol(kalman_input)

ssm <- statespace(kalman_input, kalman_target)
ssm_em <- select_Kalman_variances(ssm, kalman_input[sel_a, ], kalman_target,
                                  method = "em", n_iter = 1000,
                                  Q_init = diag(input_dim), verbose = 10,
                                  mode_diag = TRUE)

saveRDS(ssm_em, "Results/ssm_em.RDS")
ssm_em <- readRDS("Results/ssm_em.RDS")
kalman_pred <- predict(ssm_em, kalman_input, kalman_target)
final_pred <- kalman_pred %>% tail(length(sel_b))

pinball_loss(data0$Net_demand[sel_b], final_pred, quant = .8,
             output.vect = FALSE)
