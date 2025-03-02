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

qgam_eq <- Net_demand ~ s(as.numeric(Date), k = 3, bs = "cr") +
  s(toy, k = 30, bs = "cc") + s(Temp, k = 10, bs = "cr") +
  s(Load.1, bs = "cr", by =  as.factor(WeekDays)) +
  s(Load.7, bs = "cr")  + as.factor(WeekDays) + as.factor(BH) +
  s(Net_demand.1, bs = "cr") + s(Net_demand.7, bs = "cr")

qgam_xgb_kf <- function(train, test, qgam_eq, quantile) {
  # Separate train sets for qgam and xgb.
  full_data <- rbind(train, test)
  nrow_train <- nrow(train)
  idx <- sample(nrow_train, .5 * nrow_train)
  train_qgam <- train[idx, ]
  train_xgb <- train[-idx, ]

  # Train qgam model.
  qgam_model <- qgam(qgam_eq, data = train_qgam, qu = quantile)

  # Boost qgam model with xgboost.
  qgam_train_pred <- predict(qgam_model, newdata = train_xgb)
  qgam_train_res <- train_xgb$Net_demand - qgam_train_pred
  dtrain <- xgb.DMatrix(data = as.matrix(train_xgb[, -1]),
                        label = qgam_train_res)
  xgb_params <- list(objective = "reg:squarederror", eta = .1, max_depth = 3)
  xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100)

  # Kalman filtering with expectation maximization on final prediction.
  qgam_pred <- predict(qgam_model, newdata = full_data, type = "terms")
  xgb_pred <- predict(xgb_model, newdata = as.matrix(full_data[, -1]))
  input <- cbind(qgam_pred, xgb_pred) %>% scale %>% cbind(1)
  target <- full_data$Net_demand
  ssm <- statespace(input, target)
  ssm_em <- select_Kalman_variances(ssm, input[seq_len(nrow_train), ],
                                    target[seq_len(nrow_train)],
                                    Q_init = diag(ncol(input)), method = "em",
                                    n_iter = 1000, verbose = 100)
  predict(ssm_em, input, target, type = "model", compute_smooth = TRUE)
}

qgam_80 <- qgam_xgb_kf(train, test, qgam_eq, .8)

idx <- sample(nrow(train), .5 * nrow(train))
train_qgam <- train[idx, ]
train_xgb <- train[-idx, ]

qgam_eq <- Net_demand ~ s(as.numeric(Date), k = 3, bs = "cr") +
  s(toy, k = 30, bs = "cc") + s(Temp, k = 10, bs = "cr") +
  s(Load.1, bs = "cr", by =  as.factor(WeekDays)) +
  s(Load.7, bs = "cr")  + as.factor(WeekDays) + as.factor(BH) +
  s(Net_demand.1, bs = "cr") + s(Net_demand.7, bs = "cr")

qgam_model <- qgam(qgam_eq, data = train_qgam, qu = .8)

qgam_train_pred <- predict(qgam_model, newdata = train_xgb)
qgam_train_res <- train_xgb$Net_demand - qgam_train_pred
dtrain <- xgb.DMatrix(data = as.matrix(train_xgb[, -1]), label = qgam_train_res)
xgb_params <- list(objective = "reg:squarederror", eta = .1, max_depth = 3)
xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100)

qgam_pred <- predict(qgam_model, newdata = data0)
qgam_terms <- predict(qgam_model, newdata = data0, type = "terms")
xgb_pred <- predict(xgb_model, newdata = as.matrix(data0[, -1]))
boosted_pred <- qgam_pred + xgb_pred
input <- cbind(qgam_terms, boosted_pred) %>% scale %>% cbind(1)
target <- data0$Net_demand

ssm <- statespace(input, target)
ssm_em <- select_Kalman_variances(ssm, input[sel_a, ], target[sel_a],
                                  Q_init = diag(ncol(input)), method = "em",
                                  n_iter = 1000, verbose = 100)
saveRDS(ssm_em, "Results/ssm_em.RDS")

ssm_em <- readRDS("Results/ssm_em.RDS")
ssm_em_pred <- predict(ssm_em, input, target, type = "model",
                       compute_smooth = TRUE)
final_pred <- tail(ssm_em_pred$pred_mean, nrow(test))

pinball_loss(test$Net_demand, qgam_80, quant = .8, output.vect = FALSE)
rmse(test$Net_demand, final_pred)

plot(test$Net_demand, type = "l")
lines(final_pred, col = "red")

final_res <- test$Net_demand - final_pred
acf(final_res)
