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
source("score.R")

data0 <- read_delim("data/Data0.csv", delim = ",")
data1 <- read_delim("data/Data1.csv", delim = ",")

data0$Time <- as.numeric(data0$Date)
data1$Time <- as.numeric(data1$Date)

sel_a <- which(data0$Year <= 2020)
train <- data0[sel_a, ]
test <- data0[-sel_a, ]

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
  xgb_params <- list(objective = "reg:squarederror", eta = .1, max_depth = 3,
                     verbosity = 2)
  xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100)

  # Kalman filtering with expectation maximization on final prediction.
  qgam_terms <- predict(qgam_model, newdata = full_data, type = "terms")
  qgam_pred <- predict(qgam_model, newdata = full_data)
  xgb_pred <- predict(xgb_model, newdata = as.matrix(full_data[, -1]))
  boosted_pred <- qgam_pred + xgb_pred
  input <- cbind(qgam_terms, boosted_pred) %>% scale %>% cbind(1)
  target <- full_data$Net_demand
  ssm <- statespace(input, target)
  ssm_em <- select_Kalman_variances(ssm, input[seq_len(nrow_train), ],
                                    target[seq_len(nrow_train)],
                                    Q_init = diag(ncol(input)), method = "em",
                                    n_iter = 1000, verbose = 100)
  ssm_em_pred <- predict(ssm_em, input, target, type = "model",
                         compute_smooth = TRUE)
  ssm_em_pred$pred_mean
}

# Parallel processing.
nb_cores <- detectCores() - 3
cluster <- makeCluster(nb_cores)
registerDoParallel(cluster)

clusterExport(cluster, varlist = c("qgam_xgb_kf", "train", "test", "qgam_eq"))
quantiles <- seq(.1, .9, by = .1)
preds <- foreach(q = quantiles, .combine = cbind,
                 .packages = c("qgam", "xgboost", "viking", "tidyverse",
                               "magrittr")) %dopar% {
  qgam_xgb_kf(train, test, qgam_eq, q)
}

stopCluster(cluster)
preds <- cbind(data0$Net_demand, preds)

# Sequential processing.
pred_90 <- qgam_xgb_kf(train, test, qgam_eq, .9)
pred_80 <- qgam_xgb_kf(train, test, qgam_eq, .8)
pred_70 <- qgam_xgb_kf(train, test, qgam_eq, .7)
pred_60 <- qgam_xgb_kf(train, test, qgam_eq, .6)

preds <- data.frame(Net_demand = data0$Net_demand, pred_90, pred_80, pred_70,
                    pred_60)

# Train aggregator.
aggregator <- lm(Net_demand ~ ., data = preds[sel_a, ])

# Study results.
aggreg_pred <- predict(aggregator, newdata = preds)
final_pred <- aggreg_pred[-sel_a]

train_res <- train$Net_demand - aggreg_pred[sel_a]
q_norm <- qnorm(.8, mean = mean(train_res), sd = sd(train_res))
pinball_loss(test$Net_demand, final_pred + q_norm, quant = .8,
             output.vect = FALSE)

rmse(test$Net_demand, final_pred)
plot(test$Net_demand, type = "l")
lines(final_pred, col = "red")

final_res <- test$Net_demand - final_pred
plot(final_res, type = "l")

plot(cumsum(test$Net_demand - final_pred[]), type = "l", col = "red")

acf(final_res)

hist(final_res, breaks = 30)

qqnorm(final_res)
qqline(final_res, col = "red")

shapiro.test(final_res)
