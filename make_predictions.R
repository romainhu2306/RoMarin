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

# Formatting data1 so column names correspond to data0.
data1 <- data1[, -c(36, 37)]
missing_cols <- setdiff(names(data0), names(data1))
for (col in missing_cols) {
  data1[[col]] <- NA
}
data1 <- data1[, names(data0)]

data0$Time <- as.numeric(data0$Date)
data1$Time <- as.numeric(data1$Date)

# For validation.
sel_a <- which(data0$Year <= 2020)
train <- data0[sel_a, ]
test <- data0[-sel_a, ]

# For prediction on data1.
sel_a <- seq_len(nrow(data0))

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
  dtrain <- xgb.DMatrix(data = as.matrix(train_xgb[, -c(1, 2, 5, 6, 7)]),
                        label = qgam_train_res)
  xgb_params <- list(objective = "reg:squarederror", eta = .1, max_depth = 3,
                     verbosity = 2)
  xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100)

  # Kalman filtering with expectation maximization on final prediction.
  qgam_terms <- predict(qgam_model, newdata = full_data, type = "terms")
  qgam_pred <- predict(qgam_model, newdata = full_data)
  xgb_pred <- predict(xgb_model,
                      newdata = as.matrix(full_data[, -c(1, 2, 5, 6, 7)]))
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
nb_cores <- detectCores() - 2
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
preds <- as.data.frame(preds)
preds <- cbind(data0$Net_demand, preds)
colnames(preds)[1] <- "Net_demand"
write.csv(preds, "train_preds_pin.csv", row.names = FALSE)

# Train aggregator.
preds <- read_csv("train_preds_base.csv")
aggregator <- lm(Net_demand ~ ., data = preds[sel_a, -c(2, 3, 9)])

# Study results.
aggreg_pred <- predict(aggregator, newdata = preds)
final_pred <- aggreg_pred[-sel_a]

train_res <- train$Net_demand - aggreg_pred[sel_a]
q_norm <- qnorm(.8, mean = mean(train_res), sd = sd(train_res))
final_pred <- final_pred + q_norm
pinball_loss(test$Net_demand, final_pred, quant = .8,
             output.vect = FALSE)

rmse(test$Net_demand, final_pred)
plot(test$Net_demand, type = "l")
lines(final_pred, col = "red")

final_res <- test$Net_demand - final_pred
plot(final_res, type = "l")
abline(h = 0, col = "red")

plot(cumsum(final_res), type = "l", col = "red")
for (i in 2:10) {
  predq <- unlist(preds[, i])
  final_predq <- predq[-sel_a]
  train_resq <- train$Net_demand - predq[sel_a]
  q_normq <- qnorm(.8, mean = mean(train_resq), sd = sd(train_resq))
  final_predq <- final_predq + q_normq
  final_resq <- test$Net_demand - final_predq
  lines(cumsum(final_resq), type = "l", col = "blue")
}

acf(final_res)

hist(final_res, breaks = 30)

qqnorm(final_res)
qqline(final_res, col = "red")

submit <- data.frame(Id = seq_len(length(final_pred)), Net_demand = final_pred)
write.table(submit, file = "pred.csv", quote = FALSE, sep = ",", dec = ".",
            row.names = FALSE)













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
  myobjective <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    grad <- (labels < preds) - quantile
    hess <- rep(1, length(labels))
    list(grad = grad, hess = hess)
  }

  evalerror <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    u <- (labels - preds) * (quantile - (labels < preds))
    err <- sum(u) / length(u)
    list(metric = "MyError", value = err)
  }

  qgam_train_pred <- predict(qgam_model, newdata = train_xgb)
  qgam_train_res <- train_xgb$Net_demand - qgam_train_pred
  dtrain <- xgb.DMatrix(data = as.matrix(train_xgb[, -c(1, 2, 5, 6, 7)]),
                        label = qgam_train_res)
  xgb_params <- list(objective = myobjective, eval_metric = evalerror,
                     eta = .1, max_depth = 3, verbosity = 2, maximize = TRUE)
  xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100)

  # Kalman filtering with expectation maximization on final prediction.
  qgam_terms <- predict(qgam_model, newdata = full_data, type = "terms")
  qgam_pred <- predict(qgam_model, newdata = full_data)
  xgb_pred <- predict(xgb_model,
                      newdata = as.matrix(full_data[, -c(1, 2, 5, 6, 7)]))
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





idx <- sample(nrow(train), .5 * nrow(train))
train_qgam <- train[idx, ]
train_xgb <- train[-idx, ]

qgam_model <- qgam(qgam_eq, data = train_qgam, qu = .8)

qgam_train_pred <- predict(qgam_model, newdata = train_xgb)
qgam_train_res <- train_xgb$Net_demand - qgam_train_pred

dtrain <- xgb.DMatrix(data = as.matrix(train_xgb[, -c(1, 2, 5, 6, 7)]),
                      label = qgam_train_res)
xgb_params <- list(booster = "gbtree", eta = .1, objective = myobjective,
                   eval_metric = evalerror, max_depth = 3, subsample = .5,
                   colsample_bytree = .8)

xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100,
                       maximize = FALSE, verbose = TRUE)

qgam_terms <- predict(qgam_model, newdata = data0, type = "terms")
qgam_pred <- predict(qgam_model, newdata = data0)
xgb_pred <- predict(xgb_model, newdata = as.matrix(data0[, -c(1, 2, 5, 6, 7)]))
boosted_pred <- qgam_pred + xgb_pred

input <- cbind(qgam_terms, boosted_pred) %>% scale %>% cbind(1)
target <- data0$Net_demand
ssm <- statespace(input, target)
ssm_em <- select_Kalman_variances(ssm, input[sel_a, ],
                                  target[sel_a],
                                  Q_init = diag(ncol(input)), method = "em",
                                  n_iter = 1000, verbose = 100)
ssm_em_pred <- predict(ssm_em, input, target, type = "model",
                       compute_smooth = TRUE)

train_res <- train$Net_demand - ssm_em_pred$pred_mean[sel_a]
q_norm <- qnorm(.8, mean = mean(train_res), sd = sd(train_res))
final_pred <- ssm_em_pred$pred_mean[-sel_a] + q_norm
pinball_loss(test$Net_demand, final_pred, quant = .8, output.vect = FALSE)
rmse(test$Net_demand, final_pred)




idx <- sample(nrow(train), .5 * nrow(train))
train_qgam <- train[idx, ]
train_xgb <- train[-idx, ]

equation <- Net_demand ~ Time + toy + Temp + Load.1 + Load.7 + Temp_s99 +
  as.factor(WeekDays) + BH + Temp_s95_max + Temp_s99_max + Summer_break +
  Christmas_break + Temp_s95_min + Temp_s99_min + DLS

m0 <- model.matrix(equation %>% as.formula, data = train_xgb)
x0 <- m0[, -1]
m1 <- model.matrix(equation %>% as.formula, data = train_qgam)
x1 <- m1[, -1]

dtrain <- xgb.DMatrix(data = x0, label = train_xgb$Net_demand)
dtest <- xgb.DMatrix(data = x1, label = train_qgam$Net_demand)

param2 <- list(booster = "gbtree"
               , learning_rate = 100
               , objective = myobjective
               , eval_metric = evalerror
               , max_depth = 1
               , subsample = 0.5
               , colsample_bytree = 0.8)

xgb2 <- xgb.train(params = param2
                  , data = dtrain
                  , nrounds = 10000
                  , watchlist
                  , maximize = FALSE
                  , verbose = TRUE
                  , early_stopping_rounds = 100)
xgb2_pred <- predict(xgb2, dtest)
xgb2_train_res <- train_qgam$Net_demand - xgb2_pred

train_qgam <- cbind(xgb2_train_res, train_qgam)

qgam_eq <- xgb2_train_res ~ s(as.numeric(Date), k = 3, bs = "cr") +
  s(toy, k = 30, bs = "cc") + s(Temp, k = 10, bs = "cr") +
  s(Load.1, bs = "cr", by =  as.factor(WeekDays)) +
  s(Load.7, bs = "cr")  + as.factor(WeekDays) + as.factor(BH) +
  s(Net_demand.1, bs = "cr") + s(Net_demand.7, bs = "cr")

m2 <- model.matrix(equation %>% as.formula, data = data0)
x2 <- m2[, -1]

dtest2 <- xgb.DMatrix(data = x2, label = data0$Net_demand)
qgam_model <- qgam(qgam_eq, data = train_qgam, qu = .8)

xgb2_pred <- predict(xgb2, dtest2)
qgam_pred <- predict(qgam_model, data0)
boosted_pred <- xgb2_pred + qgam_pred
qgam_terms <- predict(qgam_model, data0, type = "terms")

input <- cbind(qgam_terms, boosted_pred) %>% scale %>% cbind(1)
target <- data0$Net_demand
ssm <- statespace(input, target)
ssm_em <- select_Kalman_variances(ssm, input[sel_a, ],
                                  target[sel_a],
                                  Q_init = diag(ncol(input)), method = "em",
                                  n_iter = 1000, verbose = 100)
ssm_em_pred <- predict(ssm_em, input, target, type = "model",
                       compute_smooth = TRUE)

train_res <- train$Net_demand - ssm_em_pred$pred_mean[sel_a]
q_norm <- qnorm(.8, mean = mean(train_res), sd = sd(train_res))
final_pred <- ssm_em_pred$pred_mean[-sel_a] + q_norm

pinball_loss(test$Net_demand, final_pred, quant = .8,
             output.vect = FALSE)
