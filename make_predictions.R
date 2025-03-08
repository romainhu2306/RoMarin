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

qgam_eq <- Net_demand ~ s(as.numeric(Date), k = 3, bs = "cr") +
  s(toy, k = 30, bs = "cc") + s(Temp, k = 10, bs = "cr") +
  s(Load.1, bs = "cr", by =  as.factor(WeekDays)) +
  s(Load.7, bs = "cr")  + as.factor(WeekDays) + as.factor(BH) +
  s(Solar_power.1, bs = "cr") + s(Net_demand.1, bs = "cr")

qgam_xgb_kf <- function(train, test, qgam_eq, quantile) {
  # Separate train sets for qgam and xgb.
  full_data <- rbind(train, test)
  nrow_train <- nrow(train)
  idx <- seq_len(.5 * nrow(train))
  train_qgam <- train[idx, ]
  train_xgb <- train[-idx, ]

  # Train qgam model.
  qgam_model <- qgam(qgam_eq, data = train_qgam, qu = quantile)

  # Boost qgam model with xgboost.
  qgam_train_pred <- predict(qgam_model, newdata = train_xgb)
  qgam_train_res <- train_xgb$Net_demand - qgam_train_pred
  dtrain <- xgb.DMatrix(data = as.matrix(train_xgb[, -c(1, 2, 5, 6, 7)]),
                        label = qgam_train_res)
  xgb_params <- list(objective = "reg:squarederror", eta = .1, max_depth = 3)
  xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 3000)

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
                         compute_smooth = TRUE)$pred_mean
  train_res <- train$Net_demand - ssm_em_pred[seq_len(nrow_train)]
  q_norm <- qnorm(q, mean = mean(train_res), sd = sd(train_res))
  ssm_em_pred[-seq_len(nrow_train)] <- ssm_em_pred[-seq_len(nrow_train)] + q_norm
  ssm_em_pred$pred_mean
}

# Parallel processing.
cluster <- makeCluster(9)
registerDoParallel(cluster)

clusterExport(cluster, varlist = c("qgam_xgb_kf", "data0", "data1", "qgam_eq"))
quantiles <- seq(.1, .9, by = .1)
preds <- foreach(q = quantiles, .combine = cbind,
                 .packages = c("qgam", "xgboost", "viking", "tidyverse",
                               "magrittr")) %dopar% {
  qgam_xgb_kf(data0, data1, qgam_eq, q)
}

stopCluster(cluster)
preds <- as.data.frame(preds)
preds <- cbind(full_data$Net_demand, preds)
colnames(preds)[1] <- "Net_demand"
write.csv(preds, "qgam_xgb_agg3.csv", row.names = FALSE)

# Train aggregator.
preds <- read.csv("qgam_xgb_agg2.csv")
target <- data1$Net_demand
experts <- preds[, -1]
experts[, 8] <- experts[, 8] - 500
agg <- mixture(target, experts, model = "MLpol",
               loss.type = list(name = "pinball", tau = .8))
plot(agg)

# Study results.
agg_pred <- agg$prediction

y <- full_data$Net_demand
precedent <- read.csv("qgam_xgb_agg2_penalized.csv")[, 2]
X <- agg_pred %>% scale %>% cbind(1)
ssm <- statespace(X, y)
ssm_em <- select_Kalman_variances(ssm, X[sel_a, ], y[sel_a],
                                  method = "em", Q_init <- diag(ncol(X)),
                                  n_iter = 1000, verbose = 100)
agg_pred <- predict(ssm_em, X, y, type = "model", compute_smooth = TRUE)$pred_mean

final_pred <- agg_pred[-sel_a]
final_pred <- agg_pred

train_res <- data0$Net_demand - agg_pred[sel_a]
q_norm <- qnorm(.8, mean = mean(train_res), sd = sd(train_res))
final_pred <- final_pred + q_norm
pinball_loss(data1$Net_demand, final_pred, quant = .8,
             output.vect = FALSE)

rmse(data1$Net_demand, final_pred)
plot(data1$Net_demand, type = "l")
lines(final_pred, col = "red")

final_res <- data1$Net_demand - final_pred
png("images/residuals.png", width = 600, height = 600)
plot(data1$Date, final_res, type = "l", xlab = "Time", ylab = "Residuals")
abline(h = 0, col = "red", lwd = 2)
dev.off()

png("images/acf.png", width = 600, height = 600)
acf(final_res)
dev.off()

png("images/hist.png", width = 600, height = 600)
hist(final_res, probability = TRUE, breaks = 30, col = "lightblue", xlab = "Residuals", main = "")
lines(density(final_res), col = "red", lwd = 2)
legend("topleft", legend = "Density", col = "red", lty = 1)
dev.off()

png("images/qqplot.png", width = 600, height = 600)
qqnorm(final_res, main = "")
qqline(final_res, col = "red", lwd = 2)
dev.off()

shapiro.test(final_res)

final_pred <- read.csv("qgam_xgb_agg2_penalized.csv")[, 2]
final_pred <- pred
submit <- data.frame(Id = seq_len(length(final_pred)), Net_demand = final_pred)
write.table(submit, file = "qgam_and_xgb2_penalized.csv", quote = FALSE, sep = ",", dec = ".",
            row.names = FALSE)




idx <- sample(nrow(data0), .5 * nrow(data0))
train_qgam <- data0[idx, ]
train_xgb <- data0[-idx, ]
qgam_model <- qgam(qgam_eq, data = train_qgam, qu = .8)

qgam_pred <- predict(qgam_model, train_xgb)
res <- train_xgb$Net_demand - qgam_pred

dtrain <- xgb.DMatrix(data = as.matrix(train_xgb[, -c(1, 2, 5, 6, 7)]),
                      label = res)
xgb_params <- list(objective = "reg:squarederror", eta = .1, max_depth = 3, alpha = 1, lambda = 5, subsample = .7, colsample_bytree = 1)

cv_params <- expand.grid(
                         eta = c(.01, .1, .3),
                         alpha = seq(1, 10, by = 3),
                         lambda = seq(1, 10, by = 3),
                         subsample = seq(.1, 1, by = .3),
                         colsample_bytree = seq(.1, 1, by = .3))

best_score <- Inf
best_params <- NULL
for (i in seq_len(nrow(cv_params))){
  params <- list(
                 eta = cv_params$eta[i],
                 alpha = cv_params$alpha[i],
                 lambda = cv_params$lambda[i],
                 subsample = cv_params$subsample[i],
                 colsample_bytree = cv_params$colsample_bytree[i])
  
  cv_results <- xgb.cv(params = params, data = dtrain, nrounds = 100, nfold = 5, stratified = TRUE, early_stopping_rounds = 5)
  mean_rmse <- min(cv_results$evaluation_log$test_rmse_mean)

  if (mean_rmse < best_score){
    best_score <- mean_rmse
    best_params <- params
  }
}
print(best_params)
xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 500)

pred1 <- predict(qgam_model, data1)
pred2 <- predict(xgb_model, as.matrix(data1[, -c(1, 2, 5, 6, 7)]))
pinball_loss(data1$Net_demand, pred1 + pred2, quant = .8)
rmse(data1$Net_demand, pred1 + pred2)





qgam_eq <- Net_demand ~ s(as.numeric(Date), k = 3, bs = "cr") +
  s(toy, k = 30, bs = "cc") + s(Temp, k = 10, bs = "cr") +
  s(Load.1, bs = "cr", by =  as.factor(WeekDays)) +
  s(Load.7, bs = "cr")  + as.factor(WeekDays) + as.factor(BH) +
  s(Solar_power.1, bs = "cr") + s(Net_demand.1, bs = "cr") +
  te(Wind, Wind_weighted) + te(Nebulosity, Nebulosity_weighted) +
  te(Temp_s99, Temp_s95_min) + te(Temp_s99, Temp_s95) + s(Temp_s95, bs = "cr")

qgam_model <- qgam(qgam_eq, data = data0, qu = .8)
pred_train <- predict(qgam_model)
pred_test <- predict(qgam_model, data1)
pinball_loss(data0$Net_demand, pred_train, qu = .8)
pinball_loss(data1$Net_demand, pred_test, qu = .8)
png("images/qgam_effects.png", width = 1200, height = 1200)
par(mfrow = c(2, 2))
plot(qgam_model, select = 2, shade = TRUE, col = "black", ylab = "Effect", xlab = "Time of year")
plot(qgam_model, select = 3, shade = TRUE, col = "red", shade.col = "pink", ylab = "Effect", xlab = "Temperature")
plot(qgam_model, select = 4, shade = TRUE, col = "blue", shade.col = "lightblue", ylab = "Effect", xlab = "Lagged load on Monday")
plot(qgam_model, select = 13, shade = TRUE, col = "orange", shade.col = "yellow", ylab = "Effect", xlab = "Lagged net demand")
dev.off()

png("images/nebulo_iterac.png", width = 600, height = 600)
vis.gam(qgam_model, view = c("Nebulosity", "Nebulosity_weighted"), theta = 30, ticktype = "detailed",
        color = "topo")
dev.off()

png("images/wind_iterac.png", width = 600, height = 600)
vis.gam(qgam_model, view = c("Wind", "Wind_weighted"), theta = 30, ticktype = "detailed", color = "gray")
dev.off()

png("images/stemp_iterac.png", width = 600, height = 600)
vis.gam(qgam_model, view = c("Temp_s99", "Temp_s95_min"), theta = 30, ticktype = "detailed", color = "terrain")
dev.off()

png("images/stemp_iterac.png", width = 600, height = 600)
vis.gam(qgam_model, view = c("Temp_s99", "Temp_s95"), theta = 30, ticktype = "detailed", color = "heat")
dev.off()

qgam_xgb_kf <- function(train, test, qgam_eq, quantile) {
  idx <- sample(nrow(train), .5 * nrow(train))
  train_qgam <- train[idx, ]
  train_xgb <- train[-idx, ]
  full_data <- rbind(data0, data1)
  qgam_model <- qgam(qgam_eq, data = train_qgam, qu = .8)

  qgam_pred <- predict(qgam_model, train_xgb)
  res <- train_xgb$Net_demand - qgam_pred

  dtrain <- xgb.DMatrix(data = as.matrix(train_xgb[, -c(1, 2, 5, 6, 7)]),
                        label = res)
  xgb_params <- list(objective = "reg:squarederror", eta = .1, max_depth = 3, alpha = 1, lambda = 0,
                     subsample = .7, colsample_bytree = 1, eval_metric = "rmse")
  xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 1000)

  pred1 <- predict(qgam_model, full_data)
  pred2 <- predict(xgb_model, as.matrix(full_data[, -c(1, 2, 5, 6, 7)]))
  pred1 + pred2
}

pred <- qgam_xgb_kf(data0, data1, qgam_eq, .8)
pinball_loss(data1$Net_demand, pred[-sel_a], quant = .8)
rmse(data1$Net_demand, pred[-sel_a])

q <- 0.2
myobjective <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  grad <- (labels < preds)-q
  hess <- rep(1, length(labels))
  return(list(grad = grad, hess = hess))
}

evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  u = (labels-preds)*(q-(labels<preds))
  err <- sum(u) / length(u)
  return(list(metric = "MyError", value = err))
}
