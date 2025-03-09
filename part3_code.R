# Clear environment.
rm(list = objects())

# Load libraries.
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

# Score functions.
source("score.R")

# Data.
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

# Identify data0 in the full dataset.
data0_idx <- seq_len(nrow(data0))


###############################################################################
# FIRST MODEL
###############################################################################

qgam_xgb_kf <- function(train, test, qgam_eq, quantile) {
  "
  This function trains a qgam model followed by an xgboost booster. The results
  are then smoothed using a kalman filter :

  - train : dataframe containing target and predictors for training ;
  - test : dataframe for testing, must contain the same columns as train ;
  - qgam_eq : formula used by the qgam model ;
  - quantile : aimed for by the qgam model.

  Returns a prediction vector made on the test predictors.
  "
  # Separate train sets for qgam and xgb.
  idx <- seq_len(.5 * nrow(train))
  train_qgam <- train[idx, ]
  train_xgb <- train[-idx, ]

  # Concatenate train and test sets for online learning.
  full_data <- rbind(train, test)

  # Train qgam model.
  qgam_model <- qgam(qgam_eq, data = train_qgam, qu = quantile)

  # BOOST QGAM WITH XGBOOST
  # Make a prediction with the qgam on train_xgb.
  qgam_train_pred <- predict(qgam_model, newdata = train_xgb)

  # Get the residuals on train_xgb.
  qgam_train_res <- train_xgb$Net_demand - qgam_train_pred

  # Train xgboost model on the residuals.
  dtrain <- xgb.DMatrix(data = as.matrix(train_xgb[, -c(1, 2, 5, 6, 7)]),
                        label = qgam_train_res)
  xgb_params <- list(objective = "reg:squarederror", eta = .1, max_depth = 3)
  xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 3000)

  # KALMAN FILTERING
  # Get effects from qgam model on full data.
  qgam_terms <- predict(qgam_model, newdata = full_data, type = "terms")

  # Get boosted prediction on full data.
  qgam_pred <- predict(qgam_model, newdata = full_data)
  xgb_pred <- predict(xgb_model,
                      newdata = as.matrix(full_data[, -c(1, 2, 5, 6, 7)]))
  boosted_pred <- qgam_pred + xgb_pred

  # Fit kalman filter with EM.
  input <- cbind(qgam_terms, boosted_pred) %>% scale %>% cbind(1)
  target <- full_data$Net_demand

  ssm <- statespace(input, target)
  ssm_em <- select_Kalman_variances(ssm, input[seq_len(nrow(train)), ],
                                    target[seq_len(nrow(train))],
                                    Q_init = diag(ncol(input)), method = "em",
                                    n_iter = 1000, verbose = 100)
  
  # Make prediction on fulldata.
  ssm_em_pred <- predict(ssm_em, input, target, type = "model",
                         compute_smooth = TRUE)$pred_mean

  train_res <- train$Net_demand - ssm_em_pred[seq_len(nrow_train)]
  q_norm <- qnorm(q, mean = mean(train_res), sd = sd(train_res))
  ssm_em_pred[-seq_len(nrow_train)] <- ssm_em_pred[-seq_len(nrow_train)] + q_norm

  # Return full prediction.
  ssm_em_pred
}

# Equation used for qgam model.
qgam_eq <- Net_demand ~ s(as.numeric(Date), k = 3, bs = "cr") +
  s(toy, k = 30, bs = "cc") + s(Temp, k = 10, bs = "cr") +
  s(Load.1, bs = "cr", by =  as.factor(WeekDays)) +
  s(Load.7, bs = "cr")  + as.factor(WeekDays) + as.factor(BH) +
  s(Solar_power.1, bs = "cr") + s(Net_demand.1, bs = "cr") +
  te(Wind, Wind_weighted) + te(Nebulosity, Nebulosity_weighted) +
  te(Temp_s99, Temp_s95_min) + te(Temp_s99, Temp_s95) + s(Temp_s95, bs = "cr")

# PARALLEL PROCESSING
# Setup cluster with 9 cores.
cluster <- makeCluster(9)
registerDoParallel(cluster)

# Export required variables to cluster.
clusterExport(cluster, varlist = c("qgam_xgb_kf", "data0", "data1", "qgam_eq"))

# Process for each quantile and bind predictions into a matrix.
quantiles <- seq(.1, .9, by = .1)
preds <- foreach(q = quantiles, .combine = cbind,
                 .packages = c("qgam", "xgboost", "viking", "tidyverse",
                               "magrittr")) %dopar% {
  qgam_xgb_kf(data0, data1, qgam_eq, q)
}
stopCluster(cluster)
preds <- as.data.frame(preds)

# Add target features to the predictions.
preds <- cbind(full_data$Net_demand, preds)
colnames(preds)[1] <- "Net_demand"

# ONLINE AGGREGATION
target <- full_data$Net_demand
experts <- preds[, -1]

# Apply artificial underestimation.
experts[, 8] <- experts[, 8] - 500

# Aggregate.
agg <- mixture(target, experts, model = "MLpol",
               loss.type = list(name = "pinball", tau = .8))
plot(agg)
final_pred <- agg$prediction[-data0_idx]


###############################################################################
# TUNING XGBOOST BOOSTER
###############################################################################
# Separate train sets for qgam and xgb.
idx <- sample(nrow(data0), .5 * nrow(data0))
train_qgam <- data0[idx, ]
train_xgb <- data0[-idx, ]

# Train qgam model.
qgam_model <- qgam(qgam_eq, data = train_qgam, qu = .8)

# Make a prediction with the qgam on train_xgb.
qgam_train_pred <- predict(qgam_model, train_xgb)

# Get the residuals on train_xgb.
qgam_train_res <- train_xgb$Net_demand - qgam_pred
dtrain <- xgb.DMatrix(data = as.matrix(train_xgb[, -c(1, 2, 5, 6, 7)]),
                      label = qgam_train_res)

# Define grid parameter for grid-search.
cv_params <- expand.grid(eta = c(.01, .1, .3),
                         alpha = seq(1, 10, by = 3),
                         lambda = seq(1, 10, by = 3),
                         subsample = seq(.1, 1, by = .3),
                         colsample_bytree = seq(.1, 1, by = .3))

# Grid-search using cross-validation.
best_score <- Inf
best_params <- NULL
for (i in seq_len(nrow(cv_params))) {
  # Get params.
  params <- list(eta = cv_params$eta[i],
                 alpha = cv_params$alpha[i],
                 lambda = cv_params$lambda[i],
                 subsample = cv_params$subsample[i],
                 colsample_bytree = cv_params$colsample_bytree[i])
  
  # Make cross-validation with these params.
  cv_results <- xgb.cv(params = params, data = dtrain, nrounds = 100,
                       nfold = 5, stratified = TRUE, early_stopping_rounds = 5)
  
  # Get evaluation.
  mean_rmse <- min(cv_results$evaluation_log$test_rmse_mean)
  
  # Update best parameters based on results.
  if (mean_rmse < best_score){
    best_score <- mean_rmse
    best_params <- params
  }
}
print(best_params)


###############################################################################
# UPDATED MODEL
###############################################################################
qgam_xgb_kf <- function(train, test, qgam_eq, quantile) {
  "
  Updated function of the previous model. Same arguments and output.
  "
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

# PARALLEL PROCESSING
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

# ONLINE AGGREGATION
target <- full_data$Net_demand
experts <- preds[, -1]
experts[, 8] <- experts[, 8] - 500
agg <- mixture(target, experts, model = "MLpol",
               loss.type = list(name = "pinball", tau = .8))
plot(agg)
final_pred <- agg$prediction[-data0_idx]


###############################################################################
# BAGGING
###############################################################################
# Create 10 bootstrapped sample indexes.
nb_boots <- 10
boot_samples <- replicate(nb_boots, sample(seq_len(nrow(data0)), replace = TRUE))

# Make predictions on each bootstrapped sample.
predictions <- apply(boot_samples, 2, function(idx) {qgam_xgb(data0[idx, ], data1, qgam_eq, .8)})

# ONLINE AGGREGATION
target <- full_data$Net_demand
predictions[, 3] <- predictions[, 8] - 500
agg <- mixture(target, predictions, model = "MLpol",
               loss.type = list(name = "pinball", tau = .8))
plot(agg)
final_pred <- agg$prediction[-data0_idx]


###############################################################################
###############################################################################
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


###############################################################################
# STUDY RESULTS
###############################################################################
# Example pre-made prediction.
# Use a prediction on data1 named "final_pred" otherwise.
final_pred <- read.csv("results/updated_model.csv")[, 2]

# Evaluation.
pinball_loss(data1$Net_demand, final_pred, quant = .8,
             output.vect = FALSE)
rmse(data1$Net_demand, final_pred)

# Graphical comparison.
plot(data1$Date, data1$Net_demand,
     type = "l",
     main = "Comparison of the predictions with the ground truth",
     xlab = "Time",
     ylab = "Electricity consumption")
lines(data1$Date, final_pred,
      col = "red")
legend("topright",
       legend = c("Prediction", "Ground truth"),
       col = c("red", "black"),
       lty = 1)

# Get residuals.
final_res <- data1$Net_demand - final_pred

# Plot residuals.
plot(data1$Date, final_res,
     type = "l",
     xlab = "Time",
     ylab = "Residuals")
abline(h = 0,
       col = "red",
       lwd = 2)

# Autocorrelation.
acf(final_res)

# RESIDUALS DISTRIBUTION STUDY
# Histogram.
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

# Q-Q plot.
qqnorm(final_res)
qqline(final_res, col = "red", lwd = 2)

# Normality test.
shapiro.test(final_res)
