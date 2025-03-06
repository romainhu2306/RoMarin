# Clear environment.
rm(list = objects())

# Load required libraries and files.
library(tidyverse)
library(qgam)
library(xgboost)
library(viking)
library(foreach)
library(doParallel)
library(opera)
source("score.R")
data0 <- read_delim("data/Data0.csv", delim = ",")
data1 <- read_delim("data/Data1.csv", delim = ",")

# We remove the "Id" and "Usage" column in data1 that are not present in data0.
data1 <- data1 %>% select(-c("Id", "Usage"))

# We create the "Net_demand", "Load", "Solar_power" and "Wind_power" columns
# in data1 using their lagged variants : this is required for online learning.
data1 <- data1 %>% mutate(Net_demand = lead(Net_demand.1, n = 1))
data1$Net_demand[nrow(data1)] <- data1$Net_demand[nrow(data1) - 1]
data1 <- data1 %>% mutate(Load = lead(Load.1, n = 1))
data1$Load[nrow(data1)] <- data1$Load[nrow(data1) - 1]
data1 <- data1 %>% mutate(Solar_power = lead(Solar_power.1, n = 1))
data1$Solar_power[nrow(data1)] <- data1$Solar_power[nrow(data1) - 1]
data1 <- data1 %>% mutate(Wind_power = lead(Wind_power.1, n = 1))
data1$Wind_power[nrow(data1)] <- data1$Wind_power[nrow(data1) - 1]
data1 <- data1[, names(data0)]

# We create the full concatenated data for online learning.
full_data <- rbind(data0, data1)

# Marking index of the training data.
train_idx <- seq_len(nrow(data0))

# Equation for qgam models.
qgam_eq <- Net_demand ~ s(as.numeric(Date), k = 3, bs = "cr") +
  s(toy, k = 30, bs = "cc") + s(Temp, k = 10, bs = "cr") +
  s(Load.1, bs = "cr", by =  as.factor(WeekDays)) +
  s(Load.7, bs = "cr")  + as.factor(WeekDays) + as.factor(BH) +
  s(Solar_power.1, bs = "cr") + s(Net_demand.1, bs = "cr")

qgam_xgb <- function(train, test, qgam_eq, quantile) {
  idx <- sample(nrow(train), .5 * nrow(train))
  train_qgam <- train[idx, ]
  train_xgb <- train[-idx, ]
  qgam_model <- qgam(qgam_eq, data = train_qgam, qu = .8)

  qgam_pred <- predict(qgam_model, train_xgb)
  res <- train_xgb$Net_demand - qgam_pred

  dtrain <- xgb.DMatrix(data = as.matrix(train_xgb[, -c(1, 2, 5, 6, 7)]),
                        label = res)
  xgb_params <- list(objective = "reg:squarederror", eta = .1, max_depth = 3)
  xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 3000)

  pred1 <- predict(qgam_model, test)
  pred2 <- predict(xgb_model, as.matrix(test[, -c(1, 2, 5, 6, 7)]))
  pred1 + pred2
}


# Parallel processing.
cluster <- makeCluster(9)
registerDoParallel(cluster)

clusterExport(cluster, varlist = c("qgam_xgb", "data0", "data1", "qgam_eq"))
quantiles <- seq(.1, .9, by = .1)
preds <- foreach(q = quantiles, .combine = cbind,
                 .packages = c("qgam", "xgboost", "viking", "tidyverse",
                               "magrittr")) %dopar% {
  qgam_xgb(data0, data1, qgam_eq, q)
}

stopCluster(cluster)
preds <- as.data.frame(preds)
preds <- cbind(data1$Net_demand, preds)
colnames(preds)[1] <- "Net_demand"

target <- data1$Net_demand
experts[, 2] <- experts[, 2] + 20
agg <- mixture(target, experts, model = "MLpol",
               loss.type = list(name = "pinball", tau = .8))

final_pred <- agg$prediction

pinball_loss(data1$Net_demand, final_pred, quant = .8,
             output.vect = FALSE)

rmse(data1$Net_demand, final_pred)
plot(data1$Net_demand, type = "l")
lines(final_pred, col = "red")

final_res <- data1$Net_demand - final_pred
plot(final_res, type = "l")
abline(h = 0, col = "red")

acf(final_res)
hist(final_res, breaks = 25)
plot(agg)









qgam_xgb <- function(train, test, qgam_eq, quantile) {
  boot_idx <- sample(seq_len(nrow(train)), replace = TRUE)
  train <- train[boot_idx, ]
  idx <- sample(nrow(train), .5 * nrow(train))
  train_qgam <- train[idx, ]
  train_xgb <- train[-idx, ]
  qgam_model <- qgam(qgam_eq, data = train_qgam, qu = .8)

  qgam_pred <- predict(qgam_model, train_xgb)
  res <- train_xgb$Net_demand - qgam_pred

  dtrain <- xgb.DMatrix(data = as.matrix(train_xgb[, -c(1, 2, 5, 6, 7)]),
                        label = res)
  xgb_params <- list(objective = "reg:squarederror", eta = .1, max_depth = 3)
  xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 3000)

  pred1 <- predict(qgam_model, test)
  pred2 <- predict(xgb_model, as.matrix(test[, -c(1, 2, 5, 6, 7)]))
  pred1 + pred2
}


nb_boots <- 10
boot_samples <- replicate(nb_boots, sample(seq_len(nrow(data0)), replace = TRUE))
predictions <- apply(boot_samples, 2, function(x) {qgam_xgb(data0[x, ], data1, qgam_eq, .8)})

target <- data1$Net_demand
agg <- mixture(target, predictions, model = "MLpol",
               loss.type = list(name = "pinball", tau = .8))

final_pred <- agg$prediction

pinball_loss(data1$Net_demand, final_pred, quant = .8,
             output.vect = FALSE)
plot(agg)



cluster <- makeCluster(9)
registerDoParallel(cluster)

clusterExport(cluster, varlist = c("qgam_xgb", "data0", "data1", "qgam_eq"))
quantiles <- seq(.1, .9, by = .1)
preds <- foreach(q = quantiles, .combine = cbind,
                 .packages = c("qgam", "xgboost", "viking", "tidyverse",
                               "magrittr")) %dopar% {
  nb_boots <- 10
  boot_samples <- replicate(nb_boots, sample(seq_len(nrow(data0)), replace = TRUE))
  predictions <- apply(boot_samples, 2, function(x) {qgam_xgb(data0[x, ], data1, qgam_eq, .8)})
  target <- data1$Net_demand
  agg <- mixture(target, predictions, model = "MLpol",
               loss.type = list(name = "pinball", tau = .8))
  final_pred <- agg$prediction
}



target <- data1$Net_demand
experts <- preds[, -1]
experts[, 2] <- experts[, 2] + 20
agg <- mixture(target, experts, model = "MLpol",
               loss.type = list(name = "pinball", tau = .8))

final_pred <- agg$prediction

pinball_loss(data1$Net_demand, final_pred, quant = .8,
             output.vect = FALSE)

rmse(data1$Net_demand, final_pred)