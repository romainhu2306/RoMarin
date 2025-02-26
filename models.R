rm(list = objects())

library(mgcv)
library(yarrr)
library(magrittr)
library(forecast)
library(tidyverse)
library(ranger)
library(qgam)
library(xgboost)
library(data.table)
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

qgam80 <- qgam(qgam_eq, data = train, qu = .8)
qgam80_pred <- predict(qgam80)
qgam80_res <- train$Net_demand - qgam80_pred

xgb_train <- as.matrix(train[, -1])
xgb_test <- as.matrix(test[, -1])
dtrain <- xgb.DMatrix(data = xgb_train, label = qgam80_res)
xgb_params <- list(objective = "reg:squarederror", eta = .1, max_depth = 3)
xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100)
xgb_pred <- predict(xgb_model, newdata = xgb_test)

final_pred <- predict(qgam80, newdata = test) + xgb_pred

pinball_loss(data0$Net_demand[sel_b], final_pred, quant = .8,
             output.vect = FALSE)
