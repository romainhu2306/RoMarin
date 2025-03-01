###############################################################################
###############################################################################
qgam_boost <- function(train, test, quantile, qgam_eq){
  # Separate train sets.
  idx = sample(nrow(train), .5 * nrow(train))
  train_qgam = train[idx, ]
  train_xgb = train[-idx, ]

  # qgam model.
  qgam_model <- qgam(qgam_eq, data = train_qgam, qu = quantile)

  # Boosting with xgboost.
  qgam_train_pred <- predict(qgam_model, newdata = train_xgb)
  qgam_train_res <- train_xgb$Net_demand - qgam_train_pred
  dtrain <- xgb.DMatrix(data = as.matrix(train_xgb[, -1]),
                        label = qgam_train_res)
  xgb_params <- list(objective = "reg:squarederror", eta = .1, max_depth = 3)
  xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100)

  # Boosted prediction.
  data <- rbind(train, test)
  qgam_pred <- predict(qgam_model, newdata = data, type = "terms")
  xgb_pred <- predict(xgb_model, newdata = as.matrix(data[, -1]),
                      type = "terms")
  final_pred <- qgam_pred + xgb_pred
}

###############################################################################
###############################################################################
aggreg_qgam <- function(test, aggregation){
  aggregator <- lm(Net_demand ~ ., data = aggregation)
  aggreg_pred <- predict(aggregator, newdata = test)
}

###############################################################################
###############################################################################
kalman_filter <- function(input, target){
  # Scaling inputs.
  for (j in seq_len(ncol(input))){
    input[, j] <- (input[, j] - mean(input[, j])) / sd(input[, j])
  }
  input <- cbind(input, 1)
  input_dim = ncol(input)
  
  # Online learning.
  ssm <- statespace(input, target)
  ssm_em <- select_Kalman_variances(ssm, input, target, method = "em",
                                    n_iter = 1000, Q_init = diag(input_dim),
                                    verbose = 10)
  kalman_pred <- predict(ssm, input, target)
  kalman_pred <- kalman_pred %>% tail(length(test))
}