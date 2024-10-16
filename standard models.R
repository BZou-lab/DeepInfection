# Clear workspace
rm(list = ls())

# Load required libraries
library(deepTL)
library(randomForest)
library(xgboost)
library(magrittr)
library(glmnet)
library(MASS)
library(e1071)
library(ROCR)

# Set arguments
args <- commandArgs(trailingOnly = TRUE)
fold <- as.integer(args[1])
model_name <- args[2] # Model selection argument

# File paths based on model
impfile <- paste0("/", model_name, "_imp_", fold, ".csv")
aucfile <- paste0("/", model_name, "_auc_", fold, ".csv")
prefile <- paste0("/", model_name, "_pre_", fold, ".csv")

# Load data
dat <- read.csv("Infection_clean.csv", head = TRUE)
dimsize <- ncol(dat)
y <- factor(dat[, 3], labels = c("0", "1"))
x <- as.matrix(dat[, 4:dimsize])

# Parameters
fold_num <- 10
testnum <- round(length(y) / fold_num)
n_tree <- 1000
node_size <- 3
n_epoch <- 1000
n_ensemble <- 100
esCtrl <- list(
  n.hidden = c(50, 40, 30, 20), 
  activate = "relu", 
  l1.reg = 10**-4, 
  early.stop.det = 1000, 
  n.batch = 30, 
  n.epoch = n_epoch, 
  learning.rate.adaptive = "adam", 
  plot = FALSE
)
xgb_params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eta = 0.3,
  gamma = 0,
  max_depth = 5,
  min_child_weight = 1,
  subsample = 1,
  colsample_bytree = 1
)

# Evaluation functions
acc <- function(y, x, cut = 0.5) mean((y == levels(y)[2]) == (x > cut))
auc <- function(y, x) performance(prediction(x, (y == levels(y)[2]) * 1), "auc")@y.values[[1]]
prauc <- function(y, x) performance(prediction(x, (y == levels(y)[2]) * 1), "aucpr")@y.values[[1]]

# Shuffle and split data for cross-validation
set.seed(20220301)
shuffle <- sample(length(y))
validate <- if (fold < fold_num) {
  shuffle[((fold - 1) * testnum + 1):(fold * testnum)]
} else {
  shuffle[((fold - 1) * testnum + 1):length(y)]
}
trainx <- x[-validate, ]
trainy <- y[-validate]

# Model functions
run_dnn <- function(trainx, trainy, validate) {
  dnn_obj <- importDnnet(x = trainx, y = trainy)
  dnn_mod <- ensemble_dnnet(dnn_obj, n_ensemble, esCtrl, verbose = 0)
  pred <- predict(dnn_mod, x[validate, ])[, "1"]
  return(pred)
}

run_xgb <- function(trainx, trainy, validate) {
  xgbtrain <- xgb.DMatrix(data = trainx, label = as.numeric(as.character(trainy)) - 1)
  xgbtest <- xgb.DMatrix(data = x[validate, ], label = as.numeric(as.character(y[validate])) - 1)
  xgb_mod <- xgb.train(params = xgb_params, data = xgbtrain, nrounds = 5, watchlist = list(train = xgbtrain), print_every_n = NULL, maximize = FALSE, eval_metric = "error")
  pred <- predict(xgb_mod, xgbtest)
  return(pred)
}

run_svm <- function(trainx, trainy, validate) {
  tune_result <- tune.svm(
    trainx, trainy, 
    gamma = 10 ** (-(0:4)), 
    cost = 10 ** (0:4 / 2), 
    tunecontrol = tune.control(cross = fold_num)
  )
  svm_mod <- svm(
    trainx, trainy, 
    gamma = tune_result$best.parameters$gamma, 
    cost = tune_result$best.parameters$cost, 
    probability = TRUE
  )
  pred <- attr(predict(svm_mod, x[validate, ], decision.values = TRUE, probability = TRUE), "probabilities")[, "1"]
  return(pred)
}

run_rf <- function(trainx, trainy, validate) {
  rf_mod <- randomForest(trainx, trainy, ntree = n_tree, nodesize = node_size, importance = TRUE)
  pred <- predict(rf_mod, x[validate, ], type = "prob")[, 2]
  return(pred)
}

# Run selected model
pred <- switch(model_name,
               "DNN" = run_dnn(trainx, trainy, validate),
               "XGB" = run_xgb(trainx, trainy, validate),
               "SVM" = run_svm(trainx, trainy, validate),
               "RF"  = run_rf(trainx, trainy, validate),
               stop("Invalid model name"))

# Store predictions and metrics
dfpre <- data.frame('PosCT' = y[validate], model_name = pred)
write.table(dfpre, file = prefile, append = FALSE, row.names = FALSE, col.names = TRUE, sep = ",")

dfauc <- data.frame(
  Method = model_name,
  Accuracy = round(acc(y = y[validate], x = pred), 3),
  AUC = round(auc(y = y[validate], x = pred), 3),
  PRAUC = round(prauc(y = y[validate], x = pred), 3)
)
write.table(dfauc, file = aucfile, append = FALSE, row.names = FALSE, col.names = TRUE, sep = ",")

print(paste(model_name, "modeling complete"))
