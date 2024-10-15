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
impfile <- paste0("/Perm_", model_name, "imp_", fold, ".csv")
aucfile <- paste0("/Perm_", model_name, "auc_", fold, ".csv")
prefile <- paste0("/Perm_", model_name, "pre_", fold, ".csv")

# Download and load data
dat <- read.csv("Infection_clean.csv", head = TRUE)
dimsize <- ncol(dat)
y <- factor(dat[, 3], labels = c("0", "1"))
x <- as.matrix(dat[, 4:dimsize])
colnames(x)[20] <- "GENDER"

# Parameters
pvacut <- 0.05
fold_num <- 10
n_ensemble <- 100
new_perm <- 200
new_fold <- 10
n_tree <- 1000
node_size <- 3

# Evaluation functions
acc <- function(y, x, cut = 0.5) mean((y == levels(y)[2]) == (x > cut))
auc <- function(y, x) performance(prediction(x, (y == levels(y)[2]) * 1), "auc")@y.values[[1]]
prauc <- function(y, x) performance(prediction(x, (y == levels(y)[2]) * 1), "aucpr")@y.values[[1]]

# Pathway list and continuous features
pathwaylist <- list(INSURANCE=18:19, ETHNICITY=21:24, MARITAL_STATUS=25:28, ADMISSION_TYPE=29:30, ADMISSION_LOCATION=31:34, Surgery=35:40)
conlist <- c(1:17, 20, 41:dim(x)[2])

# Shuffle and split data for cross-validation
set.seed(20220301)
shuffle <- sample(length(y))
validate <- if (fold < fold_num) {
  shuffle[((fold - 1) * length(y) / fold_num + 1):(fold * length(y) / fold_num)]
} else {
  shuffle[((fold - 1) * length(y) / fold_num + 1):length(y)]
}
trainx <- x[-validate, ]
trainy <- y[-validate]

# PermFIT-DNN function
run_permfit_dnn <- function(trainx, trainy, validate) {
  dnn_obj <- importDnnet(x = trainx, y = trainy)
  npermfit_dnn <- permfit(train = dnn_obj, k_fold = new_fold, n_perm = new_perm, pathway_list = pathwaylist, method = "ensemble_dnnet", shuffle = sample(length(trainy)), n.ensemble = n_ensemble)
  ndnn_feature <- which(npermfit_dnn@importance$importance_pval <= pvacut)
  ndnn_feature <- intersect(ndnn_feature, conlist)
  dnn_cat <- which(npermfit_dnn@block_importance$importance_pval <= pvacut)
  for (i in dnn_cat) {
    ndnn_feature <- c(ndnn_feature, pathwaylist[[dnn_cat[i]]])
  }
  dnn_mod <- ensemble_dnnet(importDnnet(x = as.matrix(x[-validate, ndnn_feature]), y = y[-validate]), n_ensemble)
  pred <- predict(dnn_mod, as.matrix(x[validate, ndnn_feature]))[, "1"]
  list(pred = pred, importance = npermfit_dnn)
}

# PermFIT-RF function
run_permfit_rf <- function(trainx, trainy, validate) {
  dnn_obj <- importDnnet(x = trainx, y = trainy)
  npermfit_rf <- permfit(train = dnn_obj, k_fold = new_fold, n_perm = new_perm, pathway_list = pathwaylist, method = "random_forest", shuffle = sample(length(trainy)), n.ensemble = n_ensemble, ntree = n_tree, nodesize = node_size)
  nrf_feature <- which(npermfit_rf@importance$importance_pval <= pvacut)
  nrf_feature <- intersect(nrf_feature, conlist)
  rf_cat <- which(npermfit_rf@block_importance$importance_pval <= pvacut)
  for (i in rf_cat) {
    nrf_feature <- c(nrf_feature, pathwaylist[[rf_cat[i]]])
  }
  rf_mod <- randomForest(as.matrix(x[-validate, nrf_feature]), y[-validate], ntree = n_tree, mtry = length(nrf_feature), nodesize = node_size)
  pred <- predict(rf_mod, as.matrix(x[validate, nrf_feature]), type = "prob")[, 2]
  list(pred = pred, importance = npermfit_rf)
}

# PermFIT-SVM function
run_permfit_svm <- function(trainx, trainy, validate) {
  dnn_obj <- importDnnet(x = trainx, y = trainy)
  npermfit_svm <- permfit(train = dnn_obj, k_fold = new_fold, n_perm = new_perm, pathway_list = pathwaylist, method = "svm", shuffle = sample(length(trainy)), n.ensemble = n_ensemble)
  nsvm_feature <- which(npermfit_svm@importance$importance_pval <= pvacut)
  nsvm_feature <- intersect(nsvm_feature, conlist)
  svm_cat <- which(npermfit_svm@block_importance$importance_pval <= pvacut)
  for (i in svm_cat) {
    nsvm_feature <- c(nsvm_feature, pathwaylist[[svm_cat[i]]])
  }
  svm_mod <- tune.svm(as.matrix(x[-validate, nsvm_feature]), y[-validate], gamma = 10**(-(0:4)), cost = 10**(0:4/2), tunecontrol = tune.control(cross = fold_num))
  svm_mod <- svm(as.matrix(x[-validate, nsvm_feature]), y[-validate], gamma = svm_mod$best.parameters$gamma, cost = svm_mod$best.parameters$cost, probability = TRUE)
  pred <- attr(predict(svm_mod, as.matrix(x[validate, nsvm_feature]), decision.values = TRUE, probability = TRUE), "probabilities")[, "1"]
  list(pred = pred, importance = npermfit_svm)
}

# PermFIT-XGB function
run_permfit_xgb <- function(trainx, trainy, validate) {
  dnn_obj <- importDnnet(x = trainx, y = trainy)
  parms <- list(booster = "gbtree", objective = "binary:logistic", eta = 0.3, gamma = 0, max_depth = 5, min_child_weight = 1, subsample = 1, colsample_bytree = 1)
  npermfit_xgb <- permfit(train = dnn_obj, k_fold = new_fold, n_perm = new_perm, pathway_list = pathwaylist, method = "xgboost", shuffle = sample(length(trainy)), params = parms)
  nxgb_feature <- which(npermfit_xgb@importance$importance_pval <= pvacut)
  nxgb_feature <- intersect(nxgb_feature, conlist)
  xgb_cat <- which(npermfit_xgb@block_importance$importance_pval <= pvacut)
  for (i in xgb_cat) {
    nxgb_feature <- c(nxgb_feature, pathwaylist[[xgb_cat[i]]])
  }
  xgbtrain <- xgb.DMatrix(data = x[-validate, nxgb_feature], label = as.numeric(as.character(y[-validate])) - 1)
  xgbtest <- xgb.DMatrix(data = x[validate, nxgb_feature], label = as.numeric(as.character(y[validate])) - 1)
  xgb_mod <- xgb.train(params = parms, data = xgbtrain, nrounds = 5, watchlist = list(train = xgbtrain), print_every_n = NULL, maximize = FALSE, eval_metric = "error")
  pred <- predict(xgb_mod, xgbtest)
  list(pred = pred, importance = npermfit_xgb)
}

# Run selected model
results <- switch(model_name,
                  "PermFIT-DNN" = run_permfit_dnn(trainx, trainy, validate),
                  "PermFIT-RF" = run_permfit_rf(trainx, trainy, validate),
                  "PermFIT-SVM" = run_permfit_svm(trainx, trainy, validate),
                  "PermFIT-XGB" = run_permfit_xgb(trainx, trainy, validate),
                  stop("Invalid model name"))

# Store predictions and metrics
dfimp <- data.frame(
  var_name = c(colnames(x)[conlist], names(pathwaylist)),
  paste0(model_name, "-IMP") = round(c(results$importance@importance$importance[conlist], results$importance@block_importance$importance), 5),
  paste0(model_name, "-PVL") = round(c(results$importance@importance$importance_pval[conlist], results$importance@block_importance$importance_pval), 5)
)
write.table(dfimp, file = impfile, append = FALSE, row.names = FALSE, col.names = TRUE, sep = ",")

dfpre <- data.frame('PosCT' = y[validate], model_name = results$pred)
write.table(dfpre, file = prefile, append = FALSE, row.names = FALSE, col.names = TRUE, sep = ",")

dfauc <- data.frame(
  Method = model_name,
  Accuracy = round(acc(y = y[validate], x = results$pred), 3),
  AUC = round(auc(y = y[validate], x = results$pred), 3),
  PRAUC = round(prauc(y = y[validate], x = results$pred), 3)
)
write.table(dfauc, file = aucfile, append = FALSE, row.names = FALSE, col.names = TRUE, sep = ",")

print(paste(model_name, "modeling complete"))
