suppressWarnings(library(xgboost))
suppressWarnings(library(randomForest))

train <- read.csv("train_sample.csv", header = T)
test <- read.csv("test_sample.csv", header = T)

predictors <- paste0("X", 1:(ncol(train) - 1))
target <- train$class
params = list("objective" = "binary:logistic")

set.seed(1)
cv <- xgb.cv(data = data.matrix(train[predictors]), 
             params = params, label = target, 
             nfold = 5, nrounds = 50, prediction = T, 
             verbose = F)
(cv.results <- cv$evaluation_log)
(bestNR <- which.min(cv.results$test_logloss_mean)) # 10

set.seed(1)
xgb.model <- xgboost(data = data.matrix(train[predictors]), 
                   params = params, label = target,
                   nrounds = bestNR, verbose = F, 
                   save_period=NULL)
xgb.preds <- round(predict(xgb.model, 
                           newdata=data.matrix(test[predictors])))

set.seed(1)
rf.model <- randomForest(x = data.matrix(train[predictors]), 
                         y = as.factor(train$class), ntree=500, 
                         importance=TRUE)
varImpPlot(rf.model, main="Variable Importance")
most <- as.integer(which.max(importance(rf.model)[,1])) #2
pred <- data.matrix(cbind(test$id, xgb.preds))
saveRDS(list(RFMostImportant = most, Forecast = pred), "W4answer.rds")
