'''

# CODE in APPENDIX of Project for DS 5220
# -*- coding: utf-8 -*-
# @Date    : 2018-12-07 20:23:49
# @Author  : Amano
# @Version : $final$

'''

####################################### Environment Setting #####################################
options(warn=-1)
options(message=-1)

## Data without Smote and without one-hot encoder  FOR TREE-BASED MODEL
setwd("C:/R/SML")
adult <- read.csv("adult_processed.csv")
any(is.na(adult))
adult <- adult[,-c(1)]
adult$BinaryIncome <- factor(adult$income)

## Prepare data
set.seed(99)
rnum <- sample(x=1:dim(adult)[1], size=19561)
Train <- adult[rnum,]
Valid <- adult[-rnum,]
rownames(Valid) <- 1:nrow(Valid)
rnum <- sample(x=1:dim(Valid)[1], size=9750)
Test <- Valid[-rnum,]
Valid <- Valid[rnum,]

######################################## Classification Tree #####################################
require(rpart)
require(rpart.plot)
require(dplyr)

# Grid Search for Best Pruned Tree
best_cp <- 0
best_eval <- 0
seq <- seq(0, 1, by = 0.01)
for (c in seq) {
  model.rpart <- rpart(BinaryIncome ~ ., method="class", data=Train[,-3], control=rpart.control(cp=c)) 
  rpart.pred <- predict(model.rpart, Valid[,-3], type="class")
  if (best_eval < sum(Valid$BinaryIncome == rpart.pred)){
    best_eval <- sum(Valid$BinaryIncome == rpart.pred)
    best_cp <- c
    best.rpart.model <- model.rpart
  }
  print(sum(rpart.pred == Valid$BinaryIncome))
}
paste("best CP:", best_cp)
# 0.01
rpart.pred <- predict(best.rpart.model, Valid[,-3], type="class")
table(pred = rpart.pred, true = Valid$BinaryIncome)
#       true
#pred    0    1
#0      6987 1151
#1       355 1257
pred <- sum(Valid$BinaryIncome == rpart.pred) / length(Valid$BinaryIncome)
paste('Accuracy', 100 * pred, "%")
# 84.55
# TP 52.2
prp(best.rpart.model, type=1, extra=1, under=TRUE, split.font=1, varlen=-10)
printcp(best.rpart.model)

######################################## Random Forest ######################################
require(randomForest)
# Random Forest
RFtrees <- randomForest(BinaryIncome ~ ., data = Train[,-3], importance = TRUE)
RFtrees
# OOB estimate of  error rate: 14.91% -> 85.09%

# Bagging
RFtrees2 <- randomForest(BinaryIncome ~ ., data = Train[,-3], ntree = 500, mtry = 15, importance = TRUE)
RFtrees2

# Grid Search for Best Parameter m
best_mtry <- 0
best_eval <- 0
for (i in 1:15) {
  RFtrees <- randomForest(BinaryIncome ~ ., data = Train[,-3], ntree = 500, mtry = i, importance = TRUE)
  predValid <- predict(RFtrees, Valid[,-3], type = "class")
  if (best_eval < sum(Valid$BinaryIncome == predValid)){
    best_eval <- sum(Valid$BinaryIncome == predValid)
    best_mtry <- i
    best.RF.model <- RFtrees
  }
  print(sum(predValid == Valid$BinaryIncome))
}

# Variable Importance
par(mfrow=c(1,2))
varImpPlot(best.RF.model, type=1, pch=19, col=1, cex=.5, main="")
varImpPlot(best.RF.model, type=2, pch=19, col=1, cex=.5, main="")

paste("Best mtry: ", best_mtry)
# 3
paste('Best Tree Average Size', mean(treesize(best.RF.model)))
# 1266.612
RanTree.prediction <- predict(best.RF.model, Valid[,-3], type = "class")
table(pred = RanTree.prediction, true = Valid$BinaryIncome)
#      true
#pred    0    1
#0     6943  982
#1      399 1426
paste('Accuracy', 100 * sum(RanTree.prediction == Valid$BinaryIncome) / length(Valid$BinaryIncome), "%")
# 85.83
# TP 59.22

require(ggplot2)
qplot(LogEdNum, LogHr, colour = BinaryIncome, shape = RanTree.prediction, data=Valid[,-3])

######################################## Boosting Tree ######################################
require(gbm)
ntree = 3000
model.boost <- gbm(income~., data=Train[,-17], distribution="bernoulli", n.trees=ntree,
                   interaction.depth=1, shrinkage = 0.08) 

# Variable Importance
par(mfrow=c(1,1))
summary(model.boost)
#The measures are based on the number of times a variable is selected for splitting,
#weighted by the squared improvement to the model as a result of each split, and averaged over all trees

#LogScaledExtra_income LogScaledExtra_income 28.31817941
#relationship                   relationship 27.52186174
#LogEdNum                           LogEdNum 11.83441076
#LogAge                               LogAge  7.23602672
#LogHr                                 LogHr  6.86195137
#live_with_spouse           live_with_spouse  6.61455319
#LogFnlwgt                         LogFnlwgt  5.19111059
#education.levels           education.levels  2.65781805

plot(model.boost, i="LogScaledExtra_income", ylab="f(LogScaledExtra_income)",
     main="Relative Income vs. LogScaledExtra_income")
plot(model.boost, i="relationship", ylab="f(relationship)",
     main="Relative Income vs. Relationship")
plot(model.boost, i="LogEdNum", ylab="f(LogEdNum)",
     main="Relative Income vs. LogEdNum")
plot(model.boost, i="LogAge", ylab="f(LogAge)",
     main="Relative Income vs. LogAge")

Boost.pred <- predict(model.boost, Valid[,-17], n.trees=ntree)
Boost.pred <- ifelse(Boost.pred > 0.5,1,0)
misClasificError <- mean(Boost.pred != Valid$income)
table(pred = Boost.pred, true = Valid$BinaryIncome)
#      true
#pred    0    1
#0     7155 1251
#1      187 1157
print(paste('Accuracy', (1-misClasificError) * 100,'%'))
# 85.25
# TP 48.05

## Data without Smote and with one-hot encoder FOR NN
setwd("C:/R/SML")
adult2 <- read.csv("adult_processed_onehot.csv")
any(is.na(adult2))
adult2 <- adult2[,-c(1)]
adult2$BinaryIncome <- factor(adult2$income)

## Prepare data
set.seed(99)
rnum <- sample(x=1:dim(adult2)[1], size=19561)
Train2 <- adult2[rnum,]
Valid2 <- adult2[-rnum,]
rownames(Valid2) <- 1:nrow(Valid2)
rnum <- sample(x=1:dim(Valid2)[1], size=9750)
Test2 <- Valid2[-rnum,]
Valid2 <- Valid2[rnum,]
Train2 <- Train2[,-29]
Valid2 <- Valid2[,-29]
Test2 <- Test2[,-29]

######################################## Neural Network ######################################
require(ANN2)
model.ann <- neuralnetwork(Train2[,-29], Train2$BinaryIncome, hiddenLayers=c(20,20,20), batchSize=500,
                           learnRate=0.001, momentum=0.5, L2=0.0, maxEpochs=2000, standardize=FALSE)
plot(model.ann)

ann.pred <- predict(model.ann, Valid2[,-29])
table(pred = ann.pred$predictions, true = Valid2$BinaryIncome)
#      true
#pred    0    1
#0     6821  975
#1     521  1433
paste('Accuracy', 100 * sum(ann.pred$predictions==Valid2$BinaryIncome) / length(Valid2$BinaryIncome), "%")
# 84.66
# TP 59.51

# Compared with more units structure
model.ann1_3 <- neuralnetwork(Train2[,-29], Train2$BinaryIncome, hiddenLayers=c(30,40,30), batchSize=500,
                              learnRate=0.001, momentum=0.5, L2=0.0, maxEpochs=2000, standardize=FALSE)
plot(model.ann1_3)

ann.pred1_3 <- predict(model.ann1_3, Valid2[,-29])
table(pred = ann.pred1_3$predictions, true = Valid2$BinaryIncome)
#       true
#pred    0    1
#0     6466  710
#1      876 1698
paste('Accuracy', 100 * sum(ann.pred1_3$predictions==Valid2$BinaryIncome) / length(Valid2$BinaryIncome), "%")
# 83.73
# 70.51

###################################### Data with SMOTE ####################################### 
#Data with Smote and without one-hot encoder  FOR TREE-BASED MODEL
setwd("C:/R/SML")
adult3 <- read.csv("adult_processed_smote.csv")
any(is.na(adult3))
adult3 <- adult3[,-c(1)]
adult3$BinaryIncome <- factor(adult3$income)

## Prepare data
set.seed(99)
rnum <- sample(x=1:dim(adult3)[1], size=22505)
Train3 <- adult3[rnum,]
Valid3 <- adult3[-rnum,]
rownames(Valid3) <- 1:nrow(Valid3)
rnum <- sample(x=1:dim(Valid3)[1], size=11250)
Test3 <- Valid3[-rnum,]
Valid3 <- Valid3[rnum,]

######################################## Classification Tree #####################################
require(rpart)
require(rpart.plot)
require(dplyr)

# Grid Search for Best Subtree
best_cp <- 0
best_eval <- 0
seq <- seq(0, 1, by = 0.01)
for (c in seq) {
  model.rpart <- rpart(BinaryIncome ~ ., method="class", data=Train3[,-16], control=rpart.control(cp=c)) 
  rpart.pred <- predict(model.rpart, Valid3[,-16], type="class")
  if (best_eval < sum(Valid3$BinaryIncome == rpart.pred)){
    best_eval <- sum(Valid3$BinaryIncome == rpart.pred)
    best_cp <- c
    best.rpart.model2 <- model.rpart
  }
  print(sum(rpart.pred == Valid3$BinaryIncome))
}
paste("best CP:", best_cp)
# 0.00
rpart.pred2 <- predict(best.rpart.model2, Valid3[,-16], type="class")
table(pred = rpart.pred2, true = Valid3$BinaryIncome)
#      true
#pred    0    1
#0     6765  911
#1      672 2902
pred2 <- sum(Valid3$BinaryIncome == rpart.pred2) / length(Valid3$BinaryIncome)
paste('Accuracy', 100 * pred2, "%")
# 85.92
# TP 76.11
prp(best.rpart.model2, type=1, extra=1, under=TRUE, split.font=1, varlen=-10)
printcp(best.rpart.model2)

######################################## Random Forest ######################################
require(randomForest)

# Grid Search for Best Parameter m
best_mtry <- 0
best_eval <- 0
for (i in 1:15) {
  RFtrees <- randomForest(BinaryIncome ~ ., data = Train3[,-16], ntree = 500, mtry = i, importance = TRUE)
  predValid <- predict(RFtrees, Valid3[,-16], type = "class")
  if (best_eval < sum(Valid3$BinaryIncome == predValid)){
    best_eval <- sum(Valid3$BinaryIncome == predValid)
    best_mtry <- i
    best.RF.model2 <- RFtrees
  }
  print(sum(predValid == Valid3$BinaryIncome))
}

# Variable Importance
par(mfrow=c(1,2))
varImpPlot(best.RF.model2, type=1, pch=19, col=1, cex=.5, main="")
varImpPlot(best.RF.model2, type=2, pch=19, col=1, cex=.5, main="")

paste("Best mtry: ", best_mtry)
# 3
paste('Best Tree Average Size', mean(treesize(best.RF.model2)))
# 1470.128
RanTree.prediction2 <- predict(best.RF.model2, Valid3[,-16], type = "class")
table(pred = RanTree.prediction2, true = Valid3$BinaryIncome)
#      true
#pred    0    1
#0     6953  888
#1      484 2925
paste('Accuracy', 100 * sum(RanTree.prediction2 == Valid3$BinaryIncome) / length(Valid3$BinaryIncome), "%")
# 87.8
# TP 76.71

require(ggplot2)
qplot(LogEdNum, LogHr, colour = BinaryIncome, shape = RanTree.prediction2, data=Valid3[,-16])

######################################## Boosting Tree ######################################
require(gbm)
ntree = 3000
model.boost2 <- gbm(income~., data=Train3[,-17], distribution="bernoulli", n.trees=ntree,
                   interaction.depth=1, shrinkage = 0.08) 

# Variable Importance
par(mfrow=c(1,1))
summary(model.boost2)
#The measures are based on the number of times a variable is selected for splitting,
#weighted by the squared improvement to the model as a result of each split, and averaged over all trees

#live_with_spouse           live_with_spouse 35.53066145
#LogScaledExtra_income LogScaledExtra_income 14.34369993
#education.levels           education.levels 11.10206926
#LogEdNum                           LogEdNum 11.04077739
#LogHr                                 LogHr 10.13674303
#LogAge                               LogAge  7.99150108
#work_classes                   work_classes  3.58239192
#occupations                     occupations  2.58202026

plot(model.boost2, i="live_with_spouse", ylab="f(live_with_spouse)",
     main="Relative Income vs. Live_with_spouse")
plot(model.boost2, i="LogScaledExtra_income", ylab="f(LogScaledExtra_income)",
     main="Relative Income vs. LogScaledExtra_income")
plot(model.boost2, i="education.levels", ylab="f(education.levels)",
     main="Relative Income vs. Education.levels")
plot(model.boost2, i="LogEdNum", ylab="f(LogEdNum)",
     main="Relative Income vs. LogEdNum")
plot(model.boost2, i="LogHr", ylab="f(LogHr)",
     main="Relative Income vs. LogHr")

Boost.pred2 <- predict(model.boost2, Valid3[,-17], n.trees=ntree)
Boost.pred2 <- ifelse(Boost.pred2 > 0.5,1,0)
misClasificError <- mean(Boost.pred2 != Valid3$income)
table(pred = Boost.pred2, true = Valid3$BinaryIncome)
#       true
#pred    0    1
#0     7153 1120
#1     284  2693
print(paste('Accuracy', (1-misClasificError) * 100,'%'))
# 87.52
# TP 70.62

## Data without Smote and with one-hot encoder FOR NN
setwd("C:/R/SML")
adult4 <- read.csv("adult_processed_smote_onehot.csv")
any(is.na(adult4))
adult4 <- adult4[,-c(1)]
adult4$BinaryIncome <- factor(adult4[,29])

## Prepare data
set.seed(99)
rnum <- sample(x=1:dim(adult4)[1], size=22505)
Train4 <- adult4[rnum,]
Valid4 <- adult4[-rnum,]
rownames(Valid4) <- 1:nrow(Valid4)
rnum <- sample(x=1:dim(Valid4)[1], size=11250)
Test4 <- Valid4[-rnum,]
Valid4 <- Valid4[rnum,]
Train4 <- Train4[,-29]
Valid4 <- Valid4[,-29]
Test4 <- Test4[,-29]

######################################## Neural Network ######################################
require(ANN2)
model.ann2 <- neuralnetwork(Train4[,-29], Train4$BinaryIncome, hiddenLayers=c(20,20,20), batchSize=500,
                           learnRate=0.001, momentum=0.5, L2=0.0, maxEpochs=2000, standardize=FALSE)
plot(model.ann2)

ann.pred2 <- predict(model.ann2, Valid4[,-29])
table(pred = ann.pred2$predictions, true = Valid4$BinaryIncome)
#      true
#pred    0    1
#0     6784  906
#1      623 2937
paste('Accuracy', 100 * sum(ann.pred2$predictions==Valid4$BinaryIncome) / length(Valid4$BinaryIncome), "%")
# 86.41
# TP 76.42

# Compared with more units structure
model.ann2_3 <- neuralnetwork(Train4[,-29], Train4$BinaryIncome, hiddenLayers=c(30,40,30), batchSize=500,
                              learnRate=0.001, momentum=0.5, L2=0.0, maxEpochs=2000, standardize=FALSE)
plot(model.ann2_3)

ann.pred2_3 <- predict(model.ann2_3, Valid4[,-29])
table(pred = ann.pred2_3$predictions, true = Valid4$BinaryIncome)
#       true
#pred    0    1
#0     6466  710
#1      876 2698
paste('Accuracy', 100 * sum(ann.pred2_3$predictions==Valid4$BinaryIncome) / length(Valid4$BinaryIncome), "%")
# 85.99
# 77.21
