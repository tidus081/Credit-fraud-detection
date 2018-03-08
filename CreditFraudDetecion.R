######################## Credit card fraud dectection ########

# Importing the dataset
dataset = read.csv('D://R/creditcard3.csv')
dataset = dataset[3:32]
head(dataset)
length(subset(dataset$Class, dataset$Class==1))/100000 
# the gap is extremly large

# Feature Scaling
dataset[-30] = scale(dataset[-30])

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Class, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

> head(dataset)
          V1          V2        V3         V4          V5          V6
1 -1.3598071 -0.07278117 2.5363467  1.3781552 -0.33832077  0.46238778
2  1.1918571  0.26615071 0.1664801  0.4481541  0.06001765 -0.08236081
3 -1.3583541 -1.34016307 1.7732093  0.3797796 -0.50319813  1.80049938
4 -0.9662717 -0.18522601 1.7929933 -0.8632913 -0.01030888  1.24720317
5 -1.1582331  0.87773675 1.5487178  0.4030339 -0.40719338  0.09592146
6 -0.4259659  0.96052304 1.1411093 -0.1682521  0.42098688 -0.02972755
           V7          V8         V9         V10        V11
1  0.23959855  0.09869790  0.3637870  0.09079417 -0.5515995
2 -0.07880298  0.08510165 -0.2554251 -0.16697441  1.6127267
3  0.79146096  0.24767579 -1.5146543  0.20764287  0.6245015
4  0.23760894  0.37743587 -1.3870241 -0.05495192 -0.2264873
5  0.59294075 -0.27053268  0.8177393  0.75307443 -0.8228429
6  0.47620095  0.26031433 -0.5686714 -0.37140720  1.3412620
          V12        V13        V14        V15        V16         V17
1 -0.61780086 -0.9913898 -0.3111694  1.4681770 -0.4704005  0.20797124
2  1.06523531  0.4890950 -0.1437723  0.6355581  0.4639170 -0.11480466
3  0.06608369  0.7172927 -0.1659459  2.3458649 -2.8900832  1.10996938
4  0.17822823  0.5077569 -0.2879237 -0.6314181 -1.0596472 -0.68409279
5  0.53819555  1.3458516 -1.1196698  0.1751211 -0.4514492 -0.23703324
6  0.35989384 -0.3580907 -0.1371337  0.5176168  0.4017259 -0.05813282
          V18         V19         V20          V21          V22
1  0.02579058  0.40399296  0.25141210 -0.018306778  0.277837576
2 -0.18336127 -0.14578304 -0.06908314 -0.225775248 -0.638671953
3 -0.12135931 -2.26185710  0.52497973  0.247998153  0.771679402
4  1.96577500 -1.23262197 -0.20803778 -0.108300452  0.005273597
5 -0.03819479  0.80348692  0.40854236 -0.009430697  0.798278495
6  0.06865315 -0.03319379  0.08496767 -0.208253515 -0.559824796
          V23         V24        V25        V26          V27
1 -0.11047391  0.06692807  0.1285394 -0.1891148  0.133558377
2  0.10128802 -0.33984648  0.1671704  0.1258945 -0.008983099
3  0.90941226 -0.68928096 -0.3276418 -0.1390966 -0.055352794
4 -0.19032052 -1.17557533  0.6473760 -0.2219288  0.062722849
5 -0.13745808  0.14126698 -0.2060096  0.5022922  0.219422230
6 -0.02639767 -0.37142658 -0.2327938  0.1059148  0.253844225
          V28 Amount Class
1 -0.02105305 149.62     0
2  0.01472417   2.69     0
3 -0.05975184 378.66     0
4  0.06145763 123.50     0
5  0.21515315  69.99     0
6  0.08108026   3.67     0
> length(subset(dataset$Class, dataset$Class==1)) # the gap is extremly large
[1] 223
> length(subset(dataset$Class, dataset$Class==1))/100000
[1] 0.00223


######################## 1 XGBoost(Extreme Gradient Boost) ########
# Fitting XGBoost to the Training set
# install.packages('xgboost')
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[-30]), label = training_set$Class, nrounds = 10)

[1]	train-rmse:0.350282 
[2]	train-rmse:0.245575 
[3]	train-rmse:0.172391 
[4]	train-rmse:0.121337 
[5]	train-rmse:0.085836 
[6]	train-rmse:0.061289 
[7]	train-rmse:0.044501 
[8]	train-rmse:0.033252 
[9]	train-rmse:0.025722 
[10]	train-rmse:0.021219 

# Predicting the Test set results
y_pred = predict(classifier, newdata = as.matrix(test_set[-30]))
y_pred = (y_pred >= 0.5)

# Making the Confusion Matrix
cm = table(test_set[, 30], y_pred)
accuracy_xgb = (cm[1,1]+cm[2,2]) /margin.table(cm)

# Applying k-Fold Cross Validation
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Class, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = xgboost(data = as.matrix(training_set[-30]), label = training_set$Class, nrounds = 10)
  y_pred = predict(classifier, newdata = as.matrix(test_fold[-30]))
  y_pred = (y_pred >= 0.5)
  cm = table(test_fold[, 30], y_pred)
  accuracy = (cm[1,1]+cm[2,2]) /margin.table(cm)
  return(accuracy)
})
accuracy_xgb_kval = mean(as.numeric(cv))
c(accuracy_xgb,accuracy_xgb_kval)

[1] 0.9993500 0.9996875

######################## 2 Support Vector Machine ##################

# Fitting Kernel SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Class ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-30])

# Making the Confusion Matrix
cm = table(test_set[, 30], y_pred)
cm
accuracy = (cm[1,1]+cm[2,2]) /margin.table(cm)
accuracy
# Applying k-Fold Cross Validation
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Class, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = Purchased ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
  y_pred = predict(classifier, newdata = test_fold[-3])
  cm = table(test_fold[, 3], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))
accuracy
# Applying Grid Search to find the best parameters
# install.packages('caret')
library(caret)
classifier = train(form = Class ~ ., data = training_set, method = 'svmRadial')
classifier
classifier$bestTune

######################## 3 Random Forest ##############################


######################## 4 multi Logistic Regression ##################


######################## 5 Naïve Bayes. ###############################
