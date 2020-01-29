
### +++++ TITLE: Machine Learning Pipeline ====


### +++++ GLOBAL SETUP ====



## Set up the working directory to home directory
setwd("~/")

## create folders
dir.create("~/MohamadAznor_BinMohamedYusof/mlp/1.ClusteringAnalysis")
dir.create("~/MohamadAznor_BinMohamedYusof/mlp/2.ExploratoryDataAnalysis")
dir.create("~/MohamadAznor_BinMohamedYusof/mlp/3.ModelPerf1_CreateEnsembles")
dir.create("~/MohamadAznor_BinMohamedYusof/mlp/4.ModelPerf2_EvaluateOnTest")


### +++++ SETUP ====



### Useful functions ====

## ipak function: install multiple R packages, if not already installed
ipak <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

## names of missing columns
nacols <- function(df) {
  colnames(df)[unlist(lapply(df, function(x) anyNA(x)))]
}



### Install and load r packages ====

## specify packages
packages <- c("tidyverse", "data.table", "lubridate", "skimr", "psych", "factoextra", "cluster",
              "ggplot2", "shiny", "plotly", "colourpicker", "caret", "caretEnsemble", "kernlab", "glmnet", "broom", "gvlma", "car", 
              "DataExplorer", "RANN", "doParallel")

# install packages
ipak(packages)

# load packages
lapply(packages, library, character.only = TRUE)
rm(packages)



### Read and clean data ====

## read in data from source
RawData <- fread("https://aisgaiap.blob.core.windows.net/aiap4-assessment/real_estate.csv", 
                 colClasses = "character")

Final_Data <- RawData[,-1]


## clean column names
OldNames <- names(RawData[,-1])
NewNames <- c("X1.Transaction_Dt", "X2.Age", "X3.MRT_Dist", "X4.Stores_No", "X5.Latitude", "X6.Longitude", "Y.Hse_Price")
setnames(Final_Data, OldNames, NewNames)
rm(OldNames)
rm(NewNames)

### convert to numeric
Final_Data <- Final_Data %>%
  mutate_at(vars(1:length(Final_Data)), as.numeric) %>%
  as.data.table()


### Apply clustering on the location data based on EDA ====
pdf("~/MohamadAznor_BinMohamedYusof/mlp/1.ClusteringAnalysis/1.Elbow_Silhoutte_Cluster_Plots.pdf")

# get the elbow plot
(elbow_method <- Final_Data %>%
  select(X4.Stores_No, X5.Latitude, X6.Longitude) %>%
  scale() %>%
  fviz_nbclust(kmeans, "wss")
 )

# get the silhoutte plot
(silhoutte_method <- Final_Data %>%
    select(X4.Stores_No, X5.Latitude, X6.Longitude) %>%
    scale() %>%
    fviz_nbclust(kmeans, "silhouette")
)


# carry out k-means clustering
set.seed(99)
num_clusters <- 7
k_means <- Final_Data %>%
  select(X4.Stores_No, X5.Latitude, X6.Longitude) %>%
  scale() %>%
  kmeans(centers = num_clusters, iter.max=50, nstart=25)

# append clustering data
Final_Data$cluster <- factor(k_means$cluster)


# geo plot for 7 clusters
ggplot(Final_Data, aes(x=X5.Latitude, y=X6.Longitude, colour = factor(cluster))) +
  geom_point(alpha=0.5, size = 5) +
  xlab("Latitude")+
  ylab("Longitude")+ 
  ggtitle("Clusters over geo location")

dev.off()


# select features for training
features <- c("X1.Transaction_Dt", "X2.Age", "X3.MRT_Dist",
              "X4.Stores_No", "Y.Hse_Price", "cluster")
Final_Data <- Final_Data %>% select(features)
rm(features)

### One-hot encoding and missing values ====

## generate one-hot encoding for clusters

# create dummy model
dummies_model <- dummyVars(~ ., data = Final_Data)
Final_Data_DumVar <- predict(dummies_model, newdata = Final_Data) %>%
  # convert to dataframe
  data.table()
str(Final_Data_DumVar)


## check for missing data

# plot missing data
pdf("~/MohamadAznor_BinMohamedYusof/mlp/2.ExploratoryDataAnalysis/1.Plot_Missing.pdf")
plot_missing(Final_Data_DumVar)
dev.off()

# if data is missing, replace with preProcess.
if (anyNA(Final_Data_DumVar)) {
  # model features 
  preProcess_missingdata_model <- preProcess(Final_Data_DumVar, method='knnImpute')
  Final_Data_DumVar_Imputed <- predict(preProcess_missingdata_model, newdata = Final_Data_DumVar)
  
  # identify columns with missing values
  Final_Data_DumVar_nacols <- nacols(Final_Data_DumVar)
  
  # replace columns with missing values 
  Final_Data_DumVar <- Final_Data_DumVar %>% select(-Final_Data_DumVar_nacols)
  Final_Data_DumVar <- Final_Data_DumVar_Imputed %>% select(Final_Data_DumVar_nacols) %>% bind_cols(Final_Data_DumVar)
  rm(Final_Data_DumVar_nacols)
  rm(Final_Data_DumVar_Imputed)
  
} else {
    paste("No missing values")
  }



### Exploring the data ====

## observe correlations
pdf("~/MohamadAznor_BinMohamedYusof/mlp/2.ExploratoryDataAnalysis/2.Feature_Correlations.pdf")
Final_Data_DumVar %>%
  select(1:5) %>%
  pairs.panels()

dev.off()


## apply multiple linear regression 1: all applicable variables
set.seed(99)
model_lm1 <- Final_Data_DumVar %>%
  # remove cluster 1 to prevent dummy variable trap
  select(-cluster.1) %>%
  lm(formula = Y.Hse_Price ~ .)

# examine the model's summary statistics
summary(model_lm1)
vif(model_lm1)
## Result: high multi-collinearity for multiple variables


## apply multiple linear regression 2: drop correlated variables
set.seed(99)
model_lm2 <- Final_Data_DumVar %>%
  # drop high vif variables
  select(-cluster.2, -cluster.4, -cluster.6, -cluster.1) %>%
  lm(formula = Y.Hse_Price ~ .)

# examine the model's summary statistics
(summary_model_lm2 <- summary(model_lm2))
(vif_model_lm2 <- vif(model_lm2))
## Results: good model fit, and low VIF
# Clusters 3 and 5 shows high negative correlation transaction date and number of stores shows high positive correlation with price

# export model coefficients
model_lm2 %>% 
  tidy() %>%
  arrange(-abs(estimate)) %>%
  write.csv("~/MohamadAznor_BinMohamedYusof/mlp/2.ExploratoryDataAnalysis/3.MultipleLinearRegression_Coef.csv")

# export vif values
write.csv(vif_model_lm2, "~/MohamadAznor_BinMohamedYusof/mlp/2.ExploratoryDataAnalysis/4.MultipleLinearRegression_VIF.csv")

## RMSE of multiple linear model

# get predictions
pred_model_lm2 <- predict(model_lm2, newdata = Final_Data_DumVar[, .(X1.Transaction_Dt, X2.Age, X3.MRT_Dist,
                                                   X4.Stores_No, cluster.3, cluster.5, cluster.7)])
# get RMSE of model
RMSE(pred_model_lm2, Final_Data_DumVar[, Y.Hse_Price]) %>%
  write.csv("~/MohamadAznor_BinMohamedYusof/mlp/2.ExploratoryDataAnalysis/5.MultipleLinearRegression_RMSE.csv")

# correlation between predictions and real values
cor(pred_model_lm2, unlist(Final_Data_DumVar[, Y.Hse_Price])) %>%
  write.csv("~/MohamadAznor_BinMohamedYusof/mlp/2.ExploratoryDataAnalysis/6.MultipleLinearRegression_Corr.csv")



### Data preparation and pre-processing ====

## split the data into training and validation 
set.seed(99)

# get rows for the training data
trainRowNumbers <- createDataPartition(Final_Data_DumVar$Y.Hse_Price , p = 0.7, list = FALSE)

# create the training dataset
train_data <- Final_Data_DumVar[trainRowNumbers, ]

# create the test dataset
test_data <- Final_Data_DumVar[-trainRowNumbers, ]

rm(trainRowNumbers)


## split the data into four partitions:
# train_data_y
# train_data_x
# test_data_y
# test_data_x

(train_data_y <- train_data %>% select(Y.Hse_Price) %>% unlist())
(train_data_x <- train_data %>% select(-Y.Hse_Price))
(test_data_y <- test_data %>% select(Y.Hse_Price) %>% unlist())
(test_data_x <- test_data %>% select(-Y.Hse_Price))



### +++++TRAIN THE MODELS ====
## Models to train:
# linear model
# support vector machines
# random forrest
# gradient boosting, tree-based models
# gradient boosing, linear-based models


### Setup hyperparameters ====
registerDoParallel(4)
getDoParWorkers()

set.seed(99)

# select cross validation method, with number of folds
k_folds <- 5
my_control <- trainControl(method = "cv",
                           number = k_folds,
                           savePredictions = "final",
                           allowParallel = TRUE)



### Train models ====

set.seed(99)
model_list <- caretList(train_data_x,
                        train_data_y,
                        trControl = my_control,
                        methodList = c("lm", "svmRadial", "rf", 
                                        "xgbTree", "xgbLinear"),
                        tuneList = NULL,
                        continue_on_fail = FALSE, 
                        preProcess = c("center", "scale"))


### Model performance 1: Ensembling models ====

options(digits = 3)

## compare the RMSE of models 
(model_results <- data.frame(
  LM = min(model_list$lm$results$RMSE),
  SVM = min(model_list$svmRadial$results$RMSE),
  RF = min(model_list$rf$results$RMSE),
  XGBT = min(model_list$xgbTree$results$RMSE),
  XGBL = min(model_list$xgbLinear$results$RMSE)
))

model_results %>% 
  write.csv("~/MohamadAznor_BinMohamedYusof/mlp/3.ModelPerf1_CreateEnsembles/1.Models_RMSE.csv")


## resample models and plot results
resamples <- resamples(model_list)
pdf("~/MohamadAznor_BinMohamedYusof/mlp/3.ModelPerf1_CreateEnsembles/2.Models_RMSE_Resamples.pdf")
dotplot(resamples, metric = "RMSE")
dev.off()


# check correlation of models
modelCor(resamples) %>%
  write.csv("~/MohamadAznor_BinMohamedYusof/mlp/3.ModelPerf1_CreateEnsembles/3.Models_Corr.csv")

# build ensemble of models 1
set.seed(99)
ensemble_1 <- caretEnsemble(model_list, 
                            metric = "RMSE", 
                            trControl = my_control)
summary(ensemble_1)
pdf("~/MohamadAznor_BinMohamedYusof/mlp/3.ModelPerf1_CreateEnsembles/4.Models_Ensemble1_RMSE.pdf")
plot(ensemble_1)
dev.off()



# stack models
set.seed(99)
ensemble_2 <- caretStack(model_list, 
                         method = "glmnet", 
                         metric = "RMSE", 
                         trControl = my_control)
print(ensemble_2) %>%
  capture.output(file = "~/MohamadAznor_BinMohamedYusof/mlp/3.ModelPerf1_CreateEnsembles/5.Models_Ensemble2_RMSE.csv")


### Model performance 2: RMSE of predictions ====

# Predictions
pred_lm <- predict.train(model_list$lm, newdata = test_data_x)
pred_svm <- predict.train(model_list$svmRadial, newdata = test_data_x)
pred_rf <- predict.train(model_list$rf, newdata = test_data_x)
pred_xgbT <- predict.train(model_list$xgbTree, newdata = test_data_x)
pred_xgbL <- predict.train(model_list$xgbLinear, newdata = test_data_x)
predict_ens1 <- predict(ensemble_1, newdata = test_data_x)
predict_ens2 <- predict(ensemble_2, newdata = test_data_x)

# RMSE
pred_RMSE <- data.frame(ensemble_1 = RMSE(predict_ens1, test_data_y),
                        ensemble_2 = RMSE(predict_ens2, test_data_y),
                        LM = RMSE(pred_lm, test_data_y),
                        SVM = RMSE(pred_svm, test_data_y),
                        RF = RMSE(pred_rf, test_data_y),
                        XGBT = RMSE(pred_xgbT, test_data_y),
                        XGBL = RMSE(pred_xgbL, test_data_y))
pred_RMSE %>% 
  capture.output(file = "~/MohamadAznor_BinMohamedYusof/mlp/4.ModelPerf2_EvaluateOnTest/1.ModelPredictions_RMSE.csv")
  

# Correlation between prediction and test data
pred_cor <- data.frame(ensemble_1 = cor(predict_ens1, test_data_y),
                       ensemble_2 = cor(predict_ens2, test_data_y),
                       LM = cor(pred_lm, test_data_y),
                       SVM = cor(pred_svm, test_data_y),
                       RF = cor(pred_rf, test_data_y),
                       XGBT = cor(pred_xgbT, test_data_y),
                       XGBL = cor(pred_xgbL, test_data_y))
pred_cor %>%
  capture.output(file = "~/MohamadAznor_BinMohamedYusof/mlp/4.ModelPerf2_EvaluateOnTest/2.Model_Corr_Pred_vs_Y.csv")


### +++++ End of Task 2 ====

