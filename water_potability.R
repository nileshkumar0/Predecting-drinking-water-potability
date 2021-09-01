##ABOUT THE DATASET###
#Title of the Project: Predicting water quality 
#Data Description: In this project, we will be predicting if the water quality is potable or not. This is data set is taken from Kaggle.  
#This dataset consists of 3276 rows and 10 columns (1 dependent variable and 9 independent variables). 
#Dependent Variable: 
#1.	Potability: Indicates if water is safe for human consumption where 1 means Potable and 0 means Not potable. 
#Independent Variables: 
#1.	pH value: PH is an important parameter in evaluating the acid-base balance of water. According to WHO ph value needs to be between 6.5 to 8.5.  ph values are numeric and ranges from 0 to 14. 
#2.	Hardness: Hardness was originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium. Hardness values are numeric and range from 47.4 to 323. 
#3.	Solids (Total dissolved solids - TDS): The water with a high TDS value indicates that water is highly mineralized. The desirable limit for TDS is 500 mg/l and the maximum limit is 1000 mg/l which is prescribed for drinking purposes. Values are numeric and ranges from 321 to more than 60,000 
#4.	Chloramines: Chlorine and chloramine are the major disinfectants used in public water systems.  Values are numeric and range from .35 to 13.1 
#5.	Sulfate: Sulfates are naturally occurring substances that are found in minerals, soil, and rocks. Values are numeric and range from 129 to 481. 
#6.	Conductivity:  Electrical conductivity (EC)  measures the ionic process of a solution that enables it to transmit current. According to WHO standards, EC value should not exceed 400 ??S/cm. Values are numeric and range from 181 to 753 
#7.	Organic_carbon: Total Organic Carbon is a measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2 mg/L as TOC in treated / drinking water, and < 4 mg/Lit in source water which is used for treatment. Values are numeric and range from 2.2 to 28.2 
#8.	Trihalomethanes: The concentration of THMs in drinking water varies according to the level of organic material in the water, the amount of chlorine required to treat the water, and the temperature of the water that is being treated. THM levels up to 80 ppm are considered safe in drinking water. Values are numeric and range from .74 to 124. 
#9.	Turbidity: The turbidity of water depends on the quantity of solid matter present in the suspended state. The mean turbidity value obtained for Wondo Genet Campus (0.98 NTU) is lower than the WHO recommended value of 5.00 NTU. Values are numeric and range from 1.45 to 6.74 


#install required libraries 
install.packages("caTools")
install.packages("ROCR")
install.packages("gains")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("randomForest")
install.packages("caret")
install.packages("ggplot2")
install.packages("e1071")
install.packages("corrplot")
install.packages("nnet")

#use of library
library(Rcpp)
library(mlbench)
library(caret)
library(lattice)
library(ggplot2)
library(forecast)


###1. Set working directory and Load the dataset
data = read.csv("water_potability.csv")
View(data);
str(data)

#set the baseline
table(data$Potability)
baseline = 1998 /(1998 +1278 ) #60.98901 %

#Data Exploration - Check Summmary and Examine missing data
summary(data)  # NA values found in ph,Sulfate & Trihalomethanes


#replace NA values with mean
data$ph[is.na(data$ph)] = mean(data$ph, na.rm=TRUE)
data$Sulfate[is.na(data$Sulfate)] = mean(data$Sulfate, na.rm=TRUE)
data$Trihalomethanes[is.na(data$Trihalomethanes)] = mean(data$Trihalomethanes, na.rm=TRUE)

#Check correlations - An indication of multicollinearity
correlations = cor(data[c(1,2,3,4,5,6,7,8,9)])
round(correlations,2)  #multicollinearity not found



###2. Partioning the dataset into training and test set
#As there are less number of potable water than non potable water, the data is unbalanced.
#While splitting we need to consider the prortion so that both the splitted datasets are
#true replication of the main dataset.

library(caTools)
set.seed(1000)
split<- sample.split(data,SplitRatio=0.7)
train= subset(data,split==TRUE)
test= subset(data,split==FALSE)

str(train)
str(test)

table(train$Potability) 
# 0 = 1398, 1 = 895
baseline.train = 1398/(1398+895) #60.96816%


table(test$Potability) 
# 0 = 600, 1 = 383 
baseline.test = 600/(600+383) #61.03764%


###3. Logistic Regression
options(scipen=999)
fit_lr.train = glm(Potability ~ ., data=train, family=binomial) #Fit Logistic Regression Model
summary(fit_lr.train) #summarize the fit, only Organic_carbon is significant at 95 CI

fit_lr.train$fitted.values #Check fitted values


#Remove highly correlated variables and run logistic regression again
fit_lr.train = glm(Potability ~ . - ph - Hardness - Solids - Chloramines - Sulfate - Conductivity - Trihalomethanes -Turbidity, data=train, family=binomial)
summary(fit_lr.train)
fit_lr.train$fitted.values


table(train$Potability, fit_lr.train$fitted.values>0.5) #Check Accuracy - 60.96816%
fit_lr.test = predict(fit_lr.train, type="response", newdata=test) #Make Predictions
table(test$Potability, fit_lr.test > 0.5) #Check Accuracy on Test data - 0.6103764

liftdata = cbind(train$Potability, fit_lr.train$fitted.values)

write.csv(liftdata, "liftdata.csv")




###6. Judging Classification Performance: ROCR Curve  to find out optimal cutoff
library(ROCR)
best_model = fit_lr.test
ROCRpred = prediction(best_model, test$Potability)
ROCRperf = performance(ROCRpred, "tpr","fpr")
plot(ROCRperf, col="black", lty=2, lwd=1)
plot(ROCRperf, col=rainbow(4))
lines(c(0,1),c(0,1))

as.numeric(performance(ROCRpred, "auc")@y.values) # 0.5069278

#Determining Optimal cut-off - Given by Youden's Index = max(sensitivity+specificity-1) = max(tpr-fpr)
str(ROCRperf)
ROCRperf@alpha.values[[1]]
ROCRperf@x.values[[1]]
ROCRperf@y.values[[1]]
difference = ROCRperf@y.values[[1]] - ROCRperf@x.values[[1]]
youden = max(difference)
youdendata = as.data.frame(cbind(alpha=ROCRperf@alpha.values[[1]],difference))
which.max(youdendata$difference) #641
youdendata
youdendata[50,] #0.3807448
write.csv(youdendata,"youdendata.csv")



#Optimal cutoff 

table(train$Potability, fit_lr.train$fitted.values>0.4254101) #Check Accuracy - 0.6066289
fit_lr.test = predict(fit_lr.train, type="response", newdata=test) #Make Predictions
table(test$Potability, fit_lr.test > 0.4254101) #Check Accuracy on Test data - 0.5747711



###4. Classsification Trees
library(rpart)
library(rpart.plot) 
library(caret)

fit_ct.train = rpart(Potability ~ ., data=train, method ="class", control = rpart.control(minbucket=90))
prp(fit_ct.train)

fit_ct.train.pred = predict(fit_ct.train, data = train, type="class")
confusionMatrix(factor(fit_ct.train.pred), factor(train$Potability)) # 0.6432621

fit_ct.test.pred = predict(fit_ct.train, newdata = test, type="class")
confusionMatrix(factor(fit_ct.test.pred), factor(test$Potability)) #0.636826


#Improving Accuracy using Random Forest Modeling - Make Sure that Outcome Variable is a Factor for conducting Random Forests
library(randomForest)
str(train)

train$Potability <- as.factor(train$Potability)
fit_rf.train = randomForest(Potability ~ ., data=train, nodesize=200, ntree=200)

fit_rf.train.pred = predict(fit_rf.train, data = train)
confusionMatrix(factor(train$Potability), factor(fit_rf.train.pred)) #0.6323594

fit_rf.test.pred = predict(fit_rf.train, newdata = test)
confusionMatrix(factor(test$Potability), factor(fit_rf.test.pred)) # 0.6328    


#Building Optimal Tree using Cross Validation
library(caret)
library(e1071)
library(rpart)
library(rpart.plot)

fitControl = trainControl(method="cv", number=10)
cartGrid = expand.grid(.cp=(1:100)*0.01)

train(Potability ~ ., data=train, method="rpart", trControl=fitControl, tuneGrid = cartGrid)
fit_cv.train = rpart(Potability ~ ., data=train, method="class", control=rpart.control(cp=0.01))

fit_cv.train.pred = predict(fit_cv.train, data = train, type ="class")
confusionMatrix(train$Potability, fit_cv.train.pred) #0.6502399

fit_cv.test.pred = predict(fit_cv.train, newdata = test, type ="class")
confusionMatrix(factor(test$Potability), factor(fit_cv.test.pred)) #0.6266531
prp(fit_cv.train)


###5. Neural Nets

library(caret)
library(nnet)
 
str(data)

#nnet(x, y, weights, size, Wts, mask, linout = FALSE, entropy = FALSE, softmax = FALSE,
#     censored = FALSE, skip = FALSE, rang = 0.7, decay = 0,
#     maxit = 100, Hess = FALSE, trace = TRUE, MaxNWts = 1000,
#     abstol = 1.0e-4, reltol = 1.0e-8, ...)

fit_nn.train = nnet(Potability ~ . , data=train, size = 50, layers =2, maxit=5000, MaxNWts = 3000, Trace=T)
summary(fit_nn.train)
topmodel = varImp(fit_nn.train) #Finding important predictors
head(topmodel,30)

fit_nn.train.pred = predict(fit_nn.train, data=train)
table(train$Potability, fit_nn.train.pred>0.50) # 0.61099

fit_nn.test.pred = predict(fit_nn.train, newdata=test, method="class", na.rm=TRUE)
table(test$Potability, fit_nn.test.pred>0.50) #0.6103764








###7. Judging Ranking Performance: Building the Lift Chart
library(gains)
best_model = fit_lr.test
gain = gains(as.integer(test$Potability)-1, best_model) #Subtracted 1 as converting factor variable into integer converts it into 1 and 2 rather than 0 and 1
plot(gain$cume.lift)
plot(gain$cume.obs)
plot(c(0,gain$cume.pct.of.total*sum(as.numeric(test$Potability)-1))~c(0,gain$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l")
lines(c(0,sum(as.integer(test$Potability)-1))~c(0, dim(test)[1]), lty=2)
str(test)

#Plot lift Chart - Alternate Code
lift = as.data.frame(cbind("Actual"=test$Potability, "Predicted"= best_model))
lift = lift[order(-lift$Predicted),] # - is added for ordering in descending order
lift$cumsum = cumsum(lift$Actual-1) #Subtracted 1 so that 1 and 2 are taken as 0 and 1
plot(lift$cumsum, type="l", lty=1)
lines(c(0,sum(as.integer(test$Potability)-1))~c(0, dim(test)[1]), lty=2)
write.csv(lift,"liftdata.csv")

















