

#reading in data
originalTrainingData<-read.csv("/home/florian/Arbeitsfläche/Data Science/Machine Learning/Course Project/pml-training.csv", header = TRUE, na.strings=c(""," ","NA"))
originalTestingData<-read.csv("/home/florian/Arbeitsfläche/Data Science/Machine Learning/Course Project/pml-testing.csv", header = TRUE, na.strings=c(""," ","NA"))

#splitting training into training and validation set
set.seed(2612)
library(caret)
trainIndex = createDataPartition(originalTrainingData$classe, p = 0.50,list=FALSE)
training = originalTrainingData[trainIndex,]
validation = originalTrainingData[-trainIndex,]
#View(originalTrainingData)
#names(originalTrainingData)
#Initial Descriptives on Data

table(training$classe)
plot(training$classe)
hist(training$raw_timestamp_part_1)
table(training$kurtosis_roll_belt)
#countig how many NAs are in all columns
na_count <-sapply(originalTrainingData, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
na_count$var<-rownames(na_count)
na_count
##->all variables with 19216 missings are deleted
keep<-na_count[na_count$na_count==0,2]

# Dataset is limited to the columns with no NAs
training2<-subset(training,select=keep)

#Preprocesssing (Box-Cox transformation to correct for skewness, center and scale each variable and then apply PCA)


require(caret)
names(training2)
trans <- preProcess(training2[,-59], 
                   method=c("center", 
                            "scale", "pca"))
trans
trainingPC <- predict(trans, training2)

validationPC <-  predict(trans, validation)


#in this object the model (classe~PC1:PC39 ist stored), so it can be used for all model types
model <- as.formula(paste('classe~', paste(colnames(trainingPC)[6:31], collapse='+')))
model
# Tests with some methods from https://topepo.github.io/caret/modelList.html
#Multinomial Regression


multinom<-train(model, method="multinom",data=trainingPC)
pred_multinom<-predict(multinom, validationPC)
confusionMatrix(validationPC$classe, pred_multinom)
#Accuray: 0.95

#ada cannot handle more than two class responses

#Linear Discriminant Analysis
lda<-train(classe~., method="lda",data=trainingPC)
pred_lda<-predict(lda, validationPC)
confusionMatrix(pred_lda,validationPC$classe)
#Accuray: 0.99

#Naive Base (running too long)
nb<-train(model, method="nb",data=trainingPC)
pred_nb<-predict(nb, validationPC)
confusionMatrix(pred_nb,validationPC$classe)
#accuracy: 0.62

#Rule-BAsed Classifier ---< r crashes every time this is run
#not run PART<-train(model, method="PART",data=trainingPC)
#not run pred_PART<-predict(PART, validationPC)
#not run confusionMatrix(pred_PART,validationPC$classe)
#not run 


#Penalized Discriminatn Analysis
pda<-train(model, method="pda",data=trainingPC)
pred_pda<-predict(pda, validationPC)
confusionMatrix(pred_pda,validationPC$classe)
#accuracy: 0.98


#Shrinkage Discriminatn Analysis
sda<-train(model, method="sda",data=trainingPC)
pred_sda<-predict(sda, validationPC)
confusionMatrix(pred_sda,validationPC$classe)
#accuracy: 0.98


#Random Forest (22:14-22:50 aborted--> too long)
rf<-train(model, method="rf",data=trainingPC)
pred_rf<-predict(rf, validationPC)
confusionMatrix(pred_rf,validationPC$classe)
#Accuray: 0.66


# Model Stacking of the three discrimant analyses models
combinedPredDF<-data.frame(pred_sda, pred_pda, pred_lda, classe=validationPC$classe)
combModFit<-train(classe~, method="gam", data=combinedPredDF)
pred_comb<-predict(combModFit, combinedPredDF)
confusionMatrix(pred_comb,validationPC$classe)

# --> Model Stacking did not yield improvement, final model is just lda

# Predicting on the test set

testingPC <- predict(trans, originalTestingData)
pred_Test<-predict(lda, newdata=testingPC)
pred_Test

pred_Test<-as.character(pred_Test)



# stuff for writing the files for submission into the workingdirectory
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}

setwd("/home/florian/Arbeitsfläche/Data Science/Machine Learning/Course Project/Submission")
pml_write_files(pred_Test)
