# customer_insurance_prediction
#Problem Statement:
You are working for a new-age insurance company and employ mutiple outreach plans to sell term insurance to your customers. Telephonic marketing campaigns still remain one of the most effective way to reach out to people however they incur a lot of cost. Hence, it is important to identify the customers that are most likely to convert beforehand so that they can be specifically targeted via call. We are given the historical marketing data of the insurance company and are required to build a ML model that will predict if a client will subscribe to the insurance. 

#Data:
The historical sales data is available as a compressed file in train.csv 

#Features: 
age (numeric)
job : type of job
marital : marital status
educational_qual : education status
call_type : contact communication type
day: last contact day of the month (numeric)
mon: last contact month of year
dur: last contact duration, in seconds (numeric)
num_calls: number of contacts performed during this campaign and for this client 
prev_outcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
Output variable (desired target):
y - has the client subscribed to the insurance.

#STEPS:
#1.Import the required libraries and load the dataset.

#2.clean the dataset such as remove the duplicates, dropping the null values, remving the outliers

#3.Exploratary data analysis

#4.Plotting the relation between the features and the target

#5.Encoding the features by using label and one-hot encoding

#6.Splitting the dataset

#7.Sampling the traing data

#8.scaling the training data using standard scaler

#9.Applying the model-logisticregression,Decision tree classifier,xgboost, and random forest classifier
Accuracy_score (logistic regression): 0.9004755059161783
auc_roc for logistic regression 0.9001540175977957
Accuracy_score for decisiontree: 1.0
auc_roc for decisiontree 0.5209102345512119
Learning rate :  0.01  Train score(xgb):  0.9042242108229989  Cross-Val score :  0.8972705715542952
Learning rate :  0.02  Train score(xgb):  0.9138951521984217  Cross-Val score :  0.8993705079939373
Learning rate :  0.03  Train score(xgb):  0.9311583990980834  Cross-Val score :  0.9004770693785753
Learning rate :  0.04  Train score(xgb):  0.9373062288613303  Cross-Val score :  0.8994817386202513
Learning rate :  0.05  Train score(xgb):  0.9415515783540023  Cross-Val score :  0.9012509167359312
Learning rate :  0.1  Train score(xgb):  0.9515395997745209  Cross-Val score :  0.9018049919327238
Learning rate :  0.11  Train score(xgb):  0.9527903043968433  Cross-Val score :  0.9032412115582066
Learning rate :  0.12  Train score(xgb):  0.9536006200676438  Cross-Val score :  0.8998137192587883
Learning rate :  0.13  Train score(xgb):  0.95472801578354  Cross-Val score :  0.9026881142130738
Learning rate :  0.14  Train score(xgb):  0.9558025648252536  Cross-Val score :  0.900919425023224
Learning rate :  0.15  Train score(xgb):  0.956190107102593  Cross-Val score :  0.9022464919571702
Learning rate :  0.2  Train score(xgb):  0.9604002254791432  Cross-Val score :  0.9014719112110694
Learning rate :  0.5  Train score(xgb):  0.9769060033821871  Cross-Val score :  0.8971590964650662
Learning rate :  0.7  Train score(xgb):  0.9838817643742954  Cross-Val score :  0.8957216545250086
Learning rate :  1  Train score(xgb):  0.9902938275084555  Cross-Val score :  0.8919627927443405
Depth  :  1  Training Accuracy(randomforest) :  0.5563169391206313  Cross val score :  0.8839305571526788
Depth  :  2  Training Accuracy(randomforest) :  0.5563169391206313  Cross val score :  0.8839305571526788
Depth  :  3  Training Accuracy(randomforest) :  0.5566340191657272  Cross val score :  0.8842070599504309
Depth  :  4  Training Accuracy(randomforest) :  0.5593468151071026  Cross val score :  0.8870549913633015
Depth  :  5  Training Accuracy(randomforest) :  0.5647371758737317  Cross val score :  0.8919486566019362
Depth  :  6  Training Accuracy(randomforest) :  0.5687006764374295  Cross val score :  0.8950729990629304
Depth  :  7  Training Accuracy(randomforest) :  0.5818242671927847  Cross val score :  0.8961235705407844
Depth  :  8  Training Accuracy(randomforest) :  0.5853649943630215  Cross val score :  0.8970083275021346
Depth  :  9  Training Accuracy(randomforest) :  0.5970617249154453  Cross val score :  0.8977825108692727
Depth  :  10  Training Accuracy(randomforest) :  0.5957229425028185  Cross val score :  0.8997179463497105
**AUROC SCORE**
Logistic regression:0.900
Decision tree:0.520
XGBoost:0.902
Random forest:0.899
Conclusion:
** XGBOOST is the best model for customer conversion prediction**



