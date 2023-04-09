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

#conclusion:

