import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
#read the data
df_train = pd.read_csv('train.csv')
print(df_train)
df=df_train.replace({'y':{'yes':1,'no':0}})
print(df)
a=df.dtypes
print(a)
# print(df_train.dtypes)
#data cleaning
b=df.isnull().sum()
print(b)
print(df.isnull().sum())
print('vjg',df.duplicated().sum())
c=df.drop_duplicates()
print('fhdf',c)
# print('nmn',df_train.info())
print('cvbcb',df.describe())
lower_threshold = []
upper_threshold = []
for i in ["age","day","dur","num_calls"]:
    q3 = df[i].quantile(0.75)
    q1 = df[i].quantile(0.25)
    iqr = q3 - q1
    u = q3 + (1.5 * iqr)
    l = q1 - (1.5 * iqr)
    print(i,":",l,u)
df_tr = df.age.clip(10.5,70.5)
df_tr1 = df.day.clip(-11.5,40.5)
df_tr2 = df_train.dur.clip(-221.0,643.0)
df_tr3 = df_train.num_calls.clip(-2.0,6.0)
print(df_tr.describe(),df_tr1.describe(),df_tr2.describe(),df_tr3.describe())
print(df_train["y"].value_counts())
sn.countplot(x="y",data=df_train)
plt.title("label")
# plt.show()

categ_variable=df_train.select_dtypes(include=["object"]).columns
print(categ_variable)
# plt.style.use("ggplot")
for column in categ_variable:
    plt.figure(figsize=(20,4))
    # ax=plt.subplot(121)
    df_train[column].value_counts(normalize=True).plot(kind="bar")
    plt.xlabel(column)
    plt.ylabel("customer count")
    plt.title(column)
    # plt.show()
cat_variable=['job','education_qual']
for column in cat_variable:
    mode=df[column].mode()[0]
    df_train[column] = df[column].replace("unknown",mode)
    print(df_train[column])
num_variable=df_train.select_dtypes(include=np.number)
ak=num_variable.head()
print(ak)
for column in ak:
    plt.figure(figsize=(10, 4))
    sn.displot(df_train[column],kde=True)
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.title(column)
    # plt.show()
#feature vs target
categ_variable = df.select_dtypes(include=["object"]).columns
for column in categ_variable:
    df.groupby(column)['y'].mean().sort_values(ascending=False).plot(kind='bar')
    #plt.show()
for i in categ_variable:
    print(i,":",df[i].unique())
#encoding the data
lab = preprocessing.LabelEncoder()
# df['job'] =lab.fit_transform(df['job'])
df['marital'] =lab.fit_transform(df['marital'])
df['education_qual'] =lab.fit_transform(df['education_qual'])
k=['mon','job','call_type','prev_outcome']
def encoding_data(df,k):
    for i in k:
        df_dummies=pd.get_dummies(df[i],prefix=i)
        df=pd.concat([df,df_dummies],axis=1)
        df.drop([i],inplace=True,axis=1)
    return df
df=encoding_data(df,k)
print(df)
#splitting dataset
tr=df.drop(['y'],axis=1)#data
print('bcbn', tr)
target=df['y']#target
print('vcb', target)
X_train, X_test, Y_train, Y_test = train_test_split(tr,target,test_size=0.2,random_state=42)
print(X_train, X_test, Y_train, Y_test)
#sampling the training data
smote = SMOTETomek(sampling_strategy=0.80)
tr_data_smote, tr_tar_smote = smote.fit_resample(X_train,Y_train)
print(tr_tar_smote.value_counts())
scaler = StandardScaler()
X_train_scal = scaler.fit_transform(tr_data_smote)
X_test_scal = scaler.transform(X_test)
#MODELS
#logisticregression
log_reg = LogisticRegression() # initialise the model, ready to be used

log_reg.fit(X_train_scal, tr_tar_smote)#scaling is not mandatory
log_score=log_reg.score(X_test_scal, Y_test)
log_pred=log_reg.predict_proba(X_test_scal)
print(log_pred)
# conf=confusion_matrix(Y_test,log_pred)
# print(conf)

acc = log_reg.score(X_test_scal, Y_test)
print("Accuracy_score:", acc)
auc_roc = roc_auc_score(Y_test, log_pred[:, 1])
print("auc_roc for logistic regression", auc_roc)
#decisiontree
DecisionTreeClass = DecisionTreeClassifier(random_state=0, splitter='best')
DecisionTreeClass.fit(tr_data_smote, tr_tar_smote)
dec_pred = DecisionTreeClass.predict(X_test)
# conf=confusion_matrix(Y_test,dec_pred)
# print(conf)
acc = DecisionTreeClass.score(tr_data_smote, tr_tar_smote)
print("Accuracy_score for decisiontree:",acc)
deci_pred = log_reg.predict_proba(X_test)[:,1]
auc_roc=roc_auc_score(Y_test, deci_pred)
print("auc_roc for decisiontree", auc_roc)
#xgboost
xgb_mod = xgb.XGBClassifier() # initialise the model, ready to be used
for lr in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.2, 0.5, 0.7, 1]:
  model = xgb.XGBClassifier(learning_rate = lr, n_estimators=100, verbosity = 0) # initialise the model
  model.fit(tr_data_smote, tr_tar_smote) #train the model
  print("Learning rate : ", lr, " Train score(xgb): ", model.score(tr_data_smote, tr_tar_smote), " Cross-Val score : ", np.mean(cross_val_score(model, X_test, Y_test, cv=10)))

#randomforest
rf= RandomForestClassifier(max_depth=2, n_estimators=100, max_features="sqrt")    #max_depth=log(no of features)
rf.fit(X_train, Y_train)
y_pred= rf.predict(X_test)
for depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    rf = RandomForestClassifier(max_depth=depth, n_estimators=100,
                                max_features="sqrt")  # will tell the DT to not grow past the given threhsold
    # Fit dt to the training set
    rf.fit(X_train, Y_train)  # the model is trained
    # rf = RandomForestClassifier(max_depth=depth, n_estimators=100,
    #                             max_features="sqrt")  # a fresh model which is not trained yet
    trainAccuracy = accuracy_score(tr_tar_smote, rf.predict(tr_data_smote))
    valAccuracy = cross_val_score(rf, X_train, Y_train,
                                  cv=10)  # syntax : cross_val_Score(freshModel,fts, target, cv= 10/5)

    print("Depth  : ", depth, " Training Accuracy(randomforest) : ", trainAccuracy, " Cross val score : ", np.mean(valAccuracy))




