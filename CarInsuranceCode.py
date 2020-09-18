import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, plot_confusion_matrix, roc_curve, auc


data = pd.read_csv('insurance_claims.csv')
data

#info on the data
data.info()

# checking the nan values
data.isnull().sum()

# data stats
des = data.describe()

data['fraud_reported'] 

fraud_output = []
response = data.iloc[:,-1]
 
# converting output to binary   
for i in  range(len(response)):
     fraud_output.append(1 if response[i] == 'Y' else 0 )
    
data['fraud_reported'] = pd.Series(fraud_output)
data

# data analysis
plt.hist(data['age'])
plt.ylabel('No. of Customers')
plt.xlabel('customer age')
plt.title("Histogram of age of customers")

data["insured_sex"].hist()
plt.title("Hstogram of insured_sex of customers")
plt.xlabel("insured_sex of customers")
plt.ylabel("No. of customers")


# checking the output if its balanced!!
sns.countplot(data.iloc[:,-1]) #imbalanced

#feature selection#
rcParams['figure.figsize'] = [10,10]
sns.heatmap(data.corr())


data.info()
#policy number in x is irrelevant for prediction
x = data.iloc[:,1:38]
x_columns = x.columns
y = data.iloc[:,-1] 


#ordinal encoder
#encode categorical features as an integer array.
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder() 
oe.fit(x)
x_enc = oe.transform(x)
x_enc = pd.DataFrame(x_enc , columns= x_columns )  
columns = pd.DataFrame(x_enc.columns)

#Label encoder
#Encode target labels with value between 0 and n_classes-1
#This transformer should be used to encode target values, i.e. y, and not the input X.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() 
le.fit(y)
y_enc = pd.DataFrame((le.transform(y)), columns =['fraud_reported'])


#extracting the features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

best_features = SelectKBest(score_func=chi2, k = 'all')
fit = best_features.fit(x_enc,y_enc)

#scores of the features
scores = pd.DataFrame(fit.scores_)

feature_scores = pd.concat([columns,scores], axis = 1)
feature_scores.columns = ['features','scores']
feature_scores.nlargest(37,'scores')

#dropping the columns with least scores
columns_to_drop = ['collision_type',    
                   'insured_hobbies',     
                   'umbrella_limit',     
                   'incident_state',     
                   'number_of_vehicles_involved',     
                   'capital_loss',     
                   'incident_type',     
                   'incident_city',
                   'authorities_contacted',    
                   'witnesses',     
                   'auto_make',
                   'policy_csl',     
                   'bodily_injuries',     
                   'property_damage',     
                   'age',     
                   'policy_state',     
                   'police_report_available',     
                   'insured_relationship',     
                   'insured_sex',     
                   'auto_year',     
                   'insured_education_level',     
                   'incident_hour_of_the_day',     
                   'policy_deductable',     
                   'insured_occupation',     
                   'auto_model']

imp_data = data.drop(columns_to_drop, axis =1) 

imp_data.info()

X = imp_data.iloc[:,1:13]
Y = imp_data.iloc[:,-1] 

#dummy coding
X = pd.get_dummies(X, drop_first=True)

#we are using an minority class over sampling technique 
#and converting it into a balanced dataset

#from imblearn.over_sampling import SMOTE
#sm = SMOTE(random_state = 24)
#X , Y = sm.fit_resample(X, Y)

from imblearn.over_sampling import RandomOverSampler
ROS = RandomOverSampler()
X, Y = ROS.fit_sample(X,Y)


#plotting the independent variable
sns.countplot(Y)

#scaling the data
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)



#splitting the training and testing data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=(0.2))

rcParams['figure.figsize'] = [5,5]

#xgboost

from xgboost import  XGBClassifier
XGB = XGBClassifier()
XGB.fit(x_train,y_train)
XGB_Predictions = XGB.predict(x_test)
print(accuracy_score(y_test, XGB_Predictions))
print(classification_report(y_test, XGB_Predictions))
plot_confusion_matrix(XGB , x_test , y_test)

#cross validation score
from sklearn.model_selection import cross_val_score
print(cross_val_score(XGB,X,Y,cv=10,scoring='accuracy'))




#random forest classifier
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(x_train,y_train)
RF_Predictions = RF.predict(x_test)
print(accuracy_score(y_test, RF_Predictions))
print(classification_report(y_test, RF_Predictions))
plot_confusion_matrix(RF , x_test , y_test)

#cross validation score
print(cross_val_score(RF,X,Y,cv=10,scoring='accuracy'))



#support vector machine classifier # best model
from sklearn import svm
SVC = svm.SVC()
SVC.fit(x_train,y_train)
SVC_Predictions = SVC.predict(x_test)
print(accuracy_score(y_test,SVC_Predictions)) 
print(classification_report(y_test, SVC_Predictions))
plot_confusion_matrix(SVC , x_test , y_test)

#cross validation score
print(cross_val_score(SVC,X,Y,cv=10,scoring='accuracy'))


#Receiving Operator Charecteristic for SVM
plt.title("ROC curve for SVM:")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

model_pred = {}

tpr , fpr , threshold = roc_curve(y_test, SVC_Predictions, pos_label = 1)
model_pred["SVC"] = [tpr, fpr]
print("AUC value = " +str(auc(tpr,fpr)))


for key, value in model_pred.items():
    model_list = model_pred[key]
    plt.plot(model_list[0], model_list[1], label=key)
    plt.legend()
plt.show()



#LDA gives you best result out of all incase you filter out minimal irrelevant features.
#since its a dimensionality reduction algorithm.
#lets try

data 
x_lda = data.iloc[:,1:38]
y_lda = data.iloc[:,-1]


columns_to_drop_lda = ["auto_model", 
                      "policy_bind_date",
                      "policy_state", 
                      "incident_date",
                      "incident_state",
                      "incident_city",
                      "incident_location", 
                      "policy_csl"]

x_lda = x_lda.drop(columns_to_drop_lda, axis=1)

x_lda = pd.get_dummies(x_lda, drop_first=True)
x_lda, y_lda = ROS.fit_sample(x_lda,y_lda)
x_lda = ss.fit_transform(x_lda)

#splitting the training and testing data
x_lda_train, x_lda_test, y_lda_train, y_lda_test = train_test_split(x_lda, y_lda, test_size=(0.2))


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA = LinearDiscriminantAnalysis()
LDA.fit(x_lda_train,y_lda_train)
LDA_Predictions = LDA.predict(x_lda_test)
print(accuracy_score(y_lda_test, LDA_Predictions))
print(classification_report(y_lda_test, LDA_Predictions))































