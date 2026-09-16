from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
import pandas as pd 

# Data Set 

data_set=pd.DataFrame({
    "age":[22,23,21,33,45,50,40,20,30,35],
    "salary":[2200,3500,5500,4100,7600,2900,7200,8100,6200,10000],
    "previous_purchase":[2,3,5,7,3,2,8,9,10,12],
    "browsing_time":[3,4,5,3,4,5,2,6,7,8],
    "purchased":[1,0,1,0,1,1,0,1,0,0]
})

first_rows=data_set.head(5)
print(first_rows)

statistics_information=data_set.describe()
print(statistics_information)

shape_of_data_set=data_set.shape
print(shape_of_data_set)

missing_values=data_set.isna()
print(missing_values)
count_missing_values=data_set.sum()
print(count_missing_values)

print(data_set.info())



# Decisions Tree   Regressor 

X1=data_set[["age","salary","previous_purchase","browsing_time"]]
Y1=data_set["purchased"]

X1_train,X1_test,Y1_train,Y1_test=train_test_split(X1,Y1,test_size=0.2,random_state=5)
model1=DecisionTreeRegressor()

model1.fit(X1_train,Y1_train)

prediction_regressor=model1.predict(X1_test)
print(prediction_regressor)

mse=mean_squared_error(Y1_test,prediction_regressor)
print("MSE =",mse)


# Decisions tree Classifier 

data_set["class"]=data_set["purchased"].apply(lambda x: "yes" if x>0 else "no")
X2=data_set[["age","salary","previous_purchase","browsing_time"]]
Y2=data_set["class"]


X2_train,X2_test,Y2_train,Y2_test=train_test_split(X2,Y2,test_size=0.2,random_state=5)
model_gini=DecisionTreeClassifier(criterion="gini",max_depth=3)
model_entropy=DecisionTreeClassifier(criterion="entropy",max_depth=3)

model_gini.fit(X2_train,Y2_train)
model_entropy.fit(X2_train,Y2_train)

prediction_gini=model_gini.predict(X2_test)
prediction_entropy=model_entropy.predict(X2_test)
print(prediction_entropy)
print(prediction_gini)

accuracy_score_gini=accuracy_score(Y2_test,prediction_gini)
accuracy_score_entropy=accuracy_score(Y2_test,prediction_entropy)
print(accuracy_score_entropy)
print(accuracy_score_gini)