import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
# df=pd.read_csv("ds_dataset2.csv")
filename=sys.argv[1];
df=pd.read_csv(filename)
df=df.drop(['First Name'], axis=1)
df=df.drop(['Last Name'], axis=1)
df=df.drop(['State'], axis=1)
df=df.drop(['Link to updated Resume (Google/ One Drive link preferred)'],axis=1)
df=df.drop(['link to Linkedin profile'],axis=1)
df=df.drop(["Certifications/Achievement/ Research papers"],axis=1)
df=df.drop(["Email Address"], axis=1)
df=df.drop(["Contact Number"], axis=1)
df=df.drop(["Emergency Contact Number"], axis=1)
df=df.drop(['Degree'], axis=1)
df=df.drop(["Course Type"],axis=1)
df=df.drop(['Current Employment Status'],axis=1)
df=df.drop(['City','Zip Code','DOB [DD/MM/YYYY]','Age','Gender','College name','University Name','Expected Graduation-year','How Did You Hear About This Internship?'],1)
df.dropna(axis = 1, how='all',inplace=True)
df["Label"] = df["Label"].replace({'eligible':1,'ineligible':0})
df["Java"]=df["Have you worked core Java"].replace({'Yes':1,'No':0})
df["DB"]=df["Have you worked on MySQL or Oracle database"].replace({'Yes':1,'No':0})
df["OOP"]=df["Have you studied OOP Concepts"].replace({'Yes':1,'No':0})
df=df.drop(["Have you worked core Java","Have you worked on MySQL or Oracle database","Have you studied OOP Concepts"],1)
rename_dict={'Major/Area of Study':'Area of Study',
             'Which-year are you studying in?':'Current Year',
             'CGPA/ percentage':'CGPA',
             'Programming Language Known other than Java (one major)':'Programming Language',
             'Rate your written communication skills [1-10]':'Written communication skills',
             'Rate your verbal communication skills [1-10]':'Verbal communication skills'}
df.rename(columns = rename_dict, inplace = True) 
df = pd.get_dummies(df)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
y = df['Label']
X = df_scaled.copy()
X=X.drop(["Label"],1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
scoring = ['f1']
scores = cross_validate(DecisionTreeClassifier(),X_train,y_train,scoring=scoring,cv=20)
sorted(scores.keys())
dtree_f1 = scores['test_f1'].mean()

scores = cross_validate(RandomForestClassifier(n_estimators=100),X_train,y_train,scoring=scoring,cv=20)
sorted(scores.keys())
rforest_f1 = scores['test_f1'].mean()

scores = cross_validate(GradientBoostingClassifier(), X_train, y_train, scoring=scoring, cv=20)
sorted(scores.keys())
xgb_f1 = scores['test_f1'].mean()

scores = cross_validate(AdaBoostClassifier(), X_train, y_train, scoring=scoring, cv=20)
sorted(scores.keys())
adab_f1 = scores['test_f1'].mean()


models_comparison = pd.DataFrame({
    'Model'       : ['Decision Tree','Random Forest','XGBoost','AdaBoost'],
    'F1_score'    : [dtree_f1, rforest_f1,xgb_f1,adab_f1],
    }, columns = ['Model', 'F1_score'])
print(models_comparison['F1_score'].max())
