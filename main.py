import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
url="https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Students%20Performance.csv"

df=pd.read_csv(url)
df.to_csv("Students Performance.csv",index=False)

print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df.info())
print(df.shape)
print(df.columns)
print(df.duplicated().sum())
df['test preparation score']=df['test preparation course'].map({'completed':1,'none':0})
x=df[['writing score','reading score','test preparation score']]
y=df['math score']
model=LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
ypred=model.predict(x_test)
print("r2 score:",r2_score(y_test,ypred))
print("mean squared error:",mean_squared_error(y_test,ypred))
sns.boxplot(x='test preparation course',y='math score',data=df)
plt.title('course vs score')
plt.show()
sns.boxplot(x='test preparation course',y='reading score',data=df)
plt.show()
sns.boxplot(x='test preparation course',y='writing score',data=df)
plt.show()
sns.heatmap(df[['math score','reading score','writing score']].corr(),annot=True,cmap='coolwarm')
plt.title('correlation between scores')
plt.show()
