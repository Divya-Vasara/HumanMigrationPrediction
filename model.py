import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('IndianMigrationHistory1.3.csv')

#Dropping unnecessary columns
cols = ['Country Origin Code','Migration by Gender Code','Country Dest Code','Country Origin Name']
df.drop(cols,axis=1,inplace=True)
df.columns = ['MigrationGender','Destination','1960','1970','1980','1990','2000']
df.drop(['1960','1970'],axis=1,inplace=True)

#Converting the years 1980,1990,2000 values which are strings into float
columns = ['1980','1990','2000']
df[columns] = df[columns].apply(pd.to_numeric,errors='coerce')

#Filling the missing values with median
df['1980'].fillna(df['1980'].median(),inplace=True)
df['1990'].fillna(df['1990'].median(),inplace=True)
df['2000'].fillna(df['2000'].median(),inplace=True)

#Changing the strings into numericals
df['MigrationGender'].replace('Female',0,inplace=True)
df['MigrationGender'].replace('Male',1,inplace=True)
df['DestinationID'] = pd.factorize(df['Destination'])[0]
df.drop('Destination',axis=1,inplace=True)
df = df.rename(columns={'DestinationID':'MigrationDestination'})

#Splitting the data into training and testing
X = df.drop('2000',axis=1)
y= df['2000']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Training the model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=150,max_features='auto')
model.fit(X_train,y_train)

#Dumping the model into a pickle file
pickle.dump(model2,open('migration.pkl','wb'))
