import pandas as pd
import catboost
from sklearn.metrics import classification_report

df=pd.read_csv('3view_sensor.csv')
df['check']=df['판정'].map(lambda x: 'normal' if x=='적정' else 'leak' if x=='부족' else 'over' if x=='과다' else 0)
df=df.reset_index()
df.columns = ['timeseries', ' 4thTemp', 'PeakCh4', '판정','check']

model=catboost.CatBoostClassifier()
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(df[[' 4thTemp','PeakCh4']], df['check'], test_size=0.1, shuffle=True, stratify=df['check'], random_state=42)
model.fit(x_train,y_train,silent=True)
y_pred = model.predict(x_valid)
print(classification_report(y_valid, y_pred))
#model save
import joblib 
joblib.dump(model, '3view_model.pkl')
