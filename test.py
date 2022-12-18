
import joblib
loaded_model = joblib.load('3view_model.pkl')
loaded_model.predict([5,35])  # 4thTemp , PeakCh4 
