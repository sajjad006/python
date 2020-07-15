import pandas as pd
from fbprophet import Prophet

def predictions(df):

	m = Prophet()
	m.fit(df)

	future = m.make_future_dataframe(periods=5)
	print(future)

	forecast = m.predict(future)
	print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
