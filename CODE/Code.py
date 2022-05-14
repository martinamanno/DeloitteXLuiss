"""
Deloitte X Luiss
Machine Learning techniques for counterfactual estimation and forecasting
Carlo Ardito, Martina Manno, Olimpia Sannucci
"""


# Import required libraries
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from prophet.plot import plot_plotly


# Open the datasets
sales = pd.read_csv('Sales.csv')
webtraf = pd.read_csv('WebTraffic.csv')
googleq = pd.read_csv('Gsearchdata.csv')


# Merging the datasets
visits = webtraf['visits']
df = sales.join(visits)
gq = googleq['Gsearchprod']
df = df.join(gq)


# Check for missing values
df.isnull()


# Additional variables to be created: purchases and sales
df['online_purchases'] = df['convrate'] * df['visits']
df['sales'] = df['online_purchases'] * df['avspend']


# Formatting dates
df['day'] = pd.to_datetime(df['day'], errors='coerce')
df['day'] = df['day'].dt.strftime('%Y %m %d')
df['day'] = pd.to_datetime(df['day'], errors='coerce')


# The complete dataset is ready and can be saved
df.to_csv('dataset.csv')


# Explore correlation between variables: compute the correlation matrix
df.corr()
corr_matrix = df.corr()
matrix = np.triu(corr_matrix)
np.fill_diagonal(matrix, False)
sns.heatmap(corr_matrix, annot=True, mask=matrix, cmap='viridis', center=0, vmin=-1, vmax=1, fmt="0.2f", linewidths=.5)


# Divide years in quarters
df['quarter'] = pd.PeriodIndex(df.day, freq='Q')
df['quarter'] = df['quarter'].astype(str)


# Create a column with only months and years
df['ym'] = df['day'].dt.strftime('%Y-%m')


# Visualize visits, online purchases and sales comparing cities
plt.figure(figsize=(15, 10))
v = sns.lineplot(data=df, x="ym", y="visits", hue="city", ci=None)
v.legend(loc='center left', bbox_to_anchor=(1, 0.5))
v.set(title='visits')

plt.figure(figsize=(15, 10))
p = sns.lineplot(data=df, x="ym", y='online_purchases', hue="city", ci=None)
p.legend(loc='center left', bbox_to_anchor=(1, 0.5))
p.set(title='online_purchases')

plt.figure(figsize=(15, 10))
s = sns.lineplot(data=df, x="ym", y="sales", hue="city", ci=None)
s.legend(loc='center left', bbox_to_anchor=(1, 0.5))
s.set(title='sales')
s.set(ylim=(285202, None))


# Analyze data by city
df_rome = df[df['city'] == 'Rome']
df_milan = df[df['city'] == 'Milan']
df_naples = df[df['city'] == 'Naples']


# Milan
plt.figure(figsize=(15, 10))
a = sns.lineplot(data=df_milan, x="day", y="visits")
a.set(title='Visits in Milan')

plt.figure(figsize=(15, 10))
b = sns.lineplot(data=df_milan, x="day", y="sales")
b.set(title='Sales in Milan')

plt.figure(figsize=(15, 10))
c = sns.lineplot(data=df_milan, x="day", y="online_purchases")
c.set(title='Online purchases in Milan')

plt.figure(figsize=(15, 10))
h = sns.lineplot(data=df_milan, x="day", y="Gsearchprod")
h.set(title='Google queries Milan')


# Rome
plt.figure(figsize=(15, 10))
d = sns.lineplot(data=df_rome, x="day", y="visits")
d.set(title='Visits in Rome')

plt.figure(figsize=(15, 10))
e = sns.lineplot(data=df_rome, x="day", y="sales")
e.set(title='Sales in Rome')

plt.figure(figsize=(15, 10))
f = sns.lineplot(data=df_rome, x="day", y="online_purchases")
f.set(title='Online purchases in Rome')

plt.figure(figsize=(15, 10))
g = sns.lineplot(data=df_rome, x="day", y="Gsearchprod")
g.set(title='Google queries Rome')


# Naples
plt.figure(figsize=(15, 10))
i = sns.lineplot(data=df_naples, x="day", y="visits")
i.set(title='Visits in Naples')

plt.figure(figsize=(15, 10))
l = sns.lineplot(data=df_naples, x="day", y="sales")
l.set(title='Sales in Naples')

plt.figure(figsize=(15, 10))
m = sns.lineplot(data=df_naples, x="day", y="online_purchases")
m.set(title='Online purchases in Naples')

plt.figure(figsize=(15, 10))
n = sns.lineplot(data=df_naples, x="day", y="Gsearchprod")
n.set(title='Google queries Naples')

# Correlation matrix for each city
df_milan.corr()
corr_matrix = df_milan.corr()
matrix = np.triu(corr_matrix)
np.fill_diagonal(matrix, False)
sns.heatmap(corr_matrix, annot=True, mask=matrix, cmap='viridis', center=0, vmin=-1, vmax=1, fmt="0.2f", linewidths=.5)

df_rome.corr()
corr_matrix = df_rome.corr()
matrix = np.triu(corr_matrix)
np.fill_diagonal(matrix, False)
sns.heatmap(corr_matrix, annot=True, mask=matrix, cmap='viridis', center=0, vmin=-1, vmax=1, fmt="0.2f", linewidths=.5)

df_naples.corr()
corr_matrix = df_naples.corr()
matrix = np.triu(corr_matrix)
np.fill_diagonal(matrix, False)
sns.heatmap(corr_matrix, annot=True, mask=matrix, cmap='viridis', center=0, vmin=-1, vmax=1, fmt="0.2f", linewidths=.5)


# Aggregate values of each city by day
df['day'] = pd.to_datetime(df['day'], errors='coerce')
df = df.groupby([df['day'].dt.date]).sum()

# The aggregate dataset is ready and can be saved
df.to_csv('dataset_final.csv')


# Seasonal decompose di sales
result = seasonal_decompose(df.sales, model='additive', period = 12)
result.plot()
pyplot.show()


# Seasonal decompose di visits
result = seasonal_decompose(df.visits, model='additive', freq = 12)
result.plot()
pyplot.show()


# Seasonal decompose di online purchases 
result = seasonal_decompose(df.online_purchases, model='additive', freq = 12)
result.plot()
pyplot.show()


# AD Fuller Test
df.reset_index(drop=False, inplace=True)


# Creating different dataframe containing the specific variables
df1 = df[['day', 'sales']]
df1 = df1.set_index(df1['day'])
df1.drop('day', axis = 1, inplace = True)

df2 = df[['day','visits']]
df2 = df2.set_index(df2['day'])
df2.drop('day', axis = 1, inplace = True)

df3 = df[['day','online_purchases']]
df3 = df3.set_index(df3['day'])
df3.drop('day', axis = 1, inplace = True)


# AD Fuller Test on sales
test_result=adfuller(df1)
def adfuller_test(df1):
    result=adfuller(df1)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
        
adfuller_test(df1)


# AD Fuller Test on visits
adfuller_test(df2)


# AD Fuller Test on online purchases
adfuller_test(df3)


# Garch analysis 
df_dummy = df.copy()
df_dummy['day'] = pd.to_datetime(df_dummy['day'], errors = 'coerce')
df_dummy['day_of_week'] = df_dummy['day'].dt.day_name()
df_dummy['month'] = df_dummy['day'].dt.month_name()
df_dummy = pd.get_dummies(df_dummy)


#differencing
df_dummy['prev_sales'] = df_dummy['sales'].shift(1)


#drop the null values and calculate the dummyerence
df_dummy = df_dummy.dropna()
df_dummy['diff_sales'] = (df_dummy['sales'] - df_dummy['prev_sales'])
df_dummy['prev_visits'] = df_dummy['visits'].shift(1)
df_dummy = df_dummy.dropna()
df_dummy['diff_visits'] = (df_dummy['visits'] - df_dummy['prev_visits'])
df_dummy.to_csv('df_dummy.csv')


# Prophet Algorithm on sales
dfd = pd.read_csv('df_dummy.csv')
dfp = df[['day','sales']]
dfp = dfp[:414] #considering just the data before the campaign
dfp.rename({'day':'ds','sales':'y'},axis =1, inplace = True)

model = Prophet(seasonality_mode= 'additive',interval_width= 0.95, weekly_seasonality= 'auto',yearly_seasonality=True,daily_seasonality='auto')
model.add_country_holidays(country_name='IT')
model.fit(dfp)
future_dates = model.make_future_dataframe(periods=75) #time period to forecast
predictions = model.predict(future_dates) # Predictions
model.plot(predictions)
model.plot_components(predictions)
yhat = predictions['yhat']
dfp = dfp.join(yhat)
r2_score(dfp.y,dfp.yhat)


# Calculating the difference bewteen actual and predicted
diff = predictions[['ds','yhat']]
y = df['sales']
diff = diff.join(y)
diff[414:486]
differenza = diff['sales'] - diff['yhat']
sum_tot_sales = diff['sales'].sum()
sum_yhat = diff['yhat'].sum()
(sum_tot_sales - sum_yhat)/sum_tot_sales



# Prophet Algorithm on visits
dfp_v = df[['day','visits']]
dfp_v = dfp_v[:414] #considering just the data before the campaign
dfp_v.rename({'day':'ds','visits':'y'},axis =1, inplace = True)

model2 = Prophet(seasonality_mode= 'additive',interval_width= 0.95, weekly_seasonality= 'auto',yearly_seasonality=True,daily_seasonality='auto')
model2.add_country_holidays(country_name='IT')
model2.fit(dfp_v)

future_dates = model2.make_future_dataframe(periods=75) #time period to forecast
predictions_v = model2.predict(future_dates)
model2.plot(predictions_v)
model2.plot_components(predictions_v)

yhat = predictions_v['yhat'] 
dfp_v = dfp_v.join(yhat)
r2_score(dfp_v.y,dfp_v.yhat)
