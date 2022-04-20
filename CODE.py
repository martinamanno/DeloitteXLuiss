import pandas as pd
import numpy as np
from datetime import datetime 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot
from matplotlib.dates import DateFormatter

#open the datasets
sales = pd.read_csv('Sales.csv')
webtraf = pd.read_csv('WebTraffic.csv')

#create a single dataset
visits = webtraf['visits']
df= sales.join(visits)
df.info()

#formatting dates
df['day'] = pd.to_datetime(df['day'], errors='coerce')
df['day'] = df['day'].dt.strftime('%Y %m %d')
df.info()


#check for missing values 
sales.isnull()
webtraf.isnull()

#create new data: purchases and sales
df['online_purchases']=df['convrate']* df['visits']
df['sales']= df['online_purchases']* df['avspend']

#from float to int
df['sales']=df['sales'].astype(int)
df['online_purchases']=df['online_purchases'].astype(int)
df.info()


#divide years in quarters
df['quarter'] = pd.PeriodIndex(df.day, freq='Q')
df['quarter']=df['quarter'].astype(str) 

#create a column with only months and years
df['ym']= df['day'].dt.strftime('%Y-%m')

#create a dataset for each city
df_rome = df[df['city'] == 'Rome']
df_milan = df[df['city'] == 'Milan']
df_naples = df[df['city'] == 'Naples']


#plot visits, sales and purcheses aggregate by city and for each city


plt.figure(figsize=(15,10))
l=sns.lineplot(data=df, x="ym", y="visits", hue="city", ci=None )
l.legend(loc='center left', bbox_to_anchor=(1, 0.5))
l.set(title= 'visits')

plt.figure(figsize=(15,10))
m=sns.lineplot(data=df, x="ym", y='online_purchases', hue="city", ci=None )
m.legend(loc='center left', bbox_to_anchor=(1, 0.5))
m.set(title= 'online_purchases')

plt.figure(figsize=(15,10))
c=sns.lineplot(data=df, x="ym", y="sales", hue="city", ci=None)
n.legend(loc='center left', bbox_to_anchor=(1, 0.5))
n.set(title= 'sales')
n.set(ylim=(285202, None))


#MILAN
f_milan['day'] = pd.to_datetime(df['day'], errors='coerce')
date_form = DateFormatter("%Y %m")

plt.figure(figsize=(15,10))
a=sns.lineplot(data=df_milan, x="day", y="visits")
a.xaxis.set_major_formatter(date_form)
a.set(title= 'Visits in Milan')

plt.figure(figsize=(15,10))
b=sns.lineplot(data=df_milan, x="day", y="sales")
b.xaxis.set_major_formatter(date_form)
b.set(title= 'Sales in Milan')

plt.figure(figsize=(15,10))
c=sns.lineplot(data=df_milan, x="day", y="online_purchases")
c.xaxis.set_major_formatter(date_form)
b.set(title= 'Online purchases in Milan')


#ROME
df_rome['day'] = pd.to_datetime(df['day'], errors='coerce')
date_form = DateFormatter("%Y %m")

plt.figure(figsize=(15,10))
d=sns.lineplot(data=df_rome, x="day", y="visits")
d.xaxis.set_major_formatter(date_form)
d.set(title= 'Visits in Rome')

plt.figure(figsize=(15,10))
e=sns.lineplot(data=df_rome, x="day", y="sales")
e.xaxis.set_major_formatter(date_form)
e.set(title= 'Sales in Rome')

plt.figure(figsize=(15,10))
f=sns.lineplot(data=df_rome, x="day", y="online_purchases")
f.xaxis.set_major_formatter(date_form)
f.set(title= 'Online purchases in Rome')


#NAPLES
df_naples['day'] = pd.to_datetime(df['day'], errors='coerce')
date_form = DateFormatter("%Y %m")

plt.figure(figsize=(15,10))
g=sns.lineplot(data=df_naples, x="day", y="visits")
g.xaxis.set_major_formatter(date_form)
g.set(title= 'Visits in Naples')

plt.figure(figsize=(15,10))
h=sns.lineplot(data=df_naples, x="day", y="sales")
h.xaxis.set_major_formatter(date_form)
h.set(title= 'Sales in Naples')

plt.figure(figsize=(15,10))
i=sns.lineplot(data=df_naples, x="day", y="online_purchases")
i.xaxis.set_major_formatter(date_form)
i.set(title= 'Online purchases in Naples')


##correlation matrix
df_milan.corr()
corr_matrix = df_milan.corr()
sns.heatmap(corr_matrix, annot=True, cmap='viridis' , center=0, vmin=-1, vmax=1, fmt="0.2f", linewidths=.5)
plt.show()

df_rome.corr()
corr_matrix = df_rome.corr()
sns.heatmap(corr_matrix, annot=True, cmap='viridis' , center=0, vmin=-1, vmax=1, fmt="0.2f", linewidths=.5)
plt.show()

df_naples.corr()
corr_matrix = df_naples.corr()
sns.heatmap(corr_matrix, annot=True, cmap='viridis' , center=0, vmin=-1, vmax=1, fmt="0.2f", linewidths=.5)
plt.show()

'''
#calculate the average aggregating the 3 cities by dates for each variables and plot them
df1= df.groupby(
    [pd.to_datetime(df.day).dt.strftime('%Y %m %d')]
)['visits'].mean().reset_index(name='average_visits')

df2= df.groupby(
    [pd.to_datetime(df.day).dt.strftime('%Y %m %d')]
)['convrate'].mean().reset_index(name='average_convrate')

df3= df.groupby(
    [pd.to_datetime(df.day).dt.strftime('%Y %m %d')]
)['avspend'].mean().reset_index(name='average_avspend')

av_conv=df2['average_convrate']
df_av= df1.join(av_conv)

av_spend=df3['average_avspend']
df_av=df_av.join(av_spend)

av_purchases=df_av['average_convrate']*df_av['average_visits']
df_av= pd.concat([df_av, av_purchases.rename('average_purchases')], axis=1)
av_sales=df_av['average_purchases']*df_av['average_avspend']
df_av= pd.concat([df_av, av_sales.rename('average_sales')], axis=1)


sns.lineplot(data=df_av, x="day", y="average_visits")
sns.lineplot(data=df_av, x="day", y="average_sales")
sns.lineplot(data=df_av, x="day", y="average_purchases")
'''

#VEDIAMO SE LASCIARE O MENO QUESTA PARTE 
#AUTOCORRELATION PLOTS
#MILAN
#visits lag = 100
df_milan2 = df_milan['visits']
df_milan2.head()
plot_acf(df_milan2, lags = 100)
plt.show()
#visits lag = 25
from statsmodels.graphics.tsaplots import plot_acf
plt.figure(figsize=(15, 7))
plot_acf(df_milan2)
plt.title("Autocorrelation visits in Milan", fontsize=15)
plt.ylabel("Correlation",fontsize=15)
plt.xlabel("Lag",fontsize=15)
plt.show()
#autocorrelation test
acorr_ljungbox(df_milan2, lags=[1], return_df=True)
#purchases
df_milan3 = df_milan[['purchases']]
df_milan3.head()
plot_acf(df_milan3, lags = 100)
plt.show()
#autocorrelation test 
from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(df_milan3, lags=[1], return_df=True)
#seasonal decompose delle visits
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df_milan2, model='additive', freq = 12)
result.plot()
pyplot.show()
#purchases
#ADfuller test
from statsmodels.tsa.stattools import adfuller
dftest = adfuller(df_milan3, autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)
#visits ADF
dftest = adfuller(df_milan2, autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)

#df milan for sales 
df_milan4 = df_milan['sales']
df_milan4.head()
dftest = adfuller(df_milan4, autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():

    print("\t",key, ": ", val)

#ROMA
#rome visits
df_rome = df_rome.set_index(df_rome['day'])
df_rome2 = df_rome['visits']
dftest = adfuller(df_rome2, autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():

    print("\t",key, ": ", val)
#purchases
df_rome3 = df_rome['purchases']
dftest = adfuller(df_rome3,autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():

    print("\t",key, ": ", val)

#autocorrelation plot sales
plot_acf(df_rome3, lags = 40) #sales
plt.show()
#autocorrelation test
acorr_ljungbox(df_rome3, lags=[40], return_df=True)
#ADF sales
df_rome4 = df_rome['sales']
dftest = adfuller(df_rome4,autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():

    print("\t",key, ": ", val)
#plot sales autocorr
plot_acf(df_rome4, lags = 50) #sales
acorr_ljungbox(df_rome4, lags=[1], return_df=True)
plt.show()
#VEDIAMO SE LASCIARE O MENO QUESTA PARTE 

#SEASONAL DECOMPOSE (we select 'additive' model to describe the seasonality)
from statsmodels.tsa.seasonal import seasonal_decompose
#seasonal decompose di tot_sales
result = seasonal_decompose(df.tot_sales, model='additive', freq = 12)
result.plot()
pyplot.show()

#tot_visits
result = seasonal_decompose(df.tot_visits, model='additive', freq = 12)
result.plot()
pyplot.show()

#online purchases 
result = seasonal_decompose(df.tot_onpurc, model='additive', freq = 12)
result.plot()
pyplot.show()

#ADF TEST 
#tot_sales
df1 = df[['day','tot_sales']]
df1 = df1.set_index(df1['day'])
df1.drop('day', axis = 1, inplace = True)
from statsmodels.tsa.stattools import adfuller
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

#tot_visits
df2 = df[['day','tot_visits']]
df2 = df2.set_index(df2['day'])
df2.drop('day', axis = 1, inplace = True)
adfuller_test(df2)

#tot_onpurc
df3 = df[['day','tot_onpurc']]
df3 = df3.set_index(df3['day'])
df3.drop('day', axis = 1, inplace = True)
adfuller_test(df3)

#check if after differencing the data is still non stationary
#tot_sales
df_diff_s = df1.copy()
df_diff_s['prev_sales'] = df_diff_s['tot_sales'].shift(1)
df_diff_s['prev_sales'] = df_diff_s['tot_sales'].shift(1)
df_diff_s = df_diff_s.dropna()
df_diff_s['diff'] = (df_diff_s['tot_sales'] - df_diff_s['prev_sales'])
dfdiff = df_diff_s[['day','diff']]
dfdiff.head()
dfdiff = dfdiff.set_index(dfdiff['day'])
dfdiff.drop('day', axis = 1, inplace = True)
adfuller_test(dfdiff) #stationary

#create dataset with dummies 
df_dummy = df.copy()
df_dummy['day'] = pd.to_datetime(df_dummy['day'], errors = 'coerce')
df_dummy['day_of_week'] = df_dummy['day'].dt.day_name()
df_dummy['month'] = df_dummy['day'].dt.month_name()
df_dummy = pd.get_dummies(df_dummy)

#PROPHET 
from fbprophet import Prophet
df = pd.read_csv('df_dummy.csv')
df.drop('Unnamed: 0', axis =1, inplace = True)
#prophet 
dfp = df[['day','tot_sales']]
dfp = dfp[:414]



#LSTM

#https://towardsdatascience.com/predicting-sales-611cb5a252de
from datetime import datetime, timedelta,date
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from __future__ import division#import Keras
import warnings
warnings.filterwarnings("ignore")
from chart_studio import plotly
import plotly.offline as pyoff
import plotly.graph_objs as go

#read the data in csv
df= pd.read_csv('/Users/martinamanno/Desktop/LUISS/CORSI da seguire/DATA SCIENCE IN ACTION/DELOITTE/progetto 2/dataset.csv')

#convert date field from string to datetime
df['day'] = pd.to_datetime(df['day'])

#show first 10 rows
df.head(10)

df['online_purchases']=df['convrate']* df['visits']
df['sales']= df['online_purchases']* df['avspend']
df.head(10)

#represent month in date field as its first day
#df['day'] = df['day'].dt.year.astype('str') + '-' + df['day'].dt.month.astype('str') + '-01'
#df['day'] = pd.to_datetime(df['day'])

#groupby date and sum the sales
#df = df.groupby('day').sales.sum().reset_index()

#plot monthly sales
plot_data = [
    go.Scatter(
        x=df['day'],
        y=df['sales'],
    )
]
plot_layout = go.Layout(
        title='Montly Sales'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

#create a new dataframe to model the difference
df_diff = df.copy()
#add previous sales to the next row
df_diff['prev_sales'] = df_diff['sales'].shift(1)
#drop the null values and calculate the difference
df_diff = df_diff.dropna()
df_diff['diff'] = (df_diff['sales'] - df_diff['prev_sales'])
df_diff.head(10)

#plot sales diff
plot_data = [
    go.Scatter(
        x=df_diff['day'],
        y=df_diff['diff'],
    )
]
plot_layout = go.Layout(
        title='Montly Sales Diff'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

#create dataframe for transformation from time series to supervised
df_supervised = df_diff.drop(['prev_sales'],axis=1)
#adding lags
for inc in range(1,13):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc)
    #drop null values
df_supervised = df_supervised.dropna().reset_index(drop=True)
df_supervised

# Import statsmodels.formula.api
import statsmodels.formula.api as smf
# Define the regression formula
#model = smf.ols(formula='diff ~ lag_1 + lag_2 +lag_3 + lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + lag_10 + lag_11 + lag_12', data=df_supervised)
model = smf.ols(formula='diff ~  lag_1  + lag_2 +lag_3 + lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + lag_10 + lag_11 + lag_12', data=df_supervised)
# Fit the regression
model_fit = model.fit()
# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)

#import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler
df_model = df_supervised.drop(['sales','day'],axis=1)
#split train and test set
train_set, test_set = df_model[0:-6].values, df_model[-6:].values
df_model

#apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)

# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)
# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)

X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

#import Keras
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split

model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=100, batch_size=1, verbose=1, shuffle=False)

y_pred = model.predict(X_test,batch_size=1)
#reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
#rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred)):
    print np.concatenate([y_pred[index],X_test[index]],axis=1)
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))
#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

#create dataframe that shows the predicted sales
result_list = []
sales_dates = list(df_sales[-7:].date)
act_sales = list(df_sales[-7:].sales)
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['date'] = sales_dates[index+1]
    result_list.append(result_dict)
df_result = pd.DataFrame(result_list)
#for multistep prediction, replace act_sales with the predicted sales

#merge with actual sales dataframe
df_sales_pred = pd.merge(df_sales,df_result,on='date',how='left')
#plot actual and predicted
plot_data = [
    go.Scatter(
        x=df_sales_pred['date'],
        y=df_sales_pred['sales'],
        name='actual'
    ),
        go.Scatter(
        x=df_sales_pred['date'],
        y=df_sales_pred['pred_value'],
        name='predicted'
    )
    
]
plot_layout = go.Layout(
        title='Sales Prediction'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
