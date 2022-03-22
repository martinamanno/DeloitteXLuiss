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
