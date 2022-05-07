#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df=pd.read_csv ("Weather_forecast.csv")

df


# In[72]:


df.index


# In[73]:


df.set_index("Date",inplace = True)


# In[74]:


df


# In[75]:


df


# In[ ]:





# In[76]:


df.info()


# In[77]:


type(df.index)


# In[78]:


df.index


# In[79]:


df.columns


# In[80]:


df=pd.read_csv ("Weather_forecast.csv",parse_dates=["Date"])#changing the index to date type


# In[81]:


df.head(10)


# In[82]:


type(df.Date[0])



# In[83]:


df=pd.read_csv ("Weather_forecast.csv",parse_dates=["Date"],index_col="Date") 
#setting date as the index and changig it to date type


# In[84]:


df.index


# In[85]:



df["2014-01"]#Retriving one months data 


# In[86]:


df["Low Temperature (C)"].resample('M').mean() #monthly low temperature mean 


# In[87]:


df["High Temperature (C)"].resample('M').mean() #monthly high temperature mean


# In[88]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[89]:


df["Average Temperature (C)"].resample('m').mean().plot() 
#just a demonestration of monthly average temperature


# In[90]:


df["Average Temperature (C)"].resample('w').mean() #weekly average temperature mean


# In[91]:


df["Average Temperature (C)"].resample('m').mean().plot(kind = "bar") 
#just a demonestration of monthly average temperature


# In[92]:


#df[["2014-01-05":"2014-12-10"],["Average Temperature (C)"]].resample('w').mean()


# In[93]:


df.columns


# In[94]:


g = df.groupby("weather")


# In[95]:


g


# In[96]:


#g.get_group('Fog')
for weather, weather_df in g:
    print(weather)
    print(weather_df)


# In[97]:


g.describe()


# In[98]:


g.mean()


# In[99]:


df


# In[100]:


df


# In[ ]:





# In[101]:


df.head(10)


# In[ ]:





# In[102]:


df


# In[103]:


filt= (df['Average Temperature (C)'] == 'weekly average temperature' ) 
   
   


# In[104]:


df.loc[filt,'Monthly Average Temperature (C)']


# In[105]:


filt= (df['Average Humidity'] == 'Monthly Average Humidity')
   
   


# In[106]:


df.loc[filt,'Weekly Average Humidity']


# In[107]:


pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)


# In[108]:


df.head()


# In[109]:


df.fillna(0, inplace=True)
   


# In[110]:


df.head()


# In[111]:


weather=(df['Monthly Average Temperature (C)'] < 60)


# In[ ]:





# In[112]:


filt= (df['Average Wind Speed'] == 'Monthly Average wind speed')
   


# In[113]:


df.loc[filt,'Weekly Average Wind Speed']


# In[114]:


filt


# In[115]:


df.dtypes


# In[116]:


print(df['Monthly Average Humidity'].unique())


# In[117]:


df.describe()


# In[118]:


df.isnull().sum()


# In[119]:


df.dtypes


# In[120]:


fig = plt.figure(figsize = (20,15))
ax = fig.gca()
df.hist(bins=50, ax = ax)


# In[ ]:





# In[121]:




# Correlation between the data Dimensionss 

import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
corrMatrix = df.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corrMatrix, annot=True,)


# In[122]:



#Data is cleaned and ready for Regression analysis
#Spliting the data 
y = df['Low Temperature (C)'].values
x = df.drop('Low Temperature (C)' , axis = 1).values
x = df[['Low Humidity','High Humidity','Average Humidity']].values
x.reshape(1, -1)
y.reshape(1, -1)
x.shape
len(x)


# In[123]:



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
le=preprocessing.LabelEncoder()
le.fit(["Date","Low Temperature (C)","High Temperature (C)","Average Temperature (C)","weekly average temperature","Monthly Average Temperature (C)","Low Humidity","High Humidity","Average Humidity","Weekly Average Humidity","Monthly Average Humidity","Wind Speed low","Wind Speed High","Average Wind Speed","Weekly Average Wind Speed","Monthly Average wind speed"])
le.fit(["Low Temperature (C)","Low Humidity","High Humidity","Average Humidity"])
labelencoder = LabelEncoder()
df["weather_transform"]= labelencoder.fit_transform(df["weather"])
df["weather"]=df["weather"].astype('category') 
df.dtypes
df["weather"]=df["weather"].cat.codes 
list(le.classes_)



# In[124]:


import scipy.stats as st
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split

X_train , X_test , Y_train , Y_test = train_test_split(x,y, test_size = 0.3 , random_state = 0)

lnr = LinearRegression().fit(X_train,Y_train)
len(Y_train)

Y_train.shape

X_train
df.head()
X_test.shape
X_test.reshape(1, -1)
Y_test.shape

#X_train = X_train.astype(dtype, casting="unsafe", copy=False)
lnr.fit(X_train,Y_train)

train_score = lnr.score(X_train, Y_train)
test_score = lnr.score(X_test, Y_test)


# In[125]:


train_score


# In[126]:


test_score


# In[127]:


from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
le=preprocessing.LabelEncoder()
le.fit(["Date","Low Temperature (C)","High Temperature (C)","Average Temperature (C)","weekly average temperature","Monthly Average Temperature (C)","Low Humidity","High Humidity","Average Humidity","Weekly Average Humidity","Monthly Average Humidity","Wind Speed low","Wind Speed High","Average Wind Speed","Weekly Average Wind Speed","Monthly Average wind speed"])
le.fit(["Low Temperature (C)","Low Humidity","High Humidity","Average Humidity"])
labelencoder = LabelEncoder()
df["weather_transform"]= labelencoder.fit_transform(df["weather"])
df["weather"]=df["weather"].astype('category') 
df.dtypes
df["weather"]=df["weather"].cat.codes 
list(le.classes_)


# In[162]:


import scipy.stats as st
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train , X_test , Y_train , Y_test = train_test_split(x,y, test_size = 0.4 , random_state = 0)

lnr = LogisticRegression().fit(X_train,Y_train)
len(Y_train)
lnr.predict(X_test [0].reshape(1,-1))
Y_train.shape

X_train
df.head()
X_test.shape
X_test.reshape(1, -1)
Y_test.shape

#X_train = X_train.astype(dtype, casting="unsafe", copy=False)
lnr.fit(X_train,Y_train)

train_score = lnr.score(X_train, Y_train)
test_score = lnr.score(X_test, Y_test)


# In[163]:


train_score


# In[164]:


test_score


# In[131]:


df


# In[132]:



## Feature Interactions:
plt.figure(figsize = (14,10))
mask = np.triu(np.ones_like(df.corr()))
sns.heatmap(df.corr(),cmap="RdYlGn",mask = mask,annot=True)
plt.show()


# In[133]:


print(df.apply(lambda x: x.nunique()))
df.describe().T.style.background_gradient(
    vmin=-1, vmax=1, cmap=sns.color_palette("vlag", as_cmap=True))


# In[134]:




def annotate_plot(plots):
    for bar in plots.patches:
        plots.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')


# In[135]:


df


# In[136]:


import pandas as pd 
plt.figure(figsize=(16,16))
unique_df = df.apply(lambda x: x.nunique())


# In[137]:


unique_df.drop(columns = ['Date'], inplace=True)  
#unique_df.drop("Date",inplace =True)
unique_df = unique_df.reset_index()
unique_df.columns = ["col","values"]
g = sns.barplot(x="col",data = unique_df, y= "values")
annotate_plot(g)
plt.xticks(rotation=90)
plt.title("Unique Value Count")
plt.tight_layout()
plt.show()


# In[138]:


plt.figure(figsize=(16,16))
g = sns.countplot(df.weather)
annotate_plot(g)
plt.xticks(rotation=60)
plt.show()


# In[139]:


#import pandas as pd 
#df= pd.df[['Date']]


# In[ ]:





# In[141]:


df


# In[142]:


df["Average Temperature (C)"].resample('y').mean().plot(kind = "bar") 
#just a demonestration of yearly average temperature


# In[143]:


df["Average Temperature (C)"].resample('q').mean().plot(kind = "bar") 
#just a demonestration of quartile average temperature


# In[144]:


df["Average Temperature (C)"].resample('w').mean().plot(kind = "bar") 
#just a demonestration of quartile average temperature


# In[145]:


df.rename({'Average Temperature (C)': 'AverageTemperatureC'}, axis=1, inplace=True)


# In[146]:


df["2014-01-01":"2014-03-01"].AverageTemperatureC.resample('w').mean().plot(kind = "bar")
#3 month weekly temperature average 


# In[147]:


#df["2014-01-01":"2014-03-01"].AverageTemperatureC.resample('d').plot(kind = "bar")
#3 month weekly temperature average 


# In[148]:


df


# In[149]:


df.index


# In[150]:


#df = pd.df['Date']


# In[151]:


#pd.plot.bar(x='Date', y=['Low Temperature (C)', 'High Temperature (C)',], color={ "Low Temperature (C)": "blue", "High Temperature (C)": "red"})


# In[152]:


df.plot.bar(x='High Temperature (C)',y='Low Temperature (C)',rot=0);


# In[153]:


df["2014-01-01":"2014-02-01"].plot.bar(x='AverageTemperatureC', y=['Low Temperature (C)', 'High Temperature (C)',], color={ "Low Temperature (C)": "blue", "High Temperature (C)": "red"})


# In[154]:


#df["2014-01-01":"2014-02-01"].plot.bar(x='Date', y=['Low Temperature (C)', 'High Temperature (C)',], color={ "Low Temperature (C)": "blue", "High Temperature (C)": "red"})


# In[155]:


#d_parser = lambda x: pd.datetime.strptime(x,'')
#df = pd.read_csv('Weather_forecast.csv', parse_dates=['Date'],date_parser = d_parser)


# In[156]:


df.index


# In[ ]:





# In[ ]:


df.index


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


fig, ax = plt.subplots()


# In[ ]:


#ax.plot(df.index, df['Low Temperature (C)'])
#ax.set_xlabel("Date")
#ax.set_ylabel("Low Temperature (C)")


# In[ ]:


#plt.show()


# In[ ]:


#df["2014-01-01":"2014-02-01"].plot.bar(x='Date', y=['Low Temperature (C)', 'High Temperature (C)',], color={ "Low Temperature (C)": "blue", "High Temperature (C)": "red"})


# In[ ]:


#import seaborn as sns


# In[ ]:


#sns.barplot(x="Date", y="Low Temperature (C)", hue="", data=)


# In[ ]:



df.insert(0,'date',range(0,len(df)))


# df.plot(y='High Temperature (C)',x='Date')

# In[ ]:


d = np.polyfit(df['date'],df['High Temperature (C)'],1)
f = np.poly1d(d)
#df.insert(6,'Rain',f(df['weather']))
ax = df.plot(x = 'AverageTemperatureC',y='High Temperature (C)')
df.plot(x='AverageTemperatureC', y='Rain',color='Red',ax=ax)


# In[ ]:


ax=df.plot.scatter(x='AverageTemperatureC', y='Weekly Average Wind Speed')
df.plot(x = 'AverageTemperatureC',y='weather',color='Red',legend=False,ax=ax)


# In[179]:


import pandas
from sklearn import linear_model

X = df[['Weekly Average Humidity','Weekly Average Wind Speed','AverageTemperatureC']]
y = df['weather']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predicted = regr.predict([[0,0,62]])

print(predicted)


# In[177]:


df.head(10)


# In[161]:





# In[ ]:





# In[ ]:




