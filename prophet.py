
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


a=pd.read_excel("/home/jash/Desktop/prophet/forecast.xlsx")


# In[3]:


a.plot(x='date', y='Policycount', style='o')
a.rename(columns={'date': 'ds', 'Policycount': 'y'}, inplace = True)


# In[4]:


from fbprophet import Prophet
i=Prophet()
i.fit(a)
future=i.make_future_dataframe(periods=180)
forecast1=i.predict(future)
fig1=i.plot(forecast1)


# In[5]:


from fbprophet.plot import add_changepoints_to_plot
changed=Prophet(changepoint_prior_scale=100)
changed.fit(a)
future=changed.make_future_dataframe(periods=180)
forecast2=changed.predict(future)
fig=changed.plot(forecast2)
p=add_changepoints_to_plot(fig.gca(),changed,forecast2)


from fbprophet.plot import add_changepoints_to_plot
m=Prophet()
m.fit(a)
future=m.make_future_dataframe(periods=180)
forecast3=m.predict(future)
fig1=m.plot(forecast3)
q=add_changepoints_to_plot(fig1.gca(),m,forecast3)


# In[9]:


from fbprophet.plot import plot_yearly
j=plot_yearly(m)


# In[11]:


k=Prophet(yearly_seasonality=13).fit(a)
j=plot_yearly(k)


# In[16]:


m2= Prophet()


forecast4 = m2.fit(a).predict(future)
fig = m2.plot_components(forecast4)


# In[33]:


a.loc[a.loc[:]['y'] > 280]=None


# In[35]:


g=Prophet()
model=g.fit(a)
fig=model.plot(model.predict(future))


# In[54]:


from fbprophet.diagnostics import cross_validation
df_cv = cross_validation(m, initial='360 days', period='180 days', horizon = '180 days')


# In[59]:


df_cv.tail()


# In[89]:


last6=df_cv.iloc[-181:]


# In[116]:


last6 = last6.reset_index(drop=True)
last6.head()


# In[117]:


#calculating mape for the last 6 months
import math
import numpy as np
mape=np.zeros(shape=(len(last6),1))
for i in range(len(last6)):
    mape[i]=math.fabs((last6.iloc[i]['yhat']-last6.loc[i]['y'])/last6.iloc[i]['y'])
    


# In[141]:


print "mape for the last 6 months is",np.mean(mape)*100,"percent"


# In[142]:


fig=plt.figure(figsize=(70,40))

plt.plot(last6['ds'],last6['yhat'],color='red',linewidth=7)
plt.plot(last6['ds'],last6['y'],color='blue',linewidth=7)


# In[143]:


from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(last6, metric='mape')

