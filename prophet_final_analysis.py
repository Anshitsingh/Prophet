
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from fbprophet import Prophet


# In[2]:


a=pd.read_excel("/home/jash/Desktop/prophet/forecast.xlsx")


# In[ ]:


a.plot(x='date', y='Policycount', style='o')
a.rename(columns={'date': 'ds', 'Policycount': 'y'}, inplace = True)
last3data=a.iloc[-90:]
print last3data.head()


# In[4]:


def forecasting(cps,period,dataframe):
    from fbprophet.plot import add_changepoints_to_plot
    changed=Prophet(changepoint_prior_scale=cps)
    changed.fit(dataframe)
    future=changed.make_future_dataframe(periods=period)
    forecast=changed.predict(future)
    fig=changed.plot(forecast)
    p=add_changepoints_to_plot(fig.gca(),changed,forecast)
    return changed

changed1=forecasting(10,180,a)
changed2=forecasting(.05,180,a)



# In[5]:


print a.iloc[1790]


# In[6]:


def plot_comp(dataframe,cps,period):
    m=Prophet(changepoint_prior_scale=cps)
    m.fit(dataframe)
    future=m.make_future_dataframe(periods=period)
    forecast=m.predict(future)
    fig = m.plot_components(forecast)

plot_comp(a,.05,180)


# In[7]:


for i in range(len(a)):
    if(a.iloc[i]['y']>280):
        a.iloc[i]['y']=220
print a.iloc[1790]


# In[8]:


def simpleforecast(cps,period,dataframe):
    changed=Prophet(changepoint_prior_scale=cps)
    changed.fit(dataframe)
    future=changed.make_future_dataframe(periods=period)
    forecast=changed.predict(future)
    fig=changed.plot(forecast)
    return changed

changed=simpleforecast(.05,180,a)


# In[9]:


def simpleforecastwithoutplot(cps,period,dataframe):
    changed=Prophet(changepoint_prior_scale=cps)
    changed.fit(dataframe)
    future=changed.make_future_dataframe(periods=period)
    forecast=changed.predict(future)
    return changed

changed=simpleforecastwithoutplot(.05,180,a)


# In[10]:


from fbprophet.diagnostics import cross_validation

def last6monthscrossval(dataframe,init,per,hor,cps,period):
    m=simpleforecastwithoutplot(cps,period,dataframe)
    df_cv = cross_validation(m, initial=str(init)+'days', period=str(per)+'days', horizon =str(hor)+'days')
    df_cv.tail()
    last6=df_cv.iloc[-181:]
    last6 = last6.reset_index(drop=True)
    return last6

last6=last6monthscrossval(a,365,90,90,.05,180)


# In[11]:


#calculating mape for the last 6 months

def last6averagemape(last6):
    mape=np.zeros(shape=(len(last6),1))
    for i in range(len(last6)):
        mape[i]=math.fabs((last6.iloc[i]['yhat']-last6.loc[i]['y'])/last6.iloc[i]['y'])
    return np.mean(mape)*100 
    
t=last6averagemape(last6)


# In[12]:


#investigating 

from fbprophet.diagnostics import cross_validation

def first3monthscrossval(dataframe,init,per,hor,cps,period):
    m=simpleforecastwithoutplot(cps,period,dataframe)
    df_cv = cross_validation(m, initial=str(init)+'days', period=str(per)+'days', horizon =str(hor)+'days')
    df_cv.tail()
    last6=df_cv.iloc[-181:-91]
    last6 = last6.reset_index(drop=True)
    return last6

def last3monthscrossval(dataframe,init,per,hor,cps,period):
    m=simpleforecastwithoutplot(cps,period,dataframe)
    df_cv = cross_validation(m, initial=str(init)+'days', period=str(per)+'days', horizon =str(hor)+'days')
    df_cv.tail()
    last6=df_cv.iloc[-90:]
    last6 = last6.reset_index(drop=True)
    return last6

f3=first3monthscrossval(a,365,90,90,.5,90)
l3=last3monthscrossval(a,365,90,90,.5,90)

print f3.head()
print l3.head()


# In[13]:


def plotdaily(dataframe):
    last6=dataframe
    fig=plt.figure(figsize=(70,40))
    plt.plot(last6['ds'],last6['yhat'],color='red',linewidth=7)
    plt.plot(last6['ds'],last6['y'],color='blue',linewidth=7)


plotdaily(l3)


# In[14]:


#estimating peaks points of high error
i=0
for i in range(len(l3)):
    if(l3.iloc[i]['y']-l3.iloc[i]['yhat']>30):
        print "date is",l3.iloc[i]['ds'],"yhat is",l3.iloc[i]['yhat'],"y is",l3.iloc[i]['y']


# In[15]:


#estimating dates of low error and y is more than yhat
i=0
for i in range(len(l3)):
    if((l3.iloc[i]['y']-l3.iloc[i]['yhat']<10)and(l3.iloc[i]['y']-l3.iloc[i]['yhat']>0)):
        print "date is",l3.iloc[i]['ds'],"yhat is",l3.iloc[i]['yhat'],"y is",l3.iloc[i]['y']


# In[16]:


#estimating peaks points of high error
i=0
flag=1
for i in range(len(l3)):
    if(l3.iloc[i]['y']-l3.iloc[i]['yhat']>30):
        if(flag==1):
            print "\n"
        flag=0
        print "date is",l3.iloc[i]['ds'],"   high error"
        
    if((l3.iloc[i]['y']-l3.iloc[i]['yhat']<15)and(l3.iloc[i]['y']-l3.iloc[i]['yhat']>0)):
        if(flag==0):
            print "\n"
        flag=1
        print "date is",l3.iloc[i]['ds'],"   less error"


# In[17]:


plot_comp(last3data,.5,90)


# In[43]:


from fbprophet.plot import plot_weekly
m=Prophet(changepoint_prior_scale=.05)
m.fit(a)
future=m.make_future_dataframe(periods=90)
forecast=m.predict(future)
k=Prophet(changepoint_prior_scale=.05)
k.fit(last3data)
future=k.make_future_dataframe(periods=90)
forecast=k.predict(future)
plot_weekly(m)
plot_weekly(k)




# In[18]:


f, (ax1, ax2) = plt.subplots(2, sharex=True,figsize=(90,60))
f.suptitle('Sharing Y axis')

ax1.plot(l3['ds'],l3['yhat'],color='red',linewidth=7)
ax1.plot(l3['ds'],l3['y'],color='blue',linewidth=7)
ax2.plot(f3['ds'],f3['yhat'],color='red',linewidth=7)
ax2.plot(f3['ds'],f3['y'],color='blue',linewidth=7)


# In[19]:



def last6averagemape(last6):
    mape=np.zeros(shape=(len(last6),1))
    for i in range(len(last6)):
        mape[i]=math.fabs((last6.iloc[i]['yhat']-last6.loc[i]['y'])/last6.iloc[i]['y'])
    return np.mean(mape)*100 
    
first3=last6averagemape(f3)
last3=last6averagemape(l3)
print "mape for the first 3 months of the year is ",first3
print "mape for the last3 months of the year is",last3


# In[20]:


#finding minimum mape for various cps values
def findoptcps(dataframe,init,per,hor):
    k=40
    mapes=np.zeros(shape=(10,1))
    k_values=np.zeros(shape=(10,1))
    while(k<90):        
        last6=last6monthscrossval(a,365,180,180,k,180)
        i=int((k-40)/5)
        mapes[i]=last6averagemape(last6)
        k_values[i]=k
        print "k=",k,"done","mape value is ",mapes[i]
        k=k+5
    return mapes,k_values    
    

mapes,k_values=findoptcps(a,730,180,180)
        
    


# In[21]:


#finding minimum values among all these
plt.plot(k_values,mapes)
min_index=np.argmin(mapes)
print "min k is",k_values[min_index][0]
print "minimum mape is",mapes[min_index][0],"for changepoint prior scale =",k_values[min_index][0]


# In[34]:



last6=last6monthscrossval(a,365,180,180,.01,180)
def plotdaily(dataframe):
    last6=dataframe
    fig=plt.figure(figsize=(70,40))
    plt.plot(last6['ds'],last6['yhat'],color='red',linewidth=7)
    plt.plot(last6['ds'],last6['y'],color='blue',linewidth=7)
#trying different cps values
last6=last6monthscrossval(a,365,180,180,.1,180)

#plotting optimum mape value
last6opt=last6monthscrossval(a,365,180,180,55,180)

plotdaily(a)


# In[ ]:


#Plotting optimum mape curve and other curves
last6=last6monthscrossval(a,365,180,180,.1,180)
f, (ax1, ax2) = plt.subplots(2, sharex=True,figsize=(90,60))
f.suptitle('Sharing Y axis')

ax1.plot(last6['ds'],last6['yhat'],color='red',linewidth=7)
ax1.plot(last6['ds'],last6['y'],color='blue',linewidth=7)
ax2.plot(last6opt['ds'],last6opt['yhat'],color='red',linewidth=7)
ax2.plot(last6opt['ds'],last6opt['y'],color='blue',linewidth=7)


# In[ ]:


from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(last6, metric='mape')


# In[ ]:


#weekly
def weekly(last6): 
    i=0
    k=0
    mape=pd.DataFrame(columns=['yhat','y'])
    while (i<len(last6)):
        s=last6.iloc[i:i+7]
        yhat=np.sum(s['yhat'])
        y=np.sum(s['y'])
        last6.loc[i:i+7,'weeklyyhat']=yhat
        last6.loc[i:i+7,'weeklyy']=y
        mape.loc[k,'yhat']=yhat
        mape.loc[k,'y']=y
        k=k+1
        i=i+7

    fig=plt.figure(figsize=(70,40))

    plt.plot(mape.index,mape['yhat'],color='red',linewidth=7)
    plt.plot(mape.index,mape['y'],color='blue',linewidth=7) 
    
weekly(last6)



# In[ ]:


#monthly

def monthly(last6):    
    i=0
    k=0
    mape=pd.DataFrame(columns=['yhat','y'])
    while (i<len(last6)):
        s=last6.iloc[i:i+30]
        yhat=np.sum(s['yhat'])
        y=np.sum(s['y'])
        last6.loc[i:i+30,'weeklyyhat']=yhat
        last6.loc[i:i+30,'weeklyy']=y
        mape.loc[k,'yhat']=yhat
        mape.loc[k,'y']=y
        k=k+1
        i=i+30

    fig=plt.figure(figsize=(70,40))
    months=pd.Series(['Jan','Feb','March','April','May','June'])
    mape.reindex(index=months)
    plt.plot(mape.index,mape['yhat'],color='red',linewidth=7)
    plt.plot(mape.index,mape['y'],color='blue',linewidth=7)    


monthly(last6)    

