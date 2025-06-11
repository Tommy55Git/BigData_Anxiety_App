#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install pymongo


# In[2]:


from pymongo import MongoClient  # type: ignore

uri = "mongodb+srv://teste:T08m10b033@projeto-bigdata.ydeu01v.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(uri)

db = client['Projeto_BD']
df_inner = db['df_inner']

doc = df_inner.find_one()
print("Documento encontrado:", doc)


# In[4]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import *

def load_data():
    spark = SparkSession.builder \
        .appName("Projeto_inner") \
        .getOrCreate()

    df = spark.read.format("mongodb")\
        .option("spark.mongodb.connection.uri", "mongodb+srv://teste:T08m10b033@projeto-bigdata.ydeu01v.mongodb.net/") \
        .option("spark.mongodb.write.connection.uri", "mongodb+srv://teste:T08m10b033@projeto-bigdata.ydeu01v.mongodb.net/Projeto_BD") \
        .option("spark.mongodb.database", "Projeto_BD")\
        .option("spark.mongodb.collection", "df_inner") \
        .load()

    df_inner = df
    df_inner = df_inner.toPandas()


# In[5]:


#df_inner = df_inner.toPandas()


# In[6]:


#df_inner.describe()


# In[7]:


#df_inner.head()


# In[9]:

def predict():
    import seaborn as sb
    import pandas as pd
    import matplotlib.pyplot as plt
    import sklearn as skl


    # In[11]:


    df_inner = df_inner.drop('_id', axis=1)


    # In[15]:


    corr = df_inner.corr()

    plt.figure(figsize=(100, 98))
    sb.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.show()


    # In[25]:


    sb.histplot(data=df_inner, x='Mental Health Condition_Bipolar', stat='percent', hue='Anxiety Level (1-10)',  multiple="dodge")


    # In[26]:


    sb.histplot(data=df_inner, x='Mental Health Condition_Bipolar', stat='percent', hue='Country_Germany',  multiple="dodge")


    # In[28]:


    sb.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sb.boxplot(x='Occupation_Doctor', y='Anxiety Level (1-10)', data=df_inner)
    plt.xticks(rotation=45)
    plt.title("Anxiety Levels by Occupation")
    plt.show()


    # In[29]:


    sb.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sb.boxplot(x='Occupation_Doctor', y='Caffeine Intake (mg/day)', data=df_inner)
    plt.xticks(rotation=45)
    plt.title("Anxiety Levels by Occupation")
    plt.show()


    # In[33]:


    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.metrics import accuracy_score


    # In[32]:


    x = df_inner.drop(columns=['Mental Health Condition_Anxiety'])
    y = df_inner['Mental Health Condition_Anxiety']

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state = 5)


    # In[34]:


    # Random Forests

    random_forest = RandomForestClassifier(n_estimators=8)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    acc


# In[ ]:




