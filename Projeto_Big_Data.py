#!/usr/bin/env python
# coding: utf-8

# In[8]:


#pip install pymongo


# In[11]:


from pymongo import MongoClient  # type: ignore

uri = "mongodb+srv://teste:T08m10b033@projeto-bigdata.ydeu01v.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(uri)

db = client['Projeto_BD']
anxiety = db['anxiety']
mental = db['mental_health']

doc = anxiety.find_one()
print("Documento encontrado:", doc)


# In[ ]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import *

def load_data1():
    spark = SparkSession.builder \
        .appName("Projeto") \
        .getOrCreate()

    df = spark.read.format("mongodb")\
        .option("spark.mongodb.connection.uri", "mongodb+srv://teste:T08m10b033@projeto-bigdata.ydeu01v.mongodb.net/") \
        .option("spark.mongodb.database", "Projeto_BD")\
        .option("spark.mongodb.collection", "anxiety") \
        .load()

    anxiety = df

    anxiety.show()



# In[15]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark1 = SparkSession.builder \
    .appName("Projeto2") \
    .getOrCreate()

df1 = spark1.read.format("mongodb")\
    .option("spark.mongodb.connection.uri", "mongodb+srv://teste:T08m10b033@projeto-bigdata.ydeu01v.mongodb.net/") \
    .option("spark.mongodb.database", "Projeto_BD")\
    .option("spark.mongodb.collection", "mental_health") \
    .load()

mental = df1

mental.show()



# In[6]:


import pandas as pd
import json

# Caminhos para os ficheiros
file_path_anxiety = '/data/anxiety.json'
file_path_mental_health = '/data/mental_health.json'

# Carrega os ficheiros JSON
with open(file_path_anxiety, 'r') as f:
    dataset_anxiety = json.load(f)

with open(file_path_mental_health, 'r') as f:
    dataset_mental_health = json.load(f)

# Converte para DataFrame
df_anxiety = pd.DataFrame(dataset_anxiety)
df_mental_health = pd.DataFrame(dataset_mental_health)

# Visualiza as primeiras linhas e estatísticas
print(df_anxiety.head())
print(df_anxiety.describe())

print(df_mental_health.head())
print(df_mental_health.describe())

