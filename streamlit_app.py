import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Mental Health Data Analysis", layout="wide")

# Title
st.title("Mental Health & Anxiety Data Dashboard")

# MongoDB connection function
@st.cache_data
def load_data_from_mongo():
    try:
        uri = "mongodb+srv://teste:T08m10b033@projeto-bigdata.ydeu01v.mongodb.net/?retryWrites=true&w=majority"
        client = MongoClient(uri)
        db = client['Projeto_BD']
        
        # Load data from collections
        anxiety_data = list(db['anxiety'].find())
        mental_data = list(db['mental_health'].find())
        df_inner_data = list(db['df_inner'].find())
        cluster_data = list(db['clusters'].find())  # <- NEW COLLECTION
        
        # Convert to DataFrames
        df_anxiety = pd.DataFrame(anxiety_data)
        df_mental = pd.DataFrame(mental_data)
        df_inner = pd.DataFrame(df_inner_data)
        df_clusters = pd.DataFrame(cluster_data)  # <- NEW DF
        
        # Remove MongoDB _id column if exists
        for df in [df_anxiety, df_mental, df_inner, df_clusters]:
            if '_id' in df.columns:
                df.drop('_id', axis=1, inplace=True)
        
        return df_anxiety, df_mental, df_inner, df_clusters
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Load data
df_anxiety, df_mental, df_inner, df_clusters = load_data_from_mongo()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Overview", "Visualizations", "Analysis", "Cluster Analysis"])

# Page 1: Data Overview
if page == "Data Overview":
    st.header("Data Overview")
    tab1, tab2, tab3 = st.tabs(["Anxiety Data", "Mental Health Data", "Combined Data"])
    
    with tab1:
        if not df_anxiety.empty:
            st.subheader("Anxiety Dataset")
            st.write(f"Shape: {df_anxiety.shape}")
            st.dataframe(df_anxiety.head())
            st.write("Statistical Summary:")
            st.dataframe(df_anxiety.describe())
        else:
            st.warning("No anxiety data available")
    
    with tab2:
        if not df_mental.empty:
            st.subheader("Mental Health Dataset")
            st.write(f"Shape: {df_mental.shape}")
            st.dataframe(df_mental.head())
            st.write("Statistical Summary:")
            st.dataframe(df_mental.describe())
        else:
            st.warning("No mental health data available")
    
    with tab3:
        if not df_inner.empty:
            st.subheader("Combined Dataset")
            st.write(f"Shape: {df_inner.shape}")
            st.dataframe(df_inner.head())
            st.write("Statistical Summary:")
            st.dataframe(df_inner.describe())
        else:
            st.warning("No combined data available")

# Page 2: Visualizations
elif page == "Visualizations":
    st.header("Data Visualizations")

    if not df_inner.empty:
        # Criar abas para cada grupo de variáveis
        tab1, tab2, tab3 = st.tabs(["📊 Sociodemográficos", "🧠 Psicológicos", "🏃 Estilo de Vida"])

        with tab1:
            st.subheader("Variáveis Sociodemográficas")
            sociodemographic_cols = ['Age', 'Gender', 'Education Level', 'Employment Status', 'Income']
            for col in sociodemographic_cols:
                if col in df_inner.columns:
                    fig = px.histogram(df_inner, x=col, color='Anxiety Level (1-10)',
                                       title=f"{col} vs Nível de Ansiedade", barmode="group")
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Variáveis Psicológicas")
            psychological_cols = ['Mental Health Condition_Anxiety', 'Mental Health Condition_Depression', 
                                  'Mental Health Condition_Bipolar', 'Therapy', 'Self Esteem']
            for col in psychological_cols:
                if col in df_inner.columns:
                    fig = px.histogram(df_inner, x=col, color='Anxiety Level (1-10)',
                                       title=f"{col} vs Nível de Ansiedade", barmode="group")
                    st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Variáveis de Estilo de Vida")
            lifestyle_cols = ['Caffeine Intake (mg/day)', 'Smoking_Yes', 'Sleep Duration (hours/day)',
                              'Exercise Frequency (days/week)', 'Social Media Usage (hours/day)']
            for col in lifestyle_cols:
                if col in df_inner.columns:
                    fig = px.histogram(df_inner, x=col, color='Anxiety Level (1-10)',
                                       title=f"{col} vs Nível de Ansiedade", barmode="group")
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Nenhum dado disponível para visualização.")


# Page 3: Analysis
elif page == "Analysis":
    st.header("Data Analysis & Insights")
    
    if not df_inner.empty:
        st.subheader("Key Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df_inner))
        
        with col2:
            if 'Anxiety Level (1-10)' in df_inner.columns:
                avg_anxiety = df_inner['Anxiety Level (1-10)'].mean()
                st.metric("Average Anxiety Level", f"{avg_anxiety:.2f}")
        
        with col3:
            if 'Mental Health Condition_Anxiety' in df_inner.columns:
                anxiety_rate = df_inner['Mental Health Condition_Anxiety'].mean() * 100
                st.metric("Anxiety Condition Rate", f"{anxiety_rate:.1f}%")
        
        st.subheader("Data Insights")
        if 'Anxiety Level (1-10)' in df_inner.columns:
            st.write("**Anxiety Level Statistics:**")
            st.write(df_inner['Anxiety Level (1-10)'].describe())
        
        st.subheader("Dataset Information")
        st.write("**Available Columns:**")
        st.write(list(df_inner.columns))
        
        st.subheader("Data Quality")
        missing_data = df_inner.isnull().sum()
        if missing_data.sum() > 0:
            st.write("**Missing Values:**")
            st.write(missing_data[missing_data > 0])
        else:
            st.write("✅ No missing values found!")
    
    else:
        st.warning("No data available for analysis")

# Page 4: Cluster Analysis
elif page == "Cluster Analysis":
    st.header("Cluster Analysis")
    
    if not df_clusters.empty:
        st.write("### Clustered Dataset")
        st.dataframe(df_clusters.head())

        if {'PCA1', 'PCA2', 'Cluster'}.issubset(df_clusters.columns):
            fig = px.scatter(
                df_clusters, x='PCA1', y='PCA2', color='Cluster',
                title="Clusters (2D PCA)", labels={'Cluster': 'Grupo'}
            )
            st.plotly_chart(fig)

        if 'Anxiety Level (1-10)' in df_clusters.columns and 'Cluster' in df_clusters.columns:
            fig2 = px.box(
                df_clusters, x='Cluster', y='Anxiety Level (1-10)',
                title="Distribuição de Níveis de Ansiedade por Cluster"
            )
            st.plotly_chart(fig2)
    else:
        st.warning("Cluster data not found.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Mental Health Data Dashboard - Built with Streamlit")
