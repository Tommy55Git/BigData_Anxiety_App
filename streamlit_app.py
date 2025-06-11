import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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
        
        # Convert to DataFrames
        df_anxiety = pd.DataFrame(anxiety_data)
        df_mental = pd.DataFrame(mental_data)
        df_inner = pd.DataFrame(df_inner_data)
        
        # Remove MongoDB _id column if exists
        for df in [df_anxiety, df_mental, df_inner]:
            if '_id' in df.columns:
                df.drop('_id', axis=1, inplace=True)
        
        return df_anxiety, df_mental, df_inner
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Load data
df_anxiety, df_mental, df_inner = load_data_from_mongo()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Overview", "Visualizations", "Analysis"])

if page == "Data Overview":
    st.header("Data Overview")
    
    # Tabs for different datasets
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

elif page == "Visualizations":
    st.header("Data Visualizations")
    
    if not df_inner.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlation heatmap using plotly instead of seaborn
            st.subheader("Correlation Heatmap")
            numeric_cols = df_inner.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                corr = df_inner[numeric_cols].corr()
                fig = px.imshow(corr, 
                               text_auto=True, 
                               aspect="auto",
                               color_continuous_scale='RdBu_r',
                               title="Correlation Matrix")
                st.plotly_chart(fig)
        
        with col2:
            # Distribution plots
            st.subheader("Data Distributions")
            if 'Anxiety Level (1-10)' in df_inner.columns:
                fig = px.histogram(df_inner, x='Anxiety Level (1-10)', 
                                 title="Distribution of Anxiety Levels")
                st.plotly_chart(fig)
        
        # Additional visualizations
        st.subheader("Additional Charts")
        
        # Check for specific columns and create visualizations
        if 'Mental Health Condition_Bipolar' in df_inner.columns and 'Anxiety Level (1-10)' in df_inner.columns:
            fig = px.histogram(df_inner, x='Mental Health Condition_Bipolar', 
                             color='Anxiety Level (1-10)', 
                             title="Bipolar Condition vs Anxiety Level")
            st.plotly_chart(fig)
        
        if 'Caffeine Intake (mg/day)' in df_inner.columns:
            fig = px.histogram(df_inner, x='Caffeine Intake (mg/day)', 
                             title="Caffeine Intake Distribution")
            st.plotly_chart(fig)
    
    else:
        st.warning("No data available for visualizations")

elif page == "Analysis":
    st.header("Data Analysis & Insights")
    
    if not df_inner.empty:
        # Basic statistics
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
        
        # Data insights
        st.subheader("Data Insights")
        
        if 'Anxiety Level (1-10)' in df_inner.columns:
            anxiety_stats = df_inner['Anxiety Level (1-10)'].describe()
            st.write("**Anxiety Level Statistics:**")
            st.write(anxiety_stats)
        
        # Show column information
        st.subheader("Dataset Information")
        st.write("**Available Columns:**")
        st.write(list(df_inner.columns))
        
        # Data quality
        st.subheader("Data Quality")
        missing_data = df_inner.isnull().sum()
        if missing_data.sum() > 0:
            st.write("**Missing Values:**")
            st.write(missing_data[missing_data > 0])
        else:
            st.write("✅ No missing values found!")
    
    else:
        st.warning("No data available for analysis")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Mental Health Data Dashboard - Built with Streamlit")
