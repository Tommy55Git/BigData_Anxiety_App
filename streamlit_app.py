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
        cluster_data = list(db['modelação/clusters'].find())  # <- NEW COLLECTION
        
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

elif page == "Visualizations":
    st.header("🌍 Mapa Global: Ansiedade e Estilo de Vida")

    # Cópia de trabalho segura do DataFrame de clusters
    df_map = df_clusters.copy()

    # Criar coluna 'Country' se não existir, a partir de colunas binárias 'Country_XXX'
    if 'Country' not in df_map.columns:
        country_cols = [col for col in df_map.columns if col.startswith('Country_')]
        if country_cols:
            df_map['Country'] = ''
            for col in country_cols:
                country_name = col.replace('Country_', '')
                # Define o país onde o valor é 1 para essa coluna binária
                df_map.loc[df_map[col] == 1, 'Country'] = country_name

    # Remover linhas sem país válido
    df_map = df_map[df_map['Country'].notna() & (df_map['Country'] != '')]

    # Verificar se todas as colunas necessárias existem no DataFrame
    required_cols = [
        'Country', 'Anxiety Level (1-10)', 'Sleep_Stress_Ratio',
        'Therapy Sessions (per month)', 'Work_Exercise_Ratio'
    ]

    if all(col in df_map.columns for col in required_cols):
        # Criar o gráfico geográfico de dispersão
        fig = px.scatter_geo(
            df_map,
            locations="Country",
            locationmode="country names",
            color="Anxiety Level (1-10)",
            size="Therapy Sessions (per month)",
            hover_name="Country",
            hover_data={
                "Anxiety Level (1-10)": ':.2f',
                "Work_Exercise_Ratio": ':.2f',
                "Therapy Sessions (per month)": ':.2f',
                "Sleep_Stress_Ratio": ':.2f'
            },
            size_max=40,
            color_continuous_scale="Reds",
            title="🌍 Ansiedade Média, Terapia e Estilo de Vida por País"
        )

        # Ajustes no layout para mostrar contornos dos países e linhas costeiras
        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            geo=dict(
                showframe=False,
                showcoastlines=True,
                showcountries=True,
                projection_type="natural earth"
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("❗ Algumas colunas obrigatórias estão ausentes no dataset. Verifique seu arquivo de dados.")


    if not df_inner.empty:
        tab1, tab2, tab3 = st.tabs(["📊 Sociodemográficos", "🧠 Psicológicos", "🏃 Estilo de Vida"])

        # --- Sociodemográficos ---
        with tab1:
            st.subheader("Variáveis Sociodemográficas")

            # Preparar colunas e dados para filtros dinâmicos
            df = df_inner.copy()  # trabalhar em uma cópia para evitar alterações globais

            # Criar coluna 'Country' a partir das dummies
            country_cols = [c for c in df.columns if c.startswith('Country_')]
            if country_cols:
                df['Country'] = ''
                for c in country_cols:
                    df.loc[df[c] == 1, 'Country'] = c.replace('Country_', '')
            else:
                df['Country'] = 'Unknown'

            # Criar coluna 'Occupation' a partir das dummies
            occupation_cols = [
                'Occupation_Artist', 'Occupation_Athlete', 'Occupation_Chef', 'Occupation_Doctor',
                'Occupation_Engineer', 'Occupation_Freelancer', 'Occupation_Lawyer', 'Occupation_Musician',
                'Occupation_Nurse', 'Occupation_Other', 'Occupation_Scientist', 'Occupation_Student',
                'Occupation_Teacher'
            ]
            occupation_cols = [c for c in occupation_cols if c in df.columns]
            if occupation_cols:
                def get_occupation(row):
                    for col in occupation_cols:
                        if row[col] == 1:
                            return col.replace('Occupation_', '')
                    return 'Unknown'
                df['Occupation'] = df.apply(get_occupation, axis=1)
            else:
                df['Occupation'] = 'Unknown'

            # Criar coluna 'Gender'
            if all(x in df.columns for x in ['Gender_Female', 'Gender_Male', 'Gender_Other']):
                def get_gender(row):
                    if row['Gender_Female'] == 1:
                        return 'Female'
                    elif row['Gender_Male'] == 1:
                        return 'Male'
                    elif row['Gender_Other'] == 1:
                        return 'Other'
                    else:
                        return 'Unknown'
                df['Gender'] = df.apply(get_gender, axis=1)
            else:
                df['Gender'] = 'Unknown'

            # Criar faixa etária
            bins = [10, 20, 30, 40, 50, 60, 70, 80]
            labels = ['10–19', '20–29', '30–39', '40–49', '50–59', '60–69', '70+']
            df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

            # Filtros do usuário
            filter_type = st.selectbox("Filtrar por:", options=["Nenhum", "País", "Ocupação", "Gênero", "Faixa Etária"])

            if filter_type == "País":
                options = sorted(df['Country'].unique())
                selected = st.selectbox("Escolha o país:", options=options)
                filtered_df = df[df['Country'] == selected]
            elif filter_type == "Ocupação":
                options = sorted(df['Occupation'].unique())
                selected = st.selectbox("Escolha a ocupação:", options=options)
                filtered_df = df[df['Occupation'] == selected]
            elif filter_type == "Gênero":
                options = sorted(df['Gender'].unique())
                selected = st.selectbox("Escolha o gênero:", options=options)
                filtered_df = df[df['Gender'] == selected]
            elif filter_type == "Faixa Etária":
                options = sorted(df['Age Group'].dropna().unique())
                selected = st.selectbox("Escolha a faixa etária:", options=options)
                filtered_df = df[df['Age Group'] == selected]
            else:
                filtered_df = df

            # Agora gráficos com filtered_df
            st.write("Distribuição de nível de ansiedade por variáveis sociodemográficas:")
            for col in ['Age', 'Gender', 'Education Level', 'Employment Status', 'Income', 'Country', 'Occupation', 'Age Group']:
                if col in filtered_df.columns:
                    fig = px.histogram(
                        filtered_df,
                        x=col,
                        color='Anxiety Level (1-10)' if 'Anxiety Level (1-10)' in filtered_df.columns else None,
                        title=f"{col} vs Nível de Ansiedade",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # --- Psicológicos ---
        with tab2:
            st.subheader("Variáveis Psicológicas")
            psychological_cols = ['Mental Health Condition_Anxiety', 'Mental Health Condition_Depression', 
                                  'Mental Health Condition_Bipolar', 'Therapy', 'Self Esteem']
            for col in psychological_cols:
                if col in df_inner.columns:
                    fig = px.histogram(df_inner, x=col, color='Anxiety Level (1-10)',
                                       title=f"{col} vs Nível de Ansiedade", barmode="group")
                    st.plotly_chart(fig, use_container_width=True)

        # --- Estilo de Vida ---
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
