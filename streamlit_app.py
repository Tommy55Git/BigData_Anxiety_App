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
page = st.sidebar.selectbox(
    "Choose a page", 
    ["Data Overview", "Visualizations", "Analysis", "Cluster Analysis", "Regression Model"]
)


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
    st.header(" Mapa Global")

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
            title=" Ansiedade Média, Terapia e Estilo de Vida por País"
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
    with tab3:
     st.subheader("Variáveis de Estilo de Vida")

    # Trabalhar em cópia para segurança
    df = df_inner.copy()

    # Criar grupos para filtros
    df['Age Group'] = pd.cut(df['Age'], bins=[10, 20, 30, 40, 50, 60, 70, 80], labels=['10–19', '20–29', '30–39', '40–49', '50–59', '60–69', '70+'], right=False)
    df['Screen Time Group'] = pd.cut(df['Screen Time per Day (Hours)'], bins=[0, 2, 4, 6, 8, 24], labels=['0–2h', '2–4h', '4–6h', '6–8h', '8h+'], right=False)
    df['Activity Group'] = pd.cut(df['Physical Activity (hrs/week)'], bins=[0, 1, 3, 5, 7, 20], labels=['0–1h', '1–3h', '3–5h', '5–7h', '7h+'], right=False)
    df['Work Group'] = pd.cut(df['Work Hours per Week'], bins=[0, 10, 20, 30, 40, 50, 60, 80], labels=['0–9h', '10–19h', '20–29h', '30–39h', '40–49h', '50–59h', '60h+'], right=False)

    # Filtro do usuário
    filter_type = st.selectbox("Filtrar por:", options=["Nenhum", "Faixa Etária", "Tempo de Tela", "Atividade Física", "Horas de Trabalho"])

    if filter_type == "Faixa Etária":
        options = sorted(df['Age Group'].dropna().unique())
        selected = st.selectbox("Escolha a faixa etária:", options=options)
        filtered_df = df[df['Age Group'] == selected]
    elif filter_type == "Tempo de Tela":
        options = sorted(df['Screen Time Group'].dropna().unique())
        selected = st.selectbox("Escolha o tempo de tela diário:", options=options)
        filtered_df = df[df['Screen Time Group'] == selected]
    elif filter_type == "Atividade Física":
        options = sorted(df['Activity Group'].dropna().unique())
        selected = st.selectbox("Escolha a atividade física semanal:", options=options)
        filtered_df = df[df['Activity Group'] == selected]
    elif filter_type == "Horas de Trabalho":
        options = sorted(df['Work Group'].dropna().unique())
        selected = st.selectbox("Escolha a carga horária semanal:", options=options)
        filtered_df = df[df['Work Group'] == selected]
    else:
        filtered_df = df

    # Gráficos com base no filtro
    st.write("Distribuição de nível de ansiedade por variáveis de estilo de vida:")
    for col in [
        'Sleep Duration (hours/day)', 'Exercise Frequency (days/week)',
        'Social Media Usage (hours/day)', 'Physical Activity (hrs/week)',
        'Work Hours per Week', 'Screen Time per Day (Hours)',
        'Age Group', 'Screen Time Group', 'Activity Group', 'Work Group'
    ]:
        if col in filtered_df.columns:
            fig = px.histogram(
                filtered_df,
                x=col,
                color='Anxiety Level (1-10)',
                title=f"{col} vs Nível de Ansiedade",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Heatmap fora dos filtros
    st.markdown("### Consumo de Cafeína e Cigarros vs Ansiedade (Sem Filtros)")

    if 'Caffeine Intake (mg/day)' in df.columns and 'Smoking_Yes' in df.columns:
        df['Caffeine_bin'] = pd.cut(df['Caffeine Intake (mg/day)'], bins=30)
        heatmap_data = df.groupby(['Caffeine_bin', 'Smoking_Yes'])['Anxiety Level (1-10)'].mean().reset_index()
        heatmap_data['Caffeine_mid'] = heatmap_data['Caffeine_bin'].apply(lambda x: x.mid)
        heatmap_data['Smoking_Status'] = heatmap_data['Smoking_Yes'].map({0: 'Não Fuma', 1: 'Fuma'})

        fig_heat = px.density_heatmap(
            heatmap_data,
            x='Caffeine_mid',
            y='Smoking_Status',
            z='Anxiety Level (1-10)',
            color_continuous_scale='Reds',
            title='Consumo de Cafeína e Cigarros x Ansiedade'
        )
        st.plotly_chart(fig_heat, use_container_width=True)


    with tab2:
      st.subheader("Variáveis Psicológicas")

    # Cópia para não mexer no original
    df = df_clusters.copy()

    # Função para mapear gênero
    def get_gender(row):
        if row.get('Gender_Female', 0) == 1:
            return 'Female'
        elif row.get('Gender_Male', 0) == 1:
            return 'Male'
        elif row.get('Gender_Other', 0) == 1:
            return 'Other'
        else:
            return 'Unknown'
    df['Gender'] = df.apply(get_gender, axis=1)

    # Mapear condição mental
    conditions = [
        'Mental Health Condition_Anxiety',
        'Mental Health Condition_Bipolar',
        'Mental Health Condition_Depression',
        'Mental Health Condition_None',
        'Mental Health Condition_PTSD'
    ]
    def get_condition(row):
        for cond in conditions:
            if row.get(cond, 0) == 1:
                return cond.replace('Mental Health Condition_', '')
        return 'Unknown'
    df['Mental Health Condition'] = df.apply(get_condition, axis=1)

    # Faixas etárias
    bins = [10, 20, 30, 40, 50, 60, 70, 80]
    labels = ['10–19', '20–29', '30–39', '40–49', '50–59', '60–69', '70+']
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    # Mapear histórico familiar e eventos
    df['Family History Anxiety'] = df.get('Family History of Anxiety_Yes', 0).map({1: 'Yes', 0: 'No'})
    df['Life Event'] = df.get('Recent Major Life Event_Yes', 0).map({1: 'Yes', 0: 'No'})

    # --------------------- FILTROS ---------------------
    st.markdown("### Filtros")

    genders = df['Gender'].unique().tolist()
    conditions_list = df['Mental Health Condition'].unique().tolist()
    age_groups = df['Age Group'].dropna().unique().tolist()
    family_hist_options = df['Family History Anxiety'].unique().tolist()
    life_event_options = df['Life Event'].unique().tolist()

    selected_gender = st.multiselect("Filtrar por Gênero", genders, default=genders)
    selected_condition = st.multiselect("Filtrar por Condição Mental", conditions_list, default=conditions_list)
    selected_ages = st.multiselect("Filtrar por Faixa Etária", age_groups, default=age_groups)
    selected_family_history = st.multiselect("Histórico Familiar de Ansiedade", family_hist_options, default=family_hist_options)
    selected_life_event = st.multiselect("Teve Evento de Vida Recente?", life_event_options, default=life_event_options)

    # Aplicar filtros
    df = df[
        (df['Gender'].isin(selected_gender)) &
        (df['Mental Health Condition'].isin(selected_condition)) &
        (df['Age Group'].isin(selected_ages)) &
        (df['Family History Anxiety'].isin(selected_family_history)) &
        (df['Life Event'].isin(selected_life_event))
    ]

    # --------------------- GRÁFICOS ---------------------

    # 1) Ansiedade média por Faixa Etária, Gênero e Condição Mental
    df_grouped_1 = df.groupby(['Age Group', 'Gender', 'Mental Health Condition'])['Anxiety Level (1-10)'].mean().reset_index()
    fig1 = px.bar(
        df_grouped_1,
        x='Age Group',
        y='Anxiety Level (1-10)',
        color='Mental Health Condition',
        barmode='group',
        facet_col='Gender',
        category_orders={'Age Group': labels},
        labels={
            'Age Group': 'Faixa Etária',
            'Anxiety Level (1-10)': 'Nível Médio de Ansiedade',
            'Mental Health Condition': 'Condição Mental'
        },
        title='Ansiedade Média por Faixa Etária, Género e Condição Mental',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig1.update_layout(yaxis=dict(title='Nível Médio de Ansiedade'))
    st.plotly_chart(fig1, use_container_width=True)

    # 2) Influência do Histórico Familiar de Ansiedade
    df_grouped_2 = df.groupby(['Age Group', 'Gender', 'Family History Anxiety'])['Anxiety Level (1-10)'].mean().reset_index()
    fig2 = px.bar(
        df_grouped_2,
        x='Age Group',
        y='Anxiety Level (1-10)',
        color='Family History Anxiety',
        barmode='group',
        facet_col='Gender',
        category_orders={'Age Group': labels},
        labels={
            'Age Group': 'Faixa Etária',
            'Anxiety Level (1-10)': 'Nível Médio de Ansiedade',
            'Family History Anxiety': 'Histórico Familiar'
        },
        title='Influência do Histórico Familiar no Nível Médio de Ansiedade',
        color_discrete_map={'Yes': 'crimson', 'No': 'steelblue'}
    )
    fig2.update_layout(yaxis=dict(title='Nível Médio de Ansiedade'))
    st.plotly_chart(fig2, use_container_width=True)

    # 3) Ansiedade média por género entre quem teve evento de vida recente
    df_event = df[df['Life Event'] == 'Yes']
    if not df_event.empty:
        df_grouped_3 = df_event.groupby('Gender')['Anxiety Level (1-10)'].mean().reset_index()
        fig3 = px.bar(
            df_grouped_3,
            x='Gender',
            y='Anxiety Level (1-10)',
            color='Anxiety Level (1-10)',
            color_continuous_scale='Reds',
            labels={
                'Gender': 'Gênero',
                'Anxiety Level (1-10)': 'Nível Médio de Ansiedade'
            },
            title='Ansiedade Média por Género (com Evento de Vida Recente)'
        )
        fig3.update_layout(yaxis=dict(title='Nível Médio de Ansiedade'))
        st.plotly_chart(fig3, use_container_width=True)

   
  
    
       



        # --- Estilo de Vida ---
    with tab3:
     st.subheader("Variáveis de Estilo de Vida")

    # ➤ Filtros interativos
    age_filter = st.slider("Faixa Etária", min_value=10, max_value=70, value=(10, 70))
    screen_time_filter = st.slider("Tempo de Tela por Dia (horas)", 0, 24, (0, 24))
    activity_filter = st.slider("Atividade Física por Semana (horas)", 0, 20, (0, 20))
    work_filter = st.slider("Horas de Trabalho por Semana", 0, 80, (0, 80))

    df_filtered = df[
        (df['Age'] >= age_filter[0]) & (df['Age'] <= age_filter[1]) &
        (df['Screen Time per Day (Hours)'] >= screen_time_filter[0]) & (df['Screen Time per Day (Hours)'] <= screen_time_filter[1]) &
        (df['Physical Activity (hrs/week)'] >= activity_filter[0]) & (df['Physical Activity (hrs/week)'] <= activity_filter[1]) &
        (df['Work Hours per Week'] >= work_filter[0]) & (df['Work Hours per Week'] <= work_filter[1])
    ].copy()

    # ➤ Gráficos de Estilo de Vida
    lifestyle_cols = [
        'Sleep Duration (hours/day)',
        'Exercise Frequency (days/week)',
        'Social Media Usage (hours/day)'
    ]

    for col in lifestyle_cols:
        if col in df_filtered.columns:
            fig = px.histogram(
                df_filtered,
                x=col,
                color='Anxiety Level (1-10)',
                title=f"{col} vs Nível de Ansiedade (Filtrado)",
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ➤ Horas de Trabalho vs Ansiedade
    work_bins = [0, 10, 20, 30, 40, 50, 60, 80]
    work_labels = ['0–9h', '10–19h', '20–29h', '30–39h', '40–49h', '50–59h', '60+h']
    df_filtered['Work Group'] = pd.cut(df_filtered['Work Hours per Week'], bins=work_bins, labels=work_labels, right=False)

    work_anxiety = df_filtered.groupby('Work Group')['Anxiety Level (1-10)'].mean().reset_index()

    fig_work = px.bar(
        work_anxiety,
        x='Work Group',
        y='Anxiety Level (1-10)',
        color='Anxiety Level (1-10)',
        color_continuous_scale='Oranges',
        title='Horas de Trabalho por Semana vs Nível de Ansiedade (Filtrado)'
    )
    st.plotly_chart(fig_work, use_container_width=True)

    # ➤ Atividade Física vs Ansiedade
    activity_bins = [0, 1, 3, 5, 7, 20]
    activity_labels = ['0–1h', '1–3h', '3–5h', '5–7h', '7+h']
    df_filtered['Activity Group'] = pd.cut(df_filtered['Physical Activity (hrs/week)'], bins=activity_bins, labels=activity_labels, right=False)

    activity_anxiety = df_filtered.groupby('Activity Group')['Anxiety Level (1-10)'].mean().reset_index()

    fig_act = px.bar(
        activity_anxiety,
        x='Activity Group',
        y='Anxiety Level (1-10)',
        color='Anxiety Level (1-10)',
        color_continuous_scale='Greens',
        title='Atividade Física Semanal vs Nível de Ansiedade (Filtrado)'
    )
    st.plotly_chart(fig_act, use_container_width=True)

    # ➤ Idade vs Tempo de Tela vs Ansiedade
    age_bins = [10, 20, 30, 40, 50, 60, 70]
    age_labels = ['10–19', '20–29', '30–39', '40–49', '50–59', '60–69']
    screen_bins = [0, 2, 4, 6, 8, 24]
    screen_labels = ['0-2h', '2-4h', '4-6h', '6-8h', '8+h']

    df_filtered['Age Group'] = pd.cut(df_filtered['Age'], bins=age_bins, labels=age_labels, right=False)
    df_filtered['Screen Time Group'] = pd.cut(df_filtered['Screen Time per Day (Hours)'], bins=screen_bins, labels=screen_labels, right=False)

    df_screen = df_filtered.groupby(['Age Group', 'Screen Time Group'])['Anxiety Level (1-10)'].mean().reset_index()

    fig_screen = px.bar(
        df_screen,
        x='Age Group',
        y='Anxiety Level (1-10)',
        color='Screen Time Group',
        barmode='group',
        category_orders={'Age Group': age_labels, 'Screen Time Group': screen_labels},
        title='Nível Médio de Ansiedade por Idade e Tempo de Tela (Filtrado)'
    )
    st.plotly_chart(fig_screen, use_container_width=True)

    # ➤ Idade vs Atividade Física vs Ansiedade
    df_filtered['Physical Activity Group'] = pd.cut(df_filtered['Physical Activity (hrs/week)'], bins=activity_bins, labels=activity_labels, right=False)

    df_activity = df_filtered.groupby(['Age Group', 'Physical Activity Group'])['Anxiety Level (1-10)'].mean().reset_index()

    fig_activity = px.scatter(
        df_activity,
        x='Age Group',
        y='Physical Activity Group',
        size='Anxiety Level (1-10)',
        color='Anxiety Level (1-10)',
        category_orders={'Age Group': age_labels, 'Physical Activity Group': activity_labels},
        color_continuous_scale='Viridis',
        title='Idade, Atividade Física e Ansiedade (Filtrado)'
    )
    st.plotly_chart(fig_activity, use_container_width=True)

    # ➤ Heatmap: Cafeína x Fumar
    if 'Caffeine Intake (mg/day)' in df_filtered.columns and 'Smoking_Yes' in df_filtered.columns:
        df_filtered['Caffeine_bin'] = pd.cut(df_filtered['Caffeine Intake (mg/day)'], bins=30)
        heatmap_data = df_filtered.groupby(['Caffeine_bin', 'Smoking_Yes'])['Anxiety Level (1-10)'].mean().reset_index()
        heatmap_data['Caffeine_mid'] = heatmap_data['Caffeine_bin'].apply(lambda x: x.mid)
        heatmap_data['Smoking_Status'] = heatmap_data['Smoking_Yes'].map({0: 'Não Fuma', 1: 'Fuma'})

        fig_heat = px.density_heatmap(
            heatmap_data,
            x='Caffeine_mid',
            y='Smoking_Status',
            z='Anxiety Level (1-10)',
            color_continuous_scale='Reds',
            title='Consumo de Cafeína e Cigarros vs Ansiedade (Filtrado)'
        )
        st.plotly_chart(fig_heat, use_container_width=True)






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

# Nova página de Modelos de Regressão
elif page == "Regression Model":
    st.header("Modelos de Regressão para Predição da Ansiedade")

    if not df_inner.empty:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Selecionar colunas relevantes
        df_reg = df_inner[[
            "Age",
            "Sleep Hours",
            "Physical Activity (hrs/week)",
            "Diet Quality (1-10)",
            "Stress Level (1-10)",
            "Caffeine Intake (mg/day)",
            "Heart Rate (bpm)",
            "Breathing Rate (breaths/min)",
            "Anxiety Level (1-10)"
        ]].dropna()

        # Definir variáveis independentes e alvo
        X = df_reg.drop(columns=["Anxiety Level (1-10)"])
        y = df_reg["Anxiety Level (1-10)"]

        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modelos a comparar
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42),
            'k-NN': KNeighborsRegressor(),
            'SVR': SVR()
        }

        # Avaliar os modelos
        model_metrics = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            model_metrics.append({
                'Model': name,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            })

        metrics_df = pd.DataFrame(model_metrics)
        st.subheader("Métricas de Avaliação")
        st.dataframe(metrics_df)

        # Gráfico de comparação
        st.subheader("Comparação de Métricas entre Modelos")
        fig_metrics, ax_metrics = plt.subplots(figsize=(10, 6))
        metrics_df.set_index('Model')[['MAE', 'RMSE', 'R2']].plot(kind='bar', ax=ax_metrics, cmap='Set2')
        ax_metrics.set_title("Comparação dos Modelos de Regressão")
        ax_metrics.set_ylabel("Valor da Métrica")
        ax_metrics.grid(True)
        st.pyplot(fig_metrics)

        # Gráficos de dispersão com linha de regressão linear
        st.subheader("Dispersões Individuais vs Ansiedade")
        sns.set_style("whitegrid")
        ncols = len(X.columns)
        fig_disp, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(6 * ncols, 5))

        if ncols == 1:
            axes = [axes]

        for idx, col in enumerate(X.columns):
            sns.regplot(x=col, y='Anxiety Level (1-10)', data=df_reg, ax=axes[idx])
            axes[idx].set_title(f"{col} vs Ansiedade")

        st.pyplot(fig_disp)

        # Gráficos Real vs Predito por Modelo
        st.subheader("Real vs Predito por Modelo")
        fig_pred, axs = plt.subplots(2, 3, figsize=(18, 10))  # 2 linhas, 3 colunas
        axs = axs.flatten()

        for i, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test)
            axs[i].scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
            axs[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            axs[i].set_title(f"{name} - Real vs Predito")
            axs[i].set_xlabel("Valor Real")
            axs[i].set_ylabel("Valor Predito")

        for j in range(i + 1, len(axs)):
            fig_pred.delaxes(axs[j])

        fig_pred.suptitle("Comparação entre Valores Reais e Preditos por Modelo", fontsize=16)
        st.pyplot(fig_pred)

    else:
        st.warning("Dados insuficientes para regressão.")



# Footer
st.sidebar.markdown("---")
st.sidebar.info("Mental Health Data Dashboard - Built with Streamlit")
