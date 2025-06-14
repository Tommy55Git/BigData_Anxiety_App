import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats

import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import coalesce, when, col, avg

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
    ["Dashboard", "Data Overview", "Visualizations", "Classification Model", "Regression Model", "Predict your Anxiety"]
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
        tab1, tab2, tab3 = st.tabs(["📊 Sociodemográficos", "🧠 Psicológicos", "Estilo de vida"])

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
        
        # GRÁFICO: Distribuição da Idade por Condição de Saúde Mental (PySpark + matplotlib)
        import pandas as pd
        import plotly.express as px
        import streamlit as st
        
        # df: pandas DataFrame já carregado
        condition_cols = [c for c in df.columns if c.startswith('Mental Health Condition_')]
        
        def get_condition(row):
            for col in condition_cols:
                if row[col] == 1:
                    return col.replace('Mental Health Condition_', '')
            return 'Unknown'
        
        df['Mental Health Condition'] = df.apply(get_condition, axis=1)
        
        # Agrupa por idade e condição
        df_grouped = (
            df.groupby(['Age', 'Mental Health Condition'])
            .size()
            .reset_index(name='count')
        )
        
        # Gráfico interativo com Plotly
        fig = px.bar(
            df_grouped,
            x='Age',
            y='count',
            color='Mental Health Condition',
            title="Distribuição da Idade por Condição de Saúde Mental",
            labels={'Age': 'Idade', 'count': 'Contagem', 'Mental Health Condition': 'Condição Mental'},
            barmode='group'
        )
        
        fig.update_layout(
            xaxis_title="Idade",
            yaxis_title="Contagem",
            legend_title="Condição Mental"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        



   
        


        

        

        
        
        
        
        
        with tab3:
            st.subheader("Estilo de Vida")
        
            # Criar coluna com nível de exercício (versão pandas)
            def get_exercise_level(row):
                if row.get("Exercise Level_Low", 0) == 1:
                    return "Low"
                elif row.get("Exercise Level_Moderate", 0) == 1:
                    return "Moderate"
                elif row.get("Exercise Level_High", 0) == 1:
                    return "High"
                else:
                    return "Unknown"
        
            df_exercise = df_clusters.copy()
            df_exercise["Exercise Level"] = df_exercise.apply(get_exercise_level, axis=1)
        
            # Criar dicionário com listas de ansiedade por nível de exercício
            exercise_levels = ["Low", "Moderate", "High"]
            ansiedade_por_nivel = {}
        
            for nivel in exercise_levels:
                valores = df_exercise[df_exercise["Exercise Level"] == nivel]["Anxiety Level (1-10)"].dropna().tolist()
                if len(valores) >= 10:
                    ansiedade_por_nivel[nivel] = valores
        
            # Criar gráfico interativo com Plotly
            import plotly.graph_objects as go
            import numpy as np
        
            fig = go.Figure()
        
            for nivel, valores in ansiedade_por_nivel.items():
                hist, bin_edges = np.histogram(valores, bins=30, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                fig.add_trace(go.Scatter(
                    x=bin_centers,
                    y=hist,
                    mode='lines',
                    name=nivel,
                    fill='tozeroy'
                ))
        
            fig.update_layout(
                title="Distribuição do Nível de Ansiedade por Nível de Exercício",
                xaxis_title="Nível de Ansiedade (1-10)",
                yaxis_title="Densidade Aproximada",
                template="plotly_white",
                hovermode="x unified"
            )
        
            st.plotly_chart(fig, use_container_width=True)
        




        import plotly.express as px
        import plotly.graph_objects as go
        
        # --- Gráfico 2: Média de Ansiedade por Tipo de Dieta ---
        
        # Identificar colunas de dieta
        diet_columns = [c for c in df_clusters.columns if c.startswith("Diet Type_")]
        
        # Função para extrair tipo de dieta
        def get_diet_type(row):
            for col in diet_columns:
                if row.get(col, 0) == 1:
                    return col.replace("Diet Type_", "")
            return "Desconhecida"
        
        # Aplicar mapeamento
        df_diet = df_clusters.copy()
        df_diet["Tipo de Dieta"] = df_diet.apply(get_diet_type, axis=1)
        
        # Agrupar dados
        df_grouped = df_diet.groupby("Tipo de Dieta")["Anxiety Level (1-10)"].mean().reset_index()
        df_grouped = df_grouped.sort_values("Anxiety Level (1-10)", ascending=False)
        
        # Criar gráfico com estilo moderno
        fig_diet = go.Figure()
        
        fig_diet.add_trace(go.Scatter(
            x=df_grouped["Tipo de Dieta"],
            y=df_grouped["Anxiety Level (1-10)"],
            mode='lines+markers+text',
            line=dict(color='mediumturquoise', width=3),
            marker=dict(size=10, symbol="circle", color='indianred'),
            text=[f'{v:.2f}' for v in df_grouped["Anxiety Level (1-10)"]],
            textposition='top center',
            name='Ansiedade Média'
        ))
        
        # Layout com melhorias visuais
        fig_diet.update_layout(
            title="📊 Média do Nível de Ansiedade por Tipo de Dieta",
            title_font_size=20,
            xaxis_title="Tipo de Dieta",
            yaxis_title="Ansiedade Média",
            xaxis=dict(tickangle=45),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=100)
        )
        
        st.plotly_chart(fig_diet, use_container_width=True)




        # --- Gráfico 3: Distribuição Ansiedade por Consumo de Álcool (bins Baixo, Médio, Alto) ---

        # Filtrar e categorizar os dados de álcool
        df_alcohol = df_clusters.copy()
        df_alcohol = df_alcohol[df_alcohol["Alcohol Consumption (drinks/week)"] <= 19]
        
        bins = [-float("inf"), 6.33, 12.66, float("inf")]
        labels = ["Baixo", "Médio", "Alto"]
        df_alcohol["AlcoholBin"] = pd.cut(df_alcohol["Alcohol Consumption (drinks/week)"], bins=bins, labels=labels)
        
        # Preparar dados para o gráfico
        dados_kde = []
        for label in labels:
            anx_values = df_alcohol[df_alcohol["AlcoholBin"] == label]["Anxiety Level (1-10)"].dropna().tolist()
            if len(anx_values) < 10:
                continue
            hist, bin_edges = np.histogram(anx_values, bins=30, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            dados_kde.append({
                'label': label,
                'x': bin_centers,
                'y': hist
            })


        fig_alcohol = go.Figure()
        for d in dados_kde:
            fig_alcohol.add_trace(go.Scatter(
                x=d['x'], y=d['y'], fill='tozeroy', mode='lines', name=d['label'], opacity=0.5
            ))
        fig_alcohol.update_layout(
            title='Distribuição do Nível de Ansiedade por Consumo de Álcool (bins: Baixo, Médio, Alto)',
            xaxis_title='Nível de Ansiedade',
            yaxis_title='Densidade'
        )
        st.plotly_chart(fig_alcohol, use_container_width=True)


        # --- Gráfico 4: Média Ansiedade por Cafeína e Fumar ---
        
        # Copiar os dados relevantes do DataFrame principal
        df_caffeine = df_clusters[["Anxiety Level (1-10)", "Caffeine Intake (mg/day)", "Smoking_Yes"]].copy()
        
        # Categorizar cafeína manualmente: Baixo < 50, Médio 50-150, Alto > 150
        bins = [-float("inf"), 50, 150, float("inf")]
        labels_caf = ["Baixo", "Médio", "Alto"]
        df_caffeine["CaffeineCat"] = pd.cut(df_caffeine["Caffeine Intake (mg/day)"], bins=bins, labels=labels_caf)
        
        # Mapear status de fumo
        df_caffeine["Smoking Status"] = df_caffeine["Smoking_Yes"].map({1: "Fuma", 0: "Não Fuma"})
        
        # Calcular média de ansiedade por combinação de fumo e cafeína
        agg_df = df_caffeine.groupby(["Smoking Status", "CaffeineCat"])["Anxiety Level (1-10)"].mean().reset_index()
        
        # Organizar dados para o gráfico
        caffeine_cats_sorted = ["Baixo", "Médio", "Alto"]
        bar_data = {}
        for status in agg_df["Smoking Status"].unique():
            subset = agg_df[agg_df["Smoking Status"] == status]
            medias = []
            for cat in caffeine_cats_sorted:
                media = subset[subset["CaffeineCat"] == cat]["Anxiety Level (1-10)"].values
                medias.append(media[0] if len(media) > 0 else None)
            bar_data[status] = medias
        
        # Criar gráfico
        fig_caffeine = go.Figure()
        for status, medias in bar_data.items():
            fig_caffeine.add_trace(go.Bar(
                name=status,
                x=caffeine_cats_sorted,
                y=medias
            ))
        
        fig_caffeine.update_layout(
            barmode='group',
            title='Média de Ansiedade por Consumo de Cafeína e Fumar',
            xaxis_title='Categoria de Cafeína',
            yaxis_title='Nível Médio de Ansiedade'
        )
        st.plotly_chart(fig_caffeine, use_container_width=True)



        # --- Gráfico 5: Nível de Ansiedade por Work/Exercise Ratio e Sono/Estresse (com dropdown) ---
        
        # Selecionar colunas relevantes
        df_activity = df_clusters[[
            "Work Hours per Week",
            "Physical Activity (hrs/week)",
            "Sleep Hours",
            "Stress Level (1-10)",
            "Anxiety Level (1-10)"
        ]].copy()
        
        # Evitar divisão por zero na atividade física
        df_activity["Physical Activity Adjusted"] = df_activity["Physical Activity (hrs/week)"].replace(0, 0.1)
        
        # Calcular as proporções
        df_activity["Work/Exercise Ratio"] = df_activity["Work Hours per Week"] / df_activity["Physical Activity Adjusted"]
        df_activity["Sleep/Stress Ratio"] = df_activity["Sleep Hours"] / (df_activity["Stress Level (1-10)"] + 1e-5)
        
        # Calcular tercis (quantis 33% e 66%)
        q1_wex = df_activity["Work/Exercise Ratio"].quantile(1/3)
        q2_wex = df_activity["Work/Exercise Ratio"].quantile(2/3)
        q1_ses = df_activity["Sleep/Stress Ratio"].quantile(1/3)
        q2_ses = df_activity["Sleep/Stress Ratio"].quantile(2/3)
        
        # Categorizar
        labels = ["Baixo", "Médio", "Alto"]
        df_activity["WorkExBucket"] = pd.cut(df_activity["Work/Exercise Ratio"],
                                              bins=[-float("inf"), q1_wex, q2_wex, float("inf")],
                                              labels=labels)
        df_activity["SleepStressBucket"] = pd.cut(df_activity["Sleep/Stress Ratio"],
                                                   bins=[-float("inf"), q1_ses, q2_ses, float("inf")],
                                                   labels=labels)
        
        # Calcular médias
        wex_avg = df_activity.groupby("WorkExBucket")["Anxiety Level (1-10)"].mean().reindex(labels)
        ses_avg = df_activity.groupby("SleepStressBucket")["Anxiety Level (1-10)"].mean().reindex(labels)
        
        # Organizar para gráfico
        x_vals = labels
        y_vals_wex = wex_avg.tolist()
        y_vals_ses = ses_avg.tolist()
        
        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Bar(x=x_vals, y=y_vals_wex, name="Work/Exercício", marker_color='teal'))
        fig_ratio.update_layout(
            updatemenus=[dict(
                buttons=[
                    dict(
                        label="Work/Exercício",
                        method="update",
                        args=[{"y": [y_vals_wex]}, {"title": "Nível de Ansiedade por Work/Exercise Ratio"}]
                    ),
                    dict(
                        label="Sono/Estresse",
                        method="update",
                        args=[{"y": [y_vals_ses]}, {"title": "Nível de Ansiedade por Sleep/Stress Ratio"}]
                    )
                ],
                direction="down",
                x=0.5,
                xanchor="center",
                y=1.1,
                yanchor="top"
            )],
            title="Nível de Ansiedade por Work/Exercise Ratio",
            yaxis_title="Ansiedade Média"
        )
        
        st.plotly_chart(fig_ratio, use_container_width=True)

        
        # --- Gráfico 7: Densidade de Ansiedade por Categoria de Sono (interativo, não invertido) ---
        st.subheader("Distribuição de Ansiedade por Categoria de Sono (interativo)")
        
        sim_data = df_clusters.copy()
        
        if "Anxiety Level (1-10)" in sim_data.columns and "Sleep Hours" in sim_data.columns:
            sim_data.rename(columns={"Anxiety Level (1-10)": "Ansiedade"}, inplace=True)
        
            # Classificar categorias de sono
            sim_data['Sleep Category'] = pd.cut(
                sim_data['Sleep Hours'],
                bins=[0, 5.5, 7.5, 10],
                labels=['Pouco Sono', 'Sono Ideal', 'Muito Sono']
            )
        
            sim_data_clean = sim_data.dropna(subset=['Sleep Category', 'Ansiedade'])
        
            if not sim_data_clean.empty:
                import plotly.graph_objects as go
                import numpy as np
        
                # Preparar densidades por categoria
                categorias = sim_data_clean['Sleep Category'].unique()
                dados_kde = {}
        
                for cat in categorias:
                    anx_vals = sim_data_clean[sim_data_clean['Sleep Category'] == cat]['Ansiedade'].dropna().tolist()
                    if len(anx_vals) < 10:
                        continue
                    hist, bin_edges = np.histogram(anx_vals, bins=30, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    dados_kde[str(cat)] = (bin_centers, hist)
        
                fig_sono = go.Figure()
        
                if dados_kde:
                    # Exibir primeira curva por padrão
                    nome_inicial = list(dados_kde.keys())[0]
                    x, y = dados_kde[nome_inicial]
                    fig_sono.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', mode='lines', name=nome_inicial))
        
                    # Botões para dropdown
                    buttons = []
                    for nome, (x_vals, y_vals) in dados_kde.items():
                        buttons.append(
                            dict(
                                label=nome,
                                method="update",
                                args=[
                                    {"x": [x_vals], "y": [y_vals]},
                                    {"layout": {"title": f"Densidade de Ansiedade - {nome}"}}
                                ]
                            )
                        )
        
                    fig_sono.update_layout(
                        updatemenus=[dict(
                            buttons=buttons,
                            direction="down",
                            x=0.5,
                            xanchor="center",
                            y=1.15,
                            yanchor="top"
                        )],
                        title=f"Densidade de Ansiedade - {nome_inicial}",
                        xaxis_title="Nível de Ansiedade (1-10)",
                        yaxis_title="Densidade Estimada",
                        showlegend=False
                    )
        
                else:
                    fig_sono.add_trace(go.Scatter(x=[], y=[]))
                    fig_sono.update_layout(
                        title="Nenhum dado disponível para as categorias de sono",
                        xaxis_title="Nível de Ansiedade",
                        yaxis_title="Densidade"
                    )
        
                st.plotly_chart(fig_sono, use_container_width=True)
        
            else:
                st.info("Dados insuficientes para o gráfico de sono.")
        else:
            st.warning("Colunas necessárias ausentes para gerar o gráfico de sono.")

        





        # --- Gráfico Interativo com Linha de Tendência Real ---
    st.markdown("---")
    st.subheader("Análise Detalhada de Fatores de Estilo de Vida e Ansiedade")

    # Selecionar colunas necessárias
    pdf = df[[
        "Recent Major Life Event_Yes",
        "Screen Time per Day (Hours)",
        "Therapy Sessions (per month)",
        "Social Interaction Score",
        "Anxiety Level (1-10)"
    ]].copy()

    # Mapear eventos recentes
    pdf['Evento Recente'] = pdf['Recent Major Life Event_Yes'].map({0: 'Não', 1: 'Sim'})

    # Categorizar variáveis contínuas com pd.cut
    pdf['Tela (horas)'] = pd.cut(pdf['Screen Time per Day (Hours)'], bins=4, labels=[f'{i}-{i+1}' for i in range(4)])
    pdf['Terapia (mês)'] = pd.cut(pdf['Therapy Sessions (per month)'], bins=4, labels=[f'{i}-{i+1}' for i in range(4)])
    pdf['Interação'] = pd.cut(pdf['Social Interaction Score'], bins=4, labels=[f'{i}-{i+1}' for i in range(4)])

    # Agregar dados
    event_data = pdf.groupby('Evento Recente', as_index=False)['Anxiety Level (1-10)'].mean()
    screen_data = pdf.groupby('Tela (horas)', as_index=False)['Anxiety Level (1-10)'].mean()
    therapy_data = pdf.groupby('Terapia (mês)', as_index=False)['Anxiety Level (1-10)'].mean()
    social_data = pdf.groupby('Interação', as_index=False)['Anxiety Level (1-10)'].mean()

    # Criar traços do gráfico
    data_traces = []

    # Evento Recente
    data_traces.append(go.Bar(
        x=event_data['Evento Recente'],
        y=event_data['Anxiety Level (1-10)'],
        name='Evento Recente',
        marker_color='lightcoral',
        visible=True
    ))
    data_traces.append(go.Scatter(
        x=event_data['Evento Recente'],
        y=event_data['Anxiety Level (1-10)'],
        name='Tendência de Ansiedade',
        mode='lines+markers',
        line=dict(color='crimson', width=3),
        visible=True
    ))

    # Tempo de Tela
    data_traces.append(go.Bar(
        x=screen_data['Tela (horas)'].astype(str),
        y=screen_data['Anxiety Level (1-10)'],
        name='Tempo de Tela',
        marker_color='lightsteelblue',
        visible=False
    ))
    data_traces.append(go.Scatter(
        x=screen_data['Tela (horas)'].astype(str),
        y=screen_data['Anxiety Level (1-10)'],
        name='Tendência de Ansiedade',
        mode='lines+markers',
        line=dict(color='darkblue', width=3),
        visible=False
    ))

    # Terapia
    data_traces.append(go.Bar(
        x=therapy_data['Terapia (mês)'].astype(str),
        y=therapy_data['Anxiety Level (1-10)'],
        name='Terapia',
        marker_color='lightgreen',
        visible=False
    ))
    data_traces.append(go.Scatter(
        x=therapy_data['Terapia (mês)'].astype(str),
        y=therapy_data['Anxiety Level (1-10)'],
        name='Tendência de Ansiedade',
        mode='lines+markers',
        line=dict(color='darkgreen', width=3),
        visible=False
    ))

    # Interação Social
    data_traces.append(go.Bar(
        x=social_data['Interação'].astype(str),
        y=social_data['Anxiety Level (1-10)'],
        name='Interação Social',
        marker_color='mediumorchid',
        visible=False
    ))
    data_traces.append(go.Scatter(
        x=social_data['Interação'].astype(str),
        y=social_data['Anxiety Level (1-10)'],
        name='Tendência de Ansiedade',
        mode='lines+markers',
        line=dict(color='darkmagenta', width=3),
        visible=False
    ))

    # Inicializar figura
    fig = go.Figure(data=data_traces)

    # Botões do menu dropdown
    buttons = []
    step = 2  # dois traços por grupo (barra + linha)
    labels = ['Evento Recente', 'Tempo de Tela', 'Terapia por Mês', 'Interação Social']
    titles = [
        'Ansiedade por Evento de Vida Recente',
        'Ansiedade por Tempo de Tela',
        'Ansiedade por Sessões de Terapia',
        'Ansiedade por Interação Social'
    ]
    x_labels = ['Evento Recente', 'Tempo de Tela (h/dia)', 'Sessões de Terapia (mês)', 'Interação Social']

    for i in range(0, len(data_traces), step):
        visibility = [False] * len(data_traces)
        visibility[i] = True
        visibility[i + 1] = True
        idx = i // step
        buttons.append(dict(
            label=labels[idx],
            method='update',
            args=[
                {'visible': visibility},
                {'title': titles[idx], 'xaxis': {'title': x_labels[idx]}}
            ]
        ))

    # Layout final
    fig.update_layout(
        updatemenus=[
            dict(
                type='dropdown',
                direction='down',
                x=0.5,
                xanchor='center',
                y=1.15,
                yanchor='top',
                buttons=buttons
            )
        ],
        title=titles[0],
        yaxis_title="Ansiedade Média",
        xaxis_title=x_labels[0],
        barmode='group',
        legend_title="Indicador",
        height=500,
        font=dict(color='black'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )

    # Exibir gráfico
    st.plotly_chart(fig, use_container_width=True)



    
    st.markdown("---")
    st.subheader("Sinais Fisiológicos Médios por Gênero")

    # Dicionário de colunas dummy
    sexo_cols = {"Feminino": "Gender_Female", "Masculino": "Gender_Male", "Outro": "Gender_Other"}

    # Lista de sinais fisiológicos (substitua conforme necessário)
    sinais_fisiologicos = [
        "Heart Rate", 
        "Respiration Rate", 
        "Body Temperature", 
        "Blood Pressure"
    ]

    try:
        # Verifica se df_pd existe e contém as colunas necessárias
        if isinstance(df_pd, pd.DataFrame) and \
           all(col in df_pd.columns for col in sexo_cols.values()) and \
           all(col in df_pd.columns for col in sinais_fisiologicos):

            # Cria a coluna categórica 'Gender' a partir das colunas dummy
            def get_gender(row):
                for genero, col in sexo_cols.items():
                    if row[col] == 1:
                        return genero
                return 'Desconhecido'

            df_pd['Gender'] = df_pd.apply(get_gender, axis=1)

            # Agrupa por gênero e calcula a média
            sinais_por_genero = df_pd.groupby('Gender')[sinais_fisiologicos].mean().T

            # Gráfico
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            sinais_por_genero.plot(kind='barh', ax=ax)
            ax.set_title("Sinais Fisiológicos Médios por Gênero")
            ax.set_xlabel("Média")
            plt.tight_layout()
            st.pyplot(fig)

        else:
            st.warning("Colunas de gênero ou sinais fisiológicos ausentes no DataFrame.")

    except NameError:
        st.error("O DataFrame 'df_pd' não está definido. Verifique o carregamento dos dados.")







        



elif page == "Classification Model":

    st.header("Modelos de Classificação para Ansiedade Alta")

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import plotly.express as px
        import plotly.graph_objects as go
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import (
            accuracy_score, classification_report, confusion_matrix
        )

        if 'df_inner' not in globals() or df_inner is None or df_inner.empty:
            st.warning("Dados não carregados ou indisponíveis. Carregue os dados antes de prosseguir.")
        else:
            colunas_independentes = [
                'Age',
                'Sleep Hours',
                'Physical Activity (hrs/week)',
                'Diet Quality (1-10)',
                'Stress Level (1-10)'
            ]

            df_class = df_inner[colunas_independentes + ["Anxiety Level (1-10)"]].dropna()
            df_class["Anxiety_High"] = (df_class["Anxiety Level (1-10)"] > 6).astype(int)

            X = df_class[colunas_independentes]
            y = df_class["Anxiety_High"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'k-NN': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'SVM': SVC(random_state=42)
            }

            model_metrics = []
            confusion_data = []

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                cm = confusion_matrix(y_test, y_pred)

                model_metrics.append({
                    'Model': name,
                    'Accuracy': accuracy,
                    'Precision': report['macro avg']['precision'],
                    'Recall': report['macro avg']['recall'],
                    'F1-Score': report['macro avg']['f1-score']
                })

                confusion_data.append((name, cm))

            metrics_df = pd.DataFrame(model_metrics)
            st.subheader("Comparação entre Modelos")
            st.dataframe(metrics_df)

            st.subheader("Gráfico Comparativo Interativo")
            fig = px.bar(
                metrics_df.melt(id_vars="Model"),
                x="Model", y="value", color="variable",
                barmode="group",
                labels={"value": "Métrica", "variable": "Tipo"},
                title="Métricas por Modelo"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Matrizes de Confusão Interativas")
            for name, cm in confusion_data:
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Baixa/Moderada', 'Alta'],
                    y=['Baixa/Moderada', 'Alta'],
                    colorscale='YlGnBu',
                    text=cm,
                    texttemplate="%{text}",
                    hovertemplate="Predito %{x}<br>Real %{y}<br>Qtd: %{z}<extra></extra>"
                ))
                fig_cm.update_layout(title=f"Matriz de Confusão - {name}",
                                     xaxis_title="Predito",
                                     yaxis_title="Real")
                st.plotly_chart(fig_cm, use_container_width=True)

    except Exception as e:
        st.warning("Erro ao executar o modelo de classificação. Verifique se os dados foram carregados corretamente.")
        st.exception(e)






elif page == "Dashboard":

    st.header("Dashboard Geral de Saúde Mental e Ansiedade")

    try:
        import plotly.express as px
        import plotly.graph_objects as go

        if 'df_inner' not in globals() or df_inner is None or df_inner.empty:
            st.warning("Dados não carregados ou indisponíveis. Carregue os dados antes de prosseguir.")
        else:
            df_dash = df_inner.copy()

            # Reconstruir coluna 'Country'
            country_cols = [c for c in df_dash.columns if c.startswith('Country_')]
            if country_cols:
                df_dash['Country'] = df_dash[country_cols].idxmax(axis=1).str.replace("Country_", "")
            else:
                df_dash['Country'] = 'Unknown'

            # Reconstruir coluna 'Mental Health Condition'
            condition_cols = [c for c in df_dash.columns if c.startswith('Mental_Health_Condition_')]
            if condition_cols:
                def get_condition(row):
                    for col in condition_cols:
                        if row.get(col, 0) == 1:
                            return col.replace("Mental_Health_Condition_", "")
                    return "Unknown"
                df_dash['Mental Health Condition'] = df_dash.apply(get_condition, axis=1)
            else:
                df_dash['Mental Health Condition'] = 'Unknown'

            # Remover registros incompletos
            df_dash = df_dash.dropna(subset=["Country", "Anxiety Level (1-10)", "Mental Health Condition"])

            # ================= GRÁFICOS ===================

            import plotly.graph_objects as go
            import pandas as pd
            import plotly.express as px
            
            # Usar DataFrame já preparado
            df_vis = df_dash.copy()
            
            # Coordenadas conhecidas
            country_coords = {
                'USA': (38.0, -97.0),
                'Brazil': (-14.2, -51.9),
                'Germany': (51.2, 10.5),
                'India': (20.6, 78.9),
                'Australia': (-25.0, 133.0),
                'Canada': (56.1, -106.3),
                'Japan': (36.2, 138.2),
                'UK': (55.3, -3.4),
                'France': (46.2, 2.2),
                'Mexico': (23.6, -102.5),
                'China': (35.9, 104.2)
            }
            
            # Emojis de bandeira por país
            flags = {
                'USA': '🇺🇸',
                'Brazil': '🇧🇷',
                'Germany': '🇩🇪',
                'India': '🇮🇳',
                'Australia': '🇦🇺',
                'Canada': '🇨🇦',
                'Japan': '🇯🇵',
                'UK': '🇬🇧',
                'France': '🇫🇷',
                'Mexico': '🇲🇽',
                'China': '🇨🇳'
            }
            
            # Calcular média de ansiedade por país
            df_country_avg = df_vis.groupby('Country', as_index=False)['Anxiety Level (1-10)'].mean()
            df_country_avg[['lat', 'lon']] = df_country_avg['Country'].apply(
                lambda x: pd.Series(country_coords.get(x, (None, None)))
            )
            df_country_avg = df_country_avg.dropna(subset=['lat', 'lon'])
            
            # Ajustar a escala
            min_ansiedade = 3.5
            max_ansiedade = 4.0
            
            # Cores normalizadas para esse intervalo
            norm_colors = px.colors.sample_colorscale(
                "Turbo",
                df_country_avg['Anxiety Level (1-10)'].apply(lambda x: (x - min_ansiedade) / (max_ansiedade - min_ansiedade))
            )
            
            # País com maior ansiedade
            top_country = df_country_avg.loc[df_country_avg['Anxiety Level (1-10)'].idxmax()]
            top_country_name = top_country['Country']
            top_country_value = top_country['Anxiety Level (1-10)']
            
            # Criar figura
            fig = go.Figure()
            
            # Adicionar países
            for i, row in df_country_avg.iterrows():
                fig.add_trace(go.Scattergeo(
                    lon=[row['lon']],
                    lat=[row['lat']],
                    text=f"{row['Country']}: {row['Anxiety Level (1-10)']:.2f}",
                    marker=dict(
                        size=10,
                        color=norm_colors[i],
                        line_color='black',
                        line_width=0.5
                    ),
                    mode='markers',
                    name=f"{flags.get(row['Country'], '')} {row['Country']}",
                    showlegend=True
                ))
            
            # Destaque
            fig.add_trace(go.Scattergeo(
                lon=[top_country['lon']],
                lat=[top_country['lat']],
                text=[f"{top_country_name}<br>{top_country_value:.2f}"],
                marker=dict(
                    size=18,
                    color='#FF1744',
                    line_color='white',
                    line_width=2,
                    symbol='star'
                ),
                mode='markers+text',
                textposition='top center',
                name=f'🔺 Destaque: {top_country_name}'
            ))
            
            # Barra de cor de 3.5 a 4.0
            fig.add_trace(go.Scattergeo(
                lon=[None],
                lat=[None],
                marker=dict(
                    size=0.1,
                    color=[3.5 + 0.05 * i for i in range(11)],  # De 3.5 a 4.0
                    cmin=3.5,
                    cmax=4.0,
                    colorscale='Turbo',
                    colorbar=dict(
                        title=dict(
                            text='Nível de Ansiedade',
                            font=dict(color='white')
                        ),
                        tickvals=[3.5, 3.75, 4.0],
                        tickfont=dict(color='white'),
                        len=0.5,
                        lenmode='fraction',
                        x=1.05,
                        xanchor='left'
                    )
                ),
                showlegend=False
            ))
            
            # Layout
            fig.update_layout(
                geo=dict(
                    projection_type='orthographic',
                    showland=True,
                    showcountries=True,
                    showcoastlines=True,
                    landcolor='rgba(255,255,255,0.1)',
                    countrycolor='white',
                    bgcolor='black'
                ),
                height=700,
                margin=dict(l=0, r=0, t=50, b=0),
                paper_bgcolor='black',
                plot_bgcolor='black',
                font=dict(color='white'),
                legend=dict(
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
            )
            
            # Exibir no Streamlit
            st.plotly_chart(fig, use_container_width=True)












            
             # 📊 Visão Geral Global da Ansiedade
            st.subheader("Visão Geral das Variáveis")
            
            # Estatísticas principais
            media_ansiedade = df_dash["Anxiety Level (1-10)"].mean()
            mediana_ansiedade = df_dash["Anxiety Level (1-10)"].median()
            media_sessoes_terapia = df_dash["Therapy Sessions (per month)"].mean()
            prop_terapia = df_dash["Therapy Sessions (per month)"].gt(0).mean() * 100
            
            # Sexo com maior ansiedade
            sexo_cols = {"Feminino": "Gender_Female", "Masculino": "Gender_Male", "Outro": "Gender_Other"}
            sexo_media = {
                sexo: df_dash[df_dash[col] == 1]["Anxiety Level (1-10)"].mean()
                for sexo, col in sexo_cols.items() if col in df_dash.columns
            }
            sexo_top = max(sexo_media, key=sexo_media.get)
            
            # País com maior ansiedade média
            if 'Country' in df_dash.columns:
                pais_top = df_dash.groupby("Country")["Anxiety Level (1-10)"].mean().idxmax()
                pais_top_valor = df_dash.groupby("Country")["Anxiety Level (1-10)"].mean().max()
            else:
                pais_top, pais_top_valor = "N/A", 0
            
            # Condição de saúde mental mais comum
            condicoes = [col for col in df_dash.columns if col.startswith("Mental Health Condition_")]
            condicao_top = max(condicoes, key=lambda c: df_dash[c].sum()).replace("Mental Health Condition_", "")
            
            # Outras métricas
            idade_media = df_dash["Age"].mean()
            sono_medio = df_dash["Sleep Hours"].mean()
            alcool_medio = df_dash["Alcohol Consumption (drinks/week)"].mean()
            tela_medio = df_dash["Screen Time per Day (Hours)"].mean()
            trabalho_medio = df_dash["Work Hours per Week"].mean()
            cafeina_medio = df_dash["Caffeine Intake (mg/day)"].mean()
            fumantes = df_dash["Smoking_Yes"].mean() * 100
            eventos = df_dash["Recent Major Life Event_Yes"].mean() * 100
            medicacao = df_dash["Medication_Yes"].mean() * 100
            
            # Estilos com gradientes suaves e texto escuro
            box_style = """
            <style>
            .box-metric {
                padding: 1rem;
                border-radius: 15px;
                text-align: center;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.15);
                margin: 0.5rem 0.3rem;
                color: #222;
                font-family: 'Segoe UI', sans-serif;
            }
            .gradient1 { background: linear-gradient(135deg, #fbc2eb, #a6c1ee); }
            .gradient2 { background: linear-gradient(135deg, #fddb92, #d1fdff); }
            .gradient3 { background: linear-gradient(135deg, #a1c4fd, #c2e9fb); }
            .gradient4 { background: linear-gradient(135deg, #d4fc79, #96e6a1); }
            .gradient5 { background: linear-gradient(135deg, #ffecd2, #fcb69f); }
            .gradient6 { background: linear-gradient(135deg, #e0c3fc, #8ec5fc); }
            .gradient7 { background: linear-gradient(135deg, #f6d365, #fda085); }
            .gradient8 { background: linear-gradient(135deg, #cfd9df, #e2ebf0); }
            </style>
            """
            st.markdown(box_style, unsafe_allow_html=True)
            
            # Função para exibir cada métrica
            def render_box(title, value, emoji, gradient_class="gradient1"):
                st.markdown(f"""
                <div class="box-metric {gradient_class}">
                    <div style='font-size:0.95rem; margin-bottom: 0.4rem'>{emoji} {title}</div>
                    <div style='font-size:1.6rem; font-weight:700;'>{value}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Blocos com colunas
            with st.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1: render_box("Média de Ansiedade", f"{media_ansiedade:.2f}", "📈", "gradient1")
                with col2: render_box("Mediana de Ansiedade", f"{mediana_ansiedade:.2f}", "📊", "gradient2")
                with col3: render_box("Média Sessões Terapia", f"{media_sessoes_terapia:.2f}", "💬", "gradient3")
                with col4: render_box("% em Terapia", f"{prop_terapia:.1f}%", "🧠", "gradient4")
            
            with st.container():
                col5, col6, col7, col8 = st.columns(4)
                with col5: render_box("Sexo Mais Ansioso", sexo_top, "🚻", "gradient5")
                with col6: render_box("País com Maior Ansiedade", f"{pais_top} ({pais_top_valor:.2f})", "🌍", "gradient6")
                with col7: render_box("Condição Mental + Comum", condicao_top, "⚠️", "gradient7")
                with col8: render_box("Idade Média", f"{idade_media:.1f} anos", "👤", "gradient8")
            
            with st.container():
                col9, col10, col11, col12 = st.columns(4)
                with col9: render_box("Sono Médio Diário", f"{sono_medio:.1f}h", "🛌", "gradient2")
                with col10: render_box("Álcool Médio", f"{alcool_medio:.1f}/sem", "🍷", "gradient3")
                with col11: render_box("Tempo de Tela", f"{tela_medio:.1f}h/dia", "📺", "gradient4")
                with col12: render_box("Horas de Trabalho", f"{trabalho_medio:.1f}/sem", "💻", "gradient1")
            
            with st.container():
                col13, col14, col15 = st.columns(3)
                with col13: render_box("Cafeína Média", f"{cafeina_medio:.0f} mg/dia", "☕", "gradient5")
                with col14: render_box("% Fumantes", f"{fumantes:.1f}%", "🚬", "gradient6")
                with col15: render_box("% Com Evento Recente", f"{eventos:.1f}%", "😰", "gradient7")
            
            with st.container():
                col16 = st.columns(1)[0]
                with col16: render_box("% Usa Medicação", f"{medicacao:.1f}%", "💊", "gradient8")
            
            st.markdown("---")









            # GRÁFICO: Proporção de Gênero por Quartis de Ansiedade (com Others)
            st.subheader("Distribuição de Género por Quartis de Ansiedade")
            
            try:
                df_quartis = df_dash.copy()
            
                # Criar quartis de ansiedade com rótulos descritivos
                df_quartis['Anxiety Quartile'] = pd.qcut(
                    df_quartis['Anxiety Level (1-10)'],
                    q=4,
                    labels=[
                        'Q1 🔹 Baixo (0–25%)',
                        'Q2 🟦 Médio-Baixo (25–50%)',
                        'Q3 🟨 Médio-Alto (50–75%)',
                        'Q4 🔺 Alto (75–100%)'
                    ]
                )
            
                # Verificar colunas binárias e criar 'Others'
                if 'Gender_Female' in df_quartis.columns and 'Gender_Male' in df_quartis.columns:
                    # Criar coluna "Others" onde nem female nem male são 1
                    df_quartis['Gender_Other'] = ((df_quartis['Gender_Female'] != 1) & (df_quartis['Gender_Male'] != 1)).astype(int)
            
                    # Agrupar por quartil
                    quartile_gender = df_quartis.groupby('Anxiety Quartile')[['Gender_Female', 'Gender_Male', 'Gender_Other']].sum()
            
                    # Calcular percentuais
                    quartile_gender_percent = quartile_gender.div(quartile_gender.sum(axis=1), axis=0) * 100
                    quartile_gender_percent = quartile_gender_percent.reset_index()
            
                    # Reestruturar para long format
                    df_long = quartile_gender_percent.melt(
                        id_vars='Anxiety Quartile',
                        value_vars=['Gender_Female', 'Gender_Male', 'Gender_Other'],
                        var_name='Gênero',
                        value_name='Percentual'
                    )
            
                    # Renomear valores
                    df_long['Gênero'] = df_long['Gênero'].map({
                        'Gender_Female': 'Feminino',
                        'Gender_Male': 'Masculino',
                        'Gender_Other': 'Outros'
                    })
            
                    import plotly.express as px
            
                    fig_quartil = px.bar(
                        df_long,
                        x='Percentual',
                        y='Anxiety Quartile',
                        color='Gênero',
                        orientation='h',
                        text=df_long['Percentual'].map(lambda x: f"{x:.1f}%"),
                    )
            
                    fig_quartil.update_layout(
                        barmode='stack',
                        xaxis=dict(range=[0, 100]),
                        yaxis_title=None,
                        legend_title='Gênero'
                    )
            
                    st.plotly_chart(fig_quartil, use_container_width=True)
            
                else:
                    st.info("Colunas de gênero binário ('Gender_Female' e 'Gender_Male') não encontradas nos dados.")
            
            except Exception as e:
                st.warning("Erro ao gerar gráfico de quartis de ansiedade por gênero.")
                st.exception(e)


    except Exception as e:
        st.warning("Erro ao carregar o dashboard.")
        st.exception(e)







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


elif page == "Predict your Anxiety":
    st.header("🧠 Predict Your Anxiety Level")
    st.markdown("Fill in your personal information below to get a prediction of your anxiety level (1-10 scale)")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Personal Information")
        age = st.slider("Age", min_value=18, max_value=80, value=30, help="Your current age")
        sleep_hours = st.slider("Sleep Hours per Night", min_value=3.0, max_value=12.0, value=7.0, step=0.5, 
                               help="Average hours of sleep per night")
        physical_activity = st.slider("Physical Activity (hours/week)", min_value=0.0, max_value=20.0, value=3.0, step=0.5,
                                    help="Hours of physical exercise per week")
        diet_quality = st.slider("Diet Quality (1-10)", min_value=1, max_value=10, value=5,
                                help="Rate your diet quality from 1 (very poor) to 10 (excellent)")
    
    with col2:
        st.subheader("🏥 Health & Lifestyle")
        stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5,
                                help="Rate your current stress level from 1 (very low) to 10 (very high)")
    
    if st.button("🔮 Predict My Anxiety Level", type="primary"):
        try:
        # Check if we have the necessary data and models
            if not df_inner.empty:
                # Prepare the input data
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Sleep Hours': [sleep_hours],
                    'Physical Activity (hrs/week)': [physical_activity],
                    'Diet Quality (1-10)': [diet_quality],
                    'Stress Level (1-10)': [stress_level],
                })
                
                # Define independent columns
                colunas_independentes = [
                    "Age",
                    "Sleep Hours", 
                    "Physical Activity (hrs/week)",
                    "Diet Quality (1-10)",
                    "Stress Level (1-10)",
                ]
                
                df_inner = df_inner[['Mental Health Condition_Anxiety'] + colunas_independentes].dropna()
            
                x = df_inner.drop(columns=['Mental Health Condition_Anxiety'])
                y = df_inner['Mental Health Condition_Anxiety']

                
                # Split train/test
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.svm import SVC
                from sklearn.metrics import accuracy_score, classification_report
                
                X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state = 5)
                
                # Classification models
                models = {
                    'Random Forest': RandomForestClassifier(random_state=42),
                    'SVM': SVC(random_state=5, probability=True)
                }
                
                # Train and evaluate models to choose the best one
                best_model = None
                best_accuracy = 0
                best_model_name = ""
                
                for name, model in models.items():
                    model.fit(X_train, Y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(Y_test, y_pred)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model
                        best_model_name = name
                
                # Make prediction with the best model
                prediction_proba = best_model.predict_proba(input_data)[0]
                prediction_class = best_model.predict(input_data)[0]
                
                # Get probability for high anxiety
                high_anxiety_prob = prediction_proba[1] if len(prediction_proba) > 1 else 0
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                # Create a nice display for the result
                col_result1, col_result2, col_result3 = st.columns([1, 2, 1])

                with col_result2:
                    if prediction_class == 1:
                        st.metric(
                            label="🎯 Anxiety Level Prediction",
                            value="High Risk",
                            delta=f"😰 {high_anxiety_prob*100:.1f}% probability"
                        )
                    else:
                        st.metric(
                            label="🎯 Anxiety Level Prediction", 
                            value="Low-Moderate Risk",
                            delta=f"😌 {(1-high_anxiety_prob)*100:.1f}% probability"
                        )
            
            # Model performance info
            st.info(f"🤖 **Model Used**: {best_model_name} (Accuracy: {best_accuracy:.2f})")
            
            # Interpretation
            st.subheader("📊 Interpretation")
            if prediction_class == 0:
                st.success("🌟 **Low-Moderate Anxiety Risk**: Your profile suggests you're likely managing stress well. Keep up the healthy habits!")
                recommendations = [
                    "Continue your current healthy lifestyle",
                    "Maintain regular physical activity", 
                    "Keep your good sleep schedule",
                    "Consider sharing your wellness strategies with others"
                ]
            else:
                st.warning("⚠️ **High Anxiety Risk**: Your profile suggests you may be at higher risk for elevated anxiety levels.")
                recommendations = [
                    "Consider speaking with a mental health professional",
                    "Practice stress-reduction techniques like meditation",
                    "Prioritize adequate sleep (7-9 hours)",
                    "Engage in regular physical exercise",
                    "Limit caffeine intake if high",
                    "Build a strong support network"
                ]
            
            st.subheader("💡 Personalized Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
            
            # Show feature importance (for Random Forest)
            if best_model_name == 'Random Forest':
                st.subheader("📈 Factors Influencing Your Prediction")
                feature_importance = pd.DataFrame({
                    'Feature': colunas_independentes,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                import plotly.express as px
                fig_importance = px.bar(
                    feature_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance in Your Anxiety Risk Prediction",
                    labels={'Importance': 'Importance Score', 'Feature': 'Health Factors'}
                )
                fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Show probability breakdown
            st.subheader("🎲 Prediction Probabilities")
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                st.metric(
                    label="Low-Moderate Anxiety",
                    value=f"{(1-high_anxiety_prob)*100:.1f}%"
                )
            
            with prob_col2:
                st.metric(
                    label="High Anxiety Risk", 
                    value=f"{high_anxiety_prob*100:.1f}%"
                )
            
            # Disclaimer
            st.info("⚠️ **Disclaimer**: This prediction is based on a machine learning classification model trained on survey data and should not replace professional medical advice. If you're experiencing persistent anxiety, please consult with a healthcare professional.")

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {str(e)}")
            st.info("Please check that all required data is available and try again.")

    
# Footer
st.sidebar.markdown("---")
st.sidebar.info("Mental Health Data Dashboard - Built with Streamlit")
