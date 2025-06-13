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
    ["Data Overview", "Visualizations", "Analysis", "Classification Model", "Regression Model"]
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
                ansiedade_por_nivel[nivel] = valores
        
            # Gerar gráfico
            plt.figure(figsize=(6, 4))
            for nivel, valores in ansiedade_por_nivel.items():
                if len(valores) < 10:
                    continue
                hist, bin_edges = np.histogram(valores, bins=30, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                plt.plot(bin_centers, hist, label=nivel, alpha=0.7)
        
            plt.title('Distribuição do Nível de Ansiedade por Nível de Exercício')
            plt.xlabel('Nível de Ansiedade (1-10)')
            plt.ylabel('Densidade Aproximada')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt.gcf())
            plt.clf()



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
        


         # --- Gráfico 6: Interativo com dropdown para Evento Recente, Tempo Tela, Terapia, Interação ---
        
        cols_desejadas = [
            "Anxiety Level (1-10)",
            "Recent Event",
            "Screen Time (hrs/day)",
            "Therapy_Yes",
            "Social Interaction Level (1-10)"
        ]
        
        # Verificar colunas existentes
        cols_existentes = [col for col in cols_desejadas if col in df_clusters.columns]
        df_interativo = df_clusters[cols_existentes].copy()
        
        def avg_anxiety_by_col(df, col_name, val_map=None):
            if col_name not in df.columns:
                return [], []
            grouped = df.groupby(col_name)["Anxiety Level (1-10)"].mean().reset_index()
            grouped = grouped.dropna()
            x_vals, y_vals = [], []
            for _, row in grouped.iterrows():
                key = row[col_name]
                if val_map:
                    key = val_map.get(key, key)
                x_vals.append(str(key))  # garantir que x seja string
                y_vals.append(round(row["Anxiety Level (1-10)"], 2))  # arredondar para melhor visual
            return x_vals, y_vals
        
        event_map = {0: "Nenhum", 1: "Sim"}
        therapy_map = {0: "Não", 1: "Sim"}
        
        dados_interativo = {}
        
        # Coletar dados por variável
        if "Recent Event" in df_interativo.columns:
            xv, yv = avg_anxiety_by_col(df_interativo, "Recent Event", event_map)
            if len(xv) > 0 and len(yv) > 0:
                dados_interativo["Evento Recente"] = (xv, yv)
        
        if "Screen Time (hrs/day)" in df_interativo.columns:
            xv, yv = avg_anxiety_by_col(df_interativo, "Screen Time (hrs/day)")
            if len(xv) > 0 and len(yv) > 0:
                dados_interativo["Tempo de Tela"] = (xv, yv)
        
        if "Therapy_Yes" in df_interativo.columns:
            xv, yv = avg_anxiety_by_col(df_interativo, "Therapy_Yes", therapy_map)
            if len(xv) > 0 and len(yv) > 0:
                dados_interativo["Terapia"] = (xv, yv)
        
        if "Social Interaction Level (1-10)" in df_interativo.columns:
            xv, yv = avg_anxiety_by_col(df_interativo, "Social Interaction Level (1-10)")
            if len(xv) > 0 and len(yv) > 0:
                dados_interativo["Interação Social"] = (xv, yv)
        
        fig_interativo = go.Figure()
        
        if dados_interativo:
            # Adiciona o primeiro gráfico com dados
            nome_inicial = list(dados_interativo.keys())[0]
            xv, yv = dados_interativo[nome_inicial]
            fig_interativo.add_trace(go.Bar(x=xv, y=yv))
            title_inicial = f"Nível Médio de Ansiedade por {nome_inicial}"
        else:
            # Criar gráfico vazio com aviso visual
            fig_interativo.add_trace(go.Bar(x=[], y=[]))
            title_inicial = "Nenhum dado disponível para variáveis selecionadas"
        
        # Criar botões para dropdown
        buttons = []
        for nome, (xv, yv) in dados_interativo.items():
            buttons.append(
                dict(
                    label=nome,
                    method="update",
                    args=[
                        {"x": [xv], "y": [yv]},
                        {"layout": {"title": f"Nível Médio de Ansiedade por {nome}"}}
                    ]
                )
            )
        
        # Layout do gráfico
        fig_interativo.update_layout(
            updatemenus=[dict(
                buttons=buttons,
                direction="down",
                x=0.5,
                xanchor="center",
                y=1.1,
                yanchor="top"
            )] if buttons else [],
            yaxis_title="Ansiedade Média",
            title=title_inicial
        )
        
        st.plotly_chart(fig_interativo, use_container_width=True)





        
        
        
        

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
