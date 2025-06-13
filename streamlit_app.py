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
    ["Data Overview", "Visualizations", "Analysis",  "Regression Model"]
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



        # --- Gráfico 2: Média de Ansiedade por Tipo de Dieta ---

        diet_columns = [c for c in df_clusters.columns if c.startswith("Diet Type_")]
        
        df_diet = df_clusters
        df_diet = df_diet.withColumn(
            "Diet Type",
            when(col(diet_columns[0]) == 1, diet_columns[0].replace("Diet Type_", ""))
        )
        for c in diet_columns[1:]:
            diet_name = c.replace("Diet Type_", "")
            df_diet = df_diet.withColumn(
                "Diet Type",
                when((col(c) == 1) & (col("Diet Type").isNull()), diet_name).otherwise(col("Diet Type"))
            )

        df_grouped = df_diet.groupBy("Diet Type").agg(avg(col("Anxiety Level (1-10)")).alias("Avg Anxiety"))
        dados = df_grouped.collect()
        diet_types = [row['Diet Type'] for row in dados]
        avg_anxieties = [row['Avg Anxiety'] for row in dados]

        fig_diet = px.line(
            x=diet_types,
            y=avg_anxieties,
            markers=True,
            title="Média do Nível de Ansiedade por Tipo de Dieta",
            labels={"x": "Tipo de Dieta", "y": "Ansiedade Média"}
        )
        st.plotly_chart(fig_diet, use_container_width=True)


        # --- Gráfico 3: Distribuição Ansiedade por Consumo de Álcool (bins Baixo, Médio, Alto) ---

        splits = [-float("inf"), 6.33, 12.66, 19.0]
        bucketizer = bucketizer(splits=splits, inputCol="Alcohol Consumption (drinks/week)", outputCol="AlcoholBin") # type: ignore
        df_alcohol = df_clusters.filter(col("Alcohol Consumption (drinks/week)") <= 19) \
                                .select("Anxiety Level (1-10)", "Alcohol Consumption (drinks/week)")
        df_alcohol = bucketizer.transform(df_alcohol)

        labels = {0.0: "Baixo", 1.0: "Médio", 2.0: "Alto"}

        dados_kde = []
        for i in range(3):
            anx_values = df_alcohol.filter(col("AlcoholBin") == i).select("Anxiety Level (1-10)").rdd.flatMap(lambda x: x).collect()
            if len(anx_values) < 10:
                continue
            hist, bin_edges = np.histogram(anx_values, bins=30, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            dados_kde.append({
                'label': labels[i],
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

        df_caffeine = df_clusters.select("Anxiety Level (1-10)", "Caffeine Intake (mg/day)", "Smoking_Yes")

        # Bucketizar cafeína em 3 categorias
        caffeine_splits = [-float("inf"), df_caffeine.agg({'Caffeine Intake (mg/day)': 'approx_percentile'}).collect()[0][0],  # aproximadamente 33% quantil
                        df_caffeine.agg({'Caffeine Intake (mg/day)': 'approx_percentile'}).collect()[1][0],  # aproximadamente 66% quantil
                        float("inf")]
        # Como approx_percentile não é trivial usar aqui, para simplificar vamos fixar bins:
        caffeine_splits = [-float("inf"), 50, 150, float("inf")]  # Exemplo: Baixo <50, Médio 50-150, Alto >150

        bucketizer_caf = bucketizer(splits=caffeine_splits, inputCol="Caffeine Intake (mg/day)", outputCol="CaffeineCat")
        df_caffeine = bucketizer_caf.transform(df_caffeine)

        # Mapear índice para label
        caf_labels = {0.0: "Baixo", 1.0: "Médio", 2.0: "Alto"}

        # Criar coluna Smoking Status
        df_caffeine = df_caffeine.withColumn("Smoking Status", when(col("Smoking_Yes") == 1, "Fuma").otherwise("Não Fuma"))

        # Agrupar e calcular média ansiedade
        df_grouped = df_caffeine.groupBy("Smoking Status", "CaffeineCat") \
                            .agg(avg(col("Anxiety Level (1-10)")).alias("Avg Anxiety"))

        dados = df_grouped.collect()
        # Organizar dados para plotly
        grouped_dict = {}
        for row in dados:
            smoking = row['Smoking Status']
            caffeine_cat = caf_labels.get(row['CaffeineCat'], "Unknown")
            media = row['Avg Anxiety']
            grouped_dict.setdefault(smoking, []).append((caffeine_cat, media))

        # Para plotly bar, criar listas para cada categoria
        caffeine_cats_sorted = ["Baixo", "Médio", "Alto"]
        bar_data = []
        for smoking_status, values in grouped_dict.items():
            # Criar lista de médias ordenadas por caffeine_cats_sorted
            vals_sorted = []
            for cat in caffeine_cats_sorted:
                v = next((m for (c, m) in values if c == cat), None)
                vals_sorted.append(v)
            bar_data.append((smoking_status, vals_sorted))

        fig_caffeine = go.Figure()
        for smoking_status, medias in bar_data:
            fig_caffeine.add_trace(go.Bar(
                name=smoking_status,
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

        df_activity = df_clusters.select("Work Hours per Week", "Physical Activity (hrs/week)",
                                        "Sleep Hours", "Stress Level (1-10)", "Anxiety Level (1-10)")

        # Ajustar Physical Activity para evitar divisão por zero
        df_activity = df_activity.withColumn(
            "Physical Activity Adjusted",
            when(col("Physical Activity (hrs/week)") == 0, 0.1).otherwise(col("Physical Activity (hrs/week)"))
        )

        df_activity = df_activity.withColumn(
            "Work/Exercise Ratio",
            col("Work Hours per Week") / col("Physical Activity Adjusted")
        ).withColumn(
            "Sleep/Stress Ratio",
            col("Sleep Hours") / (col("Stress Level (1-10)") + 1e-5)
        )

        # Usar approxQuantile para pegar quantis para bucketizer
        q1_wex, q2_wex = df_activity.approxQuantile("Work/Exercise Ratio", [1/3, 2/3], 0.01)
        q1_ses, q2_ses = df_activity.approxQuantile("Sleep/Stress Ratio", [1/3, 2/3], 0.01)

        splits_wex = [-float("inf"), q1_wex, q2_wex, float("inf")]
        splits_ses = [-float("inf"), q1_ses, q2_ses, float("inf")]

        bucketizer_wex = bucketizer(splits=splits_wex, inputCol="Work/Exercise Ratio", outputCol="WorkExBucket")
        bucketizer_ses = bucketizer(splits=splits_ses, inputCol="Sleep/Stress Ratio", outputCol="SleepStressBucket")

        df_activity = bucketizer_wex.transform(df_activity)
        df_activity = bucketizer_ses.transform(df_activity)

        labels = {0.0: "Baixo", 1.0: "Médio", 2.0: "Alto"}

        # Agrupar médias para cada bucket
        df_wex = df_activity.groupBy("WorkExBucket").agg(avg(col("Anxiety Level (1-10)")).alias("Avg Anxiety")).collect()
        df_ses = df_activity.groupBy("SleepStressBucket").agg(avg(col("Anxiety Level (1-10)")).alias("Avg Anxiety")).collect()

        dados = {
            "Work/Exercício": [(labels[row["WorkExBucket"]], row["Avg Anxiety"]) for row in df_wex],
            "Sono/Estresse": [(labels[row["SleepStressBucket"]], row["Avg Anxiety"]) for row in df_ses]
        }

        fig_ratio = go.Figure()

        # Inicialmente mostra Work/Exercise
        x_vals = [x for x, _ in dados["Work/Exercício"]]
        y_vals = [y for _, y in dados["Work/Exercício"]]

        fig_ratio.add_trace(go.Bar(x=x_vals, y=y_vals, marker_color='teal'))

        buttons = [
            dict(
                label="Work/Exercício",
                method="update",
                args=[{"x": [x_vals], "y": [y_vals]}, {"title": "Nível de Ansiedade por Work/Exercise Ratio"}]
            ),
            dict(
                label="Sono/Estresse",
                method="update",
                args=[
                    {"x": [[x for x, _ in dados["Sono/Estresse"]]], "y": [[y for _, y in dados["Sono/Estresse"]]]},
                    {"title": "Nível de Ansiedade por Sleep/Stress Ratio"}
                ]
            )
        ]

        fig_ratio.update_layout(
            updatemenus=[dict(
                buttons=buttons,
                direction="down",
                x=0.5,
                xanchor="center",
                y=1.1,
                yanchor="top"
            )],
            yaxis_title="Ansiedade Média",
            title="Nível de Ansiedade por Work/Exercise Ratio"
        )

        st.plotly_chart(fig_ratio, use_container_width=True)


        # --- Gráfico 6: Interativo com dropdown para Evento Recente, Tempo Tela, Terapia, Interação ---

        df_interativo = df_clusters.select("Anxiety Level (1-10)", "Recent Event", "Screen Time (hrs/day)",
                                        "Therapy_Yes", "Social Interaction Level (1-10)")

        def avg_anxiety_by_col(df, col_name, val_map=None):
            grouped = df.groupBy(col_name).agg(avg(col("Anxiety Level (1-10)")).alias("Avg Anxiety")).collect()
            x_vals, y_vals = [], []
            for row in grouped:
                key = row[col_name]
                if val_map:
                    key = val_map.get(key, key)
                x_vals.append(key)
                y_vals.append(row["Avg Anxiety"])
            return x_vals, y_vals

        event_map = {0: "Nenhum", 1: "Sim"}
        therapy_map = {0: "Não", 1: "Sim"}

        dados_interativo = {
            "Evento Recente": avg_anxiety_by_col(df_interativo, "Recent Event", event_map),
            "Tempo de Tela": avg_anxiety_by_col(df_interativo, "Screen Time (hrs/day)"),
            "Terapia": avg_anxiety_by_col(df_interativo, "Therapy_Yes", therapy_map),
            "Interação Social": avg_anxiety_by_col(df_interativo, "Social Interaction Level (1-10)")
        }

        fig_interativo = go.Figure()

        # Inicialmente mostrar "Evento Recente"
        fig_interativo.add_trace(go.Bar(x=dados_interativo["Evento Recente"][0], y=dados_interativo["Evento Recente"][1]))

        buttons = []
        for nome, (xv, yv) in dados_interativo.items():
            buttons.append(
                dict(
                    label=nome,
                    method="update",
                    args=[{"x": [xv], "y": [yv]}, {"title": f"Nível Médio de Ansiedade por {nome}"}]
                )
            )

        fig_interativo.update_layout(
            updatemenus=[dict(
                buttons=buttons,
                direction="down",
                x=0.5,
                xanchor="center",
                y=1.1,
                yanchor="top"
            )],
            yaxis_title="Ansiedade Média",
            title="Nível Médio de Ansiedade (Interativo)"
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
