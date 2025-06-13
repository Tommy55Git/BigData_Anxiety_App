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

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
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
        tab1, tab2 = st.tabs(["📊 Sociodemográficos", "🧠 Psicológicos"])

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

   
  
    
       




        
        
        

elif page == "Cluster Analysis":
    st.header("🔍 Análise de Clusters")

    # 1. Verificação inicial do DataFrame
    if df_clusters is None or df_clusters.empty:
        st.warning("Dados de cluster não encontrados ou o DataFrame está vazio. Certifique-se de que os dados foram carregados e processados corretamente.")
        # Opcional: st.stop() para parar a execução da página aqui se os dados forem essenciais
    else:
        st.write("### 📋 Visualização do DataFrame Clusterizado (Primeiras Linhas)")
        st.dataframe(df_clusters.head())
        st.info(f"DataFrame carregado com **{df_clusters.shape[0]}** linhas e **{df_clusters.shape[1]}** colunas.")

        # As features para análise - ajusta conforme colunas presentes em df_clusters
        # Certifique-se de que estas são colunas numéricas adequadas para clustering
        initial_features = [
            'Therapy Sessions (per month)', 'Caffeine Intake (mg/day)',
            'Stress Level (1-10)', 'Heart Rate (bpm)',
            'Physical Activity (hrs/week)', 'Sleep_Stress_Ratio',
            'Work_Exercise_Ratio', 'Anxiety Level (1-10)'
        ]

        # Filtra colunas que realmente existem em df_clusters
        features = [f for f in initial_features if f in df_clusters.columns]

        if not features:
            st.error("Nenhuma das features esperadas para clustering foi encontrada no DataFrame. Por favor, verifique os nomes das colunas e se o DataFrame foi preparado corretamente na etapa anterior.")
            st.stop() # Parar a execução se não houver features para analisar

        # Cria uma cópia do subconjunto de features
        X = df_clusters[features].copy()

        # 2. Tratamento de Valores Nulos (NaNs)
        st.write("### Limpeza de Dados para Clustering")
        nan_count = X.isnull().sum().sum()
        if nan_count > 0:
            st.warning(f"Foram encontrados **{nan_count}** valores nulos nas features selecionadas para clustering. A imputação pela média será aplicada.")
            # Imputar NaNs com a média da coluna.
            # Alternativas: X = X.dropna() para remover linhas com NaN, ou usar a mediana.
            X = X.fillna(X.mean())
            st.info("Valores nulos imputados com a média de suas respetivas colunas.")
        else:
            st.success("Não foram encontrados valores nulos nas features selecionadas para clustering.")

        # 3. Escalonamento dos dados
        st.write("### Escalonamento dos Dados")
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            st.success("Dados escalonados com sucesso para análise de clusters.")
        except Exception as e:
            st.error(f"Erro ao escalar os dados. Certifique-se de que todas as features são numéricas e não contêm valores inválidos: `{e}`")
            st.stop() # Para a execução aqui se a escalagem falhar

        # Tabs de Análise
        tabs = st.tabs([
            "📌 PCA + Variância",
            "🔵 PCA + KMeans",
            "🌿 Cluster Hierárquico",
            "🔎 DBSCAN",
            "📉 Avaliações"
        ])

        # -------------------- PCA Variância --------------------
        with tabs[0]:
            st.subheader("📌 PCA - Variância Explicada")
            st.info("Analisando a variância explicada por cada componente principal para determinar a dimensionalidade dos dados.")
            try:
                pca = PCA()
                pca.fit(X_scaled)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=np.cumsum(pca.explained_variance_ratio_),
                    mode='lines+markers',
                    name='Variância Explicada Acumulada'
                ))
                fig.add_hline(y=0.90, line_dash="dash", line_color="red", annotation_text="90% da variância")
                fig.update_layout(
                    title='Variância Explicada Acumulada pela PCA',
                    xaxis_title='Número de Componentes Principais',
                    yaxis_title='Variância Explicada Acumulada',
                    template='simple_white',
                    height=500
                )
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Erro ao gerar o gráfico de Variância Explicada da PCA: `{e}`")


        # -------------------- KMeans + PCA --------------------
        with tabs[1]:
            st.subheader("🔵 PCA + Clusters KMeans")
            st.info("Aplicando o algoritmo KMeans e visualizando os clusters nos dois primeiros componentes principais (PCA).")

            n_clusters = st.slider("Selecione o número de clusters para KMeans", min_value=2, max_value=7, value=3)

            try:
                # n_init='auto' é importante para versões recentes do scikit-learn
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                clusters = kmeans.fit_predict(X_scaled)

                df_clustered = df_clusters.copy()
                df_clustered['Cluster_ID_KMeans'] = clusters + 1 # +1 para começar a contagem dos clusters de 1

                pca_2d = PCA(n_components=2)
                pcs = pca_2d.fit_transform(X_scaled)
                pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
                pca_df['Cluster_ID'] = df_clustered['Cluster_ID_KMeans'] # Usar o ID do KMeans

                fig = px.scatter(
                    pca_df, x='PC1', y='PC2', color=pca_df['Cluster_ID'].astype(str),
                    title=f'Clusters KMeans nos 2 Primeiros Componentes (PCA) - {n_clusters} Clusters',
                    labels={'color': 'Cluster'},
                    hover_name=df_clustered.index # Opcional: para ver o índice do ponto ao passar o mouse
                )
                st.plotly_chart(fig)

                st.write("#### Sumário dos Clusters KMeans")
                # Exibir a média das features por cluster
                st.dataframe(df_clustered.groupby('Cluster_ID_KMeans')[features].mean())

            except Exception as e:
                st.error(f"Erro ao aplicar KMeans ou gerar o gráfico de clusters: `{e}`")


        # -------------------- Cluster Hierárquico --------------------
        with tabs[2]:
            st.subheader("🌿 Cluster Hierárquico com Dendrograma")
            st.info("Gerando um dendrograma para visualizar a estrutura hierárquica dos clusters. Note: pode ser denso para muitos pontos.")

            try:
                # Limite o número de amostras para o dendrograma se o dataset for muito grande
                if X_scaled.shape[0] > 5000: # Ajuste este valor se necessário
                    st.warning("O dendrograma pode ser muito denso para visualizar com muitos pontos. Usando uma amostra aleatória de 5000 pontos.")
                    sample_indices = np.random.choice(X_scaled.shape[0], 5000, replace=False)
                    X_sampled = X_scaled[sample_indices]
                    labels_sampled = [str(i) for i in df_clusters.index[sample_indices]] # Usar índices originais
                    linked = linkage(X_sampled, method='ward')
                    fig = ff.create_dendrogram(X_sampled, orientation='top', labels=labels_sampled)
                else:
                    linked = linkage(X_scaled, method='ward')
                    fig = ff.create_dendrogram(X_scaled, orientation='top', labels=[str(i) for i in df_clusters.index])

                fig.update_layout(width=1000, height=500, title='Dendrograma Hierárquico')
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Erro ao gerar o Dendrograma Hierárquico: `{e}`. Isso pode ocorrer com muitos pontos ou dados inválidos.")


        # -------------------- DBSCAN --------------------
        with tabs[3]:
            st.subheader("🔎 Clusters com DBSCAN")
            st.info("Aplicando DBSCAN, um algoritmo de clustering baseado em densidade que identifica pontos de ruído (outliers).")

            col1, col2 = st.columns(2)
            with col1:
                eps_val = st.slider("eps (Raio da vizinhança)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
            with col2:
                min_samples_val = st.slider("min_samples (Min. de pontos na vizinhança)", min_value=2, max_value=20, value=5, step=1)

            try:
                dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
                clusters_db = dbscan.fit_predict(X_scaled)

                df_db = df_clusters.copy()
                df_db['Cluster_ID_DBSCAN'] = clusters_db # -1 representa ruído

                pcs_db = PCA(n_components=2).fit_transform(X_scaled)
                db_df = pd.DataFrame(pcs_db, columns=['PC1', 'PC2'])
                db_df['Cluster_ID'] = clusters_db # Usar o ID do DBSCAN

                fig = px.scatter(
                    db_df, x='PC1', y='PC2', color=db_df['Cluster_ID'].astype(str),
                    title='Clusters DBSCAN nos 2 PCs (Pontos com ruído = -1)',
                    labels={'color': 'Cluster'},
                    hover_name=df_db.index # Opcional
                )
                st.plotly_chart(fig)

                st.write("#### Sumário dos Clusters DBSCAN")
                if -1 in clusters_db:
                    st.write("Nota: O Cluster **-1** representa pontos de ruído (outliers) não atribuídos a nenhum cluster.")
                st.dataframe(df_db.groupby('Cluster_ID_DBSCAN')[features].mean())

            except Exception as e:
                st.error(f"Erro ao aplicar DBSCAN ou gerar o gráfico de clusters: `{e}`. Tente ajustar 'eps' e 'min_samples'.")


        # -------------------- Avaliação --------------------
        with tabs[4]:
            st.subheader("📉 Métricas de Avaliação dos Clusters")
            st.info("Avaliando a qualidade dos clusters usando métricas comuns como Silhouette, Calinski-Harabasz e Davies-Bouldin.")

            # Avaliação KMeans
            st.markdown("---")
            st.markdown("### **🔵 Avaliação do KMeans**")
            try:
                # Verifique se há mais de um cluster para avaliar o Silhouette Score
                # E se não há apenas um cluster ou todos os pontos no mesmo cluster
                if len(set(clusters)) > 1 and len(set(clusters)) < X_scaled.shape[0]: # Não avaliar se todos os pontos são 1 cluster
                    sil_k = silhouette_score(X_scaled, clusters)
                    ch_k = calinski_harabasz_score(X_scaled, clusters)
                    db_k = davies_bouldin_score(X_scaled, clusters)

                    st.markdown(f"- **Silhouette Score**: `{sil_k:.4f}` (Maior é melhor, intervalo: -1 a 1)")
                    st.markdown(f"- **Calinski-Harabasz Index**: `{ch_k:.4f}` (Maior é melhor)")
                    st.markdown(f"- **Davies-Bouldin Index**: `{db_k:.4f}` (Menor é melhor)")
                else:
                    st.warning(f"KMeans formou apenas {len(set(clusters))} cluster(s) ou não há clusters válidos para avaliação. Ajuste o número de clusters.")
            except Exception as e:
                st.error(f"Erro ao calcular métricas para KMeans: `{e}`. Verifique a dimensionalidade e a formação dos clusters.")

            # Avaliação DBSCAN (removendo ruído)
            st.markdown("---")
            st.markdown("### **🔎 Avaliação do DBSCAN (sem ruído)**")
            try:
                clusters_db_arr = np.array(clusters_db)
                mask = clusters_db_arr != -1 # Máscara para remover pontos de ruído
                clusters_masked = clusters_db_arr[mask]
                X_masked = X_scaled[mask]

                # Verifique se ainda há clusters válidos (mais de 1) após remover o ruído
                # E se não há apenas um cluster ou todos os pontos no mesmo cluster após mascarar
                if len(set(clusters_masked)) > 1 and len(set(clusters_masked)) < X_masked.shape[0]:
                    sil_d = silhouette_score(X_masked, clusters_masked)
                    ch_d = calinski_harabasz_score(X_masked, clusters_masked)
                    db_d = davies_bouldin_score(X_masked, clusters_masked)

                    st.markdown(f"- **Silhouette Score**: `{sil_d:.4f}` (Maior é melhor, intervalo: -1 a 1)")
                    st.markdown(f"- **Calinski-Harabasz Index**: `{ch_d:.4f}` (Maior é melhor)")
                    st.markdown(f"- **Davies-Bouldin Index**: `{db_d:.4f}` (Menor é melhor)")
                else:
                    st.warning("DBSCAN não formou clusters válidos (mais de um, sem ruído) para avaliação, ou todos os pontos são ruído, ou formou um único cluster.")
            except Exception as e:
                st.error(f"Erro ao calcular métricas para DBSCAN: `{e}`. Verifique os parâmetros do DBSCAN e a distribuição dos dados.")


        
        
        
        
        

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
