import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import gc

# ── Configuração da página Streamlit ──────────────────────────────────────────
st.set_page_config(
    page_title="Home Credit — Motor de Risco Transacional",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS Dark/Glassmorphic personalizada
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stAlert {
        background-color: #161b22;
        border-color: #30363d;
        color: #c9d1d9;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    </style>
""", unsafe_allow_html=True)

# ── Caminhos e Configurações ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
FEATURES_AGREGADAS_DIR = os.path.join(OUTPUT_DIR, "features_agregadas")

MODEL_PATH = os.path.join(OUTPUT_DIR, "squad3_lgb_ensemble.pkl")
METADATA_PATH = os.path.join(OUTPUT_DIR, "squad3_metadata.json")

# ── Carregamento de Recursos (Cacheado) ──────────────────────────────────────
@st.cache_resource
def carregar_ativos_modelo():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        return None, None
    
    with open(MODEL_PATH, "rb") as f:
        assets = pickle.load(f)
        
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    return assets, meta

@st.cache_data
def carregar_tabelas_perifericas():
    tabelas = {}
    nomes = ["bureau_agg", "bb_agg", "prev_agg", "inst_agg", "pos_agg", "cc_agg"]
    
    progresso = st.empty()
    progresso.info("Carregando banco de dados históricos agregados...")
    
    for nome in nomes:
        caminho = os.path.join(FEATURES_AGREGADAS_DIR, f"{nome}.csv.gz")
        if os.path.exists(caminho):
            tabelas[nome] = pd.read_csv(caminho, compression="gzip")
        else:
            tabelas[nome] = None
            
    progresso.empty()
    return tabelas

# Tenta carregar os modelos e metadados
assets, meta = carregar_ativos_modelo()

# ── Layout da Interface ───────────────────────────────────────────────────────
st.title("🏦 Motor de Risco Transacional — Home Credit")
st.subheader("Painel de Decisão Comercial de Crédito B2C")

if assets is None or meta is None:
    st.error("⚠️ Modelos não encontrados no diretório `/output`!")
    st.info("""
        **Como resolver:**
        1. Certifique-se de que o dataset da competição está extraído em `/home-credit-default-risk`.
        2. Execute o pipeline técnico completo executando o arquivo `squad3_pipeline.py` no terminal.
        3. Após a conclusão bem-sucedida, os arquivos de modelo `squad3_lgb_ensemble.pkl` e metadados `squad3_metadata.json` serão gerados automaticamente na pasta `/output`.
    """)
else:
    # Carregar dados históricos e exibir na barra lateral
    tabelas_perifericas = carregar_tabelas_perifericas()
    
    st.sidebar.image("https://img.shields.io/badge/Squad_3-Home_Credit-58a6ff?style=for-the-badge&logo=home-assistant", use_container_width=True)
    st.sidebar.header("Configurações do Motor")
    st.sidebar.markdown(f"**Threshold Ótimo:** `{meta['threshold_otimo']:.3f}`")
    st.sidebar.markdown(f"**Taxa Recuperação:** `{meta['premissas']['taxa_recuperacao']*100:.0f}%`")
    st.sidebar.markdown(f"**Prazo Médio:** `{meta['premissas']['prazo_medio_meses']} meses`")
    
    st.sidebar.markdown("---")
    st.sidebar.write("⚡ **Modo Ensemble**: Previsão ponderada de 5 sub-modelos calibrados via validação cruzada 5-fold.")
    
    # ── Upload de Arquivos ────────────────────────────────────────────────────
    st.write("### 📂 Processamento de Novos Clientes")
    st.markdown("Suba um arquivo `.csv` contendo as informações cadastrais dos clientes do dia (formato da tabela `application_test.csv`).")
    
    arquivo_upload = st.file_uploader("Selecione o arquivo CSV", type=["csv"])
    
    if arquivo_upload is not None:
        try:
            # Carrega dados
            df_novos = pd.read_csv(arquivo_upload)
            n_linhas = len(df_novos)
            st.success(f"✅ Arquivo carregado com sucesso! {n_linhas:,} registros encontrados.")
            
            with st.spinner("Processando dados e realizando feature engineering..."):
                # 1. Limpeza e Tratamento da Tabela Principal
                # Import do LabelEncoder para fallback se necessário
                from sklearn.preprocessing import LabelEncoder
                
                df_proc = df_novos.copy()
                
                # DAYS_EMPLOYED = 365243 é código para "não empregado / aposentado"
                df_proc["FLAG_DAYS_EMPLOYED_ANOMALY"] = (df_proc["DAYS_EMPLOYED"] == 365243).astype(np.int8)
                df_proc["DAYS_EMPLOYED"] = df_proc["DAYS_EMPLOYED"].replace(365243, np.nan)

                # Conversão temporal (dias → anos)
                df_proc["AGE_YEARS"]          = (df_proc["DAYS_BIRTH"]        / -365.25).astype(np.float32)
                df_proc["YEARS_EMPLOYED"]     = (df_proc["DAYS_EMPLOYED"]     / -365.25).astype(np.float32)
                df_proc["YEARS_REGISTRATION"] = (df_proc["DAYS_REGISTRATION"] / -365.25).astype(np.float32)
                df_proc["YEARS_ID_PUBLISH"]   = (df_proc["DAYS_ID_PUBLISH"]   / -365.25).astype(np.float32)

                # Removemos as originais em dias
                df_proc.drop(columns=["DAYS_BIRTH", "DAYS_REGISTRATION",
                                  "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"],
                        errors="ignore", inplace=True)

                # Ratios financeiros
                df_proc["CREDIT_INCOME_RATIO"]  = df_proc["AMT_CREDIT"]  / (df_proc["AMT_INCOME_TOTAL"] + 1e-5)
                df_proc["ANNUITY_INCOME_RATIO"] = df_proc["AMT_ANNUITY"] / (df_proc["AMT_INCOME_TOTAL"] + 1e-5)
                df_proc["CREDIT_GOODS_RATIO"]   = df_proc["AMT_CREDIT"]  / (df_proc["AMT_GOODS_PRICE"]  + 1e-5)
                df_proc["INCOME_PER_MEMBER"]    = df_proc["AMT_INCOME_TOTAL"] / (df_proc["CNT_FAM_MEMBERS"] + 1e-5)
                df_proc["ANNUITY_CREDIT_RATIO"] = df_proc["AMT_ANNUITY"] / (df_proc["AMT_CREDIT"] + 1e-5)
                df_proc["LOAN_VALUE_GAP"]       = df_proc["AMT_CREDIT"]  - df_proc["AMT_GOODS_PRICE"]

                # Flags de nulo
                colunas_criticas_null = [
                    "AMT_ANNUITY", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL",
                    "YEARS_EMPLOYED", "CNT_FAM_MEMBERS", "EXT_SOURCE_1",
                    "EXT_SOURCE_2", "EXT_SOURCE_3",
                ]
                for col in colunas_criticas_null:
                    if col in df_proc.columns:
                        df_proc[f"FLAG_{col}_NULL"] = df_proc[col].isna().astype(np.int8)

                # EXT_SOURCE: scores de crédito externos
                ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
                ext_present = [c for c in ext_cols if c in df_proc.columns]
                if ext_present:
                    df_proc["EXT_SOURCE_MEAN"]    = df_proc[ext_present].mean(axis=1)
                    df_proc["EXT_SOURCE_MIN"]     = df_proc[ext_present].min(axis=1)
                    df_proc["EXT_SOURCE_MAX"]     = df_proc[ext_present].max(axis=1)
                    df_proc["EXT_SOURCE_STD"]     = df_proc[ext_present].std(axis=1)
                    df_proc["EXT_SOURCE_PRODUCT"] = df_proc[ext_present].prod(axis=1)

                # Encoding categóricas
                encoders = assets["categorical_encoders"]
                cat_cols = df_proc.select_dtypes("object").columns.tolist()
                for col in cat_cols:
                    if col in encoders:
                        le = encoders[col]
                        valid_classes = set(le.classes_)
                        df_proc[col] = df_proc[col].astype(str).apply(lambda val: val if val in valid_classes else 'unknown')
                        
                        if 'unknown' not in le.classes_:
                            le_classes = le.classes_.tolist() + ['unknown']
                            le.classes_ = np.array(le_classes)
                        
                        df_proc[col] = le.transform(df_proc[col])
                    else:
                        le = LabelEncoder()
                        df_proc[col] = le.fit_transform(df_proc[col].astype(str))

                # 2. Joins com as Tabelas Periféricas Pré-Agregadas
                for nome_tbl, tbl in tabelas_perifericas.items():
                    if tbl is not None:
                        df_proc = df_proc.merge(tbl, on="SK_ID_CURR", how="left")
                
                # Reordenar colunas e preencher nulos de colunas ausentes
                colunas_modelo = meta["features"]
                X_infer = df_proc.reindex(columns=colunas_modelo, fill_value=np.nan)
                
                # 3. Inferência em Ensemble (média das predições de cada fold)
                modelos = assets["models"]
                preds = np.zeros(len(X_infer))
                for model in modelos:
                    preds += model.predict_proba(X_infer)[:, 1] / len(modelos)
                
            # 4. Decisão Racional Comercial
            threshold = meta["threshold_otimo"]
            df_novos["Probability"] = preds
            
            # Mapeamento de decisões de negócio
            # - Bloquear: Probabilidade maior que o threshold ótimo (alto risco)
            # - Revisar: Zona cinzenta entre 80% do threshold e o threshold ótimo (risco moderado)
            # - Aprovar: Probabilidade menor que 80% do threshold (baixo risco)
            limiar_revisao = threshold * 0.8
            
            decisoes = []
            for p in preds:
                if p > threshold:
                    decisoes.append("BLOQUEAR")
                elif p >= limiar_revisao:
                    decisoes.append("REVISAR")
                else:
                    decisoes.append("APROVAR")
                    
            df_novos["Decision"] = decisoes
            
            # ── Exibição de Resultados ────────────────────────────────────────
            st.write("### 📈 Painel Analítico das Decisões")
            
            # Métricas rápidas
            col1, col2, col3 = st.columns(3)
            aprovados = decisoes.count("APROVAR")
            revisoes = decisoes.count("REVISAR")
            bloqueados = decisoes.count("BLOQUEAR")
            
            col1.metric("🟢 Clientes Aprovados", f"{aprovados} ({aprovados/n_linhas*100:.1f}%)")
            col2.metric("🟡 Em Análise (Revisar)", f"{revisoes} ({revisoes/n_linhas*100:.1f}%)")
            col3.metric("🔴 Clientes Bloqueados", f"{bloqueados} ({bloqueados/n_linhas*100:.1f}%)")
            
            # Download dos resultados
            st.write("#### 📥 Download do Lote Processado")
            csv = df_novos.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Baixar CSV com Score e Decisão",
                data=csv,
                file_name="clientes_avaliados_home_credit.csv",
                mime="text/csv",
                key="download_btn"
            )
            
            # Tabela de Amostra
            st.write("#### 👁️ Amostra dos Clientes Processados (Top 25)")
            cols_exibicao = ["SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "Probability", "Decision"]
            st.dataframe(
                df_novos[cols_exibicao].head(25).style.map(
                    lambda v: 'background-color: #2ea043; color: white;' if v == 'APROVAR' else (
                              'background-color: #d29922; color: white;' if v == 'REVISAR' else (
                              'background-color: #f78166; color: white;' if v == 'BLOQUEAR' else '')
                    ), subset=["Decision"]
                )
            )
            
            gc.collect()
            
        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
            st.info("Verifique se o CSV carregado está no mesmo formato de dados do dataset original.")
