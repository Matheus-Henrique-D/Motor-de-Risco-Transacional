# 🏦 Motor de Risco Transacional Multi-Tabelas
### Squad 3 — Home Credit Finanças | Hackathon de Data Science

---

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-3.x-00A86B?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow?style=for-the-badge)

</div>

---

## 📌 Contexto do Projeto

A **Home Credit Finanças** é uma instituição financeira com foco em clientes **desbancarizados** — pessoas que não possuem histórico de crédito tradicional (sem Serasa/SPC). O desafio central é avaliar o **risco de inadimplência** desses clientes com base exclusivamente em dados de comportamento de consumo, micro-transações e histórico interno.

Este projeto foi desenvolvido no contexto de um **hackathon corporativo de nível sênior**, onde a Squad 3 atua como consultoria independente de Data Science para a Home Credit Finanças.

---

## 🎯 Objetivo

Construir um **modelo preditivo de risco de crédito** capaz de:

- Identificar clientes com alta probabilidade de inadimplência
- Minimizar o **impacto financeiro** do modelo (custo de Falso Positivo vs. Falso Negativo)
- Encontrar o **threshold ótimo de negócio**, não apenas o de melhor AUC
- Suportar decisões do comitê de crédito com features interpretáveis

---

## 🗂️ Estrutura do Repositório

```
Motor-de-Risco-Transacional/
│
├── docs/                              ← Pasta de documentação e diretrizes do projeto
│   └── DESAFIO.md                     # Descrição detalhada e regras das Squads
│
├── home-credit-default-risk/          ← Dataset original do Kaggle (Local / Não versionado)
│   ├── application_train.csv          # Tabela principal — treino (307.511 clientes)
│   ├── application_test.csv           # Tabela principal — teste (48.744 clientes)
│   └── ...                            # Outros 6 arquivos CSV de comportamento transacional
│
├── output/                            ← Gerado automaticamente ao rodar o pipeline
│   ├── squad3_dashboard.png           # Dashboard visual analítico completo (PNG)
│   ├── squad3_submission.csv          # Probabilidades preditas para o conjunto de testes
│   ├── squad3_feature_importance.csv  # Ranking de relevância de todas as 248 features
│   ├── squad3_lgb_ensemble.pkl        # Modelos treinados (Ensemble de 5 folds) + Encoders
│   ├── squad3_metadata.json           # Lista de features e threshold ótimo de decisão (0.931)
│   └── features_agregadas/            # [GZIP] Agregações históricas comprimidas para o Streamlit
│
├── app.py                             ← ⭐ Interface Gráfica Interativa (Streamlit)
├── amostra_novos_clientes.csv         ← Amostra leve de 100 clientes para demonstração rápida do app
├── download_dataset.py                ← Script utilitário para download dos dados via KaggleHub
├── squad3_pipeline.py                 ← Script técnico do pipeline (treinamento e engenharia)
├── train_baseline.py                  ← Script de modelo de baseline simples
├── process_features.py                ← Script de suporte de features
└── README.md                          ← Este arquivo
```

---

## 🗺️ Arquitetura dos Dados

O dataset possui um **Snowflake Schema** — uma tabela central conectada a 6 tabelas periféricas:

```
application_train / test   ← Tabela Central (1 linha = 1 cliente)
        │
        ├── bureau.csv              (via SK_ID_CURR) → histórico em bureaus externos
        │       └── bureau_balance  (via SK_ID_BUREAU) → status mensal de cada crédito
        │
        └── previous_application   (via SK_ID_CURR) → propostas anteriores na Home Credit
                ├── POS_CASH_balance        (via SK_ID_PREV) → saldo mensal POS/cash
                ├── installments_payments   (via SK_ID_PREV) → parcelas geradas vs. pagas
                └── credit_card_balance     (via SK_ID_PREV) → comportamento de cartão
```

> ⚠️ **Regra de ouro dos joins:** As tabelas periféricas possuem relação **1:N** com o cliente. Elas **nunca** são unidas diretamente — apenas após **agregação por `SK_ID_CURR`**.

---

## 🧠 Pipeline Técnico (`squad3_pipeline.py`)

O pipeline é composto por **8 blocos sequenciais**:

| Bloco | Descrição |
|:---:|:---|
| **1** | Carregamento dos 8 CSVs + diagnóstico de nulos e desbalanceamento |
| **2** | Limpeza da tabela principal: anomalias, conversão temporal, ratios financeiros, flags de nulo, combinações de EXT_SOURCE |
| **3A** | Feature engineering — `bureau.csv`: atividade de crédito, razão dívida/crédito |
| **3B** | Feature engineering — `bureau_balance.csv`: status mensal com **janelas de 3, 6 e 12 meses** |
| **3C** | Feature engineering — `installments_payments.csv`: atraso real, taxa de pagamento, tendência |
| **3D** | Feature engineering — `POS_CASH_balance.csv`: DPD por janela temporal |
| **3E** | Feature engineering — `credit_card_balance.csv`: utilização de limite, comportamento de saque |
| **3F** | Feature engineering — `previous_application.csv`: taxa de aprovação/rejeição histórica |
| **4** | Join seguro com **assert de cardinalidade** (garante zero duplicatas) |
| **5** | Modelagem: **LightGBM** + `StratifiedKFold(5)` + `class_weight='balanced'` |
| **6** | Análise financeira: **Profit Curve** com 300 thresholds → threshold ótimo de negócio |
| **7** | Dashboard visual em dark mode com 5 gráficos analíticos |
| **8** | Exportação: submission, feature importance, dashboard |

---

## ⚙️ Decisões Técnicas Críticas

### 1. Desbalanceamento de Classes
A variável TARGET possui proporção de **~1:11** (8% inadimplentes).

```
Classe 0 (Adimplente)  : 282.686 (91,93%)
Classe 1 (Inadimplente):  24.825  (8,07%)
```

**Solução:** `class_weight='balanced'` no LightGBM + validação com AUC-ROC (não Acurácia).

---

### 2. Anti Data Leakage
Tabelas com histórico mensal são filtradas para incluir **apenas registros anteriores à aplicação**:

```python
# bureau_balance: apenas meses históricos (nunca meses futuros)
bb_hist = bb[bb["MONTHS_BALANCE"] < 0]
```

---

### 3. Nulos como Informação (Não Apenas como Problema)
Colunas com nulos estruturais geram **flags binárias** antes de qualquer imputação:

```python
# O nulo em AMT_ANNUITY do bureau significa que o contrato não tem anuidade
bureau["FLAG_AMT_ANNUITY_NULL"] = bureau["AMT_ANNUITY"].isna().astype(int)
```

---

### 4. Features Temporais por Janela
Todas as tabelas de comportamento são agregadas em **3 janelas independentes**:

| Janela | Captura |
|:---:|:---|
| **3 meses** | Comportamento recente (curto prazo) |
| **6 meses** | Tendência de médio prazo |
| **12 meses** | Padrão histórico (longo prazo) |

Feature de tendência criada automaticamente:
```python
inst_agg["inst_delay_trend"] = inst_agg["inst_delay_mean_3m"] - inst_agg["inst_delay_mean_12m"]
# Positivo = piorando | Negativo = melhorando
```

---

## 💰 Métricas de Negócio

> **Acurácia é proibida como métrica primária.** O modelo é avaliado pelo impacto financeiro real.

| Resultado | Significado | Impacto Financeiro |
|:---|:---|:---|
| **True Positive** | Inadimplente corretamente bloqueado | ✅ Perda evitada = `AMT_CREDIT` |
| **True Negative** | Adimplente corretamente aprovado | ✅ Receita = `AMT_ANNUITY × prazo` |
| **False Positive** | Bom cliente rejeitado | ❌ Receita perdida = `AMT_ANNUITY × 24 meses` |
| **False Negative** | Mau cliente aprovado | ❌ Perda = `AMT_CREDIT × 80%` |

**Premissas financeiras:**
- Taxa de recuperação em inadimplência: **20%**
- Prazo médio estimado: **24 meses**

---

## 🚀 Como Executar

### Pré-requisitos

Instale as dependências de machine learning e interface:
```bash
pip install streamlit lightgbm pandas numpy scikit-learn matplotlib
```

---

### Opção A: Executar a Aplicação Streamlit (Demonstração Rápida)
Se você deseja apenas testar a interface do aplicativo de decisão comercial imediatamente (com o modelo e as features agregadas que já estão versionados):

1. Execute no terminal:
   ```bash
   streamlit run app.py
   ```
2. A página abrirá no seu navegador. Faça o upload do arquivo [amostra_novos_clientes.csv](file:///c:/Users/natha/Documents/GitHub/Motor-de-Risco-Transacional/amostra_novos_clientes.csv) (localizado na raiz do projeto) para ver as predições de probabilidade de default e as decisões automáticas de crédito (**Aprovar / Revisar / Bloquear**).

---

### Opção B: Rodar o Pipeline Técnico (Treinamento do Zero)
Se deseja treinar os modelos novamente, recalcular a engenharia de variáveis e recalcular o threshold ótimo:

1. **Autenticação no Kaggle**: Obtenha seu token de API `kaggle.json` no site do Kaggle e mova para `C:\Users\natha\.kaggle\kaggle.json` (instruções completas em [docs/DESAFIO.md](file:///c:/Users/natha/Documents/GitHub/Motor-de-Risco-Transacional/docs/DESAFIO.md)).
2. **Download**: Execute o script utilitário de download na raiz:
   ```bash
   python download_dataset.py
   ```
3. **Mover dados**: Extraia os CSVs na pasta `home-credit-default-risk/`.
4. **Execução**: Rode o pipeline:
   ```bash
   python squad3_pipeline.py
   ```
   > ⏱️ **Tempo estimado:** 10–25 minutos.

### Saídas geradas em `output/`

* `squad3_dashboard.png`: Gráficos de Profit Curve, matriz de confusão e importância de variáveis.
* `squad3_lgb_ensemble.pkl`: O ensemble de 5 modelos de classificação LightGBM.
* `squad3_metadata.json`: O valor de threshold ótimo de negócio (`0.931`) e colunas utilizadas.

---

## 📊 Outputs do Dashboard

O arquivo `squad3_dashboard.png` contém **5 visualizações**:

1. **Profit Curve** — Custo financeiro total × threshold (com ponto ótimo marcado)
2. **Distribuição TARGET** — Visualização do desbalanceamento de classes
3. **Precision / Recall / F1** — Trade-off por threshold
4. **Matriz de Confusão** — Com valores reais (não apenas contagens)
5. **Top-25 Features** — Coloridas por origem (tabela periférica vs. cadastral)

---

## 📐 Parâmetros do Modelo (LightGBM)

```python
lgb_params = {
    "objective":         "binary",
    "metric":            "auc",
    "n_estimators":      2000,
    "learning_rate":     0.03,
    "num_leaves":        63,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "class_weight":      "balanced",
    "early_stopping":    100,
}
```

---

## 👥 Equipe

| Papel | Responsabilidade |
|:---|:---|
| **Tech Lead / Diretor** | Arquitetura do pipeline, revisão de data leakage, validação de negócio |
| **Squad 3** | Implementação do feature engineering, modelagem e análise financeira |

---

## 📚 Referências

- [Home Credit Default Risk — Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk)
- [Fonte dos Dados (Kaggle Dataset)](https://www.kaggle.com/c/home-credit-default-risk/data) — Fonte oficial para download dos dados
- [HomeCredit_columns_description.csv](./home-credit-default-risk/HomeCredit_columns_description.csv) — Dicionário completo de todas as colunas
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)

---

<div align="center">

**Squad 3 — Home Credit Finanças**
*Motor de Risco Transacional Multi-Tabelas*

</div>