# 🏆 Desafio de Data Science: Consultorias de Machine Learning

Este documento descreve as diretrizes, squads, clientes e entregáveis do desafio de Data Science.

> [!IMPORTANT]
> **Identificação do Grupo**
> - **Grupo:** Grupo 2
> - **Tema:** Tema 3 (Squad 3)
> - **Cliente:** Home Credit Finanças
> - **Foco:** Motor de Risco Transacional Multi-Tabelas

---

## 📌 Visão Geral do Desafio

Neste desafio de nível sênior, as consultorias enfrentarão problemas reais e complexos de ciência de dados, tais como:
- Extremo desbalanceamento de classes
- Variáveis anonimizadas ou com alto índice de valores nulos (exigindo imputação avançada)
- Múltiplas tabelas relacionais (necessidade de joins complexos)
- Criação de métricas de negócio muito específicas (custo do falso positivo vs. falso negativo)

O desafio é dividido em **4 Squads (Consultorias)**, onde cada uma assume um cliente diferente.

---

## 👥 As 4 Squads e seus Desafios

### Squad 1: Detecção de Fraude Silenciosa em Tempo Real
- **Cliente:** GlobalPay Solutions (Gateway de Pagamentos B2B2C)
- **O Desafio:** Bloquear transações fraudulentas de cartão de crédito no exato momento da compra, sem gerar atrito para os bons clientes.
- **Dificuldade:** 
  - Brutal desbalanceamento de classes (apenas 0,17% de fraudes).
  - Variáveis originais transformadas via PCA ($V_1, V_2, \dots, V_{28}$), impossibilitando intuição de negócio básica.
  - Dependência puramente matemática nas etapas *Explore* e *Modify*.
  - Amostragem cuidadosa (*Sample*) para não enviesar o modelo (sub-amostragem ou pesos de classe).
  - Acurácia é inútil; a métrica principal de *Assess* deve ser **Precision-Recall AUC**.
- **Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Squad 2: Manutenção Preditiva e Custo de Falha
- **Cliente:** Scania Logística Pesada (Frota de Caminhões de Transporte de Minério)
- **O Desafio:** Prever se o Sistema de Ar Comprimido (APS) dos caminhões vai falhar na próxima viagem, permitindo o envio preventivo para a oficina.
- **Dificuldade:**
  - Alto grau de valores nulos (*missing values*) que não ocorrem ao acaso, exigindo imputação inteligente em *Modify*.
  - Desafio financeiro em *Assess*: enviar um caminhão à toa para a oficina (Falso Positivo) custa **$10**, enquanto deixar o caminhão quebrar na estrada (Falso Negativo) custa **$500**.
  - Otimização do modelo focada em minimizar o **custo total**, não em acerto geral.
- **Dataset:** [APS Failure at Scania Trucks](https://www.kaggle.com/datasets/uciml/aps-failure-at-scania-trucks-data-set)

### ⭐ Squad 3 (Grupo 2 - Nosso Tema): Motor de Risco Transacional Multi-Tabelas
- **Cliente:** Home Credit Finanças (Instituição focada em desbancarizados)
- **O Desafio:** Avaliar o risco de inadimplência de clientes que não possuem histórico de crédito tradicional (sem Serasa/SPC), utilizando dados de comportamento de consumo e telecomunicações.
- **Dificuldade:**
  - Engenharia de dados pesada.
  - Dataset composto por dezenas de tabelas relacionais (histórico de pagamentos, aplicações anteriores, etc.).
  - A equipe gastará **70% do tempo nas etapas Explore e Modify**, realizando Joins e criando agregações (ex: média de atraso nos últimos 3 meses).
  - É o cenário mais próximo do dia a dia de um Cientista de Dados em um grande banco.
- **Dataset:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)

### Squad 4: Predição de Sinistro de Alta Assimetria
- **Cliente:** Porto Seguro Auto (Seguradora de Veículos)
- **O Desafio:** Prever com precisão quais motoristas irão acionar o seguro (bater o carro) no próximo ano, para ajustar o valor da apólice (prêmio).
- **Dificuldade:**
  - Dados quase totalmente anonimizados e altamente desbalanceados.
  - Grande quantidade de "ruído" (bons motoristas batem por azar, ruins não batem por sorte).
  - Em *Model*, algoritmos baseados em árvores simples costumam falhar, exigindo Gradient Boosting (XGBoost/LightGBM).
  - Em *Assess*, a métrica de avaliação exigida é o **Coeficiente de Gini Normalizado**.
- **Dataset:** [Porto Seguro’s Safe Driver Prediction](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data)

---

## 📦 Entregáveis Exigidos (Para Todas as Squads)

Para aplicar o framework **SEMMA** com rigor, a nota final será composta pela entrega de dois produtos:

### 📱 Entregável 1: A Aplicação (O Produto Final de Previsão)
O cliente deseja um produto funcional e não apenas um notebook Jupyter.
- **Interface Gráfica:** Utilizar Streamlit (recomendado por ser fácil e em Python), Gradio ou Tkinter.
- **Modelo Embarcado:** O modelo `.pkl` gerado na etapa de *Assess* deve estar importado na aplicação.
- **Funcionalidade:** O usuário deve poder fazer o upload de um arquivo `.csv` novo (simulando os clientes do dia) e o aplicativo deve retornar o mesmo arquivo contendo uma nova coluna com a **Probabilidade (Score)** e a **Decisão (Bloquear/Aprovar/Revisar)**.

### 📄 Entregável 2: A Documentação Executiva (Traduzindo o SEMMA)
Um relatório em PDF focado em negócios, estruturado sob a ótica do framework:
1. **Sumário Executivo:** O problema resolvido e o resultado principal.
2. **A Jornada (SEMMA):**
   - **Sample & Explore:** Como garantiram a confiabilidade do modelo (ex: validação temporal/OOT)? Quais foram os maiores problemas dos dados originais (nulos, outliers) e o que eles ensinaram sobre o negócio?
   - **Modify:** Quais foram as 3 melhores variáveis (*features*) criadas a partir dos dados originais que mais ajudaram na previsão?
   - **Model:** Qual foi o algoritmo vencedor após os testes de hiperparâmetros?
   - **Assess (O Valor Financeiro):** Clímax do relatório. Usando a Curva ROC, matriz de confusão e métricas de custo, provar quanto a empresa economizará usando o modelo (ex: *"Dos 100 caminhões que quebrariam, prevemos 85, gerando uma economia de R$ X mil"*).

---

## 🤝 O Pitch de Consultoria (A Defesa da Nota)

- **Evento:** *Board Meeting* (marcado para o último dia do projeto).
- **Tempo:** 15 minutos por squad.
- **Apresentação:** Voltada para o "Diretor da Empresa Cliente".
- **Requisitos:** Demonstração ao vivo do aplicativo funcionando e defesa das escolhas do SEMMA, focando no ganho financeiro mensurado em *Assess*.
- **Objetivo:** Traduzir a complexidade de um modelo de Machine Learning em valor financeiro claro para o cliente.
