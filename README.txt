🎓 Previsão de Evasão no Ensino Superior com Inteligência Artificial

Este repositório contém o código-fonte, os experimentos e a aplicação desenvolvida no projeto “Aplicação de Inteligência Artificial para Previsão de Evasão no Ensino Superior”, cujo objetivo é identificar estudantes em risco de evasão a partir de dados acadêmicos, socioeconômicos e demográficos.

O projeto foi desenvolvido como trabalho final da disciplina, incluindo análise de dados, modelagem preditiva, avaliação experimental rigorosa e uma aplicação web interativa.

📌 Objetivo do Projeto

Aplicar técnicas de Aprendizado de Máquina para prever o status acadêmico de estudantes.

Comparar o desempenho de dois modelos:

Multi-Layer Perceptron (MLP)

Random Forest

Avaliar os modelos em dois cenários distintos:

Multiclasse: Graduate, Dropout e Enrolled

Binário: Graduate vs Dropout (classe Enrolled removida)

Disponibilizar uma aplicação em Streamlit para uso prático e demonstrativo.

🗂️ Estrutura do Repositório
.
├── dataset_com_id.csv
├── notebooks/
│   ├── treino_multiclasse.ipynb
│   └── treino_binario.ipynb
├── resultados/
│   ├── resultados_multiclasse.csv
│   ├── resultados_binario.csv
│   ├── confusion_mlp_multiclasse.png
│   ├── confusion_rf_multiclasse.png
│   ├── confusion_mlp_binario.png
│   └── confusion_rf_binario.png
├── modelos/
│   ├── mlp_multiclasse.pkl
│   ├── rf_multiclasse.pkl
│   ├── scaler_multiclasse.pkl
│   ├── label_encoder_multiclasse.pkl
│   ├── mlp_binario.pkl
│   ├── rf_binario.pkl
│   ├── scaler_binario.pkl
│   └── label_encoder_binario.pkl
├── app/
│   └── app_streamlit.py
├── Relatório_Final___Aplicação_de_IA_para_Previsão_de_Evasão_no_Ensino_Superior.pdf
└── README.md


⚠️ Os nomes de arquivos podem variar conforme sua organização local, mas a lógica geral segue essa estrutura.

📊 Base de Dados

Origem: Higher Education Students Performance Dataset

Total de registros: 4.424 estudantes

Variável alvo (Target):

Graduate

Dropout

Enrolled

Foi adicionada uma coluna ID sequencial para permitir consultas individuais na aplicação.

Cenários de Modelagem

Multiclasse: mantém as três classes originais.

Binário: remove a classe Enrolled, focando apenas no desfecho final:

Sucesso acadêmico (Graduate)

Evasão (Dropout)

⚙️ Pré-processamento

Remoção da coluna Target antes da predição.

Normalização das variáveis numéricas com StandardScaler, aplicando a transformação:

𝑧
=
𝑥
−
𝜇
𝜎
z=
σ
x−μ
	​


onde:

𝑥
x é o valor original

𝜇
μ é a média da feature no conjunto de treino

𝜎
σ é o desvio padrão da feature no conjunto de treino

A padronização é essencial para o bom funcionamento da MLP e foi aplicada somente com parâmetros aprendidos no treino, evitando vazamento de dados.

🧠 Modelos Utilizados
🔹 Multi-Layer Perceptron (MLP)

Rede neural feedforward

Treinada com algoritmo Adam

Funções de ativação testadas: logistic, tanh, relu

🔹 Random Forest

Ensemble de árvores de decisão

Alta robustez para dados tabulares

Menor sensibilidade a outliers e escalas diferentes

🔍 Metodologia Experimental

30 execuções independentes para cada cenário

Divisão 80% treino / 20% teste com amostragem estratificada

GridSearchCV (k = 5) para otimização de hiperparâmetros

Métrica principal: Acurácia

Avaliação adicional:

Boxplots de desempenho

Matrizes de confusão

Curva de loss (MLP)

📈 Resultados Principais
Cenário	Modelo	Acurácia Média
Multiclasse	MLP	74,73%
Multiclasse	Random Forest	77,62%
Binário	MLP	90,50%
Binário	Random Forest	91,04%

📌 O Random Forest apresentou:

Maior acurácia

Menor variância entre execuções

Melhor capacidade de detectar evasão no cenário binário

🖥️ Aplicação Web (Streamlit)

A aplicação desenvolvida permite:

📊 Visão Geral do dataset

🔍 Previsão por ID (aluno existente)

➕ Previsão de Novo Aluno (simulação)

📉 Comparação de Modelos (boxplots e matrizes de confusão)

Executar a aplicação:
pip install -r requirements.txt
streamlit run app/app_streamlit.py

📄 Relatório Final

O relatório completo do projeto está disponível neste repositório:

📘 Relatório_Final___Aplicação_de_IA_para_Previsão_de_Evasão_no_Ensino_Superior.pdf

Ele descreve:

Motivação

Base de dados

Metodologia

Resultados experimentais

Aplicação desenvolvida

Conclusões e trabalhos futuros

🚀 Trabalhos Futuros

Testar técnicas de balanceamento (ex: SMOTE) para a classe Enrolled

Avaliar modelos de Gradient Boosting (XGBoost, LightGBM)

Incorporar métricas adicionais (Recall, F1-score)

Integração com sistemas acadêmicos reais

👨‍💻 Autores

Antônio Henrique Carlos

Clístenes Erasmo Alves

Everton Barbosa

Jônatas Henrique

Pedro Bullé