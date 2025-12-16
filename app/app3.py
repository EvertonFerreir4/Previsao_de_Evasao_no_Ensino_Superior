
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import warnings
import json
from collections import Counter

warnings.filterwarnings("ignore")

# --------------------------
# CONFIGURAÇÕES
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_NAME = "dataset_com_id.csv"

ARTIFACTS = {
    "multiclasse": {
        "mlp": "mlp_multiclasse.pkl",
        "rf": "rf_multiclasse.pkl",
        "scaler": "scaler_multiclasse.pkl",
        "le": "label_encoder_multiclasse.pkl",
        "mlp_stats": "mlp_stats_multiclasse.pkl",
        "rf_stats": "rf_stats_multiclasse.pkl",
        "cm_mlp": "confusion_mlp_multiclasse.png",
        "cm_rf": "confusion_rf_multiclasse.png",
        "results_csv": "resultados_multiclasse.csv"
    },
    "binario": {
        "mlp": "mlp_binario.pkl",
        "rf": "rf_binario.pkl",
        "scaler": "scaler_binario.pkl",
        "le": "label_encoder_binario.pkl",
        "mlp_stats": "mlp_stats_binario.pkl",
        "rf_stats": "rf_stats_binario.pkl",
        "cm_mlp": "confusion_mlp_binario.png",
        "cm_rf": "confusion_rf_binario.png",
        "results_csv": "resultados_binario.csv"
    }
}

def fp(fname):
    return os.path.join(BASE_DIR, fname)

# --------------------------
# UTILITÁRIAS DE CARREGAMENTO
# --------------------------
def safe_joblib_load(path):
    """Tenta carregar com joblib; retorna obj ou None (sem lançar)."""
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

@st.cache_data
def load_dataset():
    p = fp(CSV_NAME)
    if os.path.exists(p):
        return pd.read_csv(p)
    else:
        return None

@st.cache_resource
def load_artifacts():
    out = {}
    for scen, mapping in ARTIFACTS.items():
        loaded = {}
        for k, fname in mapping.items():
            if k in ("cm_mlp", "cm_rf", "results_csv"):
                loaded[k] = fp(fname)
            else:
                loaded[k] = safe_joblib_load(fp(fname))
        out[scen] = loaded
    return out

# --------------------------
# INÍCIO DO APP
# --------------------------
st.set_page_config(page_title="Previsão de Evasão — App Final", layout="wide", page_icon="🎓")
st.title("🎓 Previsão de Evasão")

# dataset e artefatos
df = load_dataset()
if df is None:
    st.error(f"CSV '{CSV_NAME}'")
    st.stop()

artifacts = load_artifacts()

FEATURES = [c for c in df.columns if c not in ("ID", "Target")]

# SIDEBAR
st.sidebar.header("Configurações")
scenario = st.sidebar.selectbox("Cenário", ["multiclasse", "binario"],
                                format_func=lambda x: "Multiclasse" if x == "multiclasse" else "Binário")
model_choice = st.sidebar.selectbox("Modelo", ["mlp", "rf"], format_func=lambda x: "MLP" if x == "mlp" else "Random Forest")
show_load_info = st.sidebar.checkbox("Mostrar status dos artefatos", value=False)

loaded = artifacts[scenario]
model = loaded.get("mlp") if model_choice == "mlp" and isinstance(loaded.get("mlp"), object) else loaded.get("rf") if model_choice == "rf" and isinstance(loaded.get("rf"), object) else loaded.get(model_choice)
scaler = loaded.get("scaler")
le = loaded.get("le")
mlp_stats = loaded.get("mlp_stats")
rf_stats = loaded.get("rf_stats")

cm_mlp_path = loaded.get("cm_mlp")
cm_rf_path = loaded.get("cm_rf")
results_csv_path = loaded.get("results_csv")

# Mostrar status rápido
if show_load_info:
    st.sidebar.subheader("Artefatos (arquivo/caminho)")
    for k, fname in ARTIFACTS[scenario].items():
        p = fp(fname)
        status = "OK" if os.path.exists(p) else "FALTANDO"
        st.sidebar.write(f"- {fname}: {status}")

# Top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de alunos", len(df))
col2.metric("Graduados (%)", f"{(df['Target'] == 'Graduate').mean() * 100:.1f}%")
col3.metric("Evadidos (%)", f"{(df['Target'] == 'Dropout').mean() * 100:.1f}%")
col4.metric("Matriculados (%)", f"{(df['Target'] == 'Enrolled').mean() * 100:.1f}%")

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Prever por ID", "Prever Novo Aluno", "Comparar Modelos"])

# --------------------------
# TAB 1 — VISÃO GERAL
# --------------------------
with tab1:
    st.header("Visão Geral do Dataset")

    # distribuição Target
    fig = px.histogram(df, x="Target", color="Target", title="Distribuição da variável Target")
    st.plotly_chart(fig, use_container_width=True)

    # principais variáveis numéricas
    st.subheader("Distribuição das Principais Features")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "ID" in num_cols:
        num_cols.remove("ID")
    top_features = num_cols[:8] 
    cols = st.columns(2)
    for i, c in enumerate(top_features):
        with cols[i % 2]:
            st.markdown(f"**{c}**")
            fig = px.histogram(df, x=c, nbins=25, title=f"Distribuição de {c}")
            st.plotly_chart(fig, use_container_width=True)

    # correlação
    st.subheader("Mapa de Correlação (features numéricas)")
    num = df.select_dtypes(include=[np.number]).drop(columns=["ID"], errors="ignore")
    if num.shape[1] > 1:
        corr = num.corr()
        fig_corr = px.imshow(corr, title="Correlação entre features numéricas")
        st.plotly_chart(fig_corr, use_container_width=True)

# --------------------------
# TAB 2 — PREVER POR ID
# --------------------------
with tab2:
    st.header("🔍 Prever Aluno por ID")
    id_min, id_max = int(df["ID"].min()), int(df["ID"].max())
    id_sel = st.number_input("ID do aluno", min_value=id_min, max_value=id_max, value=id_min)

    if st.button("Prever Aluno"):
        row = df[df["ID"] == id_sel]
        if row.empty:
            st.error("ID não encontrado.")
        else:
            st.subheader("Dados do aluno (linha selecionada)")
            st.dataframe(row.T)

            # Preparar X
            X_raw = row[FEATURES]
            # Aplica scaler se disponível
            try:
                X_scaled = scaler.transform(X_raw) if scaler is not None else X_raw.values
            except Exception:
                X_scaled = X_raw.values

            # Verifica modelo
            if model is None:
                st.error("Modelo selecionado não está carregado. Verifique os artefatos na pasta do app.")
            else:
                # predição
                try:
                    pred = model.predict(X_scaled)[0]
                    label = le.inverse_transform([pred])[0] if le is not None and hasattr(le, "inverse_transform") else pred
                    st.success(f"Predição ({model_choice.upper()}): {label}")
                except Exception as e:
                    st.error(f"Erro ao predizer: {e}")
                    pred = None

                # probabilidades (se disponíveis)
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(X_scaled)[0]
                        dfp = pd.DataFrame({"classe": le.classes_ if le is not None and hasattr(le, "classes_") else list(range(len(probs))), "prob": probs})
                        fig = px.bar(dfp, x="classe", y="prob", title="Probabilidades da Predição")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.info("predict_proba indisponível para este artefato/modelo.")

                # Loss curve MLP
                if model_choice == "mlp":
                    st.subheader("Curva de loss (MLP)")
                    loss = mlp_stats.get("loss_curve") if mlp_stats else None
                    if loss:
                        fig = px.line(y=loss, labels={"y": "loss", "x": "epoch"}, title="Loss curve (MLP)")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Loss curve MLP não disponível nos artefatos.")

# --------------------------
# TAB 3 — PREVER NOVO ALUNO
# --------------------------
with tab3:
    st.header("➕ Prever Novo Aluno (inserir manualmente)")
    with st.form("form_novo"):
        values = {}
        cols = st.columns(2)
        for i, feat in enumerate(FEATURES):
            col = cols[i % 2]
            default = df[feat].median() if pd.api.types.is_numeric_dtype(df[feat]) else df[feat].mode()[0]
            if pd.api.types.is_numeric_dtype(df[feat]):
                values[feat] = col.number_input(feat, value=float(default))
            else:
                values[feat] = col.selectbox(feat, df[feat].unique())
        ok = st.form_submit_button("Prever")

    if ok:
        X_new = pd.DataFrame([values])[FEATURES]
        try:
            X_scaled = scaler.transform(X_new) if scaler is not None else X_new.values
        except Exception:
            X_scaled = X_new.values

        if model is None:
            st.error("Modelo não carregado. Não é possível prever.")
        else:
            try:
                pred = model.predict(X_scaled)[0]
                label = le.inverse_transform([pred])[0] if le is not None and hasattr(le, "inverse_transform") else pred
                st.success(f"Predição: {label}")
            except Exception as e:
                st.error(f"Erro durante predição: {e}")

            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X_scaled)[0]
                    dfp = pd.DataFrame({"classe": le.classes_ if le is not None and hasattr(le, "classes_") else list(range(len(probs))), "prob": probs})
                    fig = px.bar(dfp, x="classe", y="prob", title="Probabilidades da Predição")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("predict_proba indisponível para este artefato/modelo.")

# --------------------------
# TAB 4 — COMPARAR MODELOS
# --------------------------
with tab4:
    st.header("📊 Comparação dos Modelos")

    colA, colB = st.columns(2)

    # Boxplots de acurácias (se CSVs disponíveis)
    with colA:
        st.subheader("Multiclasse")
        path = fp(ARTIFACTS["multiclasse"]["results_csv"])
        if os.path.exists(path):
            dfm = pd.read_csv(path)
            if {"accuracy_mlp", "accuracy_rf"}.issubset(dfm.columns):
                melt = dfm.melt(value_vars=["accuracy_mlp", "accuracy_rf"])
                fig = px.box(melt, x="variable", y="value", title="Acurácias — Multiclasse", points="all")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("CSV multiclasse encontrado mas sem colunas 'accuracy_mlp'/'accuracy_rf'.")
        else:
            st.info("CSV multiclasse não encontrado.")

    with colB:
        st.subheader("Binário")
        pathb = fp(ARTIFACTS["binario"]["results_csv"])
        if os.path.exists(pathb):
            dfb = pd.read_csv(pathb)
            if {"accuracy_mlp", "accuracy_rf"}.issubset(dfb.columns):
                meltb = dfb.melt(value_vars=["accuracy_mlp", "accuracy_rf"])
                figb = px.box(meltb, x="variable", y="value", title="Acurácias — Binário", points="all")
                st.plotly_chart(figb, use_container_width=True)
            else:
                st.info("CSV binário encontrado mas sem colunas esperadas.")
        else:
            st.info("CSV binário não encontrado.")

    # Matrizes de confusão (mantidas apenas aqui)
    st.subheader("Matrizes de Confusão")
    cols_cm = st.columns(2)

    # MLP confusion
    if cm_mlp_path and os.path.exists(cm_mlp_path):
        cols_cm[0].image(cm_mlp_path, caption="Matriz de Confusão — MLP", use_container_width=True)
    else:
        cols_cm[0].info("PNG da matriz MLP não encontrado para o cenário atual.")

    # RF confusion
    if cm_rf_path and os.path.exists(cm_rf_path):
       cols_cm[1].image(cm_rf_path, caption="Matriz de Confusão — RF", use_container_width=True)
    else:
        cols_cm[1].info("PNG da matriz RF não encontrado para o cenário atual.")
