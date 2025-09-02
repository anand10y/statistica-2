
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Progres Bacalaureat – Analytics", layout="wide")

# ------------------------
# Helper functions
# ------------------------
REQUIRED_COLS = ["Nume", "Clasa", "Proba", "Evaluare", "Simulare", "Bacalaureat"]

@st.cache_data
def load_excel(file, sheet_name=None):
    if sheet_name is None:
        df = pd.read_excel(file, engine="openpyxl")
    else:
        df = pd.read_excel(file, sheet_name=sheet_name, engine="openpyxl")
    return df

def check_columns(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return missing

def as_number(series):
    # Convert to numeric safely (e.g., if there are strings like "9,50")
    s = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def compute_indicators(df):
    # Ensure numeric
    for col in ["Evaluare", "Simulare", "Bacalaureat"]:
        df[col] = as_number(df[col])

    # Basic counts
    n_elevi = df["Nume"].nunique()

    # Means for each stage
    mean_eval = df["Evaluare"].mean()
    mean_sim = df["Simulare"].mean()
    mean_bac = df["Bacalaureat"].mean()

    # Progress metrics
    prog_eval_bac = (df["Bacalaureat"] - df["Evaluare"]).mean()
    prog_sim_eval = (df["Simulare"] - df["Evaluare"]).mean()
    prog_bac_sim = (df["Bacalaureat"] - df["Simulare"]).mean()

    # Percent metrics (optional display)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_eval_bac = np.nanmean((df["Bacalaureat"] - df["Evaluare"]) / df["Evaluare"] * 100.0)
        pct_sim_eval = np.nanmean((df["Simulare"] - df["Evaluare"]) / df["Evaluare"] * 100.0)
        pct_bac_sim = np.nanmean((df["Bacalaureat"] - df["Simulare"]) / df["Simulare"] * 100.0)

    return {
        "n_elevi": n_elevi,
        "mean_eval": mean_eval,
        "mean_sim": mean_sim,
        "mean_bac": mean_bac,
        "prog_eval_bac": prog_eval_bac,
        "prog_sim_eval": prog_sim_eval,
        "prog_bac_sim": prog_bac_sim,
        "pct_eval_bac": pct_eval_bac,
        "pct_sim_eval": pct_sim_eval,
        "pct_bac_sim": pct_bac_sim,
    }

def plot_stage_means(means_dict, title="Medii pe etape (selecție curentă)"):
    stages = ["Evaluare", "Simulare", "Bacalaureat"]
    values = [means_dict["mean_eval"], means_dict["mean_sim"], means_dict["mean_bac"]]
    fig, ax = plt.subplots()
    ax.bar(stages, values)
    ax.set_title(title)
    ax.set_ylabel("Medie")
    ax.set_xlabel("Etapă")
    return fig

def plot_progress_bars(df, by="Clasa", agg_on="Bacalaureat- Evaluare", title="Progres mediu pe grup"):
    # agg_on can be "bac_eval", "sim_eval", "bac_sim"
    temp = df.copy()
    temp["bac_eval"] = as_number(temp["Bacalaureat"]) - as_number(temp["Evaluare"])
    temp["sim_eval"] = as_number(temp["Simulare"]) - as_number(temp["Evaluare"])
    temp["bac_sim"] = as_number(temp["Bacalaureat"]) - as_number(temp["Simulare"])

    mapping = {
        "Bacalaureat vs Evaluare": "bac_eval",
        "Simulare vs Evaluare": "sim_eval",
        "Bacalaureat vs Simulare": "bac_sim"
    }
    key = mapping.get(agg_on, "bac_eval")

    g = temp.groupby(by)[key].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    g.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(by)
    ax.set_ylabel("Progres mediu (puncte)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def plot_per_student_lines(df, title="Evoluția fiecărui elev"):
    # Line for each student within selection
    temp = df.copy()
    temp["Evaluare"] = as_number(temp["Evaluare"])
    temp["Simulare"] = as_number(temp["Simulare"])
    temp["Bacalaureat"] = as_number(temp["Bacalaureat"])

    fig, ax = plt.subplots()
    x = ["Evaluare", "Simulare", "Bacalaureat"]
    for nume, g in temp.groupby("Nume"):
        y = [g["Evaluare"].mean(), g["Simulare"].mean(), g["Bacalaureat"].mean()]
        ax.plot(x, y, marker="o", alpha=0.6)
    ax.set_title(title)
    ax.set_ylabel("Notă")
    ax.set_xlabel("Etapă")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig

def plot_hist_progress(df, metric="bac_eval", title="Distribuția progresului"):
    temp = df.copy()
    temp["bac_eval"] = as_number(temp["Bacalaureat"]) - as_number(temp["Evaluare"])
    temp["sim_eval"] = as_number(temp["Simulare"]) - as_number(temp["Evaluare"])
    temp["bac_sim"] = as_number(temp["Bacalaureat"]) - as_number(temp["Simulare"])

    fig, ax = plt.subplots()
    ax.hist(temp[metric].dropna(), bins=15)
    ax.set_title(title)
    ax.set_xlabel("Progres (puncte)")
    ax.set_ylabel("Număr elevi")
    return fig

def heatmap_progress_by_class_proba(df, title="Progres mediu (Bac - Evaluare) pe Clasă x Probă"):
    # pivot of mean progress (bac - eval)
    temp = df.copy()
    temp["bac_eval"] = as_number(temp["Bacalaureat"]) - as_number(temp["Evaluare"])
    pv = temp.pivot_table(index="Clasa", columns="Proba", values="bac_eval", aggfunc="mean")
    fig, ax = plt.subplots()
    im = ax.imshow(pv.values, aspect="auto")
    ax.set_xticks(np.arange(len(pv.columns)))
    ax.set_yticks(np.arange(len(pv.index)))
    ax.set_xticklabels(pv.columns, rotation=45, ha="right")
    ax.set_yticklabels(pv.index)
    ax.set_title(title)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            val = pv.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center")
    plt.tight_layout()
    return fig


def plot_line_by_class(df, title="Evoluția mediilor pe etape – pe clase"):
    temp = df.copy()
    temp["Evaluare"] = as_number(temp["Evaluare"])
    temp["Simulare"] = as_number(temp["Simulare"])
    temp["Bacalaureat"] = as_number(temp["Bacalaureat"])
    
    grouped = temp.groupby("Clasa")[["Evaluare", "Simulare", "Bacalaureat"]].mean()

    fig, ax = plt.subplots()
    x = ["Evaluare", "Simulare", "Bacalaureat"]
    for clasa, row in grouped.iterrows():
        y = row.values
        ax.plot(x, y, marker="o", label=clasa)
    ax.set_title(title)
    ax.set_ylabel("Medie")
    ax.set_xlabel("Etapă")
    ax.legend(title="Clasa", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig

def plot_line_by_proba(df, title="Evoluția mediilor pe etape – pe probe"):
    temp = df.copy()
    temp["Evaluare"] = as_number(temp["Evaluare"])
    temp["Simulare"] = as_number(temp["Simulare"])
    temp["Bacalaureat"] = as_number(temp["Bacalaureat"])
    
    grouped = temp.groupby("Proba")[["Evaluare", "Simulare", "Bacalaureat"]].mean()

    fig, ax = plt.subplots()
    x = ["Evaluare", "Simulare", "Bacalaureat"]
    for proba, row in grouped.iterrows():
        y = row.values
        ax.plot(x, y, marker="o", label=proba)
    ax.set_title(title)
    ax.set_ylabel("Medie")
    ax.set_xlabel("Etapă")
    ax.legend(title="Proba", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig


def plot_scatter_by_class(df, title="Distribuția notelor pe etape – pe clase"):
    temp = df.copy()
    temp["Evaluare"] = as_number(temp["Evaluare"])
    temp["Simulare"] = as_number(temp["Simulare"])
    temp["Bacalaureat"] = as_number(temp["Bacalaureat"])

    fig, ax = plt.subplots()
    x = ["Evaluare", "Simulare", "Bacalaureat"]
    for clasa, g in temp.groupby("Clasa"):
        y = [g["Evaluare"].mean(), g["Simulare"].mean(), g["Bacalaureat"].mean()]
        ax.scatter(x, y, label=clasa, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Medie")
    ax.set_xlabel("Etapă")
    ax.legend(title="Clasa", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig

def plot_scatter_by_proba(df, title="Distribuția notelor pe etape – pe probe"):
    temp = df.copy()
    temp["Evaluare"] = as_number(temp["Evaluare"])
    temp["Simulare"] = as_number(temp["Simulare"])
    temp["Bacalaureat"] = as_number(temp["Bacalaureat"])

    fig, ax = plt.subplots()
    x = ["Evaluare", "Simulare", "Bacalaureat"]
    for proba, g in temp.groupby("Proba"):
        y = [g["Evaluare"].mean(), g["Simulare"].mean(), g["Bacalaureat"].mean()]
        ax.scatter(x, y, label=proba, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Medie")
    ax.set_xlabel("Etapă")
    ax.legend(title="Proba", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig

# ------------------------
# UI
# ------------------------
st.title("📊 Progres Bacalaureat – pe clase și probe")

with st.sidebar:
    st.header("🔧 Setări / Încărcare date")
    uploaded = st.file_uploader("Încarcă fișier Excel (.xlsx) cu coloanele: Nume, Clasa, Proba, Evaluare, Simulare, Bacalaureat", type=["xlsx"])
    sheet = st.text_input("Nume foaie (opțional, lasă gol pentru prima foaie)", value="")
    show_student_plots = st.checkbox("Afișează grafice pentru fiecare elev (agregat în aceeași figură")
    show_scatter_plots = st.checkbox("Afișează grafice scatter (puncte) pentru clase și probe", value="True", value="False")
    st.markdown("---")
    st.caption("Sfat: dacă ai notele cu virgule, aplicația le normalizează automat.")

if uploaded is None:
    st.info("Încarcă un fișier Excel pentru a începe.")
    st.stop()

sheet_arg = sheet.strip() if sheet.strip() else None
df = load_excel(uploaded, sheet_name=sheet_arg)
missing = check_columns(df)
if missing:
    st.error(f"Fișierul nu are coloanele obligatorii: {missing}. Coloanele găsite sunt: {list(df.columns)}")
    st.stop()

# Clean numeric columns once
for c in ["Evaluare", "Simulare", "Bacalaureat"]:
    df[c] = as_number(df[c])

# Filters
classes = sorted(df["Clasa"].dropna().unique().tolist())
probes = sorted(df["Proba"].dropna().unique().tolist())

col1, col2, col3 = st.columns([1,1,1])
with col1:
    clasa_sel = st.multiselect("Alege clasa/le", classes, default=classes)
with col2:
    proba_sel = st.multiselect("Alege proba/probele", probes, default=probes)
with col3:
    top_n = st.number_input("Top N îmbunătățiri (per selecție)", min_value=1, max_value=1000, value=10, step=1)

mask = df["Clasa"].isin(clasa_sel) & df["Proba"].isin(proba_sel)
df_sel = df.loc[mask].copy()

if df_sel.empty:
    st.warning("Selecția curentă nu are date. Alege alte clase/probe.")
    st.stop()

# Indicators
inds = compute_indicators(df_sel)

st.subheader("📌 Indicatori generali (selecție curentă)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Număr elevi (unic)", f"{inds['n_elevi']}")
m2.metric("Medie Evaluare", f"{inds['mean_eval']:.2f}" if pd.notna(inds['mean_eval']) else "—")
m3.metric("Medie Simulare", f"{inds['mean_sim']:.2f}" if pd.notna(inds['mean_sim']) else "—")
m4.metric("Medie Bacalaureat", f"{inds['mean_bac']:.2f}" if pd.notna(inds['mean_bac']) else "—")

m5, m6, m7 = st.columns(3)
m5.metric("Progres mediu Bac vs Evaluare (puncte)", f"{inds['prog_eval_bac']:.2f}" if pd.notna(inds['prog_eval_bac']) else "—", delta=f"{inds['pct_eval_bac']:.1f}%" if pd.notna(inds['pct_eval_bac']) else None)
m6.metric("Progres mediu Sim vs Evaluare (puncte)", f"{inds['prog_sim_eval']:.2f}" if pd.notna(inds['prog_sim_eval']) else "—", delta=f"{inds['pct_sim_eval']:.1f}%" if pd.notna(inds['pct_sim_eval']) else None)
m7.metric("Progres mediu Bac vs Simulare (puncte)", f"{inds['prog_bac_sim']:.2f}" if pd.notna(inds['prog_bac_sim']) else "—", delta=f"{inds['pct_bac_sim']:.1f}%" if pd.notna(inds['pct_bac_sim']) else None)

st.markdown("---")

# Means bar
fig_means = plot_stage_means(inds, title="Medii pe etape – selecție curentă")
st.pyplot(fig_means, clear_figure=True)

# Progress by class (within selected probes)
st.subheader("🏫 Progres mediu pe clasă (în selecție)")
fig_prog_class = plot_progress_bars(df_sel, by="Clasa", agg_on="Bacalaureat vs Evaluare", title="Progres mediu (Bac - Evaluare) pe clasă")
st.pyplot(fig_prog_class, clear_figure=True)

# Progress by proba (within selected classes)
st.subheader("📚 Progres mediu pe probă (în selecție)")
fig_prog_proba = plot_progress_bars(df_sel, by="Proba", agg_on="Bacalaureat vs Evaluare", title="Progres mediu (Bac - Evaluare) pe probă")
st.pyplot(fig_prog_proba, clear_figure=True)

# Evoluție pe clase
st.subheader("📊 Evoluția mediilor pe etape – pe clase")
fig_line_class = plot_line_by_class(df_sel)
st.pyplot(fig_line_class, clear_figure=True)

# Evoluție pe probe
st.subheader("📊 Evoluția mediilor pe etape – pe probe")
fig_line_proba = plot_line_by_proba(df_sel)
st.pyplot(fig_line_proba, clear_figure=True)

if show_scatter_plots:
    # Scatter pe clase
st.subheader("🔵 Distribuția notelor pe etape – pe clase (scatter)")
fig_scatter_class = plot_scatter_by_class(df_sel)
st.pyplot(fig_scatter_class, clear_figure=True)

    # Scatter pe probe
    st.subheader("🔵 Distribuția notelor pe etape – pe probe (scatter)")
    fig_scatter_proba = plot_scatter_by_proba(df_sel)
    st.pyplot(fig_scatter_proba, clear_figure=True)



# Heatmap Class x Proba
st.subheader("🗺️ Hartă progres (Bac - Evaluare) pe Clasă x Probă (în selecție)")
fig_heat = heatmap_progress_by_class_proba(df_sel, title="Progres mediu (Bac - Evaluare) pe Clasă x Probă")
st.pyplot(fig_heat, clear_figure=True)

# Distributions
st.subheader("📈 Distribuții progres")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.caption("Bac - Evaluare")
    fig_hist1 = plot_hist_progress(df_sel, metric="bac_eval", title="Distribuție (Bac - Evaluare)")
    st.pyplot(fig_hist1, clear_figure=True)
with col_b:
    st.caption("Simulare - Evaluare")
    fig_hist2 = plot_hist_progress(df_sel, metric="sim_eval", title="Distribuție (Simulare - Evaluare)")
    st.pyplot(fig_hist2, clear_figure=True)
with col_c:
    st.caption("Bac - Simulare")
    fig_hist3 = plot_hist_progress(df_sel, metric="bac_sim", title="Distribuție (Bac - Simulare)")
    st.pyplot(fig_hist3, clear_figure=True)

# Top improvements (per selection)
st.subheader(f"🏅 Top {top_n} îmbunătățiri (Bac - Evaluare) în selecție")
df_sel["Progres_Bac_minus_Evaluare"] = as_number(df_sel["Bacalaureat"]) - as_number(df_sel["Evaluare"])
top_df = (df_sel
          .sort_values("Progres_Bac_minus_Evaluare", ascending=False)
          .loc[:, ["Nume", "Clasa", "Proba", "Evaluare", "Simulare", "Bacalaureat", "Progres_Bac_minus_Evaluare"]]
          .head(int(top_n)))
st.dataframe(top_df, use_container_width=True)

# Optional per-student plot
if show_student_plots:
    st.subheader("👩‍🎓👨‍🎓 Evoluția fiecărui elev (note medii pe elev în selecție)")
    fig_students = plot_per_student_lines(df_sel, title="Evoluția fiecărui elev (selecție curentă)")
    st.pyplot(fig_students, clear_figure=True)

st.markdown("---")
st.caption("Notă: Aplicația presupune existența coloanelor Nume, Clasa, Proba, Evaluare, Simulare, Bacalaureat. Valorile non-numerice sunt ignorate (NaN).")
