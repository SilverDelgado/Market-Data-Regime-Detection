import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import numpy as np
from datetime import datetime, timedelta
import scipy.stats

# --- Configuración de la Página ---
st.set_page_config(
    page_title="HMM Market Regime Analysis",
    page_icon="📈",
    layout="wide"
)

# --- Constantes de Rutas ---
# Rutas ajustadas según la estructura de tu proyecto.
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "dataset_raw" / "DUKASCOPY_EURUSD_15_2000-01-01_2025-01-01.csv"
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "dataset_preprocessed" / "DUKASCOPY_EURUSD_15_2000-01-01_2025-01-01_preprocessed_families_pca.csv"
EVAL_RESULTS_DIR = PROJECT_ROOT / "src" / "data" / "results"

# --- Funciones de Carga de Datos (con caché para rendimiento) ---

@st.cache_data
def load_raw_data():
    """Carga el dataset de precios original."""
    try:
        df = pd.read_csv(DATA_RAW_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de datos brutos en '{DATA_RAW_PATH}'.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el dataset original: {e}")
        return None

@st.cache_data
def load_features_data():
    """Carga el dataset con los features preprocesados (PCA, etc.)."""
    try:
        df = pd.read_csv(FEATURES_DATA_PATH)
        time_col = 'datetime' if 'datetime' in df.columns else 'timestamp'
        df[time_col] = pd.to_datetime(df[time_col])
        df.rename(columns={time_col: 'datetime'}, inplace=True)
        return df.sort_values('datetime')
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de features en '{FEATURES_DATA_PATH}'.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el dataset de features: {e}")
        return None

@st.cache_data
def find_evaluation_files():
    """Encuentra todos los archivos de evaluación JSON disponibles."""
    try:
        json_files = list(EVAL_RESULTS_DIR.glob("*_summary.json"))
        return sorted(json_files, key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception as e:
        st.error(f"Error buscando archivos de evaluación en '{EVAL_RESULTS_DIR}': {e}")
        return []

@st.cache_data
def load_json_data(file_path):
    """Carga y parsea el archivo JSON con las estadísticas del modelo."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error al cargar o parsear el archivo JSON {file_path}: {e}")
        return None

@st.cache_data
def load_states_timeseries_data(json_path):
    """Carga el CSV con la serie temporal de estados, asociado al JSON."""
    try:
        csv_filename = json_path.name.replace('_summary.json', '_states.csv')
        states_csv_path = json_path.parent / csv_filename
        
        if not states_csv_path.exists():
            st.warning(f"No se encontró el archivo de serie temporal de estados en '{states_csv_path}'.")
            return None
            
        df = pd.read_csv(states_csv_path)
        time_col = 'datetime' if 'datetime' in df.columns else 'timestamp'
        df[time_col] = pd.to_datetime(df[time_col])
        df.rename(columns={time_col: 'datetime'}, inplace=True)
        return df.sort_values('datetime')
    except Exception as e:
        st.error(f"Error cargando el archivo de serie temporal de estados {states_csv_path}: {e}")
        return None

def merge_data(raw_df, features_df, states_df):
    """Fusiona los datos de precios, features y estados."""
    if features_df is None or states_df is None:
        st.error("No se pueden fusionar los datos porque faltan los archivos de features o de estados.")
        return None
    
    try:
        # --- Paso 1: Normalizar columnas de tiempo a timezone-naive ---
        for df in [raw_df, features_df, states_df]:
            if df is not None:
                time_col = 'datetime' if 'datetime' in df.columns else 'timestamp'
                if df[time_col].dt.tz is not None:
                    df[time_col] = df[time_col].dt.tz_localize(None)
                if time_col != 'datetime':
                     df.rename(columns={time_col: 'datetime'}, inplace=True)

        # --- Paso 2: Fusionar features con estados ---
        # Estos dos archivos deben estar bien alineados
        features_with_states = pd.merge_asof(
            features_df.sort_values('datetime'),
            states_df.sort_values('datetime'),
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta('1 minute')
        )
        features_with_states.dropna(subset=['state'], inplace=True)

        # --- Paso 3: Fusionar el resultado con los datos de precios brutos ---
        if raw_df is not None:
            final_merged = pd.merge_asof(
                raw_df.sort_values('datetime'),
                features_with_states,
                on='datetime',
                direction='nearest',
                tolerance=pd.Timedelta('15 minutes')
            )
            return final_merged.dropna(subset=['state'])
        else:
            return features_with_states

    except Exception as e:
        st.error(f"Error al fusionar los datos: {e}")
        return None

# --- Funciones de Visualización ---

def plot_market_states(df, date_range, price_col='Close'):
    """Crea el gráfico principal de precios con los estados HMM coloreados."""
    if df is None or df.empty or price_col not in df.columns:
        st.warning(f"No se puede generar el gráfico. Faltan datos o la columna '{price_col}'.")
        return None
        
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    mask = (df['datetime'] >= start_date) & (df['datetime'] <= end_date)
    df_filtered = df.loc[mask].copy()

    if df_filtered.empty:
        st.warning("No hay datos en el rango de fechas seleccionado.")
        return None

    unique_states = sorted(df_filtered['state'].dropna().unique().astype(int))
    colors = px.colors.qualitative.Plotly
    state_colors = {state: colors[i % len(colors)] for i, state in enumerate(unique_states)}

    fig = go.Figure()

    df_filtered['state_change'] = df_filtered['state'].diff().ne(0).cumsum()
    for i, group in df_filtered.groupby('state_change'):
        state = int(group['state'].iloc[0])
        fig.add_trace(go.Scatter(
            x=group['datetime'], y=group[price_col], mode='lines',
            line=dict(color=state_colors[state], width=2),
            name=f'Estado {state}', legendgroup=f'Estado {state}', showlegend=False
        ))
    
    for state in unique_states:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color=state_colors[state], width=5),
            name=f'Estado {state}', legendgroup=f'Estado {state}'
        ))

    fig.update_layout(
        title="Precio del Mercado con Regímenes HMM Coloreados", height=600,
        xaxis_title="Fecha", yaxis_title="Precio", legend_title="Estados HMM", hovermode='x unified'
    )
    return fig

def plot_return_distributions(df):
    """Genera un gráfico de densidad de las distribuciones de log_return por estado."""
    if 'log_return' not in df.columns:
        st.warning("La columna 'log_return' no se encuentra en los datos para graficar la distribución.")
        return

    returns_df = df[['state', 'log_return']].dropna()
    returns_df['log_return'] = returns_df['log_return'] * 100
    unique_states = sorted(returns_df['state'].unique().astype(int))
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    min_ret, max_ret = returns_df['log_return'].quantile(0.001), returns_df['log_return'].quantile(0.999)
    x_range = np.linspace(min_ret, max_ret, 500)

    for i, state in enumerate(unique_states):
        state_returns = returns_df[returns_df['state'] == state]['log_return']
        if len(state_returns) < 2: continue
        mu, sigma = state_returns.mean(), state_returns.std()
        if sigma > 0:
            pdf_y = scipy.stats.norm.pdf(x_range, loc=mu, scale=sigma)
            fig.add_trace(go.Scatter(
                x=x_range, y=pdf_y, mode='lines',
                name=f'Estado {state} (μ={mu:.3f}, σ={sigma:.3f})',
                line=dict(color=colors[i % len(colors)])
            ))

    fig.update_layout(
        title="Distribuciones de Densidad de Retornos por Régimen de Mercado",
        xaxis_title="Retorno Logarítmico a 15 min (%)", yaxis_title="Densidad de Probabilidad",
        legend_title="Estados HMM", hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Aplicación Principal de Streamlit ---

def main():
    st.title("📈 Dashboard de Análisis de Regímenes de Mercado con HMM")
    st.markdown("---")

    # --- Carga de Datos en el Sidebar ---
    st.sidebar.header("📂 Selección de Modelo")
    json_files = find_evaluation_files()
    if not json_files:
        st.error(f"No se encontraron archivos de evaluación en la carpeta '{EVAL_RESULTS_DIR}'.")
        return

    model_names = [f.name for f in json_files]
    selected_model_name = st.sidebar.selectbox("Seleccionar ejecución de modelo:", model_names, index=0)
    selected_json_path = EVAL_RESULTS_DIR / selected_model_name

    with st.spinner("Cargando todos los datos..."):
        json_data = load_json_data(selected_json_path)
        states_timeseries_data = load_states_timeseries_data(selected_json_path)
        features_data = load_features_data()
        raw_data = load_raw_data()

    if json_data is None: st.stop()

    merged_data = merge_data(raw_data, features_data, states_timeseries_data)

    st.sidebar.header("⚙️ Controles de Visualización")
    if merged_data is not None:
        min_date, max_date = merged_data['datetime'].min().date(), merged_data['datetime'].max().date()
        default_start = max(min_date, max_date - timedelta(days=90))
        date_range = st.sidebar.date_input(
            "Seleccionar rango de fechas:", value=(default_start, max_date),
            min_value=min_date, max_value=max_date, key="date_range_selector"
        )
    else:
        st.sidebar.warning("Controles de fecha deshabilitados.")
        date_range = (datetime.now().date() - timedelta(days=90), datetime.now().date())

    st.header(f"Visualización del Modelo: `{selected_model_name}`")
    price_col_map = {'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close'} # Columnas de precio de datos brutos
    selected_price_col = st.selectbox("Seleccionar columna de precio para graficar:", list(price_col_map.keys()), index=3)
    
    if merged_data is not None and len(date_range) == 2:
        price_chart = plot_market_states(merged_data, date_range, price_col=price_col_map[selected_price_col])
        if price_chart:
            st.plotly_chart(price_chart, use_container_width=True)
    else:
        st.info("El gráfico de precios no se puede mostrar. Revisa que todos los archivos de datos necesarios existan.")

    st.header("🔬 Análisis Detallado del Modelo")
    n_states = len(json_data.get("transmat", []))
    features_used = json_data.get("features_used", [])
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Clasificación", "🔄 Transiciones", "⚖️ Features", "⏳ Duración", "📈 Retornos"
    ])

    with tab1:
        st.subheader("Tabla de Características y Clasificación de Estados")
        stats = json_data.get("conditional_stats", [])
        if stats:
            summary_list, total_obs = [], sum(s.get('count', 0) for s in stats)
            for i in range(n_states):
                state_stats = next((s for s in stats if s['state'] == i), None)
                if state_stats:
                    summary_list.append({
                        "Estado": i,
                        "Frecuencia (%)": f"{(state_stats.get('count', 0) / total_obs * 100):.2f}%" if total_obs > 0 else "N/A",
                        "Persistencia": f"{json_data.get('transmat', [[0]*n_states]*n_states)[i][i]:.3f}",
                        "Volatilidad Media (PCA)": state_stats.get("mean_pca_volatilidad", 0)
                    })
            summary_df = pd.DataFrame(summary_list).sort_values("Volatilidad Media (PCA)").reset_index(drop=True)
            st.markdown("Añade tu propia clasificación para cada estado basándote en las métricas.")
            st.data_editor(summary_df, use_container_width=True, hide_index=True, key="classification_editor")
        else:
            st.warning("No se encontraron estadísticas condicionales en el archivo JSON.")

    with tab2:
        st.subheader("Matriz de Transición de Estados")
        transmat = np.array(json_data.get("transmat", []))
        if transmat.any():
            fig = px.imshow(
                transmat, text_auto=".3f", labels=dict(x="Estado Siguiente", y="Estado Actual", color="Probabilidad"),
                x=[f"Estado {i}" for i in range(n_states)], y=[f"Estado {i}" for i in range(n_states)],
                color_continuous_scale='Blues'
            )
            fig.update_layout(title="Probabilidad de Transición entre Estados")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Distribución de Features por Estado")
        if merged_data is not None and not merged_data.empty:
            available_features = [f for f in features_used if f in merged_data.columns]
            if available_features:
                selected_feature = st.selectbox("Seleccionar Feature para visualizar:", available_features)
                fig = px.box(
                    merged_data, x="state", y=selected_feature, color="state", points="outliers",
                    labels={"state": "Estado HMM", selected_feature: selected_feature},
                    title=f"Distribución de '{selected_feature}' por Estado"
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Duración de los Episodios por Estado")
        durations = json_data.get("durations_by_state", {})
        if durations:
            state_to_show = st.selectbox("Seleccionar Estado:", list(durations.keys()), format_func=lambda x: f"Estado {x}")
            if state_to_show:
                duration_values = durations[state_to_show]
                if duration_values:
                    fig = px.histogram(
                        x=duration_values, nbins=50,
                        labels={'x': 'Duración (en intervalos de 15 min)', 'y': 'Frecuencia'},
                        title=f"Histograma de Duración para el Estado {state_to_show}"
                    )
                    mean_dur, median_dur = np.mean(duration_values), np.median(duration_values)
                    st.metric(f"Duración Promedio (Estado {state_to_show})", f"{mean_dur:.2f} intervalos")
                    st.metric(f"Duración Mediana (Estado {state_to_show})", f"{median_dur:.2f} intervalos")
                    st.plotly_chart(fig, use_container_width=True)

    with tab5:
        if merged_data is not None and not merged_data.empty:
            plot_return_distributions(merged_data)
        else:
            st.info("Esta visualización requiere datos fusionados para mostrar las distribuciones.")


if __name__ == "__main__":
    main()