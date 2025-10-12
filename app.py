import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import joblib


st.set_page_config(
    page_title="HMM Market States Visualization",
    page_icon="📈",
    layout="wide"
)

DATA_RAW = Path("data/dataset_raw/DUKASCOPY_EURUSD_15_2000-01-01_2025-01-01.csv")
RESULTS_DIR = Path("src/data/results")

@st.cache_data
def load_raw_data():
    try:
        df = pd.read_csv(DATA_RAW)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    except Exception as e:
        st.error(f"Error cargando dataset original: {e}")
        return None

@st.cache_data
def load_evaluation_files():
    try:
        eval_files = list(RESULTS_DIR.glob("eval_grid_best_*_states.csv"))
        return eval_files
    except Exception as e:
        st.error(f"Error buscando archivos de evaluación: {e}")
        return []

@st.cache_data
def load_evaluation_data(file_path):
    try:
        df = pd.read_csv(file_path, index_col="datetime", parse_dates=True)
        df.reset_index(inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df.sort_values('datetime')
    except Exception as e:
        st.error(f"Error cargando archivo de evaluación {file_path}: {e}")
        return None

def merge_data(raw_df, eval_df):
    try:
        raw_df = raw_df.copy()
        eval_df = eval_df.copy()
        
        raw_df['datetime'] = raw_df['timestamp']
        
        if raw_df['datetime'].dt.tz is not None:
            raw_df['datetime'] = raw_df['datetime'].dt.tz_convert('UTC')
        if eval_df['datetime'].dt.tz is not None:
            eval_df['datetime'] = eval_df['datetime'].dt.tz_convert('UTC')
        
        # Si una tiene timezone y la otra no, normalizar a sin timezone
        if raw_df['datetime'].dt.tz is not None and eval_df['datetime'].dt.tz is None:
            raw_df['datetime'] = raw_df['datetime'].dt.tz_localize(None)
        elif raw_df['datetime'].dt.tz is None and eval_df['datetime'].dt.tz is not None:
            eval_df['datetime'] = eval_df['datetime'].dt.tz_localize(None)
        
        # merge por datetime
        merged = pd.merge_asof(
            raw_df.sort_values('datetime'),
            eval_df.sort_values('datetime'),
            on='datetime',
            direction='nearest'
        )
        
        return merged
    except Exception as e:
        st.error(f"Error fusionando datos: {e}")
        return None

def create_state_colors(n_states):
    """Crear paleta de colores para los estados"""
    colors = px.colors.qualitative.Set1[:n_states]
    if n_states > len(px.colors.qualitative.Set1):
        colors.extend(px.colors.qualitative.Set2[:n_states - len(px.colors.qualitative.Set1)])
    return colors

def plot_market_states(df, date_range, price_col):
    """Crear gráfico del mercado con estados coloreados"""
    try:
        if price_col not in df.columns:
            st.error(f"La columna '{price_col}' no existe en los datos. Columnas disponibles: {list(df.columns)}")
            return None
            
        start_date, end_date = date_range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if df['datetime'].dt.tz is not None:
            start_dt = start_dt.tz_localize('UTC') if start_dt.tz is None else start_dt.tz_convert('UTC')
            end_dt = end_dt.tz_localize('UTC') if end_dt.tz is None else end_dt.tz_convert('UTC')
        else:
            start_dt = start_dt.tz_localize(None) if start_dt.tz is not None else start_dt
            end_dt = end_dt.tz_localize(None) if end_dt.tz is not None else end_dt
        
        mask = (df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)
        df_filtered = df[mask].copy()
        
        if df_filtered.empty:
            st.warning("No hay datos en el rango de fechas seleccionado")
            return None
        
        # estados únicos
        unique_states = sorted([int(state) for state in df_filtered['state'].dropna().unique()])
        colors = create_state_colors(len(unique_states))
    
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Precio EURUSD con Estados HMM', 'Probabilidades de Estados'),
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3]
        )
        
        df_filtered = df_filtered.sort_values('datetime').reset_index(drop=True)
        
        # cambios de estado
        state_changes = df_filtered['state'].ne(df_filtered['state'].shift()).cumsum()
        legend_added = {state: False for state in unique_states}
        
        for segment_id in state_changes.unique():
            segment_data = df_filtered[state_changes == segment_id]
            if len(segment_data) > 0:
                current_state = int(segment_data['state'].iloc[0])
                color_idx = unique_states.index(current_state)
                
                fig.add_trace(
                    go.Scatter(
                        x=segment_data['datetime'],
                        y=segment_data[price_col],
                        mode='lines',
                        line=dict(
                            color=colors[color_idx],
                            width=1.5
                        ),
                        name=f'Estado {current_state}',
                        legendgroup=f'state_{current_state}',
                        showlegend=not legend_added[current_state]  # Solo mostrar leyenda si no se ha añadido ya
                    ),
                    row=1, col=1
                )
                legend_added[current_state] = True
        
        # -------

        prob_columns = [col for col in df_filtered.columns if col.startswith('p_state')]
        for i, prob_col in enumerate(prob_columns):
            state_num = prob_col.replace('p_state', '')
            if int(state_num) in unique_states:
                fig.add_trace(
                    go.Scatter(
                        x=df_filtered['datetime'],
                        y=df_filtered[prob_col],
                        mode='lines',
                        line=dict(color=colors[unique_states.index(int(state_num))], width=1),
                        name=f'P(Estado {state_num})',
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            title="Visualización de Estados HMM en el Mercado EURUSD",
            height=700,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(title_text="Fecha", row=2, col=1)
        fig.update_yaxes(title_text="Precio", row=1, col=1)
        fig.update_yaxes(title_text="Probabilidad", row=2, col=1, range=[0, 1])
        
        return fig
        
    except Exception as e:
        st.error(f"Error creando gráfico: {e}")
        return None

def main():
    st.title("📈 Visualización de Estados HMM en el Mercado")
    st.markdown("---")
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        raw_data = load_raw_data()
        eval_files = load_evaluation_files()
    
    if raw_data is None:
        st.error("No se pudo cargar el dataset original")
        return
    
    if not eval_files:
        st.error("No se encontraron archivos de evaluación en src/data/results/")
        return
    
    st.sidebar.header("Configuración")
    
    model_names = [f.stem.replace("_states", "").replace("eval_", "") for f in eval_files]
    selected_model_idx = st.sidebar.selectbox(
        "Seleccionar Modelo HMM:",
        range(len(model_names)),
        format_func=lambda x: model_names[x]
    )
    
    selected_file = eval_files[selected_model_idx]
    
    with st.spinner(f"Cargando modelo {model_names[selected_model_idx]}..."):
        eval_data = load_evaluation_data(selected_file)
    
    if eval_data is None:
        st.error("No se pudo cargar el archivo de evaluación seleccionado")
        return
    
    with st.spinner("Procesando datos..."):
        merged_data = merge_data(raw_data, eval_data)
    
    if merged_data is None:
        st.error("No se pudieron fusionar los datos")
        return
    
    #-------
    st.sidebar.subheader("Rango de Fechas")
    
    min_date = merged_data['datetime'].min().date()
    max_date = merged_data['datetime'].max().date()
    
    default_end = max_date
    default_start = max(min_date, default_end - timedelta(days=160))
    
    date_range = st.sidebar.date_input(
        "Seleccionar rango:",
        value=(default_start, default_end),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) != 2:
        st.warning("Por favor selecciona un rango de fechas válido")
        return
    
    st.sidebar.subheader("Información del Modelo")
    unique_states = sorted([int(state) for state in merged_data['state'].dropna().unique()])
    st.sidebar.write(f"**Estados detectados:** {len(unique_states)}")
    st.sidebar.write(f"**Estados:** {unique_states}")
    

    st.sidebar.subheader("Configuración de Precio")
    
    price_mapping = {
        'close': 'close_x',  # close de raw y csv son iguales cogemos x
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close_csv': 'close_y',
    }
    
    available_price_cols = []
    price_display_names = []
    
    for display_name, actual_col in price_mapping.items():
        if actual_col in merged_data.columns:
            available_price_cols.append(actual_col)
            price_display_names.append(display_name)
    
    if not available_price_cols:
        for col in merged_data.columns:
            if any(price in col.lower() for price in ['close', 'open', 'high', 'low']):
                if col not in ['state'] and not col.startswith('p_state'):
                    available_price_cols.append(col)
                    price_display_names.append(col)
    
    if not available_price_cols:
        st.error("No se encontró ninguna columna de precio válida")
        return
    
    default_idx = 0
    if 'close' in price_display_names:
        default_idx = price_display_names.index('close')
    elif 'open' in price_display_names:
        default_idx = price_display_names.index('open')
    
    selected_display = st.sidebar.selectbox(
        "Columna de precio:",
        price_display_names,
        index=default_idx,
        help="'close' es recomendado para análisis de fin de período."
    )
    selected_idx = price_display_names.index(selected_display)
    price_col = available_price_cols[selected_idx]
    
    st.subheader(f"Modelo: {model_names[selected_model_idx]}")
    st.metric("Estados detectados", len(unique_states))
    
    with st.spinner("Generando gráfico..."):
        fig = plot_market_states(merged_data, date_range, price_col)
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        start_date, end_date = date_range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        if merged_data['datetime'].dt.tz is not None:
            start_dt = start_dt.tz_localize('UTC') if start_dt.tz is None else start_dt.tz_convert('UTC')
            end_dt = end_dt.tz_localize('UTC') if end_dt.tz is None else end_dt.tz_convert('UTC')
        else:
            start_dt = start_dt.tz_localize(None) if start_dt.tz is not None else start_dt
            end_dt = end_dt.tz_localize(None) if end_dt.tz is not None else end_dt
        
        mask = (merged_data['datetime'] >= start_dt) & (merged_data['datetime'] <= end_dt)
        period_data = merged_data[mask]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Observaciones en el período",
                f"{len(period_data):,}"
            )
        
        with col2:
            if len(period_data) > 1:
                price_change = period_data[price_col].iloc[-1] - period_data[price_col].iloc[0]
                st.metric(
                    "Cambio de precio",
                    f"{price_change:.4f}",
                    delta=f"{(price_change/period_data[price_col].iloc[0]*100):.2f}%"
                )
            else:
                st.metric("Cambio de precio", "N/A")
        
        with col3:
            volatility = period_data[price_col].std()
            st.metric(
                "Volatilidad (período)",
                f"{volatility:.4f}"
            )
        st.subheader("Resumen por Estado (período seleccionado)")
        
        summary_data = []
        period_unique_states = sorted([int(state) for state in period_data['state'].dropna().unique()])
        for state in period_unique_states:
            state_data = period_data[period_data['state'] == state]
            if len(state_data) > 0:
                summary_data.append({
                    'Estado': state,
                    'Observaciones': len(state_data),
                    'Porcentaje': f"{len(state_data)/len(period_data)*100:.1f}%",
                    'Precio Promedio': f"{state_data[price_col].mean():.4f}",
                    'Volatilidad': f"{state_data[price_col].std():.4f}",
                    'Precio Min': f"{state_data[price_col].min():.4f}",
                    'Precio Max': f"{state_data[price_col].max():.4f}"
                })
        
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

if __name__ == "__main__":
    main()