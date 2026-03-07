import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import re
from datetime import datetime

# Научный стиль графиков
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.facecolor': 'white',
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'errorbar.capsize': 3,
})

# ============================================================================
# БАЗА ДАННЫХ ИОННЫХ РАДИУСОВ (Шеннон)
# ============================================================================
IONIC_RADII = {
    # Формат: (ион, заряд, КЧ): (кристаллический радиус, ионный радиус)
    # Для B-сайта используем КЧ=6, для Ba - КЧ=12, для O - фиксированное значение
    ('Ba', 2, 12): 1.61,  # ионный радиус Ba2+ в XII координации
    ('O', -2, 6): 1.4,    # по заданию
    
    # B-катионы (КЧ=6)
    ('Ce', 4, 6): 0.87,
    ('Zr', 4, 6): 0.72,
    ('Sn', 4, 6): 0.69,
    ('Ti', 4, 6): 0.605,
    ('Hf', 4, 6): 0.71,
    
    # D-допанты (акцепторы, обычно 3+, КЧ=6)
    ('Gd', 3, 6): 0.938,
    ('Sm', 3, 6): 0.958,
    ('Y', 3, 6): 0.9,
    ('In', 3, 6): 0.8,
    ('Sc', 3, 6): 0.745,
    ('Dy', 3, 6): 0.912,
    ('Ho', 3, 6): 0.901,
    ('Yb', 3, 6): 0.868,
    ('Eu', 3, 6): 0.947,
    ('Nd', 3, 6): 0.983,
    ('Pr', 3, 6): 0.99,
    ('Tb', 3, 6): 0.923,
    ('Er', 3, 6): 0.89,
    ('Tm', 3, 6): 0.88,
    ('Lu', 3, 6): 0.861,
}

# Цветовая карта для B-катионов
B_COLORS = {
    'Ce': '#E41A1C',  # красный
    'Zr': '#377EB8',  # синий
    'Sn': '#4DAF4A',  # зеленый
    'Ti': '#984EA3',  # фиолетовый
    'Hf': '#FF7F00',  # оранжевый
    'default': '#999999'  # серый
}

# Маркеры для D-допантов
D_MARKERS = {
    'Gd': 'o', 'Sm': 's', 'Y': '^', 'In': 'D', 'Sc': 'v',
    'Dy': 'P', 'Ho': '*', 'Yb': 'X', 'Eu': 'p', 'Nd': 'h',
    'Pr': 'H', 'Tb': 'd', 'Er': '8', 'Tm': 'p', 'Lu': 'P',
    'default': 'o'
}

# ============================================================================
# ФУНКЦИИ ДЛЯ ЗАГРУЗКИ И ОБРАБОТКИ ДАННЫХ
# ============================================================================
def extract_year_from_doi(doi):
    """Извлечение года из DOI (если есть в строке)"""
    if pd.isna(doi) or doi == '':
        return None
    
    # Пробуем найти год в DOI (часто есть в виде /2009/ или .2009.)
    match = re.search(r'[\./](19|20)\d{2}[\./]', str(doi))
    if match:
        return int(match.group(0).strip('./'))
    return None

def get_ionic_radius(element, charge, coordination):
    """Получение ионного радиуса из базы"""
    key = (element, charge, coordination)
    return IONIC_RADII.get(key, None)

def calculate_descriptors(row):
    """Расчет всех дескрипторов для одной строки"""
    B = row['B']
    D = row['D']
    x = row.get('x_boundary', 0)  # используем границу растворимости
    
    # Получаем радиусы
    r_Ba = IONIC_RADII.get(('Ba', 2, 12), None)
    r_O = IONIC_RADII.get(('O', -2, 6), None)
    
    # Для B и D определяем заряды по умолчанию
    # B обычно 4+, D обычно 3+
    r_B = IONIC_RADII.get((B, 4, 6), None)
    r_D = IONIC_RADII.get((D, 3, 6), None)
    
    # Если не нашли по стандартным зарядам, пробуем другие
    if r_B is None:
        # Попробуем другие возможные заряды для B
        possible_charges_B = [4, 3, 2]
        for charge in possible_charges_B:
            r_B = IONIC_RADII.get((B, charge, 6), None)
            if r_B:
                break
    
    if r_D is None:
        # Попробуем другие возможные заряды для D
        possible_charges_D = [3, 2, 4]
        for charge in possible_charges_D:
            r_D = IONIC_RADII.get((D, charge, 6), None)
            if r_D:
                break
    
    if None in [r_Ba, r_O, r_B, r_D]:
        return {
            'r_B': r_B,
            'r_D': r_D,
            'dr': None,
            'dr_rel': None,
            'r_avg_B': None,
            'tolerance_factor': None
        }
    
    # Расчеты
    dr = abs(r_D - r_B)
    dr_rel = dr / r_B if r_B != 0 else None
    r_avg_B = (1 - x) * r_B + x * r_D
    
    # Tolerance factor: t = (r_Ba + r_O) / [√2 * (r_avg_B + r_O)]
    tolerance_factor = (r_Ba + r_O) / (np.sqrt(2) * (r_avg_B + r_O))
    
    return {
        'r_B': r_B,
        'r_D': r_D,
        'dr': dr,
        'dr_rel': dr_rel,
        'r_avg_B': r_avg_B,
        'tolerance_factor': tolerance_factor
    }

def process_data(df):
    """Основная функция обработки данных"""
    df_processed = df.copy()
    
    # Переименование колонок для удобства
    column_mapping = {
        'A': 'A_element',  # обычно Ba
        'B': 'B_element',
        'D': 'D_element',
        'x(inv,in)': 'x_inv_in',
        'x(inv,end)': 'x_inv_end',
        'x(boundary)': 'x_boundary',
        'Impurity phase(s)': 'impurity',
        'x(max)': 'x_max',
        'doi': 'doi'
    }
    
    # Переименовываем только существующие колонки
    for old, new in column_mapping.items():
        if old in df_processed.columns:
            df_processed.rename(columns={old: new}, inplace=True)
    
    # Заполняем пропуски
    for col in ['x_boundary', 'x_max']:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    if 'impurity' in df_processed.columns:
        df_processed['impurity'] = df_processed['impurity'].fillna('none')
        df_processed['has_impurity'] = df_processed['impurity'] != 'none'
    
    # Извлекаем год из DOI
    if 'doi' in df_processed.columns:
        df_processed['year'] = df_processed['doi'].apply(extract_year_from_doi)
    
    # Рассчитываем дескрипторы для каждой строки
    descriptors_list = []
    for idx, row in df_processed.iterrows():
        desc = calculate_descriptors(row)
        descriptors_list.append(desc)
    
    descriptors_df = pd.DataFrame(descriptors_list)
    
    # Объединяем с исходными данными
    result = pd.concat([df_processed, descriptors_df], axis=1)
    
    # Дополнительные параметры
    if 'x_boundary' in result.columns and 'x_inv_end' in result.columns:
        result['x_rel_boundary'] = result['x_boundary'] / result['x_inv_end']
    
    if 'x_max' in result.columns and 'x_boundary' in result.columns:
        result['x_rel_max'] = result['x_max'] / result['x_boundary']
    
    return result

# ============================================================================
# ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ
# ============================================================================
def plot_solubility_vs_dr(df, ax):
    """График 1: x(boundary) vs Δr"""
    for b_element in df['B_element'].unique():
        mask = df['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        for _, row in df[mask].iterrows():
            d_element = row['D_element']
            marker = D_MARKERS.get(d_element, D_MARKERS['default'])
            
            ax.scatter(
                row['dr'], row['x_boundary'],
                color=color, marker=marker, s=80,
                alpha=0.7, edgecolors='black', linewidth=0.5,
                label=f"{b_element}-{d_element}" if _ == mask.idxmax() else ""
            )
    
    ax.set_xlabel('Δr = |r(D) - r(B)| (Å)')
    ax.set_ylabel('x(boundary)')
    ax.set_title('Solubility Limit vs Radius Difference')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_tolerance_factor(df, ax):
    """График 2: x(boundary) vs tolerance factor"""
    for b_element in df['B_element'].unique():
        mask = df['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        ax.scatter(
            df.loc[mask, 'tolerance_factor'],
            df.loc[mask, 'x_boundary'],
            color=color, s=100, alpha=0.7,
            edgecolors='black', linewidth=0.5,
            label=b_element
        )
    
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Ideal cubic (t=1)')
    ax.set_xlabel('Tolerance Factor (t) at x = x(boundary)')
    ax.set_ylabel('x(boundary)')
    ax.set_title('Solubility Limit vs Tolerance Factor')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_heatmap_dr(df):
    """График 3: Тепловая карта Δr"""
    # Создаем матрицу Δr
    b_elements = sorted(df['B_element'].unique())
    d_elements = sorted(df['D_element'].unique())
    
    dr_matrix = pd.DataFrame(index=b_elements, columns=d_elements)
    x_boundary_matrix = pd.DataFrame(index=b_elements, columns=d_elements)
    
    for _, row in df.iterrows():
        dr_matrix.loc[row['B_element'], row['D_element']] = row['dr']
        x_boundary_matrix.loc[row['B_element'], row['D_element']] = row['x_boundary']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Тепловая карта для Δr
    im = ax.imshow(dr_matrix.values.astype(float), cmap='viridis', aspect='auto')
    
    # Добавляем значения x(boundary) как текст
    for i in range(len(b_elements)):
        for j in range(len(d_elements)):
            x_val = x_boundary_matrix.iloc[i, j]
            if not pd.isna(x_val):
                ax.text(j, i, f'{x_val:.2f}', ha='center', va='center', 
                       color='white', fontweight='bold')
    
    ax.set_xticks(range(len(d_elements)))
    ax.set_yticks(range(len(b_elements)))
    ax.set_xticklabels(d_elements)
    ax.set_yticklabels(b_elements)
    ax.set_xlabel('Dopant (D)')
    ax.set_ylabel('B-site cation (B)')
    ax.set_title('Radius Difference Δr (color) with x(boundary) (text)')
    
    plt.colorbar(im, ax=ax, label='Δr (Å)')
    plt.tight_layout()
    return fig

def plot_xmax_vs_xboundary(df, ax):
    """График 4: x(max) vs x(boundary)"""
    valid = df.dropna(subset=['x_max', 'x_boundary'])
    
    # Цвет по Δr
    scatter = ax.scatter(
        valid['x_boundary'], valid['x_max'],
        c=valid['dr'], cmap='coolwarm', s=100,
        alpha=0.7, edgecolors='black', linewidth=0.5
    )
    
    # Линия y=x
    min_val = min(valid['x_boundary'].min(), valid['x_max'].min())
    max_val = max(valid['x_boundary'].max(), valid['x_max'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'k--', alpha=0.5, label='y = x')
    
    ax.set_xlabel('x(boundary)')
    ax.set_ylabel('x(max conductivity)')
    ax.set_title('Conductivity Maximum vs Solubility Limit')
    
    plt.colorbar(scatter, ax=ax, label='Δr (Å)')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_xmax_vs_tolerance(df, ax):
    """График 5: x(max) vs tolerance factor"""
    valid = df.dropna(subset=['x_max', 'tolerance_factor'])
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        ax.scatter(
            valid.loc[mask, 'tolerance_factor'],
            valid.loc[mask, 'x_max'],
            color=color, s=100, alpha=0.7,
            edgecolors='black', linewidth=0.5,
            label=b_element
        )
    
    ax.set_xlabel('Tolerance Factor (t) at x = x(max)')
    ax.set_ylabel('x(max conductivity)')
    ax.set_title('Conductivity Maximum vs Tolerance Factor')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_relative_position(df, ax):
    """График 6: x(max)/x(boundary) vs Δr/r_B"""
    valid = df.dropna(subset=['x_rel_max', 'dr_rel'])
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        ax.scatter(
            valid.loc[mask, 'dr_rel'],
            valid.loc[mask, 'x_rel_max'],
            color=color, s=100, alpha=0.7,
            edgecolors='black', linewidth=0.5,
            label=b_element
        )
    
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='x(max) = x(boundary)')
    ax.set_xlabel('Δr / r(B) (relative radius difference)')
    ax.set_ylabel('x(max) / x(boundary)')
    ax.set_title('Position of Conductivity Maximum Relative to Solubility Limit')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_pca(df):
    """График 7: PCA анализ"""
    # Подготовка признаков
    features = ['dr', 'dr_rel', 'tolerance_factor', 'x_boundary']
    
    # Удаляем строки с пропусками
    plot_df = df.dropna(subset=features + ['B_element', 'has_impurity'])
    
    if len(plot_df) < 3:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data for PCA', 
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    X = plot_df[features].values
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # По B-катионам
    for b_element in plot_df['B_element'].unique():
        mask = plot_df['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        ax1.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            color=color, s=100, alpha=0.7,
            edgecolors='black', linewidth=0.5,
            label=b_element
        )
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_title('PCA: Colored by B-site')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # По наличию примесей
    colors = {True: '#E41A1C', False: '#377EB8'}
    for has_impurity in [True, False]:
        mask = plot_df['has_impurity'] == has_impurity
        ax2.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            color=colors[has_impurity], s=100, alpha=0.7,
            edgecolors='black', linewidth=0.5,
            label=f'Impurity: {has_impurity}'
        )
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.set_title('PCA: Colored by Impurity Presence')
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def plot_impurity_phase_diagram(df, ax):
    """График 8: Диаграмма t - Δr с указанием примесных фаз"""
    # Определяем тип примесной фазы
    def get_impurity_type(impurity):
        if pd.isna(impurity) or impurity == 'none':
            return 'none'
        elif 'Ba' in str(impurity) and ('O4' in str(impurity) or 'O5' in str(impurity)):
            return 'BaD2O4'
        elif '2O3' in str(impurity) or 'O3' in str(impurity):
            return 'D2O3'
        else:
            return 'other'
    
    df = df.copy()
    df['impurity_type'] = df['impurity'].apply(get_impurity_type)
    
    # Цвета для типов примесей
    impurity_colors = {
        'none': '#4DAF4A',
        'BaD2O4': '#E41A1C',
        'D2O3': '#377EB8',
        'other': '#984EA3'
    }
    
    valid = df.dropna(subset=['tolerance_factor', 'dr'])
    
    for imp_type, color in impurity_colors.items():
        mask = valid['impurity_type'] == imp_type
        if mask.any():
            ax.scatter(
                valid.loc[mask, 'tolerance_factor'],
                valid.loc[mask, 'dr'],
                color=color, s=100, alpha=0.7,
                edgecolors='black', linewidth=0.5,
                label=imp_type
            )
    
    ax.set_xlabel('Tolerance Factor (t)')
    ax.set_ylabel('Δr (Å)')
    ax.set_title('Impurity Phase Formation: t-Δr Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_temporal_trend(df, ax):
    """График 9: x(boundary) по годам"""
    valid = df.dropna(subset=['year', 'x_boundary'])
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        ax.scatter(
            valid.loc[mask, 'year'],
            valid.loc[mask, 'x_boundary'],
            color=color, s=100, alpha=0.7,
            edgecolors='black', linewidth=0.5,
            label=b_element
        )
    
    ax.set_xlabel('Year')
    ax.set_ylabel('x(boundary)')
    ax.set_title('Solubility Limit Evolution Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

# ============================================================================
# ОСНОВНОЕ STREAMLIT-ПРИЛОЖЕНИЕ
# ============================================================================
def main():
    st.set_page_config(
        page_title="Perovskite Solubility Analyzer",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Perovskite Solubility and Conductivity Analysis")
    st.markdown("---")
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Загрузка файла
        uploaded_file = st.file_uploader(
            "Upload Excel file", 
            type=['xlsx', 'xls'],
            help="Upload your data file with perovskite compositions"
        )
        
        if uploaded_file is None:
            st.info("Please upload a data file to begin")
            st.markdown("""
            ### Expected format:
            - **A**: A-site element (usually Ba)
            - **B**: B-site element
            - **D**: Dopant element
            - **x(inv,in)**: Start of investigated range
            - **x(inv,end)**: End of investigated range
            - **x(boundary)**: Solubility limit
            - **Impurity phase(s)**: Impurity phases observed
            - **x(max)**: Concentration at max conductivity
            - **doi**: DOI reference
            """)
        
        st.markdown("---")
        st.header("📊 Plot Settings")
        
        # Настройки отображения
        show_grid = st.checkbox("Show grid", value=True)
        show_legend = st.checkbox("Show legend", value=True)
        
        # Фильтры
        st.markdown("---")
        st.header("🔍 Filters")
        
    # Основная область
    if uploaded_file is not None:
        try:
            # Загрузка данных
            df = pd.read_excel(uploaded_file)
            
            with st.spinner("Processing data..."):
                # Обработка данных
                df_processed = process_data(df)
                
                # Применение фильтров из боковой панели
                with st.sidebar:
                    # Фильтр по B-элементу
                    b_elements = ['All'] + sorted(df_processed['B_element'].unique().tolist())
                    selected_b = st.multiselect(
                        "B-site elements",
                        options=sorted(df_processed['B_element'].unique()),
                        default=sorted(df_processed['B_element'].unique())
                    )
                    
                    # Фильтр по D-элементу
                    selected_d = st.multiselect(
                        "Dopant elements",
                        options=sorted(df_processed['D_element'].unique()),
                        default=sorted(df_processed['D_element'].unique())
                    )
                    
                    # Фильтр по наличию примесей
                    impurity_filter = st.radio(
                        "Impurity phases",
                        options=['All', 'With impurities', 'Without impurities']
                    )
                
                # Применяем фильтры
                filtered_df = df_processed.copy()
                
                if selected_b:
                    filtered_df = filtered_df[filtered_df['B_element'].isin(selected_b)]
                
                if selected_d:
                    filtered_df = filtered_df[filtered_df['D_element'].isin(selected_d)]
                
                if impurity_filter == 'With impurities':
                    filtered_df = filtered_df[filtered_df['has_impurity'] == True]
                elif impurity_filter == 'Without impurities':
                    filtered_df = filtered_df[filtered_df['has_impurity'] == False]
            
            # Статистика
            st.subheader("📈 Data Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total entries", len(filtered_df))
            with col2:
                st.metric("Unique B-site", filtered_df['B_element'].nunique())
            with col3:
                st.metric("Unique Dopants", filtered_df['D_element'].nunique())
            with col4:
                imp_count = filtered_df['has_impurity'].sum() if 'has_impurity' in filtered_df.columns else 0
                st.metric("With impurities", imp_count)
            
            # Таблица с данными
            with st.expander("📋 View processed data"):
                st.dataframe(filtered_df, use_container_width=True)
                
                # Кнопка для скачивания
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download processed data as CSV",
                    csv,
                    "perovskite_data_processed.csv",
                    "text/csv"
                )
            
            st.markdown("---")
            
            # Выбор графиков
            st.subheader("📊 Visualization")
            
            plot_types = {
                "Solubility vs Radius Difference": plot_solubility_vs_dr,
                "Solubility vs Tolerance Factor": plot_tolerance_factor,
                "Δr Heatmap with x(boundary)": lambda df, ax: None,  # специальная обработка
                "x(max) vs x(boundary)": plot_xmax_vs_xboundary,
                "x(max) vs Tolerance Factor": plot_xmax_vs_tolerance,
                "Relative Position of Maximum": plot_relative_position,
                "PCA Analysis": lambda df, ax: None,  # специальная обработка
                "Impurity Phase Diagram (t-Δr)": plot_impurity_phase_diagram,
                "Temporal Trends": plot_temporal_trend,
            }
            
            selected_plots = st.multiselect(
                "Select plots to display",
                options=list(plot_types.keys()),
                default=["Solubility vs Radius Difference", "x(max) vs x(boundary)"]
            )
            
            # Создаем сетку графиков
            if selected_plots:
                n_plots = len(selected_plots)
                n_cols = min(2, n_plots)
                n_rows = (n_plots + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
                if n_plots == 1:
                    axes = np.array([axes])
                axes = axes.flatten()
                
                for i, plot_name in enumerate(selected_plots):
                    ax = axes[i]
                    
                    if plot_name == "Δr Heatmap with x(boundary)":
                        # Специальная обработка для тепловой карты
                        plt.close(fig)  # закрываем текущую фигуру
                        heatmap_fig = plot_heatmap_dr(filtered_df)
                        st.pyplot(heatmap_fig)
                        continue
                    
                    elif plot_name == "PCA Analysis":
                        # Специальная обработка для PCA
                        plt.close(fig)
                        pca_fig = plot_pca(filtered_df)
                        st.pyplot(pca_fig)
                        continue
                    
                    else:
                        # Обычные графики
                        plot_func = plot_types[plot_name]
                        plot_func(filtered_df, ax)
                        
                        # Применяем настройки
                        if not show_grid:
                            ax.grid(False)
                        if not show_legend:
                            ax.legend().remove()
                
                # Если есть обычные графики для отображения
                if any(p not in ["Δr Heatmap with x(boundary)", "PCA Analysis"] for p in selected_plots):
                    # Убираем пустые подграфики
                    for j in range(i+1, len(axes)):
                        fig.delaxes(axes[j])
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Дополнительная аналитика
            st.markdown("---")
            st.subheader("🔬 Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top combinations by solubility:**")
                if 'x_boundary' in filtered_df.columns:
                    top_sol = filtered_df.nlargest(5, 'x_boundary')[
                        ['B_element', 'D_element', 'x_boundary', 'impurity']
                    ]
                    st.dataframe(top_sol, use_container_width=True)
            
            with col2:
                st.markdown("**Correlations with descriptors:**")
                desc_cols = ['dr', 'dr_rel', 'tolerance_factor', 'x_boundary', 'x_max']
                desc_df = filtered_df[desc_cols].dropna()
                
                if len(desc_df) > 1:
                    corr = desc_df.corr()['x_boundary'].drop('x_boundary').sort_values(ascending=False)
                    
                    corr_df = pd.DataFrame({
                        'Descriptor': corr.index,
                        'Correlation with x(boundary)': corr.values
                    })
                    
                    # Форматирование
                    corr_df['Correlation with x(boundary)'] = corr_df['Correlation with x(boundary)'].apply(
                        lambda x: f'{x:.3f}'
                    )
                    
                    st.dataframe(corr_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
    
    else:
        # Плейсхолдер при отсутствии файла
        st.info("👈 Please upload an Excel file to begin analysis")
        
        # Показываем пример данных
        st.markdown("### Example of expected data format:")
        example_data = pd.DataFrame({
            'A': ['Ba', 'Ba', 'Ba'],
            'B': ['Ce', 'Ce', 'Zr'],
            'D': ['Gd', 'Sm', 'Y'],
            'x(inv,in)': [0.05, 0, 0],
            'x(inv,end)': [0.2, 0.25, 0.3],
            'x(boundary)': [0.15, 0.22, 0.27],
            'Impurity phase(s)': ['-', 'BaSm2O4', 'Y2O3'],
            'x(max)': [0.15, 0.15, 0.25],
            'doi': ['10.1016/j.jallcom.2009.05.108', 
                   '10.1016/j.jpowsour.2008.01.036',
                   '10.1016/j.ssi.2007.01.015']
        })
        
        st.dataframe(example_data, use_container_width=True)

if __name__ == "__main__":
    main()
