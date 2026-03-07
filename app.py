import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.signal import find_peaks
import re
from datetime import datetime
import openpyxl

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
    # Проверяем наличие необходимых ключей
    if 'B_element' not in row.index or 'D_element' not in row.index:
        return {
            'r_B': None,
            'r_D': None,
            'dr': None,
            'dr_rel': None,
            'r_avg_B': None,
            'tolerance_factor': None
        }
    
    B = row['B_element']
    D = row['D_element']
    
    # Получаем x_boundary, если есть
    x = row.get('x_boundary', 0)
    if pd.isna(x):
        x = 0
    
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
    
    # Выводим названия колонок для отладки
    print("Original columns:", df_processed.columns.tolist())
    
    # Создаем маппинг на основе первых строк или стандартных названий
    column_mapping = {}
    
    # Пробуем определить колонки по их содержимому или позиции
    # Обычно в таких таблицах порядок колонок фиксирован:
    # A, B, D, x(inv,in), x(inv,end), x(boundary), Impurity phase(s), x(max), doi
    
    expected_order = [
        'A_element', 'B_element', 'D_element', 
        'x_inv_in', 'x_inv_end', 'x_boundary', 
        'impurity', 'x_max', 'doi'
    ]
    
    # Если колонок ровно 9, используем позиционное соответствие
    if len(df_processed.columns) == 9:
        for i, col_name in enumerate(expected_order):
            column_mapping[df_processed.columns[i]] = col_name
    else:
        # Пробуем найти соответствия по частичному совпадению
        for col in df_processed.columns:
            col_lower = str(col).lower()
            
            if 'a' == col_lower or 'a-site' in col_lower or 'site a' in col_lower:
                column_mapping[col] = 'A_element'
            elif 'b' == col_lower or 'b-site' in col_lower or 'site b' in col_lower:
                column_mapping[col] = 'B_element'
            elif 'd' == col_lower or 'dopant' in col_lower:
                column_mapping[col] = 'D_element'
            elif 'x(inv,in)' in col_lower or 'inv in' in col_lower or 'x_inv_in' in col_lower:
                column_mapping[col] = 'x_inv_in'
            elif 'x(inv,end)' in col_lower or 'inv end' in col_lower or 'x_inv_end' in col_lower:
                column_mapping[col] = 'x_inv_end'
            elif 'x(boundary)' in col_lower or 'boundary' in col_lower:
                column_mapping[col] = 'x_boundary'
            elif 'impurity' in col_lower or 'phase' in col_lower:
                column_mapping[col] = 'impurity'
            elif 'x(max)' in col_lower or 'max' in col_lower:
                column_mapping[col] = 'x_max'
            elif 'doi' in col_lower:
                column_mapping[col] = 'doi'
    
    # Применяем маппинг
    df_processed.rename(columns=column_mapping, inplace=True)
    
    # Проверяем, что все необходимые колонки есть
    required_cols = ['B_element', 'D_element']
    missing_cols = [col for col in required_cols if col not in df_processed.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.write("Available columns:", df_processed.columns.tolist())
        # Пробуем использовать первые колонки как A, B, D если они есть
        if len(df_processed.columns) >= 3:
            st.warning("Trying to use first three columns as A, B, D elements")
            col_names = list(df_processed.columns)
            df_processed.rename(columns={
                col_names[0]: 'A_element',
                col_names[1]: 'B_element',
                col_names[2]: 'D_element'
            }, inplace=True)
    
    # Заполняем пропуски
    for col in ['x_boundary', 'x_max']:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Обработка примесей
    if 'impurity' in df_processed.columns:
        df_processed['impurity'] = df_processed['impurity'].fillna('none').astype(str)
        df_processed['impurity'] = df_processed['impurity'].replace(['-', '--', ''], 'none')
        df_processed['has_impurity'] = df_processed['impurity'] != 'none'
    else:
        df_processed['impurity'] = 'none'
        df_processed['has_impurity'] = False
    
    # Извлекаем год из DOI
    if 'doi' in df_processed.columns:
        df_processed['year'] = df_processed['doi'].apply(extract_year_from_doi)
    else:
        df_processed['year'] = None
    
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
        with np.errstate(divide='ignore', invalid='ignore'):
            result['x_rel_boundary'] = result['x_boundary'] / result['x_inv_end']
    
    if 'x_max' in result.columns and 'x_boundary' in result.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            result['x_rel_max'] = result['x_max'] / result['x_boundary']
            result['x_diff_norm'] = (result['x_max'] - result['x_boundary']) / result['x_boundary']
    
    return result

# ============================================================================
# НОВЫЕ ФУНКЦИИ ДЛЯ РАСЧЕТА СТАТИСТИКИ
# ============================================================================
def calculate_correlations(df, features):
    """Расчет корреляций Пирсона и Спирмена с p-value"""
    corr_data = []
    for i, f1 in enumerate(features):
        for f2 in features[i+1:]:
            valid = df[[f1, f2]].dropna()
            if len(valid) > 3:
                try:
                    pearson_r, pearson_p = stats.pearsonr(valid[f1], valid[f2])
                    spearman_r, spearman_p = stats.spearmanr(valid[f1], valid[f2])
                    corr_data.append({
                        'Feature 1': f1,
                        'Feature 2': f2,
                        'Pearson r': f'{pearson_r:.3f}',
                        'Pearson p': f'{pearson_p:.3e}',
                        'Spearman ρ': f'{spearman_r:.3f}',
                        'Spearman p': f'{spearman_p:.3e}',
                        'N': len(valid)
                    })
                except:
                    continue
    return pd.DataFrame(corr_data)

def calculate_t_series(row, x_points=50):
    """Рассчитать tolerance factor для ряда x"""
    if pd.isna(row.get('r_B')) or pd.isna(row.get('r_D')):
        return None
    
    r_Ba = 1.61
    r_O = 1.4
    r_B = row['r_B']
    r_D = row['r_D']
    x_max = row.get('x_boundary', 0.5)
    if pd.isna(x_max) or x_max <= 0:
        x_max = 0.5
    
    x_vals = np.linspace(0, min(x_max, 0.8), x_points)
    r_avg = (1 - x_vals) * r_B + x_vals * r_D
    t_vals = (r_Ba + r_O) / (np.sqrt(2) * (r_avg + r_O))
    
    return x_vals, t_vals

def feature_importance_analysis(df):
    """Random Forest анализ важности признаков"""
    # Подготовка данных
    plot_df = df.dropna(subset=['x_boundary', 'dr', 'tolerance_factor', 'B_element'])
    
    if len(plot_df) < 10:
        return None, None
    
    # One-hot encoding для B_element
    X = pd.get_dummies(plot_df[['dr', 'tolerance_factor', 'B_element']], 
                       columns=['B_element'])
    y = plot_df['x_boundary']
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Важность признаков
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Оценка качества
    r2 = rf.score(X, y)
    
    return importance_df, r2

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
# НОВЫЕ ФУНКЦИИ ДЛЯ ГРАФИКОВ (ПРЕДЛОЖЕННЫЕ УЛУЧШЕНИЯ)
# ============================================================================
def plot_b_site_statistics(df):
    """График 10: Статистика по B-элементам (столбчатая диаграмма с ошибками)"""
    stats_df = df.groupby('B_element')['x_boundary'].agg(['mean', 'median', 'count', 'std']).round(3)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Столбчатая диаграмма со стандартным отклонением
    b_sites = stats_df.index
    x_pos = np.arange(len(b_sites))
    
    ax1.bar(x_pos, stats_df['mean'], yerr=stats_df['std'], 
            capsize=5, color=[B_COLORS.get(b, B_COLORS['default']) for b in b_sites],
            edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(b_sites)
    ax1.set_ylabel('Mean x(boundary)')
    ax1.set_title('Average Solubility by B-site')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Количество образцов
    ax2.bar(x_pos, stats_df['count'], 
            color=[B_COLORS.get(b, B_COLORS['default']) for b in b_sites],
            edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(b_sites)
    ax2.set_ylabel('Number of samples')
    ax2.set_title('Sample Count by B-site')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, stats_df

def plot_top_dopants_violin(df):
    """График 11: Violin plot для топ-10 допантов по растворимости"""
    # Берем топ-10 допантов по медиане
    dopant_stats = df.groupby('D_element')['x_boundary'].agg(['median', 'count'])
    top_dopants = dopant_stats.nlargest(10, 'median').index
    plot_df = df[df['D_element'].isin(top_dopants)].dropna(subset=['x_boundary'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Порядок для отображения
    order = dopant_stats.loc[top_dopants].sort_values('median', ascending=False).index
    
    # Violin plot
    sns.violinplot(data=plot_df, x='D_element', y='x_boundary', 
                   order=order, ax=ax, cut=0)
    
    # Добавляем количество образцов
    for i, d in enumerate(order):
        count = dopant_stats.loc[d, 'count']
        ax.text(i, ax.get_ylim()[1]*0.95, f'n={int(count)}', 
                ha='center', fontsize=9)
    
    ax.set_xlabel('Dopant Element')
    ax.set_ylabel('x(boundary)')
    ax.set_title('Top 10 Dopants by Solubility (Violin Plot)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_xmax_vs_boundary_histogram(df):
    """График 12: Гистограмма распределения (x_max - x_boundary)/x_boundary"""
    valid = df.dropna(subset=['x_max', 'x_boundary'])
    diff = (valid['x_max'] - valid['x_boundary']) / valid['x_boundary']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n, bins, patches = ax.hist(diff, bins=20, edgecolor='black', 
                                color='skyblue', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, 
               label='x(max) = x(boundary)')
    ax.axvline(x=diff.median(), color='blue', linestyle=':', linewidth=2,
               label=f'Median = {diff.median():.3f}')
    
    # Добавляем проценты
    within_10pct = (abs(diff) < 0.1).mean()
    ax.text(0.05, 0.95, f'Within ±10%: {within_10pct:.1%}', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('(x_max - x_boundary) / x_boundary')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Conductivity Maximum Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_correlation_heatmap(df):
    """График 13: Тепловая карта корреляций"""
    features = ['dr', 'dr_rel', 'tolerance_factor', 'x_boundary', 'x_max']
    corr_df = df[features].dropna()
    
    if len(corr_df) < 5:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data for correlation', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Расчет корреляций
    pearson_corr = corr_df.corr(method='pearson')
    spearman_corr = corr_df.corr(method='spearman')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pearson
    sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, ax=ax1,
                cbar_kws={'label': 'Pearson r'})
    ax1.set_title('Pearson Correlations')
    
    # Spearman
    sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, ax=ax2,
                cbar_kws={'label': 'Spearman ρ'})
    ax2.set_title('Spearman Correlations')
    
    plt.tight_layout()
    return fig

def plot_publication_matrix(df):
    """График 14: Тепловая карта количества публикаций по B-D парам"""
    pub_matrix = df.groupby(['B_element', 'D_element']).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(pub_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Number of publications'},
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Dopant (D)')
    ax.set_ylabel('B-site (B)')
    ax.set_title('Research Intensity: B-D Combinations')
    
    plt.tight_layout()
    return fig, pub_matrix

def plot_distribution_kde(df):
    """График 15: Распределение x_boundary (гистограмма + KDE по B)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Общее распределение
    ax1.hist(df['x_boundary'].dropna(), bins=20, edgecolor='black', 
             alpha=0.7, color='gray', density=True)
    sns.kdeplot(data=df['x_boundary'].dropna(), ax=ax1, color='red', 
                linewidth=2, label='KDE')
    ax1.set_xlabel('x(boundary)')
    ax1.set_ylabel('Density')
    ax1.set_title('Overall Distribution of Solubility Limits')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # По B-элементам
    for b_element in df['B_element'].unique():
        data = df[df['B_element'] == b_element]['x_boundary'].dropna()
        if len(data) > 1:
            sns.kdeplot(data=data, label=b_element, ax=ax2, 
                        linewidth=2, color=B_COLORS.get(b_element, 'gray'))
    
    ax2.set_xlabel('x(boundary)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution by B-site')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_shift_vs_dr_bubble(df):
    """График 16: Пузырьковая диаграмма зависимости смещения от Δr"""
    valid = df.dropna(subset=['x_max', 'x_boundary', 'dr', 'tolerance_factor'])
    
    if len(valid) < 3:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Пузырьковая диаграмма
    scatter = ax.scatter(
        valid['dr'],
        valid['x_max'] - valid['x_boundary'],
        s=valid['x_boundary'] * 1000,  # размер = растворимость
        c=valid['tolerance_factor'],   # цвет = tolerance factor
        alpha=0.6,
        cmap='coolwarm',
        edgecolors='black',
        linewidth=0.5
    )
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Добавить подписи для точек с большим отклонением
    valid_sorted = valid.nlargest(5, 'x_diff_norm')
    for idx, row in valid_sorted.iterrows():
        ax.annotate(f"{row['B_element']}-{row['D_element']}", 
                    (row['dr'], row['x_max'] - row['x_boundary']),
                    fontsize=8, ha='center')
    
    ax.set_xlabel('Δr (Å)')
    ax.set_ylabel('x_max - x_boundary')
    ax.set_title('Shift of Conductivity Maximum from Solubility Limit\n(Bubble size = x_boundary)')
    
    plt.colorbar(scatter, ax=ax, label='Tolerance Factor')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_tolerance_evolution(df):
    """График 17: Эволюция tolerance factor с допированием"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Выбираем до 10 случайных систем для наглядности
    sample_size = min(10, len(df))
    sampled_df = df.sample(sample_size, random_state=42) if sample_size > 0 else df
    
    for idx, row in sampled_df.iterrows():
        result = calculate_t_series(row)
        if result:
            x_vals, t_vals = result
            label = f"{row['B_element']}-{row['D_element']}"
            if not pd.isna(row.get('x_boundary')):
                # Отмечаем точку границы растворимости
                boundary_idx = np.argmin(np.abs(x_vals - row['x_boundary']))
                ax.plot(x_vals, t_vals, linewidth=2, alpha=0.7, label=label)
                ax.plot(x_vals[boundary_idx], t_vals[boundary_idx], 'ro', 
                       markersize=8, alpha=0.8)
            else:
                ax.plot(x_vals, t_vals, linewidth=2, alpha=0.7, label=label)
    
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, 
               label='Ideal cubic (t=1)', linewidth=2)
    ax.set_xlabel('Dopant concentration x')
    ax.set_ylabel('Tolerance Factor')
    ax.set_title('Evolution of Tolerance Factor with Doping\n(Red dots = solubility limit)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_critical_dr_threshold(df):
    """График 18: Критический Δr для образования примесей"""
    valid = df.dropna(subset=['dr', 'has_impurity']).sort_values('dr')
    
    if len(valid) < 5:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Кумулятивная доля примесей
    x_vals = valid['dr'].values
    y_vals = valid['has_impurity'].astype(int).cumsum() / np.arange(1, len(valid)+1)
    
    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='Cumulative fraction')
    ax.fill_between(x_vals, 0, y_vals, alpha=0.3, color='blue')
    
    # Находим точку перегиба (порог)
    y_diff = np.diff(y_vals)
    if len(y_diff) > 5:
        # Сглаживаем для поиска пика
        from scipy.ndimage import gaussian_filter1d
        y_diff_smooth = gaussian_filter1d(y_diff, sigma=2)
        threshold_idx = np.argmax(y_diff_smooth[:len(y_diff_smooth)//2])  # ищем в первой половине
        if threshold_idx < len(x_vals)-1:
            ax.axvline(x=x_vals[threshold_idx], color='red', linestyle='--', 
                       linewidth=2, label=f'Threshold Δr ≈ {x_vals[threshold_idx]:.3f} Å')
    
    ax.set_xlabel('Δr (Å)')
    ax.set_ylabel('Cumulative fraction with impurities')
    ax.set_title('Critical Radius Difference for Impurity Formation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_dopant_comparison_boxplot(df, selected_dopant):
    """График 19: Сравнение одного допанта на разных B-сайтах"""
    plot_df = df[df['D_element'] == selected_dopant].dropna(subset=['x_boundary'])
    
    if len(plot_df) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f'Insufficient data for {selected_dopant}', 
                ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Boxplot
    bp = ax.boxplot([plot_df[plot_df['B_element'] == b]['x_boundary'] 
                     for b in plot_df['B_element'].unique()],
                    labels=plot_df['B_element'].unique(),
                    patch_artist=True)
    
    # Раскрашиваем боксы
    for i, b in enumerate(plot_df['B_element'].unique()):
        bp['boxes'][i].set_facecolor(B_COLORS.get(b, 'lightgray'))
        bp['boxes'][i].set_alpha(0.7)
    
    # Наложить точки (strip plot)
    for b in plot_df['B_element'].unique():
        data = plot_df[plot_df['B_element'] == b]['x_boundary']
        x_pos = list(plot_df['B_element'].unique()).index(b) + 1
        # Добавляем случайный шум для x-позиции
        x_jittered = np.random.normal(x_pos, 0.05, len(data))
        ax.scatter(x_jittered, data, color='red', s=50, 
                   alpha=0.5, zorder=3)
    
    ax.set_xlabel('B-site')
    ax.set_ylabel(f'x(boundary) for {selected_dopant} doping')
    ax.set_title(f'Solubility of {selected_dopant} on Different B-sites')
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig

def plot_feature_importance(df):
    """График 20: Важность признаков Random Forest"""
    importance_df, r2 = feature_importance_analysis(df)
    
    if importance_df is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data for feature importance', 
                ha='center', va='center')
        return fig, None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Горизонтальная столбчатая диаграмма
    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['importance'], 
            color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Random Forest: Factors Affecting Solubility\n(R² = {r2:.3f})')
    ax.invert_yaxis()  # Чтобы самое важное было сверху
    ax.grid(True, alpha=0.3, axis='x')
    
    # Добавляем значения на концах столбцов
    for i, v in enumerate(importance_df['importance']):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    return fig, importance_df

def plot_goldschmidt_bubble(df):
    """График 21: t-Δr фазовая диаграмма с пузырьками"""
    valid = df.dropna(subset=['tolerance_factor', 'dr', 'x_boundary'])
    
    if len(valid) < 3:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Основные точки
    scatter = ax.scatter(
        valid['tolerance_factor'],
        valid['dr'],
        s=valid['x_boundary'] * 2000,  # размер = растворимость
        c=valid['year'] if 'year' in valid.columns and valid['year'].notna().any() else valid['x_boundary'],
        alpha=0.6,
        cmap='viridis',
        edgecolors='black',
        linewidth=0.5
    )
    
    # Выделяем точки с примесями
    impurity_points = valid[valid['has_impurity']]
    if len(impurity_points) > 0:
        ax.scatter(
            impurity_points['tolerance_factor'],
            impurity_points['dr'],
            s=impurity_points['x_boundary'] * 2000,
            facecolors='none',
            edgecolors='red',
            linewidth=2,
            label='With impurities'
        )
    
    # Добавляем подписи для некоторых точек
    for idx, row in valid.nlargest(5, 'x_boundary').iterrows():
        ax.annotate(f"{row['B_element']}-{row['D_element']}", 
                    (row['tolerance_factor'], row['dr']),
                    fontsize=8, ha='center')
    
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel('Tolerance Factor')
    ax.set_ylabel('Δr (Å)')
    ax.set_title('Goldschmidt Diagram with Solubility Information\n(Bubble size = x_boundary)')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Year' if 'year' in valid.columns and valid['year'].notna().any() else 'x(boundary)')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

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
            return
        
        st.markdown("---")
        st.header("📊 Plot Settings")
        
        # Настройки отображения
        show_grid = st.checkbox("Show grid", value=True)
        show_legend = st.checkbox("Show legend", value=True)
        
        # Фильтры
        st.markdown("---")
        st.header("🔍 Filters")
        
        # Загрузка и обработка данных
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            with st.spinner("Processing data..."):
                df_processed = process_data(df)
                
                # Фильтр по B-элементу
                if 'B_element' in df_processed.columns:
                    selected_b = st.multiselect(
                        "B-site elements",
                        options=sorted(df_processed['B_element'].unique()),
                        default=sorted(df_processed['B_element'].unique())
                    )
                else:
                    selected_b = []
                    st.error("B_element column not found")
                
                # Фильтр по D-элементу
                if 'D_element' in df_processed.columns:
                    selected_d = st.multiselect(
                        "Dopant elements",
                        options=sorted(df_processed['D_element'].unique()),
                        default=sorted(df_processed['D_element'].unique())
                    )
                else:
                    selected_d = []
                    st.error("D_element column not found")
                
                # Фильтр по наличию примесей
                impurity_filter = st.radio(
                    "Impurity phases",
                    options=['All', 'With impurities', 'Without impurities']
                )
                
                # Применяем фильтры
                filtered_df = df_processed.copy()
                
                if selected_b and 'B_element' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['B_element'].isin(selected_b)]
                
                if selected_d and 'D_element' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['D_element'].isin(selected_d)]
                
                if impurity_filter == 'With impurities' and 'has_impurity' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['has_impurity'] == True]
                elif impurity_filter == 'Without impurities' and 'has_impurity' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['has_impurity'] == False]
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
    
    # Основная область
    if uploaded_file is not None and len(filtered_df) > 0:
        # Статистика
        st.subheader("📈 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total entries", len(filtered_df))
        with col2:
            st.metric("Unique B-site", filtered_df['B_element'].nunique() if 'B_element' in filtered_df.columns else 0)
        with col3:
            st.metric("Unique Dopants", filtered_df['D_element'].nunique() if 'D_element' in filtered_df.columns else 0)
        with col4:
            imp_count = filtered_df['has_impurity'].sum() if 'has_impurity' in filtered_df.columns else 0
            st.metric("With impurities", imp_count)
        
        # ============================================================================
        # НОВЫЕ МЕТРИКИ И СТАТИСТИКА (Предложение 1, 2, 3)
        # ============================================================================
        st.markdown("---")
        st.subheader("📊 Quick Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Solubility Statistics by B-site**")
            if 'x_boundary' in filtered_df.columns and 'B_element' in filtered_df.columns:
                b_stats = filtered_df.groupby('B_element')['x_boundary'].agg(['mean', 'median', 'count', 'std']).round(3)
                st.dataframe(b_stats, use_container_width=True)
        
        with col2:
            st.markdown("**Top 5 Dopants by Median Solubility**")
            if 'x_boundary' in filtered_df.columns and 'D_element' in filtered_df.columns:
                d_stats = filtered_df.groupby('D_element')['x_boundary'].agg(['median', 'count']).round(3)
                top_d = d_stats.nlargest(5, 'median')
                st.dataframe(top_d, use_container_width=True)
        
        with col3:
            st.markdown("**Conductivity Maximum Position**")
            if 'x_max' in filtered_df.columns and 'x_boundary' in filtered_df.columns:
                valid = filtered_df.dropna(subset=['x_max', 'x_boundary'])
                if len(valid) > 0:
                    diff = (valid['x_max'] - valid['x_boundary']) / valid['x_boundary']
                    within_10pct = (abs(diff) < 0.1).mean()
                    st.metric("Within ±10% of boundary", f"{within_10pct:.1%}")
                    st.metric("Median relative position", f"{diff.median():.3f}")
                    st.metric("Samples with x(max) > x(boundary)", f"{(diff > 0).mean():.1%}")
        
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
        
        # ============================================================================
        # ВКЛАДКИ С ГРАФИКАМИ
        # ============================================================================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Basic Statistics", 
            "🔬 Solubility Analysis", 
            "⚡ Conductivity Analysis",
            "📈 Advanced Visualization",
            "🤖 ML Insights"
        ])
        
        # ============================================================================
        # ВКЛАДКА 1: BASIC STATISTICS
        # ============================================================================
        with tab1:
            st.subheader("Basic Statistical Analysis")
            
            # График 10: Статистика по B-элементам
            if len(filtered_df) > 0:
                fig, stats_df = plot_b_site_statistics(filtered_df)
                st.pyplot(fig)
                plt.close(fig)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # График 12: Гистограмма распределения x_max относительно x_boundary
                if 'x_max' in filtered_df.columns and 'x_boundary' in filtered_df.columns:
                    fig = plot_xmax_vs_boundary_histogram(filtered_df)
                    st.pyplot(fig)
                    plt.close(fig)
            
            with col2:
                # График 15: Распределение x_boundary
                if 'x_boundary' in filtered_df.columns:
                    fig = plot_distribution_kde(filtered_df)
                    st.pyplot(fig)
                    plt.close(fig)
            
            # График 13: Корреляционная матрица
            if len(filtered_df) > 0:
                fig = plot_correlation_heatmap(filtered_df)
                st.pyplot(fig)
                plt.close(fig)
            
            # Детальная корреляция с p-value
            st.subheader("Detailed Correlations with p-values")
            features = ['dr', 'dr_rel', 'tolerance_factor', 'x_boundary', 'x_max']
            available_features = [f for f in features if f in filtered_df.columns]
            if len(available_features) >= 2:
                corr_df = calculate_correlations(filtered_df, available_features)
                if len(corr_df) > 0:
                    st.dataframe(corr_df, use_container_width=True)
        
        # ============================================================================
        # ВКЛАДКА 2: SOLUBILITY ANALYSIS
        # ============================================================================
        with tab2:
            st.subheader("Solubility Limit Analysis")
            
            # Выбор графиков для отображения
            solubility_plots = st.multiselect(
                "Select solubility plots to display",
                options=[
                    "Solubility vs Radius Difference",
                    "Solubility vs Tolerance Factor",
                    "Δr Heatmap with x(boundary)",
                    "Top Dopants Violin Plot",
                    "Critical Δr Threshold",
                    "Research Intensity Matrix"
                ],
                default=["Solubility vs Radius Difference", "Δr Heatmap with x(boundary)"]
            )
            
            n_plots = len(solubility_plots)
            if n_plots > 0:
                n_cols = min(2, n_plots)
                n_rows = (n_plots + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
                if n_plots == 1:
                    axes = np.array([axes])
                axes = axes.flatten()
                
                plot_idx = 0
                for plot_name in solubility_plots:
                    ax = axes[plot_idx]
                    
                    if plot_name == "Solubility vs Radius Difference":
                        if 'dr' in filtered_df.columns and 'x_boundary' in filtered_df.columns:
                            plot_solubility_vs_dr(filtered_df, ax)
                        else:
                            ax.text(0.5, 0.5, 'Required data missing', ha='center', va='center')
                    
                    elif plot_name == "Solubility vs Tolerance Factor":
                        if 'tolerance_factor' in filtered_df.columns and 'x_boundary' in filtered_df.columns:
                            plot_tolerance_factor(filtered_df, ax)
                        else:
                            ax.text(0.5, 0.5, 'Required data missing', ha='center', va='center')
                    
                    elif plot_name == "Top Dopants Violin Plot":
                        if 'x_boundary' in filtered_df.columns and 'D_element' in filtered_df.columns:
                            # Временно сохраняем и показываем отдельно
                            plt.close(fig)
                            violin_fig = plot_top_dopants_violin(filtered_df)
                            st.pyplot(violin_fig)
                            plt.close(violin_fig)
                            plot_idx -= 1  # не занимаем место в сетке
                    
                    elif plot_name == "Critical Δr Threshold":
                        if 'dr' in filtered_df.columns and 'has_impurity' in filtered_df.columns:
                            # Временно сохраняем и показываем отдельно
                            plt.close(fig)
                            threshold_fig = plot_critical_dr_threshold(filtered_df)
                            st.pyplot(threshold_fig)
                            plt.close(threshold_fig)
                            plot_idx -= 1
                    
                    elif plot_name == "Research Intensity Matrix":
                        if 'B_element' in filtered_df.columns and 'D_element' in filtered_df.columns:
                            # Временно сохраняем и показываем отдельно
                            plt.close(fig)
                            matrix_fig, pub_matrix = plot_publication_matrix(filtered_df)
                            st.pyplot(matrix_fig)
                            plt.close(matrix_fig)
                            st.dataframe(pub_matrix, use_container_width=True)
                            plot_idx -= 1
                    
                    # Применяем настройки
                    if not show_grid:
                        ax.grid(False)
                    if not show_legend:
                        ax.legend().remove()
                    
                    plot_idx += 1
                
                # Убираем пустые подграфики
                for j in range(plot_idx, len(axes)):
                    fig.delaxes(axes[j])
                
                if plot_idx > 0:
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            
            # График 3: Тепловая карта Δr (отдельно, если выбрана)
            if "Δr Heatmap with x(boundary)" in solubility_plots:
                if 'dr' in filtered_df.columns and 'B_element' in filtered_df.columns and 'D_element' in filtered_df.columns:
                    heatmap_fig = plot_heatmap_dr(filtered_df)
                    st.pyplot(heatmap_fig)
                    plt.close(heatmap_fig)
        
        # ============================================================================
        # ВКЛАДКА 3: CONDUCTIVITY ANALYSIS
        # ============================================================================
        with tab3:
            st.subheader("Conductivity Maximum Analysis")
            
            if 'x_max' in filtered_df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # График 4: x(max) vs x(boundary)
                    if 'x_boundary' in filtered_df.columns:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plot_xmax_vs_xboundary(filtered_df, ax)
                        if not show_grid:
                            ax.grid(False)
                        if not show_legend:
                            ax.legend().remove()
                        st.pyplot(fig)
                        plt.close(fig)
                
                with col2:
                    # График 5: x(max) vs tolerance factor
                    if 'tolerance_factor' in filtered_df.columns:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plot_xmax_vs_tolerance(filtered_df, ax)
                        if not show_grid:
                            ax.grid(False)
                        if not show_legend:
                            ax.legend().remove()
                        st.pyplot(fig)
                        plt.close(fig)
                
                # График 6: Относительное положение
                if 'x_rel_max' in filtered_df.columns and 'dr_rel' in filtered_df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_relative_position(filtered_df, ax)
                    if not show_grid:
                        ax.grid(False)
                    if not show_legend:
                        ax.legend().remove()
                    st.pyplot(fig)
                    plt.close(fig)
                
                # График 16: Пузырьковая диаграмма смещения
                st.subheader("Shift Analysis")
                fig = plot_shift_vs_dr_bubble(filtered_df)
                st.pyplot(fig)
                plt.close(fig)
            
            else:
                st.info("x_max data not available in the dataset")
        
        # ============================================================================
        # ВКЛАДКА 4: ADVANCED VISUALIZATION
        # ============================================================================
        with tab4:
            st.subheader("Advanced Visualization")
            
            adv_plots = st.multiselect(
                "Select advanced plots",
                options=[
                    "PCA Analysis",
                    "Impurity Phase Diagram (t-Δr)",
                    "Temporal Trends",
                    "Tolerance Factor Evolution",
                    "Goldschmidt Bubble Diagram",
                    "Dopant Comparison by B-site"
                ],
                default=["PCA Analysis", "Goldschmidt Bubble Diagram"]
            )
            
            for plot_name in adv_plots:
                if plot_name == "PCA Analysis":
                    fig = plot_pca(filtered_df)
                    st.pyplot(fig)
                    plt.close(fig)
                
                elif plot_name == "Impurity Phase Diagram (t-Δr)":
                    if 'tolerance_factor' in filtered_df.columns and 'dr' in filtered_df.columns:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        plot_impurity_phase_diagram(filtered_df, ax)
                        if not show_grid:
                            ax.grid(False)
                        if not show_legend:
                            ax.legend().remove()
                        st.pyplot(fig)
                        plt.close(fig)
                
                elif plot_name == "Temporal Trends":
                    if 'year' in filtered_df.columns and 'x_boundary' in filtered_df.columns:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plot_temporal_trend(filtered_df, ax)
                        if not show_grid:
                            ax.grid(False)
                        if not show_legend:
                            ax.legend().remove()
                        st.pyplot(fig)
                        plt.close(fig)
                
                elif plot_name == "Tolerance Factor Evolution":
                    fig = plot_tolerance_evolution(filtered_df)
                    st.pyplot(fig)
                    plt.close(fig)
                
                elif plot_name == "Goldschmidt Bubble Diagram":
                    fig = plot_goldschmidt_bubble(filtered_df)
                    st.pyplot(fig)
                    plt.close(fig)
                
                elif plot_name == "Dopant Comparison by B-site":
                    if 'D_element' in filtered_df.columns and 'B_element' in filtered_df.columns:
                        selected_d = st.selectbox(
                            "Select dopant for comparison",
                            options=sorted(filtered_df['D_element'].unique())
                        )
                        fig = plot_dopant_comparison_boxplot(filtered_df, selected_d)
                        st.pyplot(fig)
                        plt.close(fig)
        
        # ============================================================================
        # ВКЛАДКА 5: ML INSIGHTS
        # ============================================================================
        with tab5:
            st.subheader("Machine Learning Insights")
            
            # График 20: Важность признаков
            st.markdown("**Feature Importance Analysis (Random Forest)**")
            fig, importance_df = plot_feature_importance(filtered_df)
            st.pyplot(fig)
            plt.close(fig)
            
            if importance_df is not None:
                st.dataframe(importance_df, use_container_width=True)
            
            # Статистика по публикациям
            st.markdown("---")
            st.subheader("Publication Analysis")
            
            if 'doi' in filtered_df.columns:
                st.metric("Total publications", filtered_df['doi'].nunique())
            
            if 'year' in filtered_df.columns:
                year_stats = filtered_df['year'].dropna()
                if len(year_stats) > 0:
                    st.metric("Year range", f"{int(year_stats.min())} - {int(year_stats.max())}")
                    st.metric("Median year", int(year_stats.median()))
                    
                    # Гистограмма по годам
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(year_stats, bins=20, edgecolor='black', alpha=0.7)
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Number of publications')
                    ax.set_title('Publication Year Distribution')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
    
    else:
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

