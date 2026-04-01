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
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata
import re
from datetime import datetime
import openpyxl
import warnings
warnings.filterwarnings('ignore')

# Константы
AVOGADRO_NUMBER = 6.02214076e23  # моль⁻¹
OXYGEN_RADIUS = 1.4  # Å
PREFACTOR_VOLUME = 16 * np.pi / 3  # 16π/3 для расчета объема сфер

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
    # Для A-сайта используем КЧ=12, для B-сайта КЧ=6, для O - фиксированное значение
    ('Ba', 2, 12): 1.61,
    ('Sr', 2, 12): 1.44,
    ('O', -2, 6): 1.4,
    
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
    ('La', 3, 6): 1.032,
    ('Pr', 3, 6): 0.99,
    ('Tb', 3, 6): 0.923,
    ('Er', 3, 6): 0.89,
    ('Tm', 3, 6): 0.88,
    ('Lu', 3, 6): 0.861,
    ('Ca', 2, 6): 1.00,
}

# ============================================================================
# БАЗА ДАННЫХ ЭЛЕКТРООТРИЦАТЕЛЬНОСТИ (Полинг)
# ============================================================================
ELECTRONEGATIVITY = {
    'Ba': 0.89,
    'Sr': 0.95,
    'Ce': 1.12,
    'Zr': 1.33,
    'Sn': 1.96,
    'Ti': 1.54,
    'Hf': 1.3,
    'Gd': 1.20,
    'Sm': 1.17,
    'Y': 1.22,
    'In': 1.78,
    'Sc': 1.36,
    'Dy': 1.22,
    'Ho': 1.23,
    'Yb': 1.22,
    'Eu': 1.20,
    'Nd': 1.14,
    'La': 1.10,
    'Pr': 1.13,
    'Tb': 1.20,
    'Er': 1.24,
    'Tm': 1.25,
    'Lu': 1.27,
    'Ca': 1.00,
    'O': 3.44,
}

# ============================================================================
# БАЗА ДАННЫХ СВОЙСТВ БАЗОВЫХ СТРУКТУР
# ============================================================================
MATERIAL_PROPERTIES = {
    'BaCeO3': {
        'band_gap': 2.299,
        'E_form': -3.550,
        'density': 6.034,
        'M_molar': 341.36,
        'r_A': 1.61,
        'r_B': 0.87,
        'r_O': 1.4
    },
    'BaSnO3': {
        'band_gap': 0.372,
        'E_form': -2.587,
        'density': 7.097,
        'M_molar': 336.03,
        'r_A': 1.61,
        'r_B': 0.69,
        'r_O': 1.4
    },
    'BaHfO3': {
        'band_gap': 3.539,
        'E_form': -3.787,
        'density': 8.332,
        'M_molar': 428.72,
        'r_A': 1.61,
        'r_B': 0.71,
        'r_O': 1.4
    },
    'BaZrO3': {
        'band_gap': 3.116,
        'E_form': -3.639,
        'density': 6.148,
        'M_molar': 348.54,
        'r_A': 1.61,
        'r_B': 0.72,
        'r_O': 1.4
    },
    'BaTiO3': {
        'band_gap': None,
        'E_form': -1.685,
        'density': 4.547,
        'M_molar': 233.19,
        'r_A': 1.61,
        'r_B': 0.605,
        'r_O': 1.4
    },
    'SrSnO3': {
        'band_gap': 1.555,
        'E_form': -2.631,
        'density': 6.355,
        'M_molar': 302.34,
        'r_A': 1.44,
        'r_B': 0.69,
        'r_O': 1.4
    }
}

# ============================================================================
# АТОМНЫЕ МАССЫ (г/моль)
# ============================================================================
ATOMIC_MASSES = {
    'Ba': 137.33,
    'Sr': 87.62,
    'Ce': 140.12,
    'Zr': 91.22,
    'Sn': 118.71,
    'Ti': 47.87,
    'Hf': 178.49,
    'Gd': 157.25,
    'Sm': 150.36,
    'Y': 88.91,
    'In': 114.82,
    'Sc': 44.96,
    'Dy': 162.50,
    'Ho': 164.93,
    'Yb': 173.05,
    'Eu': 151.96,
    'Nd': 144.24,
    'La': 138.91,
    'Pr': 140.91,
    'Tb': 158.93,
    'Er': 167.26,
    'Tm': 168.93,
    'Lu': 174.97,
    'Ca': 40.08,
    'O': 16.00,
}

# Цветовая карта для B-катионов
B_COLORS = {
    'Ce': '#E41A1C',
    'Zr': '#377EB8',
    'Sn': '#4DAF4A',
    'Ti': '#984EA3',
    'Hf': '#FF7F00',
    'default': '#999999'
}

# Маркеры для D-допантов
D_MARKERS = {
    'Gd': 'o', 'Sm': 's', 'Y': '^', 'In': 'D', 'Sc': 'v',
    'Dy': 'P', 'Ho': '*', 'Yb': 'X', 'Eu': 'p', 'Nd': 'h',
    'Pr': 'H', 'Tb': 'd', 'Er': '8', 'Tm': 'p', 'Lu': 'P',
    'default': 'o'
}

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================
def extract_year_from_doi(doi):
    """Извлечение года из DOI (если есть в строке)"""
    if pd.isna(doi) or doi == '':
        return None
    
    match = re.search(r'[\./](19|20)\d{2}[\./]', str(doi))
    if match:
        return int(match.group(0).strip('./'))
    return None

def get_ionic_radius(element, charge, coordination):
    """Получение ионного радиуса из базы"""
    key = (element, charge, coordination)
    return IONIC_RADII.get(key, None)

def get_electronegativity(element):
    """Получение электроотрицательности по Полингу"""
    return ELECTRONEGATIVITY.get(element, None)

def get_atomic_mass(element):
    """Получение атомной массы элемента"""
    return ATOMIC_MASSES.get(element, None)

def get_base_properties(a_element, b_element):
    """Получение свойств базовой структуры ABO₃"""
    compound = f"{a_element}{b_element}O3"
    return MATERIAL_PROPERTIES.get(compound, None)

def calculate_molar_mass(a_element, b_element, d_element, x):
    """Расчет молярной массы для AB₁₋xDxO₃₋x/₂"""
    M_A = get_atomic_mass(a_element)
    M_B = get_atomic_mass(b_element)
    M_D = get_atomic_mass(d_element)
    M_O = ATOMIC_MASSES['O']
    
    if None in [M_A, M_B, M_D]:
        return None
    
    return M_A + (1 - x) * M_B + x * M_D + (3 - x/2) * M_O

def calculate_volume_cations(r_A, r_B, r_D, r_O, x):
    """
    Расчет объема катионов и анионов по сферической модели
    Vcations = 16π/3 × [r(A)³ + (1-x)·r(B)³ + x·r(D)³ + (3-x/2)·r(O)³]
    """
    if None in [r_A, r_B, r_D, r_O]:
        return None
    
    term_A = r_A ** 3
    term_B = (1 - x) * (r_B ** 3)
    term_D = x * (r_D ** 3)
    term_O = (3 - x/2) * (r_O ** 3)
    
    return PREFACTOR_VOLUME * (term_A + term_B + term_D + term_O)

def calculate_cell_volume(molar_mass, density):
    """
    Расчет объема элементарной ячейки
    Vcell = M / (ρ × N_A)
    """
    if None in [molar_mass, density] or density <= 0:
        return None
    return molar_mass / (density * AVOGADRO_NUMBER) * 1e24  # перевод в Å³

def calculate_free_volume(V_cell, V_cations):
    """Расчет свободного объема"""
    if None in [V_cell, V_cations]:
        return None
    return V_cell - V_cations

def calculate_packing_factor(V_cations, V_cell):
    """Расчет фактора упаковки"""
    if None in [V_cations, V_cell] or V_cell <= 0:
        return None
    return V_cations / V_cell

def calculate_free_volume_fraction(V_free, V_cell):
    """Расчет доли свободного объема"""
    if None in [V_free, V_cell] or V_cell <= 0:
        return None
    return V_free / V_cell

def calculate_formation_energy(base_properties, d_element, x):
    """
    Расчет энергии образования твердого раствора
    E_form = (1-x)·E_form(BO₃) + x·E_form(DO₁.₅) + ΔE_mixing
    """
    if base_properties is None:
        return None
    
    # Энергия образования DO₁.₅ (оценка по тренду)
    # В реальности нужно из базы, здесь используем приближение
    E_form_D = {
        'Gd': -2.8, 'Sm': -2.7, 'Y': -2.9, 'In': -2.5, 'Sc': -3.0,
        'Dy': -2.8, 'Ho': -2.8, 'Yb': -2.7, 'Eu': -2.6, 'Nd': -2.6,
        'La': -2.5, 'Pr': -2.6, 'Tb': -2.7, 'Er': -2.8, 'Tm': -2.8, 'Lu': -2.9
    }.get(d_element, -2.7)
    
    E_form_base = base_properties.get('E_form', 0)
    E_form_mixed = (1 - x) * E_form_base + x * E_form_D
    
    return E_form_mixed

def calculate_band_gap(base_properties, d_element, x):
    """
    Расчет ширины запрещенной зоны твердого раствора (линейная интерполяция)
    """
    if base_properties is None:
        return None
    
    band_gap_base = base_properties.get('band_gap', None)
    if band_gap_base is None:
        return None
    
    # Оценка band gap для допантов (аппроксимация)
    band_gap_D = {
        'Gd': 5.2, 'Sm': 4.8, 'Y': 5.5, 'In': 3.7, 'Sc': 6.0,
        'Dy': 5.0, 'Ho': 5.1, 'Yb': 4.5, 'Eu': 4.6, 'Nd': 4.7,
        'La': 5.6, 'Pr': 4.9, 'Tb': 5.0, 'Er': 5.2, 'Tm': 5.3, 'Lu': 5.8
    }.get(d_element, 5.0)
    
    return (1 - x) * band_gap_base + x * band_gap_D

def calculate_lattice_strain_energy(r_B, r_D, x):
    """
    Оценка энергии деформации решетки
    ε_strain ~ (Δr)² × (1 - x) × x
    """
    if None in [r_B, r_D]:
        return None
    
    dr = abs(r_D - r_B)
    return dr ** 2 * (1 - x) * x

def calculate_tolerance_factor(r_A, r_avg_B, r_O):
    """Расчет фактора толерантности Гольдшмидта"""
    if None in [r_A, r_avg_B, r_O]:
        return None
    return (r_A + r_O) / (np.sqrt(2) * (r_avg_B + r_O))

def calculate_t_gradient(r_B, r_D, x, r_A, r_O, x_points=100):
    """
    Расчет градиента изменения t с ростом x
    dt/dx ≈ (t(x+Δx) - t(x)) / Δx
    """
    if None in [r_B, r_D, r_A, r_O]:
        return None
    
    x_vals = np.linspace(0, x, x_points)
    t_vals = []
    
    for xi in x_vals:
        r_avg = (1 - xi) * r_B + xi * r_D
        t = (r_A + r_O) / (np.sqrt(2) * (r_avg + r_O))
        t_vals.append(t)
    
    if len(t_vals) > 1:
        dt = t_vals[-1] - t_vals[0]
        dx = x_vals[-1] - x_vals[0]
        return dt / dx if dx > 0 else None
    return None

def process_x_boundary(value, inv_end=None, treat_lower_as_exact=False):
    """
    Специальная обработка для x(boundary)
    Возвращает (числовое_значение, тип_значения, исходная_строка)
    тип_значения: 'exact' - точное значение, 'lower_bound' - нижняя оценка, 'none' - нет данных
    
    Если treat_lower_as_exact=True, то все '-' преобразуются в 'exact' с значением inv_end
    """
    if pd.isna(value) or value == '' or value == '-':
        if inv_end is not None and not pd.isna(inv_end) and inv_end != '':
            try:
                numeric_value = float(inv_end)
                if treat_lower_as_exact:
                    return numeric_value, 'exact', str(value)
                else:
                    return numeric_value, 'lower_bound', str(value)
            except (ValueError, TypeError):
                return None, 'none', str(value)
        return None, 'none', str(value)
    
    try:
        numeric_value = float(value)
        return numeric_value, 'exact', str(value)
    except (ValueError, TypeError):
        return None, 'none', str(value)

# ============================================================================
# ФУНКЦИЯ РАСЧЕТА ВСЕХ ДЕСКРИПТОРОВ (РАСШИРЕННАЯ)
# ============================================================================
def calculate_descriptors(row):
    """Расчет всех дескрипторов для одной строки (расширенная версия)"""
    # Проверка наличия необходимых ключей
    if 'B_element' not in row.index or 'D_element' not in row.index:
        return {
            'r_B': None, 'r_D': None, 'dr': None, 'dr_rel': None,
            'r_avg_B': None, 'tolerance_factor': None, 't_gradient': None,
            't_range': None, 'χ_B': None, 'χ_D': None, 'χ_A': None,
            'χ_avg_B': None, 'Δχ': None, 'Δχ_gradient': None,
            'V_cations': None, 'V_cell': None, 'V_free': None,
            'packing_factor': None, 'free_volume_fraction': None,
            'E_form': None, 'band_gap': None, 'lattice_strain_energy': None,
            'oxygen_vacancy_conc': None, 'molar_mass': None
        }
    
    A = row.get('A_element', 'Ba')
    B = row['B_element']
    D = row['D_element']
    
    # Получаем x_boundary, если есть
    x = row.get('x_boundary_value', 0)
    if pd.isna(x):
        x = 0
    
    # Получаем радиусы
    r_A = IONIC_RADII.get((A, 2, 12), None)
    r_O = IONIC_RADII.get(('O', -2, 6), None)
    
    # Для B и D определяем заряды по умолчанию
    r_B = IONIC_RADII.get((B, 4, 6), None)
    r_D = IONIC_RADII.get((D, 3, 6), None)
    
    # Если не нашли по стандартным зарядам, пробуем другие
    if r_B is None:
        possible_charges_B = [4, 3, 2]
        for charge in possible_charges_B:
            r_B = IONIC_RADII.get((B, charge, 6), None)
            if r_B:
                break
    
    if r_D is None:
        possible_charges_D = [3, 2, 4]
        for charge in possible_charges_D:
            r_D = IONIC_RADII.get((D, charge, 6), None)
            if r_D:
                break
    
    # Получаем электроотрицательности
    χ_A = get_electronegativity(A)
    χ_B = get_electronegativity(B)
    χ_D = get_electronegativity(D)
    
    # Получаем свойства базовой структуры
    base_props = get_base_properties(A, B)
    
    # Расчеты
    dr = abs(r_D - r_B) if None not in [r_B, r_D] else None
    dr_rel = dr / r_B if (dr is not None and r_B is not None and r_B != 0) else None
    
    r_avg_B = (1 - x) * r_B + x * r_D if None not in [r_B, r_D] else None
    
    tolerance_factor = calculate_tolerance_factor(r_A, r_avg_B, r_O) if None not in [r_A, r_avg_B, r_O] else None
    
    t_gradient = calculate_t_gradient(r_B, r_D, x, r_A, r_O) if None not in [r_B, r_D, r_A, r_O] and x > 0 else None
    
    t_range = (calculate_tolerance_factor(r_A, r_B, r_O) - tolerance_factor) if None not in [r_A, r_B, r_O, tolerance_factor] else None
    
    χ_avg_B = (1 - x) * χ_B + x * χ_D if None not in [χ_B, χ_D] else None
    
    Δχ = abs(χ_avg_B - χ_A) if None not in [χ_avg_B, χ_A] else None
    
    Δχ_gradient = (χ_avg_B - χ_B) / x if (χ_avg_B is not None and χ_B is not None and x > 0) else None
    
    # Объемные характеристики
    V_cations = calculate_volume_cations(r_A, r_B, r_D, r_O, x) if None not in [r_A, r_B, r_D, r_O] else None
    
    molar_mass = calculate_molar_mass(A, B, D, x)
    
    # Используем плотность из базовой структуры, если доступна
    density = base_props.get('density', None) if base_props is not None else None
    V_cell = calculate_cell_volume(molar_mass, density) if None not in [molar_mass, density] else None
    
    V_free = calculate_free_volume(V_cell, V_cations) if None not in [V_cell, V_cations] else None
    
    packing_factor = calculate_packing_factor(V_cations, V_cell) if None not in [V_cations, V_cell] else None
    
    free_volume_fraction = calculate_free_volume_fraction(V_free, V_cell) if None not in [V_free, V_cell] else None
    
    # Термодинамические характеристики
    E_form = calculate_formation_energy(base_props, D, x) if base_props is not None else None
    
    band_gap = calculate_band_gap(base_props, D, x) if base_props is not None else None
    
    lattice_strain_energy = calculate_lattice_strain_energy(r_B, r_D, x) if None not in [r_B, r_D] else None
    
    oxygen_vacancy_conc = x / 2 if x is not None else None
    
    return {
        'r_B': r_B,
        'r_D': r_D,
        'dr': dr,
        'dr_rel': dr_rel,
        'r_avg_B': r_avg_B,
        'tolerance_factor': tolerance_factor,
        't_gradient': t_gradient,
        't_range': t_range,
        'χ_B': χ_B,
        'χ_D': χ_D,
        'χ_A': χ_A,
        'χ_avg_B': χ_avg_B,
        'Δχ': Δχ,
        'Δχ_gradient': Δχ_gradient,
        'V_cations': V_cations,
        'V_cell': V_cell,
        'V_free': V_free,
        'packing_factor': packing_factor,
        'free_volume_fraction': free_volume_fraction,
        'E_form': E_form,
        'band_gap': band_gap,
        'lattice_strain_energy': lattice_strain_energy,
        'oxygen_vacancy_conc': oxygen_vacancy_conc,
        'molar_mass': molar_mass
    }

# ============================================================================
# ФУНКЦИЯ ПРОЦЕССИНГА ДАННЫХ (РАСШИРЕННАЯ)
# ============================================================================
def process_data(df, treat_lower_as_exact=False):
    """Основная функция обработки данных (расширенная версия)"""
    df_processed = df.copy()
    
    # Выводим названия колонок для отладки
    print("Original columns:", df_processed.columns.tolist())
    
    # Создаем маппинг на основе первых строк или стандартных названий
    column_mapping = {}
    
    # Обычно в таких таблицах порядок колонок фиксирован:
    # A, B, D, x(inv,in), x(inv,end), x(boundary), Impurity phase(s), x(max), doi,
    # далее могут быть колонки с электроотрицательностями
    expected_order = [
        'A_element', 'B_element', 'D_element',
        'x_inv_in', 'x_inv_end', 'x_boundary',
        'impurity', 'x_max', 'doi'
    ]
    
    # Если колонок ровно 9, используем позиционное соответствие
    if len(df_processed.columns) >= 9:
        for i, col_name in enumerate(expected_order[:len(df_processed.columns)]):
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
            elif 'χa' in col_lower or 'χ a' in col_lower:
                column_mapping[col] = 'χA_raw'
            elif 'χb' in col_lower or 'χ b' in col_lower:
                column_mapping[col] = 'χB_raw'
            elif 'χd' in col_lower or 'χ d' in col_lower:
                column_mapping[col] = 'χD_raw'
    
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
    
    # Заполняем пропуски и конвертируем числовые колонки
    for col in ['x_inv_in', 'x_inv_end', 'x_max']:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Обработка электроотрицательностей из файла
    for col in ['χA_raw', 'χB_raw', 'χD_raw']:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            df_processed[col.replace('_raw', '')] = df_processed[col]
    
    # СПЕЦИАЛЬНАЯ ОБРАБОТКА ДЛЯ x_boundary с учетом treat_lower_as_exact
    if 'x_boundary' in df_processed.columns:
        # Создаем колонки для хранения обработанных значений
        x_boundary_values = []
        x_boundary_types = []
        x_boundary_raw = []
        
        for idx, row in df_processed.iterrows():
            x_boundary_raw_val = row['x_boundary']
            x_inv_end_val = row.get('x_inv_end', None)
            
            numeric_val, val_type, raw_str = process_x_boundary(
                x_boundary_raw_val, x_inv_end_val, treat_lower_as_exact
            )
            
            x_boundary_values.append(numeric_val)
            x_boundary_types.append(val_type)
            x_boundary_raw.append(raw_str)
        
        df_processed['x_boundary_value'] = x_boundary_values
        df_processed['x_boundary_type'] = x_boundary_types
        df_processed['x_boundary_raw'] = x_boundary_raw
        
        # Для обратной совместимости оставляем исходную колонку, но создаем новую для числовых данных
        df_processed['x_boundary_original'] = df_processed['x_boundary']
        df_processed['x_boundary'] = df_processed['x_boundary_value']
    else:
        df_processed['x_boundary_value'] = None
        df_processed['x_boundary_type'] = 'none'
        df_processed['x_boundary_raw'] = ''
    
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
    
    # Рассчитываем дескрипторы для каждой строки (расширенная версия)
    descriptors_list = []
    for idx, row in df_processed.iterrows():
        desc = calculate_descriptors(row)
        descriptors_list.append(desc)
    
    descriptors_df = pd.DataFrame(descriptors_list)
    
    # Объединяем с исходными данными
    result = pd.concat([df_processed, descriptors_df], axis=1)
    
    # Дополнительные параметры
    if 'x_boundary_value' in result.columns and 'x_inv_end' in result.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            result['x_rel_boundary'] = result['x_boundary_value'] / result['x_inv_end']
    
    if 'x_max' in result.columns and 'x_boundary_value' in result.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            result['x_rel_max'] = result['x_max'] / result['x_boundary_value']
            result['x_diff_norm'] = (result['x_max'] - result['x_boundary_value']) / result['x_boundary_value']
    
    # Расчет ширины окна растворимости
    if 'x_boundary_value' in result.columns and 'x_inv_in' in result.columns:
        result['solubility_window'] = result['x_boundary_value'] - result['x_inv_in']
    
    # Расчет стабильности
    if 'x_boundary_value' in result.columns and 'x_max' in result.columns:
        result['stability_margin'] = (result['x_boundary_value'] - result['x_max']) / result['x_boundary_value']
    
    return result

# ============================================================================
# ФУНКЦИИ ДЛЯ РАСЧЕТА СТАТИСТИКИ (СУЩЕСТВУЮЩИЕ, НЕ ИЗМЕНЯЮТСЯ)
# ============================================================================
def calculate_correlations(df, features, include_lower_bounds=True):
    """Расчет корреляций Пирсона и Спирмена с p-value"""
    corr_data = []
    
    if include_lower_bounds:
        df_filtered = df.dropna(subset=features)
    else:
        df_filtered = df[df['x_boundary_type'] == 'exact'].dropna(subset=features)
    
    for i, f1 in enumerate(features):
        for f2 in features[i+1:]:
            valid = df_filtered[[f1, f2]].dropna()
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
                        'N': len(valid),
                        'Includes lower bounds': include_lower_bounds
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
    
    x_max_val = row.get('x_boundary_value', 0.5)
    if pd.isna(x_max_val) or x_max_val <= 0:
        x_max_val = 0.5
    
    x_vals = np.linspace(0, min(x_max_val, 0.8), x_points)
    r_avg = (1 - x_vals) * r_B + x_vals * r_D
    t_vals = (r_Ba + r_O) / (np.sqrt(2) * (r_avg + r_O))
    
    return x_vals, t_vals

def feature_importance_analysis(df):
    """Random Forest анализ важности признаков"""
    plot_df = df.dropna(subset=['x_boundary_value', 'dr', 'tolerance_factor', 'B_element'])
    
    if len(plot_df) < 10:
        return None, None
    
    X = pd.get_dummies(plot_df[['dr', 'tolerance_factor', 'B_element']],
                       columns=['B_element'])
    y = plot_df['x_boundary_value']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    r2 = rf.score(X, y)
    
    return importance_df, r2

def get_dopant_statistics(df, include_lower_bounds=True, treat_lower_as_exact=False):
    """Получение статистики по допантам с учетом типа значений
    
    Параметры:
    - include_lower_bounds: включать ли lower_bound значения
    - treat_lower_as_exact: если True, lower_bound значения учитываются как точные
    """
    if 'x_boundary_value' not in df.columns or 'D_element' not in df.columns:
        return pd.DataFrame()
    
    # Подготовка данных
    stats_data = []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('x_boundary_value')):
            continue
        
        dopant = row['D_element']
        x_val = row['x_boundary_value']
        x_type = row['x_boundary_type']
        
        if x_type == 'exact':
            stats_data.append({
                'Dopant': dopant,
                'x_boundary_value': x_val,
                'type': 'exact'
            })
        elif x_type == 'lower_bound' and include_lower_bounds:
            if treat_lower_as_exact:
                stats_data.append({
                    'Dopant': dopant,
                    'x_boundary_value': x_val,
                    'type': 'exact_estimated'
                })
            else:
                stats_data.append({
                    'Dopant': dopant,
                    'x_boundary_value': x_val,
                    'type': 'lower_bound'
                })
    
    if len(stats_data) == 0:
        return pd.DataFrame()
    
    stats_df = pd.DataFrame(stats_data)
    
    # Группировка по допантам
    stats_list = []
    for dopant in stats_df['Dopant'].unique():
        dopant_data = stats_df[stats_df['Dopant'] == dopant]
        
        exact_data = dopant_data[dopant_data['type'] == 'exact']['x_boundary_value']
        exact_est_data = dopant_data[dopant_data['type'] == 'exact_estimated']['x_boundary_value']
        lower_data = dopant_data[dopant_data['type'] == 'lower_bound']['x_boundary_value']
        
        all_values = pd.concat([exact_data, exact_est_data, lower_data])
        
        if len(all_values) > 0:
            stats_list.append({
                'Dopant': dopant,
                'Count': len(all_values),
                'Mean': all_values.mean(),
                'Median': all_values.median(),
                'Std': all_values.std(),
                'Min': all_values.min(),
                'Max': all_values.max(),
                'Exact values': len(exact_data) + len(exact_est_data),
                'Lower bounds': len(lower_data),
                'Includes lower bounds': include_lower_bounds,
                'Treat as exact': treat_lower_as_exact
            })
    
    if len(stats_list) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(stats_list).sort_values('Median', ascending=False)

# ============================================================================
# СУЩЕСТВУЮЩИЕ ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ (НЕ ИЗМЕНЯЮТСЯ)
# ============================================================================
def plot_solubility_vs_dr(df, ax):
    """График 1: x(boundary) vs Δr"""
    combinations = {}
    
    for idx, row in df.iterrows():
        if pd.isna(row.get('dr')) or pd.isna(row.get('x_boundary_value')):
            continue
            
        b_element = row['B_element']
        d_element = row['D_element']
        combo_key = f"{b_element}-{d_element}"
        
        if combo_key not in combinations:
            combinations[combo_key] = {
                'b_element': b_element,
                'd_element': d_element,
                'exact_values': [],
                'lower_bounds': []
            }
        
        if row['x_boundary_type'] == 'exact':
            combinations[combo_key]['exact_values'].append((row['dr'], row['x_boundary_value']))
        else:
            combinations[combo_key]['lower_bounds'].append((row['dr'], row['x_boundary_value']))
    
    for combo_key, data in combinations.items():
        b_element = data['b_element']
        color = B_COLORS.get(b_element, B_COLORS['default'])
        marker = D_MARKERS.get(data['d_element'], D_MARKERS['default'])
        
        if data['exact_values']:
            dr_exact, x_exact = zip(*data['exact_values'])
            ax.scatter(
                dr_exact, x_exact,
                color=color, marker=marker, s=80,
                alpha=1.0, edgecolors='black', linewidth=0.5,
                label=combo_key
            )
        
        if data['lower_bounds']:
            dr_lower, x_lower = zip(*data['lower_bounds'])
            ax.scatter(
                dr_lower, x_lower,
                facecolors='none',
                edgecolors=color,
                marker=marker, s=80,
                alpha=1.0, linewidth=1.5,
                label=f"{combo_key} (≥)" if not data['exact_values'] else ""
            )
    
    ax.set_xlabel('Δr = |r(D) - r(B)| (Å)')
    ax.set_ylabel('x(boundary)')
    ax.set_title('Solubility Limit vs Radius Difference\n(Hollow markers = lower bound estimates)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_tolerance_factor(df, ax):
    """График 2: x(boundary) vs tolerance factor"""
    for b_element in df['B_element'].unique():
        mask = df['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        exact_mask = mask & (df['x_boundary_type'] == 'exact')
        lower_mask = mask & (df['x_boundary_type'] == 'lower_bound')
        
        ax.scatter(
            df.loc[exact_mask, 'tolerance_factor'],
            df.loc[exact_mask, 'x_boundary_value'],
            color=color, s=100, alpha=0.9,
            edgecolors='black', linewidth=0.5,
            label=f"{b_element} (exact)"
        )
        
        ax.scatter(
            df.loc[lower_mask, 'tolerance_factor'],
            df.loc[lower_mask, 'x_boundary_value'],
            color=color, s=100, alpha=0.3,
            edgecolors='black', linewidth=0.5,
            label=f"{b_element} (≥)"
        )
    
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Ideal cubic (t=1)')
    ax.set_xlabel('Tolerance Factor (t) at x = x(boundary)')
    ax.set_ylabel('x(boundary)')
    ax.set_title('Solubility Limit vs Tolerance Factor\n(Transparent = lower bound estimates)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_heatmap_dr(df):
    """График 3: Тепловая карта Δr"""
    b_elements = sorted(df['B_element'].unique())
    d_elements = sorted(df['D_element'].unique())
    
    dr_matrix = pd.DataFrame(index=b_elements, columns=d_elements)
    x_boundary_matrix = pd.DataFrame(index=b_elements, columns=d_elements)
    x_boundary_type_matrix = pd.DataFrame(index=b_elements, columns=d_elements)
    
    for _, row in df.iterrows():
        dr_matrix.loc[row['B_element'], row['D_element']] = row['dr']
        x_boundary_matrix.loc[row['B_element'], row['D_element']] = row['x_boundary_value']
        x_boundary_type_matrix.loc[row['B_element'], row['D_element']] = row['x_boundary_type']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(dr_matrix.values.astype(float), cmap='viridis', aspect='auto')
    
    for i in range(len(b_elements)):
        for j in range(len(d_elements)):
            x_val = x_boundary_matrix.iloc[i, j]
            x_type = x_boundary_type_matrix.iloc[i, j]
            if not pd.isna(x_val):
                text_color = 'white'
                if x_type == 'lower_bound':
                    text = f'≥{x_val:.2f}'
                else:
                    text = f'{x_val:.2f}'
                ax.text(j, i, text, ha='center', va='center',
                       color=text_color, fontweight='bold', fontsize=8)
    
    ax.set_xticks(range(len(d_elements)))
    ax.set_yticks(range(len(b_elements)))
    ax.set_xticklabels(d_elements)
    ax.set_yticklabels(b_elements)
    ax.set_xlabel('Dopant (D)')
    ax.set_ylabel('B-site cation (B)')
    ax.set_title('Radius Difference Δr (color) with x(boundary) (text)\n(≥ indicates lower bound)')
    
    plt.colorbar(im, ax=ax, label='Δr (Å)')
    plt.tight_layout()
    return fig

def plot_xmax_vs_xboundary(df, ax):
    """График 4: x(max) vs x(boundary)"""
    valid = df.dropna(subset=['x_max', 'x_boundary_value'])
    
    exact_valid = valid[valid['x_boundary_type'] == 'exact']
    lower_valid = valid[valid['x_boundary_type'] == 'lower_bound']
    
    if len(exact_valid) > 0:
        scatter_exact = ax.scatter(
            exact_valid['x_boundary_value'], exact_valid['x_max'],
            c=exact_valid['dr'], cmap='coolwarm', s=100,
            alpha=0.9, edgecolors='black', linewidth=0.5,
            label='Exact values'
        )
    
    if len(lower_valid) > 0:
        scatter_lower = ax.scatter(
            lower_valid['x_boundary_value'], lower_valid['x_max'],
            c=lower_valid['dr'], cmap='coolwarm', s=100,
            alpha=0.3, edgecolors='black', linewidth=0.5,
            label='Lower bounds (≥)',
            marker='s'
        )
    
    min_val = min(valid['x_boundary_value'].min(), valid['x_max'].min())
    max_val = max(valid['x_boundary_value'].max(), valid['x_max'].max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'k--', alpha=0.5, label='y = x')
    
    ax.set_xlabel('x(boundary)')
    ax.set_ylabel('x(max conductivity)')
    ax.set_title('Conductivity Maximum vs Solubility Limit')
    
    if len(exact_valid) > 0:
        plt.colorbar(scatter_exact, ax=ax, label='Δr (Å)')
    elif len(lower_valid) > 0:
        plt.colorbar(scatter_lower, ax=ax, label='Δr (Å)')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_xmax_vs_tolerance(df, ax):
    """График 5: x(max) vs tolerance factor"""
    valid = df.dropna(subset=['x_max', 'tolerance_factor'])
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        exact_mask = mask & (valid['x_boundary_type'] == 'exact')
        lower_mask = mask & (valid['x_boundary_type'] == 'lower_bound')
        
        if exact_mask.any():
            ax.scatter(
                valid.loc[exact_mask, 'tolerance_factor'],
                valid.loc[exact_mask, 'x_max'],
                color=color, s=100, alpha=0.9,
                edgecolors='black', linewidth=0.5,
                label=f"{b_element} (exact)"
            )
        
        if lower_mask.any():
            ax.scatter(
                valid.loc[lower_mask, 'tolerance_factor'],
                valid.loc[lower_mask, 'x_max'],
                color=color, s=100, alpha=0.3,
                edgecolors='black', linewidth=0.5,
                label=f"{b_element} (≥)"
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
        
        exact_mask = mask & (valid['x_boundary_type'] == 'exact')
        lower_mask = mask & (valid['x_boundary_type'] == 'lower_bound')
        
        if exact_mask.any():
            ax.scatter(
                valid.loc[exact_mask, 'dr_rel'],
                valid.loc[exact_mask, 'x_rel_max'],
                color=color, s=100, alpha=0.9,
                edgecolors='black', linewidth=0.5,
                label=f"{b_element} (exact)"
            )
        
        if lower_mask.any():
            ax.scatter(
                valid.loc[lower_mask, 'dr_rel'],
                valid.loc[lower_mask, 'x_rel_max'],
                color=color, s=100, alpha=0.3,
                edgecolors='black', linewidth=0.5,
                label=f"{b_element} (≥)"
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
    features = ['dr', 'dr_rel', 'tolerance_factor', 'x_boundary_value']
    
    plot_df = df.dropna(subset=features + ['B_element', 'has_impurity'])
    
    if len(plot_df) < 3:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data for PCA',
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    X = plot_df[features].values
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for b_element in plot_df['B_element'].unique():
        mask = plot_df['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        exact_mask = mask & (plot_df['x_boundary_type'] == 'exact')
        if exact_mask.any():
            ax1.scatter(
                X_pca[exact_mask, 0], X_pca[exact_mask, 1],
                color=color, s=100, alpha=0.9,
                edgecolors='black', linewidth=0.5,
                label=f"{b_element} (exact)"
            )
        
        lower_mask = mask & (plot_df['x_boundary_type'] == 'lower_bound')
        if lower_mask.any():
            ax1.scatter(
                X_pca[lower_mask, 0], X_pca[lower_mask, 1],
                color=color, s=100, alpha=0.3,
                edgecolors='black', linewidth=0.5,
                label=f"{b_element} (≥)"
            )
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_title('PCA: Colored by B-site')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    colors = {True: '#E41A1C', False: '#377EB8'}
    for has_impurity in [True, False]:
        mask = plot_df['has_impurity'] == has_impurity
        if mask.any():
            exact_mask = mask & (plot_df['x_boundary_type'] == 'exact')
            if exact_mask.any():
                ax2.scatter(
                    X_pca[exact_mask, 0], X_pca[exact_mask, 1],
                    color=colors[has_impurity], s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f'Impurity: {has_impurity} (exact)'
                )
            
            lower_mask = mask & (plot_df['x_boundary_type'] == 'lower_bound')
            if lower_mask.any():
                ax2.scatter(
                    X_pca[lower_mask, 0], X_pca[lower_mask, 1],
                    color=colors[has_impurity], s=100, alpha=0.3,
                    edgecolors='black', linewidth=0.5,
                    label=f'Impurity: {has_impurity} (≥)'
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
            exact_mask = mask & (valid['x_boundary_type'] == 'exact')
            lower_mask = mask & (valid['x_boundary_type'] == 'lower_bound')
            
            if exact_mask.any():
                ax.scatter(
                    valid.loc[exact_mask, 'tolerance_factor'],
                    valid.loc[exact_mask, 'dr'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{imp_type} (exact)"
                )
            
            if lower_mask.any():
                ax.scatter(
                    valid.loc[lower_mask, 'tolerance_factor'],
                    valid.loc[lower_mask, 'dr'],
                    color=color, s=100, alpha=0.3,
                    edgecolors='black', linewidth=0.5,
                    label=f"{imp_type} (≥)"
                )
    
    ax.set_xlabel('Tolerance Factor (t)')
    ax.set_ylabel('Δr (Å)')
    ax.set_title('Impurity Phase Formation: t-Δr Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_temporal_trend(df, ax):
    """График 9: x(boundary) по годам"""
    valid = df.dropna(subset=['year', 'x_boundary_value'])
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        exact_mask = mask & (valid['x_boundary_type'] == 'exact')
        lower_mask = mask & (valid['x_boundary_type'] == 'lower_bound')
        
        if exact_mask.any():
            ax.scatter(
                valid.loc[exact_mask, 'year'],
                valid.loc[exact_mask, 'x_boundary_value'],
                color=color, s=100, alpha=0.9,
                edgecolors='black', linewidth=0.5,
                label=f"{b_element} (exact)"
            )
        
        if lower_mask.any():
            ax.scatter(
                valid.loc[lower_mask, 'year'],
                valid.loc[lower_mask, 'x_boundary_value'],
                color=color, s=100, alpha=0.3,
                edgecolors='black', linewidth=0.5,
                label=f"{b_element} (≥)"
            )
    
    ax.set_xlabel('Year')
    ax.set_ylabel('x(boundary)')
    ax.set_title('Solubility Limit Evolution Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_b_site_statistics(df, include_lower_bounds=True, treat_lower_as_exact=False):
    """График 10: Статистика по B-элементам (столбчатая диаграмма с ошибками)
    
    Параметры:
    - include_lower_bounds: включать ли lower_bound значения в анализ
    - treat_lower_as_exact: если True, lower_bound значения учитываются как точные
    """
    # Подготовка данных для статистики
    stats_data = []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('x_boundary_value')):
            continue
        
        b_element = row['B_element']
        x_val = row['x_boundary_value']
        x_type = row['x_boundary_type']
        
        if x_type == 'exact':
            stats_data.append({
                'B_element': b_element,
                'x_boundary_value': x_val,
                'type': 'exact'
            })
        elif x_type == 'lower_bound' and include_lower_bounds:
            if treat_lower_as_exact:
                # Обрабатываем как точное значение
                stats_data.append({
                    'B_element': b_element,
                    'x_boundary_value': x_val,
                    'type': 'exact_estimated'
                })
            else:
                # Оставляем как lower_bound
                stats_data.append({
                    'B_element': b_element,
                    'x_boundary_value': x_val,
                    'type': 'lower_bound'
                })
    
    if len(stats_data) == 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
        return fig, pd.DataFrame()
    
    stats_df = pd.DataFrame(stats_data)
    
    # Расчет статистики по B-элементам
    b_stats_list = []
    for b_element in stats_df['B_element'].unique():
        b_data = stats_df[stats_df['B_element'] == b_element]
        
        # Разделяем по типам для отображения
        exact_data = b_data[b_data['type'] == 'exact']['x_boundary_value']
        lower_data = b_data[b_data['type'] == 'lower_bound']['x_boundary_value']
        exact_est_data = b_data[b_data['type'] == 'exact_estimated']['x_boundary_value']
        
        # Объединяем все значения для статистики
        all_values = pd.concat([exact_data, exact_est_data, lower_data])
        
        if len(all_values) > 0:
            b_stats_list.append({
                'B_element': b_element,
                'mean': all_values.mean(),
                'median': all_values.median(),
                'std': all_values.std(),
                'count': len(all_values),
                'count_exact': len(exact_data) + len(exact_est_data),
                'count_lower': len(lower_data),
                'min': all_values.min(),
                'max': all_values.max()
            })
    
    if len(b_stats_list) == 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
        return fig, pd.DataFrame()
    
    b_stats = pd.DataFrame(b_stats_list).sort_values('mean', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    b_sites = b_stats['B_element'].tolist()
    x_pos = np.arange(len(b_sites))
    
    # Столбчатая диаграмма с ошибками (используем min и max для lower_bound)
    means = b_stats['mean'].values
    stds = b_stats['std'].values
    
    # Для B-элементов с lower_bound значениями показываем стрелку вверх
    bars = ax1.bar(x_pos, means, yerr=stds,
                   capsize=5, 
                   color=[B_COLORS.get(b, B_COLORS['default']) for b in b_sites],
                   edgecolor='black', linewidth=0.5, alpha=0.8,
                   error_kw={'linewidth': 1.5, 'ecolor': 'black'})
    
    # Добавляем маркеры для lower_bound значений
    for i, b in enumerate(b_sites):
        b_info = b_stats[b_stats['B_element'] == b].iloc[0]
        if b_info['count_lower'] > 0 and not treat_lower_as_exact:
            # Добавляем стрелку вверх, показывающую что истинное значение может быть выше
            max_val = b_info['max']
            if max_val > means[i] + stds[i]:
                ax1.annotate('↑', xy=(x_pos[i], max_val), 
                            xytext=(x_pos[i], max_val + 0.02),
                            ha='center', fontsize=14, color='red',
                            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(b_sites)
    ax1.set_ylabel('x(boundary)')
    title = f'Average Solubility by B-site\n(n={len(stats_df)} samples'
    if treat_lower_as_exact:
        title += ', lower bounds treated as exact)'
    elif include_lower_bounds:
        title += ', including lower bounds)'
    else:
        title += ', exact only)'
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(bottom=0)
    
    # Вторая диаграмма - количество образцов
    bottom = np.zeros(len(b_sites))
    
    count_exact = b_stats['count_exact'].values
    count_lower = b_stats['count_lower'].values
    
    ax2.bar(x_pos, count_exact, bottom=bottom,
            label='Exact', color='darkblue', edgecolor='black', linewidth=0.5)
    bottom += count_exact
    
    if include_lower_bounds and not treat_lower_as_exact:
        ax2.bar(x_pos, count_lower, bottom=bottom,
                label='Lower bound (≥)', color='lightblue', edgecolor='black', linewidth=0.5, alpha=0.7)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(b_sites)
    ax2.set_ylabel('Number of samples')
    ax2.set_title('Sample Count by B-site')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, b_stats

def plot_top_dopants_violin(df, include_lower_bounds=True, treat_lower_as_exact=False):
    """График 11: Violin plot для топ-10 допантов по растворимости, отсортированных по ионному радиусу
    
    Параметры:
    - include_lower_bounds: включать ли lower_bound значения
    - treat_lower_as_exact: если True, lower_bound значения учитываются как точные
    """
    dopant_stats = get_dopant_statistics(df, include_lower_bounds, treat_lower_as_exact)
    
    if len(dopant_stats) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    top_dopants = dopant_stats.head(10)['Dopant'].tolist()
    
    # Получаем ионные радиусы для допантов
    dopant_radii = {}
    for dopant in top_dopants:
        radius = IONIC_RADII.get((dopant, 3, 6), None)
        if radius is None:
            for charge in [2, 4]:
                radius = IONIC_RADII.get((dopant, charge, 6), None)
                if radius:
                    break
        dopant_radii[dopant] = radius if radius else 0
    
    # Сортируем по ионному радиусу
    sorted_dopants = sorted(top_dopants, key=lambda d: dopant_radii.get(d, 0))
    
    # Собираем данные для violin plot
    plot_data = []
    exact_points = []
    lower_points = []
    exact_est_points = []
    
    for d in sorted_dopants:
        d_data = []
        
        # Точные значения
        exact_vals = df[(df['D_element'] == d) & 
                        (df['x_boundary_type'] == 'exact')]['x_boundary_value'].dropna()
        for val in exact_vals:
            plot_data.append({'Dopant': d, 'x_value': val, 'type': 'exact'})
            exact_points.append({'Dopant': d, 'x_value': val})
        
        # Lower bounds
        if include_lower_bounds:
            lower_vals = df[(df['D_element'] == d) & 
                            (df['x_boundary_type'] == 'lower_bound')]['x_boundary_value'].dropna()
            if treat_lower_as_exact:
                for val in lower_vals:
                    plot_data.append({'Dopant': d, 'x_value': val, 'type': 'exact_estimated'})
                    exact_est_points.append({'Dopant': d, 'x_value': val})
            else:
                for val in lower_vals:
                    plot_data.append({'Dopant': d, 'x_value': val, 'type': 'lower_bound'})
                    lower_points.append({'Dopant': d, 'x_value': val})
    
    if len(plot_data) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data after processing', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    plot_df = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Подготовка данных для violin plot
    violin_data = []
    positions = []
    violin_labels = []
    radius_values = []
    
    for i, d in enumerate(sorted_dopants):
        d_data = plot_df[plot_df['Dopant'] == d]['x_value'].dropna()
        if len(d_data) > 0:
            violin_data.append(d_data)
            positions.append(i + 1)
            violin_labels.append(f"{d}")
            radius_values.append(dopant_radii.get(d, 0))
    
    if violin_data:
        parts = ax.violinplot(violin_data, positions=positions, showmeans=False, 
                              showmedians=True, widths=0.7)
        
        cmap = plt.cm.viridis
        norm = plt.Normalize(min(radius_values), max(radius_values))
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(cmap(norm(radius_values[i])))
            pc.set_alpha(0.6)
            pc.set_edgecolor('navy')
            pc.set_linewidth(1)
        
        if 'cmedians' in parts:
            parts['cmedians'].set_color('red')
            parts['cmedians'].set_linewidth(2)
        
        # Добавляем точки
        for i, d in enumerate(sorted_dopants):
            # Точные значения
            exact_vals = [p['x_value'] for p in exact_points if p['Dopant'] == d]
            if len(exact_vals) > 0:
                x_pos = np.random.normal(positions[i], 0.05, len(exact_vals))
                ax.scatter(x_pos, exact_vals, color='darkred', s=80,
                          alpha=0.9, zorder=3, edgecolors='black', linewidth=1,
                          label='Exact values' if i == 0 else "")
            
            # Точные из lower bounds (если treat_as_exact)
            exact_est_vals = [p['x_value'] for p in exact_est_points if p['Dopant'] == d]
            if len(exact_est_vals) > 0 and treat_lower_as_exact:
                x_pos = np.random.normal(positions[i], 0.05, len(exact_est_vals))
                ax.scatter(x_pos, exact_est_vals, color='orange', s=80,
                          alpha=0.8, zorder=3, edgecolors='black', linewidth=1,
                          label='Lower bounds (treated as exact)' if i == 0 else "")
            
            # Нижние оценки
            lower_vals = [p['x_value'] for p in lower_points if p['Dopant'] == d]
            if len(lower_vals) > 0 and not treat_lower_as_exact:
                x_pos = np.random.normal(positions[i], 0.08, len(lower_vals))
                ax.scatter(x_pos, lower_vals, color='darkorange', s=80,
                          alpha=0.7, zorder=3, marker='D', edgecolors='black', linewidth=1,
                          label='Lower bounds (≥)' if i == 0 else "")
        
        y_top = ax.get_ylim()[1]
        
        # Добавляем информацию о количестве образцов
        for i, d in enumerate(sorted_dopants):
            stats_row = dopant_stats[dopant_stats['Dopant'] == d].iloc[0]
            exact_count = int(stats_row['Exact values'])
            lower_count = int(stats_row['Lower bounds'])
            
            text = f'n={exact_count}'
            if lower_count > 0:
                if treat_lower_as_exact:
                    text += f' (inc. {lower_count}≥)'
                else:
                    text += f' (+{lower_count}≥)'
            
            ax.text(positions[i], y_top * 0.98, text,
                    ha='center', fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))
        
        ax.set_xlabel('Dopant Element', fontsize=14, fontweight='bold')
        ax.set_ylabel('Solid solution range', fontsize=14, fontweight='bold')
        
        title = f'Top 10 Dopants by Solubility - Sorted by Ionic Radius\n'
        if treat_lower_as_exact:
            title += '(Lower bounds treated as exact values)'
        elif include_lower_bounds:
            title += '(Including lower bounds as estimates)'
        else:
            title += '(Exact values only)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(violin_labels, fontsize=10)
        ax.set_ylim(bottom=0, top=1.0)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Ionic Radius (Å)', fontsize=10)
        
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), loc='upper right', fontsize=9,
                     framealpha=0.9, edgecolor='black')
        
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        for y in [0.2, 0.4, 0.6, 0.8]:
            ax.axhline(y=y, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig

def plot_xmax_vs_boundary_histogram(df):
    """График 12: Гистограмма распределения (x_max - x_boundary)/x_boundary"""
    valid = df.dropna(subset=['x_max', 'x_boundary_value'])
    
    exact_valid = valid[valid['x_boundary_type'] == 'exact']
    lower_valid = valid[valid['x_boundary_type'] == 'lower_bound']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if len(exact_valid) > 0:
        diff_exact = (exact_valid['x_max'] - exact_valid['x_boundary_value']) / exact_valid['x_boundary_value']
        
        n, bins, patches = ax1.hist(diff_exact, bins=20, edgecolor='black',
                                    color='skyblue', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2,
                   label='x(max) = x(boundary)')
        ax1.axvline(x=diff_exact.median(), color='blue', linestyle=':', linewidth=2,
                   label=f'Median = {diff_exact.median():.3f}')
        
        within_10pct = (abs(diff_exact) < 0.1).mean()
        ax1.text(0.05, 0.95, f'Within ±10%: {within_10pct:.1%}',
                transform=ax1.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax1.set_xlabel('(x_max - x_boundary) / x_boundary')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Exact values (n={len(exact_valid)})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    if len(lower_valid) > 0:
        diff_lower = (lower_valid['x_max'] - lower_valid['x_boundary_value']) / lower_valid['x_boundary_value']
        
        n, bins, patches = ax2.hist(diff_lower, bins=20, edgecolor='black',
                                    color='lightgreen', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2,
                   label='x(max) = x(boundary)')
        ax2.axvline(x=diff_lower.median(), color='blue', linestyle=':', linewidth=2,
                   label=f'Median = {diff_lower.median():.3f}')
        
        within_10pct = (abs(diff_lower) < 0.1).mean()
        ax2.text(0.05, 0.95, f'Within ±10%: {within_10pct:.1%}',
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.set_xlabel('(x_max - x_boundary) / x_boundary')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Lower bounds (n={len(lower_valid)})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Distribution of Conductivity Maximum Position')
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df, include_lower_bounds=True):
    """График 13: Тепловая карта корреляций"""
    features = ['dr', 'dr_rel', 'tolerance_factor', 'x_boundary_value', 'x_max']
    
    if include_lower_bounds:
        corr_df = df[features].dropna()
    else:
        corr_df = df[df['x_boundary_type'] == 'exact'][features].dropna()
    
    if len(corr_df) < 5:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data for correlation',
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    corr_df_renamed = corr_df.rename(columns={
        'x_boundary_value': 'x_boundary'
    })
    
    pearson_corr = corr_df_renamed.corr(method='pearson')
    spearman_corr = corr_df_renamed.corr(method='spearman')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, ax=ax1,
                cbar_kws={'label': 'Pearson r'})
    ax1.set_title(f'Pearson Correlations\n(n={len(corr_df)})')
    
    sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, ax=ax2,
                cbar_kws={'label': 'Spearman ρ'})
    ax2.set_title(f'Spearman Correlations\n(n={len(corr_df)})')
    
    plt.suptitle(f'Correlation Matrix ({"including" if include_lower_bounds else "excluding"} lower bounds)')
    plt.tight_layout()
    return fig

def plot_publication_matrix(df):
    """График 14: Тепловая карта количества публикаций по B-D парам"""
    pub_matrix = df.groupby(['B_element', 'D_element']).size().unstack(fill_value=0)
    exact_matrix = df[df['x_boundary_type'] == 'exact'].groupby(['B_element', 'D_element']).size().unstack(fill_value=0)
    lower_matrix = df[df['x_boundary_type'] == 'lower_bound'].groupby(['B_element', 'D_element']).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(pub_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Number of publications'},
                linewidths=0.5, linecolor='gray')
    
    for i in range(len(pub_matrix.index)):
        for j in range(len(pub_matrix.columns)):
            b_elem = pub_matrix.index[i]
            d_elem = pub_matrix.columns[j]
            
            exact_count = exact_matrix.loc[b_elem, d_elem] if (b_elem in exact_matrix.index and d_elem in exact_matrix.columns) else 0
            lower_count = lower_matrix.loc[b_elem, d_elem] if (b_elem in lower_matrix.index and d_elem in lower_matrix.columns) else 0
            
            if exact_count > 0 or lower_count > 0:
                text = f'{int(exact_count)}'
                if lower_count > 0:
                    text += f' (+{int(lower_count)}≥)'
                ax.text(j + 0.5, i + 0.7, text,
                       ha='center', va='center', fontsize=8,
                       color='black' if pub_matrix.iloc[i, j] < pub_matrix.max().max()/2 else 'white')
    
    ax.set_xlabel('Dopant (D)')
    ax.set_ylabel('B-site (B)')
    ax.set_title('Research Intensity: B-D Combinations\n(numbers show exact (+lower bound) counts)')
    
    plt.tight_layout()
    return fig, pub_matrix

def plot_distribution_kde(df):
    """График 15: Распределение x_boundary (гистограмма + KDE по B)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    exact_data = df[df['x_boundary_type'] == 'exact']['x_boundary_value'].dropna()
    lower_data = df[df['x_boundary_type'] == 'lower_bound']['x_boundary_value'].dropna()
    
    if len(exact_data) > 0:
        ax1.hist(exact_data, bins=20, edgecolor='black',
                 alpha=0.7, color='blue', density=True, label='Exact')
    
    if len(lower_data) > 0:
        ax1.hist(lower_data, bins=20, edgecolor='black',
                 alpha=0.3, color='green', density=True, label='Lower bound')
    
    sns.kdeplot(data=exact_data, ax=ax1, color='darkblue',
                linewidth=2, label='KDE (exact)')
    if len(lower_data) > 0:
        sns.kdeplot(data=lower_data, ax=ax1, color='darkgreen',
                    linewidth=2, linestyle='--', label='KDE (lower)')
    
    ax1.set_xlabel('x(boundary)')
    ax1.set_ylabel('Density')
    ax1.set_title('Overall Distribution of Solubility Limits')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for b_element in df['B_element'].unique():
        color = B_COLORS.get(b_element, 'gray')
        
        exact_data_b = df[(df['B_element'] == b_element) &
                          (df['x_boundary_type'] == 'exact')]['x_boundary_value'].dropna()
        if len(exact_data_b) > 1:
            sns.kdeplot(data=exact_data_b, label=f"{b_element} (exact)", ax=ax2,
                        linewidth=2, color=color, linestyle='-')
        
        lower_data_b = df[(df['B_element'] == b_element) &
                          (df['x_boundary_type'] == 'lower_bound')]['x_boundary_value'].dropna()
        if len(lower_data_b) > 1:
            sns.kdeplot(data=lower_data_b, label=f"{b_element} (≥)", ax=ax2,
                        linewidth=2, color=color, linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('x(boundary)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution by B-site (dashed = lower bounds)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_shift_vs_dr_bubble(df):
    """График 16: Пузырьковая диаграмма зависимости смещения от Δr"""
    valid = df.dropna(subset=['x_max', 'x_boundary_value', 'dr', 'tolerance_factor'])
    
    if len(valid) < 3:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    exact_valid = valid[valid['x_boundary_type'] == 'exact']
    lower_valid = valid[valid['x_boundary_type'] == 'lower_bound']
    
    if len(exact_valid) > 0:
        scatter_exact = ax.scatter(
            exact_valid['dr'],
            exact_valid['x_max'] - exact_valid['x_boundary_value'],
            s=exact_valid['x_boundary_value'] * 1000,
            c=exact_valid['tolerance_factor'],
            alpha=0.7,
            cmap='coolwarm',
            edgecolors='black',
            linewidth=0.5,
            label='Exact values'
        )
    
    if len(lower_valid) > 0:
        scatter_lower = ax.scatter(
            lower_valid['dr'],
            lower_valid['x_max'] - lower_valid['x_boundary_value'],
            s=lower_valid['x_boundary_value'] * 1000,
            c=lower_valid['tolerance_factor'],
            alpha=0.3,
            cmap='coolwarm',
            edgecolors='black',
            linewidth=0.5,
            marker='s',
            label='Lower bounds'
        )
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    valid_sorted = valid.nlargest(5, 'x_diff_norm')
    for idx, row in valid_sorted.iterrows():
        marker_style = ' (≥)' if row['x_boundary_type'] == 'lower_bound' else ''
        ax.annotate(f"{row['B_element']}-{row['D_element']}{marker_style}",
                    (row['dr'], row['x_max'] - row['x_boundary_value']),
                    fontsize=8, ha='center')
    
    ax.set_xlabel('Δr (Å)')
    ax.set_ylabel('x_max - x_boundary')
    ax.set_title('Shift of Conductivity Maximum from Solubility Limit\n(Bubble size = x_boundary)')
    ax.legend()
    
    if len(exact_valid) > 0:
        plt.colorbar(scatter_exact, ax=ax, label='Tolerance Factor')
    elif len(lower_valid) > 0:
        plt.colorbar(scatter_lower, ax=ax, label='Tolerance Factor')
    
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_tolerance_evolution(df):
    """График 17: Эволюция tolerance factor с допированием"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sample_size = min(10, len(df))
    sampled_df = df.sample(sample_size, random_state=42) if sample_size > 0 else df
    
    for idx, row in sampled_df.iterrows():
        result = calculate_t_series(row)
        if result:
            x_vals, t_vals = result
            label = f"{row['B_element']}-{row['D_element']}"
            
            linestyle = '--' if row['x_boundary_type'] == 'lower_bound' else '-'
            alpha = 0.5 if row['x_boundary_type'] == 'lower_bound' else 0.9
            
            if not pd.isna(row.get('x_boundary_value')):
                boundary_idx = np.argmin(np.abs(x_vals - row['x_boundary_value']))
                ax.plot(x_vals, t_vals, linewidth=2, alpha=alpha,
                       linestyle=linestyle, label=label)
                marker_color = 'red' if row['x_boundary_type'] == 'exact' else 'orange'
                ax.plot(x_vals[boundary_idx], t_vals[boundary_idx], 'o',
                       color=marker_color, markersize=8, alpha=alpha,
                       markeredgecolor='black')
            else:
                ax.plot(x_vals, t_vals, linewidth=2, alpha=alpha,
                       linestyle=linestyle, label=label)
    
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5,
               label='Ideal cubic (t=1)', linewidth=2)
    ax.set_xlabel('Dopant concentration x')
    ax.set_ylabel('Tolerance Factor')
    ax.set_title('Evolution of Tolerance Factor with Doping\n(Red dots = exact, orange = lower bound)')
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
    
    exact_valid = valid[valid['x_boundary_type'] == 'exact']
    lower_valid = valid[valid['x_boundary_type'] == 'lower_bound']
    
    if len(exact_valid) > 0:
        x_exact = exact_valid['dr'].values
        y_exact = exact_valid['has_impurity'].astype(int).cumsum() / np.arange(1, len(exact_valid)+1)
        ax.plot(x_exact, y_exact, 'b-', linewidth=2, label='Exact values')
        ax.fill_between(x_exact, 0, y_exact, alpha=0.2, color='blue')
    
    if len(lower_valid) > 0:
        x_lower = lower_valid['dr'].values
        y_lower = lower_valid['has_impurity'].astype(int).cumsum() / np.arange(1, len(lower_valid)+1)
        ax.plot(x_lower, y_lower, 'g--', linewidth=2, label='Lower bounds', alpha=0.7)
        ax.fill_between(x_lower, 0, y_lower, alpha=0.1, color='green')
    
    if len(exact_valid) > 5:
        y_diff = np.diff(y_exact)
        y_diff_smooth = gaussian_filter1d(y_diff, sigma=2)
        threshold_idx = np.argmax(y_diff_smooth[:len(y_diff_smooth)//2])
        if threshold_idx < len(x_exact)-1:
            ax.axvline(x=x_exact[threshold_idx], color='red', linestyle='--',
                       linewidth=2, label=f'Threshold Δr ≈ {x_exact[threshold_idx]:.3f} Å')
    
    ax.set_xlabel('Δr (Å)')
    ax.set_ylabel('Cumulative fraction with impurities')
    ax.set_title('Critical Radius Difference for Impurity Formation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_dopant_comparison_boxplot(df, selected_dopant, include_lower_bounds=True):
    """График 19: Сравнение одного допанта на разных B-сайтах"""
    if include_lower_bounds:
        plot_df = df[df['D_element'] == selected_dopant].dropna(subset=['x_boundary_value'])
    else:
        plot_df = df[(df['D_element'] == selected_dopant) &
                     (df['x_boundary_type'] == 'exact')].dropna(subset=['x_boundary_value'])
    
    if len(plot_df) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f'Insufficient data for {selected_dopant}',
                ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    b_sites = plot_df['B_element'].unique()
    data_by_b = []
    for b in b_sites:
        if include_lower_bounds:
            data = plot_df[plot_df['B_element'] == b]['x_boundary_value'].dropna()
        else:
            data = plot_df[(plot_df['B_element'] == b) &
                          (plot_df['x_boundary_type'] == 'exact')]['x_boundary_value'].dropna()
        data_by_b.append(data)
    
    bp = ax.boxplot(data_by_b,
                    labels=b_sites,
                    patch_artist=True)
    
    for i, b in enumerate(b_sites):
        bp['boxes'][i].set_facecolor(B_COLORS.get(b, 'lightgray'))
        bp['boxes'][i].set_alpha(0.7)
    
    for i, b in enumerate(b_sites):
        exact_data = plot_df[(plot_df['B_element'] == b) &
                            (plot_df['x_boundary_type'] == 'exact')]['x_boundary_value']
        if len(exact_data) > 0:
            x_pos = np.random.normal(i + 1, 0.05, len(exact_data))
            ax.scatter(x_pos, exact_data, color='red', s=50,
                       alpha=0.7, zorder=3, label='Exact' if i == 0 else '')
        
        lower_data = plot_df[(plot_df['B_element'] == b) &
                            (plot_df['x_boundary_type'] == 'lower_bound')]['x_boundary_value']
        if len(lower_data) > 0:
            x_pos = np.random.normal(i + 1, 0.05, len(lower_data))
            ax.scatter(x_pos, lower_data, color='red', s=50,
                       alpha=0.2, zorder=2, marker='s', label='Lower bound' if i == 0 else '')
    
    ax.set_xlabel('B-site')
    ax.set_ylabel(f'x(boundary) for {selected_dopant} doping')
    ax.set_title(f'Solubility of {selected_dopant} on Different B-sites\n({len(plot_df)} samples)')
    ax.legend()
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
    
    importance_df['feature_clean'] = importance_df['feature'].str.replace('B_element_', 'B=')
    
    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['importance'],
            color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature_clean'])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Random Forest: Factors Affecting Solubility\n(R² = {r2:.3f}, includes lower bounds)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(importance_df['importance']):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    return fig, importance_df

def plot_goldschmidt_bubble(df):
    """График 21: t-Δr фазовая диаграмма с пузырьками"""
    valid = df.dropna(subset=['tolerance_factor', 'dr', 'x_boundary_value'])
    
    if len(valid) < 3:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    exact_valid = valid[valid['x_boundary_type'] == 'exact']
    lower_valid = valid[valid['x_boundary_type'] == 'lower_bound']
    
    if len(exact_valid) > 0:
        scatter_exact = ax.scatter(
            exact_valid['tolerance_factor'],
            exact_valid['dr'],
            s=exact_valid['x_boundary_value'] * 2000,
            c=exact_valid['year'] if 'year' in exact_valid.columns and exact_valid['year'].notna().any() else exact_valid['x_boundary_value'],
            alpha=0.7,
            cmap='viridis',
            edgecolors='black',
            linewidth=0.5,
            label='Exact values'
        )
    
    if len(lower_valid) > 0:
        scatter_lower = ax.scatter(
            lower_valid['tolerance_factor'],
            lower_valid['dr'],
            s=lower_valid['x_boundary_value'] * 2000,
            c=lower_valid['year'] if 'year' in lower_valid.columns and lower_valid['year'].notna().any() else lower_valid['x_boundary_value'],
            alpha=0.3,
            cmap='viridis',
            edgecolors='black',
            linewidth=0.5,
            marker='s',
            label='Lower bounds'
        )
    
    impurity_points = valid[valid['has_impurity']]
    if len(impurity_points) > 0:
        ax.scatter(
            impurity_points['tolerance_factor'],
            impurity_points['dr'],
            s=impurity_points['x_boundary_value'] * 2000,
            facecolors='none',
            edgecolors='red',
            linewidth=2,
            label='With impurities'
        )
    
    for idx, row in valid.nlargest(5, 'x_boundary_value').iterrows():
        marker_style = ' (≥)' if row['x_boundary_type'] == 'lower_bound' else ''
        ax.annotate(f"{row['B_element']}-{row['D_element']}{marker_style}",
                    (row['tolerance_factor'], row['dr']),
                    fontsize=8, ha='center')
    
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel('Tolerance Factor')
    ax.set_ylabel('Δr (Å)')
    ax.set_title('Goldschmidt Diagram with Solubility Information\n(Bubble size = x_boundary)')
    
    if len(exact_valid) > 0:
        cbar = plt.colorbar(scatter_exact, ax=ax)
    elif len(lower_valid) > 0:
        cbar = plt.colorbar(scatter_lower, ax=ax)
    cbar.set_label('Year' if 'year' in valid.columns and valid['year'].notna().any() else 'x(boundary)')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# ============================================================================
# НОВЫЕ ФУНКЦИИ ДЛЯ ГРАФИКОВ (ОБЪЕМНЫЕ И ТЕРМОДИНАМИЧЕСКИЕ)
# ============================================================================
def plot_free_volume_vs_xboundary(df, ax):
    """График 30: Свободный объем vs x_boundary"""
    valid = df.dropna(subset=['free_volume_fraction', 'x_boundary_value'])
    
    if len(valid) < 3:
        ax.text(0.5, 0.5, 'Insufficient data for free volume analysis',
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        exact_mask = mask & (valid['x_boundary_type'] == 'exact')
        lower_mask = mask & (valid['x_boundary_type'] == 'lower_bound')
        
        if exact_mask.any():
            ax.scatter(
                valid.loc[exact_mask, 'free_volume_fraction'],
                valid.loc[exact_mask, 'x_boundary_value'],
                color=color, s=100, alpha=0.9,
                edgecolors='black', linewidth=0.5,
                label=f"{b_element} (exact)"
            )
        
        if lower_mask.any():
            ax.scatter(
                valid.loc[lower_mask, 'free_volume_fraction'],
                valid.loc[lower_mask, 'x_boundary_value'],
                color=color, s=100, alpha=0.3,
                edgecolors='black', linewidth=0.5,
                label=f"{b_element} (≥)",
                marker='s'
            )
    
    ax.set_xlabel('Free Volume Fraction φ = V_free / V_cell')
    ax.set_ylabel('x(boundary)')
    ax.set_title('Solubility Limit vs Free Volume Fraction')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_formation_energy_vs_xboundary(df, ax):
    """График 31: Энергия образования vs x_boundary"""
    valid = df.dropna(subset=['E_form', 'x_boundary_value'])
    
    if len(valid) < 3:
        ax.text(0.5, 0.5, 'Insufficient data for formation energy analysis',
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        exact_mask = mask & (valid['x_boundary_type'] == 'exact')
        lower_mask = mask & (valid['x_boundary_type'] == 'lower_bound')
        
        if exact_mask.any():
            ax.scatter(
                valid.loc[exact_mask, 'E_form'],
                valid.loc[exact_mask, 'x_boundary_value'],
                color=color, s=100, alpha=0.9,
                edgecolors='black', linewidth=0.5,
                label=f"{b_element} (exact)"
            )
        
        if lower_mask.any():
            ax.scatter(
                valid.loc[lower_mask, 'E_form'],
                valid.loc[lower_mask, 'x_boundary_value'],
                color=color, s=100, alpha=0.3,
                edgecolors='black', linewidth=0.5,
                label=f"{b_element} (≥)",
                marker='s'
            )
    
    ax.set_xlabel('Formation Energy (eV/atom)')
    ax.set_ylabel('x(boundary)')
    ax.set_title('Solubility Limit vs Formation Energy')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_3d_stability_phase(df):
    """График 32: 3D фазовая диаграмма (t, Δr, x_boundary)"""
    valid = df.dropna(subset=['tolerance_factor', 'dr', 'x_boundary_value'])
    
    if len(valid) < 4:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.text2D(0.5, 0.5, 'Insufficient data for 3D plot',
                  ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    exact_valid = valid[valid['x_boundary_type'] == 'exact']
    lower_valid = valid[valid['x_boundary_type'] == 'lower_bound']
    
    if len(exact_valid) > 0:
        scatter_exact = ax.scatter(
            exact_valid['tolerance_factor'],
            exact_valid['dr'],
            exact_valid['x_boundary_value'],
            c=exact_valid['x_boundary_value'],
            cmap='viridis',
            s=80,
            alpha=0.9,
            edgecolors='black',
            linewidth=0.5,
            label='Exact values'
        )
    
    if len(lower_valid) > 0:
        ax.scatter(
            lower_valid['tolerance_factor'],
            lower_valid['dr'],
            lower_valid['x_boundary_value'],
            c=lower_valid['x_boundary_value'],
            cmap='viridis',
            s=80,
            alpha=0.3,
            edgecolors='black',
            linewidth=0.5,
            marker='s',
            label='Lower bounds'
        )
    
    ax.set_xlabel('Tolerance Factor (t)')
    ax.set_ylabel('Δr (Å)')
    ax.set_zlabel('x(boundary)')
    ax.set_title('3D Stability Phase Diagram\n(Bubble color = x_boundary)')
    ax.legend()
    
    if len(exact_valid) > 0 or len(lower_valid) > 0:
        plt.colorbar(scatter_exact if len(exact_valid) > 0 else scatter_lower, ax=ax, label='x(boundary)')
    
    return fig

def plot_pairplot_volumetric(df, include_lower_bounds=True):
    """График 33: Pairplot объемных и радиусных параметров"""
    features = ['dr', 'tolerance_factor', 'free_volume_fraction', 'packing_factor', 'x_boundary_value']
    
    if include_lower_bounds:
        plot_df = df[features].dropna()
    else:
        plot_df = df[df['x_boundary_type'] == 'exact'][features].dropna()
    
    if len(plot_df) < 10:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data for pairplot',
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    plot_df_renamed = plot_df.rename(columns={
        'free_volume_fraction': 'φ (free volume)',
        'packing_factor': 'packing factor',
        'x_boundary_value': 'x_boundary'
    })
    
    fig = sns.pairplot(plot_df_renamed, diag_kind='kde',
                       plot_kws={'alpha': 0.6, 'edgecolor': 'black', 's': 50},
                       diag_kws={'color': 'navy', 'alpha': 0.7})
    
    fig.fig.suptitle(f'Correlation Between Volumetric and Geometric Parameters\n'
                     f'({"including" if include_lower_bounds else "excluding"} lower bounds)',
                     y=1.02, fontsize=14)
    
    return fig

def plot_density_prediction(df):
    """График 34: Сравнение расчетной и справочной плотности"""
    valid = df.dropna(subset=['molar_mass', 'packing_factor'])
    
    if len(valid) < 3:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data for density prediction',
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    exact_valid = valid[valid['x_boundary_type'] == 'exact']
    lower_valid = valid[valid['x_boundary_type'] == 'lower_bound']
    
    # Расчет теоретической плотности из молярной массы и объема ячейки
    # Используем плотность из базовой структуры как референс
    base_densities = []
    for idx, row in valid.iterrows():
        a_elem = row.get('A_element', 'Ba')
        b_elem = row['B_element']
        base_props = get_base_properties(a_elem, b_elem)
        if base_props:
            base_densities.append(base_props['density'])
        else:
            base_densities.append(None)
    
    valid = valid.copy()
    valid['base_density'] = base_densities
    valid = valid.dropna(subset=['base_density'])
    
    if len(valid) == 0:
        ax.text(0.5, 0.5, 'No base density data available',
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    exact_valid = valid[valid['x_boundary_type'] == 'exact']
    lower_valid = valid[valid['x_boundary_type'] == 'lower_bound']
    
    if len(exact_valid) > 0:
        ax.scatter(
            exact_valid['base_density'],
            exact_valid['base_density'] * (1 - exact_valid['free_volume_fraction']),
            color='blue', s=100, alpha=0.9,
            edgecolors='black', linewidth=0.5,
            label='Predicted density (exact)'
        )
    
    if len(lower_valid) > 0:
        ax.scatter(
            lower_valid['base_density'],
            lower_valid['base_density'] * (1 - lower_valid['free_volume_fraction']),
            color='green', s=100, alpha=0.5,
            edgecolors='black', linewidth=0.5,
            label='Predicted density (lower bounds)',
            marker='s'
        )
    
    min_val = min(valid['base_density'].min(), valid['base_density'].min())
    max_val = max(valid['base_density'].max(), valid['base_density'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y = x (ideal)')
    
    ax.set_xlabel('Reference Density from Base Structure (g/cm³)')
    ax.set_ylabel('Predicted Density (g/cm³)')
    ax.set_title('Density Prediction from Volumetric Analysis\n'
                 'ρ_pred = ρ_base × (1 - φ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_radar_chart(df):
    """График 35: Radar chart для сравнения систем"""
    # Выбираем топ-5 систем по растворимости
    valid = df.dropna(subset=['x_boundary_value', 'tolerance_factor', 'dr', 'free_volume_fraction', 'packing_factor'])
    valid = valid[valid['x_boundary_type'] == 'exact']
    
    if len(valid) < 3:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data for radar chart (need exact values)',
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    top_systems = valid.nlargest(5, 'x_boundary_value')
    
    # Нормализованные параметры
    params = ['x_boundary_value', 'tolerance_factor', 'free_volume_fraction', 'packing_factor', 'dr_rel']
    param_names = ['Solubility', 'Tolerance\nFactor', 'Free Volume\nFraction', 'Packing\nFactor', 'Relative\nΔr']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_systems)))
    
    for idx, (_, row) in enumerate(top_systems.iterrows()):
        values = []
        for param in params:
            val = row[param]
            # Нормализация
            if param == 'x_boundary_value':
                norm_val = val / valid['x_boundary_value'].max()
            elif param == 'tolerance_factor':
                norm_val = val / valid['tolerance_factor'].max()
            elif param == 'free_volume_fraction':
                norm_val = val / valid['free_volume_fraction'].max() if valid['free_volume_fraction'].max() > 0 else 0
            elif param == 'packing_factor':
                norm_val = val / valid['packing_factor'].max()
            elif param == 'dr_rel':
                norm_val = val / valid['dr_rel'].max() if valid['dr_rel'].max() > 0 else 0
            else:
                norm_val = 0
            values.append(norm_val)
        
        values += values[:1]
        
        label = f"{row['B_element']}-{row['D_element']}"
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx], label=label)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(param_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title('Material Property Profile\n(Top 5 Systems by Solubility)', pad=20, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    return fig

# ============================================================================
# ФУНКЦИЯ ПОСТРОЕНИЯ КОМПЛЕКСНОЙ КОРРЕЛЯЦИОННОЙ МАТРИЦЫ
# ============================================================================
def plot_comprehensive_correlation_matrix(df, include_lower_bounds=True):
    """График: Полная корреляционная матрица со всеми дескрипторами"""
    features = ['dr', 'dr_rel', 'tolerance_factor', 't_gradient', 't_range',
                'Δχ', 'Δχ_gradient', 'free_volume_fraction', 'packing_factor',
                'E_form', 'lattice_strain_energy', 'oxygen_vacancy_conc',
                'x_boundary_value', 'x_max']
    
    available_features = [f for f in features if f in df.columns]
    
    if include_lower_bounds:
        corr_df = df[available_features].dropna()
    else:
        corr_df = df[df['x_boundary_type'] == 'exact'][available_features].dropna()
    
    if len(corr_df) < 5:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.text(0.5, 0.5, 'Insufficient data for comprehensive correlation',
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Переименовываем для читаемости
    rename_map = {
        'dr': 'Δr',
        'dr_rel': 'Δr/r_B',
        'tolerance_factor': 't',
        't_gradient': 'dt/dx',
        't_range': 'Δ t',
        'Δχ': '|Δχ|',
        'Δχ_gradient': 'dΔχ/dx',
        'free_volume_fraction': 'φ',
        'packing_factor': 'η',
        'E_form': 'E_form',
        'lattice_strain_energy': 'ε_strain',
        'oxygen_vacancy_conc': '[V_O]',
        'x_boundary_value': 'x_boundary',
        'x_max': 'x_max'
    }
    
    corr_df_renamed = corr_df.rename(columns=rename_map)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    mask = np.triu(np.ones_like(corr_df_renamed.corr(), dtype=bool))
    sns.heatmap(corr_df_renamed.corr(), mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8})
    
    ax.set_title(f'Comprehensive Correlation Matrix\n'
                 f'({"including" if include_lower_bounds else "excluding"} lower bounds, n={len(corr_df)})',
                 fontsize=14, pad=20)
    
    plt.tight_layout()
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
            - **x(boundary)**: Solubility limit ("-" means ≥ x(inv,end))
            - **Impurity phase(s)**: Impurity phases observed
            - **x(max)**: Concentration at max conductivity
            - **doi**: DOI reference
            """)
            return
        
        # ============================================================================
        # ПОДСКАЗКА С ПОЯСНЕНИЯМИ ВЕЛИЧИН (ДОБАВИТЬ ЗДЕСЬ)
        # ============================================================================
        with st.expander("ℹ️ Variable Definitions & Descriptions", expanded=False):
            st.markdown("""
            ### 📐 **Geometric Parameters**
            
            | Symbol | Description | Formula / Notes |
            |--------|-------------|-----------------|
            | **r_A, r_B, r_D** | Ionic radii of A, B, and D cations | Shannon radii (Å) |
            | **Δr** | Radius difference | `Δr = |r_D - r_B|` (Å) |
            | **Δr/r_B** | Relative radius difference | `Δr_rel = Δr / r_B` |
            | **r_avg_B** | Average B-site radius | `r_avg = (1-x)·r_B + x·r_D` |
            | **t** | Tolerance factor (Goldschmidt) | `t = (r_A + r_O) / [√2·(r_avg_B + r_O)]` |
            | **Δt** | Tolerance factor change | `Δt = t(x=0) - t(x_boundary)` |
            | **dt/dx** | Tolerance factor gradient | Rate of change of t with x |
            
            ### ⚡ **Electronegativity Parameters**
            
            | Symbol | Description | Formula / Notes |
            |--------|-------------|-----------------|
            | **χ_A, χ_B, χ_D** | Electronegativity (Pauling scale) | Values from Pauling scale |
            | **χ_avg_B** | Average B-site electronegativity | `χ_avg = (1-x)·χ_B + x·χ_D` |
            | **Δχ** | Electronegativity difference | `Δχ = |χ_avg_B - χ_A|` |
            | **dΔχ/dx** | Electronegativity gradient | Rate of change of Δχ with x |
            
            ### 🧊 **Volumetric Parameters**
            
            | Symbol | Description | Formula / Notes |
            |--------|-------------|-----------------|
            | **V_cations** | Cation + anion volume | `V_cat = (16π/3)·[r_A³ + (1-x)r_B³ + x·r_D³ + (3-x/2)r_O³]` (Å³) |
            | **V_cell** | Unit cell volume | `V_cell = M / (ρ·N_A)` (Å³) |
            | **V_free** | Free volume | `V_free = V_cell - V_cations` (Å³) |
            | **φ** | Free volume fraction | `φ = V_free / V_cell` |
            | **η** | Packing factor | `η = V_cations / V_cell` |
            
            ### 🔥 **Thermodynamic Parameters**
            
            | Symbol | Description | Formula / Notes |
            |--------|-------------|-----------------|
            | **E_form** | Formation energy | `E_form = (1-x)·E_form(BO₃) + x·E_form(DO₁.₅)` (eV/atom) |
            | **E_g** | Band gap | Linear interpolation between base and dopant oxides (eV) |
            | **ε_strain** | Lattice strain energy | `ε_strain = Δr²·(1-x)·x` (arb. units) |
            | **[V_O]** | Oxygen vacancy concentration | `[V_O] = x/2` (per formula unit) |
            | **M** | Molar mass | `M = M_A + (1-x)M_B + x·M_D + (3-x/2)M_O` (g/mol) |
            
            ### 📊 **Composition Parameters**
            
            | Symbol | Description | Notes |
            |--------|-------------|-------|
            | **x_boundary** | Solubility limit | Maximum x for single-phase solid solution |
            | **x_max** | Optimum conductivity concentration | x where conductivity is maximum |
            | **x_inv_in** | Start of investigated range | Minimum x studied |
            | **x_inv_end** | End of investigated range | Maximum x studied |
            | **x(max)/x(boundary)** | Relative position | Ratio of conductivity max to solubility limit |
            
            ### 🧪 **Phase Parameters**
            
            | Symbol | Description |
            |--------|-------------|
            | **Impurity phase(s)** | Secondary phases observed beyond solubility limit |
            | **has_impurity** | Boolean indicating impurity presence |
            
            ### 📈 **Statistical Parameters**
            
            | Symbol | Description |
            |--------|-------------|
            | **Pearson r** | Linear correlation coefficient |
            | **Spearman ρ** | Rank correlation coefficient |
            | **p-value** | Statistical significance |
            | **R²** | Coefficient of determination |
            """)

        st.markdown("---")
        st.header("📊 Plot Settings")
        
        show_grid = st.checkbox("Show grid", value=True)
        show_legend = st.checkbox("Show legend", value=True)
        
        st.markdown("---")
        st.header("🔢 Data Processing")
        include_lower_bounds = st.checkbox(
            "Include lower bound estimates (≥)", 
            value=True,
            help="When enabled, '-' in x(boundary) is treated as ≥ x(inv,end)"
        )
        
        treat_lower_as_exact = st.checkbox(
            "Treat lower bounds as exact values (use x(inv,end) as solubility limit)",
            value=False,
            help="When enabled, all '-' in x(boundary) are treated as exact values equal to x(inv,end). "
                 "This is useful for statistical analysis where you want to include the full range."
        )
        
        st.markdown("---")
        st.header("🔍 Filters")
        
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            with st.spinner("Processing data..."):
                df_processed = process_data(df, treat_lower_as_exact)
                
                if 'B_element' in df_processed.columns:
                    selected_b = st.multiselect(
                        "B-site elements",
                        options=sorted(df_processed['B_element'].unique()),
                        default=sorted(df_processed['B_element'].unique())
                    )
                else:
                    selected_b = []
                    st.error("B_element column not found")
                
                if 'D_element' in df_processed.columns:
                    selected_d = st.multiselect(
                        "Dopant elements",
                        options=sorted(df_processed['D_element'].unique()),
                        default=sorted(df_processed['D_element'].unique())
                    )
                else:
                    selected_d = []
                    st.error("D_element column not found")
                
                impurity_filter = st.radio(
                    "Impurity phases",
                    options=['All', 'With impurities', 'Without impurities']
                )
                
                x_boundary_type_filter = st.multiselect(
                    "x(boundary) type",
                    options=['exact', 'lower_bound'],
                    default=['exact', 'lower_bound'] if include_lower_bounds else ['exact'],
                    help="Select which types of x(boundary) to include"
                )
                
                filtered_df = df_processed.copy()
                
                if selected_b and 'B_element' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['B_element'].isin(selected_b)]
                
                if selected_d and 'D_element' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['D_element'].isin(selected_d)]
                
                if impurity_filter == 'With impurities' and 'has_impurity' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['has_impurity'] == True]
                elif impurity_filter == 'Without impurities' and 'has_impurity' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['has_impurity'] == False]
                
                if 'x_boundary_type' in filtered_df.columns:
                    if treat_lower_as_exact:
                        # Если treat_lower_as_exact=True, то все lower_bound становятся exact
                        x_boundary_type_filter = ['exact']
                        filtered_df = filtered_df[filtered_df['x_boundary_type'].isin(x_boundary_type_filter)]
                    else:
                        x_boundary_type_filter = st.multiselect(
                            "x(boundary) type",
                            options=['exact', 'lower_bound'],
                            default=['exact', 'lower_bound'] if include_lower_bounds else ['exact'],
                            help="Select which types of x(boundary) to include"
                        )
                        filtered_df = filtered_df[filtered_df['x_boundary_type'].isin(x_boundary_type_filter)]
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
    
    if uploaded_file is not None and len(filtered_df) > 0:
        st.subheader("📈 Data Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total entries", len(filtered_df))
        with col2:
            st.metric("Unique B-site", filtered_df['B_element'].nunique() if 'B_element' in filtered_df.columns else 0)
        with col3:
            st.metric("Unique Dopants", filtered_df['D_element'].nunique() if 'D_element' in filtered_df.columns else 0)
        with col4:
            imp_count = filtered_df['has_impurity'].sum() if 'has_impurity' in filtered_df.columns else 0
            st.metric("With impurities", imp_count)
        with col5:
            if 'x_boundary_type' in filtered_df.columns:
                exact_count = (filtered_df['x_boundary_type'] == 'exact').sum()
                lower_count = (filtered_df['x_boundary_type'] == 'lower_bound').sum()
                st.metric("Exact / Lower", f"{exact_count} / {lower_count}")
        
        st.markdown("---")
        st.subheader("📊 Quick Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Solubility Statistics by B-site**")
            if 'x_boundary_value' in filtered_df.columns and 'B_element' in filtered_df.columns:
                if include_lower_bounds:
                    df_stats = filtered_df.dropna(subset=['x_boundary_value'])
                else:
                    df_stats = filtered_df[filtered_df['x_boundary_type'] == 'exact'].dropna(subset=['x_boundary_value'])
                
                if len(df_stats) > 0:
                    b_stats = df_stats.groupby('B_element')['x_boundary_value'].agg(['mean', 'median', 'count', 'std']).round(3)
                    st.dataframe(b_stats, use_container_width=True)
        
        with col2:
            st.markdown("**Top 5 Dopants by Median Solubility**")
            dopant_stats = get_dopant_statistics(filtered_df, include_lower_bounds)
            if len(dopant_stats) > 0:
                top_d = dopant_stats.head(5)[['Dopant', 'Median', 'Count', 'Exact values', 'Lower bounds']]
                st.dataframe(top_d, use_container_width=True)
        
        with col3:
            st.markdown("**Conductivity Maximum Position**")
            if 'x_max' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
                valid = filtered_df.dropna(subset=['x_max', 'x_boundary_value'])
                if len(valid) > 0:
                    diff = (valid['x_max'] - valid['x_boundary_value']) / valid['x_boundary_value']
                    within_10pct = (abs(diff) < 0.1).mean()
                    st.metric("Within ±10% of boundary", f"{within_10pct:.1%}")
                    st.metric("Median relative position", f"{diff.median():.3f}")
                    st.metric("Samples with x(max) > x(boundary)", f"{(diff > 0).mean():.1%}")
        
        with st.expander("📋 View processed data"):
            display_cols = ['B_element', 'D_element', 'x_boundary_value', 'x_boundary_type',
                           'x_boundary_raw', 'x_inv_in', 'x_inv_end', 'x_max',
                           'dr', 'tolerance_factor', 'free_volume_fraction', 'packing_factor',
                           'E_form', 'Δχ', 'has_impurity', 'year']
            available_cols = [col for col in display_cols if col in filtered_df.columns]
            st.dataframe(filtered_df[available_cols], use_container_width=True)
            
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download processed data as CSV",
                csv,
                "perovskite_data_processed.csv",
                "text/csv"
            )
        
        st.markdown("---")
        
        # Вкладки
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Basic Statistics",
            "🔬 Solubility Analysis",
            "⚡ Conductivity Analysis",
            "📈 Advanced Visualization",
            "🧊 Volumetric & Thermodynamic",
            "🤖 ML Insights"
        ])
        
        # ============================================================================
        # ВКЛАДКА 1: BASIC STATISTICS
        # ============================================================================
        with tab1:
            st.subheader("Basic Statistical Analysis")
            
            if len(filtered_df) > 0:
                fig, stats_df = plot_b_site_statistics(filtered_df, include_lower_bounds, treat_lower_as_exact)
                st.pyplot(fig)
                plt.close(fig)
            
            elif plot_name == "Top Dopants Violin Plot":
                if 'x_boundary_value' in filtered_df.columns and 'D_element' in filtered_df.columns:
                    plt.close(fig)
                    violin_fig = plot_top_dopants_violin(filtered_df, include_lower_bounds, treat_lower_as_exact)
                    st.pyplot(violin_fig)
                    plt.close(violin_fig)
                    plot_idx -= 1
            
            dopant_stats = get_dopant_statistics(filtered_df, include_lower_bounds, treat_lower_as_exact)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'x_max' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
                    fig = plot_xmax_vs_boundary_histogram(filtered_df)
                    st.pyplot(fig)
                    plt.close(fig)
            
            with col2:
                if 'x_boundary_value' in filtered_df.columns:
                    fig = plot_distribution_kde(filtered_df)
                    st.pyplot(fig)
                    plt.close(fig)
            
            if len(filtered_df) > 0:
                fig = plot_correlation_heatmap(filtered_df, include_lower_bounds)
                st.pyplot(fig)
                plt.close(fig)
            
            # Полная корреляционная матрица
            st.subheader("Comprehensive Correlation Matrix")
            fig = plot_comprehensive_correlation_matrix(filtered_df, include_lower_bounds)
            st.pyplot(fig)
            plt.close(fig)
            
            st.subheader("Detailed Correlations with p-values")
            features = ['dr', 'dr_rel', 'tolerance_factor', 'free_volume_fraction',
                        'packing_factor', 'x_boundary_value', 'x_max']
            available_features = [f for f in features if f in filtered_df.columns]
            if len(available_features) >= 2:
                corr_df = calculate_correlations(filtered_df, available_features, include_lower_bounds)
                if len(corr_df) > 0:
                    st.dataframe(corr_df, use_container_width=True)
        
        # ============================================================================
        # ВКЛАДКА 2: SOLUBILITY ANALYSIS
        # ============================================================================
        with tab2:
            st.subheader("Solubility Limit Analysis")
            
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
                        if 'dr' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
                            plot_solubility_vs_dr(filtered_df, ax)
                        else:
                            ax.text(0.5, 0.5, 'Required data missing', ha='center', va='center')
                    
                    elif plot_name == "Solubility vs Tolerance Factor":
                        if 'tolerance_factor' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
                            plot_tolerance_factor(filtered_df, ax)
                        else:
                            ax.text(0.5, 0.5, 'Required data missing', ha='center', va='center')
                    
                    elif plot_name == "Top Dopants Violin Plot":
                        if 'x_boundary_value' in filtered_df.columns and 'D_element' in filtered_df.columns:
                            plt.close(fig)
                            violin_fig = plot_top_dopants_violin(filtered_df, include_lower_bounds)
                            st.pyplot(violin_fig)
                            plt.close(violin_fig)
                            plot_idx -= 1
                    
                    elif plot_name == "Critical Δr Threshold":
                        if 'dr' in filtered_df.columns and 'has_impurity' in filtered_df.columns:
                            plt.close(fig)
                            threshold_fig = plot_critical_dr_threshold(filtered_df)
                            st.pyplot(threshold_fig)
                            plt.close(threshold_fig)
                            plot_idx -= 1
                    
                    elif plot_name == "Research Intensity Matrix":
                        if 'B_element' in filtered_df.columns and 'D_element' in filtered_df.columns:
                            plt.close(fig)
                            matrix_fig, pub_matrix = plot_publication_matrix(filtered_df)
                            st.pyplot(matrix_fig)
                            plt.close(matrix_fig)
                            st.dataframe(pub_matrix, use_container_width=True)
                            plot_idx -= 1
                    
                    if not show_grid:
                        ax.grid(False)
                    if not show_legend:
                        ax.legend().remove()
                    
                    plot_idx += 1
                
                for j in range(plot_idx, len(axes)):
                    fig.delaxes(axes[j])
                
                if plot_idx > 0:
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            
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
                    if 'x_boundary_value' in filtered_df.columns:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plot_xmax_vs_xboundary(filtered_df, ax)
                        if not show_grid:
                            ax.grid(False)
                        if not show_legend:
                            ax.legend().remove()
                        st.pyplot(fig)
                        plt.close(fig)
                
                with col2:
                    if 'tolerance_factor' in filtered_df.columns:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plot_xmax_vs_tolerance(filtered_df, ax)
                        if not show_grid:
                            ax.grid(False)
                        if not show_legend:
                            ax.legend().remove()
                        st.pyplot(fig)
                        plt.close(fig)
                
                if 'x_rel_max' in filtered_df.columns and 'dr_rel' in filtered_df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_relative_position(filtered_df, ax)
                    if not show_grid:
                        ax.grid(False)
                    if not show_legend:
                        ax.legend().remove()
                    st.pyplot(fig)
                    plt.close(fig)
                
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
                    if 'year' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
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
                        selected_dopant = st.selectbox(
                            "Select dopant for comparison",
                            options=sorted(filtered_df['D_element'].unique())
                        )
                        fig = plot_dopant_comparison_boxplot(filtered_df, selected_dopant, include_lower_bounds)
                        st.pyplot(fig)
                        plt.close(fig)
        
        # ============================================================================
        # ВКЛАДКА 5: VOLUMETRIC & THERMODYNAMIC
        # ============================================================================
        with tab5:
            st.subheader("Volumetric and Thermodynamic Analysis")
            
            vol_plots = st.multiselect(
                "Select volumetric/thermodynamic plots",
                options=[
                    "Free Volume vs Solubility",
                    "Formation Energy vs Solubility",
                    "3D Stability Phase Diagram",
                    "Pairplot: Volumetric Parameters",
                    "Density Prediction",
                    "Radar Chart: Material Profiles"
                ],
                default=["Free Volume vs Solubility", "Formation Energy vs Solubility"]
            )
            
            for plot_name in vol_plots:
                if plot_name == "Free Volume vs Solubility":
                    if 'free_volume_fraction' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        plot_free_volume_vs_xboundary(filtered_df, ax)
                        if not show_grid:
                            ax.grid(False)
                        if not show_legend:
                            ax.legend().remove()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning("Free volume data not available. Check if V_cell and V_cations were calculated.")
                
                elif plot_name == "Formation Energy vs Solubility":
                    if 'E_form' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        plot_formation_energy_vs_xboundary(filtered_df, ax)
                        if not show_grid:
                            ax.grid(False)
                        if not show_legend:
                            ax.legend().remove()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning("Formation energy data not available.")
                
                elif plot_name == "3D Stability Phase Diagram":
                    if 'tolerance_factor' in filtered_df.columns and 'dr' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
                        fig = plot_3d_stability_phase(filtered_df)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning("Required data for 3D plot not available.")
                
                elif plot_name == "Pairplot: Volumetric Parameters":
                    if 'free_volume_fraction' in filtered_df.columns and 'packing_factor' in filtered_df.columns:
                        fig = plot_pairplot_volumetric(filtered_df, include_lower_bounds)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning("Volumetric parameters not available.")
                
                elif plot_name == "Density Prediction":
                    fig = plot_density_prediction(filtered_df)
                    st.pyplot(fig)
                    plt.close(fig)
                
                elif plot_name == "Radar Chart: Material Profiles":
                    fig = plot_radar_chart(filtered_df)
                    st.pyplot(fig)
                    plt.close(fig)
            
            st.subheader("Volumetric Statistics")
            if 'free_volume_fraction' in filtered_df.columns:
                vol_stats = filtered_df['free_volume_fraction'].dropna()
                if len(vol_stats) > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Free Volume Fraction", f"{vol_stats.mean():.3f}")
                    with col2:
                        st.metric("Median Free Volume Fraction", f"{vol_stats.median():.3f}")
                    with col3:
                        st.metric("Std Dev", f"{vol_stats.std():.3f}")
        
        # ============================================================================
        # ВКЛАДКА 6: ML INSIGHTS
        # ============================================================================
        with tab6:
            st.subheader("Machine Learning Insights")
            
            st.markdown("**Feature Importance Analysis (Random Forest)**")
            fig, importance_df = plot_feature_importance(filtered_df)
            st.pyplot(fig)
            plt.close(fig)
            
            if importance_df is not None:
                st.dataframe(importance_df, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Publication Analysis")
            
            if 'doi' in filtered_df.columns:
                st.metric("Total publications", filtered_df['doi'].nunique())
            
            if 'year' in filtered_df.columns:
                year_stats = filtered_df['year'].dropna()
                if len(year_stats) > 0:
                    st.metric("Year range", f"{int(year_stats.min())} - {int(year_stats.max())}")
                    st.metric("Median year", int(year_stats.median()))
                    
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
        
        st.markdown("### Example of expected data format:")
        st.markdown("**Note:** In x(boundary) column, '-' means the solubility is at least x(inv,end)")
        example_data = pd.DataFrame({
            'A': ['Ba', 'Ba', 'Ba'],
            'B': ['Ce', 'Ce', 'Zr'],
            'D': ['Gd', 'Sm', 'Y'],
            'x(inv,in)': [0.05, 0, 0],
            'x(inv,end)': [0.2, 0.25, 0.3],
            'x(boundary)': ['-', '0.22', '0.27'],
            'Impurity phase(s)': ['-', 'BaSm2O4', 'Y2O3'],
            'x(max)': ['0.15', '0.15', '0.25'],
            'doi': ['10.1016/j.jallcom.2009.05.108',
                   '10.1016/j.jpowsour.2008.01.036',
                   '10.1016/j.ssi.2007.01.015']
        })
        
        st.dataframe(example_data, use_container_width=True)
        
        st.markdown("""
        ### Key improvements in this version:
        - **New volumetric descriptors**: V_cations, V_cell, V_free, packing factor, free volume fraction
        - **New thermodynamic descriptors**: Formation energy, band gap, lattice strain energy
        - **New plots**: Free volume vs solubility, formation energy vs solubility, 3D phase diagram, pairplot, density prediction, radar chart
        - **Comprehensive correlation matrix** with all 15+ descriptors
        - **"-" in x(boundary)** is now interpreted as **lower bound estimate** (≥ x(inv,end))
        - Visual distinction between exact values and lower bounds (transparent markers)
        - Separate statistics for exact values and lower bounds
        - **New tab**: Volumetric & Thermodynamic Analysis
        """)

if __name__ == "__main__":
    main()
