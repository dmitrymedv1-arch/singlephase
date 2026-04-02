import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata
import re
from datetime import datetime
import openpyxl
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import xgboost as xgb
from itertools import combinations
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
# БАЗА ДАННЫХ ЗАРЯДОВ ИОНОВ
# ============================================================================
IONIC_CHARGES = {
    'Ba': 2,
    'Sr': 2,
    'Ce': 4,
    'Zr': 4,
    'Sn': 4,
    'Ti': 4,
    'Hf': 4,
    'Gd': 3,
    'Sm': 3,
    'Y': 3,
    'In': 3,
    'Sc': 3,
    'Dy': 3,
    'Ho': 3,
    'Yb': 3,
    'Eu': 3,
    'Nd': 3,
    'La': 3,
    'Pr': 3,
    'Tb': 3,
    'Er': 3,
    'Tm': 3,
    'Lu': 3,
    'Ca': 2,
    'O': -2,
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
# НОВЫЙ КЛАСС: РАСЧЕТЧИК ДЕСКРИПТОРОВ ПЕРОВСКИТОВ
# ============================================================================
class PerovskiteDescriptorCalculator:
    """Класс для расчета всех физико-химических дескрипторов перовскитов"""
    
    def __init__(self, a_element='Ba'):
        self.a_element = a_element
        self.r_O = IONIC_RADII.get(('O', -2, 6), 1.4)
        self.χ_O = ELECTRONEGATIVITY.get('O', 3.44)
        self.r_A = IONIC_RADII.get((a_element, 2, 12), None)
        self.χ_A = ELECTRONEGATIVITY.get(a_element, None)
        self.z_A = IONIC_CHARGES.get(a_element, 2)
    
    def get_ionic_radius(self, element, charge=None, coordination=6):
        """Получение ионного радиуса с автоматическим определением заряда"""
        if charge is None:
            charge = IONIC_CHARGES.get(element, 4)
        return IONIC_RADII.get((element, charge, coordination), None)
    
    def get_electronegativity(self, element):
        return ELECTRONEGATIVITY.get(element, None)
    
    def get_charge(self, element):
        return IONIC_CHARGES.get(element, None)
    
    def calculate_descriptors(self, b_element, d_element, x, r_B=None, r_D=None, χ_B=None, χ_D=None):
        """
        Расчет полного набора дескрипторов для твердого раствора AB_{1-x}D_xO_{3-x/2}
        
        Parameters
        ----------
        b_element : str
            Элемент B-сайта
        d_element : str
            Элемент допанта
        x : float
            Концентрация допанта
        r_B : float, optional
            Ионный радиус B (если None, берется из базы)
        r_D : float, optional
            Ионный радиус D (если None, берется из базы)
        χ_B : float, optional
            Электроотрицательность B (если None, берется из базы)
        χ_D : float, optional
            Электроотрицательность D (если None, берется из базы)
        
        Returns
        -------
        dict
            Словарь с рассчитанными дескрипторами
        """
        # Получение значений из базы, если не переданы
        if r_B is None:
            r_B = self.get_ionic_radius(b_element)
        if r_D is None:
            r_D = self.get_ionic_radius(d_element)
        if χ_B is None:
            χ_B = self.get_electronegativity(b_element)
        if χ_D is None:
            χ_D = self.get_electronegativity(d_element)
        
        z_B = self.get_charge(b_element)
        z_D = self.get_charge(d_element)
        
        descriptors = {}
        
        # ====================================================================
        # 1. ГЕОМЕТРИЧЕСКИЕ ДЕСКРИПТОРЫ (радиусы, размерный мисфит)
        # ====================================================================
        
        # Базовые радиусы
        descriptors['r_B'] = r_B
        descriptors['r_D'] = r_D
        descriptors['dr'] = abs(r_D - r_B) if None not in [r_B, r_D] else None
        descriptors['dr_rel'] = descriptors['dr'] / r_B if (descriptors['dr'] is not None and r_B is not None and r_B != 0) else None
        
        # Средний радиус B-сайта
        if None not in [r_B, r_D]:
            descriptors['r_avg_B'] = (1 - x) * r_B + x * r_D
        else:
            descriptors['r_avg_B'] = None
        
        # Квадратичный размерный мисфит (size misfit) - более физичный дескриптор
        # Основан на теории упругости: энергия деформации ∝ (Δr)²
        if None not in [r_B, r_D, x]:
            descriptors['size_misfit'] = x * (r_D - r_B) ** 2 / r_B
            descriptors['elastic_misfit'] = ((r_D - r_B) / r_B) ** 2 * x * (1 - x)
        else:
            descriptors['size_misfit'] = None
            descriptors['elastic_misfit'] = None
        
        # Tolerance factor Гольдшмидта
        if None not in [self.r_A, descriptors['r_avg_B'], self.r_O]:
            descriptors['tolerance_factor'] = (self.r_A + self.r_O) / (np.sqrt(2) * (descriptors['r_avg_B'] + self.r_O))
        else:
            descriptors['tolerance_factor'] = None
        
        # Tolerance factor при x=0
        if None not in [self.r_A, r_B, self.r_O]:
            descriptors['t0'] = (self.r_A + self.r_O) / (np.sqrt(2) * (r_B + self.r_O))
        else:
            descriptors['t0'] = None
        
        # Градиент tolerance factor
        if None not in [r_B, r_D, x, self.r_A, self.r_O] and x > 0:
            x_vals = np.linspace(0, x, 100)
            t_vals = []
            for xi in x_vals:
                r_avg = (1 - xi) * r_B + xi * r_D
                t = (self.r_A + self.r_O) / (np.sqrt(2) * (r_avg + self.r_O))
                t_vals.append(t)
            if len(t_vals) > 1:
                descriptors['t_gradient'] = (t_vals[-1] - t_vals[0]) / x_vals[-1]
            else:
                descriptors['t_gradient'] = None
        else:
            descriptors['t_gradient'] = None
        
        # Диапазон изменения t
        if None not in [descriptors['t0'], descriptors['tolerance_factor']]:
            descriptors['t_range'] = descriptors['t0'] - descriptors['tolerance_factor']
        else:
            descriptors['t_range'] = None
        
        # Склонность к октаэдрическим наклонам (оценка)
        # По Goldschmidt: при t < 0.96 начинаются наклоны
        if descriptors['tolerance_factor'] is not None:
            descriptors['octahedral_tilting'] = max(0, 0.96 - descriptors['tolerance_factor']) * 10
        else:
            descriptors['octahedral_tilting'] = None
        
        # ====================================================================
        # 2. ЭЛЕКТРОННЫЕ ДЕСКРИПТОРЫ (электроотрицательность, прочность связи)
        # ====================================================================
        
        descriptors['χ_B'] = χ_B
        descriptors['χ_D'] = χ_D
        descriptors['χ_A'] = self.χ_A
        
        # Средняя электроотрицательность B-сайта
        if None not in [χ_B, χ_D]:
            descriptors['χ_avg_B'] = (1 - x) * χ_B + x * χ_D
        else:
            descriptors['χ_avg_B'] = None
        
        # Разница электроотрицательностей между B-сайтом и A-сайтом
        if None not in [descriptors['χ_avg_B'], self.χ_A]:
            descriptors['Δχ'] = abs(descriptors['χ_avg_B'] - self.χ_A)
        else:
            descriptors['Δχ'] = None
        
        # Градиент Δχ
        if None not in [χ_B, χ_D, x] and x > 0:
            descriptors['Δχ_gradient'] = (χ_D - χ_B) / x if χ_D is not None and χ_B is not None else None
        else:
            descriptors['Δχ_gradient'] = None
        
        # ЭФФЕКТИВНАЯ ЭЛЕКТРООТРИЦАТЕЛЬНОСТЬ ОТНОСИТЕЛЬНО КИСЛОРОДА
        # Δχ_eff = |χ_avg_B - χ_O| - |χ_A - χ_O|
        # Оценивает, насколько B-сайт более/менее электроотрицателен относительно кислорода, чем A-сайт
        if None not in [descriptors['χ_avg_B'], self.χ_A, self.χ_O]:
            descriptors['Δχ_eff'] = abs(descriptors['χ_avg_B'] - self.χ_O) - abs(self.χ_A - self.χ_O)
        else:
            descriptors['Δχ_eff'] = None
        
        # Прочность связи по Полингу (квадрат разности электроотрицательностей)
        if None not in [χ_B, self.χ_O]:
            descriptors['bond_strength_B'] = (χ_B - self.χ_O) ** 2
        else:
            descriptors['bond_strength_B'] = None
        
        if None not in [χ_D, self.χ_O]:
            descriptors['bond_strength_D'] = (χ_D - self.χ_O) ** 2
        else:
            descriptors['bond_strength_D'] = None
        
        # Дисперсия электроотрицательности на B-сайте
        if None not in [χ_B, χ_D, x]:
            descriptors['electronegativity_variance'] = x * (1 - x) * (χ_D - χ_B) ** 2
        else:
            descriptors['electronegativity_variance'] = None
        
        # ====================================================================
        # 3. ИОННЫЕ ДЕСКРИПТОРЫ (ионный потенциал, поляризующая способность)
        # ====================================================================
        
        # Ионный потенциал (z/r) - мера поляризующей способности катиона
        if None not in [z_B, r_B] and r_B is not None and r_B > 0:
            descriptors['ionic_potential_B'] = z_B / r_B
        else:
            descriptors['ionic_potential_B'] = None
        
        if None not in [z_D, r_D] and r_D is not None and r_D > 0:
            descriptors['ionic_potential_D'] = z_D / r_D
        else:
            descriptors['ionic_potential_D'] = None
        
        # Средний ионный потенциал B-сайта
        if None not in [descriptors['ionic_potential_B'], descriptors['ionic_potential_D']]:
            descriptors['ionic_potential_avg'] = (1 - x) * descriptors['ionic_potential_B'] + x * descriptors['ionic_potential_D']
        else:
            descriptors['ionic_potential_avg'] = None
        
        # Разность ионных потенциалов
        if None not in [descriptors['ionic_potential_B'], descriptors['ionic_potential_D']]:
            descriptors['Δ_ionic_potential'] = abs(descriptors['ionic_potential_D'] - descriptors['ionic_potential_B'])
        else:
            descriptors['Δ_ionic_potential'] = None
        
        # ====================================================================
        # 4. ОБЪЕМНЫЕ ДЕСКРИПТОРЫ
        # ====================================================================
        
        # Объем катионов и анионов
        if None not in [self.r_A, r_B, r_D, self.r_O]:
            term_A = self.r_A ** 3
            term_B = (1 - x) * (r_B ** 3)
            term_D = x * (r_D ** 3)
            term_O = (3 - x/2) * (self.r_O ** 3)
            descriptors['V_cations'] = PREFACTOR_VOLUME * (term_A + term_B + term_D + term_O)
        else:
            descriptors['V_cations'] = None
        
        # Молярная масса
        M_A = ATOMIC_MASSES.get(self.a_element, None)
        M_B = ATOMIC_MASSES.get(b_element, None)
        M_D = ATOMIC_MASSES.get(d_element, None)
        M_O = ATOMIC_MASSES['O']
        
        if None not in [M_A, M_B, M_D]:
            descriptors['molar_mass'] = M_A + (1 - x) * M_B + x * M_D + (3 - x/2) * M_O
        else:
            descriptors['molar_mass'] = None
        
        # Объем ячейки (из плотности базовой структуры)
        base_props = MATERIAL_PROPERTIES.get(f"{self.a_element}{b_element}O3", None)
        if base_props is not None and descriptors['molar_mass'] is not None:
            density = base_props.get('density', None)
            if density is not None and density > 0:
                descriptors['V_cell'] = descriptors['molar_mass'] / (density * AVOGADRO_NUMBER) * 1e24
            else:
                descriptors['V_cell'] = None
        else:
            descriptors['V_cell'] = None
        
        # Свободный объем
        if None not in [descriptors['V_cell'], descriptors['V_cations']]:
            descriptors['V_free'] = descriptors['V_cell'] - descriptors['V_cations']
            descriptors['packing_factor'] = descriptors['V_cations'] / descriptors['V_cell']
            descriptors['free_volume_fraction'] = descriptors['V_free'] / descriptors['V_cell']
        else:
            descriptors['V_free'] = None
            descriptors['packing_factor'] = None
            descriptors['free_volume_fraction'] = None
        
        # ====================================================================
        # 5. ТЕРМОДИНАМИЧЕСКИЕ ДЕСКРИПТОРЫ
        # ====================================================================
        
        # Энергия образования
        if base_props is not None:
            E_form_D = {
                'Gd': -2.8, 'Sm': -2.7, 'Y': -2.9, 'In': -2.5, 'Sc': -3.0,
                'Dy': -2.8, 'Ho': -2.8, 'Yb': -2.7, 'Eu': -2.6, 'Nd': -2.6,
                'La': -2.5, 'Pr': -2.6, 'Tb': -2.7, 'Er': -2.8, 'Tm': -2.8, 'Lu': -2.9
            }.get(d_element, -2.7)
            E_form_base = base_props.get('E_form', 0)
            descriptors['E_form'] = (1 - x) * E_form_base + x * E_form_D
        else:
            descriptors['E_form'] = None
        
        # Ширина запрещенной зоны
        if base_props is not None:
            band_gap_base = base_props.get('band_gap', None)
            if band_gap_base is not None:
                band_gap_D = {
                    'Gd': 5.2, 'Sm': 4.8, 'Y': 5.5, 'In': 3.7, 'Sc': 6.0,
                    'Dy': 5.0, 'Ho': 5.1, 'Yb': 4.5, 'Eu': 4.6, 'Nd': 4.7,
                    'La': 5.6, 'Pr': 4.9, 'Tb': 5.0, 'Er': 5.2, 'Tm': 5.3, 'Lu': 5.8
                }.get(d_element, 5.0)
                descriptors['band_gap'] = (1 - x) * band_gap_base + x * band_gap_D
            else:
                descriptors['band_gap'] = None
        else:
            descriptors['band_gap'] = None
        
        # Энергия деформации решетки (уже есть, но добавим еще квадратичную форму)
        if None not in [r_B, r_D, x]:
            descriptors['lattice_strain_energy'] = (r_D - r_B) ** 2 * (1 - x) * x
        else:
            descriptors['lattice_strain_energy'] = None
        
        # Концентрация кислородных вакансий
        descriptors['oxygen_vacancy_conc'] = x / 2 if x is not None else None
        
        # ====================================================================
        # 6. ГИБРИДНЫЕ ДЕСКРИПТОРЫ (комбинации базовых параметров)
        # ====================================================================
        
        # t * (1 + dr/r_B) - комбинация геометрии и мисфита
        if None not in [descriptors['tolerance_factor'], descriptors['dr_rel']]:
            descriptors['t_dr_hybrid'] = descriptors['tolerance_factor'] * (1 + descriptors['dr_rel'])
        else:
            descriptors['t_dr_hybrid'] = None
        
        # Δχ * dr - произведение (часто дает хорошую разделимость)
        if None not in [descriptors['Δχ'], descriptors['dr']]:
            descriptors['Δχ_dr_hybrid'] = descriptors['Δχ'] * descriptors['dr']
        else:
            descriptors['Δχ_dr_hybrid'] = None
        
        # ε_strain / t - нормализованная энергия деформации
        if None not in [descriptors['lattice_strain_energy'], descriptors['tolerance_factor']] and descriptors['tolerance_factor'] != 0:
            descriptors['strain_t_ratio'] = descriptors['lattice_strain_energy'] / descriptors['tolerance_factor']
        else:
            descriptors['strain_t_ratio'] = None
        
        # oxygen_vacancy_conc * free_volume_fraction - эффективная подвижность вакансий
        if None not in [descriptors['oxygen_vacancy_conc'], descriptors['free_volume_fraction']]:
            descriptors['vacancy_mobility_proxy'] = descriptors['oxygen_vacancy_conc'] * descriptors['free_volume_fraction']
        else:
            descriptors['vacancy_mobility_proxy'] = None
        
        return descriptors

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

def process_x_boundary(value, inv_end=None):
    """
    Специальная обработка для x(boundary)
    Возвращает (числовое_значение, тип_значения, исходная_строка)
    тип_значения: 'exact' - точное значение, 'lower_bound' - нижняя оценка, 'none' - нет данных
    """
    if pd.isna(value) or value == '' or value == '-' or value == '—':
        if inv_end is not None and not pd.isna(inv_end) and inv_end != '' and inv_end != '-':
            try:
                numeric_value = float(inv_end)
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
# ФУНКЦИЯ РАСЧЕТА ВСЕХ ДЕСКРИПТОРОВ (ОБНОВЛЕННАЯ С НОВЫМ КЛАССОМ)
# ============================================================================
@st.cache_data
def calculate_descriptors(row, aggregate_lower_bounds=False):
    """Расчет всех дескрипторов для одной строки (обновленная версия с новыми дескрипторами)
    
    Parameters
    ----------
    row : pandas.Series
        Строка с данными
    aggregate_lower_bounds : bool
        Если True, для lower_bound записей используется x_boundary = x_inv_end
        Если False, для lower_bound записей x_boundary = None (исключается из точных расчетов)
    """
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
            'oxygen_vacancy_conc': None, 'molar_mass': None,
            # Новые дескрипторы
            'size_misfit': None, 'elastic_misfit': None, 't0': None,
            'Δχ_eff': None, 'bond_strength_B': None, 'bond_strength_D': None,
            'electronegativity_variance': None, 'ionic_potential_B': None,
            'ionic_potential_D': None, 'ionic_potential_avg': None,
            'Δ_ionic_potential': None, 'octahedral_tilting': None,
            't_dr_hybrid': None, 'Δχ_dr_hybrid': None, 'strain_t_ratio': None,
            'vacancy_mobility_proxy': None, 'log_x_boundary': None,
            'solubility_energy_proxy': None
        }
    
    A = row.get('A_element', 'Ba')
    B = row['B_element']
    D = row['D_element']
    
    # Получаем x_boundary_value в зависимости от режима агрегации
    if aggregate_lower_bounds and row.get('x_boundary_type') == 'lower_bound':
        x = row.get('x_inv_end', 0)
        if pd.isna(x):
            x = 0
    else:
        x = row.get('x_boundary_value', 0)
        if pd.isna(x):
            x = 0
    
    # Используем новый класс для расчета дескрипторов
    calculator = PerovskiteDescriptorCalculator(a_element=A)
    
    # Получаем радиусы и электроотрицательности
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
    
    χ_B = get_electronegativity(B)
    χ_D = get_electronegativity(D)
    
    # Расчет всех дескрипторов через класс
    descriptors = calculator.calculate_descriptors(B, D, x, r_B, r_D, χ_B, χ_D)
    
    # Логарифмическая целевая переменная
    if x is not None and x > 0:
        epsilon = 1e-6
        descriptors['log_x_boundary'] = np.log10(x + epsilon)
        descriptors['solubility_energy_proxy'] = -np.log(x + epsilon)
    else:
        descriptors['log_x_boundary'] = None
        descriptors['solubility_energy_proxy'] = None
    
    # Добавляем информацию о типе x_boundary
    descriptors['x_boundary_original'] = x
    descriptors['x_boundary_type'] = row.get('x_boundary_type', 'exact')
    
    return descriptors

# ============================================================================
# ФУНКЦИЯ ПРОЦЕССИНГА ДАННЫХ (РАСШИРЕННАЯ)
# ============================================================================
@st.cache_data
def process_data(df, aggregate_lower_bounds=False):
    """Основная функция обработки данных (расширенная версия)
    
    Parameters
    ----------
    df : pandas.DataFrame
        Исходные данные
    aggregate_lower_bounds : bool
        Если True, для lower_bound записей используется x_boundary = x_inv_end
        Если False, для lower_bound записей x_boundary = None (исключается из точных расчетов)
    """
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
            # Обработка строк с #ЗНАЧ! и другими ошибками Excel
            df_processed[col] = df_processed[col].apply(
                lambda x: np.nan if isinstance(x, str) and ('#ЗНАЧ' in x or 'ERROR' in x.upper()) else x
            )
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Обработка электроотрицательностей из файла
    for col in ['χA_raw', 'χB_raw', 'χD_raw']:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            df_processed[col.replace('_raw', '')] = df_processed[col]
    
    # СПЕЦИАЛЬНАЯ ОБРАБОТКА ДЛЯ x_boundary
    if 'x_boundary' in df_processed.columns:
        # Создаем колонки для хранения обработанных значений
        x_boundary_values = []
        x_boundary_types = []
        x_boundary_raw = []
        
        for idx, row in df_processed.iterrows():
            x_boundary_raw_val = row['x_boundary']
            x_inv_end_val = row.get('x_inv_end', None)
            
            # Обработка строк с #ЗНАЧ!
            if isinstance(x_boundary_raw_val, str) and '#ЗНАЧ' in x_boundary_raw_val:
                x_boundary_raw_val = '-'
            
            numeric_val, val_type, raw_str = process_x_boundary(x_boundary_raw_val, x_inv_end_val)
            
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
        df_processed['impurity'] = df_processed['impurity'].replace(['-', '--', '', '#ЗНАЧ!'], 'none')
        df_processed['has_impurity'] = df_processed['impurity'] != 'none'
    else:
        df_processed['impurity'] = 'none'
        df_processed['has_impurity'] = False
    
    # Извлекаем год из DOI
    if 'doi' in df_processed.columns:
        df_processed['year'] = df_processed['doi'].apply(extract_year_from_doi)
    else:
        df_processed['year'] = None
    
    # ============================================================================
    # НОВАЯ ЛОГИКА: В АГРЕГИРОВАННОМ РЕЖИМЕ ВЫПОЛНЯЕМ ДЕДУПЛИКАЦИЮ
    # ============================================================================
    if aggregate_lower_bounds:
        # Создаем агрегированную колонку для x_boundary
        df_processed['x_boundary_aggregated'] = df_processed.apply(
            lambda row: row['x_inv_end'] if row['x_boundary_type'] == 'lower_bound' else row['x_boundary_value'],
            axis=1
        )
        
        # Для каждой уникальной комбинации (B_element, D_element) берем максимальное значение
        # Это важно, так как если есть несколько записей с разными x_inv_end для одной пары,
        # реальная растворимость ≥ максимального из исследованных значений
        grouped = df_processed.groupby(['B_element', 'D_element'])['x_boundary_aggregated'].max().reset_index()
        grouped.rename(columns={'x_boundary_aggregated': 'x_boundary_aggregated_max'}, inplace=True)
        
        # Объединяем обратно
        df_processed = df_processed.merge(grouped, on=['B_element', 'D_element'], how='left')
        
        # Для каждой строки устанавливаем максимальное значение
        # Это гарантирует, что для пары B-D с несколькими записями мы используем максимальную нижнюю оценку
        df_processed['x_boundary_value'] = df_processed['x_boundary_aggregated_max']
        
        # Обновляем тип для агрегированных записей
        df_processed['x_boundary_type_original'] = df_processed['x_boundary_type']
        df_processed['x_boundary_type'] = df_processed.apply(
            lambda row: 'exact' if row['x_boundary_type_original'] == 'exact' 
            else ('aggregated_lower_bound' if row['x_boundary_type_original'] == 'lower_bound' else 'none'),
            axis=1
        )
        
        # Удаляем временные колонки
        df_processed.drop(columns=['x_boundary_aggregated', 'x_boundary_aggregated_max'], inplace=True)
    
    # Рассчитываем дескрипторы для каждой строки (расширенная версия с новым классом)
    descriptors_list = []
    for idx, row in df_processed.iterrows():
        desc = calculate_descriptors(row, aggregate_lower_bounds)
        descriptors_list.append(desc)
    
    descriptors_df = pd.DataFrame(descriptors_list)
    
    # Объединяем с исходными данными
    result = pd.concat([df_processed, descriptors_df], axis=1)
    
    # Добавляем колонку system для удобства
    result['system'] = result.apply(
        lambda row: f"{row.get('A_element', 'Ba')}{row['B_element']}O3 + {row.get('x_boundary_value', 0):.2f}{row['D_element']}",
        axis=1
    )
    
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
# ФУНКЦИИ ДЛЯ РАСЧЕТА СТАТИСТИКИ
# ============================================================================
@st.cache_data
def calculate_correlations(df, features, include_lower_bounds=True, aggregate_lower_bounds=False):
    """Расчет корреляций Пирсона и Спирмена с p-value"""
    corr_data = []
    
    if include_lower_bounds:
        df_filtered = df.dropna(subset=features)
    else:
        if aggregate_lower_bounds:
            # В режиме агрегации все точки считаются точными
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

@st.cache_data
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

@st.cache_data
def feature_importance_analysis(df, selected_features=None):
    """Random Forest анализ важности признаков с возможностью выбора признаков"""
    if selected_features is None:
        selected_features = ['dr', 'tolerance_factor', 'size_misfit', 'elastic_misfit',
                            'Δχ', 'Δχ_eff', 'ionic_potential_avg', 'free_volume_fraction',
                            'lattice_strain_energy', 'oxygen_vacancy_conc']
    
    available_features = [f for f in selected_features if f in df.columns]
    
    if len(available_features) < 2:
        return None, None
    
    plot_df = df.dropna(subset=['x_boundary_value'] + available_features)
    
    if len(plot_df) < 10:
        return None, None
    
    X = plot_df[available_features].copy()
    # Добавляем one-hot кодирование для B_element
    if 'B_element' in plot_df.columns:
        X = pd.concat([X, pd.get_dummies(plot_df['B_element'], prefix='B')], axis=1)
    
    y = plot_df['x_boundary_value']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    r2 = rf.score(X, y)
    
    return importance_df, r2

@st.cache_data
def compare_ml_models(df, selected_features=None, target='x_boundary_value'):
    """Сравнение нескольких моделей ML с кросс-валидацией"""
    if selected_features is None:
        selected_features = ['dr', 'tolerance_factor', 'size_misfit', 'elastic_misfit',
                            'Δχ', 'Δχ_eff', 'ionic_potential_avg', 'free_volume_fraction',
                            'lattice_strain_energy', 'oxygen_vacancy_conc']
    
    available_features = [f for f in selected_features if f in df.columns]
    
    if len(available_features) < 2:
        return None, None, None
    
    plot_df = df.dropna(subset=[target] + available_features)
    
    if len(plot_df) < 10:
        return None, None, None
    
    X = plot_df[available_features].copy()
    if 'B_element' in plot_df.columns:
        X = pd.concat([X, pd.get_dummies(plot_df['B_element'], prefix='B')], axis=1)
    
    y = plot_df[target]
    
    # Определяем модели
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        
        model.fit(X, y)
        train_r2 = model.score(X, y)
        
        results.append({
            'Model': name,
            'CV R² (mean)': f'{scores.mean():.3f}',
            'CV R² (std)': f'{scores.std():.3f}',
            'Train R²': f'{train_r2:.3f}',
            'CV MAE': f'{mae_scores.mean():.3f}',
            'CV MAE (std)': f'{mae_scores.std():.3f}'
        })
    
    return pd.DataFrame(results), models, X, y

@st.cache_data
def calculate_shap_values(model, X, feature_names):
    """Расчет SHAP values для модели"""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        return explainer, shap_values
    except Exception as e:
        st.warning(f"SHAP calculation failed: {e}")
        return None, None

@st.cache_data
def perform_clustering(df, features, eps=0.5, min_samples=3):
    """DBSCAN кластеризация для выделения семейств систем"""
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < 2:
        return None, None
    
    plot_df = df.dropna(subset=available_features + ['x_boundary_value'])
    
    if len(plot_df) < 5:
        return None, None
    
    X = plot_df[available_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(X_scaled)
    
    return labels, plot_df

@st.cache_data
def get_dopant_statistics(df, include_lower_bounds=True, aggregate_lower_bounds=False):
    """Получение статистики по допантам с учетом типа значений
    
    В агрегированном режиме используем максимальную нижнюю оценку для каждой пары B-D
    """
    if 'x_boundary_value' not in df.columns or 'D_element' not in df.columns:
        return pd.DataFrame()
    
    if aggregate_lower_bounds:
        # В агрегированном режиме: для каждой пары (B_element, D_element) берем максимальное значение
        grouped = df.groupby(['B_element', 'D_element'])['x_boundary_value'].max().reset_index()
        df_stats = grouped.dropna(subset=['x_boundary_value'])
        
        if len(df_stats) == 0:
            return pd.DataFrame()
        
        stats_list = []
        for dopant in df_stats['D_element'].unique():
            dopant_data = df_stats[df_stats['D_element'] == dopant]['x_boundary_value']
            
            # Подсчет количества уникальных пар B-D для этого допанта
            unique_pairs = df_stats[df_stats['D_element'] == dopant].groupby('B_element').size().count()
            
            stats_list.append({
                'Dopant': dopant,
                'Count': len(dopant_data),
                'Unique B-D pairs': unique_pairs,
                'Mean': dopant_data.mean(),
                'Median': dopant_data.median(),
                'Std': dopant_data.std(),
                'Min': dopant_data.min(),
                'Max': dopant_data.max(),
                'Exact values': len(dopant_data),  # В агрегированном режиме все считаем точными для статистики
                'Lower bounds': 0
            })
        
        return pd.DataFrame(stats_list).sort_values('Median', ascending=False)
    
    else:
        # Стандартный режим
        if include_lower_bounds:
            df_stats = df.dropna(subset=['x_boundary_value'])
        else:
            df_stats = df[df['x_boundary_type'] == 'exact'].dropna(subset=['x_boundary_value'])
        
        if len(df_stats) == 0:
            return pd.DataFrame()
        
        stats_list = []
        for dopant in df_stats['D_element'].unique():
            dopant_data = df_stats[df_stats['D_element'] == dopant]['x_boundary_value']
            
            types_in_dopant = df[df['D_element'] == dopant]['x_boundary_type'].value_counts()
            exact_count = types_in_dopant.get('exact', 0)
            lower_count = types_in_dopant.get('lower_bound', 0)
            
            stats_list.append({
                'Dopant': dopant,
                'Count': len(dopant_data),
                'Mean': dopant_data.mean(),
                'Median': dopant_data.median(),
                'Std': dopant_data.std(),
                'Min': dopant_data.min(),
                'Max': dopant_data.max(),
                'Exact values': exact_count,
                'Lower bounds': lower_count,
                'Includes lower bounds': include_lower_bounds
            })
        
        return pd.DataFrame(stats_list).sort_values('Median', ascending=False)

# ============================================================================
# НОВЫЕ ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ (SHAP, PDP, CLUSTERING)
# ============================================================================
def plot_shap_summary(shap_values, X, feature_names, ax):
    """График SHAP summary"""
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance')
    return ax

def plot_shap_force(explainer, shap_values, X, idx, ax):
    """График SHAP force plot для одного предсказания"""
    shap.force_plot(explainer.expected_value, shap_values[idx,:], X.iloc[idx,:], 
                    feature_names=X.columns.tolist(), matplotlib=True, show=False)
    return ax

def plot_partial_dependence(model, X, feature_name, ax, grid_resolution=50):
    """Partial Dependence Plot для одного признака"""
    if feature_name not in X.columns:
        ax.text(0.5, 0.5, f'{feature_name} not in features', ha='center', va='center')
        return ax
    
    feature_values = np.linspace(X[feature_name].min(), X[feature_name].max(), grid_resolution)
    pdp_values = []
    
    X_temp = X.copy()
    for val in feature_values:
        X_temp[feature_name] = val
        pred = model.predict(X_temp)
        pdp_values.append(pred.mean())
    
    ax.plot(feature_values, pdp_values, 'b-', linewidth=2)
    ax.fill_between(feature_values, np.array(pdp_values) - np.std(pdp_values), 
                     np.array(pdp_values) + np.std(pdp_values), alpha=0.2)
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Partial Dependence')
    ax.set_title(f'PDP: {feature_name}')
    ax.grid(True, alpha=0.3)
    return ax

def plot_ice_curves(model, X, feature_name, ax, n_ice=20, grid_resolution=30):
    """Individual Conditional Expectation (ICE) curves"""
    if feature_name not in X.columns:
        ax.text(0.5, 0.5, f'{feature_name} not in features', ha='center', va='center')
        return ax
    
    feature_values = np.linspace(X[feature_name].min(), X[feature_name].max(), grid_resolution)
    
    # Выбираем случайные наблюдения для ICE
    sample_idx = np.random.choice(len(X), min(n_ice, len(X)), replace=False)
    
    for idx in sample_idx:
        ice_values = []
        X_temp = X.iloc[[idx]].copy()
        for val in feature_values:
            X_temp[feature_name] = val
            pred = model.predict(X_temp)
            ice_values.append(pred[0])
        ax.plot(feature_values, ice_values, 'gray', alpha=0.3, linewidth=0.8)
    
    # Добавляем PDP поверх
    pdp_values = []
    X_temp = X.copy()
    for val in feature_values:
        X_temp[feature_name] = val
        pred = model.predict(X_temp)
        pdp_values.append(pred.mean())
    ax.plot(feature_values, pdp_values, 'b-', linewidth=2, label='PDP')
    
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Predicted x_boundary')
    ax.set_title(f'ICE Curves + PDP: {feature_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax

def plot_contour_t_dr(df, ax, aggregate_mode=False):
    """2D contour plot: x_boundary как функция от t и dr"""
    valid = df.dropna(subset=['tolerance_factor', 'dr', 'x_boundary_value'])
    
    if len(valid) < 10:
        ax.text(0.5, 0.5, 'Insufficient data for contour plot', ha='center', va='center')
        return ax
    
    # Создаем сетку
    ti = np.linspace(valid['tolerance_factor'].min(), valid['tolerance_factor'].max(), 50)
    dri = np.linspace(valid['dr'].min(), valid['dr'].max(), 50)
    T, DR = np.meshgrid(ti, dri)
    
    # Интерполяция
    points = valid[['tolerance_factor', 'dr']].values
    values = valid['x_boundary_value'].values
    Z = griddata(points, values, (T, DR), method='cubic')
    
    # Контурный график
    contour = ax.contourf(T, DR, Z, levels=20, cmap='viridis', alpha=0.8)
    ax.contour(T, DR, Z, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    
    # Добавляем точки данных
    if aggregate_mode:
        ax.scatter(valid['tolerance_factor'], valid['dr'], c='red', s=20, alpha=0.5, edgecolors='black')
    else:
        exact = valid[valid['x_boundary_type'] == 'exact']
        lower = valid[valid['x_boundary_type'] == 'lower_bound']
        ax.scatter(exact['tolerance_factor'], exact['dr'], c='red', s=30, alpha=0.7, edgecolors='black', label='Exact')
        ax.scatter(lower['tolerance_factor'], lower['dr'], c='orange', s=30, alpha=0.3, edgecolors='black', marker='s', label='Lower bound')
    
    ax.axvline(x=1.0, color='white', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Tolerance Factor (t)')
    ax.set_ylabel('Δr (Å)')
    ax.set_title('Solubility Limit as Function of t and Δr')
    if not aggregate_mode:
        ax.legend()
    
    plt.colorbar(contour, ax=ax, label='x(boundary)')
    return ax

def plot_parallel_coordinates(df, ax, features=None, target='x_boundary_value'):
    """Parallel Coordinates Plot для многомерного анализа"""
    if features is None:
        features = ['dr', 'tolerance_factor', 'size_misfit', 'Δχ', 'free_volume_fraction']
    
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < 3:
        ax.text(0.5, 0.5, 'Insufficient data for parallel coordinates', ha='center', va='center')
        return ax
    
    plot_df = df.dropna(subset=available_features + [target])
    
    if len(plot_df) < 10:
        ax.text(0.5, 0.5, 'Insufficient data for parallel coordinates', ha='center', va='center')
        return ax
    
    # Нормализуем данные
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    plot_df_norm = plot_df[available_features].copy()
    plot_df_norm_scaled = scaler.fit_transform(plot_df_norm)
    
    # Создаем параллельные координаты
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Цветовая кодировка по x_boundary
    cmap = plt.cm.viridis
    norm = plt.Normalize(plot_df[target].min(), plot_df[target].max())
    
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        values = plot_df_norm_scaled[i]
        color = cmap(norm(row[target]))
        ax.plot(range(len(available_features)), values, color=color, alpha=0.5, linewidth=1)
    
    ax.set_xticks(range(len(available_features)))
    ax.set_xticklabels(available_features, rotation=45, ha='right')
    ax.set_ylabel('Normalized Value')
    ax.set_title(f'Parallel Coordinates Plot (color = {target})')
    ax.grid(True, alpha=0.3)
    
    # Добавляем colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=target)
    
    return fig

def plot_violin_by_impurity(df, feature, ax, aggregate_mode=False):
    """Violin plot с разделением по наличию примесей"""
    valid = df.dropna(subset=[feature, 'x_boundary_value', 'has_impurity'])
    
    if len(valid) < 10:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return ax
    
    data_no_imp = valid[~valid['has_impurity']]['x_boundary_value']
    data_imp = valid[valid['has_impurity']]['x_boundary_value']
    
    violin_data = [data_no_imp, data_imp]
    labels = ['No impurities', 'With impurities']
    colors = ['#4DAF4A', '#E41A1C']
    
    parts = ax.violinplot(violin_data, positions=[1, 2], showmeans=False, showmedians=True, widths=0.7)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    parts['cmedians'].set_color('red')
    parts['cmedians'].set_linewidth(2)
    
    # Добавляем точки
    for i, data in enumerate(violin_data):
        x_pos = np.random.normal(i + 1, 0.05, len(data))
        ax.scatter(x_pos, data, color='black', s=20, alpha=0.3)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel('x(boundary)')
    ax.set_title(f'Solubility Distribution: {feature}')
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax

def plot_pca_loadings(df, features=None, ax=None):
    """PCA loadings plot (correlation circle)"""
    if features is None:
        features = ['dr', 'tolerance_factor', 'size_misfit', 'elastic_misfit', 
                   'Δχ', 'Δχ_eff', 'ionic_potential_avg', 'free_volume_fraction']
    
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < 3:
        if ax is not None:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return None
    
    plot_df = df[available_features].dropna()
    
    if len(plot_df) < 10:
        if ax is not None:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(plot_df)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()
    
    # Круг корреляций
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', alpha=0.5)
    ax.add_artist(circle)
    
    # Стрелки нагрузок
    for i, feature in enumerate(available_features):
        comp1 = pca.components_[0, i]
        comp2 = pca.components_[1, i]
        ax.arrow(0, 0, comp1, comp2, head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)
        ax.text(comp1 * 1.05, comp2 * 1.05, feature, fontsize=10, ha='center')
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA Loadings Plot (Correlation Circle)')
    ax.set_aspect('equal')
    
    return fig

def plot_3d_interactive(df, x='tolerance_factor', y='dr', z='x_boundary_value', color='size_misfit'):
    """Интерактивный 3D график с plotly"""
    available_x = x if x in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    available_y = y if y in df.columns else df.select_dtypes(include=[np.number]).columns[1]
    available_z = z if z in df.columns else 'x_boundary_value'
    available_color = color if color in df.columns else df.select_dtypes(include=[np.number]).columns[2]
    
    plot_df = df.dropna(subset=[available_x, available_y, available_z, available_color])
    
    if len(plot_df) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for 3D plot", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure(data=[go.Scatter3d(
        x=plot_df[available_x],
        y=plot_df[available_y],
        z=plot_df[available_z],
        mode='markers',
        marker=dict(
            size=plot_df[available_z] * 50 + 5,
            color=plot_df[available_color],
            colorscale='Viridis',
            colorbar=dict(title=available_color),
            showscale=True,
            line=dict(width=0.5, color='black')
        ),
        text=plot_df['system'] if 'system' in plot_df.columns else plot_df.index,
        hovertemplate='<b>%{text}</b><br>' +
                      f'{available_x}: %{{x:.3f}}<br>' +
                      f'{available_y}: %{{y:.3f}}<br>' +
                      f'{available_z}: %{{z:.3f}}<br>' +
                      f'{available_color}: %{{marker.color:.3f}}<extra></extra>'
    )])
    
    fig.update_layout(
        title='3D Stability Phase Diagram',
        scene=dict(
            xaxis_title=available_x,
            yaxis_title=available_y,
            zaxis_title=available_z
        ),
        width=800,
        height=700
    )
    
    return fig

def plot_sankey_diagram(df, max_systems=20):
    """Sankey diagram для потока данных от B-site к допанту к примесям"""
    if 'B_element' not in df.columns or 'D_element' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Required columns missing", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Агрегируем данные
    df_agg = df.groupby(['B_element', 'D_element', 'has_impurity']).size().reset_index(name='count')
    
    # Ограничиваем количество систем для читаемости
    if len(df_agg) > max_systems:
        df_agg = df_agg.nlargest(max_systems, 'count')
    
    # Создаем узлы
    nodes = []
    node_indices = {}
    
    # B-site узлы
    b_sites = df_agg['B_element'].unique()
    for b in b_sites:
        node_indices[f'B_{b}'] = len(nodes)
        nodes.append(f'B: {b}')
    
    # D-site узлы
    d_sites = df_agg['D_element'].unique()
    for d in d_sites:
        node_indices[f'D_{d}'] = len(nodes)
        nodes.append(f'D: {d}')
    
    # Impurity узлы
    node_indices['No impurity'] = len(nodes)
    nodes.append('No impurity')
    node_indices['With impurity'] = len(nodes)
    nodes.append('With impurity')
    
    # Создаем связи
    links = []
    
    for _, row in df_agg.iterrows():
        b_idx = node_indices[f'B_{row["B_element"]}']
        d_idx = node_indices[f'D_{row["D_element"]}']
        
        # Связь B -> D
        links.append({
            'source': b_idx,
            'target': d_idx,
            'value': row['count']
        })
        
        # Связь D -> impurity status
        imp_idx = node_indices['With impurity'] if row['has_impurity'] else node_indices['No impurity']
        links.append({
            'source': d_idx,
            'target': imp_idx,
            'value': row['count']
        })
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=nodes,
            color='lightblue'
        ),
        link=dict(
            source=[l['source'] for l in links],
            target=[l['target'] for l in links],
            value=[l['value'] for l in links]
        )
    )])
    
    fig.update_layout(title='Sankey Diagram: B-site → Dopant → Impurity', height=600)
    return fig

def generate_insights(df):
    """Автоматическая генерация физических инсайтов на основе данных"""
    insights = []
    
    # Корреляционный анализ
    if 'x_boundary_value' in df.columns and 'lattice_strain_energy' in df.columns:
        valid = df[['x_boundary_value', 'lattice_strain_energy']].dropna()
        if len(valid) > 5:
            spearman_r, spearman_p = stats.spearmanr(valid['x_boundary_value'], valid['lattice_strain_energy'])
            if spearman_p < 0.05:
                if spearman_r < -0.5:
                    insights.append(f"**Strong negative correlation** between x_boundary and lattice_strain_energy (Spearman ρ = {spearman_r:.2f}, p < 0.05). Systems with higher strain energy have significantly lower solubility.")
                elif spearman_r > 0.5:
                    insights.append(f"**Strong positive correlation** between x_boundary and lattice_strain_energy (Spearman ρ = {spearman_r:.2f}, p < 0.05). Counterintuitive: strain may promote solubility in this system.")
    
    # Анализ tolerance factor
    if 'tolerance_factor' in df.columns and 'x_boundary_value' in df.columns:
        valid = df.dropna(subset=['tolerance_factor', 'x_boundary_value'])
        if len(valid) > 10:
            t_opt_range = valid[(valid['tolerance_factor'] >= 0.96) & (valid['tolerance_factor'] <= 1.04)]
            t_outside = valid[(valid['tolerance_factor'] < 0.96) | (valid['tolerance_factor'] > 1.04)]
            
            if len(t_opt_range) > 0 and len(t_outside) > 0:
                mean_opt = t_opt_range['x_boundary_value'].mean()
                mean_out = t_outside['x_boundary_value'].mean()
                if mean_opt > mean_out:
                    ratio = mean_opt / mean_out if mean_out > 0 else float('inf')
                    insights.append(f"Systems with tolerance factor in the optimal range [0.96-1.04] have **{ratio:.1f}x higher** average solubility (t_opt: {mean_opt:.3f} vs t_out: {mean_out:.3f}).")
    
    # Анализ Δr порога
    if 'dr' in df.columns and 'has_impurity' in df.columns:
        valid = df.dropna(subset=['dr', 'has_impurity'])
        if len(valid) > 10:
            impure = valid[valid['has_impurity']]['dr'].dropna()
            if len(impure) > 3:
                dr_threshold = impure.quantile(0.25)
                insights.append(f"Impurity phases typically appear when Δr > {dr_threshold:.3f} Å (lower quartile of impurity-containing systems).")
    
    # Анализ свободного объема
    if 'free_volume_fraction' in df.columns and 'x_boundary_value' in df.columns:
        valid = df.dropna(subset=['free_volume_fraction', 'x_boundary_value'])
        if len(valid) > 10:
            high_vol = valid[valid['free_volume_fraction'] > valid['free_volume_fraction'].median()]
            low_vol = valid[valid['free_volume_fraction'] <= valid['free_volume_fraction'].median()]
            if len(high_vol) > 0 and len(low_vol) > 0:
                mean_high = high_vol['x_boundary_value'].mean()
                mean_low = low_vol['x_boundary_value'].mean()
                if mean_high > mean_low:
                    ratio = mean_high / mean_low if mean_low > 0 else float('inf')
                    insights.append(f"Higher free volume fraction correlates with **{ratio:.1f}x higher solubility** (above median: {mean_high:.3f} vs below: {mean_low:.3f}).")
    
    # Анализ электроотрицательности
    if 'Δχ_eff' in df.columns and 'x_boundary_value' in df.columns:
        valid = df.dropna(subset=['Δχ_eff', 'x_boundary_value'])
        if len(valid) > 5:
            spearman_r, spearman_p = stats.spearmanr(valid['Δχ_eff'], valid['x_boundary_value'])
            if spearman_p < 0.05 and abs(spearman_r) > 0.4:
                direction = "positive" if spearman_r > 0 else "negative"
                insights.append(f"Effective electronegativity difference Δχ_eff shows **{direction} correlation** with solubility (Spearman ρ = {spearman_r:.2f}, p < 0.05).")
    
    # Общий вывод
    if len(insights) == 0:
        insights.append("Insufficient data for automated insights. Add more data points to enable pattern detection.")
    
    return insights

# ============================================================================
# СУЩЕСТВУЮЩИЕ ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ (СОХРАНЕНЫ БЕЗ ИЗМЕНЕНИЙ)
# ============================================================================
def plot_solubility_vs_dr(df, ax, aggregate_mode=False):
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
        
        if aggregate_mode:
            # В режиме агрегации все точки показываем единым стилем
            combinations[combo_key]['exact_values'].append((row['dr'], row['x_boundary_value'], row['x_boundary_type']))
        else:
            if row['x_boundary_type'] == 'exact':
                combinations[combo_key]['exact_values'].append((row['dr'], row['x_boundary_value'], row['x_boundary_type']))
            else:
                combinations[combo_key]['lower_bounds'].append((row['dr'], row['x_boundary_value'], row['x_boundary_type']))
    
    for combo_key, data in combinations.items():
        b_element = data['b_element']
        color = B_COLORS.get(b_element, B_COLORS['default'])
        marker = D_MARKERS.get(data['d_element'], D_MARKERS['default'])
        
        if data['exact_values']:
            dr_exact, x_exact, types = zip(*data['exact_values'])
            
            if aggregate_mode:
                # Единый стиль для всех точек, но с пометкой lower_bound
                scatter = ax.scatter(
                    dr_exact, x_exact,
                    color=color, marker=marker, s=80,
                    alpha=1.0, edgecolors='black', linewidth=0.5,
                    label=combo_key
                )
                # Добавляем пометку для lower_bound точек
                for i, (dr_val, x_val, t) in enumerate(zip(dr_exact, x_exact, types)):
                    if t == 'lower_bound':
                        ax.annotate('≥', (dr_val, x_val), textcoords="offset points", 
                                   xytext=(5, 5), ha='center', fontsize=8, fontweight='bold')
            else:
                ax.scatter(
                    dr_exact, x_exact,
                    color=color, marker=marker, s=80,
                    alpha=1.0, edgecolors='black', linewidth=0.5,
                    label=combo_key
                )
        
        if not aggregate_mode and data['lower_bounds']:
            dr_lower, x_lower, _ = zip(*data['lower_bounds'])
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
    if aggregate_mode:
        ax.set_title('Solubility Limit vs Radius Difference\n(≥ indicates lower bound estimate)')
    else:
        ax.set_title('Solubility Limit vs Radius Difference\n(Hollow markers = lower bound estimates)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_tolerance_factor(df, ax, aggregate_mode=False):
    """График 2: x(boundary) vs tolerance factor"""
    for b_element in df['B_element'].unique():
        mask = df['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        if aggregate_mode:
            # В режиме агрегации все точки показываем единым стилем
            data_mask = mask & (df['x_boundary_value'].notna())
            if data_mask.any():
                ax.scatter(
                    df.loc[data_mask, 'tolerance_factor'],
                    df.loc[data_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element}"
                )
                # Добавляем пометку для lower_bound точек
                lower_mask = data_mask & (df['x_boundary_type'] == 'lower_bound')
                for idx in df[lower_mask].index:
                    ax.annotate('≥', 
                               (df.loc[idx, 'tolerance_factor'], df.loc[idx, 'x_boundary_value']),
                               textcoords="offset points", xytext=(5, 5), 
                               ha='center', fontsize=8, fontweight='bold')
        else:
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
    if aggregate_mode:
        ax.set_title('Solubility Limit vs Tolerance Factor\n(≥ indicates lower bound estimate)')
    else:
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

def plot_xmax_vs_xboundary(df, ax, aggregate_mode=False):
    """График 4: x(max) vs x(boundary)"""
    valid = df.dropna(subset=['x_max', 'x_boundary_value'])
    
    if aggregate_mode:
        # В режиме агрегации все точки показываем единым стилем с цветовой кодировкой по Δr
        if len(valid) > 0:
            scatter = ax.scatter(
                valid['x_boundary_value'], valid['x_max'],
                c=valid['dr'], cmap='coolwarm', s=100,
                alpha=0.9, edgecolors='black', linewidth=0.5,
                label='All values'
            )
            # Добавляем пометку для lower_bound точек
            lower_valid = valid[valid['x_boundary_type'] == 'lower_bound']
            for idx in lower_valid.index:
                ax.annotate('≥', 
                           (valid.loc[idx, 'x_boundary_value'], valid.loc[idx, 'x_max']),
                           textcoords="offset points", xytext=(5, 5), 
                           ha='center', fontsize=8, fontweight='bold')
    else:
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
    
    if aggregate_mode and len(valid) > 0:
        plt.colorbar(scatter, ax=ax, label='Δr (Å)')
    elif not aggregate_mode:
        if 'scatter_exact' in locals() and len(exact_valid) > 0:
            plt.colorbar(scatter_exact, ax=ax, label='Δr (Å)')
        elif 'scatter_lower' in locals() and len(lower_valid) > 0:
            plt.colorbar(scatter_lower, ax=ax, label='Δr (Å)')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_xmax_vs_tolerance(df, ax, aggregate_mode=False):
    """График 5: x(max) vs tolerance factor"""
    valid = df.dropna(subset=['x_max', 'tolerance_factor'])
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        if aggregate_mode:
            data_mask = mask & (valid['x_max'].notna())
            if data_mask.any():
                ax.scatter(
                    valid.loc[data_mask, 'tolerance_factor'],
                    valid.loc[data_mask, 'x_max'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element}"
                )
                lower_mask = data_mask & (valid['x_boundary_type'] == 'lower_bound')
                for idx in valid[lower_mask].index:
                    ax.annotate('≥', 
                               (valid.loc[idx, 'tolerance_factor'], valid.loc[idx, 'x_max']),
                               textcoords="offset points", xytext=(5, 5), 
                               ha='center', fontsize=8, fontweight='bold')
        else:
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

def plot_relative_position(df, ax, aggregate_mode=False):
    """График 6: x(max)/x(boundary) vs Δr/r_B"""
    valid = df.dropna(subset=['x_rel_max', 'dr_rel'])
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        if aggregate_mode:
            data_mask = mask & (valid['x_rel_max'].notna())
            if data_mask.any():
                ax.scatter(
                    valid.loc[data_mask, 'dr_rel'],
                    valid.loc[data_mask, 'x_rel_max'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element}"
                )
                lower_mask = data_mask & (valid['x_boundary_type'] == 'lower_bound')
                for idx in valid[lower_mask].index:
                    ax.annotate('≥', 
                               (valid.loc[idx, 'dr_rel'], valid.loc[idx, 'x_rel_max']),
                               textcoords="offset points", xytext=(5, 5), 
                               ha='center', fontsize=8, fontweight='bold')
        else:
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

def plot_b_site_statistics(df, include_lower_bounds=True, aggregate_mode=False):
    """График 10: Статистика по B-элементам (столбчатая диаграмма с ошибками)
    
    В агрегированном режиме:
    - Используем максимальную нижнюю оценку для каждой пары B-D
    - Показываем диапазон значений как ошибку (min-max)
    """
    if aggregate_mode:
        # В режиме агрегации используем все точки, но с дедупликацией по парам B-D
        # Группируем по B_element и D_element, берем максимальное значение
        grouped = df.groupby(['B_element', 'D_element'])['x_boundary_value'].max().reset_index()
        
        # Затем группируем по B_element для статистики
        df_stats_raw = grouped.groupby('B_element')['x_boundary_value'].agg(
            mean='mean',
            median='median',
            count='count',
            std='std',
            min='min',
            max='max'
        ).round(3)
        
        # Переименовываем колонки для совместимости
        df_stats_raw = df_stats_raw.rename(columns={
            'mean': 'mean',
            'median': 'median',
            'count': 'count',
            'std': 'std'
        })
        
        # Также считаем количество точных и агрегированных записей
        type_counts = df.groupby('B_element')['x_boundary_type'].value_counts().unstack(fill_value=0)
        if 'exact' in type_counts.columns:
            df_stats_raw['exact_count'] = type_counts['exact']
        else:
            df_stats_raw['exact_count'] = 0
        if 'aggregated_lower_bound' in type_counts.columns:
            df_stats_raw['lower_bound_count'] = type_counts['aggregated_lower_bound']
        else:
            df_stats_raw['lower_bound_count'] = 0
        
        stats_df = df_stats_raw
        
        # Для отображения ошибки используем не стандартное отклонение, а диапазон
        # Для этого создадим колонки min и max для каждого B-элемента
        min_max_df = df.groupby('B_element')['x_boundary_value'].agg(['min', 'max']).round(3)
        stats_df['min'] = min_max_df['min']
        stats_df['max'] = min_max_df['max']
        
        # Ошибка как расстояние от среднего до максимума (для верхней планки)
        # и от минимума до среднего (для нижней планки)
        stats_df['error_upper'] = stats_df['max'] - stats_df['mean']
        stats_df['error_lower'] = stats_df['mean'] - stats_df['min']
        
    else:
        # Стандартный режим
        if include_lower_bounds:
            df_stats = df.dropna(subset=['x_boundary_value'])
        else:
            df_stats = df[df['x_boundary_type'] == 'exact'].dropna(subset=['x_boundary_value'])
        
        stats_df = df_stats.groupby('B_element')['x_boundary_value'].agg(
            mean='mean', 
            median='median', 
            count='count', 
            std='std'
        ).round(3)
        
        # Подсчет типов значений
        type_counts = df.groupby('B_element')['x_boundary_type'].value_counts().unstack(fill_value=0)
        if 'exact' in type_counts.columns:
            stats_df['exact_count'] = type_counts['exact']
        if 'lower_bound' in type_counts.columns:
            stats_df['lower_bound_count'] = type_counts['lower_bound']
        
        # Для стандартного режима ошибка = std
        stats_df['error_upper'] = stats_df['std']
        stats_df['error_lower'] = stats_df['std']
        stats_df['min'] = stats_df['mean'] - stats_df['std']
        stats_df['max'] = stats_df['mean'] + stats_df['std']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    b_sites = stats_df.index
    x_pos = np.arange(len(b_sites))
    
    # График средних значений с асимметричными ошибками
    if aggregate_mode:
        # Асимметричные ошибки (нижняя и верхняя)
        error_lower = stats_df['error_lower'].values
        error_upper = stats_df['error_upper'].values
        error_bars = [error_lower, error_upper]
        
        bars = ax1.bar(x_pos, stats_df['mean'], 
                       color=[B_COLORS.get(b, B_COLORS['default']) for b in b_sites],
                       edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # Добавляем асимметричные error bars
        ax1.errorbar(x_pos, stats_df['mean'], 
                     yerr=[error_lower, error_upper],
                     fmt='none', ecolor='black', capsize=5, capthick=1.5, elinewidth=1.5)
        
        # Добавляем точки для минимальных и максимальных значений
        ax1.scatter(x_pos, stats_df['min'], color='red', s=50, 
                   marker='v', zorder=5, label='Minimum value', alpha=0.7)
        ax1.scatter(x_pos, stats_df['max'], color='green', s=50, 
                   marker='^', zorder=5, label='Maximum value', alpha=0.7)
        
        # Добавляем аннотации с диапазоном
        for i, (b, row) in enumerate(stats_df.iterrows()):
            ax1.annotate(f'[{row["min"]:.2f}-{row["max"]:.2f}]', 
                        (x_pos[i], row['max'] + 0.02),
                        ha='center', fontsize=8, rotation=0,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
        
    else:
        ax1.bar(x_pos, stats_df['mean'], yerr=stats_df['std'],
                capsize=5, color=[B_COLORS.get(b, B_COLORS['default']) for b in b_sites],
                edgecolor='black', linewidth=0.5, alpha=0.8)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(b_sites)
    ax1.set_ylabel('Mean x(boundary)')
    
    if aggregate_mode:
        title = f'Average Solubility by B-site (Aggregated mode)\n'
        title += f'Bars show mean, error bars show range [min-max]'
        ax1.set_title(title)
    else:
        title = f'Average Solubility by B-site\n({len(df_stats)} samples)'
        ax1.set_title(title)
    ax1.grid(True, alpha=0.3, axis='y')
    if aggregate_mode:
        ax1.legend(loc='upper right')
    
    # График количества образцов
    bottom = np.zeros(len(b_sites))
    if 'exact_count' in stats_df.columns:
        ax2.bar(x_pos, stats_df['exact_count'], bottom=bottom,
                label='Exact', color='darkblue', edgecolor='black', linewidth=0.5)
        bottom += stats_df['exact_count']
    if aggregate_mode and 'lower_bound_count' in stats_df.columns:
        ax2.bar(x_pos, stats_df['lower_bound_count'], bottom=bottom,
                label='Aggregated (≥)', color='lightblue', edgecolor='black', linewidth=0.5, alpha=0.7)
    elif not aggregate_mode and 'lower_bound_count' in stats_df.columns:
        ax2.bar(x_pos, stats_df['lower_bound_count'], bottom=bottom,
                label='Lower bound', color='lightblue', edgecolor='black', linewidth=0.5, alpha=0.7)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(b_sites)
    ax2.set_ylabel('Number of samples')
    
    if aggregate_mode:
        ax2.set_title('Sample Count by B-site (after aggregation)')
    else:
        ax2.set_title('Sample Count by B-site')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, stats_df

def plot_top_dopants_violin(df, include_lower_bounds=True, aggregate_mode=False):
    """График 11: Violin plot для топ-10 допантов по растворимости, отсортированных по ионному радиусу"""
    dopant_stats = get_dopant_statistics(df, include_lower_bounds, aggregate_mode)
    
    if len(dopant_stats) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig
    
    top_dopants = dopant_stats.head(10)['Dopant'].tolist()
    
    dopant_radii = {}
    for dopant in top_dopants:
        radius = IONIC_RADII.get((dopant, 3, 6), None)
        if radius is None:
            for charge in [2, 4]:
                radius = IONIC_RADII.get((dopant, charge, 6), None)
                if radius:
                    break
        dopant_radii[dopant] = radius if radius else 0
    
    sorted_dopants = sorted(top_dopants, key=lambda d: dopant_radii.get(d, 0))
    
    expanded_data = []
    
    for _, row in df.iterrows():
        if row['D_element'] not in sorted_dopants:
            continue
            
        if pd.isna(row.get('x_boundary_value')):
            continue
        
        # В режиме агрегации все точки считаем точными для распределения
        if aggregate_mode or row['x_boundary_type'] == 'exact':
            x_inv_in = row.get('x_inv_in', 0)
            if pd.isna(x_inv_in):
                x_inv_in = 0
            
            x_boundary = row['x_boundary_value']
            range_width = x_boundary - x_inv_in
            n_points = max(3, min(10, int(range_width * 20)))
            
            for x in np.linspace(x_inv_in, x_boundary, n_points):
                expanded_data.append({
                    'D_element': row['D_element'],
                    'x_value': x,
                    'type': 'exact',
                    'original_max': x_boundary
                })
        elif row['x_boundary_type'] == 'lower_bound' and include_lower_bounds and not aggregate_mode:
            x_inv_in = row.get('x_inv_in', 0)
            x_inv_end = row.get('x_inv_end', row['x_boundary_value'])
            
            if pd.isna(x_inv_in) or x_inv_in == '-':
                x_inv_in = 0
            if pd.isna(x_inv_end) or x_inv_end == '-':
                x_inv_end = row['x_boundary_value']
            
            try:
                x_inv_in = float(x_inv_in)
                x_inv_end = float(x_inv_end)
            except (ValueError, TypeError):
                x_inv_in = 0
                x_inv_end = row['x_boundary_value']
            
            range_width = x_inv_end - x_inv_in
            
            if range_width < 0.05:
                n_points = 2
            elif range_width < 0.1:
                n_points = 3
            elif range_width < 0.2:
                n_points = 5
            else:
                n_points = min(10, int(range_width * 15))
            
            for x in np.linspace(x_inv_in, x_inv_end, n_points):
                expanded_data.append({
                    'D_element': row['D_element'],
                    'x_value': x,
                    'type': 'lower_bound',
                    'weight': 1.0 / n_points
                })
    
    expanded_df = pd.DataFrame(expanded_data)
    
    if len(expanded_df) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data after expansion', ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    plot_data = []
    positions = []
    violin_labels = []
    radius_values = []
    
    for i, d in enumerate(sorted_dopants):
        d_data = expanded_df[expanded_df['D_element'] == d]['x_value'].dropna()
        if len(d_data) > 0:
            plot_data.append(d_data)
            positions.append(i + 1)
            violin_labels.append(f"{d}")
            radius_values.append(dopant_radii.get(d, 0))
    
    if plot_data:
        parts = ax.violinplot(plot_data, positions=positions, showmeans=False, showmedians=True, widths=0.7)
        
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
        
        for i, d in enumerate(sorted_dopants):
            if aggregate_mode:
                # В режиме агрегации показываем все точки единым стилем
                all_data = df[(df['D_element'] == d) & (df['x_boundary_value'].notna())]['x_boundary_value'].dropna()
                if len(all_data) > 0:
                    x_pos = np.random.normal(positions[i], 0.05, len(all_data))
                    ax.scatter(x_pos, all_data, color='darkred', s=80,
                              alpha=0.9, zorder=3, edgecolors='black', linewidth=1,
                              label='All values' if i == 0 else "")
                    # Добавляем пометку для lower_bound точек
                    lower_data = df[(df['D_element'] == d) & 
                                   (df['x_boundary_type'] == 'lower_bound') &
                                   (df['x_boundary_value'].notna())]['x_boundary_value'].dropna()
                    for j, (x_val, x_pos_j) in enumerate(zip(lower_data, x_pos[:len(lower_data)])):
                        ax.annotate('≥', (x_pos_j, x_val), textcoords="offset points", 
                                   xytext=(5, 5), ha='center', fontsize=8, fontweight='bold')
            else:
                exact_data = df[(df['D_element'] == d) &
                               (df['x_boundary_type'] == 'exact')]['x_boundary_value'].dropna()
                if len(exact_data) > 0:
                    x_pos = np.random.normal(positions[i], 0.05, len(exact_data))
                    ax.scatter(x_pos, exact_data, color='darkred', s=80,
                              alpha=0.9, zorder=3, edgecolors='black', linewidth=1,
                              label='Exact values' if i == 0 else "")
                
                lower_data = df[(df['D_element'] == d) &
                               (df['x_boundary_type'] == 'lower_bound')]['x_boundary_value'].dropna()
                if len(lower_data) > 0:
                    x_pos = np.random.normal(positions[i], 0.08, len(lower_data))
                    ax.scatter(x_pos, lower_data, color='darkorange', s=80,
                              alpha=0.7, zorder=3, marker='D', edgecolors='black', linewidth=1,
                              label='Lower bounds (≥)' if i == 0 else "")
        
        y_top = ax.get_ylim()[1]
        
        for i, d in enumerate(sorted_dopants):
            stats_row = dopant_stats[dopant_stats['Dopant'] == d].iloc[0]
            exact_count = int(stats_row['Exact values'])
            lower_count = int(stats_row['Lower bounds'])
            
            if aggregate_mode:
                text = f'n={exact_count + lower_count}'
            else:
                text = f'n={exact_count}'
                if lower_count > 0:
                    text += f' (+{lower_count}≥)'
            
            ax.text(positions[i], y_top * 0.98, text,
                    ha='center', fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))
        
        for i, d in enumerate(sorted_dopants):
            all_values = df[df['D_element'] == d]['x_boundary_value'].dropna()
            if len(all_values) > 0:
                min_val = all_values.min()
                max_val = all_values.max()
                
                if aggregate_mode:
                    range_text = f'[{min_val:.2f}-{max_val:.2f}]'
                else:
                    stats_row = dopant_stats[dopant_stats['Dopant'] == d].iloc[0]
                    lower_count = int(stats_row['Lower bounds'])
                    
                    if lower_count == 0:
                        range_text = f'[{min_val:.2f}-{max_val:.2f}]'
                    else:
                        range_text = f'{min_val:.2f} to ≥{max_val:.2f}'
                
                if i % 2 == 0:
                    y_pos = y_top * 0.92
                else:
                    y_pos = y_top * 0.89
                
                ax.text(positions[i], y_pos, range_text,
                        ha='center', fontsize=8, va='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7, edgecolor='gray'))
        
        ax.set_xlabel('Dopant Element', fontsize=14, fontweight='bold')
        ax.set_ylabel('Solid solution range', fontsize=14, fontweight='bold')
        
        if aggregate_mode:
            title = f'Top 10 Dopants by Solubility - Sorted by Ionic Radius\n'
            title += f'(Aggregated mode: lower bounds included as exact values)'
        else:
            title = f'Top 10 Dopants by Solubility - Sorted by Ionic Radius\n'
            title += f'({"Including" if include_lower_bounds else "Excluding"} lower bounds)'
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

def plot_correlation_heatmap(df, include_lower_bounds=True, aggregate_mode=False):
    """График 13: Тепловая карта корреляций"""
    features = ['dr', 'dr_rel', 'tolerance_factor', 'x_boundary_value', 'x_max']
    
    if aggregate_mode:
        corr_df = df[features].dropna()
    else:
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
    
    if aggregate_mode:
        ax1.set_title(f'Pearson Correlations\n(n={len(corr_df)}, aggregated mode)')
    else:
        ax1.set_title(f'Pearson Correlations\n(n={len(corr_df)})')
    
    sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, ax=ax2,
                cbar_kws={'label': 'Spearman ρ'})
    
    if aggregate_mode:
        ax2.set_title(f'Spearman Correlations\n(n={len(corr_df)}, aggregated mode)')
    else:
        ax2.set_title(f'Spearman Correlations\n(n={len(corr_df)})')
    
    mode_text = "aggregated" if aggregate_mode else "standard"
    plt.suptitle(f'Correlation Matrix ({mode_text} mode, {"including" if include_lower_bounds else "excluding"} lower bounds)')
    plt.tight_layout()
    return fig

def plot_distribution_kde(df, aggregate_mode=False):
    """График 15: Распределение x_boundary (гистограмма + KDE по B)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if aggregate_mode:
        # В режиме агрегации все точки считаем точными
        all_data = df['x_boundary_value'].dropna()
        if len(all_data) > 0:
            ax1.hist(all_data, bins=20, edgecolor='black',
                     alpha=0.7, color='blue', density=True, label='All values')
            sns.kdeplot(data=all_data, ax=ax1, color='darkblue',
                        linewidth=2, label='KDE')
        ax1.set_title('Overall Distribution of Solubility Limits (Aggregated)')
    else:
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
        ax1.set_title('Overall Distribution of Solubility Limits')
    
    ax1.set_xlabel('x(boundary)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for b_element in df['B_element'].unique():
        color = B_COLORS.get(b_element, 'gray')
        
        if aggregate_mode:
            data_b = df[(df['B_element'] == b_element) & (df['x_boundary_value'].notna())]['x_boundary_value'].dropna()
            if len(data_b) > 1:
                sns.kdeplot(data=data_b, label=f"{b_element}", ax=ax2,
                            linewidth=2, color=color, linestyle='-')
        else:
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
    if aggregate_mode:
        ax2.set_title('Distribution by B-site (aggregated mode)')
    else:
        ax2.set_title('Distribution by B-site (dashed = lower bounds)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_shift_vs_dr_bubble(df, aggregate_mode=False):
    """График 16: Пузырьковая диаграмма зависимости смещения от Δr"""
    valid = df.dropna(subset=['x_max', 'x_boundary_value', 'dr', 'tolerance_factor'])
    
    if len(valid) < 3:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if aggregate_mode:
        scatter = ax.scatter(
            valid['dr'],
            valid['x_max'] - valid['x_boundary_value'],
            s=valid['x_boundary_value'] * 1000,
            c=valid['tolerance_factor'],
            alpha=0.7,
            cmap='coolwarm',
            edgecolors='black',
            linewidth=0.5,
            label='All values'
        )
        # Добавляем пометку для lower_bound точек
        lower_valid = valid[valid['x_boundary_type'] == 'lower_bound']
        for idx in lower_valid.index:
            ax.annotate('≥', 
                       (valid.loc[idx, 'dr'], valid.loc[idx, 'x_max'] - valid.loc[idx, 'x_boundary_value']),
                       textcoords="offset points", xytext=(5, 5), 
                       ha='center', fontsize=8, fontweight='bold')
    else:
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
        marker_style = ' (≥)' if not aggregate_mode and row['x_boundary_type'] == 'lower_bound' else ''
        ax.annotate(f"{row['B_element']}-{row['D_element']}{marker_style}",
                    (row['dr'], row['x_max'] - row['x_boundary_value']),
                    fontsize=8, ha='center')
    
    ax.set_xlabel('Δr (Å)')
    ax.set_ylabel('x_max - x_boundary')
    ax.set_title('Shift of Conductivity Maximum from Solubility Limit\n(Bubble size = x_boundary)')
    ax.legend()
    
    if aggregate_mode and 'scatter' in locals():
        plt.colorbar(scatter, ax=ax, label='Tolerance Factor')
    elif not aggregate_mode:
        if 'scatter_exact' in locals():
            plt.colorbar(scatter_exact, ax=ax, label='Tolerance Factor')
        elif 'scatter_lower' in locals():
            plt.colorbar(scatter_lower, ax=ax, label='Tolerance Factor')
    
    ax.grid(True, alpha=0.3)
    
    return fig

# ============================================================================
# СУЩЕСТВУЮЩИЕ ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ (ПРОДОЛЖЕНИЕ)
# ============================================================================
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

def plot_comprehensive_correlation_matrix(df, include_lower_bounds=True):
    """График: Полная корреляционная матрица со всеми дескрипторами"""
    features = ['dr', 'dr_rel', 'tolerance_factor', 't_gradient', 't_range',
                'Δχ', 'Δχ_gradient', 'free_volume_fraction', 'packing_factor',
                'E_form', 'lattice_strain_energy', 'oxygen_vacancy_conc',
                'size_misfit', 'elastic_misfit', 'Δχ_eff', 'ionic_potential_avg',
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
        'size_misfit': 'size_misfit',
        'elastic_misfit': 'elastic_misfit',
        'Δχ_eff': 'Δχ_eff',
        'ionic_potential_avg': 'z/r_avg',
        'x_boundary_value': 'x_boundary',
        'x_max': 'x_max'
    }
    
    corr_df_renamed = corr_df.rename(columns=rename_map)
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
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

def plot_free_volume_vs_xboundary(df, ax, aggregate_mode=False):
    """График 30: Свободный объем vs x_boundary"""
    valid = df.dropna(subset=['free_volume_fraction', 'x_boundary_value'])
    
    if len(valid) < 3:
        ax.text(0.5, 0.5, 'Insufficient data for free volume analysis',
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        if aggregate_mode:
            data_mask = mask & (valid['x_boundary_value'].notna())
            if data_mask.any():
                ax.scatter(
                    valid.loc[data_mask, 'free_volume_fraction'],
                    valid.loc[data_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element}"
                )
                lower_mask = data_mask & (valid['x_boundary_type'] == 'lower_bound')
                for idx in valid[lower_mask].index:
                    ax.annotate('≥', 
                               (valid.loc[idx, 'free_volume_fraction'], valid.loc[idx, 'x_boundary_value']),
                               textcoords="offset points", xytext=(5, 5), 
                               ha='center', fontsize=8, fontweight='bold')
        else:
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

def plot_formation_energy_vs_xboundary(df, ax, aggregate_mode=False):
    """График 31: Энергия образования vs x_boundary"""
    valid = df.dropna(subset=['E_form', 'x_boundary_value'])
    
    if len(valid) < 3:
        ax.text(0.5, 0.5, 'Insufficient data for formation energy analysis',
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        if aggregate_mode:
            data_mask = mask & (valid['x_boundary_value'].notna())
            if data_mask.any():
                ax.scatter(
                    valid.loc[data_mask, 'E_form'],
                    valid.loc[data_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element}"
                )
                lower_mask = data_mask & (valid['x_boundary_type'] == 'lower_bound')
                for idx in valid[lower_mask].index:
                    ax.annotate('≥', 
                               (valid.loc[idx, 'E_form'], valid.loc[idx, 'x_boundary_value']),
                               textcoords="offset points", xytext=(5, 5), 
                               ha='center', fontsize=8, fontweight='bold')
        else:
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

# ============================================================================
# НОВЫЕ ФУНКЦИИ ДЛЯ SOLUBILITY ANALYSIS (2a-2f)
# ============================================================================
def plot_solubility_vs_delta_chi(df, ax, aggregate_mode=False):
    """График 2a: x(boundary) vs Δχ (разница электроотрицательностей)"""
    valid = df.dropna(subset=['Δχ', 'x_boundary_value'])
    
    if len(valid) < 3:
        ax.text(0.5, 0.5, 'Insufficient data for Δχ analysis',
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        if aggregate_mode:
            data_mask = mask & (valid['x_boundary_value'].notna())
            if data_mask.any():
                ax.scatter(
                    valid.loc[data_mask, 'Δχ'],
                    valid.loc[data_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element}"
                )
                lower_mask = data_mask & (valid['x_boundary_type'] == 'lower_bound')
                for idx in valid[lower_mask].index:
                    ax.annotate('≥', 
                               (valid.loc[idx, 'Δχ'], valid.loc[idx, 'x_boundary_value']),
                               textcoords="offset points", xytext=(5, 5), 
                               ha='center', fontsize=8, fontweight='bold')
        else:
            exact_mask = mask & (valid['x_boundary_type'] == 'exact')
            lower_mask = mask & (valid['x_boundary_type'] == 'lower_bound')
            
            if exact_mask.any():
                ax.scatter(
                    valid.loc[exact_mask, 'Δχ'],
                    valid.loc[exact_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element} (exact)"
                )
            
            if lower_mask.any():
                ax.scatter(
                    valid.loc[lower_mask, 'Δχ'],
                    valid.loc[lower_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.3,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element} (≥)",
                    marker='s'
                )
    
    ax.set_xlabel('Electronegativity Difference Δχ = |χ_avg_B - χ_A|')
    ax.set_ylabel('x(boundary)')
    if aggregate_mode:
        ax.set_title('Solubility Limit vs Electronegativity Difference\n(≥ indicates lower bound estimate)')
    else:
        ax.set_title('Solubility Limit vs Electronegativity Difference\n(Hollow markers = lower bound estimates)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_solubility_vs_avg_chi(df, ax, aggregate_mode=False):
    """График 2b: x(boundary) vs χ_avg_B (средняя электроотрицательность B-сайта)"""
    valid = df.dropna(subset=['χ_avg_B', 'x_boundary_value'])
    
    if len(valid) < 3:
        ax.text(0.5, 0.5, 'Insufficient data for χ_avg_B analysis',
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        if aggregate_mode:
            data_mask = mask & (valid['x_boundary_value'].notna())
            if data_mask.any():
                ax.scatter(
                    valid.loc[data_mask, 'χ_avg_B'],
                    valid.loc[data_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element}"
                )
                lower_mask = data_mask & (valid['x_boundary_type'] == 'lower_bound')
                for idx in valid[lower_mask].index:
                    ax.annotate('≥', 
                               (valid.loc[idx, 'χ_avg_B'], valid.loc[idx, 'x_boundary_value']),
                               textcoords="offset points", xytext=(5, 5), 
                               ha='center', fontsize=8, fontweight='bold')
        else:
            exact_mask = mask & (valid['x_boundary_type'] == 'exact')
            lower_mask = mask & (valid['x_boundary_type'] == 'lower_bound')
            
            if exact_mask.any():
                ax.scatter(
                    valid.loc[exact_mask, 'χ_avg_B'],
                    valid.loc[exact_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element} (exact)"
                )
            
            if lower_mask.any():
                ax.scatter(
                    valid.loc[lower_mask, 'χ_avg_B'],
                    valid.loc[lower_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.3,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element} (≥)",
                    marker='s'
                )
    
    ax.set_xlabel('Average B-site Electronegativity χ_avg_B = (1-x)·χ_B + x·χ_D')
    ax.set_ylabel('x(boundary)')
    if aggregate_mode:
        ax.set_title('Solubility Limit vs Average B-site Electronegativity\n(≥ indicates lower bound estimate)')
    else:
        ax.set_title('Solubility Limit vs Average B-site Electronegativity\n(Hollow markers = lower bound estimates)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_solubility_vs_avg_radius(df, ax, aggregate_mode=False):
    """График 2c: x(boundary) vs r_avg_B (средний ионный радиус B-сайта)"""
    valid = df.dropna(subset=['r_avg_B', 'x_boundary_value'])
    
    if len(valid) < 3:
        ax.text(0.5, 0.5, 'Insufficient data for r_avg_B analysis',
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        if aggregate_mode:
            data_mask = mask & (valid['x_boundary_value'].notna())
            if data_mask.any():
                ax.scatter(
                    valid.loc[data_mask, 'r_avg_B'],
                    valid.loc[data_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element}"
                )
                lower_mask = data_mask & (valid['x_boundary_type'] == 'lower_bound')
                for idx in valid[lower_mask].index:
                    ax.annotate('≥', 
                               (valid.loc[idx, 'r_avg_B'], valid.loc[idx, 'x_boundary_value']),
                               textcoords="offset points", xytext=(5, 5), 
                               ha='center', fontsize=8, fontweight='bold')
        else:
            exact_mask = mask & (valid['x_boundary_type'] == 'exact')
            lower_mask = mask & (valid['x_boundary_type'] == 'lower_bound')
            
            if exact_mask.any():
                ax.scatter(
                    valid.loc[exact_mask, 'r_avg_B'],
                    valid.loc[exact_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element} (exact)"
                )
            
            if lower_mask.any():
                ax.scatter(
                    valid.loc[lower_mask, 'r_avg_B'],
                    valid.loc[lower_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.3,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element} (≥)",
                    marker='s'
                )
    
    ax.set_xlabel('Average B-site Ionic Radius r_avg_B = (1-x)·r_B + x·r_D (Å)')
    ax.set_ylabel('x(boundary)')
    if aggregate_mode:
        ax.set_title('Solubility Limit vs Average B-site Radius\n(≥ indicates lower bound estimate)')
    else:
        ax.set_title('Solubility Limit vs Average B-site Radius\n(Hollow markers = lower bound estimates)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_solubility_vs_formation_energy(df, ax, aggregate_mode=False):
    """График 2d: x(boundary) vs E_form (энергия образования)"""
    valid = df.dropna(subset=['E_form', 'x_boundary_value'])
    
    if len(valid) < 3:
        ax.text(0.5, 0.5, 'Insufficient data for formation energy analysis',
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        if aggregate_mode:
            data_mask = mask & (valid['x_boundary_value'].notna())
            if data_mask.any():
                ax.scatter(
                    valid.loc[data_mask, 'E_form'],
                    valid.loc[data_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element}"
                )
                lower_mask = data_mask & (valid['x_boundary_type'] == 'lower_bound')
                for idx in valid[lower_mask].index:
                    ax.annotate('≥', 
                               (valid.loc[idx, 'E_form'], valid.loc[idx, 'x_boundary_value']),
                               textcoords="offset points", xytext=(5, 5), 
                               ha='center', fontsize=8, fontweight='bold')
        else:
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
    if aggregate_mode:
        ax.set_title('Solubility Limit vs Formation Energy\n(≥ indicates lower bound estimate)')
    else:
        ax.set_title('Solubility Limit vs Formation Energy\n(Hollow markers = lower bound estimates)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_solubility_vs_strain_energy(df, ax, aggregate_mode=False):
    """График 2e: x(boundary) vs lattice_strain_energy (энергия деформации решетки)"""
    valid = df.dropna(subset=['lattice_strain_energy', 'x_boundary_value'])
    
    if len(valid) < 3:
        ax.text(0.5, 0.5, 'Insufficient data for lattice strain energy analysis',
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        if aggregate_mode:
            data_mask = mask & (valid['x_boundary_value'].notna())
            if data_mask.any():
                ax.scatter(
                    valid.loc[data_mask, 'lattice_strain_energy'],
                    valid.loc[data_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element}"
                )
                lower_mask = data_mask & (valid['x_boundary_type'] == 'lower_bound')
                for idx in valid[lower_mask].index:
                    ax.annotate('≥', 
                               (valid.loc[idx, 'lattice_strain_energy'], valid.loc[idx, 'x_boundary_value']),
                               textcoords="offset points", xytext=(5, 5), 
                               ha='center', fontsize=8, fontweight='bold')
        else:
            exact_mask = mask & (valid['x_boundary_type'] == 'exact')
            lower_mask = mask & (valid['x_boundary_type'] == 'lower_bound')
            
            if exact_mask.any():
                ax.scatter(
                    valid.loc[exact_mask, 'lattice_strain_energy'],
                    valid.loc[exact_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element} (exact)"
                )
            
            if lower_mask.any():
                ax.scatter(
                    valid.loc[lower_mask, 'lattice_strain_energy'],
                    valid.loc[lower_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.3,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element} (≥)",
                    marker='s'
                )
    
    ax.set_xlabel('Lattice Strain Energy ε_strain = (Δr)²·(1-x)·x (arb. units)')
    ax.set_ylabel('x(boundary)')
    if aggregate_mode:
        ax.set_title('Solubility Limit vs Lattice Strain Energy\n(≥ indicates lower bound estimate)')
    else:
        ax.set_title('Solubility Limit vs Lattice Strain Energy\n(Hollow markers = lower bound estimates)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    return ax

def plot_solubility_vs_vacancy_conc(df, ax, aggregate_mode=False):
    """График 2f: x(boundary) vs oxygen_vacancy_conc (концентрация кислородных вакансий)"""
    valid = df.dropna(subset=['oxygen_vacancy_conc', 'x_boundary_value'])
    
    if len(valid) < 3:
        ax.text(0.5, 0.5, 'Insufficient data for oxygen vacancy analysis',
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    for b_element in valid['B_element'].unique():
        mask = valid['B_element'] == b_element
        color = B_COLORS.get(b_element, B_COLORS['default'])
        
        if aggregate_mode:
            data_mask = mask & (valid['x_boundary_value'].notna())
            if data_mask.any():
                ax.scatter(
                    valid.loc[data_mask, 'oxygen_vacancy_conc'],
                    valid.loc[data_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element}"
                )
                lower_mask = data_mask & (valid['x_boundary_type'] == 'lower_bound')
                for idx in valid[lower_mask].index:
                    ax.annotate('≥', 
                               (valid.loc[idx, 'oxygen_vacancy_conc'], valid.loc[idx, 'x_boundary_value']),
                               textcoords="offset points", xytext=(5, 5), 
                               ha='center', fontsize=8, fontweight='bold')
        else:
            exact_mask = mask & (valid['x_boundary_type'] == 'exact')
            lower_mask = mask & (valid['x_boundary_type'] == 'lower_bound')
            
            if exact_mask.any():
                ax.scatter(
                    valid.loc[exact_mask, 'oxygen_vacancy_conc'],
                    valid.loc[exact_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.9,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element} (exact)"
                )
            
            if lower_mask.any():
                ax.scatter(
                    valid.loc[lower_mask, 'oxygen_vacancy_conc'],
                    valid.loc[lower_mask, 'x_boundary_value'],
                    color=color, s=100, alpha=0.3,
                    edgecolors='black', linewidth=0.5,
                    label=f"{b_element} (≥)",
                    marker='s'
                )
    
    ax.set_xlabel('Oxygen Vacancy Concentration [V_O] = x/2 (per formula unit)')
    ax.set_ylabel('x(boundary)')
    if aggregate_mode:
        ax.set_title('Solubility Limit vs Oxygen Vacancy Concentration\n(≥ indicates lower bound estimate)')
    else:
        ax.set_title('Solubility Limit vs Oxygen Vacancy Concentration\n(Hollow markers = lower bound estimates)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
            - **x(boundary)**: Solubility limit ("-" means ≥ x(inv,end))
            - **Impurity phase(s)**: Impurity phases observed
            - **x(max)**: Concentration at max conductivity
            - **doi**: DOI reference
            """)
            return
        
        # ============================================================================
        # ПОДСКАЗКА С ПОЯСНЕНИЯМИ ВЕЛИЧИН
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
            | **size_misfit** | Quadratic size misfit | `size_misfit = x·(Δr)²/r_B` |
            | **elastic_misfit** | Elastic misfit energy proxy | `elastic_misfit = (Δr/r_B)²·x·(1-x)` |
            
            ### ⚡ **Electronegativity Parameters**
            
            | Symbol | Description | Formula / Notes |
            |--------|-------------|-----------------|
            | **χ_A, χ_B, χ_D** | Electronegativity (Pauling scale) | Values from Pauling scale |
            | **χ_avg_B** | Average B-site electronegativity | `χ_avg = (1-x)·χ_B + x·χ_D` |
            | **Δχ** | Electronegativity difference | `Δχ = |χ_avg_B - χ_A|` |
            | **dΔχ/dx** | Electronegativity gradient | Rate of change of Δχ with x |
            | **Δχ_eff** | Effective Δχ relative to oxygen | `Δχ_eff = |χ_avg_B - χ_O| - |χ_A - χ_O|` |
            | **bond_strength** | Pauling bond strength proxy | `(χ - χ_O)²` |
            | **χ_variance** | Electronegativity variance on B-site | `x·(1-x)·(χ_D - χ_B)²` |
            
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
            
            ### ⚛️ **Ionic Parameters**
            
            | Symbol | Description | Formula / Notes |
            |--------|-------------|-----------------|
            | **z/r** | Ionic potential | `z/r` (Å⁻¹) - measure of polarizing power |
            | **Δ(z/r)** | Difference in ionic potential | `|(z_D/r_D) - (z_B/r_B)|` |
            | **octahedral_tilting** | Tilting tendency estimate | `max(0, 0.96 - t) × 10` |
            
            ### 🎯 **Hybrid Parameters**
            
            | Symbol | Description | Formula / Notes |
            |--------|-------------|-----------------|
            | **t·(1+Δr/r_B)** | Combined geometry-misfit | `t × (1 + Δr_rel)` |
            | **Δχ·Δr** | Product of Δχ and Δr | Often gives good separation |
            | **ε_strain/t** | Normalized strain energy | `ε_strain / t` |
            | **[V_O]·φ** | Vacancy mobility proxy | `[V_O] × φ` |
            
            ### 📊 **Composition Parameters**
            
            | Symbol | Description | Notes |
            |--------|-------------|-------|
            | **x_boundary** | Solubility limit | Maximum x for single-phase solid solution |
            | **log_x_boundary** | Logarithmic solubility | `log10(x_boundary + ε)` |
            | **solubility_energy_proxy** | Solubility energy proxy | `-ln(x_boundary + ε)` |
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
            | **SHAP values** | Feature contribution to predictions |
            """)

        st.markdown("---")
        st.header("📊 Plot Settings")
        
        show_grid = st.checkbox("Show grid", value=True)
        show_legend = st.checkbox("Show legend", value=True)
        
        st.markdown("---")
        st.header("🔢 Data Processing")
        
        include_lower_bounds = st.checkbox(
            "Include lower bound estimates (≥) in visual distinction",
            value=True,
            help="When enabled, '-' in x(boundary) is shown as transparent/outline markers (standard mode)"
        )
        
        # НОВЫЙ ПАРАМЕТР: Режим агрегации
        aggregate_lower_bounds = st.checkbox(
            "🔗 Aggregate lower bounds with x(inv,end) [RECOMMENDED]",
            value=True,
            help="When enabled, '-' in x(boundary) is treated as x(inv,end) and included in all statistics and plots as exact values (with '≥' annotation). This gives a conservative estimate of average solubility."
        )
        
        st.markdown("---")
        st.header("🔍 Filters")
        
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            with st.spinner("Processing data..."):
                # Передаем параметр aggregate_lower_bounds в process_data
                df_processed = process_data(df, aggregate_lower_bounds=aggregate_lower_bounds)
                
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
                    default=['exact', 'lower_bound'] if include_lower_bounds and not aggregate_lower_bounds else ['exact'],
                    help="Select which types of x(boundary) to include (in aggregated mode, only exact type is shown)"
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
                
                if 'x_boundary_type' in filtered_df.columns and not aggregate_lower_bounds:
                    filtered_df = filtered_df[filtered_df['x_boundary_type'].isin(x_boundary_type_filter)]
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
    
    if uploaded_file is not None and len(filtered_df) > 0:
        # Отображение информации о режиме обработки
        if aggregate_lower_bounds:
            st.info("🔄 **Aggregated Mode Active**: All '-' entries are treated as x(inv,end) values. Points marked with '≥' indicate these are lower bound estimates.")
        else:
            st.info("📊 **Standard Mode Active**: Exact values shown as filled markers, lower bounds (≥) as transparent/outline markers.")
        
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
                df_stats = filtered_df.dropna(subset=['x_boundary_value'])
                
                if len(df_stats) > 0:
                    b_stats = df_stats.groupby('B_element')['x_boundary_value'].agg(['mean', 'median', 'count', 'std']).round(3)
                    st.dataframe(b_stats, use_container_width=True)
        
        with col2:
            st.markdown("**Top 5 Dopants by Median Solubility**")
            dopant_stats = get_dopant_statistics(filtered_df, include_lower_bounds, aggregate_lower_bounds)
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
                           'dr', 'tolerance_factor', 'size_misfit', 'elastic_misfit',
                           'Δχ', 'Δχ_eff', 'ionic_potential_avg', 'free_volume_fraction',
                           'packing_factor', 'E_form', 'log_x_boundary', 'solubility_energy_proxy',
                           'has_impurity', 'year', 'system']
            # Убираем дубликаты колонок из DataFrame
            filtered_df = filtered_df.loc[:, ~filtered_df.columns.duplicated()]
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
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Basic Statistics",
            "🔬 Solubility Analysis",
            "⚡ Conductivity Analysis",
            "📈 Advanced Visualization",
            "🧊 Volumetric & Thermodynamic",
            "🤖 ML Insights",
            "🎯 SHAP Analysis"
        ])
        
        # ============================================================================
        # ВКЛАДКА 1: BASIC STATISTICS
        # ============================================================================
        with tab1:
            st.subheader("Basic Statistical Analysis")
            
            if len(filtered_df) > 0:
                fig, stats_df = plot_b_site_statistics(filtered_df, include_lower_bounds, aggregate_lower_bounds)
                st.pyplot(fig)
                plt.close(fig)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'x_max' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
                    fig = plot_xmax_vs_boundary_histogram(filtered_df)
                    st.pyplot(fig)
                    plt.close(fig)
            
            with col2:
                if 'x_boundary_value' in filtered_df.columns:
                    fig = plot_distribution_kde(filtered_df, aggregate_lower_bounds)
                    st.pyplot(fig)
                    plt.close(fig)
            
            if len(filtered_df) > 0:
                fig = plot_correlation_heatmap(filtered_df, include_lower_bounds, aggregate_lower_bounds)
                st.pyplot(fig)
                plt.close(fig)
            
            # Полная корреляционная матрица
            st.subheader("Comprehensive Correlation Matrix")
            fig = plot_comprehensive_correlation_matrix(filtered_df, include_lower_bounds)
            st.pyplot(fig)
            plt.close(fig)
            
            st.subheader("Detailed Correlations with p-values")
            features = ['dr', 'dr_rel', 'tolerance_factor', 'size_misfit', 'elastic_misfit',
                        'Δχ', 'Δχ_eff', 'ionic_potential_avg', 'free_volume_fraction',
                        'packing_factor', 'x_boundary_value', 'x_max', 'log_x_boundary']
            available_features = [f for f in features if f in filtered_df.columns]
            if len(available_features) >= 2:
                corr_df = calculate_correlations(filtered_df, available_features, include_lower_bounds, aggregate_lower_bounds)
                if len(corr_df) > 0:
                    st.dataframe(corr_df, use_container_width=True)
            
            # Автоматическая генерация инсайтов
            st.subheader("💡 Automated Physical Insights")
            insights = generate_insights(filtered_df)
            for insight in insights:
                st.info(insight)
        
        # ============================================================================
        # ВКЛАДКА 2: SOLUBILITY ANALYSIS (ОБНОВЛЕНА С НОВЫМИ ГРАФИКАМИ)
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
                    "Research Intensity Matrix",
                    "2a: Solubility vs Electronegativity Difference (Δχ)",
                    "2b: Solubility vs Average B-site Electronegativity (χ_avg_B)",
                    "2c: Solubility vs Average B-site Radius (r_avg_B)",
                    "2d: Solubility vs Formation Energy (E_form)",
                    "2e: Solubility vs Lattice Strain Energy (ε_strain)",
                    "2f: Solubility vs Oxygen Vacancy Concentration ([V_O])",
                    "Contour Plot: t-Δr vs x(boundary)",
                    "Parallel Coordinates Plot",
                    "Violin Plot by Impurity"
                ],
                default=["Solubility vs Radius Difference", "Δr Heatmap with x(boundary)", "2a: Solubility vs Electronegativity Difference (Δχ)"]
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
                            plot_solubility_vs_dr(filtered_df, ax, aggregate_lower_bounds)
                        else:
                            ax.text(0.5, 0.5, 'Required data missing', ha='center', va='center')
                    
                    elif plot_name == "Solubility vs Tolerance Factor":
                        if 'tolerance_factor' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
                            plot_tolerance_factor(filtered_df, ax, aggregate_lower_bounds)
                        else:
                            ax.text(0.5, 0.5, 'Required data missing', ha='center', va='center')
                    
                    elif plot_name == "Top Dopants Violin Plot":
                        if 'x_boundary_value' in filtered_df.columns and 'D_element' in filtered_df.columns:
                            plt.close(fig)
                            violin_fig = plot_top_dopants_violin(filtered_df, include_lower_bounds, aggregate_lower_bounds)
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
                    
                    elif plot_name == "2a: Solubility vs Electronegativity Difference (Δχ)":
                        if 'Δχ' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
                            plot_solubility_vs_delta_chi(filtered_df, ax, aggregate_lower_bounds)
                        else:
                            ax.text(0.5, 0.5, 'Δχ data not available', ha='center', va='center')
                    
                    elif plot_name == "2b: Solubility vs Average B-site Electronegativity (χ_avg_B)":
                        if 'χ_avg_B' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
                            plot_solubility_vs_avg_chi(filtered_df, ax, aggregate_lower_bounds)
                        else:
                            ax.text(0.5, 0.5, 'χ_avg_B data not available', ha='center', va='center')
                    
                    elif plot_name == "2c: Solubility vs Average B-site Radius (r_avg_B)":
                        if 'r_avg_B' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
                            plot_solubility_vs_avg_radius(filtered_df, ax, aggregate_lower_bounds)
                        else:
                            ax.text(0.5, 0.5, 'r_avg_B data not available', ha='center', va='center')
                    
                    elif plot_name == "2d: Solubility vs Formation Energy (E_form)":
                        if 'E_form' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
                            plot_solubility_vs_formation_energy(filtered_df, ax, aggregate_lower_bounds)
                        else:
                            ax.text(0.5, 0.5, 'E_form data not available', ha='center', va='center')
                    
                    elif plot_name == "2e: Solubility vs Lattice Strain Energy (ε_strain)":
                        if 'lattice_strain_energy' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
                            plot_solubility_vs_strain_energy(filtered_df, ax, aggregate_lower_bounds)
                        else:
                            ax.text(0.5, 0.5, 'lattice_strain_energy data not available', ha='center', va='center')
                    
                    elif plot_name == "2f: Solubility vs Oxygen Vacancy Concentration ([V_O])":
                        if 'oxygen_vacancy_conc' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
                            plot_solubility_vs_vacancy_conc(filtered_df, ax, aggregate_lower_bounds)
                        else:
                            ax.text(0.5, 0.5, 'oxygen_vacancy_conc data not available', ha='center', va='center')
                    
                    elif plot_name == "Contour Plot: t-Δr vs x(boundary)":
                        if 'tolerance_factor' in filtered_df.columns and 'dr' in filtered_df.columns and 'x_boundary_value' in filtered_df.columns:
                            plot_contour_t_dr(filtered_df, ax, aggregate_lower_bounds)
                        else:
                            ax.text(0.5, 0.5, 'Required data missing', ha='center', va='center')
                    
                    elif plot_name == "Parallel Coordinates Plot":
                        if 'x_boundary_value' in filtered_df.columns:
                            plt.close(fig)
                            parallel_fig = plot_parallel_coordinates(filtered_df, ax)
                            if isinstance(parallel_fig, plt.Figure):
                                st.pyplot(parallel_fig)
                                plt.close(parallel_fig)
                            plot_idx -= 1
                    
                    elif plot_name == "Violin Plot by Impurity":
                        if 'x_boundary_value' in filtered_df.columns and 'has_impurity' in filtered_df.columns:
                            plot_violin_by_impurity(filtered_df, 'x_boundary_value', ax, aggregate_lower_bounds)
                        else:
                            ax.text(0.5, 0.5, 'Required data missing', ha='center', va='center')
                    
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
                        plot_xmax_vs_xboundary(filtered_df, ax, aggregate_lower_bounds)
                        if not show_grid:
                            ax.grid(False)
                        if not show_legend:
                            ax.legend().remove()
                        st.pyplot(fig)
                        plt.close(fig)
                
                with col2:
                    if 'tolerance_factor' in filtered_df.columns:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plot_xmax_vs_tolerance(filtered_df, ax, aggregate_lower_bounds)
                        if not show_grid:
                            ax.grid(False)
                        if not show_legend:
                            ax.legend().remove()
                        st.pyplot(fig)
                        plt.close(fig)
                
                if 'x_rel_max' in filtered_df.columns and 'dr_rel' in filtered_df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_relative_position(filtered_df, ax, aggregate_lower_bounds)
                    if not show_grid:
                        ax.grid(False)
                    if not show_legend:
                        ax.legend().remove()
                    st.pyplot(fig)
                    plt.close(fig)
                
                st.subheader("Shift Analysis")
                fig = plot_shift_vs_dr_bubble(filtered_df, aggregate_lower_bounds)
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
                    "PCA Loadings Plot",
                    "Impurity Phase Diagram (t-Δr)",
                    "Temporal Trends",
                    "Tolerance Factor Evolution",
                    "Goldschmidt Bubble Diagram",
                    "Dopant Comparison by B-site",
                    "3D Interactive Plot",
                    "Sankey Diagram"
                ],
                default=["PCA Analysis", "Goldschmidt Bubble Diagram", "3D Interactive Plot"]
            )
            
            for plot_name in adv_plots:
                if plot_name == "PCA Analysis":
                    fig = plot_pca(filtered_df)
                    st.pyplot(fig)
                    plt.close(fig)
                
                elif plot_name == "PCA Loadings Plot":
                    fig = plot_pca_loadings(filtered_df)
                    if fig is not None:
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
                
                elif plot_name == "3D Interactive Plot":
                    st.plotly_chart(plot_3d_interactive(filtered_df), use_container_width=True)
                
                elif plot_name == "Sankey Diagram":
                    st.plotly_chart(plot_sankey_diagram(filtered_df), use_container_width=True)
        
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
                        plot_free_volume_vs_xboundary(filtered_df, ax, aggregate_lower_bounds)
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
                        plot_formation_energy_vs_xboundary(filtered_df, ax, aggregate_lower_bounds)
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
            
            # Выбор признаков для ML
            ml_features = st.multiselect(
                "Select features for ML analysis",
                options=[f for f in ['dr', 'tolerance_factor', 'size_misfit', 'elastic_misfit',
                                     'Δχ', 'Δχ_eff', 'ionic_potential_avg', 'free_volume_fraction',
                                     'lattice_strain_energy', 'oxygen_vacancy_conc', 'packing_factor']
                         if f in filtered_df.columns],
                default=[f for f in ['dr', 'tolerance_factor', 'size_misfit', 'Δχ_eff', 'free_volume_fraction']
                         if f in filtered_df.columns][:5]
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Feature Importance (Random Forest)**")
                fig, importance_df = plot_feature_importance(filtered_df)
                if fig is not None:
                    st.pyplot(fig)
                    plt.close(fig)
                if importance_df is not None:
                    st.dataframe(importance_df, use_container_width=True)
            
            with col2:
                st.markdown("**Model Comparison**")
                models_df, models, X, y = compare_ml_models(filtered_df, ml_features, 'x_boundary_value')
                if models_df is not None:
                    st.dataframe(models_df, use_container_width=True)
            
            # Clustering
            st.markdown("**Clustering Analysis (DBSCAN)**")
            cluster_features = st.multiselect(
                "Features for clustering",
                options=ml_features,
                default=ml_features[:min(3, len(ml_features))]
            )
            
            if len(cluster_features) >= 2:
                eps = st.slider("DBSCAN eps", 0.1, 2.0, 0.5, 0.05)
                min_samples = st.slider("min_samples", 2, 10, 3, 1)
                
                labels, plot_df = perform_clustering(filtered_df, cluster_features, eps, min_samples)
                if labels is not None:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(plot_df[cluster_features[0]], plot_df[cluster_features[1]], 
                                        c=labels, cmap='viridis', s=100, edgecolors='black')
                    ax.set_xlabel(cluster_features[0])
                    ax.set_ylabel(cluster_features[1])
                    ax.set_title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
                    plt.colorbar(scatter, ax=ax, label='Cluster')
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.write(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
                    st.write(f"Number of noise points: {(labels == -1).sum()}")
        
        # ============================================================================
        # ВКЛАДКА 7: SHAP ANALYSIS
        # ============================================================================
        with tab7:
            st.subheader("SHAP Analysis for Model Interpretability")
            
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** values show how each feature contributes to the prediction.
            - **Red** = higher feature value increases prediction
            - **Blue** = higher feature value decreases prediction
            - **x-axis** = impact on solubility prediction
            """)
            
            shap_features = st.multiselect(
                "Select features for SHAP analysis",
                options=[f for f in ['dr', 'tolerance_factor', 'size_misfit', 'elastic_misfit',
                                     'Δχ', 'Δχ_eff', 'ionic_potential_avg', 'free_volume_fraction',
                                     'lattice_strain_energy', 'oxygen_vacancy_conc']
                         if f in filtered_df.columns],
                default=[f for f in ['dr', 'tolerance_factor', 'size_misfit', 'Δχ_eff']
                         if f in filtered_df.columns][:4]
            )
            
            if len(shap_features) >= 2:
                # Обучаем модель для SHAP
                plot_df = filtered_df.dropna(subset=shap_features + ['x_boundary_value'])
                
                if len(plot_df) > 10:
                    X_shap = plot_df[shap_features].copy()
                    if 'B_element' in plot_df.columns:
                        X_shap = pd.concat([X_shap, pd.get_dummies(plot_df['B_element'], prefix='B')], axis=1)
                    y_shap = plot_df['x_boundary_value']
                    
                    model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
                    model.fit(X_shap, y_shap)
                    
                    # SHAP values
                    explainer, shap_values = calculate_shap_values(model, X_shap, X_shap.columns.tolist())
                    
                    if explainer is not None and shap_values is not None:
                        # Summary plot
                        st.markdown("**SHAP Summary Plot**")
                        fig, ax = plt.subplots(figsize=(10, 8))
                        shap.summary_plot(shap_values, X_shap, feature_names=X_shap.columns.tolist(), show=False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Bar plot
                        st.markdown("**SHAP Feature Importance (Mean |SHAP|)**")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values, X_shap, feature_names=X_shap.columns.tolist(), 
                                        plot_type="bar", show=False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Individual force plot
                        st.markdown("**SHAP Force Plot for a Random Sample**")
                        sample_idx = np.random.randint(0, len(X_shap))
                        fig, ax = plt.subplots(figsize=(12, 2))
                        shap.force_plot(explainer.expected_value, shap_values[sample_idx,:], 
                                       X_shap.iloc[sample_idx,:], feature_names=X_shap.columns.tolist(),
                                       matplotlib=True, show=False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Partial Dependence Plots
                        st.markdown("**Partial Dependence Plots (PDP)**")
                        pdp_features = st.multiselect(
                            "Select features for PDP",
                            options=shap_features,
                            default=shap_features[:min(2, len(shap_features))]
                        )
                        
                        if pdp_features:
                            n_pdp = len(pdp_features)
                            fig, axes = plt.subplots(1, n_pdp, figsize=(6*n_pdp, 5))
                            if n_pdp == 1:
                                axes = [axes]
                            
                            for i, feat in enumerate(pdp_features):
                                plot_partial_dependence(model, X_shap, feat, axes[i])
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        # ICE plots
                        st.markdown("**ICE Curves (Individual Conditional Expectation)**")
                        ice_feature = st.selectbox("Select feature for ICE plots", options=shap_features)
                        if ice_feature:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            plot_ice_curves(model, X_shap, ice_feature, ax)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                    else:
                        st.warning("SHAP calculation failed. Try with different features.")
                else:
                    st.warning(f"Insufficient data for SHAP analysis. Need at least 10 samples, have {len(plot_df)}.")
            else:
                st.info("Select at least 2 features for SHAP analysis.")
    
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
        
        #### 🆕 **NEW: Advanced Descriptor System**
        - **PerovskiteDescriptorCalculator class** - Modular calculation of all descriptors
        - **New geometric descriptors**: size_misfit, elastic_misfit, octahedral_tilting
        - **New electronic descriptors**: Δχ_eff (oxygen-weighted), bond_strength, electronegativity_variance
        - **New ionic descriptors**: ionic_potential (z/r), Δ_ionic_potential
        - **New hybrid descriptors**: t_dr_hybrid, Δχ_dr_hybrid, strain_t_ratio, vacancy_mobility_proxy
        - **Logarithmic target**: log_x_boundary, solubility_energy_proxy for linearized correlations
        
        #### 🤖 **NEW: Machine Learning & Interpretability**
        - **SHAP analysis** with summary plots, force plots, and feature importance
        - **Model comparison**: Random Forest, Gradient Boosting, XGBoost with cross-validation
        - **Partial Dependence Plots (PDP)** and **ICE curves** for individual features
        - **DBSCAN clustering** to identify material families
        - **Feature importance analysis** with customizable feature selection
        
        #### 📊 **NEW: Advanced Visualizations**
        - **Contour plots** - x_boundary as function of t and Δr
        - **Parallel Coordinates Plot** for multivariate analysis
        - **Interactive 3D plots** with plotly
        - **PCA loadings plot** (correlation circle)
        - **Sankey diagram** for data flow (B-site → Dopant → Impurity)
        - **Violin plots with impurity separation**
        - **Comprehensive correlation matrix** with 15+ descriptors
        
        #### 💡 **NEW: Automated Insights**
        - **Rule-based insight generation** - automatically detects and reports:
          - Strong correlations with solubility
          - Optimal tolerance factor ranges
          - Critical Δr thresholds for impurity formation
          - Free volume fraction effects
          - Electronegativity trends
        
        #### 🚀 **Technical Improvements**
        - **@st.cache_data decorators** for all heavy computations
        - **Modular PerovskiteDescriptorCalculator class**
        - **System column** for easy identification (e.g., "BaCeO3 + 0.15Gd")
        - **Enhanced error handling** for missing data
        - **Configurable descriptor selection** in sidebar
        
        ### Key Features:
        - **Aggregated Mode** - Treats '-' entries as x(inv,end) for conservative estimates
        - **Visual distinction for lower bounds** - '≥' annotations on plots
        - **Extended Solubility Analysis** - 6 new plots (2a-2f) for comprehensive descriptor analysis
        - **Improved statistics** - Mean solubility includes lower bound estimates when aggregated
        """)

if __name__ == "__main__":
    main()
