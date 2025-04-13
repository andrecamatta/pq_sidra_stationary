import pandas as pd
import numpy as np
import traceback
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from scipy.stats import kruskal, linregress
from sktime.transformations.base import BaseTransformer
from sktime.transformations.series.difference import Differencer
from sktime.transformations.series.boxcox import BoxCoxTransformer, LogTransformer # Reverting based on documentation
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.compose import TransformerPipeline
# from sktime.transformations.series.boxcox import BoxCoxTransformer # Already imported above
from sktime.transformations.series.detrend import Detrender # Alternativa para tendência
from sktime.transformations.compose import Id # Identity transformer for passthrough

# Funções Auxiliares para Verificações
# ====================================

def check_stationarity(series: pd.Series) -> bool:
    # Verifica estacionariedade (ADF Test)
    adf_pvalue = adfuller(series.astype(float).dropna(), autolag='AIC')[1]
    return adf_pvalue < 0.05

def check_seasonality(series: pd.Series, seasonality: int = 12) -> bool:
    # Verifica sazonalidade (STL + Kruskal-Wallis)
    ts = series.astype(float).dropna()
    if len(ts) < 2 * seasonality:
        return False # Dados insuficientes

    # Decomposição STL
    stl = STL(ts, period=seasonality, robust=True)
    result = stl.fit()
    seasonal = result.seasonal

    # Teste Kruskal-Wallis
    groups = [seasonal.iloc[i::seasonality] for i in range(seasonality)]
    groups = [g for g in groups if not g.empty]
    if len(groups) < 2:
         return False
    stat, p_value = kruskal(*groups)
    kruskal_result = p_value < 0.05

    # Força Sazonal
    total_var = np.var(ts)
    if total_var < 1e-9:
        return False # Série constante
    seasonal_var = np.var(seasonal)
    seasonal_strength = seasonal_var / total_var
    seasonal_strength_result = seasonal_strength > 0.10 # Limiar heurístico

    return kruskal_result and seasonal_strength_result
    # Removido try-except para simplificar


def check_proportional_variance(series: pd.Series) -> bool:
    # Verifica variância proporcional (heterocedasticidade)
    ts = series.astype(float).dropna()
    if len(ts) < 2:
        return False
    abs_diff = abs(ts.diff()).dropna()
    if len(abs_diff) < 2:
        return False
    aligned_ts = ts.iloc[1:][abs_diff.index]
    if len(aligned_ts) != len(abs_diff) or len(aligned_ts) < 2:
         # print("Aviso: Problema de alinhamento ou dados insuficientes na verificação de variância.") # Comentário removido
         return False

    # Regressão linear
    slope, intercept, r_value, p_value, std_err = linregress(aligned_ts, abs_diff)
    # Verifica se a inclinação é significativamente positiva
    return p_value / 2 < 0.05 and slope > 0
    # Removido try-except para simplificar


# Funções Fábrica de Transformadores
# ==================================

def passthrough_transformer():
    # Retorna transformador identidade
    return Id()

def delta_seasonal_transformer(seasonality=12):
    return Differencer(lags=seasonality)

def delta_seasonal_delta_transformer(seasonality=12):
    return Differencer(lags=[1, seasonality])

def ln_transformer():
    # Retorna transformador Log (assume dados positivos)
    return LogTransformer()

def delta_ln_transformer():
    return TransformerPipeline(steps=[
        ("log", LogTransformer()),
        ("diff", Differencer(lags=1))
    ])

def delta_delta_ln_transformer():
    return TransformerPipeline(steps=[
        ("log", LogTransformer()),
        ("diff1", Differencer(lags=1)),
        ("diff2", Differencer(lags=1))
    ])

def delta_transformer():
    return Differencer(lags=1)

def delta_delta_transformer():
    return Differencer(lags=[1, 1])


# Lógica de Seleção do Transformador
# ==================================

def transform_and_check(series: pd.Series, transformer: BaseTransformer) -> bool:
    # Aplica transformador e verifica estacionariedade
    try:
        transformer.fit(series)
        transformed_series = transformer.transform(series)
        return check_stationarity(transformed_series)
    except Exception:
        # Falha na transformação ou verificação
        return False

def select_transformer(series: pd.Series, seasonality: int = 12) -> BaseTransformer:
    # Seleciona o transformador sktime apropriado
    series = pd.to_numeric(series, errors='coerce').dropna()
    if series.empty:
        print("Aviso: Série vazia ou NaN. Usando Passthrough.")
        return passthrough_transformer()
    if np.isinf(series).any():
        print("Aviso: Série contém Inf. Substituindo por NaN.")
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            print("Aviso: Série vazia após remover Inf. Usando Passthrough.")
            return passthrough_transformer()

    if check_stationarity(series):
        print("Série já estacionária. Usando Passthrough.")
        return passthrough_transformer()

    candidates = []
    if check_seasonality(series, seasonality):
        print("Sazonalidade detectada.")
        candidates.extend([
            ('ΔSazonal', delta_seasonal_transformer(seasonality)),
            ('ΔSazonalΔ', delta_seasonal_delta_transformer(seasonality)),
        ])

    is_positive = (series > 0).all()
    if check_proportional_variance(series):
        print("Variância proporcional detectada.")
        if is_positive:
             candidates.extend([
                 ('ln(x)', ln_transformer()),
                 ('Δln(x)', delta_ln_transformer()),
                 ('ΔΔln(x)', delta_delta_ln_transformer()),
             ])
        else:
             print("Aviso: Valores não positivos, pulando transformações log.")

    print("Tentando diferenciação geral.")
    candidates.extend([
        ('Δx', delta_transformer()),
        ('ΔΔx', delta_delta_transformer()),
    ])
    candidates.append(('Passthrough', passthrough_transformer())) # Último recurso

    print(f"\nSelecionando para série (head): \n{series.head()}")
    for name, transformer_factory in candidates:
        # Usar a factory para criar nova instância a cada verificação
        current_transformer = transformer_factory # Assumindo que as funções retornam instâncias
        print(f"Verificando: {name} ({type(current_transformer).__name__})")
        if transform_and_check(series.copy(), current_transformer):
            print(f"Selecionado: {name}")
            # Retorna uma NOVA instância da factory selecionada
            if name == 'ΔSazonal': return delta_seasonal_transformer(seasonality)
            if name == 'ΔSazonalΔ': return delta_seasonal_delta_transformer(seasonality)
            if name == 'ln(x)': return ln_transformer()
            if name == 'Δln(x)': return delta_ln_transformer()
            if name == 'ΔΔln(x)': return delta_delta_ln_transformer()
            if name == 'Δx': return delta_transformer()
            if name == 'ΔΔx': return delta_delta_transformer()
            if name == 'Passthrough': return passthrough_transformer()
            print(f"Aviso: Nome não correspondente {name}. Retornando Passthrough.") # Salvaguarda
            return passthrough_transformer()

    print("Nenhum transformador adequado encontrado. Usando Passthrough.") # Salvaguarda
    return passthrough_transformer()


# Classe Transformadora Customizada sktime
# ========================================

class StationarityTransformer(BaseTransformer):
    # Aplica automaticamente a transformação apropriada para tornar colunas estacionárias.
    _tags = {
        "scitype:transform-input": "Series", "scitype:transform-output": "Series",
        "scitype:instancewise": False, "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None", "fit_is_empty": False,
        "capability:inverse_transform": True, "handles-missing-data": False,
    }

    def __init__(self, seasonality=12):
        self.seasonality = seasonality
        self.columns_transformers_ = {}
        super().__init__()

    def _fit(self, X, y=None):
        # Ajusta o transformador para cada coluna
        self.columns_transformers_ = {}
        for col in X.columns:
            print(f"\n--- Ajustando coluna: {col} ---")
            series = X[col]
            transformer = select_transformer(series, self.seasonality)
            try:
                transformer.fit(series)
                self.columns_transformers_[col] = transformer
                print(f"Ajustado {type(transformer).__name__} para '{col}'")
            except Exception as e:
                 print(f"Erro ao ajustar {type(transformer).__name__} para {col}: {e}. Usando Passthrough.")
                 passthrough = passthrough_transformer()
                 passthrough.fit(series) # Ajusta o passthrough
                 self.columns_transformers_[col] = passthrough
        return self

    def _transform(self, X, y=None):
        # Aplica transformações ajustadas
        self.check_is_fitted()
        transformed_columns = {}
        for col in X.columns:
            if col in self.columns_transformers_:
                transformer = self.columns_transformers_[col]
                try:
                    transformed_columns[col] = transformer.transform(X[col])
                except Exception as e:
                    print(f"Erro ao transformar {col} com {type(transformer).__name__}: {e}. Retornando original.")
                    transformed_columns[col] = X[col].copy()
            else:
                print(f"Aviso: Nenhum transformador para '{col}'. Retornando original.")
                transformed_columns[col] = X[col].copy()
        return pd.DataFrame(transformed_columns, index=X.index)

    def _inverse_transform(self, X, y=None):
        # Aplica transformações inversas
        self.check_is_fitted()
        inverse_transformed_columns = {}
        for col in X.columns:
             if col in self.columns_transformers_:
                transformer = self.columns_transformers_[col]
                if hasattr(transformer, 'inverse_transform'):
                    try:
                        inverse_transformed_columns[col] = transformer.inverse_transform(X[col])
                    except Exception as e:
                        print(f"Erro na inversa de {col} com {type(transformer).__name__}: {e}. Retornando dados.")
                        inverse_transformed_columns[col] = X[col].copy()
                else:
                    print(f"Aviso: {type(transformer).__name__} não tem inversa para '{col}'. Retornando dados.")
                    inverse_transformed_columns[col] = X[col].copy()
             else:
                print(f"Aviso: Nenhum transformador para inversa de '{col}'. Retornando dados.")
                inverse_transformed_columns[col] = X[col].copy()
        return pd.DataFrame(inverse_transformed_columns, index=X.index)

# End of class StationarityTransformer

# Lógica Principal de Execução
# ============================
if __name__ == "__main__":
    input_file = 'sidra_data.xlsx'
    output_file = 'stationary_data.xlsx'
    try:
        print(f"Carregando dados de {input_file}...")
        df = pd.read_excel(input_file, index_col='Date', parse_dates=True)
        print("Dados carregados com sucesso.")

        if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
             print("Aviso: Índice não é DatetimeIndex/PeriodIndex. Tentando converter...")
             try:
                 df.index = pd.to_datetime(df.index)
                 print("Índice convertido para DatetimeIndex.")
             except Exception as e:
                 print(f"Erro ao converter índice: {e}. Prosseguindo...")

        print("Garantindo que as colunas sejam numéricas...")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().all():
                 print(f"Aviso: Coluna '{col}' é inteiramente NaN após coerção.")

        # df = df.ffill() # Opcional: Lidar com NaNs

        print("\nInicializando o Stationarity Transformer...")
        stationarity_transformer = StationarityTransformer(seasonality=12)

        print("\nAjustando o transformador aos dados...")
        stationarity_transformer.fit(df.dropna(axis=1, how='all')) # Ajusta apenas em colunas não-NaN

        print("\nTransformando dados para séries estacionárias...")
        # Transforma o DataFrame original, que pode ter NaNs onde colunas foram dropadas no fit
        stationary_df = stationarity_transformer.transform(df) 

        print("\n--- Transformadores Selecionados ---")
        for col, transformer in stationarity_transformer.columns_transformers_.items():
            print(f"Coluna '{col}': {type(transformer).__name__}")
            if isinstance(transformer, TransformerPipeline):
                 print("  Passos do Pipeline:")
                 for name, step in transformer.steps:
                     print(f"  - {name}: {type(step).__name__}")
            elif isinstance(transformer, Differencer):
                 print(f"  Lags: {transformer.lags}")

        print(f"\nSalvando dados estacionários em {output_file}...")
        stationary_df.to_excel(output_file)
        print("Dados estacionários salvos com sucesso.")

    except FileNotFoundError:
        print(f"Erro: Arquivo de entrada '{input_file}' não encontrado.")
    except ImportError as e:
         print(f"Erro: Biblioteca necessária ausente: {e}")
         print("Execute: uv add pandas numpy statsmodels scipy scikit-learn sktime openpyxl")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        traceback.print_exc()
