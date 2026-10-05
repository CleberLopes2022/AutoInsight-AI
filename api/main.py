from pathlib import Path
import json

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# =========================================================
# 1. CAMINHOS DO PROJETO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "modelo"
    / "autoinsight_hgb_v1.joblib"
)

METADATA_PATH = (
    BASE_DIR
    / "modelo"
    / "metadata.json"
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "fipe_api.parquet"
)


# =========================================================
# 2. CARREGAR MODELO
# =========================================================

try:
    modelo = joblib.load(MODEL_PATH)

except Exception as erro:
    raise RuntimeError(
        f"Erro ao carregar o modelo: {erro}"
    )


# =========================================================
# 3. CARREGAR METADATA
# =========================================================

try:
    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as arquivo:

        metadata = json.load(arquivo)

except Exception as erro:
    raise RuntimeError(
        f"Erro ao carregar metadata: {erro}"
    )


# =========================================================
# 4. CARREGAR BASE FIPE
# =========================================================

try:
    df_fipe = pd.read_parquet(DATA_PATH)

except Exception as erro:
    raise RuntimeError(
        f"Erro ao carregar base FIPE: {erro}"
    )


# Converter data
df_fipe["data_referencia"] = pd.to_datetime(
    df_fipe["data_referencia"]
)


# Garantir código FIPE como string
df_fipe["codigo_fipe"] = (
    df_fipe["codigo_fipe"]
    .astype(str)
    .str.strip()
)


# =========================================================
# 5. FASTAPI
# =========================================================

app = FastAPI(
    title="AutoInsight AI API",
    description=(
        "API inteligente para consulta e previsão "
        "de preços de veículos baseada na FIPE."
    ),
    version="1.0.0"
)


# =========================================================
# 6. SCHEMA DE ENTRADA
# =========================================================

class VeiculoRequest(BaseModel):

    codigo_fipe: str
    ano_modelo: int


# =========================================================
# 7. HOME
# =========================================================

@app.get("/")
def home():

    return {
        "projeto": "AutoInsight AI",
        "status": "online",
        "modelo": metadata["nome"],
        "versao": metadata["versao"]
    }


# =========================================================
# 8. HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "modelo_carregado": True,
        "features": len(metadata["features"]),
        "registros_fipe": len(df_fipe)
    }


# =========================================================
# 9. LISTAR MARCAS
# =========================================================

@app.get("/marcas")
def listar_marcas():

    marcas = (
        df_fipe["nome_marca"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return {
        "total": len(marcas),
        "marcas": marcas
    }


# =========================================================
# 10. LISTAR MODELOS POR MARCA
# =========================================================

@app.get("/modelos")
def listar_modelos(marca: str):

    marca = marca.strip()

    filtro = (
        df_fipe["nome_marca"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        == marca.casefold()
    )

    modelos = (
        df_fipe.loc[
            filtro,
            [
                "codigo_fipe",
                "nome_modelo"
            ]
        ]
        .dropna()
        .drop_duplicates()
        .sort_values(
            "nome_modelo"
        )
    )

    if modelos.empty:

        raise HTTPException(
            status_code=404,
            detail="Marca não encontrada."
        )

    return {
        "marca": marca,
        "total": len(modelos),
        "modelos": modelos.to_dict(
            orient="records"
        )
    }


# =========================================================
# 11. LISTAR ANOS POR CÓDIGO FIPE
# =========================================================

@app.get("/anos")
def listar_anos(codigo_fipe: str):

    codigo_fipe = codigo_fipe.strip()

    anos = (
        df_fipe.loc[
            df_fipe["codigo_fipe"]
            == codigo_fipe,
            "ano_modelo"
        ]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values(
            ascending=False
        )
        .tolist()
    )

    if not anos:

        raise HTTPException(
            status_code=404,
            detail="Código FIPE não encontrado."
        )

    return {
        "codigo_fipe": codigo_fipe,
        "anos": anos
    }

# =========================================================
# 12. HISTÓRICO DO VEÍCULO
# =========================================================

@app.get("/historico")
def historico_veiculo(
    codigo_fipe: str,
    ano_modelo: int
):

    codigo_fipe = codigo_fipe.strip()

    historico = (
        df_fipe[
            (
                df_fipe["codigo_fipe"]
                == codigo_fipe
            )
            &
            (
                df_fipe["ano_modelo"]
                == ano_modelo
            )
        ]
        .sort_values("data_referencia")
        .copy()
    )

    # -----------------------------------------------------
    # VEÍCULO NÃO ENCONTRADO
    # -----------------------------------------------------

    if historico.empty:

        raise HTTPException(
            status_code=404,
            detail="Veículo não encontrado na base FIPE."
        )

    # -----------------------------------------------------
    # CALCULAR VARIAÇÃO MENSAL
    # -----------------------------------------------------

    historico["variacao_mensal_pct"] = (
        historico["preco"]
        .pct_change(fill_method=None)
        .mul(100)
    )

    # -----------------------------------------------------
    # IDENTIFICAÇÃO
    # -----------------------------------------------------

    ultimo = historico.iloc[-1]

    # -----------------------------------------------------
    # PREPARAR SÉRIE
    # -----------------------------------------------------

    serie = []

    for _, linha in historico.iterrows():

        variacao = linha["variacao_mensal_pct"]

        serie.append(
            {
                "data": (
                    linha["data_referencia"]
                    .strftime("%Y-%m-%d")
                ),

                "preco": round(
                    float(linha["preco"]),
                    2
                ),

                "variacao_mensal_pct": (
                    None
                    if pd.isna(variacao)
                    else round(
                        float(variacao),
                        3
                    )
                )
            }
        )

    # -----------------------------------------------------
    # RESPOSTA
    # -----------------------------------------------------

    return {

        "codigo_fipe":
            codigo_fipe,

        "marca":
            str(ultimo["nome_marca"]),

        "modelo":
            str(ultimo["nome_modelo"]),

        "ano_modelo":
            int(ano_modelo),

        "total_registros":
            len(serie),

        "periodo": {

            "inicio":
                historico[
                    "data_referencia"
                ]
                .min()
                .strftime("%Y-%m-%d"),

            "fim":
                historico[
                    "data_referencia"
                ]
                .max()
                .strftime("%Y-%m-%d")
        },

        "historico":
            serie
    }

@app.get("/debug/features")
def debug_features(
    codigo_fipe: str,
    ano_modelo: int
):

    codigo_fipe = codigo_fipe.strip()

    historico = (
        df_fipe[
            (df_fipe["codigo_fipe"] == codigo_fipe)
            &
            (df_fipe["ano_modelo"] == ano_modelo)
        ]
        .sort_values("data_referencia")
        .copy()
    )

    if historico.empty:
        raise HTTPException(
            status_code=404,
            detail="Veículo não encontrado."
        )

    # -----------------------------------------
    # Variações
    # -----------------------------------------

    historico["variacao_mensal_pct"] = (
        historico["preco"]
        .pct_change(fill_method=None)
        .mul(100)
    )

    historico["variacao_3m_pct"] = (
        historico["preco"]
        .pct_change(
            periods=3,
            fill_method=None
        )
        .mul(100)
    )

    historico["variacao_6m_pct"] = (
        historico["preco"]
        .pct_change(
            periods=6,
            fill_method=None
        )
        .mul(100)
    )

    historico["variacao_12m_pct"] = (
        historico["preco"]
        .pct_change(
            periods=12,
            fill_method=None
        )
        .mul(100)
    )

    # -----------------------------------------
    # Momentum
    # -----------------------------------------

    historico["momentum_1_3"] = (
        historico["variacao_mensal_pct"]
        - historico["variacao_3m_pct"] / 3
    )

    historico["momentum_3_6"] = (
        historico["variacao_3m_pct"] / 3
        - historico["variacao_6m_pct"] / 6
    )

    # -----------------------------------------
    # Volatilidade
    # -----------------------------------------

    historico["volatilidade_6m"] = (
        historico["variacao_mensal_pct"]
        .rolling(6)
        .std()
    )
    # -----------------------------------------
    # Média móvel
    # -----------------------------------------

    media_6m = (
        historico["preco"]
        .rolling(6)
        .mean()
    )

    historico["dist_media_6m_pct"] = (
        (
            historico["preco"]
            - media_6m
        )
        / media_6m
        * 100
    )

    # -----------------------------------------
    # Features temporais
    # -----------------------------------------

    historico["mes"] = (
        historico["data_referencia"]
        .dt.month
    )

    historico["idade_veiculo"] = (
        historico["data_referencia"]
        .dt.year
        - historico["ano_modelo"]
    )

    # -----------------------------------------
    # Último registro
    # -----------------------------------------

    ultimo = historico.iloc[-1]

    features_modelo = {}

    for feature in metadata["features"]:

        valor = ultimo[feature]

        if pd.isna(valor):
            features_modelo[feature] = None

        elif hasattr(valor, "item"):
            features_modelo[feature] = valor.item()

        else:
            features_modelo[feature] = valor

    return {
        "codigo_fipe": codigo_fipe,
        "ano_modelo": ano_modelo,
        "data_referencia": (
            ultimo["data_referencia"]
            .strftime("%Y-%m-%d")
        ),
        "features": features_modelo
    }


# =========================================================
# 12. PREVISÃO
# =========================================================

@app.post("/predict")
def predict(request: VeiculoRequest):

    # -----------------------------------------------------
    # Normalizar código FIPE
    # -----------------------------------------------------

    codigo_fipe = (
        request.codigo_fipe
        .strip()
    )


    # -----------------------------------------------------
    # Buscar histórico
    # -----------------------------------------------------

    historico = (
        df_fipe[
            (
                df_fipe["codigo_fipe"]
                == codigo_fipe
            )
            &
            (
                df_fipe["ano_modelo"]
                == request.ano_modelo
            )
        ]
        .sort_values(
            "data_referencia"
        )
        .copy()
    )


    # -----------------------------------------------------
    # Verificar existência
    # -----------------------------------------------------

    if historico.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                "Veículo não encontrado "
                "na base FIPE."
            )
        )


    # -----------------------------------------------------
    # Verificar histórico mínimo
    # -----------------------------------------------------

    if len(historico) < 13:

        raise HTTPException(
            status_code=422,
            detail=(
                "Histórico insuficiente "
                "para realizar a previsão."
            )
        )


    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================


    # -----------------------------------------------------
    # Variação mensal
    # -----------------------------------------------------

    historico["variacao_mensal_pct"] = (
        historico["preco"]
        .pct_change(
            fill_method=None
        )
        .mul(100)
    )


    # -----------------------------------------------------
    # Variação 3 meses
    # -----------------------------------------------------

    historico["variacao_3m_pct"] = (
        historico["preco"]
        .pct_change(
            periods=3,
            fill_method=None
        )
        .mul(100)
    )


    # -----------------------------------------------------
    # Variação 6 meses
    # -----------------------------------------------------

    historico["variacao_6m_pct"] = (
        historico["preco"]
        .pct_change(
            periods=6,
            fill_method=None
        )
        .mul(100)
    )


    # -----------------------------------------------------
    # Variação 12 meses
    # -----------------------------------------------------

    historico["variacao_12m_pct"] = (
        historico["preco"]
        .pct_change(
            periods=12,
            fill_method=None
        )
        .mul(100)
    )


    # -----------------------------------------------------
    # Momentum 1 x 3 meses
    # -----------------------------------------------------

    historico["momentum_1_3"] = (
        historico["variacao_mensal_pct"]
        -
        historico["variacao_3m_pct"]
    )


    # -----------------------------------------------------
    # Momentum 3 x 6 meses
    # -----------------------------------------------------

    historico["momentum_3_6"] = (
        historico["variacao_3m_pct"]
        -
        historico["variacao_6m_pct"]
    )


    # -----------------------------------------------------
    # Volatilidade 6 meses
    # -----------------------------------------------------

    historico["volatilidade_6m"] = (
        historico[
            "variacao_mensal_pct"
        ]
        .rolling(6)
        .std()
    )


    # -----------------------------------------------------
    # Média móvel 6 meses
    # -----------------------------------------------------

    media_6m = (
        historico["preco"]
        .rolling(6)
        .mean()
    )


    # -----------------------------------------------------
    # Distância para média de 6 meses
    # -----------------------------------------------------

    historico["dist_media_6m_pct"] = (
        (
            historico["preco"]
            - media_6m
        )
        /
        media_6m
        *
        100
    )


    # -----------------------------------------------------
    # Mês
    # -----------------------------------------------------

    historico["mes"] = (
        historico[
            "data_referencia"
        ]
        .dt.month
    )


    # -----------------------------------------------------
    # Idade do veículo
    # -----------------------------------------------------

    historico["idade_veiculo"] = (
        historico[
            "data_referencia"
        ]
        .dt.year
        -
        historico[
            "ano_modelo"
        ]
    )


    # =====================================================
    # ÚLTIMO REGISTRO DISPONÍVEL
    # =====================================================

    ultimo = historico.iloc[-1]


    # =====================================================
    # FEATURES DO MODELO
    # =====================================================

    features = metadata["features"]


    dados_modelo = pd.DataFrame(
        [
            {
                feature: ultimo[feature]
                for feature in features
            }
        ]
    )


    # =====================================================
    # VERIFICAR FEATURES AUSENTES
    # =====================================================

    if dados_modelo.isnull().any().any():

        faltantes = (
            dados_modelo
            .columns[
                dados_modelo
                .isnull()
                .any()
            ]
            .tolist()
        )

        raise HTTPException(
            status_code=422,
            detail={
                "erro": (
                    "Não foi possível calcular "
                    "todas as features."
                ),
                "features_faltantes":
                    faltantes
            }
        )


    # =====================================================
    # PREVISÃO DO MODELO
    # =====================================================

    try:

        variacao_prevista = float(
            modelo.predict(
                dados_modelo
            )[0]
        )

    except Exception as erro:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Erro ao executar "
                f"o modelo: {erro}"
            )
        )


    # =====================================================
    # PREÇO ATUAL
    # =====================================================

    preco_atual = float(
        ultimo["preco"]
    )


    # =====================================================
    # PREÇO PREVISTO
    # =====================================================

    preco_previsto = (
        preco_atual
        *
        (
            1
            +
            variacao_prevista
            / 100
        )
    )


    # =====================================================
    # TENDÊNCIA
    # =====================================================

    if variacao_prevista > 0.5:

        tendencia = "alta"

    elif variacao_prevista < -0.5:

        tendencia = "queda"

    else:

        tendencia = "estável"


    # =====================================================
    # RESPOSTA
    # =====================================================

    return {

        "codigo_fipe":
            codigo_fipe,

        "marca":
            str(
                ultimo["nome_marca"]
            ),

        "modelo":
            str(
                ultimo["nome_modelo"]
            ),

        "ano_modelo":
            int(
                request.ano_modelo
            ),

        "data_referencia":
            ultimo[
                "data_referencia"
            ].strftime(
                "%Y-%m-%d"
            ),

        "preco_atual":
            round(
                preco_atual,
                2
            ),

        "variacao_prevista_pct":
            round(
                variacao_prevista,
                3
            ),

        "preco_previsto":
            round(
                preco_previsto,
                2
            ),

        "tendencia":
            tendencia,

        "modelo_ml":
            (
                f'{metadata["nome"]} '
                f'v{metadata["versao"]}'
            )
    }