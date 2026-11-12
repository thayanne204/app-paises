"""Classificador de países por prioridade de ajuda.

Inteligência Artificial · Unidade III · UNIFSA · Engenharia de Software · 2026.2
Autor(a): Thayanne 
"""
import altair as alt
import joblib
import pandas as pd
import streamlit as st

AUTOR = "Thayanne Oliveira Silva"         
DATA_TREINO = "12/11/2026"
# 1. Configuração da página
st.set_page_config(page_title="Classificação de países", layout="centered")


# 2. Carregamento do modelo (executado uma única vez e mantido em cache)
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_paises.pkl")


pacote = carregar_modelo()
modelo = pacote["modelo"]      # pipeline: StandardScaler -> PCA -> KMeans
colunas = pacote["colunas"]    # ordem das colunas usada no treinamento
nomes = pacote["nomes"]        # nome de cada grupo

escalonador, pca, kmeans = modelo[0], modelo[1], modelo[-1]

# 3. Descrição de cada campo: rótulo exibido e valor inicial (mediana da base)
CAMPOS = {
    "child_mort": ("Mortalidade infantil (mortes por 1.000 nascidos vivos)", 19.3),
    "exports": ("Exportações (% do PIB per capita)", 35.0),
    "health": ("Gasto com saúde (% do PIB per capita)", 6.3),
    "imports": ("Importações (% do PIB per capita)", 43.3),
    "income": ("Renda líquida por pessoa (US$)", 9960.0),
    "inflation": ("Inflação anual (%)", 5.4),
    "life_expec": ("Expectativa de vida (anos)", 73.1),
    "total_fer": ("Filhos por mulher", 2.4),
    "gdpp": ("PIB per capita (US$)", 4660.0),
}
ROTULOS_CURTOS = {
    "child_mort": "Mortalidade infantil",
    "exports": "Exportações",
    "health": "Gasto com saúde",
    "imports": "Importações",
    "income": "Renda por pessoa",
    "inflation": "Inflação",
    "life_expec": "Expectativa de vida",
    "total_fer": "Filhos por mulher",
    "gdpp": "PIB per capita",
}


@st.cache_data
def perfil_dos_grupos():
    """Média aproximada de cada grupo na escala original.

    Parte dos centroides do KMeans (espaço do PCA) e desfaz o PCA e a
    padronização do próprio pipeline.
    """
    originais = escalonador.inverse_transform(pca.inverse_transform(kmeans.cluster_centers_))
    return pd.DataFrame(originais, columns=colunas)


# 4. Interface
st.title("Classificação de países por prioridade de ajuda")
st.write(
    "Aplicativo desenvolvido na disciplina de Inteligência Artificial (UNIFSA) para "
    "estimar, a partir de nove indicadores socioeconômicos e de saúde, a qual grupo "
    "um país mais se assemelha: **Desenvolvidos**, **Em desenvolvimento** ou "
    "**Prioridade de ajuda**."
)
st.caption(f"Autor(a): {AUTOR}")

valores = {}
esquerda, direita = st.columns(2)
for i, coluna in enumerate(colunas):
    rotulo, inicial = CAMPOS[coluna]
    with esquerda if i % 2 == 0 else direita:
        valores[coluna] = st.number_input(rotulo, value=inicial, key=coluna)

# 5. Classificação
if st.button("Classificar", type="primary"):
    entrada = pd.DataFrame([valores])[colunas]   # uma linha, colunas na ordem do treinamento
    grupo = int(modelo.predict(entrada)[0])      # o pipeline padroniza, aplica o PCA e agrupa
    st.success(f"Grupo {grupo}: {nomes[grupo]}")

    # Obrigatória 2: valores informados em tabela
    st.subheader("Valores informados")
    informados = entrada.rename(columns=ROTULOS_CURTOS)
    st.dataframe(informados, hide_index=True, width="stretch")

    # Atividade 1: comparação com a média do grupo
    st.subheader(f"Comparação com o perfil médio do grupo ({nomes[grupo]})")
    media = perfil_dos_grupos().iloc[grupo]
    comparacao = pd.DataFrame({
        "Indicador": [ROTULOS_CURTOS[c] for c in colunas],
        "País informado": entrada.iloc[0].values,
        "Média do grupo": media.values,
    })
    comparacao["Diferença"] = comparacao["País informado"] - comparacao["Média do grupo"]
    st.dataframe(comparacao.round(2), hide_index=True, width="stretch")
    st.caption(
        "A média do grupo é reconstruída a partir do centroide no espaço do PCA; "
        "é uma aproximação do perfil típico."
    )

    # Atividade 2: país no plano PC1 x PC2 (etapas de padronização e PCA do pipeline)
    st.subheader("Posição no plano PC1 × PC2")
    pais = modelo[:-1].transform(entrada)[0]          # 5 componentes; usamos só os 2 primeiros
    centros = kmeans.cluster_centers_
    var = pca.explained_variance_ratio_
    pontos = pd.DataFrame({
        "PC1": list(centros[:, 0]) + [pais[0]],
        "PC2": list(centros[:, 1]) + [pais[1]],
        "Ponto": [f"Centro: {nomes[i]}" for i in range(len(centros))] + ["País informado"],
        "Tipo": ["Centro do grupo"] * len(centros) + ["País informado"],
    })
    grafico = (
        alt.Chart(pontos)
        .mark_point(size=200, filled=True)
        .encode(
            x=alt.X("PC1:Q", title=f"PC1 ({var[0]:.1%} da variância)"),
            y=alt.Y("PC2:Q", title=f"PC2 ({var[1]:.1%} da variância)"),
            color=alt.Color("Ponto:N", title=None),
            shape=alt.Shape("Tipo:N", title=None),
            tooltip=["Ponto", alt.Tooltip("PC1:Q", format=".2f"), alt.Tooltip("PC2:Q", format=".2f")],
        )
        .properties(height=400)
    )
    st.altair_chart(grafico, width="stretch")
    st.caption(
        f"O PCA do pipeline tem {pca.n_components_} componentes; "
        "apenas PC1 e PC2 estão representados."
    )

    # Obrigatória 3: limitações e uso responsável
    st.warning(
        "**Limitações e uso responsável.** O modelo agrupa países por semelhança estatística "
        "com base em apenas nove indicadores e em dados de um único período; o grupo atribuído "
        "não é um diagnóstico nem uma avaliação definitiva do país. O resultado não deve ser "
        "usado isoladamente para decidir sobre ajuda financeira, políticas públicas ou "
        "rankings. Utilize-o como apoio, junto com outras fontes e com análise humana."
    )

# Atividade 3: Sobre o modelo (versão inicial do model card)
st.divider()
st.header("Sobre o modelo")
st.markdown(
    f"""
- **Origem dos dados:** base *Country-data* (HELP International, Kaggle), com nove
  indicadores socioeconômicos e de saúde por país.
- **Data de treinamento:** {DATA_TREINO}
- **Pipeline:** padronização (StandardScaler) → PCA → agrupamento (KMeans)
- **Componentes do PCA:** {pca.n_components_} (explicam
  {pca.explained_variance_ratio_.sum():.1%} da variância)
- **Número de grupos:** {kmeans.n_clusters} ({", ".join(nomes[i] for i in sorted(nomes))})
- **Uso pretendido:** fins educacionais.
- **Limitações:** ver o aviso exibido após a classificação.
"""
)
