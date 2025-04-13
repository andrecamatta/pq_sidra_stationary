# Processador de Estacionariedade de Dados do SIDRA

Este projeto contém scripts Python para extrair dados do Sistema IBGE de Recuperação Automática (SIDRA) e processá-los para análise de estacionariedade.

## Descrição

O objetivo principal é automatizar o processo de obtenção de séries temporais específicas do SIDRA e aplicar testes de estacionariedade, salvando os resultados.

## Funcionalidades

*   **Extração de Dados:** O script `sidra_data_extractor.py` (suposição, nome baseado no padrão) é responsável por conectar-se à API do SIDRA e baixar os dados necessários, salvando-os em `sidra_data.xlsx`.
*   **Processamento de Estacionariedade:** O script `stationarity_processor.py` carrega os dados extraídos (ou pré-existentes em `sidra_data.xlsx`), aplica testes de estacionariedade (como ADF, KPSS - suposição) e salva os resultados em `stationary_data.xlsx`.

## Requisitos

*   Python 3.x
*   Dependências listadas em `pyproject.toml` (gerenciadas com `uv`)

## Instalação

1.  Clone o repositório:
    ```bash
    git clone https://github.com/andrecamatta/pq_sidra_stationary.git
    cd pq_sidra_stationary
    ```
2.  Instale as dependências (assumindo o uso de `uv`):
    ```bash
    uv sync
    ```
    *(Se você usar outro gerenciador como pip ou conda, ajuste o comando)*

## Uso

1.  **Extrair Dados (se necessário):**
    ```bash
    python sidra_data_extractor.py
    ```
    *(Este passo pode não ser necessário se `sidra_data.xlsx` já contiver os dados desejados)*

2.  **Processar Estacionariedade:**
    ```bash
    python stationarity_processor.py
    ```
    Os resultados serão salvos em `stationary_data.xlsx`.

## Dados

*   `sidra_data.xlsx`: Contém os dados brutos extraídos do SIDRA.
*   `stationary_data.xlsx`: Contém os resultados da análise de estacionariedade.

*(Este é um README inicial. Sinta-se à vontade para adicionar mais detalhes sobre as tabelas específicas do SIDRA, os testes de estacionariedade utilizados, configurações, etc.)*
