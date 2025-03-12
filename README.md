# Detecção de Fraude em Cartões de Crédito com Machine Learning

Este projeto utiliza Machine Learning para detectar transações fraudulentas em cartões de crédito. O modelo é treinado em um conjunto de dados altamente desbalanceado, onde a grande maioria das transações é legítima e uma pequena fração é fraudulenta.

## Visão Geral

O script Python neste repositório executa as seguintes etapas:

1.  **Obtenção dos Dados:**
    *   Utiliza a API do Kaggle para baixar o dataset "Credit Card Fraud Detection" ([https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)).  Este dataset contém transações de cartões de crédito, com as colunas 'Time', 'V1'...'V28' (componentes principais obtidos por PCA, para anonimizar os dados), 'Amount' (valor da transação) e 'Class' (0 para transações legítimas, 1 para fraudulentas).
    * É preciso ter o arquivo `kaggle.json` em um diretório reconhecido, ou autenticar previamente.

2.  **Análise Exploratória de Dados (EDA):**
    *   Gera um gráfico de barras mostrando a distribuição das classes (transações legítimas vs. fraudulentas). Isso ilustra o desbalanceamento do dataset.

3.  **Pré-processamento dos Dados:**
    *   **Normalização:** Normaliza a coluna 'Amount' (valor da transação) usando `StandardScaler`. Isso coloca os valores em uma escala comum, o que é importante para algoritmos como a Regressão Logística.

4.  **Balanceamento de Classes:**
    *   Utiliza a técnica SMOTE (Synthetic Minority Oversampling Technique) para lidar com o desbalanceamento de classes. O SMOTE cria novas amostras sintéticas da classe minoritária (fraudes), aumentando o número de exemplos de fraude e melhorando a capacidade do modelo de aprender a identificar esses casos.

5.  **Divisão dos Dados:**
    *   Divide os dados (após o balanceamento) em conjuntos de treinamento e teste usando `train_test_split`. O conjunto de treinamento é usado para treinar o modelo, e o conjunto de teste é usado para avaliar o desempenho do modelo em dados não vistos.

6.  **Modelagem (Regressão Logística):**
    *   Cria e treina um modelo de Regressão Logística (`LogisticRegression`) usando os dados de treinamento. A Regressão Logística é um algoritmo de classificação que estima a probabilidade de uma transação ser fraudulenta.

7.  **Avaliação do Modelo:**
    *   **Matriz de Confusão:** Gera uma matriz de confusão que mostra o número de Verdadeiros Positivos (VP), Verdadeiros Negativos (VN), Falsos Positivos (FP) e Falsos Negativos (FN).
    *   **Relatório de Classificação:** Calcula e exibe métricas de avaliação como precisão, recall (revocação), F1-score e suporte para cada classe.
    *   **Visualização da Matriz de Confusão:** Apresenta a matriz de confusão em um mapa de calor (`heatmap`) para facilitar a interpretação.

8. **Otimização de Hiperparâmetros (Grid Search):**
    *   Realiza uma busca em grade (`GridSearchCV`) para encontrar os melhores hiperparâmetros para o modelo de Regressão Logística.
    *   O hiperparâmetro `C` (parâmetro de regularização) é otimizado.
    *   Utiliza validação cruzada (`cv=5`) para avaliar o desempenho de cada combinação de hiperparâmetros de forma mais robusta.
    * Seleciona o melhor estimador, com a melhor combinação de parâmetros.

9.  **Salvamento do Modelo:**
    *   Salva o melhor modelo treinado (após a otimização) em um arquivo `modelo_detecao_fraude.pkl` usando a biblioteca `joblib`. Isso permite que o modelo seja carregado e reutilizado posteriormente sem precisar ser treinado novamente.

## Requisitos

*   Python 3.6+
*   Bibliotecas Python:
    *   `pandas` (para manipulação de dados)
    *   `matplotlib` (para visualizações)
    *   `seaborn` (para visualizações)
    *   `scikit-learn` (para Machine Learning)
    *   `imblearn` (para balanceamento de classes, especificamente SMOTE)
    *   `kaggle` (para baixar o dataset)
    *   `joblib` (para salvar o modelo)

## Instalação

1.  **Clone este repositório:**

    ```bash
    git clone <https://github.com/juliaSchaedler/Deteccao_fraude.git>
    cd <nome_da_pasta_do_repositório>
    ```

2.  **Crie um ambiente virtual (recomendado):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate  # Windows
    ```

3.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure a API do Kaggle:**
   *  Tenha uma conta no Kaggle.
   *  Vá para a página da sua conta ("My Account").
   *  Na seção "API", clique em "Create New API Token". Isso baixará um arquivo `kaggle.json`.
   *  Coloque o arquivo `kaggle.json` em um dos seguintes locais:
      *  `~/.kaggle/kaggle.json` (Linux/macOS)
      *  `C:\Users\<seu_usuario>\.kaggle\kaggle.json` (Windows)
     Ou, antes de executar o script, rode:
      ```bash
      kaggle.api.authenticate()
      ```

5.  **Execute o script:**

    ```bash
    python main.py
    ```

## Saída

O script:

*   Exibe um gráfico da distribuição das classes (transações legítimas vs. fraudulentas).
*   Exibe uma matriz de confusão visualizada como um mapa de calor.
*   Imprime um relatório de classificação com métricas de avaliação.
*   Salva o modelo treinado no arquivo `modelo_detecao_fraude.pkl`.

## Considerações Importantes

*   **Desbalanceamento de Classes:** O dataset é altamente desbalanceado, o que é um desafio comum em problemas de detecção de fraude. O uso do SMOTE ajuda a mitigar esse problema.
*   **Interpretabilidade:** A Regressão Logística é um modelo relativamente interpretável.  É possível analisar os coeficientes do modelo para entender quais variáveis são mais importantes para a detecção de fraude (embora as variáveis `V1` a `V28` sejam anonimizadas).
*   **Métricas de Avaliação:**  Em problemas de detecção de fraude, o *recall* (capacidade de identificar *todos* os casos de fraude) é frequentemente mais importante do que a precisão.  Um falso negativo (classificar uma transação fraudulenta como legítima) pode ter consequências mais graves do que um falso positivo (classificar uma transação legítima como fraudulenta).  Observe o relatório de classificação para avaliar o recall.
*   **Dados Anonimizados:** As variáveis `V1` a `V28` são componentes principais obtidos por PCA.  Isso significa que não temos informações sobre o significado original dessas variáveis.
* **Outros Modelos:** Experimente outros modelos de classificação (como Random Forest, Gradient Boosting, Redes Neurais) para comparar o desempenho.
* **Validação Cruzada:** Para uma avaliação mais robusta do desempenho, você deve considerar o uso de técnicas como validação cruzada estratificada repetida. O código já utiliza a validação cruzada, mas apenas para a busca de hiperparâmetros.
* **Dados em Tempo Real:** Em um cenário real, a detecção de fraude é feita em tempo real.  Este projeto fornece um modelo treinado que poderia ser integrado a um sistema de processamento de transações para avaliar o risco de fraude em tempo real.

## Contribuições

Contribuições são bem-vindas! Se você encontrar bugs, tiver sugestões ou quiser adicionar novas funcionalidades (por exemplo, implementar outros algoritmos, melhorar a visualização, adicionar análise de importância de variáveis), sinta-se à vontade para abrir um *issue* ou enviar um *pull request*.
