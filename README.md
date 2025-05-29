# RECONHECIMENTO DE SINAIS DINÂMICOS DA LÍNGUA BRASILEIRA DE SINAIS POR MEIO DE REDES NEURAIS CONVOLUCIONAIS

---

## ✅ Requisitos

- **Python 3.10.x**
  - Esta foi a versão utilizada em todo o desenvolvimento. Recomenda-se fortemente manter essa versão para garantir compatibilidade com bibliotecas como `torch`, `tensorflow`, `opencv` e `imageio`.

- **Sistema Operacional**
  - Recomenda-se Linux (Ubuntu ou derivados.

- **Outros**
  - Git
  - pyenv para gerenciar versões do Python

---

## 🚀 Instalação

### 1. Configure o Python 3.10

Usando pyenv:

pyenv install 3.10.12
pyenv local 3.10.12

Verifique a versão:

python --version  # Deve retornar Python 3.10.x

### 2. Crie e ative o ambiente virtual

python -m venv venv
source venv/bin/activate

### 3. Instale as dependências

pip install --upgrade pip
pip install -e .

---

## 📁 Estrutura do Projeto

- `data/raw_videos/` → Vídeos brutos da base de dados (diretamente das fontes `vlibras` e `ines_gov`)
- `data/numpy_data/` → Arquivos `.npy` com os frames tratados, redimensionados e normalizados
- `data/split/` → Dados organizados em `train/`, `val/` e `test/` para os experimentos
- `notebooks/` → Notebooks de treinamento das CNNs e do pipeline de pré-processamento (executáveis manualmente)
- `reports/` → Relatórios gerados pelas CNNs com métricas por classe e matrizes de confusão
- `saved_models/` → Arquivos `.pth` com os pesos salvos de cada arquitetura testada
- `src/models/` → Arquiteturas CNNs customizadas.  
  Contém os **códigos completos das redes neurais**, com todas as **configurações específicas**, camadas, métodos de avaliação e geração de relatórios.
- `src/preprocess/` → Pipeline completo de pré-processamento.  
  Inclui os **módulos responsáveis por carregar vídeos, extrair e tratar frames**, aplicar normalização, redimensionamento e salvar os dados em formato `.npy`.  Ali também estão definidos todos os **parâmetros utilizados**, além de um script auxiliar para execução rápida (`run_preprocessing.py`).
- `src/preprocess/run_preprocessing.py` → Script rápido para execução do pré-processamento completo (versão não visual, usada durante o desenvolvimento)
- `requirements.txt` → Lista de bibliotecas necessárias
- `setup.py` → Script de instalação e configuração do projeto como pacote Python

---

## ⚙️ Como utilizar

### 1. Pré-processamento

Os dados já estão pré-processados.  
Para repetir o processo:

1. Apague o **somente o conteúdo, não o direotiro** de:
   - data/numpy_data/
   - data/split/train/, val/, test/
2. Execute notebooks/preprocess.ipynb

---

### 2. Treinar os Modelos

Use os notebooks:

- alexnet_test.ipynb
- densenet_test.ipynb
- inceptionv3_test.ipynb
- resnet_test.ipynb
- vgg16net_test.ipynb

Eles geram:

- Modelos treinados e salvos em `saved_models/`
- Relatórios de desempenho e matrizes de confusão salvos em `notebooks/`  
  (podem ser movidos manualmente para `reports/` para organização)

> ⚠️ Caso deseje repetir os testes ou treinar novamente os modelos, **apague apenas os arquivos `.pth` em `saved_models/** Não é necessário apagar os diretórios.**

---

## 📊 Relatórios

Em `reports/`, estão organizados:

- Relatórios de desempenho por arquitetura (em `.txt`)
- Matrizes de confusão dos conjuntos de validação e teste (em `.png`)
- Todos os arquivos presentes correspondem à **última rodada de testes** executada por mim

> ⚠️ Caso execute novos testes, os relatórios e imagens serão gerados novamente nos `notebooks/`  
> Você pode movê-los manualmente para `reports/` para manter a organização.

