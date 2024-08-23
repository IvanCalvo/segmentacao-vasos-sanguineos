# Comparação de Redes Neurais na Segmentação de Vasos Sanguíneos em Imagens Médicas

### Autor: Ivan Duarte Calvo
### Orientador: Prof. Dr. Cesar H. Comin

Esse é o repositório com os _notebooks_ e gráficos gerados no desenvolvimento do Trabalho de Conclusão de Curso realizado no Departamento da Computação da Universidade Federeal de São Carlos no curso de Bacharelado em Ciência da Compuação.

## Notebooks

No repositório há notebooks utilizados para preparar o _dataset_, treinar os modelos, testar _checkpoints_ e criar _plots_ para visualização dos resultados. Todos eles estão presentes na pasta `src/code`

- `Normalização do dataset.ipynb`: Notebook utilizado para calcular os valores necessários para a normalização do conjunto de dados.
- `Run.ipynb`: Notebook utilizado para o treinamento e teste de um modelo.
- `Testes.ipynb`: Notebook utilizado para carregar e testar um _checkpoint_ criado no notebook `Run.ipynb`.
- `Plots.ipynb`: Notebook utilizado para criar os _plots_. Os dados estão na pasta `data/runsDataTorchseg`

## Referências 

Curso de Visão Computacional do Prof. Dr. Cesar Comin, presente no repositório https://github.com/chcomin/curso-visao-computacional-2024/

Biblioteca _TorchSeg_ disponível em: https://github.com/isaaccorley/torchseg

_TensorBoard_ disponível em: https://github.com/tensorflow/tensorboard/

_clDice_: https://github.com/jocpae/clDice