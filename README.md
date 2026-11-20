# ILT Data Extraction & IAT — Linux

Versão **1.1.0** (estável). Interface gráfica para **treinar** uma rede neural
convolucional (ConvNeXt-Tiny), **classificar** grandes conjuntos de imagens e
**revisar/rotular** os resultados no IAT (Interactive Analysis Tool).

Plataformas: Windows 11 · Ubuntu 24.04 · Ubuntu 26.04 · Python 3.11/3.12 · PyTorch 2.5.1 (2.7.1 em GPUs RTX 50xx).

---

## 1. Instalação e primeira execução

| Sistema | Instalar (uma vez) | Abrir |
|---|---|---|
| Windows 11 | `INSTALAR.bat` | `ILT.bat` (ou atalho na Área de Trabalho) |
| Ubuntu 24.04 / 26.04 | `bash START_HERE_LINUX.sh` | `./run_gui.sh` |

O instalador cria um ambiente isolado (`.venv`) na própria pasta e **detecta a
placa de vídeo** (`src/gpu_profile.py`) para escolher o PyTorch:

| GPU detectada | Perfil | PyTorch |
|---|---|---|
| NVIDIA Maxwell/Pascal/Volta/Turing (GTX 9xx–16xx, RTX 20xx, Quadro P4000…) | `cu118` | 2.5.1 + CUDA 11.8 |
| NVIDIA Ampere/Ada/Hopper (RTX 30xx/40xx, A-series…) | `cu124` | 2.5.1 + CUDA 12.4 |
| NVIDIA Blackwell (RTX 50xx) | `cu128` | 2.7.1 + CUDA 12.8 |
| Sem GPU NVIDIA, GPU muito antiga ou driver antigo | `cpu` | 2.5.1 CPU |

Depois da instalação há um teste real na GPU; se ela não funcionar com o perfil
escolhido, o instalador passa automaticamente para CPU. Para forçar um perfil:
`ILT_PROFILE=cpu|cu118|cu124|cu128`. Basta o driver NVIDIA; o CUDA vem dentro do
PyTorch. Trocou de placa ou atualizou o driver: rode o instalador de novo.

Se não houver Python 3.11/3.12 com Tkinter, o instalador usa um Python 3.12
local (Windows: python.org na pasta `.python312`; Ubuntu 26.04: Miniforge na
pasta do ILT), sem alterar o Python do sistema. Versões de todas as
dependências são fixadas (`requirements-common.txt`). `pyswisseph` (ciclo
lunar) é opcional.

A interface padrão é web (Dash/Plotly) e roda só no próprio computador
(`127.0.0.1`), abrindo sozinha no navegador. Para encerrar, use o botão
**Encerrar** na barra lateral. A interface clássica (Tk) continua disponível:
`ILT_classico.bat` / `./run_gui.sh --classic`.

Diagnóstico: `DIAGNOSTICO.bat` (Windows) ou `./diagnose_linux.sh` (Ubuntu).

---

## 2. Fluxo de trabalho

### Treino (`src/train.py`)
- Entrada: pasta já dividida em `train/valid/test/<classe>/` **ou** uma pasta
  apenas com `<classe>/imagens`. No segundo caso a divisão é feita por
  percentual ou número de imagens, sem alterar o dataset original
  (`symlink` → `hardlink` → cópia).
- Classes podem ser excluídas, renomeadas ou agrupadas (mesmo nome final).
- Saídas: `model_<projeto>`, `labels_<projeto>.json`, `stats_<projeto>.csv`,
  `matrix_<projeto>.csv`, `training_config_<projeto>.json`,
  `dataset_split_<projeto>.csv` e os artefatos de avaliação do conjunto de teste.
- Retomada (`Resume = T`) é bloqueada se classes ou métrica de validação mudarem.
- Se já existir `model_<projeto>` na pasta de saída, a interface pergunta se
  deve sobrescrever (ou retomar, quando há checkpoint parcial) antes de iniciar;
  o worker nunca espera resposta no console.
- Métricas macro (recall, precisão, F1) são sempre a média sobre as classes
  reais do modelo.
- **Receita de treino** (card próprio na aba): normalização ImageNet, imagem
  completada em quadrado sem esticar, augmentation (rotação/espelho/brilho),
  AdamW com warmup e redução no platô, LR 10× na cabeça, label smoothing. O
  botão "Receita legada (1.0)" restaura o comportamento antigo para comparação.
  A receita fica gravada no modelo e a classificação reaplica o mesmo
  pré-processamento (modelos antigos continuam classificados como antes).
  Detalhes e referências em `docs/METRICAS_E_REFERENCIAS.md`, seção 7.

### Classificação (`src/main.py`)
- Varre a pasta de entrada recursivamente, divide em batches (padrão 8 000
  imagens), extrai embeddings e predições top-3 e projeta com openTSNE em um
  referencial comum ao batch base.
- Saída do projeto: `dataframes/`, `embeddings/`, `images/`, `complete.csv`,
  `project_settings.json` e, opcionalmente, `clusters/` (clustering por classe).
- **Modo auditoria:** usa o rótulo da pasta de origem como rótulo inicial,
  mantendo a predição do modelo em colunas separadas.

### IAT
Projeção/Scatter, Galeria & Rotulagem, Análise Temporal e Espacial,
Clustering, Treino + Classificação integrado, Segmentação, Exportação e Logs.
Os rótulos manuais ficam em `manual_label`; `source_label` preserva a origem e
`preds1..3` a previsão do modelo. A coluna `var_norm` guarda a **incerteza da
predição** `1 − (top-1 − top-2)` (0 = segura, 1 = empate), usada nos filtros da
Galeria e na prioridade de revisão.

As medidas da Segmentação (perímetro de Crofton, Feret por *rotating calipers*,
ESD, eixos da elipse de momentos, biovolume, estatísticas de intensidade) e a
escala de densidade comum da Análise Temporal estão definidas, com as
referências, em [`docs/METRICAS_E_REFERENCIAS.md`](docs/METRICAS_E_REFERENCIAS.md).

Recursos de visualização da interface web: miniatura ao passar o mouse na
Projeção; na Análise Temporal, projeções alternativas (PCA, t-SNE, UMAP,
UMAP supervisionado, PaCMAP), gráfico de evolução e diversidade por intervalo
e janela de exportação completa (formatos, dpi, tamanho em cm, cores, fontes,
um arquivo por intervalo, GIF); o painel **Origem → Previsão** (alluvial/Sankey
com duas ou três colunas entre pasta de origem, classe prevista e rótulo
revisado); "Agrupar / renomear classes" com matriz de sugestões, valendo para
todos os gráficos da Análise Temporal; na Galeria, imagem com zoom e barra de
escala calculada pelo pixel size; no Clustering, clique no nó do dendrograma para
cortar e grupos com imagens representativas (ver, rotular, abrir na Galeria);
na Segmentação, prévia ao vivo original | máscara | sobreposição com
histograma e limiar; em Exportar, pacote para o **EcoTaxa** (TSV + imagens).

### Pixel size
Calibração por projeto em **µm/pixel** (padrão 12.87; aceita vírgula ou ponto).
No treino é apenas metadado; na classificação e no IAT é salva em
`project_settings.json` e usada nas métricas físicas da segmentação. Na
classificação vale o valor digitado: o pixel size do treino do modelo só
preenche o campo ao escolher a pasta do modelo, e apenas se o campo ainda
estiver no padrão. No ciclo integrado do IAT, o novo modelo recebe o pixel size
do projeto.

---

## 3. Desempenho e uso de hardware

Os ajustes abaixo são automáticos; todos podem ser sobrescritos por variável de
ambiente antes de `./run_gui.sh`.

| Recurso | Comportamento padrão | Variável |
|---|---|---|
| Workers do DataLoader no treino | `-1` = automático (CPUs − 2, máx. 8) | campo *Num. workers* |
| Workers na extração de features | automático (CPUs − 2, máx. 8) | `ILT_INFER_WORKERS` |
| Tamanho do lote de inferência | `auto` pela VRAM livre (128 / 256 / 512; reduzido pela metade em falta de memória) | `ILT_INFER_BATCH_SIZE` |
| Tamanho do lote de treino | campo *Batch size* (`auto` = pela VRAM; reduzido em falta de memória) | — |
| Precisão mista | bfloat16 em GPUs CC ≥ 8.0 (RTX 30xx+), float16 + *loss scaling* nas anteriores | `ILT_AMP_DTYPE=bf16/fp16` |
| `torch.compile` | desligado (opcional; primeira época mais lenta; Linux) | `ILT_TORCH_COMPILE=1` |
| Threads do t-SNE / kNN (CPU) | CPUs − 1 | `ILT_TSNE_JOBS` |
| kNN e PCA da análise estrutural | na GPU (torch) a partir de 3 000 imagens; CPU abaixo disso ou sem CUDA | `ILT_IAT_GPU=0` |
| UMAP / PaCMAP | todos os núcleos a partir de 20 000 imagens (sem semente fixa); determinístico abaixo | `ILT_UMAP_DETERMINISTIC=1` |
| Segmentação em lote | CPUs − 1 threads (máx. 8) | `ILT_SEG_WORKERS` |
| Layout `channels_last` | ativo só em GPUs com Tensor Cores (CC ≥ 7.0) | `ILT_CHANNELS_LAST=0/1` |
| Cache de embeddings no clustering | todos os batches, até 30 % da RAM livre (máx. 12 GiB) | `ILT_CLUSTER_EMBED_CACHE=<n>` |
| Miniaturas da galeria (web) | carregadas só quando visíveis; cache do navegador 7 dias com `ETag`; 6 000 em memória | `ILT_IAT_THUMB_CACHE` |
| Miniaturas em voo na galeria (clássica) | 8 | `ILT_IAT_THUMB_INFLIGHT` |
| Threads de cópia/links | 6 | `ILT_FILE_WORKERS` |
| Limite de arquivos abertos (Linux) | elevado para 65 536 na classificação e na interface, sem passar do máximo do sistema | — |

Precisão mista (AMP) é usada automaticamente em GPU. Com `seed ≠ -1` o treino
continua determinístico (cuDNN *benchmark* só é ligado com `seed = -1`).
Justificativas e referências dos ajustes de desempenho:
`docs/METRICAS_E_REFERENCIAS.md`, seção 8.

---

## 4. Estrutura do código (`src/`)

| Camada | Módulos |
|---|---|
| Workers científicos | `train.py`, `main.py`, `batch.py`, `features.py`, `projections.py`, `data.py`, `class_clustering.py` |
| Runtime de hardware | `torch_runtime.py` (GPU/cuDNN/DataLoader), `hw_resources.py` (CPU/RAM, sem torch) |
| Contratos e serviços | `ilt_contracts.py`, `ilt_clustering.py`, `project_settings.py`, `training_dataset.py`, `training_evaluation.py`, `training_export.py`, `project_model_cycle.py`, `platform_utils.py`, `aux.py` |
| Dashboard | `ilt_gui.py`, `dashboard_execution_mixin.py`, `dashboard_widgets.py`, `training_dataset_ui_mixin.py`, `training_results.py`, `mascot_games_mixin.py` |
| IAT | `iat_workspace.py`, `iat_backend.py`, `iat_widgets.py`, `iat_gallery_mixin.py`, `iat_export_mixin.py`, `iat_training_export_mixin.py`, `iat_model_cycle_mixin.py`, `iat_io.py`, `iat_filename_filters.py`, `iat_temporal_*.py`, `iat_origin_flow.py` |
| Interface web | `ilt_web.py`, `web/`: `controller.py` e `iat_core.py` executam os mesmos métodos do dashboard/IAT clássicos sem Tk; `pages/` = telas; `theme.py`, `components.py`, `assets/` = layout |
| Tema e textos | `ui_resources.py` (cores, PT/EN), `ui_contrast.py` (contraste WCAG), `segmentation_docs.py` (equações e referências da segmentação) |

Documentação complementar em [`docs/`](docs/): métricas e referências
científicas (`METRICAS_E_REFERENCIAS.md`), análise temporal
(`ANALISE_TEMPORAL_ESPACIAL.md`) e integração com visualizadores
(`VIEWER_INTEGRATION_GUIDE.md`).

---

## 5. Observações

- Nomes de arquivo de imagem devem ser únicos no dataset classificado (são a
  chave entre CSV, imagem e embedding).
- `pyswisseph` (cálculo lunar) usa a Swiss Ephemeris, com licença dupla
  AGPL/profissional — veja [`docs/ANALISE_TEMPORAL_ESPACIAL.md`](docs/ANALISE_TEMPORAL_ESPACIAL.md)
  antes de redistribuir.
- Ao empacotar para outra máquina, não inclua `.venv/`, `.ilt_linux_profile`,
  `output/`, `build/`, `dist/` nem pastas `__pycache__/`.
