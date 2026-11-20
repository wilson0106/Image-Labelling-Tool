# Métricas, fórmulas e referências científicas (ILT 1.1.0)

Este documento descreve exatamente o que o código calcula nas partes revisadas
em 2026-11-12 e as referências que fundamentam cada escolha. Os itens seguem a
numeração da avaliação técnica (itens 8, 24, 28–31, 40 e 48).

---

## 1. Treino — métricas macro (item 8)

**Código:** `src/train.py` (`compute_macro`, `run`).

- Recall, precisão e F1 por classe vêm da matriz de confusão do conjunto de
  validação/teste; os valores *macro* são a **média aritmética simples sobre as
  classes reais do modelo** (`len(labels_<projeto>.json)`).
- O antigo campo "Divisor (macro)" (`numClasses` na configuração) permitia
  dividir as médias por um número arbitrário, escalando as métricas em silêncio.
  O campo foi removido das interfaces (clássica e web). A configuração continua
  aceitando `numClasses` por compatibilidade; se o valor for diferente do número
  real de classes, o `train.py` registra um aviso no log e usa o número real.
- `evaluation_<projeto>.json` (`training_evaluation.py`) continua registrando
  `configured_macro_divisor`, agora sempre igual ao número de classes.

Referência: Sokolova, M. & Lapalme, G. (2009). *A systematic analysis of
performance measures for classification tasks.* Information Processing &
Management, 45(4), 427–437.

---

## 2. Incerteza da predição no IAT (item 24)

**Código:** `src/projections.py` (`margin_uncertainty`), `src/iat_backend.py`
(`ensure_required_columns`).

A coluna `var_norm` (rotulada "Variância" nas versões anteriores) nunca era
calculada pelo pipeline atual: o histograma e o filtro da Galeria ficavam
vazios e o componente "alta variância" da prioridade de revisão era descartado.
Ela passa a conter a **incerteza pela margem top-1 − top-2**:

```
u = 1 − (p₁ − p₂)          p₁, p₂ = duas maiores probabilidades softmax
u ∈ [0, 1]: 0 = decisão segura (p₁ = 1), 1 = empate entre as duas melhores classes
```

- Projetos novos recebem a coluna na classificação (`compute_projections`).
- Projetos antigos recebem o valor ao serem abertos no IAT (a partir de
  `confs1`/`confs2`); valores já existentes não são sobrescritos.
- Modelos com uma única classe repetem top-1 como top-2 → `u = 1`.
- Rótulos de interface: "Incerteza (margem top-1 − top-2)"; na exportação CSV a
  coluna de saída chama-se `uncertainty_margin`.
- A prioridade de revisão (`analyze_embedding_space`, `_recompute_review_priority`)
  usa `u` diretamente como componente "incerteza".

Referências: Settles, B. (2009). *Active Learning Literature Survey.* Computer
Sciences Technical Report 1648, University of Wisconsin–Madison (margin
sampling, §3.1). Scheffer, T., Decomain, C. & Wrobel, S. (2001). *Active hidden
Markov models for information extraction.* IDA 2001, LNCS 2189, 309–318.

---

## 3. Segmentação — medidas morfométricas (itens 28–31)

**Código:** `src/iat_backend.py` (`segmentation_metrics`, `crofton_perimeter`,
`pixel_corner_hull`, `feret_diameters`, `moment_ellipse`),
`src/segmentation_docs.py` (textos exibidos na aba), exportação em
`src/iat_export_mixin.py`.

Notação: `M` = máscara binária do objeto; `A` = número de pixels de `M`;
`s` = pixel size (µm/pixel). Toda a geometria é calculada sobre o **conjunto de
pixels**, cada pixel tratado como um quadrado unitário; por isso solidez e
retangularidade nunca ultrapassam 1 e um objeto de 1 pixel tem Feret 1 × √2.

| Medida | Coluna | Definição |
|---|---|---|
| Área | `Area (pxl)` | `A` (µm² = A·s²) |
| Perímetro | `Perimeter (pxl)` | Estimador de Crofton com 4 direções (0°, 45°, 90°, 135°) sobre `M` |
| Circularidade | `circularity` | `4πA / P²`, limitada a 1 |
| ESD | `ESD (pxl)` | diâmetro esférico equivalente `2·√(A/π)` |
| Feret máximo | `Feret_max` | maior distância entre vértices da envoltória convexa dos cantos dos pixels de borda |
| Feret mínimo | `Feret_min` | menor largura da envoltória sobre todas as direções (*rotating calipers*; a largura mínima ocorre com uma aresta da envoltória encostada no calibre) |
| Alongamento | `Elongation` | `Feret_max / Feret_min` |
| Retângulo mínimo | `MinRect_w (pxl)`, `MinRect_h (pxl)` | lados do retângulo de área mínima que envolve a envoltória (`cv2.minAreaRect`) |
| Retangularidade | `Rectangularity` | `A / (MinRect_w · MinRect_h)` |
| Área da envoltória | `Hull Area (pxl)` | área do polígono convexo |
| Solidez | `Solidity` | `A / Hull Area` |
| Eixos maior/menor | `Major (pxl)`, `Minor (pxl)` | elipse com os mesmos momentos centrais de 2ª ordem de `M`: `4·√λ₁`, `4·√λ₂` (autovalores da matriz de covariância dos pixels, acrescida de 1/12 por eixo pela área do pixel) |
| Excentricidade | `Eccentricity` | `√(1 − (Minor/Major)²)` |
| Orientação | `Orientation (deg)` | `½·atan2(2μ₁₁, μ₂₀ − μ₀₂)`, em graus, coordenadas de imagem |
| Biovolume | `Biovolume (pxl3)` | elipsoide de revolução `π·Major·Minor²/6` (µm³ = px³·s³) |
| Intensidade | `Mean/Median/Std/Min/Max Intensity`, `Skewness`, `Kurtosis` | estatísticas dos níveis de cinza (0–255) dentro de `M`; curtose de Fisher (normal = 0) |
| Contraste | `Contrast` | `max − min` da intensidade em `M` |
| % no bordo | `Percent_on_border` | `100 · pixels de M na borda da imagem / A` |
| Orifícios | `Number Holes` | contornos internos (`cv2.RETR_CCOMP`) |
| Versão | `Metrics Version` | `2` para este conjunto de definições |

### O que mudou em relação às segmentações anteriores (versão 1)

- **Perímetro/circularidade:** antes `cv2.arcLength` sobre o contorno em
  passos de pixel, que superestima o perímetro de objetos pequenos e enviesa a
  circularidade para baixo; além disso misturava área por contagem de pixels
  com perímetro poligonal. O estimador de Crofton tem erro ≈ 1 % em discos
  digitais (teste: disco r = 50 px → 312,1 vs. 2πr = 314,2) e é não enviesado
  para formas com orientação isotrópica; em quadrados alinhados aos eixos
  subestima ≈ 6 % (limitação conhecida das 4 direções).
- **Feret:** antes eram os lados do retângulo de área mínima, que não são os
  diâmetros de Feret (o lado curto é ≥ Feret mínimo). Agora: Feret verdadeiro
  (teste: quadrado 60 px → 84,85 / 60,00; retângulo 80×20 a 30° → 83,6 / 21,9,
  com a espessura de 1 px dos contornos rasterizados).
- **Solidez/retangularidade:** antes `A_pixels / área_poligonal`, que podia
  passar de 1 em objetos pequenos; agora a mesma geometria de pixels nos dois
  termos.
- **Novas medidas:** perímetro, ESD, eixos maior/menor, excentricidade,
  orientação, biovolume, desvio-padrão/mín./máx./assimetria/curtose da
  intensidade, altura da imagem — compatíveis com as de ZooProcess/EcoTaxa.
- Arquivos de segmentação antigos continuam legíveis (as novas colunas aparecem
  vazias e a Galeria avisa "Métricas v1"). Para obter as medidas atuais, gere a
  segmentação novamente na aba Segmentação.
- As unidades físicas continuam sendo conversões na exibição/exportação:
  comprimentos × s, áreas × s², volumes × s³.

Referências:
- Legland, D., Kiêu, K. & Devaux, M.-F. (2007). *Computation of Minkowski
  measures on 2D and 3D binary images.* Image Analysis & Stereology, 26, 83–92.
- Lehmann, G. & Legland, D. (2012). *Efficient N-Dimensional surface estimation
  using Crofton formula and run-length encoding.* Insight Journal
  (hal-00673187).
- Shamos, M. I. (1978). *Computational Geometry.* Ph.D. thesis, Yale
  University. Toussaint, G. T. (1983). *Solving geometric problems with the
  rotating calipers.* Proc. IEEE MELECON '83.
- Hu, M.-K. (1962). *Visual pattern recognition by moment invariants.* IRE
  Transactions on Information Theory, 8(2), 179–187.
- Gorsky, G. et al. (2010). *Digital zooplankton image analysis using the
  ZooScan integrated system.* Journal of Plankton Research, 32(3), 285–303.
- Vandromme, P. et al. (2012). *Assessing biases in computing size spectra of
  automatically classified zooplankton from imaging systems: a case study with
  the ZooScan integrated system.* Methods in Oceanography, 1–2, 3–21.
- Picheral, M. et al. (2017). *EcoTaxa, a tool for the taxonomic classification
  of images.* http://ecotaxa.obs-vlfr.fr (definições de `object_area`,
  `object_major`, `object_minor`, `object_esd`, `object_perim.`).

---

## 4. Análise Temporal — escala de densidade comum (item 40)

**Código:** `src/iat_temporal_figure.py` (`class_density`,
`density_reference`, `density_levels`), `src/iat_temporal_spatial.py`,
`src/web/pages/iat_temporal.py`.

As manchas de cada classe são três faixas a 17 %, 38 % e 64 % de um pico de
densidade (histograma 2-D de 55 × 55 células nos limites fixos da projeção,
suavizado com Gaussiana σ = 1,4 célula). Até a versão anterior o pico era **o de
cada classe em cada quadro**: um quadro com 20 imagens parecia tão denso quanto
um com 2 000, e a comparação A/B/C, a animação e o GIF induziam leitura errada.

Agora, por padrão (**"Escala de densidade comum a todos os quadros"** ligada),
o pico de referência de cada classe é o **maior pico suavizado que a classe
atinge em qualquer quadro do recorte atual**. O quadro mais denso mantém as
faixas completas; os demais ficam proporcionalmente mais tênues, e os quadros
passam a ser comparáveis entre si. Desligando a opção, volta-se ao comportamento
anterior (pico por quadro). A referência é recalculada em "Aplicar e gerar
intervalos" e vale para a tela, as comparações, a animação e as exportações
PNG/SVG/PDF/GIF.

Referências: Tufte, E. R. (1983). *The Visual Display of Quantitative
Information.* Graphics Press (small multiples com escalas comuns). Cleveland,
W. S. (1993). *Visualizing Data.* Hobart Press (escalas comuns em painéis
múltiplos). Silverman, B. W. (1986). *Density Estimation for Statistics and Data
Analysis.* Chapman & Hall (suavização de histogramas por kernel gaussiano).

---

## 5. Treino — modelo já existente (item 48)

**Código:** `src/train.py` (`_resolve_existing_model`, `_non_interactive`),
`src/dashboard_execution_mixin.py` (`training_existing_model_state`,
`training_conflict`, `start_training`, `_launch`), `src/web/app.py`,
`src/web/pages/modals.py`, `src/iat_model_cycle_mixin.py`.

Quando `model_<projeto>` já existia na pasta de saída, o `train.py` chamava
`input()` para perguntar o que fazer. Lançado pela interface (sem console), o
processo parava à espera de uma resposta impossível ou falhava com `EOFError`.

- A interface verifica antes de iniciar: se há modelo completo, pergunta
  **Sobrescrever / Cancelar**; se há checkpoint parcial (`_last_epoch`) com
  `Resume = T`, retoma automaticamente como antes; parcial com `Resume = F`
  oferece **Retomar / Sobrescrever / Cancelar**.
- A decisão vai na configuração (`existingModel = overwrite | resume`). Ao
  sobrescrever, os checkpoints antigos (`model_<p>` e `_last_epoch`) são
  removidos antes do treino, para que o novo melhor modelo seja publicado desde
  a primeira época.
- Os workers são lançados com `stdin` fechado e `ILT_NONINTERACTIVE=1`;
  sem decisão e sem terminal, o `train.py` termina com uma mensagem clara em vez
  de travar. No terminal, o prompt interativo continua existindo.

---

## 7. Treino — receita de pré-processamento, augmentation e otimização (etapa 2, itens 1–5)

**Código:** `src/training_recipe.py` (presets e validação, sem PyTorch),
`src/features.py` (`get_transforms`, `SquareCanvas`, `preprocess_from_checkpoint`),
`src/ilt_losses.py` (`FocalLoss`), `src/train.py` (`build_optimizer`,
`LrSchedule`, checkpoint formato 3), interfaces em `src/ilt_gui.py` e
`src/web/pages/training.py`.

A **receita** reúne as escolhas que mudam *como* a rede é treinada; fica
gravada em `training_config_<projeto>.json` e dentro do checkpoint
(`ilt_state.recipe`, `ilt_state.preprocess`, `class_names`, `arch`). Dois
presets estão disponíveis nas duas interfaces:

| Campo (config) | Recomendada | Legada (1.0) | O que faz |
|---|---|---|---|
| `inputNormalize` | `imagenet` | `none` | subtrai a média e divide pelo desvio por canal do pré-treino (0,485/0,456/0,406; 0,229/0,224/0,225) |
| `inputResize` | `pad` | `stretch` | `pad`: completa a imagem em quadrado com a cor mediana da borda e redimensiona para 224×224 preservando a forma; `stretch`: deforma para 224×224 |
| `augment` | `true` | `false` | só no treino: espelhos horizontal/vertical, rotação uniforme em [0°, 360°) com moldura expandida (nada é cortado) e brilho/contraste ±10 % |
| `optimizer` | `adamw` | `adam` | AdamW = Adam com *weight decay* desacoplado |
| `weightDecay` | 0,05 | 0 | não aplicado a parâmetros 1-D (LayerNorm, bias, layer-scale) |
| `scheduler` | `plateau` | `none` | `plateau`: LR × 0,5 quando a métrica de validação não melhora por `patience//2` épocas (piso 1 %); `cosine`: cosseno do fim do warmup até `numEpochs`; `none`: constante |
| `warmupEpochs` | 3 | 0 | crescimento linear da LR nas primeiras épocas |
| `headLrMult` | 10 | 1 | LR da camada classificadora nova = `lr × headLrMult` |
| `labelSmoothing` | 0,1 | 0 | só com `lossName = ce` |
| `stochasticDepth` | 0,1 | 0,1 | probabilidade de descarte de blocos residuais (padrão do `torchvision` para ConvNeXt-Tiny; já estava ativa) |

Observações:

- **Por que normalizar:** os pesos `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` foram
  treinados com essa normalização; sem ela as ativações do primeiro bloco ficam
  fora da distribuição de pré-treino e a transferência é subaproveitada. A
  classificação lê o pré-processamento gravado no checkpoint e o reaplica;
  modelos antigos (sem registro) são tratados como legados (`none`/`stretch`),
  então continuam sendo classificados exatamente como antes.
- **Por que não esticar:** o alongamento é caráter taxonômico (copépodes,
  quetognatos, cadeias); `pad` preserva a razão de aspecto.
- **Augmentation:** imagens de plâncton não têm orientação canônica; a rotação
  com moldura expandida faz a escala aparente variar até 1/√2, o que funciona
  como variação de escala. A aleatoriedade usa `random`, semeado por *worker*
  do DataLoader pelo PyTorch.
- **Schedule:** com a configuração padrão (500 épocas, *patience* 10) um
  cosseno até a época 500 quase não decai antes da parada antecipada; por isso
  o padrão é `plateau`, que reduz a LR quando a validação estagna e dá uma
  segunda fase de melhora antes de o *patience* esgotar. Use `cosine` em
  treinos com número fixo de épocas (sem parada antecipada).
- **Focal loss:** implementação própria (`ilt_losses.FocalLoss`) a partir de
  `log_softmax`: `FL = −(1 − p_t)^γ · log p_t`, média sobre o lote. É
  numericamente equivalente ao pacote `focal-loss-torch` (a menos do `eps` de
  1e-16), mas sem o `assert torch.all(...)` que sincronizava a GPU a cada
  passo. O pacote deixou de ser dependência.
- **Métrica de validação e parada antecipada** não mudaram.
- **Retomar treino:** exige a mesma receita do checkpoint (bloqueado com
  mensagem clara, como já acontecia com a métrica de validação). Checkpoints de
  versões anteriores (formato < 3) são retomados automaticamente com a receita
  legada e o otimizador de grupo único original.
- **Comparação controlada:** para medir o efeito da receita, treine o mesmo
  dataset com "Receita legada (1.0)" e com "Receita recomendada" e compare
  `evaluation_<projeto>.json`.
- O `stats_<projeto>.csv` ganhou a coluna `LR` (taxa do *backbone* na época).

Referências:
- Liu, Z. et al. (2022). *A ConvNet for the 2020s.* CVPR (receita de
  fine-tuning: AdamW, warmup + cosseno, label smoothing, stochastic depth).
- Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization.*
  ICLR (AdamW). Loshchilov, I. & Hutter, F. (2017). *SGDR: Stochastic Gradient
  Descent with Warm Restarts.* ICLR (cosseno).
- Goyal, P. et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in
  1 Hour.* arXiv:1706.02677 (warmup linear).
- Howard, J. & Ruder, S. (2018). *Universal Language Model Fine-tuning for Text
  Classification.* ACL (taxas discriminativas por camada).
- Müller, R., Kornblith, S. & Hinton, G. (2019). *When Does Label Smoothing
  Help?* NeurIPS.
- Huang, G. et al. (2016). *Deep Networks with Stochastic Depth.* ECCV.
- Lin, T.-Y. et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.
- Orenstein, E. C. & Beijbom, O. (2017). *Transfer learning and deep feature
  extraction for planktonic image data sets.* WACV (rotações/espelhos).
- Luo, J. Y. et al. (2018). *Automated plankton image analysis using
  convolutional neural networks.* Limnology & Oceanography: Methods, 16,
  814–827.
- Lumini, A. & Nanni, L. (2019). *Deep learning and transfer learning features
  for plankton classification.* Ecological Informatics, 51, 33–43.
  Kyathanahally, S. P. et al. (2021). *Deep learning classification of Lake
  Zooplankton.* Frontiers in Microbiology, 12, 746297 (entrada sem distorção).
- Bengio, Y. (2012). *Practical recommendations for gradient-based training of
  deep architectures* (redução da LR em platô), in *Neural Networks: Tricks of
  the Trade*, Springer.

---

## 8. Desempenho (etapa 3, itens 15, 20, 26, 39, 71, 105)

Nenhum destes ajustes altera as definições científicas; eles mudam onde e como
os mesmos cálculos são executados.

| Item | Código | O que mudou | Efeito esperado |
|---|---|---|---|
| 15 | `features.compute_features` | não chama mais `seed_everything(0)` por batch (ligava `cudnn.deterministic` e anulava o `cudnn.benchmark`); `freeze_bn` (sem efeito em ConvNeXt, que usa LayerNorm) removido | extração de features sem penalidade de kernels determinísticos |
| 105 | `torch_runtime.amp_dtype` | precisão mista em `bfloat16` em GPUs CC ≥ 8.0 (Ampere, Ada, Hopper, Blackwell); `float16` + `GradScaler` nas anteriores (Pascal/Volta/Turing, onde bf16 é emulado). `ILT_AMP_DTYPE` força | sem *overflow*/*loss scaling* em GPUs novas; mesmo comportamento de antes na P4000 |
| 105 | `torch_runtime.maybe_compile` | `torch.compile` opcional (`ILT_TORCH_COMPILE=1`); os checkpoints continuam vindo do modelo original (chaves inalteradas) | 10–30 % em Ampere+ após a primeira época (compilação lenta; Linux) |
| 105 (59/11) | `torch_runtime.resolve_batch_size` | lote de inferência `auto` pela VRAM livre (128/256/512) e lote de treino `auto` opcional; em `OutOfMemoryError` o lote é reduzido pela metade e o batch/época recomeça | usa a GPU disponível sem configurar à mão |
| 26 | `iat_backend._gpu_pca`, `_gpu_knn_indices` | PCA (`torch.pca_lowrank`, mesmo modelo randomizado de Halko et al. 2011) e kNN (expansão ‖a−b‖² = ‖a‖² − 2a·b + ‖b‖² em blocos + `topk`) na GPU para seleções ≥ 3 000 imagens; CPU/scikit-learn abaixo disso ou sem CUDA | kNN de 50 mil × 50-D em ~segundo em vez de minutos |
| 20 | `_project_validation_space` | UMAP/PaCMAP com todos os núcleos a partir de 20 000 imagens (o `umap-learn` fica em uma thread quando `random_state` é fixo); abaixo disso a semente fixa é mantida | análise estrutural de seleções grandes muito mais rápida; o resultado pode variar levemente entre execuções nesse regime |
| 39 | `run_segmentation` | decodificação + medidas em `ThreadPoolExecutor` (OpenCV libera o GIL); ordem das linhas preservada; "Parar" cancela a fila | ganho próximo ao nº de núcleos |
| 71 | `web/pages/iat_gallery.py`, `assets/hud.js`, `web/pages/full_image.py` | miniaturas com `data-src` + `IntersectionObserver` (só as visíveis são pedidas); resposta com `ETag` (arquivo + cache de segmentação) e `max-age` de 7 dias; 304 quando nada mudou | páginas de 500 imagens abrem de imediato; voltar a uma página não decodifica nem transfere de novo |
| correção 2026-11-17 | `features.compute_features`, `torch_runtime.loader_kwargs` | o DataLoader de cada batch da classificação não usa mais processos persistentes. Com `persistent_workers` e `pin_memory`, o PyTorch registra um gancho `atexit` por processo auxiliar que mantém o processo encerrado e 2 pipes abertos até o fim do programa; com 8 auxiliares eram 16 descritores perdidos por batch e o limite de 1 024 do Ubuntu estourava perto do 60º batch (`OSError: [Errno 24] Too many open files`). O treino, que reutiliza os mesmos DataLoaders a cada época, mantém os processos persistentes | descritores constantes em qualquer número de batches, sem mudança de velocidade (medido: 8 pipes do 1º ao 12º batch, antes 20 → 152) |
| correção 2026-11-17 | `hw_resources.raise_open_file_limit` | no início da classificação (`main.py`) e das interfaces (`ilt_web.py`, `ilt_gui.py`), só no Linux, o limite *soft* de arquivos abertos sobe para 65 536 (sem passar do limite *hard*; não exige administrador) | margem de segurança para execuções longas e para o ciclo integrado do IAT, que classifica dentro do processo da interface |

Referências: Micikevicius, P. et al. (2018). *Mixed Precision Training.* ICLR.
Kalamkar, D. et al. (2019). *A Study of BFLOAT16 for Deep Learning Training.*
arXiv:1905.12322. Halko, N., Martinsson, P.-G. & Tropp, J. A. (2011). *Finding
structure with randomness: probabilistic algorithms for constructing approximate
matrix decompositions.* SIAM Review, 53(2), 217–288. Johnson, J., Douze, M. &
Jégou, H. (2021). *Billion-scale similarity search with GPUs.* IEEE Trans. Big
Data, 7(3), 535–547 (busca de vizinhos por blocos em GPU). McInnes, L., Healy, J.
& Melville, J. (2018). *UMAP.* arXiv:1802.03426 (documentação do `umap-learn`
sobre `random_state` e paralelismo).

---

## 9. Visualização e exportação (etapa 4, itens 42, 64, 81, 83, 92, 95 e pedidos adicionais)

### 9.1 Análise Temporal — projeções alternativas, incluindo UMAP supervisionado

**Código:** `src/iat_temporal_analysis.py` (`temporal_projection`,
`PROJECTION_METHODS`), páginas web e clássica da Análise Temporal.

Além das coordenadas x/y do projeto (t-SNE da classificação), a aba pode usar
coordenadas calculadas a partir dos embeddings de **todas as imagens
carregadas**: PCA 2-D, t-SNE recalculado (openTSNE, cosseno), UMAP, **UMAP
supervisionado** (rótulos manuais atuais) e PaCMAP. Etapas: embeddings
normalizados (L2) → PCA para 50 dimensões (PCA incremental quando há mais de
60 000 imagens) → projeção. O resultado fica em cache em
`<projeto>/projections/`, chaveado pelo método, pelo conjunto de imagens e
(no supervisionado) pelos rótulos; trocar de projeção depois é instantâneo.
Todos os quadros, comparações, manchas, animação e exportações usam o mesmo
referencial escolhido; os eixos das figuras recebem o nome da projeção.

**Cuidado com o UMAP supervisionado:** os rótulos puxam as imagens da mesma
classe para perto (peso 0,35). Ele mostra a estrutura *dada* a anotação e é útil
para apresentar grupos já validados, mas **não deve ser usado para procurar
erros de rótulo** nem para concluir que classes são separáveis; para isso use
o UMAP não supervisionado, o t-SNE ou a PCA.

Referências: McInnes, L., Healy, J. & Melville, J. (2018). *UMAP: Uniform
Manifold Approximation and Projection for Dimension Reduction.*
arXiv:1802.03426 (inclui a variante supervisionada, `target_weight`). Poličar,
P. G., Stražar, M. & Zupan, B. (2019). *openTSNE: a modular Python library for
t-SNE dimensionality reduction and embedding.* bioRxiv 731877. Wang, Y. et al.
(2021). *Understanding how dimension reduction tools work: an empirical approach
to deciphering t-SNE, UMAP, TriMap, and PaCMAP.* JMLR 22(201), 1–73. Ross, D. A.
et al. (2008). *Incremental learning for robust visual tracking.* IJCV 77,
125–141 (PCA incremental).

### 9.2 Análise Temporal — evolução entre intervalos e diversidade (item 42)

**Código:** `iat_temporal_analysis.frame_class_counts`, `diversity_indices`;
gráfico "Evolução entre intervalos" e botão "Exportar contagens (CSV)".

Por intervalo: número de imagens por classe (linhas), composição em % (área
empilhada) e diversidade. Com `p_i` = fração das imagens da classe *i* no
intervalo:

| Índice | Fórmula | Interpretação |
|---|---|---|
| N | Σ contagens | imagens no intervalo |
| S | nº de classes com contagem > 0 | riqueza observada |
| H′ | −Σ p_i ln p_i (nats) | entropia de Shannon |
| N1 | exp(H′) | número de Hill de ordem 1 (nº efetivo de classes) |
| N2 | 1 / Σ p_i² | número de Hill de ordem 2 (inverso de Simpson) |
| J | H′ / ln S | equitabilidade de Pielou (só com S > 1) |

S, N1 e N2 têm a mesma unidade (nº de classes) e dividem o mesmo eixo no
gráfico; H′ e J aparecem no resumo e no CSV. Clicar num ponto do gráfico leva
ao intervalo. Os valores descrevem **contagens de imagens**, não abundância por
volume amostrado, e a riqueza observada depende do número de imagens (compare
intervalos de tamanho semelhante ou use N1/N2, menos sensíveis à amostragem).

Referências: Shannon, C. E. (1948). Bell System Technical Journal, 27,
379–423. Simpson, E. H. (1949). Nature, 163, 688. Pielou, E. C. (1966).
Journal of Theoretical Biology, 13, 131–144. Hill, M. O. (1973). *Diversity and
evenness: a unifying notation and its consequences.* Ecology, 54(2), 427–432.
Jost, L. (2006). *Entropy and diversity.* Oikos, 113(2), 363–375. Chao, A. et
al. (2014). *Rarefaction and extrapolation with Hill numbers.* Ecological
Monographs, 84(1), 45–67.

### 9.3 Janela de exportação de figuras (pedido adicional)

`src/web/pages/iat_temporal.py` (editor) e `iat_temporal_figure.save_figure`,
`save_frame_images`, `save_animated_gif`:

- prévia grande feita pelo mesmo desenho do arquivo (só a resolução é menor),
  com tamanho final em cm, polegadas e pixels;
- formatos PNG, TIFF (compressão LZW, sem perda), JPEG (qualidade 95), SVG e
  PDF; resolução em dpi (também usada nas nuvens de pontos rasterizadas de
  SVG/PDF); fundo transparente; recorte das margens;
- tamanho em centímetros com predefinições (automático, apresentação 16:9,
  A4 e A3 paisagem, pôster) e ampliação automática quando textos ou legenda
  não cabem (as letras nunca são reduzidas);
- tema claro/escuro, cor de cada texto, fonte única para todos os textos ou
  por elemento (somente fontes realmente instaladas, sem substituição
  silenciosa), tamanhos em pt, nomes das classes na legenda;
- pasta e nome do arquivo na própria janela (nada é sobrescrito: sufixo _2,
  _3…), opção "um arquivo por intervalo" (todos com o mesmo tamanho) e GIF com
  duração por quadro e dpi próprios; exportações longas rodam em segundo
  plano e avisam ao terminar.

Para revistas: 300 dpi para imagens, 600 dpi para gráficos com texto, TIFF ou
PDF; verifique as instruções de cada revista.

### 9.4 Projeção — miniatura ao passar o mouse (item 64)

O ponto sob o cursor mostra a miniatura, o nome, o rótulo manual, a previsão
com a confiança, a incerteza (margem) e o rótulo de origem. A leitura é feita
nos dados já desenhados, sem bloquear a interface. Em Overview com manchas
(sem pontos individuais) não há ponto para mostrar.

### 9.5 Clustering — dendrograma interativo e grupos representativos (itens 81 e 83)

- Clicar em um nó (ponto no alto de cada ∩) define o corte logo abaixo dele
  (0,999 × altura do nó / dist_max), separando seus dois ramos; o controle
  deslizante continua disponível.
- Para o corte atual, o painel "Grupos no corte atual" lista os grupos do
  `fcluster(criterion='distance')` (os mesmos de "Relabel por grupo"), do maior
  para o menor, com a composição de rótulos, a fração do rótulo dominante e as
  4 imagens mais **típicas** de cada grupo: as de maior similaridade de
  cosseno com a direção média dos embeddings (L2) do grupo. Cada grupo pode
  ser visto inteiro, rotulado de uma vez (pelo mesmo procedimento da Galeria:
  desfazer, salvar automaticamente, classes marcadas para atualizar o
  clustering) ou aberto na Galeria (a seleção passa a ser o grupo).

Referências: Dhillon, I. S. & Modha, D. S. (2001). *Concept decompositions
for large sparse text data using clustering.* Machine Learning, 42, 143–175
(centroide esférico/cosseno). Müllner, D. (2011). *Modern hierarchical,
agglomerative clustering algorithms.* arXiv:1109.2378.

### 9.6 Segmentação — prévia ao vivo e histograma (item 92)

A prévia mostra lado a lado **original | máscara | sobreposição** (máscara em
vermelho e contorno em amarelo), calculadas no momento com o método e o raio
escolhidos — pela mesma função usada na segmentação em lote — antes de gerar.
Os botões ◀ ▶ percorrem as imagens do escopo (seleção, marcadas, batch ou
projeto). O histograma dos níveis de cinza separa pixels do objeto e do fundo
e marca o limiar quando o método tem um valor global único: t* de Otsu (Otsu,
Otsu + maior componente, primeira etapa do watershed), 243 (limiar fixo) ou o
percentil 35; nos métodos locais (rolling ball, black-hat, Sauvola) e no
k-means não há um único nível de cinza. As métricas exibidas ao lado são as da
segmentação já gerada.

Referência: Otsu, N. (1979). *A threshold selection method from gray-level
histograms.* IEEE Trans. Systems, Man, and Cybernetics, 9(1), 62–66.

### 9.7 Exportação para o EcoTaxa (item 95)

**Código:** `src/ecotaxa_export.py`; aba "EcoTaxa (TSV)" em Exportar.

Gera `ecotaxa_<nome>.tsv` no formato de importação do EcoTaxa: primeira linha
com os nomes das colunas, segunda com os tipos (`[t]` texto, `[f]` número),
colunas `img_file_name`, `img_rank`, `object_id`, `object_date` (AAAAMMDD),
`object_time` (HHMMSS), `object_depth_min/max` (m), `object_annotation_category`,
`object_annotation_status`, as medidas da segmentação em pixels
(`object_area`, `object_perim.`, `object_major`, `object_minor`, `object_angle`,
`object_circ.`, `object_feret`, `object_feretmin`, `object_esd`, …, intensidades),
a previsão (`object_pred_category`, `object_pred_score`,
`object_uncertainty_margin`) e `sample_id`, `acq_id`, `process_id`,
`process_pixel` (µm/pixel), `process_software`, `process_segmentation_method`,
`process_metrics_version`. Data, hora e profundidade vêm do nome da imagem
quando presentes. Como no ZooProcess, as medidas ficam em pixels e a calibração
em `process_pixel`. Com "Copiar as imagens" ligado é criada a pasta
`ecotaxa_<nome>/` com o TSV e as imagens, pronta para compactar e importar.

Status da anotação: **automático** (padrão) = `validated` quando o rótulo foi
alterado no IAT ou veio da pasta de origem no modo auditoria, `predicted` nos
demais; ou todas `predicted`; ou todas `validated` (use só se todas as imagens
foram revisadas).

Referências: Picheral, M., Colin, S. & Irisson, J.-O. (2017). *EcoTaxa, a tool
for the taxonomic classification of images.* https://ecotaxa.obs-vlfr.fr.
Gorsky, G. et al. (2010). J. Plankton Res. 32(3), 285–303 (ZooProcess, nomes e
unidades das medidas).

---

## 10. Análise Temporal — Origem → Previsão (pedido adicional, 2026-11-13)

**Código:** `src/iat_origin_flow.py` (tabelas, resumo, leitura de versões
anteriores e figura estática) e `src/web/pages/iat_origin.py` (painel e janela
de exportação). Somente leitura.

**Dados.** Para cada imagem do recorte da subaba (mesmos filtros, classes de
interesse e intervalos), duas ou três destas colunas, escolhidas livremente e
sempre nesta ordem:

- *origem* = `source_label`, a última pasta do caminho da imagem na
  classificação. É gravada em toda classificação, com ou sem o modo auditoria;
  em projetos antigos vem da última pasta de `klass`. O modo auditoria só faz o
  rótulo manual inicial ser igual à pasta;
- *previsão* = `preds1` (top-1) do modelo escolhido. "Atual" usa os CSVs do
  projeto; "Antes de vNNN" lê a cópia que o ciclo integrado gravou antes de
  aplicar a versão vNNN (`model_history/vNNN/dataframes_before/`), ou seja, as
  previsões do modelo anterior, associadas pelo CSV e pelo nome da imagem;
- *rótulo revisado* = `manual_label`, o rótulo atual do IAT. Antes de qualquer
  revisão ele é a própria previsão, ou a pasta no modo auditoria.

**Medidas.** O resumo compara as duas primeiras colunas marcadas, a e b:
origem × previsão, origem × rótulo revisado ou previsão × rótulo revisado. Com
n(i, j) = número de imagens da classe i em a e j em b e N = Σ n(i, j):

- concordância = Σᵢ n(i, i) / N (concordância global da tabela de contingência
  a × b; Congalton 1991). Origem × previsão: concordância com a pasta. Origem ×
  rótulo: rótulos atuais iguais à pasta. Previsão × rótulo: previsões mantidas
  no rótulo atual, o que só mede acerto do modelo depois de revisão;
- para cada classe i de a: proporção que mantém a mesma classe em b,
  n(i, i) / Σⱼ n(i, j), e o principal outro destino, argmax₍ⱼ ≠ ᵢ₎ n(i, j);
- maior divergência = maior n(i, j) com j ≠ i;
- com as três colunas: entre as imagens em que pasta ≠ previsão, quantas têm
  hoje o rótulo da pasta, o da previsão ou outro.

**Interpretação.** A concordância só equivale a acurácia (ou *recall* por
classe) quando as pastas foram conferidas por especialistas. Pastas de
treino montadas manualmente costumam ter erros de rótulo; as divergências
pasta ≠ previsão são justamente as candidatas a revisão (Northcutt, Jiang &
Chuang 2021). Se as pastas não forem classes (estações, datas, lotes), o painel
avisa quando menos da metade das imagens vem de pastas com o nome de uma classe
prevista pelo modelo.

**Nomes.** "Unir grafias" junta nomes que diferem só em maiúsculas e espaços
(`medusa ` e `Medusa`), usando a grafia do modelo. Com a opção desligada os
nomes são comparados exatamente.

**Grupos e nomes nos gráficos (2026-11-17).** "Agrupar / renomear classes…"
(`src/iat_display_groups.py`, `src/web/pages/iat_groups.py`) abre uma matriz
para a coluna escolhida: cada linha é um nome (rótulo revisado, pasta de origem
ou previsão) e cada coluna uma classe de referência, as previsões do modelo
(para a própria coluna Previsão, os rótulos atuais). A célula (i, j) mostra
n(i, j) / Σⱼ n(i, j), a porcentagem das imagens do nome i na classe j. A
sugestão de cada linha é a classe dominante, argmaxⱼ n(i, j); "Aceitar
sugestões" usa as que atingem o limite escolhido (70 % por padrão). Clicar numa
célula usa aquela classe como nome; o nome também pode ser digitado. Nomes
iguais numa coluna formam um grupo.

- O rótulo revisado define as classes de **todos os gráficos da aba**:
  projeção (pontos e manchas), evolução entre intervalos, diversidade, resumo,
  coluna Rótulo revisado do alluvial e todas as exportações (figuras, GIF,
  um arquivo por intervalo, CSV). Origem e Previsão valem para as colunas
  correspondentes do alluvial. As classes de interesse continuam filtrando
  pelos nomes originais.
- Com grupos, contagens e índices de diversidade passam a contar grupos
  (S = grupos presentes; H′, N1 e N2 sobre as proporções dos grupos). A
  concordância do alluvial compara os nomes mostrados: unir classes que o
  modelo confunde aumenta a concordância, como em qualquer agregação de
  categorias de uma matriz de confusão.
- A sugestão segue o modelo e pode errar; confira antes de aceitar.
- Os nomes ficam em `<projeto>/exports/temporal/grupos_de_exibicao.json` e
  voltam ao carregar a projeção. A tabela do alluvial exportada vem acompanhada
  de `<nome>_grupos.json`. Os rótulos das imagens nunca mudam.

**Diagrama.** Alluvial = Parallel Sets do Plotly (`parcats`): cada coluna
empilha as categorias na mesma ordem alfabética, então uma classe fica de frente
para ela mesma; a largura de cada faixa é proporcional ao número de imagens.
Sankey = mesmo fluxo com nós posicionados pelo Plotly. Faixas menores que o
limite escolhido (0,5 % do total por padrão) ficam fora do desenho, mas entram
nas contagens, na tabela e no CSV. A figura salva é desenhada pelo matplotlib
com os mesmos fluxos e cores da tela, em PNG, TIFF LZW, JPEG, SVG ou PDF.

**Exportação.** "Exportar tabela (CSV)" grava em `<projeto>/exports/temporal/`
todas as combinações do recorte nas colunas marcadas (`origem`,
`previsao_modelo`, `rotulo_revisado`), com `imagens`, `pct_do_total`,
`pct_da_<primeira coluna>`, `<primeira>_igual_<segunda>`, modelo, recorte,
eixo, se as grafias foram unidas e se há grupos de exibição.

Referências:
Congalton, R. G. (1991). A review of assessing the accuracy of classifications
of remotely sensed data. *Remote Sensing of Environment* 37(1), 35–46.
Kosara, R., Bendix, F. & Hauser, H. (2006). Parallel Sets: interactive
exploration and visual analysis of categorical data. *IEEE Transactions on
Visualization and Computer Graphics* 12(4), 558–568.
Rosvall, M. & Bergstrom, C. T. (2010). Mapping change in large networks.
*PLoS ONE* 5(1), e8694.
Schmidt, M. (2008). The Sankey diagram in energy and material flow management.
Part I: History. *Journal of Industrial Ecology* 12(1), 82–94.
Northcutt, C. G., Jiang, L. & Chuang, I. L. (2021). Confident learning:
estimating uncertainty in dataset labels. *Journal of Artificial Intelligence
Research* 70, 1373–1411.

---

## 6. Código removido nesta revisão (sem uso verificado)

- `src/inference.py` (CLI não referenciado por nenhum módulo, script ou teste).
- `src/data.py`: funções de geração de fundos (`create_csv`, `get_image`,
  `map_of_images`, `generate_bkg`); permanecem `read_labels` e
  `label_predictions`, usadas por `main.py`.
- `src/projections.py`: `tsne_fit` (scikit-learn), não utilizada.
- `src/aux.py`: chaves `backgrounds`, `map_factor`, `make_bkg`.
- `src/web/pages/iat_validation.py` e, no IAT clássico, os métodos da aba
  "Validação assistida" (`_build_validation`, `_send_validation_to_gallery`,
  `_show_validation_scatter`), sem entrada de menu desde a versão anterior. A
  prioridade de revisão continua disponível em "Colorir por" na Projeção.
- Página-ponte `iat_bridge.layout` e o lançador da interface clássica a partir
  da web (`open_classic_iat`); `iat_bridge.py` mantém apenas a lista de subabas.
