# Changelog

## 1.1.0 — 2026-11-11

### Correções e ajustes (2026-11-17)
- **Agrupar / renomear classes na Análise Temporal.** Matriz com sugestões
  (cada nome × classes previstas, classe dominante sugerida, aceitar acima de
  um limite, clique na célula ou nome digitado) para o rótulo revisado, a
  origem e a previsão. Os nomes do rótulo revisado valem para todos os
  gráficos e exportações da aba (projeção, evolução, diversidade, resumo,
  alluvial, figuras, GIF e CSV) e ficam salvos no projeto; os rótulos das
  imagens não mudam. Substitui a lista de nomes anterior do alluvial.
- **Barra de escala da imagem (Galeria).** A imagem abre ajustada à janela,
  com zoom (Ajustar, 1×, 2×, 4×, −/+, Ctrl + roda). A barra fica sobre a
  imagem, com comprimento 1, 2 ou 5 × 10ⁿ µm calculado pelo pixel size do
  projeto e pelo zoom, rótulo em µm ou mm e o pixel size em uso; pode ser
  ocultada. A janela mostra também o tamanho físico da imagem.
- **Classificação: `OSError: [Errno 24] Too many open files`.** O DataLoader
  criado para cada batch usava processos persistentes; com memória fixada para
  a GPU, o PyTorch registra um gancho de saída por processo auxiliar que mantém
  o processo encerrado e 2 pipes abertos até o fim do programa. Eram 16
  descritores perdidos por batch e a classificação parava perto do 60º batch
  (limite de 1 024 do Ubuntu). A inferência não usa mais processos
  persistentes: descritores constantes, mesma velocidade. Afetava também o
  ciclo integrado do IAT. O treino não muda. Como margem de segurança, a
  classificação e as interfaces elevam o limite de arquivos abertos para
  65 536 no Linux. `docs/METRICAS_E_REFERENCIAS.md`, seção 8.
- **Pixel size da classificação voltava para 12,87 µm/pixel ao iniciar.** Ao
  iniciar, a pasta do modelo era revalidada e o pixel size gravado no treino do
  modelo substituía o valor digitado. Agora o valor digitado sempre vale; o
  pixel size do treino só preenche o campo quando a pasta do modelo é escolhida
  e o campo ainda está no padrão. Na interface web, Iniciar usa exatamente os
  valores mostrados nos campos, mesmo sem sair do campo antes de clicar.
- **Pixel size no ciclo integrado.** A versão clássica gravava 12,87 fixo no
  novo modelo; agora usa o pixel size do projeto. Na web, o valor do ciclo
  acompanha o projeto aberto e as mudanças feitas no IAT.
- **Origem → Previsão:** colunas independentes (origem, previsão, rótulo
  revisado; quaisquer duas ou as três), resumo e tabela para as duas primeiras
  colunas marcadas, cores por qualquer coluna e "Agrupar / renomear…" para unir
  ou renomear classes só no diagrama e nas exportações, com os nomes usados
  gravados junto da tabela. `docs/METRICAS_E_REFERENCIAS.md`, seção 10.

### Análise Temporal — Origem → Previsão (2026-11-13) — `docs/METRICAS_E_REFERENCIAS.md`, seção 10
- Novo painel na subaba Análise Temporal e Espacial (web): diagrama alluvial
  (Parallel Sets) ou Sankey da classe da pasta de origem para a classe
  prevista pelo modelo, com terceira coluna opcional do rótulo revisado.
- Usa colunas já existentes (`source_label`, `preds1`, `manual_label`); a
  classificação não mudou. Modelo atual ou versões anteriores guardadas pelo
  ciclo integrado (`model_history/<versão>/dataframes_before`).
- Recorte por intervalo ou todos os intervalos, com os filtros e classes da
  aba; cores pela origem ou pela previsão; faixas pequenas ocultáveis; junção
  de grafias que diferem só em maiúsculas/espaços; concordância, maior
  confusão, tabela por pasta e aviso quando as pastas não parecem classes.
- "Salvar figura…" com prévia fiel (PNG, TIFF LZW, JPEG, SVG, PDF; dpi, cm,
  tema, fonte, títulos) e "Exportar tabela (CSV)".
- `load_points` ganhou a opção `with_origin` (só a interface web usa); a
  interface clássica lê exatamente as mesmas colunas de antes.

### Correção importante (2026-11-13)
- Classificação: com mais de um batch a extração de features entrava em
  recursão infinita a partir do segundo batch (`RecursionError`). Introduzido
  na etapa 3 (o executor do modelo era guardado como submódulo do próprio
  modelo); afetava `main.py` e o ciclo integrado do IAT. Corrigido e
  verificado com 3 batches.

### Visualização e exportação (2026-11-13, etapa 4) — `docs/METRICAS_E_REFERENCIAS.md`, seção 9
- Análise Temporal: projeções alternativas calculadas dos embeddings (PCA,
  t-SNE, UMAP, **UMAP supervisionado**, PaCMAP), com cache por projeto; também
  na interface clássica. Eixos das figuras com o nome da projeção.
- Análise Temporal (web): gráfico "Evolução entre intervalos" (contagem,
  composição %, diversidade S/N1/N2; clique leva ao intervalo), resumo com
  H′ e J, exportação das contagens em CSV.
- Análise Temporal (web): nova janela "Salvar figura" — prévia fiel, PNG/TIFF
  LZW/JPEG/SVG/PDF, dpi, transparência, recorte, tamanho em cm com
  predefinições, tema claro/escuro, fontes instaladas, cores, nomes das
  classes, pasta e nome na própria janela (sem sobrescrever), um arquivo por
  intervalo; GIF com duração por quadro e dpi; exportações em segundo plano.
- Projeção: miniatura com rótulo, previsão, confiança e incerteza ao passar o
  mouse sobre um ponto.
- Clustering: clique em um nó do dendrograma corta logo abaixo dele; painel
  com os grupos do corte (tamanho, composição, 4 imagens mais típicas por
  cosseno ao centroide), ver o grupo inteiro, rotular o grupo, abrir na Galeria.
- Segmentação: prévia ao vivo original | máscara | sobreposição com o método
  e o raio atuais, navegação ◀ ▶ pelo escopo e histograma objeto/fundo com o
  limiar (Otsu, fixo, percentil).
- Exportar: aba EcoTaxa (TSV no formato de importação + cópia das imagens).

### Desempenho (2026-11-12, etapa 3) — `docs/METRICAS_E_REFERENCIAS.md`, seção 8
- Treino e inferência: bfloat16 em GPUs Ampere ou mais novas (sem *loss
  scaling*), float16 nas anteriores; `torch.compile` opcional
  (`ILT_TORCH_COMPILE=1`); lote de inferência `auto` pela VRAM livre e lote de
  treino `auto` opcional, ambos reduzidos pela metade em falta de memória.
- Inferência não chama mais `seed_everything`/`cudnn.deterministic` por batch
  (`freeze_bn` sem efeito em ConvNeXt removido).
- IAT › Projeção, análise estrutural: kNN e PCA na GPU com `torch` a partir de
  3 000 imagens (CPU/scikit-learn abaixo ou sem CUDA; `ILT_IAT_GPU=0`
  desliga); UMAP e PaCMAP usam todos os núcleos a partir de 20 000 imagens
  (`umap-learn` roda em uma só thread quando há semente fixa).
- IAT › Segmentação: lote processado em paralelo (CPUs − 1 threads, máx. 8,
  `ILT_SEG_WORKERS`); "Parar" não espera a fila terminar.
- IAT › Galeria (web): miniaturas carregadas só quando entram na área visível;
  cache do navegador por 7 dias com validação `ETag` (304 quando nada mudou,
  inclusive após regerar a segmentação); cache em memória 6 000.

### Treino — receita recomendada (2026-11-12, etapa 2) — `docs/METRICAS_E_REFERENCIAS.md`, seção 7
- Novo card "Receita de treino" (clássica e web) com presets **Recomendada** e
  **Legada (1.0)**: normalização ImageNet; imagem completada em quadrado com a
  cor de fundo em vez de esticada; augmentation no treino (espelhos, rotação
  0–360°, brilho/contraste ±10 %); AdamW (wd 0,05, sem decaimento em
  norms/bias); warmup linear + redução da LR no platô (ou cosseno); LR 10× na
  camada classificadora; label smoothing 0,1 (CE); stochastic depth 0,1.
- Focal loss própria (`ilt_losses.py`) a partir de `log_softmax`, sem a
  sincronização de GPU por batch do pacote `focal-loss-torch`, que deixou de
  ser dependência.
- Checkpoint formato 3: grava receita, pré-processamento, arquitetura e nomes
  das classes. A classificação lê o pré-processamento do checkpoint; modelos
  antigos continuam sendo classificados com o pipeline antigo. Retomar treino
  exige a mesma receita; checkpoints antigos são retomados com a receita legada.
- `stats_<projeto>.csv` ganhou a coluna `LR`; o log mostra a LR por época.

### Correções científicas (2026-11-12) — detalhes e referências em `docs/METRICAS_E_REFERENCIAS.md`
- Treino: o campo "Divisor (macro)" foi removido; as métricas macro são sempre
  a média sobre as classes reais (um divisor diferente na configuração gera
  aviso e é ignorado).
- Treino: modelo já existente na pasta de saída não trava mais o worker à
  espera de `input()`; a interface pergunta Sobrescrever / Retomar / Cancelar e
  o worker roda com `stdin` fechado (`ILT_NONINTERACTIVE=1`).
- IAT: a coluna `var_norm` ("Variância"), que nunca era calculada, passa a ser a
  incerteza pela margem `1 − (top-1 − top-2)`; preenchida na classificação e,
  para projetos antigos, ao abrir no IAT. Rótulos, histograma, filtro e
  exportação (`uncertainty_margin`) atualizados.
- Segmentação (medidas versão 2): perímetro pelo estimador de Crofton (4
  direções) em vez de passos de pixel; Feret máximo/mínimo verdadeiros
  (*rotating calipers* sobre a envoltória dos pixels) em vez dos lados do
  retângulo mínimo; solidez e retangularidade com a mesma geometria (≤ 1);
  novas colunas ESD, eixos maior/menor, excentricidade, orientação, biovolume
  elipsoidal, desvio/mín./máx./assimetria/curtose da intensidade, altura da
  imagem e `Metrics Version`. Segmentações antigas continuam legíveis
  (avisadas como v1); gere de novo para obter as medidas atuais.
- Análise Temporal: faixas de densidade relativas a um pico por classe comum a
  todos os quadros (opção "Escala de densidade comum", ligada por padrão) —
  quadros, comparações A/B/C, animação e GIF passam a ser comparáveis.
- Removido código sem uso: `inference.py`, geração de fundos em `data.py`,
  `projections.tsne_fit`, página de validação assistida (web e clássica) e a
  página-ponte do IAT.

### Correções (2026-11-12)
- Treino: a tabela de classes do dataset, a pré-visualização das classes do
  modelo e o texto de detecção não apareciam (o conteúdo recebia o estilo no
  lugar do texto em saídas `allow_duplicate`); o mesmo afetava o status do
  modelo na Classificação. Verificado em navegador real (renomear/agrupar,
  percentuais, número de imagens, desmarcar classes).
- Faixa de abas do IAT não aparece mais fora do IAT.

### Instalação e GPU
- Pacotes para Windows 11, Ubuntu 24.04 e Ubuntu 26.04.
- Detecção automática da placa de vídeo (`src/gpu_profile.py`): NVIDIA
  Maxwell–Turing → CUDA 11.8; Ampere/Ada/Hopper → CUDA 12.4; Blackwell
  (RTX 50xx) → CUDA 12.8 / PyTorch 2.7.1; demais → CPU. Teste real na GPU
  após a instalação, com troca automática para CPU se falhar.
- Dependências com versões fixadas (`requirements-common.txt`) e perfis
  `requirements-{cpu,cu118,cu124,cu128}.txt`; `pyswisseph` opcional.
- Windows: `INSTALAR.bat`, `ILT.bat`, `ILT_classico.bat`, `DIAGNOSTICO.bat`;
  Python 3.12 privado baixado do python.org (SHA-256 verificado) quando
  necessário; atalho na Área de Trabalho.

### Interface web (ajustes, rodada 2)
- Correção principal: botões que abriam janelas sem alterar a página (Sobre,
  fechar ✕, exportar figura/GIF da Análise Temporal, avaliação do modelo no
  ciclo) geravam erro interno do Dash (resposta “sem mudança” inválida para
  saídas curinga); corrigido.
- Galeria: após rotular, as marcações antigas somem imediatamente (o estado
  marcado de cada carta vem do servidor a cada redesenho).
- Galeria: marcação de imagens tratada no navegador (um único manipulador de
  clique), instantânea mesmo com 500 imagens; janelas fecham com ✕, Esc ou
  clique fora.
- Projeção: seleção (laço/retângulo/clique) atualiza só o gráfico; filtros,
  rótulos, contagens e manchas do Overview ficam em cache enquanto os dados não
  mudam (laço em 677 mil imagens: ~0,2 s). Escala do painel de classes
  funciona e o painel tem rolagem. Filtro por nome da imagem removido (a Análise
  Temporal cobre esse uso).
- Histograma de variância avisa quando o projeto não tem valores (`var_norm`
  vazio).
- Ciclo integrado: campo de pixel size (registrado no `training_config` do
  modelo); seleção de batches sem reler os CSVs.
- Log do IAT: removida a mensagem “Diagnóstico Tk” (não se aplica à versão web).

### Interface web (ajustes após uso com projeto real)
- Projeção e Temporal: listas de classes voltaram a ter altura (eram comprimidas
  pelo painel lateral); menu de batch passa a abrir acima dos gráficos.
- Galeria: clique na imagem marca/desmarca (Shift+clique marca intervalo) sem
  redesenhar a página; janela da imagem fecha no ✕.
- “Sobre”, “Configurações” e confirmações: identificadores duplicados nas
  janelas impediam a abertura; corrigido (verificação automática sem duplicatas).
- Temporal: manchas desenhadas como imagem transparente (mesmas faixas e
  opacidades da versão clássica), sem preencher o fundo; comparação com três
  painéis alinhada; legenda lateral.
- Validação assistida removida da interface web (continua na clássica).
- Jogos: imagem cabe na carta da memória; “Onde está o mascote?” volta a exibir
  o cenário.

### Interface web (etapa 2 — IAT completo e jogos)
- Todas as subabas do IAT na interface web: Projeção (WebGL, laço/retângulo,
  manchas por classe), Galeria (miniaturas, Shift+clique, similaridade,
  imagem em tamanho real com barra de escala), Validação assistida, Análise
  Temporal (Plotly; exportação PNG/SVG/PDF/GIF com o mesmo `render_figure`),
  Clustering (dendrograma WebGL), Treino + Classificação integrado,
  Segmentação (equações e referências), Exportar (CSV, imagens, dataset de
  treino) e Logs. `web/iat_core.py` executa os métodos originais do IAT.
- Correção: `_training_batch_from_gid` e `_training_unique_path` não eram
  `staticmethod`; a exportação de dataset de treino falhava com escopo
  “selecionadas”/Overview (também na interface clássica).
- Jogos do mascote: níveis Fácil, Médio e Difícil (Difícil = jogo original) e
  gráfico de evolução (tempo, tentativas, erros e acertos dentro e entre níveis).

### Interface web (etapa 1)
- Nova interface em Dash/Plotly no estilo HUD (painéis translúcidos, cantos
  arredondados, fundo preto), aberta no navegador local: Treino, Classificação
  e nova aba Aparência (temas, cores, fontes, transparência, raio; salvo em
  `input/ui_theme.json`).
- A lógica de Treino/Classificação não foi reescrita: `web/controller.py`
  executa os mesmos métodos do dashboard clássico (`TrainingDatasetUIMixin`,
  `ILTProcessMixin`, `ILTApp`) por um adaptador sem Tk.
- Log, progresso e gráficos atualizados por deltas no navegador (WebGL);
  monitor de GPU/VRAM via `nvidia-smi` na barra lateral.
- `./run_gui.sh --classic` mantém a interface Tk.

## 1.0.0 — 2026-11-11 (primeira versão estável publicada)

Consolida a linha interna 6.x (último build interno: 6.8.1-stable + revisões
de 09–10/11/2026) em uma versão estável única.

### Desempenho / hardware
- **Treino:** acurácia, perda e matriz de confusão acumuladas na GPU e lidas
  uma vez por época (antes: 3 sincronizações GPU→CPU por lote). Laços de
  treino/validação/teste unificados em `_run_epoch`.
- **Treino:** workers do DataLoader automáticos (`numWorkers = -1`); antes o
  padrão era 0, com decodificação de imagens no processo principal.
- **Classificação:** extração de features com workers automáticos (antes 1).
- **t-SNE (openTSNE / scikit-learn):** multithread (antes 1 thread).
- **`channels_last`** em GPUs com Tensor Cores (RTX 20xx+); Pascal mantém NCHW.
- **cuDNN benchmark** apenas quando `seed = -1` (preserva o determinismo).
- **Clustering por classe:** cache de embeddings limitado por RAM em vez de
  4 batches fixos; cada `.npz` é descomprimido uma vez, não uma vez por classe.
- **Validação assistida:** concordância kNN vetorizada (antes um laço Python por
  ponto); resultados idênticos.
- **Galeria:** decodificação reduzida de JPEG para miniaturas e pool de
  threads proporcional às CPUs.
- **CLI `inference.py`:** workers automáticos, `inference_mode` e uma cópia
  GPU→CPU por lote.
- Novos módulos `torch_runtime.py` (GPU) e `hw_resources.py` (CPU/RAM, sem torch).

### Correções
- Monitor da classificação: as regex esperavam “PNG scan complete”, mas o
  `batch.py` imprime “Image scan complete”. Os totais não eram detectados e a
  interface lia o `batches.csv` inteiro na thread do Tk.
- `ScrollableFrame` acumulava um `bind_all` da roda do mouse a cada troca de
  página; agora há um único despachante.
- Callbacks de idioma e cache de fontes cresciam a cada navegação (widgets
  destruídos); agora são descartados com a página.
- Log de execução limitado (6 000 linhas) e fila de saída processada em blocos,
  evitando congelamentos com saída muito rápida.
- `RangeSlider` recalculava o histograma a cada movimento do mouse.
- `RoundedPanel` redesenhava várias vezes por evento de layout.

### Interface / contraste
- Botões primários: texto escuro sobre verde (9,1:1; antes branco 2,1:1).
- Cor do texto escolhida automaticamente pelo fundo em botões, selos e
  indicadores (`ui_contrast.py`, critério WCAG AA 4,5:1).
- Bordas de cartões/campos mais visíveis; texto “desabilitado” legível (4,9:1).
- Indicadores de seleção do ciclo integrado em azul mais profundo (5,45:1).

### Interface (revisão de layout)
- Tema preto neutro com texto branco; tons de azul-escuro fixos convertidos
  para cinzas neutros; faixa em gradiente azul→violeta sob a barra superior.
- Cartões do IAT com cantos arredondados (`RoundedPanel`) e marcador em gradiente.
- Botões mais discretos (menor padding, bordas finas) no IAT e na Análise Temporal.
- **Análise Temporal e Espacial:** paleta de classes curada e muito distinta
  (calculada sobre as classes exibidas; ΔE CIELAB mínimo 27,5 até 20 classes);
  correção da faixa branca ao redor da figura (o redimensionamento assumia 100 dpi);
  redesenho no redimensionamento com *debounce*.
- **Clustering:** dendrograma sem números nos eixos; margens fixas no lugar de
  `tight_layout`; ao arrastar o corte reutiliza seleção e artefato já
  carregados (sem reler batches).
- **Treino + Classificação (Integrado):** “Como funciona?” substituído por um
  log em tempo real do processamento da própria subaba (linhas de progresso
  do tqdm são filtradas).
- **Segmentação:** equação implementada e referência científica de cada método
  (`segmentation_docs.py`) e definições das métricas.

### Documentação
- README reescrito; históricos `CHANGELOG_*`, `QA_*`, relatórios e logs de
  build/setup removidos. Conteúdo ainda relevante movido para `docs/`.

## Histórico interno resumido (6.x)

- **6.8.x:** port Linux completo da versão Windows; perfis P4000/RTX 4090/CPU;
  pixel size por projeto; datasets flexíveis (pasta de classes ou
  train/valid/test) com agrupamento/renomeação; ciclo Treino + Classificação
  integrado no IAT; contratos de rótulos e clustering centralizados.
- **Revisões 09–10/11/2026:** Análise Temporal e Espacial (eixos data, hora,
  profundidade e atributo; comparação e animação; ciclo lunar astronômico);
  filtros por nome de imagem no Overview; exportação de figuras PNG/PDF/SVG/GIF.
