# IAT — Análise Temporal e Espacial

Subaba somente leitura: não altera CSVs, rótulos, embeddings nem projeções.
Código: `src/iat_temporal_analysis.py` (leitura/filtros),
`src/iat_temporal_spatial.py` (interface), `src/iat_temporal_figure.py` e
`src/iat_temporal_export_ui.py` (figuras), `src/iat_temporal_lunar.py` (ciclo lunar),
`src/iat_origin_flow.py` e `src/web/pages/iat_origin.py` (painel Origem → Previsão).

## Recursos
- Eixos: Data (dias), Hora (minutos), Profundidade (mm) e Atributo por regex
  com grupo de captura. Filtros combináveis de data, hora, profundidade,
  atributo e classes.
- Modos: Individual; Comparar dois intervalos; Animação (pausa, velocidade,
  navegação).
- Usa por padrão as colunas `x`/`y` existentes (t-SNE da classificação). Em
  **Projeção (coordenadas)** é possível trocar por PCA, t-SNE recalculado, UMAP,
  **UMAP supervisionado** (rótulos manuais) ou PaCMAP, calculados a partir dos
  embeddings de todas as imagens carregadas e guardados em cache em
  `<projeto>/projections/`. O UMAP supervisionado mostra a estrutura dada a
  anotação e não serve para procurar erros de rótulo. Mesma escala e cores
  determinísticas por classe em cada comparação.
- **Evolução entre intervalos** (web): contagem por classe, composição (%) e
  diversidade (S, Hill N1 = exp H′, Hill N2 = 1/Σp²; H′ e Pielou J no resumo e no
  CSV). Clique em um ponto para ir ao intervalo. "Exportar contagens (CSV)"
  grava a tabela em `<projeto>/exports/temporal/`.
- **Salvar figura** (web): prévia fiel, PNG/TIFF/JPEG/SVG/PDF, dpi, fundo
  transparente, recorte de margens, tamanho em cm com predefinições, tema
  claro/escuro, fontes instaladas, cores de cada texto, nomes de classes,
  pasta e nome do arquivo na própria janela e "um arquivo por intervalo"; GIF
  com duração por quadro e dpi. Detalhes em `METRICAS_E_REFERENCIAS.md`, seção 9.
- **Origem → Previsão** (web): diagrama alluvial ou Sankey com duas ou três
  colunas escolhidas livremente: classe da pasta de origem (`source_label`,
  gravada em toda classificação, com ou sem auditoria), classe prevista
  (`preds1`) pelo modelo atual ou por uma versão anterior guardada pelo ciclo
  integrado, e rótulo revisado (`manual_label`). Recorte por intervalo ou todos
  os intervalos, cores por qualquer coluna, faixas pequenas ocultáveis, junção
  de grafias (maiúsculas/espaços), grupos e nomes só de exibição ("Agrupar /
  renomear…"), concordância, maior divergência e tabela para as duas primeiras
  colunas. "Salvar figura…" (PNG/TIFF/JPEG/SVG/PDF) e "Exportar tabela (CSV)".
  Detalhes em `METRICAS_E_REFERENCIAS.md`, seção 10.
- **Agrupar / renomear classes** (web): matriz com sugestões para unir ou
  renomear classes em todos os gráficos e exportações da aba (rótulo revisado)
  e nas colunas Origem e Previsão do alluvial. Cada linha mostra como as imagens
  do nome se distribuem nas classes previstas; a classe dominante é sugerida e
  pode ser aceita de uma vez acima de um limite. Os rótulos das imagens não
  mudam; os nomes ficam em `exports/temporal/grupos_de_exibicao.json`.
  Detalhes em `METRICAS_E_REFERENCIAS.md`, seção 10.
- Manchas apenas para classes com ≥ 20 imagens; contornos desligados por padrão.
- **Escala de densidade comum a todos os quadros** (ligada por padrão): as três
  faixas de cada classe (17/38/64 %) são relativas ao maior pico suavizado que a
  classe atinge em qualquer quadro do recorte, e não ao pico do próprio quadro.
  Assim o quadro mais denso mantém as faixas completas e os demais ficam
  proporcionalmente mais tênues; Individual, Comparar, Animação e GIF passam a
  ser comparáveis. Desligada, cada quadro usa o próprio pico (comportamento
  anterior; um quadro com 20 imagens parece tão denso quanto um com 2 000).
  Referências: Tufte 1983 (*small multiples*), Cleveland 1993 (*Visualizing
  Data*); detalhes em `METRICAS_E_REFERENCIAS.md`.
- Exportação PNG/PDF/SVG/GIF com títulos A/B/C, nomes de classe editáveis e
  dimensões em polegadas; a prévia usa o mesmo tamanho físico da exportação.

## Filtro por nome da imagem (Overview)
Datas reconhecidas: `YYYYMMDD`, `YYYY-MM-DD`, `YYYY_MM_DD`, `YYYY.MM.DD`,
`DD-MM-YYYY`, `DD_MM_YYYY`, `DD.MM.YYYY`, `YYYYMMDDHHMMSS`.
Horas: `HHMMSS`, `HH-MM-SS`, `HH_MM_SS`, `HH:MM:SS`, `HHMM` após separador.
Exemplo: `estacaoB_20261103_143055_calanus.png` → 2026-11-03, 14:30:55.
Regex com grupo nomeado: `(?P<valor>estacao[A-Z])_`.
Imagens sem data/hora reconhecível ficam fora de filtros que exigem esse campo.

## Ciclo lunar astronômico
- Em *Eixo de evolução: Data*, escolha **Lunar (astronômico)** e informe
  latitude/longitude (graus decimais, Sul/Oeste negativos) e o fuso UTC do
  relógio das imagens (ex.: `-3`). Horário de verão não é inferido.
- Usa `pyswisseph` (Swiss Ephemeris, Moshier) para a conjunção Sol–Lua
  topocêntrica; intervalos de Lua nova a Lua nova em horário local.
- Fase lunar não equivale a visibilidade, marés ou iluminação local.
- Instalação em ambiente antigo:
  `./.venv/bin/python -m pip install "pyswisseph>=2.10.3,<3"`

## Limitações científicas
- Coordenadas `x,y` de batches diferentes só são comparáveis se calculadas no
  **mesmo referencial de projeção**; a interface não consegue certificar isso a
  partir dos CSVs. Selecionar a versão das features não comprova o referencial.
- t-SNE privilegia estrutura local; diferenças de densidade, forma e distância
  não provam, isoladamente, alterações biológicas (Wattenberg, Viégas & Johnson
  2016, *How to use t-SNE effectively*, Distill).
- Contagens de imagens não são abundância normalizada por esforço amostral.
- No painel Origem → Previsão, a concordância com a pasta só equivale a
  acurácia quando as pastas foram conferidas por especialistas; se as pastas
  forem estações, datas ou lotes, o diagrama apenas mostra a pasta dos arquivos.
- Classes com mais de 4 000 pontos são amostradas apenas para desenho; os
  totais usam todos os dados.

## Licença da dependência astronômica
A Swiss Ephemeris (via `pyswisseph`) tem licenciamento duplo (AGPL /
profissional). Para redistribuir o ILT ou integrá-lo a produto não compatível
com AGPL, confirme as obrigações ou obtenha licença adequada.
