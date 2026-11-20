# Integração futura do ILT Viewer

O ILT centraliza clustering e contratos de rótulos. Um `app.py`/Viewer
externo deve consumir esses módulos em vez de copiar a lógica.

## APIs estáveis a reutilizar

### Clustering puro — `src/ilt_clustering.py`
- `hierarchical_linkage(...)`
- `save_cluster_artifact(...)`
- `read_cluster_artifact(path)`
- `cluster_groups(artifact, cut01)`
- `validate_clustering_request(...)`
- `l2_normalize(...)`
- `apply_pca_if_needed(...)`

O Viewer não deve chamar `scipy.cluster.hierarchy.fcluster` diretamente para
reimplementar o `cut`: use `cluster_groups(...)`.

### Operações sobre um projeto — `src/iat_backend.py`
- `IATProject(project_dir)`
- `gather_embeddings(project, rows, ...)`
- `generate_cluster_artifact(project, class_label, ...)`
- `generate_selection_cluster_artifact(project, rows, ...)`
- `reconcile_class_cluster_artifacts(project, ...)`

Essas funções já resolvem a associação entre rows, embeddings e artifacts no
formato do projeto.

### Contratos de labels — `src/ilt_contracts.py`
- `resolve_prediction_column(columns)`
- `resolve_scatter_class_column(color_mode, columns)`
- `sanitize_filename(value)`
- `ensure_unique_sanitized_labels(labels)`
- `validate_embedding_matrix(...)`

Semântica:
- `manual_label`: decisão humana atual;
- `source_label`: origem/pasta preservada;
- `preds1`/prediction: previsão independente do modelo;
- `gid`: identificador estável entre batches.

## Ordem recomendada para o Viewer

1. Abrir projeto e escolher batch apenas na UI.
2. Mostrar scatter/ordihull sem alterar CSVs.
3. Ler o artifact da classe e desenhar dendrograma.
4. Alterar o `cut` chamando `cluster_groups(...)`, ainda somente leitura.
5. Mostrar grupos paginados (por exemplo 8.000 pontos/página).
6. Somente após testes de paridade, habilitar relabel project-wide usando
   `IATProject.apply_labels(...)` + `save_dirty()` e, ao final,
   `reconcile_class_cluster_artifacts(...)`.

A etapa 6 deve ser transacional do ponto de vista da UI: preview primeiro,
confirmação explícita, gravação depois.

## O que NÃO deve existir no Viewer

Evite implementações próprias equivalentes a:
- `_load_cluster_artifact` se apenas repetir `read_cluster_artifact`;
- `fcluster(...)`/cache de grupos se repetir `cluster_groups`;
- normalização/PCA própria para o clustering;
- regras próprias de nomes de `.npz`;
- heurísticas próprias para escolher `manual_label`, `source_label` ou previsão.

Este pacote não contém o Viewer `app.py`; ele deve ser integrado como
consumidor desses contratos quando o Viewer for fornecido.
