#!/usr/bin/env python3
"""Equations and scientific references for the IAT segmentation methods.

Static, presentation-only data.  Each entry describes what
``iat_backend._threshold_mask`` actually computes for that method; where the
implementation is a simplification of the published method this is stated
explicitly.  Notation: I(x,y) is the 8-bit gray image, M the binary mask.
"""
from __future__ import annotations

COMMON_POSTPROCESS = {
    'pt': 'Pós-processamento comum: M ← M ∘ B₃ₓ₃ (abertura morfológica) e '
          'maior componente conexo (vizinhança-8).',
    'en': 'Common post-processing: M ← M ∘ B₃ₓ₃ (morphological opening) and '
          'largest connected component (8-neighbourhood).',
}

AUTO_POLARITY = {
    'pt': 'Auto-polaridade (heurística do ILT): entre a máscara escura e a clara '
          'escolhe a de menor  s = |f − 0,18| + 0,7·b  (f = fração da imagem '
          'ocupada, b = fração da borda tocada).',
    'en': 'Auto-polarity (ILT heuristic): between the dark and light masks keep '
          'the one minimising  s = |f − 0.18| + 0.7·b  (f = image fraction '
          'covered, b = fraction of the border touched).',
}

OTSU_REF = ('Otsu, N. (1979). A threshold selection method from gray-level '
            'histograms. IEEE Trans. Systems, Man, and Cybernetics, 9(1), 62–66.')
OTSU_EQ = ('t* = argmaxₜ σ²_B(t),   σ²_B(t) = ω₀(t)·ω₁(t)·[μ₀(t) − μ₁(t)]²\n'
           'M = [ I ≤ t* ]  (objeto escuro)  ou  [ I > t* ]  (objeto claro)')

SEGMENTATION_DOCS = {
    'otsu': {
        'eq': OTSU_EQ,
        'note': {'pt': 'Limiar global; polaridade automática.',
                 'en': 'Global threshold; automatic polarity.'},
        'polarity': True,
        'ref': OTSU_REF,
    },
    'otsu_lcc': {
        'eq': OTSU_EQ + '\nM ← maior componente conexo de M (área mínima)',
        'note': {'pt': 'Otsu seguido de filtro pelo maior componente.',
                 'en': 'Otsu followed by largest-component filtering.'},
        'polarity': True,
        'ref': OTSU_REF + '\nRosenfeld, A. & Pfaltz, J. L. (1966). Sequential operations '
               'in digital picture processing. Journal of the ACM, 13(4), 471–494.',
    },
    'rolling_ball': {
        'eq': 'B = G_σ * I   (kernel gaussiano k = 2r + 1)\n'
              'D = max(B − I, 0)\n'
              'M = [ D > t*_Otsu(D) ]',
        'note': {'pt': 'Implementação aproxima o fundo do “rolling ball” por um '
                       'desfoque gaussiano de raio r; depois Otsu sobre o resíduo.',
                 'en': 'The implementation approximates the rolling-ball background '
                       'with a Gaussian blur of radius r, then Otsu on the residual.'},
        'polarity': False,
        'ref': 'Sternberg, S. R. (1983). Biomedical image processing. '
               'Computer, 16(1), 22–34.\n' + OTSU_REF,
    },
    'fixed_243': {
        'eq': 'M = [ I < 243 ]   (objeto escuro em fundo claro)\n'
              'M = [ I > 12 ]    (polaridade invertida)',
        'note': {'pt': 'Limiar fixo no estilo ZooProcess/ZooScan.',
                 'en': 'Fixed threshold in the ZooProcess/ZooScan style.'},
        'polarity': True,
        'ref': 'Gorsky, G. et al. (2010). Digital zooplankton image analysis using '
               'the ZooScan integrated system. Journal of Plankton Research, '
               '32(3), 285–303.',
    },
    'black_hat_otsu': {
        'eq': 'BH(I) = (I • B) − I      (fechamento por elipse B de diâmetro 2r + 1)\n'
              'M = [ BH > t*_Otsu(BH) ]',
        'note': {'pt': 'Realça estruturas escuras menores que B antes do limiar.',
                 'en': 'Enhances dark structures smaller than B before thresholding.'},
        'polarity': False,
        'ref': 'Serra, J. (1982). Image Analysis and Mathematical Morphology. '
               'Academic Press.\n' + OTSU_REF,
    },
    'adaptive_sauvola': {
        'eq': 'T(x,y) = m(x,y) · [ 1 + k · ( s(x,y) / R − 1 ) ]\n'
              'k = 0,34 ;  R = 0,5 ;  janela (2r + 1)² ;  I normalizada em [0, 1]\n'
              'M = [ I < T ]',
        'note': {'pt': 'm e s: média e desvio-padrão locais.',
                 'en': 'm and s: local mean and standard deviation.'},
        'polarity': True,
        'ref': 'Sauvola, J. & Pietikäinen, M. (2000). Adaptive document image '
               'binarization. Pattern Recognition, 33(2), 225–236.',
    },
    'percentile_dark': {
        'eq': 'T = P₃₅(I)   (percentil 35 das intensidades)\n'
              'M = [ I ≤ T ]',
        'note': {'pt': 'Método p-tile: assume que ~35 % dos pixels são do objeto escuro.',
                 'en': 'P-tile method: assumes ~35 % of pixels belong to the dark object.'},
        'polarity': True,
        'ref': 'Doyle, W. (1962). Operations useful for similarity-invariant pattern '
               'recognition. Journal of the ACM, 9(2), 259–267.',
    },
    'watershed_otsu': {
        'eq': 'M₀ = Otsu(I) com auto-polaridade\n'
              'D(p) = min_{q ∉ M₀} ‖p − q‖₂   (transformada de distância)\n'
              'M = maior componente de [ D > 0,45 · max D ]',
        'note': {'pt': 'Usa apenas a etapa de marcadores (núcleos) do watershed; '
                       'não executa a inundação completa.',
                 'en': 'Uses only the marker (core) step of watershed; the full '
                       'flooding is not performed.'},
        'polarity': True,
        'ref': 'Beucher, S. & Lantuéjoul, C. (1979). Use of watersheds in contour '
               'detection. Int. Workshop on Image Processing, Rennes.\n'
               'Borgefors, G. (1986). Distance transformations in digital images. '
               'Computer Vision, Graphics, and Image Processing, 34(3), 344–371.',
    },
    'kmeans_intensity': {
        'eq': 'min_{μ₁,μ₂,μ₃} Σᵢ min_j (Iᵢ − μ_j)²     (k = 3, k-means++)\n'
              'M = [ rótulo(I) = argmin_j μ_j ]   (agrupamento mais escuro)',
        'note': {'pt': 'Agrupa intensidades em 3 classes e mantém a mais escura.',
                 'en': 'Clusters intensities into 3 groups and keeps the darkest.'},
        'polarity': True,
        'ref': 'Lloyd, S. (1982). Least squares quantization in PCM. IEEE Trans. '
               'Information Theory, 28(2), 129–137.\n'
               'Arthur, D. & Vassilvitskii, S. (2007). k-means++: the advantages of '
               'careful seeding. Proc. 18th ACM-SIAM SODA, 1027–1035.',
    },
}

METRIC_FORMULAS = {
    'pt': ('A = nº de pixels de M (px²; µm² = px² · s²)      P = perímetro de Crofton, 4 direções (px; µm = px · s)\n'
           'Circularidade = 4πA / P²  (≤ 1)      ESD = 2·√(A/π)      s = pixel size (µm/pixel)\n'
           'Feret_max = maior distância entre vértices da envoltória convexa dos pixels\n'
           'Feret_min = menor largura da envoltória (rotating calipers)      Alongamento = Feret_max / Feret_min\n'
           'Retangularidade = A / (w·h)_retângulo de área mínima      Solidez = A / A_envoltória convexa  (≤ 1)\n'
           'Eixos Maior/Menor = elipse com os mesmos momentos centrais de 2ª ordem (4·√λ₁, 4·√λ₂)\n'
           'Excentricidade = √(1 − (Menor/Maior)²)      Biovolume = π·Maior·Menor²/6  (elipsoide; µm³ = px³ · s³)\n'
           'Intensidade: média, mediana, desvio-padrão, mín., máx., assimetria e curtose (Fisher) dos pixels de M\n'
           'Contraste = max(I_M) − min(I_M)      % no bordo = 100 · pixels de M na borda / A\n'
           'Toda a geometria usa o conjunto de pixels (cada pixel = quadrado unitário), por isso Solidez e '
           'Retangularidade nunca passam de 1.\n'
           'Referências: Legland, Kiêu & Devaux (2007) Image Anal. Stereol. 26:83–92; Lehmann & Legland (2012) '
           'Crofton perimeter; Toussaint (1983) rotating calipers; Hu (1962) IRE Trans. Inf. Theory 8:179–187; '
           'Gorsky et al. (2010) J. Plankton Res. 32:285–303; Vandromme et al. (2012) Methods Oceanogr. 1–2:3–21.'),
    'en': ('A = number of pixels of M (px²; µm² = px² · s²)      P = Crofton perimeter, 4 directions (px; µm = px · s)\n'
           'Circularity = 4πA / P²  (≤ 1)      ESD = 2·√(A/π)      s = pixel size (µm/pixel)\n'
           'Feret_max = largest distance between vertices of the convex hull of the pixels\n'
           'Feret_min = smallest width of the hull (rotating calipers)      Elongation = Feret_max / Feret_min\n'
           'Rectangularity = A / (w·h)_minimum-area rectangle      Solidity = A / A_convex hull  (≤ 1)\n'
           'Major/Minor axes = ellipse with the same second central moments (4·√λ₁, 4·√λ₂)\n'
           'Eccentricity = √(1 − (Minor/Major)²)      Biovolume = π·Major·Minor²/6  (ellipsoid; µm³ = px³ · s³)\n'
           'Intensity: mean, median, standard deviation, min, max, skewness and (Fisher) kurtosis of the pixels of M\n'
           'Contrast = max(I_M) − min(I_M)      % on border = 100 · border pixels of M / A\n'
           'All geometry uses the pixel set (each pixel = unit square), so Solidity and Rectangularity never exceed 1.\n'
           'References: Legland, Kiêu & Devaux (2007) Image Anal. Stereol. 26:83–92; Lehmann & Legland (2012) '
           'Crofton perimeter; Toussaint (1983) rotating calipers; Hu (1962) IRE Trans. Inf. Theory 8:179–187; '
           'Gorsky et al. (2010) J. Plankton Res. 32:285–303; Vandromme et al. (2012) Methods Oceanogr. 1–2:3–21.'),
}

# Fields available for ordering the Gallery and for export (shown in the UI).
METRIC_FIELDS_TEXT = {
    'pt': ('Área (px² / µm²) • Perímetro de Crofton (px / µm) • ESD (px / µm) • Circularidade (adim.) • '
           'Alongamento = Feret máx./mín. (adim.) • Retangularidade (adim.) • Solidez (adim.) • '
           'Feret máx./mín. (px / µm) • Eixos maior/menor (px / µm) • Excentricidade (adim.) • '
           'Biovolume elipsoidal (px³ / µm³) • Intensidade média/mediana/desvio (0–255) • Contraste (0–255) • '
           'Percentual no bordo (%) • Número de orifícios (contagem). Segmentações antigas (sem “Metrics Version”) '
           'trazem Feret = lados do retângulo mínimo e perímetro em passos de pixel; gere a segmentação de novo para '
           'obter as medidas atuais.'),
    'en': ('Area (px² / µm²) • Crofton perimeter (px / µm) • ESD (px / µm) • Circularity (dimensionless) • '
           'Elongation = max/min Feret (dimensionless) • Rectangularity (dimensionless) • Solidity (dimensionless) • '
           'Max/min Feret (px / µm) • Major/minor axes (px / µm) • Eccentricity (dimensionless) • '
           'Ellipsoidal biovolume (px³ / µm³) • Mean/median/std intensity (0–255) • Contrast (0–255) • '
           'Percent on border (%) • Number of holes (count). Older segmentations (no “Metrics Version”) carry '
           'Feret = sides of the minimum rectangle and pixel-step perimeter; regenerate the segmentation to obtain '
           'the current measurements.'),
}


def method_doc_text(method: str, lang: str = 'pt') -> tuple[str, str]:
    """Return ``(equation_block, reference_block)`` ready for display."""
    lang = 'en' if lang == 'en' else 'pt'
    doc = SEGMENTATION_DOCS.get(method) or SEGMENTATION_DOCS['otsu']
    eq = doc['eq']
    if lang == 'en':
        eq = (eq.replace('(objeto escuro)', '(dark object)')
                .replace('(objeto claro)', '(light object)')
                .replace('(objeto escuro em fundo claro)', '(dark object on light background)')
                .replace('(polaridade invertida)', '(inverted polarity)')
                .replace('(fechamento por elipse B de diâmetro 2r + 1)',
                         '(closing with ellipse B of diameter 2r + 1)')
                .replace('(percentil 35 das intensidades)', '(35th intensity percentile)')
                .replace('(transformada de distância)', '(distance transform)')
                .replace('maior componente de', 'largest component of')
                .replace('maior componente conexo de M (área mínima)',
                         'largest connected component of M (minimum area)')
                .replace('com auto-polaridade', 'with auto-polarity')
                .replace('(kernel gaussiano k = 2r + 1)', '(Gaussian kernel k = 2r + 1)')
                .replace('janela', 'window').replace('normalizada em', 'normalised to')
                .replace('rótulo', 'label').replace('(agrupamento mais escuro)', '(darkest cluster)')
                .replace('0,34', '0.34').replace('0,45', '0.45').replace('0,5', '0.5'))
    lines = [eq, '', doc['note'][lang]]
    if doc.get('polarity'):
        lines.append(AUTO_POLARITY[lang])
    lines.append(COMMON_POSTPROCESS[lang])
    head = 'Referência' if lang == 'pt' else 'Reference'
    return '\n'.join(lines), f'{head}:\n{doc["ref"]}'
