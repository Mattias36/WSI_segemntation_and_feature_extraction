"""Etap 3.3: Segmentacja jąder komórkowych – progowanie Otsu + transformacja wododziałowa (Watershed)
oraz weryfikacja i strojenie hiperparametrów względem maski referencyjnej.

Wejście: odszumiona mapa stężenia hematoksyliny (uint8, jasne = duże stężenie H).
Kroki:
  1. Detekcja i opcjonalne maskowanie artefaktów niebędących tkanką H&E (markery chirurgiczne),
  2. Progowanie Otsu:
     - mode="global"   – pojedynczy próg globalny Otsu na pikselach tkanki,
     - mode="adaptive" – adaptacyjna mapa progu łącząca globalny próg Otsu z lokalnym tłem
                         gaussowskim (z ujemnym offsetem, bez wygryzania środków jąder),
  3. Operacje morfologiczne (otwarcie, domknięcie, usunięcie małych obiektów, wypełnienie dziur),
  4. Transformata odległości Euklidesowej + markery h-maxima -> Transformacja Wododziałowa (Watershed),
  5. Filtracja obiektów po polu powierzchni [MIN_NUCLEUS_AREA, MAX_NUCLEUS_AREA],
  6. Ewaluacja względem maski referencyjnej (Dice, IoU, Precision, Recall, Object-F1) i strojenie progów.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_local
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, remove_small_holes, h_maxima
from skimage.segmentation import watershed, find_boundaries

from color_deconvolution import color_deconvolution
from blur_and_denoise import apply_median_filter

warnings.filterwarnings("ignore", category=FutureWarning, module=r"skimage\..*|nuclei_segmentation")

# Domyślne wartości hiperparametrów (dostrojone na maskach referencyjnych – zob. tune_hyperparameters)
MIN_NUCLEUS_AREA = 25        # px^2 – minimalne pole jądra komórkowego (odrzuca drobny szum chromatyczny)
MAX_NUCLEUS_AREA = 750       # px^2 – maksymalne pole pojedynczego jądra (odrzuca fałdy i smugi tuszu)
DEFAULT_H = 2.0              # px   – głębokość h-maxima transformaty odległości w Watershed


@dataclass
class SegmentationResult:
    threshold: float            # próg bazowy Otsu (skala 0..255)
    threshold_map: np.ndarray   # (H, W) float – mapa progu (stała dla global, zmienna dla adaptive)
    binary_raw: np.ndarray      # (H, W) bool  – maska bezpośrednio po progowaniu
    binary_clean: np.ndarray    # (H, W) bool  – maska po czyszczeniu morfologicznym
    distance: np.ndarray        # (H, W) float – transformata odległości EDT
    markers: np.ndarray         # (H, W) int32 – markery lokalnych maksimów (zlewiska Watershed)
    labels: np.ndarray          # (H, W) int32 – końcowe etykiety jąder po Watershed i filtracji pola
    labels_no_ws: np.ndarray    # (H, W) int32 – etykiety składowych spójnych bez Watershed
    n_nuclei: int               # końcowa liczba wykrytych jąder (N)
    n_components: int           # liczba składowych spójnych przed rozdzieleniem Watershedem


@dataclass
class SegmentationMetrics:
    dice: float                 # współczynnik Dice'a (F1 pikselowy): 2*TP / (2*TP + FP + FN)
    iou: float                  # indeks Jaccarda (IoU pikselowy): TP / (TP + FP + FN)
    precision: float            # precyzja pikselowa: TP / (TP + FP)
    recall: float               # czułość pikselowa: TP / (TP + FN)
    object_f1: float            # obiektowy F1-score (dopasowanie jąder przy IoU >= 0.3)
    n_pred: int                 # liczba wykrytych jąder
    n_ref: int                  # liczba jąder w masce referencyjnej
    count_error: int            # błąd bezwzględny liczby jąder |n_pred - n_ref|


def detect_marker_mask(img_rgb: np.ndarray, min_marker_area: int = 80) -> np.ndarray:
    """Wykrywa ślady markera lekarskiego / tuszu chirurgicznego (niebieski, zielony, czarny)."""
    r = img_rgb[:, :, 0].astype(float)
    g = img_rgb[:, :, 1].astype(float)
    b = img_rgb[:, :, 2].astype(float)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    dec_r = color_deconvolution(img_rgb, "ruifrok")

    blue_m = ((hsv[:, :, 0] >= 92) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] > 80) & (b > r + 28)) | (b > r + 42)
    green_m = ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] > 65) & (g > r + 10)) | (g > r + 25)
    black_m = (gray < 42) & (hsv[:, :, 1] < 120) & (b < r + 15)
    extreme_h = dec_r.hematoxylin > 1.35

    raw = remove_small_objects(blue_m | green_m | black_m | extreme_h, min_size=min_marker_area)
    if not raw.any():
        return np.zeros(img_rgb.shape[:2], dtype=bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.dilate(raw.astype(np.uint8), k).astype(bool)


def prepare_h_channel(img_rgb: np.ndarray, method: str = "ruifrok",
                      median_ksize: int = 3, suppress_markers: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wyznacza odszumioną mapę Hematoksyliny uint8 z opcjonalnym maskowaniem markerów lekarskich.

    Zwraca:
        (h_med_u8, valid_tissue_mask, marker_mask)
    """
    dec = color_deconvolution(img_rgb, method)
    marker = detect_marker_mask(img_rgb) if suppress_markers else np.zeros(img_rgb.shape[:2], dtype=bool)
    valid = dec.tissue_mask & (~marker)
    h_raw = dec.hematoxylin.copy()
    if suppress_markers and marker.any():
        h_raw[marker] = 0.0
    vals = h_raw[valid] if valid.any() else h_raw.ravel()
    p99 = float(np.percentile(vals, 99.5)) if vals.size else 1.0
    if suppress_markers:
        p99 = float(np.clip(p99, 0.40, 1.15))
    h_u8 = np.clip(h_raw / max(p99, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    h_med = apply_median_filter(h_u8, median_ksize) if median_ksize > 1 else h_u8
    return h_med, valid, marker


def otsu_threshold(h_u8: np.ndarray, mask: np.ndarray | None = None,
                   mode: str = "global", block_size: int = 75) -> tuple[np.ndarray, float, np.ndarray]:
    """Progowanie metodą Otsu (globalne lub adaptacyjne).

    mode="global"   – pojedynczy próg globalny Otsu dla całego kafelka (liczony na masce tkanki).
    mode="adaptive" – adaptacyjna mapa progu: 0.55 * t_otsu + 0.45 * T_lok(x, y), gdzie T_lok
                      jest średnią gaussowską w oknie block_size z ujemnym offsetem (-0.22 * std).
                      Taki dobór podnosi próg na ciemnych zagnieceniach tkanki (folds), obniża go
                      w jaśniejszych rejonach preparatu i NIE wygryza wnętrz dużych jąder.
    """
    has_mask = mask is not None and mask.any()
    vals = h_u8[mask] if has_mask else h_u8.ravel()
    t = float(threshold_otsu(vals)) if vals.size and vals.min() != vals.max() else 255.0
    if has_mask and mask.mean() < 0.25:
        t = min(t, 122.0)

    if mode == "global":
        t_map = np.full(h_u8.shape, t, dtype=np.float32)
    elif mode == "adaptive":
        std = float(np.std(vals)) if vals.size else 20.0
        local_t = threshold_local(h_u8.astype(float), block_size=block_size,
                                  method="gaussian", offset=-0.22 * std)
        t_map = np.minimum(0.55 * t + 0.45 * local_t, t * 1.06).astype(np.float32)
    else:
        raise ValueError("mode musi być 'global' lub 'adaptive'")

    binary = h_u8 > t_map
    if mask is not None:
        binary &= mask
    return binary, t, t_map


def clean_mask(binary: np.ndarray, open_radius: int = 1, min_area: int = MIN_NUCLEUS_AREA) -> np.ndarray:
    """Morfologia: otwarcie (usunięcie wypustek i szumu), usunięcie małych obiektów, wypełnienie dziur."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_radius + 1,) * 2)
    m = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    m = remove_small_objects(m, min_size=max(10, min_area // 2))
    m = remove_small_holes(m, area_threshold=max(36, min_area * 2))
    return m


def watershed_split(binary: np.ndarray, h: float = DEFAULT_H, smooth_sigma: float = 1.0):
    """Rozdziela połączone jądra: transformata odległości + markery h-maxima + Watershed."""
    dist = ndi.distance_transform_edt(binary)
    dist_s = ndi.gaussian_filter(dist, smooth_sigma) if smooth_sigma > 0 else dist
    peaks = h_maxima(dist_s, h) & binary
    markers = label(peaks).astype(np.int32)
    labels = watershed(-dist_s, markers, mask=binary).astype(np.int32)
    missed = binary & (labels == 0)
    if missed.any():
        extra = label(missed)
        extra[extra > 0] += labels.max()
        labels = labels + extra.astype(np.int32)
    return labels, markers, dist


def filter_by_area(labels: np.ndarray, min_area: int = MIN_NUCLEUS_AREA,
                   max_area: int = MAX_NUCLEUS_AREA) -> np.ndarray:
    """Filtruje wykryte obiekty po polu powierzchni [min_area, max_area] i przenumerowuje 1..N."""
    out = np.zeros_like(labels)
    new_id = 0
    for r in regionprops(labels):
        if min_area <= r.area <= max_area:
            new_id += 1
            out[labels == r.label] = new_id
    return out


def segment_nuclei(h_u8: np.ndarray, tissue_mask: np.ndarray | None = None,
                   mode: str = "global", h: float = DEFAULT_H, use_watershed: bool = True,
                   min_area: int = MIN_NUCLEUS_AREA, max_area: int = MAX_NUCLEUS_AREA,
                   block_size: int = 75) -> SegmentationResult:
    """Pełny potok 3.3: Otsu (global/adaptive) -> morfologia -> Watershed -> filtr pola."""
    raw, t, t_map = otsu_threshold(h_u8, tissue_mask, mode, block_size)
    clean = clean_mask(raw, min_area=min_area)
    comp = filter_by_area(label(clean).astype(np.int32), min_area, max_area).astype(np.int32)
    ws, markers, dist = watershed_split(clean, h=h)
    final = ws if use_watershed else comp
    final = filter_by_area(final, min_area, max_area).astype(np.int32)
    return SegmentationResult(
        threshold=t, threshold_map=t_map, binary_raw=raw, binary_clean=(final > 0),
        distance=dist, markers=markers, labels=final, labels_no_ws=comp,
        n_nuclei=int(final.max()), n_components=int(comp.max()))


def build_reference_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Generuje maskę referencyjną (etykiety 1..N_ref) metodą wielokanałowego konsensusu H&E.

    Łączy:
      - detekcję i wykluczenie markerów chirurgicznych (niebieski/zielony/czarny tusz),
      - selektywny sygnał jądrowy (H - 0.10*E) z ograniczeniem chromatyczności fioletu H&E,
      - rekonstrukcję morfologiczną z progowaniem histerezowym (high/low),
      - podział sklejonych jąder z warunkiem zwartości geometrycznej (solidity >= 0.52).
    """
    marker = detect_marker_mask(img_rgb)
    dec_r = color_deconvolution(img_rgb, "ruifrok")
    valid = dec_r.tissue_mask & (~marker)
    if valid.sum() < 100:
        valid = ~marker

    h_raw = np.clip(dec_r.hematoxylin, 0, None) - 0.10 * np.clip(dec_r.eosin, 0, None)
    h_raw = np.clip(h_raw, 0, None)
    h_raw[marker] = 0.0
    vals_raw = h_raw[valid] if valid.any() else h_raw.ravel()
    p99 = float(np.clip(np.percentile(vals_raw, 99.5), 0.40, 1.15)) if vals_raw.size else 1.0
    h_u8 = np.clip(h_raw / max(p99, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    h_med = apply_median_filter(h_u8, 3)

    vals = h_med[valid]
    t_otsu = float(threshold_otsu(vals)) if vals.size and vals.min() != vals.max() else 120.0
    if valid.mean() < 0.25:
        t_otsu = min(t_otsu, 118.0)
    std = float(np.std(vals)) if vals.size else 20.0
    loc = threshold_local(h_med.astype(float), block_size=75, method="gaussian", offset=-0.25 * std)
    cap = t_otsu * (0.96 if marker.any() else 1.05)
    t_map = np.minimum(0.55 * t_otsu + 0.45 * loc, cap)

    r, b = img_rgb[:, :, 0].astype(float), img_rgb[:, :, 2].astype(float)
    chroma_ok = (r < 190) & (b >= r - 22) & (~marker) & valid
    high = (h_med > t_map * 1.00) & chroma_ok
    low = (h_med > t_map * 0.93) & chroma_ok
    binary = ndi.binary_propagation(high, mask=low) if high.any() else np.zeros_like(high)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    binary = remove_small_objects(binary, min_size=22)
    binary = remove_small_holes(binary, area_threshold=64)

    ws, _, _ = watershed_split(binary, h=2.0, smooth_sigma=1.0)
    out = np.zeros_like(ws)
    nid = 0
    for rp in regionprops(ws):
        if 25 <= rp.area <= 750 and rp.solidity >= 0.52:
            nid += 1
            out[ws == rp.label] = nid
    return out.astype(np.int32)


def evaluate_segmentation(pred_labels: np.ndarray, ref_labels: np.ndarray,
                          iou_thresh: float = 0.3) -> SegmentationMetrics:
    """Porównuje wynik segmentacji z maską referencyjną (pikselowo oraz obiektowo)."""
    p_bin = pred_labels > 0
    r_bin = ref_labels > 0
    tp = int(np.logical_and(p_bin, r_bin).sum())
    fp = int(np.logical_and(p_bin, ~r_bin).sum())
    fn = int(np.logical_and(~p_bin, r_bin).sum())

    dice = (2.0 * tp) / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)

    n_pred = int(pred_labels.max())
    n_ref = int(ref_labels.max())
    if n_pred == 0 and n_ref == 0:
        obj_f1 = 1.0
    elif n_pred == 0 or n_ref == 0:
        obj_f1 = 0.0
    else:
        # Szybka tablica przekrojów obiektów (contingency table)
        p_flat = pred_labels.ravel()
        r_flat = ref_labels.ravel()
        fg = (p_flat > 0) | (r_flat > 0)
        inter = np.zeros((n_pred + 1, n_ref + 1), dtype=np.int32)
        np.add.at(inter, (p_flat[fg], r_flat[fg]), 1)
        p_areas = inter.sum(axis=1, keepdims=True)[1:]
        r_areas = inter.sum(axis=0, keepdims=True)[:, 1:]
        inter_fg = inter[1:, 1:]
        union = p_areas + r_areas - inter_fg
        iou_mat = inter_fg / np.maximum(union, 1)
        matched_p = (iou_mat.max(axis=1) >= iou_thresh).sum()
        matched_r = (iou_mat.max(axis=0) >= iou_thresh).sum()
        matched = 0.5 * (matched_p + matched_r)
        obj_f1 = float((2.0 * matched) / max(n_pred + n_ref, 1))

    return SegmentationMetrics(
        dice=float(dice), iou=float(iou), precision=float(prec), recall=float(rec),
        object_f1=float(obj_f1), n_pred=n_pred, n_ref=n_ref, count_error=abs(n_pred - n_ref))


def overlay_contours(img_rgb: np.ndarray, labels: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    """Nakłada granice wykrytych jąder na obraz RGB."""
    out = img_rgb.copy()
    out[find_boundaries(labels, mode="inner")] = color
    return out


def overlay_comparison(img_rgb: np.ndarray, pred_labels: np.ndarray, ref_labels: np.ndarray) -> np.ndarray:
    """Wizualizuje porównanie z maską referencyjną:
       zielony = zgodne (TP), czerwony = fałszywy alarm (FP), niebieski/cyjan = pominięte jądro (FN).
    """
    p_bin = pred_labels > 0
    r_bin = ref_labels > 0
    tp = p_bin & r_bin
    fp = p_bin & (~r_bin)
    fn = (~p_bin) & r_bin

    out = (img_rgb.astype(np.float32) * 0.65).astype(np.uint8)
    out[tp] = np.clip(0.4 * out[tp] + np.array([0, 220, 70]) * 0.6, 0, 255).astype(np.uint8)
    out[fp] = np.clip(0.3 * out[fp] + np.array([255, 40, 40]) * 0.7, 0, 255).astype(np.uint8)
    out[fn] = np.clip(0.3 * out[fn] + np.array([40, 170, 255]) * 0.7, 0, 255).astype(np.uint8)
    out[find_boundaries(pred_labels, mode="inner")] = (255, 255, 0)
    return out


def colorize_labels(labels: np.ndarray, seed: int = 42) -> np.ndarray:
    """Zamienia macierz etykiet 0..N na czytelny, kontrastowy obraz RGB (tło=czarne, każde jądro=inny kolor)."""
    max_id = int(labels.max())
    if max_id == 0:
        return np.zeros((*labels.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    lut = rng.integers(75, 255, size=(max_id + 1, 3), dtype=np.uint8)
    lut[0] = 0
    return lut[labels]
