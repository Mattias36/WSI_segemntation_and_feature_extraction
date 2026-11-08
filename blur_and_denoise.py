"""
Etap 3.2 potoku: Detekcja nieostrości i filtracja szumów (Blur Detection & Noise Filtering).

Założenia z Sekcji 3 (Etap 2) specyfikacji projektu:
1. Detekcja nieostrości (Blur Detection):
   - Wyznaczenie operatora Laplasjana: ∇²I = ∂²I/∂x² + ∂²I/∂y²
   - Ocena ostrości kafelka poprzez wariancję Laplasjana: Var(∇²I) (Pech-Pacheco et al., 2000).
   - Wyznaczenie lokalnej mapy ostrości (local sharpness map) w ruchomym oknie (np. 31x31 px),
     pozwalającej wykryć zarówno globalny blur optyczny, jak i lokalną utratę ostrości
     na zagnieceniach preparatu (tissue folds).
2. Filtracja szumów (Noise Filtering):
   - Wygładzenie szumów wysokoczęstotliwościowych (ziarnistość barwnika, szum matrycy oraz
     artefakty kompresji JPEG ze skanów WSI) nieliniowym filtrem medianowym (median filter),
   - Zachowanie ostrych krawędzi granic jąder komórkowych (edge-preserving smoothing) przed
     progowaniem Otsu i transformacją wododziałową (Watershed) w Etapie 3.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Domyślny próg wariancji Laplasjana (w skali szarości RGB) poniżej którego kafelek
# uznawany jest za dotknięty nieostrością optyczną (optical blur).
BLUR_THRESHOLD_GRAY = 300.0

# Domyślny próg wariancji Laplasjana na 8-bitowym kanale Hematoksyliny (H).
BLUR_THRESHOLD_H = 420.0

# Domyślny rozmiar jądra filtru medianowego (3x3 lub 5x5 dla kafelków 256x256 px).
DEFAULT_MEDIAN_KSIZE = 3


@dataclass
class BlurAndFilterResult:
    """Wynik detekcji nieostrości i filtracji medianowej pojedynczego obrazu/kanału."""
    laplacian: np.ndarray           # (H, W) float64 – odpowiedź operatora Laplasjana ∇²I
    sharpness_var: float            # globalna wariancja Laplasjana Var(∇²I) przed filtracją
    sharpness_after_median: float   # wariancja Laplasjana po filtracji medianowej
    is_blurry: bool                 # flaga nieostrości (sharpness_var < threshold)
    local_sharpness_map: np.ndarray # (H, W) float64 – lokalna wariancja Laplasjana w oknie
    filtered: np.ndarray            # (H, W) lub (H, W, 3) uint8 – obraz po filtrze medianowym
    edge_preservation_corr: float   # korelacja gradientu Sobela przed i po filtracji medianowej


def compute_laplacian(img: np.ndarray, ksize: int = 1) -> np.ndarray:
    """
    Wyznacza dyskretny operator Laplasjana ∇²I (cv2.CV_64F, domyślnie standardowa maska 3x3: ksize=1).
    Jeśli podano obraz wielokanałowy RGB, konwertuje go najpierw do skali szarości.
    """
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(arr, cv2.CV_64F, ksize=ksize)


def laplacian_variance(img: np.ndarray, mask: np.ndarray | None = None,
                       ksize: int = 1) -> float:
    """
    Oblicza wariancję operatora Laplasjana Var(∇²I) jako miarę ostrości obrazu.
    Opcjonalnie ogranicza obliczenia do maski tkanki (mask=True).
    """
    lap = compute_laplacian(img, ksize=ksize)
    if mask is not None and np.any(mask):
        return float(np.var(lap[mask]))
    return float(lap.var())


def local_laplacian_variance_map(img: np.ndarray, win_size: int = 31) -> np.ndarray:
    """
    Wyznacza lokalną mapę ostrości (lokalną wariancję Laplasjana w oknie win_size x win_size):
        Var_local(L) = E[L²] - (E[L])²
    Umożliwia wizualizację obszarów lokalnego rozmycia (np. na zagnieceniach tkanki).
    """
    lap = compute_laplacian(img)
    mean_l = cv2.boxFilter(lap, cv2.CV_64F, (win_size, win_size), normalize=True)
    mean_l2 = cv2.boxFilter(lap * lap, cv2.CV_64F, (win_size, win_size), normalize=True)
    return np.clip(mean_l2 - mean_l * mean_l, 0.0, None)


def apply_median_filter(img: np.ndarray, ksize: int = DEFAULT_MEDIAN_KSIZE) -> np.ndarray:
    """
    Wygładza szumy filtrem medianowym o nieparzystym rozmiarze okna ksize (np. 3 lub 5).
    Filtr medianowy usuwa szum impulsowy i artefakty blokowe JPEG przy zachowaniu
    ostrych krawędzi jąder komórkowych.
    """
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError(f"Rozmiar okna filtru medianowego ksize musi być nieparzysty >= 3, podano {ksize}")
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.medianBlur(arr, ksize)


def _sobel_gradient_magnitude(img_u8: np.ndarray) -> np.ndarray:
    """Pomocnicza funkcja obliczająca moduł gradientu Sobela do oceny zachowania krawędzi."""
    if img_u8.ndim == 3:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(img_u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_u8, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def analyze_and_filter(img_u8: np.ndarray,
                       ksize: int = DEFAULT_MEDIAN_KSIZE,
                       blur_threshold: float = BLUR_THRESHOLD_H,
                       win_size: int = 31) -> BlurAndFilterResult:
    """
    Pełny przebieg Etapu 3.2 dla pojedynczego obrazu (np. 8-bitowej mapy stężenia Hematoksyliny
    lub obrazu w skali szarości):
      1. wyznaczenie operatora Laplasjana i jego wariancji Var(∇²I),
      2. klasyfikacja ostrości względem progu blur_threshold,
      3. wyznaczenie lokalnej mapy ostrości (local_sharpness_map),
      4. filtracja szumów filtrem medianowym (cv2.medianBlur),
      5. ocena zachowania krawędzi (korelacja modułu gradientu Sobela przed i po filtracji).
    """
    arr = np.asarray(img_u8)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    lap = compute_laplacian(arr)
    sharp_before = float(lap.var())
    local_map = local_laplacian_variance_map(arr, win_size=win_size)

    filtered = apply_median_filter(arr, ksize=ksize)
    sharp_after = float(compute_laplacian(filtered).var())

    g_before = _sobel_gradient_magnitude(arr).ravel()
    g_after = _sobel_gradient_magnitude(filtered).ravel()
    if np.std(g_before) > 0 and np.std(g_after) > 0:
        edge_corr = float(np.corrcoef(g_before, g_after)[0, 1])
    else:
        edge_corr = 1.0

    return BlurAndFilterResult(
        laplacian=lap,
        sharpness_var=sharp_before,
        sharpness_after_median=sharp_after,
        is_blurry=bool(sharp_before < blur_threshold),
        local_sharpness_map=local_map,
        filtered=filtered,
        edge_preservation_corr=edge_corr,
    )
