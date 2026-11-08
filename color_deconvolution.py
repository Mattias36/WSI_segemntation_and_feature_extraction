"""
Etap 3.1 potoku: Wstępna filtracja i dekonwolucja barwna (Color Deconvolution).

Model fizyczny (prawo Lamberta-Beera):
    I_c = I_0 * exp(-sum_s  A_cs * C_s)            dla kanału c ∈ {R, G, B}
    OD_c = -log10(I_c / I_0) = sum_s A_cs * C_s

Gdzie:
    OD  – gęstość optyczna (liniowa względem stężenia barwnika, w przeciwieństwie do RGB),
    A   – macierz barwników (kolumny = znormalizowane wektory OD barwników H, E, rezydualny),
    C   – stężenia barwników w danym pikselu.

Separacja:  C = A^{-1} · OD   (Ruifrok & Johnston, 2001)

Dostępne macierze barwników:
    * "ruifrok"  – stała, literaturowa macierz H/E/DAB (Ruifrok & Johnston 2001),
    * "macenko"  – macierz estymowana adaptacyjnie z danego kafelka (SVD w przestrzeni OD,
                   Macenko et al. 2009) – odporna na zmienność barwienia (stain variability).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Natężenie światła padającego (tło szkiełka). 255 dla obrazów 8-bitowych.
I0_DEFAULT = 255.0

# Literaturowe wektory OD barwników (Ruifrok & Johnston 2001), wiersze: H, E, DAB.
RUIFROK_HED = np.array([
    [0.650, 0.704, 0.286],   # Hematoksylina
    [0.072, 0.990, 0.105],   # Eozyna
    [0.268, 0.570, 0.776],   # DAB / kanał rezydualny
])


@dataclass
class DeconvolutionResult:
    """Wynik dekonwolucji jednego kafelka."""
    od: np.ndarray            # (H, W, 3) gęstość optyczna
    concentrations: np.ndarray  # (H, W, 3) stężenia [H, E, residual]
    stain_matrix: np.ndarray  # (3, 3) wiersze = wektory OD barwników
    tissue_mask: np.ndarray   # (H, W) bool – piksele tkanki (OD > beta)

    @property
    def hematoxylin(self) -> np.ndarray:
        return self.concentrations[..., 0]

    @property
    def eosin(self) -> np.ndarray:
        return self.concentrations[..., 1]

    @property
    def residual(self) -> np.ndarray:
        return self.concentrations[..., 2]


# --------------------------------------------------------------------------- #
# Wstępna filtracja
# --------------------------------------------------------------------------- #
def prefilter_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """
    Wstępne przygotowanie obrazu przed transformacją do OD:
      * usunięcie kanału alfa (OpenSlide zwraca RGBA),
      * konwersja do float,
      * ograniczenie zakresu do [1, I0] – zabezpieczenie przed log(0) dla
        nasyconych czarnych pikseli (np. czarny marker / zagniecenia).
    """
    img = np.asarray(img_rgb)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"Oczekiwano obrazu RGB (H, W, 3), otrzymano {img.shape}")
    img = img[..., :3].astype(np.float64)
    return np.clip(img, 1.0, I0_DEFAULT)


def rgb_to_od(img_rgb: np.ndarray, i0: float = I0_DEFAULT) -> np.ndarray:
    """Transformacja RGB -> gęstość optyczna: OD = -log10(I / I0)."""
    img = prefilter_rgb(img_rgb)
    return -np.log10(img / i0)


def od_to_rgb(od: np.ndarray, i0: float = I0_DEFAULT) -> np.ndarray:
    """Transformacja odwrotna OD -> RGB (uint8)."""
    rgb = i0 * np.power(10.0, -od)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def tissue_mask_from_od(od: np.ndarray, beta: float = 0.15) -> np.ndarray:
    """Maska tkanki: piksel uznany za tkankę, jeśli OD w każdym kanale > beta (Macenko 2009)."""
    return np.all(od > beta, axis=-1)


# --------------------------------------------------------------------------- #
# Macierze barwników
# --------------------------------------------------------------------------- #
def _normalize_rows(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def _complete_stain_matrix(he: np.ndarray) -> np.ndarray:
    """Uzupełnia macierz 2x3 (H, E) trzecim wektorem ortogonalnym (kanał rezydualny)."""
    third = np.cross(he[0], he[1])
    third = np.abs(third)  # wymuszenie dodatnich wartości OD
    return _normalize_rows(np.vstack([he, third]))


def ruifrok_stain_matrix() -> np.ndarray:
    """Stała macierz H/E/rezydualny (Ruifrok & Johnston 2001)."""
    return _normalize_rows(RUIFROK_HED.copy())


def macenko_stain_matrix(od: np.ndarray, beta: float = 0.15, alpha: float = 1.0) -> np.ndarray:
    """
    Adaptacyjna estymacja wektorów barwników (Macenko et al., 2009):
      1. odrzucenie pikseli przezroczystych (OD < beta),
      2. SVD / rozkład własny macierzy kowariancji OD,
      3. rzut na płaszczyznę 2 największych wektorów własnych,
      4. odporne ekstrema kąta (percentyle alpha i 100-alpha) -> wektory H i E.
    Gdy tkanki jest za mało, następuje powrót do macierzy Ruifroka.
    """
    od_flat = od.reshape(-1, 3)
    od_hat = od_flat[np.all(od_flat > beta, axis=1)]
    if od_hat.shape[0] < 100:
        return ruifrok_stain_matrix()

    eigvals, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
    v_main = eigvecs[:, 2]
    if np.sum(v_main) < 0:
        v_main = -v_main
    # W kafelkach ze skompresowanych JPEG-iem skanów WSI (.svs) dwie mniejsze
    # wartości własne bywają niemal równe (eigvals[1] < 1.5 * eigvals[0]), a
    # szum bloków 8x8 w kanale B może zamienić miejscami wektory eigvecs[:, 0]
    # i eigvecs[:, 1]. Wówczas wybieramy kierunek o większej składowej w kanale R.
    if eigvals[1] < 2.5 * eigvals[0] and abs(eigvecs[0, 0]) > abs(eigvecs[0, 1]):
        v_sec = eigvecs[:, 0]
    else:
        v_sec = eigvecs[:, 1]
    plane = np.column_stack([v_main, v_sec])       # 2 główne kierunki w przestrzeni OD
    proj = od_hat @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    min_phi, max_phi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)

    v1 = plane @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v2 = plane @ np.array([np.cos(max_phi), np.sin(max_phi)])
    v1, v2 = np.abs(v1), np.abs(v2)

    # Heurystyka: hematoksylina ma większą składową w kanale R (absorbuje czerwień)
    he = np.vstack([v1, v2]) if v1[0] > v2[0] else np.vstack([v2, v1])
    return _complete_stain_matrix(_normalize_rows(he))


# --------------------------------------------------------------------------- #
# Dekonwolucja
# --------------------------------------------------------------------------- #
def color_deconvolution(img_rgb: np.ndarray, method: str = "ruifrok",
                        beta: float = 0.15) -> DeconvolutionResult:
    """
    Pełna dekonwolucja barwna kafelka.

    Args:
        img_rgb: obraz RGB uint8 (H, W, 3)
        method:  "ruifrok" (stała macierz) lub "macenko" (estymowana z obrazu)
        beta:    próg OD dla maski tkanki
    """
    od = rgb_to_od(img_rgb)
    if method == "ruifrok":
        stain = ruifrok_stain_matrix()
    elif method == "macenko":
        stain = macenko_stain_matrix(od, beta=beta)
    else:
        raise ValueError(f"Nieznana metoda: {method}")

    h, w, _ = od.shape
    # OD (N x 3) = C (N x 3) · S (3 x 3)   =>   C = OD · S^{-1}
    conc = od.reshape(-1, 3) @ np.linalg.inv(stain)
    conc = np.clip(conc, 0, None).reshape(h, w, 3)  # stężenia nie mogą być ujemne

    return DeconvolutionResult(od=od, concentrations=conc, stain_matrix=stain,
                               tissue_mask=tissue_mask_from_od(od, beta))


def stain_to_rgb(result: DeconvolutionResult, stain_idx: int) -> np.ndarray:
    """Rekonstrukcja obrazu RGB zawierającego tylko jeden barwnik (0=H, 1=E, 2=residual)."""
    c = result.concentrations[..., stain_idx:stain_idx + 1]      # (H, W, 1)
    od_single = c * result.stain_matrix[stain_idx][None, None, :]  # (H, W, 3)
    return od_to_rgb(od_single)


def channel_to_uint8(channel: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """
    Skalowanie mapy stężeń do uint8 [0, 255] z odcięciem ekstremów (percentyl),
    jasne = wysokie stężenie barwnika. Wejście dla progowania Otsu w etapie 3.3.
    """
    hi = np.percentile(channel, percentile)
    if hi <= 0:
        return np.zeros_like(channel, dtype=np.uint8)
    return (np.clip(channel / hi, 0, 1) * 255).astype(np.uint8)

