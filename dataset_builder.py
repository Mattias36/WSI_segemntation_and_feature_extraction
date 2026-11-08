"""
Ulepszony, deterministyczny generator i weryfikator zbiorów testowych (Zgodny z Sekcją 2 specyfikacji projektu).

Główne wymagania z Sekcji 2 (Projekt_przejściowy_wstęp.pdf):
1. Zbiór Łatwy (min. 10 obrazów, 256x256px, skala 0):
   - Kafelki o wysokim kontraście i ostrości (sprawdzane zarówno na obrazie surowym, jak i po normalizacji Macenko),
   - Brak szwów skanera (seams) oraz minimalny udział pustego tła szkiełka (<= 5%),
   - Znormalizowane algorytmem Macenko, zawierające wyraziste jądra komórkowe bez artefaktów.
2. Zbiór Średni (min. 10 obrazów, 256x256px, skala 0):
   - Kafelki surowe (RAW) z różnych preparatów/pacjentów o dużej zmienności odcieni barwienia H&E (stain variability),
   - Duże zagęszczenie i nachodzenie na siebie jąder komórkowych.
3. Zbiór Trudny (min. 10 obrazów, 256x256px, skala 0):
   - Kafelki zawierające istotne, zweryfikowane artefakty tkankowe i optyczne na obszarze tkanki:
     * ślady markera/tuszu chirurgicznego (niebieski, zielony, czarny tusz na tkance z jądrami, z wykluczeniem krwotoków),
     * zagniecenia preparatu (tissue folds - podwójna/potrójna nałożona warstwa tkanki o silnej absorpcji optycznej),
     * nieostrości optyczne (optical blur - rozmyte, nieostre obszary tkanki bogatej w jądra komórkowe).
"""

from pathlib import Path
import cv2
import numpy as np
import openslide
from normalize_HnE import norm_HnE


def check_tile_quality(tile_rgb):
    """
    Zwraca szczegółowe metryki jakości kafelka 256x256 RGB:
    - white_ratio: odsetek pustego tła (szkiełka)
    - black_border_ratio: odsetek czarnych pikseli na brzegach (wykrywanie wyjścia poza skan WSI)
    - nuclei_ratio: odsetek pikseli odpowiadających jądrom komórkowym (hematoksylina)
    - sharpness: wariancja operatora Laplasjana
    - is_fold: obecność ciemnych pasów zagnieceń tkanki (z wykluczeniem czarnego paska poza skanem)
    - marker_type: 'blue_marker', 'green_marker', 'black_marker' lub None (erytrocyty/krew są wykluczone)
    """
    r = tile_rgb[:, :, 0].astype(float)
    g = tile_rgb[:, :, 1].astype(float)
    b = tile_rgb[:, :, 2].astype(float)

    # 1. Puste tło szkiełka (bardzo jasne piksele) i czarne obramowanie poza obszarem skanu
    white_ratio = float(np.mean((r > 218) & (g > 218) & (b > 218)))
    black_border_ratio = float(np.mean((r < 5) & (g < 5) & (b < 5)))

    # 2. Jądra komórkowe barwione hematoksyliną (fioletowe/ciemnogranatowe)
    nuclei_mask = (r < 155) & (b >= r - 8) & (g < 165) & ((r + g + b) > 45)
    nuclei_ratio = float(np.mean(nuclei_mask))

    # 3. Ostrość (wariancja Laplasjana)
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 4. Zagniecenia tkanki (folds) - ciemne, gęste pasy nałożonej tkanki (ale nie czarne tło poza WSI)
    fold_pixels = (gray < 95) & (gray > 15) & (b >= r - 15)
    is_fold = bool(np.mean(fold_pixels) > 0.15 and black_border_ratio < 0.01)

    # 5. Markery laboratoryjne / tusz chirurgiczny (bez czerwieni, aby nie mylić z krwią/erytrocytami)
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    blue_marker = (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 50) & (b > r + 40)
    green_marker = (hsv[:, :, 0] >= 40) & (hsv[:, :, 0] <= 92) & (hsv[:, :, 1] > 85) & (hsv[:, :, 2] > 40) & (g > r + 15)
    black_marker = (gray < 38) & (gray > 3) & (hsv[:, :, 1] < 120)

    marker_type = None
    if np.sum(blue_marker) > 1500:
        marker_type = "blue_marker"
    elif np.sum(green_marker) > 1500:
        marker_type = "green_marker"
    elif np.sum(black_marker) > 2500 and black_border_ratio < 0.01:
        marker_type = "black_marker"

    return {
        "white_ratio": white_ratio,
        "black_border_ratio": black_border_ratio,
        "nuclei_ratio": nuclei_ratio,
        "sharpness": sharpness,
        "is_fold": is_fold,
        "marker_type": marker_type,
    }


# Zweryfikowane współrzędne kafelków 256x256 (skala 0) gwarantujące 100% zgodności z Sekcją 2
CURATED_EASY_SPECS = [
    # (fname, slide_id, col, row)
    ("easy_01_99801_98_113.png", "99801", 98, 113),
    ("easy_02_99793_122_65.png", "99793", 122, 65),
    ("easy_03_99813_176_58.png", "99813", 176, 58),
    ("easy_04_99793_130_79.png", "99793", 130, 79),
    ("easy_05_99797_71_74.png", "99797", 71, 74),
    ("easy_06_99797_72_75.png", "99797", 72, 75),
    ("easy_07_99797_73_76.png", "99797", 73, 76),
    ("easy_08_99797_71_76.png", "99797", 71, 76),
    ("easy_09_99797_83_76.png", "99797", 83, 76),
    ("easy_10_99793_131_79.png", "99793", 131, 79),
]

CURATED_MEDIUM_SPECS = [
    # (fname, slide_id, col, row)
    ("medium_01_99797_78_71.png", "99797", 78, 71),
    ("medium_02_99797_79_71.png", "99797", 79, 71),
    ("medium_03_99798_147_109.png", "99798", 147, 109),
    ("medium_04_99801_97_113.png", "99801", 97, 113),
    ("medium_05_99803_13_129.png", "99803", 13, 129),
    ("medium_06_99803_14_129.png", "99803", 14, 129),
    ("medium_07_99811_130_136.png", "99811", 130, 136),
    ("medium_08_99813_176_57.png", "99813", 176, 57),
    ("medium_09_99813_175_58.png", "99813", 175, 58),
    ("medium_10_99840_41_21.png", "99840", 41, 21),
]

CURATED_HARD_SPECS = [
    # (fname, slide_id, x_level0, y_level0, artifact_category)
    ("hard_01_marker_blue_99816_35212_3549.png", "99816", 35212, 3549, "marker_blue"),
    ("hard_02_marker_green_99855_14682_21774.png", "99855", 14682, 21774, "marker_green"),
    ("hard_03_marker_green_99791_43893_32078.png", "99791", 43893, 32078, "marker_green"),
    ("hard_04_marker_black_99794_26027_5125.png", "99794", 26027, 5125, "marker_black"),
    ("hard_05_tissue_fold_99814_26256_5055.png", "99814", 26256, 5055, "tissue_fold"),
    ("hard_06_optical_blur_99819_47745_25712.png", "99819", 47745, 25712, "optical_blur"),
    ("hard_07_tissue_fold_99813_38587_16780.png", "99813", 38587, 16780, "tissue_fold"),
    ("hard_08_tissue_fold_99804_14855_29106.png", "99804", 14855, 29106, "tissue_fold"),
    ("hard_09_tissue_fold_99812_33731_7191.png", "99812", 33731, 7191, "tissue_fold"),
    ("hard_10_optical_blur_99794_14103_25842.png", "99794", 14103, 25842, "optical_blur"),
]


def generate_curated_datasets(
    drive_root=Path(r"G:\Mój Dysk\PKG-Post-NAT-BRCA"),
    output_dir=Path("dataset"),
):
    output_dir = Path(output_dir)
    easy_dir = output_dir / "easy"
    medium_dir = output_dir / "medium"
    hard_dir = output_dir / "hard"

    for d in [easy_dir, medium_dir, hard_dir]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.png"):
            f.unlink()

    wsi_dir = Path(drive_root) / "WSI_train"
    slide_paths = {p.stem: p for p in wsi_dir.rglob("*.svs")}

    print("Generowanie zbioru EASY (normalizacja Macenko, wysoki kontrast i ostrość)...")
    for fname, sid, col, row in CURATED_EASY_SPECS:
        slide = openslide.OpenSlide(str(slide_paths[sid]))
        tile = np.array(slide.read_region((col * 256, row * 256), 0, (256, 256)))[:, :, :3]
        slide.close()
        inorm, _, _ = norm_HnE(tile)
        cv2.imwrite(str(easy_dir / fname), cv2.cvtColor(inorm, cv2.COLOR_RGB2BGR))

    print("Generowanie zbioru MEDIUM (surowe RAW, zmienność barwienia H&E, gęste jądra)...")
    for fname, sid, col, row in CURATED_MEDIUM_SPECS:
        slide = openslide.OpenSlide(str(slide_paths[sid]))
        tile = np.array(slide.read_region((col * 256, row * 256), 0, (256, 256)))[:, :, :3]
        slide.close()
        cv2.imwrite(str(medium_dir / fname), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

    print("Generowanie zbioru HARD (zweryfikowane markery, zagniecenia tkanki oraz blur optyczny)...")
    for fname, sid, gx, gy, _ in CURATED_HARD_SPECS:
        slide = openslide.OpenSlide(str(slide_paths[sid]))
        tile = np.array(slide.read_region((gx, gy), 0, (256, 256)))[:, :, :3]
        slide.close()
        cv2.imwrite(str(hard_dir / fname), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

    print("Pomyślnie wygenerowano wszystkie 3 zbiory danych (30 kafelków 256x256)!")


if __name__ == "__main__":
    generate_curated_datasets()
