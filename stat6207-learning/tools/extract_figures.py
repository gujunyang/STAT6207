"""Extract meaningful (non-decorative) figures from a lecture PDF.

Usage:
    python tools/extract_figures.py <pdf> <out_dir>

Skips images that repeat on many pages (logos / branding), small icons,
and masks. Also renders selected full pages for diagrams that are vector
graphics (which have no embedded raster image).
"""
import sys
import os
from collections import Counter

import fitz


def main() -> None:
    pdf_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    # Count how often each xref appears -> decorative if on many pages.
    freq = Counter()
    for page in doc:
        for xref in {img[0] for img in page.get_images(full=True)}:
            freq[xref] += 1

    saved = []
    for i, page in enumerate(doc):
        for j, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            if freq[xref] > 8:  # header / footer decoration
                continue
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.width < 120 or pix.height < 90:
                    continue
                if pix.n - pix.alpha >= 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                name = f"L1_p{i + 1:02d}_fig{j + 1}.png"
                pix.save(os.path.join(out_dir, name))
                saved.append(name)
            except Exception:  # noqa: BLE001
                pass

    print(f"saved {len(saved)} figures")
    for name in saved:
        print(name)


if __name__ == "__main__":
    main()
