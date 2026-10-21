"""Extract text + images from a lecture PDF.

Usage:
    python tools/extract_pdf.py <pdf_path> <out_dir> [--text out.txt]

Writes <out_dir>/page_XX_img_N.png for every embedded raster image and
<out_dir>/../<name>.txt for the whole text (unless --text given).
"""
import sys
import os
import fitz  # PyMuPDF


def main() -> None:
    pdf_path = sys.argv[1]
    out_dir = sys.argv[2]
    text_path = None
    if "--text" in sys.argv:
        text_path = sys.argv[sys.argv.index("--text") + 1]
    else:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        text_path = os.path.join(out_dir, "..", base + ".txt")

    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    chunks = []
    n_img = 0

    for i, page in enumerate(doc):
        chunks.append(f"\n=====PAGE {i + 1}=====\n")
        chunks.append(page.get_text())
        images = page.get_images(full=True)
        chunks.append(f"[IMAGES] {len(images)}\n")
        for j, img in enumerate(images):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if pix.width < 40 or pix.height < 40:
                    continue
                name = f"page_{i + 1:02d}_img_{j + 1}.png"
                pix.save(os.path.join(out_dir, name))
                n_img += 1
            except Exception as exc:  # noqa: BLE001
                chunks.append(f"[image extract failed xref={xref}: {exc}]\n")

    with open(text_path, "w", encoding="utf-8") as fh:
        fh.write("".join(chunks))
    print(f"pages={doc.page_count} images_saved={n_img} text={text_path}")


if __name__ == "__main__":
    main()
