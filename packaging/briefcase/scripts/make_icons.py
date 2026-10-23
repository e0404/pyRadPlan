"""Render the application icons and installer images from the pyRadPlan logo SVGs.

Produces the files Briefcase looks for next to ``icon = ".../pyradplan"``:
``pyradplan.ico`` (Windows), ``pyradplan.icns`` (macOS) and ``pyradplan-<N>.png``
(Linux), and the two bitmaps of the Windows installer (``installer_background`` and
``installer_banner``): ``installer-dialog.bmp``, the side panel of the welcome and
finish pages, and ``installer-banner.bmp``, the banner across the top of the other
pages. Requires PySide6 (SVG rendering) and Pillow (ICO/ICNS/BMP writing), both
available in a ``pyRadPlan[gui]`` development environment.
"""

import sys
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

HERE = Path(__file__).resolve().parent
LOGOS = HERE.parents[2] / "pyRadPlan" / "gui" / "assets" / "logos"
SVG = LOGOS / "pyradplan_logo_skull_square.svg"
WORDMARK_SVG = LOGOS / "pyradplan_logo_full_landscape.svg"
#: Left edge of the "pyRadPlan" lettering in the landscape logo's viewBox; the skull
#: left of it is cropped off, as the side panel shows the skull on its own.
WORDMARK_LEFT = 335.0
OUT = HERE.parent / "src" / "pyradplan_app" / "resources"
SIZES = (16, 32, 64, 128, 256, 512, 1024)

#: Sizes of the WiX dialog bitmap (welcome/finish pages) and banner bitmap (other pages).
DIALOG_SIZE = (493, 312)
BANNER_SIZE = (493, 58)
#: Width of the dialog bitmap's logo panel; WiX draws the page text right of it.
PANEL_WIDTH = 164
PANEL_COLOR = QColor("#eef3f8")


def _to_pil(image: QImage) -> Image.Image:
    image = image.convertToFormat(QImage.Format.Format_ARGB32)
    ptr = image.constBits()
    size = (image.width(), image.height())
    return Image.frombuffer("RGBA", size, bytes(ptr), "raw", "BGRA", 0, 1)


def render(renderer: QSvgRenderer, size: int) -> Image.Image:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    return _to_pil(image)


def render_installer_images(skull: QSvgRenderer, wordmark: QSvgRenderer) -> dict[str, Image.Image]:
    """Return the installer's dialog and banner bitmaps (opaque RGB), by file name."""
    dialog = QImage(*DIALOG_SIZE, QImage.Format.Format_RGB32)
    dialog.fill(QColor("white"))
    painter = QPainter(dialog)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.fillRect(0, 0, PANEL_WIDTH, DIALOG_SIZE[1], PANEL_COLOR)
    logo, top = 128, 66
    skull.render(painter, QRectF((PANEL_WIDTH - logo) / 2, top, logo, logo))
    view = wordmark.viewBoxF()
    width = PANEL_WIDTH - 24
    target = QRectF(12, top + logo + 18, width, width * view.height() / view.width())
    # The renderer draws the whole drawing, also outside a cropped viewBox.
    painter.setClipRect(target)
    wordmark.render(painter, target)
    painter.end()

    banner = QImage(*BANNER_SIZE, QImage.Format.Format_RGB32)
    banner.fill(QColor("white"))
    painter = QPainter(banner)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    logo = BANNER_SIZE[1] - 12
    skull.render(painter, QRectF(BANNER_SIZE[0] - logo - 10, 6, logo, logo))
    painter.end()

    return {
        "installer-dialog.bmp": _to_pil(dialog).convert("RGB"),
        "installer-banner.bmp": _to_pil(banner).convert("RGB"),
    }


def main() -> int:
    app = QGuiApplication.instance() or QGuiApplication(["make_icons", "-platform", "offscreen"])
    renderer = QSvgRenderer(str(SVG))
    assert renderer.isValid(), SVG
    wordmark = QSvgRenderer(str(WORDMARK_SVG))
    assert wordmark.isValid(), WORDMARK_SVG
    view = wordmark.viewBoxF()
    wordmark.setViewBox(
        QRectF(WORDMARK_LEFT, view.top(), view.right() - WORDMARK_LEFT, view.height())
    )
    OUT.mkdir(parents=True, exist_ok=True)
    images = {size: render(renderer, size) for size in SIZES}
    for size, img in images.items():
        img.save(OUT / f"pyradplan-{size}.png")
    images[256].save(OUT / "pyradplan.ico", sizes=[(s, s) for s in SIZES if s <= 256])
    images[1024].save(OUT / "pyradplan.icns")
    for name, img in render_installer_images(renderer, wordmark).items():
        img.save(OUT / name)
    print(f"wrote icons and installer images to {OUT}")
    del app
    return 0


if __name__ == "__main__":
    sys.exit(main())
