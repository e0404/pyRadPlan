"""Render the application icons from the pyRadPlan logo SVG.

Produces the files Briefcase looks for next to ``icon = ".../pyradplan"``:
``pyradplan.ico`` (Windows), ``pyradplan.icns`` (macOS) and ``pyradplan-<N>.png``
(Linux). Requires PySide6 (SVG rendering) and Pillow (ICO/ICNS writing), both
available in a ``pyRadPlan[gui]`` development environment.
"""

import sys
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

HERE = Path(__file__).resolve().parent
SVG = (
    HERE.parents[2] / "pyRadPlan" / "gui" / "assets" / "logos" / "pyradplan_logo_skull_square.svg"
)
OUT = HERE.parent / "src" / "pyradplan_app" / "resources"
SIZES = (16, 32, 64, 128, 256, 512, 1024)


def render(renderer: QSvgRenderer, size: int) -> Image.Image:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    ptr = image.constBits()
    pil = Image.frombuffer("RGBA", (size, size), bytes(ptr), "raw", "BGRA", 0, 1)
    return pil


def main() -> int:
    app = QGuiApplication.instance() or QGuiApplication(["make_icons", "-platform", "offscreen"])
    renderer = QSvgRenderer(str(SVG))
    assert renderer.isValid(), SVG
    OUT.mkdir(parents=True, exist_ok=True)
    images = {size: render(renderer, size) for size in SIZES}
    for size, img in images.items():
        img.save(OUT / f"pyradplan-{size}.png")
    images[256].save(OUT / "pyradplan.ico", sizes=[(s, s) for s in SIZES if s <= 256])
    images[1024].save(OUT / "pyradplan.icns")
    print(f"wrote icons to {OUT}")
    del app
    return 0


if __name__ == "__main__":
    sys.exit(main())
