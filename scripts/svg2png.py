"""Dev helper: rasterize an SVG with QtSvg (offscreen)."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtGui import QImage, QPainter          # noqa: E402
from PySide6.QtSvg import QSvgRenderer              # noqa: E402
from PySide6.QtWidgets import QApplication          # noqa: E402

app = QApplication([])
r = QSvgRenderer(sys.argv[1])
if not r.isValid():
    raise SystemExit("invalid svg")
size = r.defaultSize()
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
img = QImage(int(size.width() * scale), int(size.height() * scale),
             QImage.Format_ARGB32)
img.fill(0xFFFFFFFF)
p = QPainter(img)
r.render(p)
p.end()
img.save(sys.argv[2])
print("wrote", sys.argv[2], img.width(), "x", img.height())
