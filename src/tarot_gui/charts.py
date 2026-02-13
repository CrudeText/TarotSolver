"""
Chart widgets for the League Parameters tab: population pie chart, generation flow, GA visual.
"""
from __future__ import annotations

import math
from typing import List, Tuple

from PySide6 import QtCore, QtGui, QtWidgets


# Pie slice colors (distinct, theme-neutral)
PIE_COLORS = [
    (0x4A90D9,),  # blue
    (0x50C878,),  # emerald
    (0xE6B800,),  # gold
    (0xD9534F,),  # red
    (0x9B59B6,),  # purple
    (0x1ABC9C,),  # teal
    (0xE67E22,),  # orange
    (0x95A5A6,),  # gray
]


def _rgb(c: int) -> QtGui.QColor:
    return QtGui.QColor((c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF)


def _gaussian(x: float, mu: float, sigma: float) -> float:
    """Gaussian PDF at x."""
    if sigma <= 0:
        return 0.0
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


class PopulationPieWidget(QtWidgets.QWidget):
    """Pie chart of population by group, with Total Agents tally."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(180)
        self.setMinimumHeight(130)
        self.setMaximumWidth(220)
        self.setMaximumHeight(150)
        self._slices: List[Tuple[str, int]] = []
        self._total = 0
        self._colors: List[int] | None = None

    def set_data(
        self,
        slices: List[Tuple[str, int]],
        total: int,
        colors: List[int] | None = None,
    ) -> None:
        self._slices = slices
        self._total = total
        self._colors = colors
        self.update()

    def paintEvent(self, event: QtCore.QEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = self.rect()
            w, h = rect.width(), rect.height()
            # Bigger pie: use most of the widget
            pie_size = min(110, w - 50, h - 30)
            cx = pie_size // 2 + 8
            cy = h // 2
            pie_rect = QtCore.QRectF(cx - pie_size / 2, cy - pie_size / 2, pie_size, pie_size)
            total = self._total
            if total == 0:
                painter.setPen(QtGui.QPen(_rgb(0x505050), 1))
                painter.setBrush(QtGui.QBrush(_rgb(0x404040)))
                painter.drawPie(pie_rect, 0, 360 * 16)
            else:
                start = 0
                for i, (_, count) in enumerate(self._slices):
                    span = int(360 * 16 * count / total) if total else 0
                    if span > 0:
                        hex_val = (
                            self._colors[i]
                            if self._colors and i < len(self._colors)
                            else PIE_COLORS[i % len(PIE_COLORS)][0]
                        )
                        color = _rgb(hex_val)
                        painter.setPen(QtGui.QPen(color.darker(120), 1))
                        painter.setBrush(QtGui.QBrush(color))
                        painter.drawPie(pie_rect, start, span)
                        start += span
            # Total label: right of pie
            font = painter.font()
            font.setPointSize(11)
            painter.setFont(font)
            painter.setPen(self.palette().color(QtGui.QPalette.ColorRole.WindowText))
            painter.drawText(
                int(pie_size + 20), 0, int(w - pie_size - 24), h,
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                f"Total Agents\n{self._total}",
            )
        finally:
            painter.end()


class GenerationFlowWidget(QtWidgets.QWidget):
    """Diagram: Population → Tournament → Fitness → Elites+Parents → Mutation/Clone → Next gen."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMinimumWidth(520)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )

    def paintEvent(self, event: QtCore.QEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = self.rect()
            w, h = rect.width(), rect.height()
            boxes = ["Population", "Tournament", "Fitness", "Elites +\nParents", "Mutation\n/ Clone", "Next Gen"]
            n = len(boxes)
            gap = 16
            box_w = max(70, (w - 40 - (n - 1) * gap) // n)
            total_w = n * box_w + (n - 1) * gap
            x0 = max(20, (w - total_w) // 2)
            y = max(10, (h - 50) // 2)
            pen = QtGui.QPen(self.palette().color(QtGui.QPalette.ColorRole.WindowText), 1)
            painter.setPen(pen)
            painter.setBrush(QtGui.QBrush(self.palette().color(QtGui.QPalette.ColorRole.Base)))
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            for i, label in enumerate(boxes):
                rx = x0 + i * (box_w + gap)
                box_rect = QtCore.QRectF(rx, y, box_w, 50)
                painter.drawRoundedRect(box_rect, 4, 4)
                painter.drawText(
                    box_rect,
                    QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap,
                    label,
                )
                if i < n - 1:
                    ax1 = rx + box_w + 2
                    ax2 = rx + box_w + gap - 2
                    ay = y + 25
                    painter.drawLine(int(ax1), int(ay), int(ax2), int(ay))
                    # Arrowhead
                    arrow = QtGui.QPolygonF([
                        QtCore.QPointF(ax2 - 5, ay - 4),
                        QtCore.QPointF(ax2 - 5, ay + 4),
                        QtCore.QPointF(ax2 + 1, ay),
                    ])
                    painter.drawPolygon(arrow)
        finally:
            painter.end()


class GAVisualWidget(QtWidgets.QWidget):
    """Visual: elite fraction, mutation prob, and Gaussian distribution for mutation std."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(180)
        self._elite_pct = 10.0
        self._mutation_pct = 50.0
        self._mutation_std = 0.1

    def set_params(self, elite_pct: float, mutation_pct: float, mutation_std: float) -> None:
        self._elite_pct = elite_pct
        self._mutation_pct = mutation_pct
        self._mutation_std = max(0.01, mutation_std)
        self.update()

    def paintEvent(self, event: QtCore.QEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
            rect = self.rect()
            w, h = rect.width(), rect.height()
            font = painter.font()
            font.setPointSize(9)
            painter.setFont(font)
            pen_color = self.palette().color(QtGui.QPalette.ColorRole.WindowText)
            painter.setPen(QtGui.QPen(pen_color, 1))

            bar_x, bar_w = 85, max(100, w - 120)
            bar_h = 14

            # Bar 1: Elite %
            y = 8
            painter.drawText(4, y + 12, "Elite %")
            painter.setBrush(QtGui.QBrush(_rgb(0x4A90D9)))
            elite_w = max(2, int(bar_w * self._elite_pct / 100))
            painter.drawRect(bar_x, y, elite_w, bar_h)
            painter.setBrush(QtGui.QBrush(_rgb(0x404040)))
            painter.drawRect(bar_x + elite_w, y, bar_w - elite_w, bar_h)
            painter.drawText(bar_x + bar_w + 6, y + 12, f"{self._elite_pct:.0f}%")

            # Bar 2: Mutation prob
            y += 28
            painter.drawText(4, y + 12, "Mut. %")
            painter.setBrush(QtGui.QBrush(_rgb(0xE6B800)))
            mut_w = max(2, int(bar_w * self._mutation_pct / 100))
            painter.drawRect(bar_x, y, mut_w, bar_h)
            painter.setBrush(QtGui.QBrush(_rgb(0x404040)))
            painter.drawRect(bar_x + mut_w, y, bar_w - mut_w, bar_h)
            painter.drawText(bar_x + bar_w + 6, y + 12, f"{self._mutation_pct:.0f}%")

            # Gaussian distribution for mutation std
            y += 36
            painter.drawText(4, y + 10, "Mutation\ndistribution")
            std = self._mutation_std
            gx = bar_x
            gy = y + 24
            gw = bar_w
            gh = 70
            # Draw axis box
            painter.setPen(QtGui.QPen(pen_color, 1))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(int(gx), int(gy), int(gw), int(gh))
            # Gaussian curve: N(0, std), x from -3*std to +3*std
            n_pts = 80
            xs = [-3 * std + 6 * std * i / (n_pts - 1) for i in range(n_pts)]
            max_pdf = _gaussian(0, 0, std) if std > 0 else 1.0
            ys = [_gaussian(x, 0, std) / max_pdf for x in xs]
            # Scale to graph area (leave 4px margin)
            path = QtGui.QPainterPath()
            for i in range(n_pts):
                px = gx + 4 + (gw - 8) * (i / (n_pts - 1))
                py = gy + gh - 4 - (gh - 8) * ys[i]
                if i == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)
            # Fill under curve first
            fill_path = QtGui.QPainterPath(path)
            fill_path.lineTo(gx + gw - 4, gy + gh - 4)
            fill_path.lineTo(gx + 4, gy + gh - 4)
            fill_path.closeSubpath()
            fill = QtGui.QColor(0x9B59B6)
            fill.setAlpha(60)
            painter.setBrush(QtGui.QBrush(fill))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawPath(fill_path)
            # Draw curve on top
            painter.setPen(QtGui.QPen(_rgb(0x9B59B6), 2))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            # Labels
            painter.setPen(QtGui.QPen(pen_color, 1))
            painter.drawText(int(gx + gw + 6), int(gy + gh // 2 + 4), f"σ={std:.2f}")
        finally:
            painter.end()
