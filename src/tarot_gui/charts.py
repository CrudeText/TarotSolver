"""
Chart widgets for the League Parameters tab: population pie chart, generation flow, GA visual.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

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

# Reference/evolving subdivision colors
REFERENCE_COLOR = 0x606060
EVOLVING_COLOR_GA = 0x4A90D9
EVOLVING_COLOR_PLAY = 0x50C878


def _rgb(c: int) -> QtGui.QColor:
    return QtGui.QColor((c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF)


def _gaussian(x: float, mu: float, sigma: float) -> float:
    """Gaussian PDF at x."""
    if sigma <= 0:
        return 0.0
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


@dataclass
class GroupSliceData:
    """Data for one group slice in the population pie."""

    name: str
    total: int
    ga_eligible: int
    play_in_league: int
    reference: int
    color: int


def _angle_from_point(cx: float, cy: float, px: float, py: float) -> float:
    """Angle in degrees (0=3 o'clock, counter-clockwise positive) from center to point."""
    dx = px - cx
    dy = py - cy
    if dx == 0 and dy == 0:
        return 0.0
    rad = math.atan2(-dy, dx)
    deg = math.degrees(rad)
    return deg if deg >= 0 else deg + 360


def _point_in_pie_circle(cx: float, cy: float, radius: float, px: float, py: float) -> bool:
    """True if point (px,py) is inside the pie circle."""
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy <= radius * radius


class PopulationPieWidget(QtWidgets.QWidget):
    """
    Pie chart of population by group with subdivisions (evolving vs reference).

    Shows GA-eligible count, play-in-league count, reference agents, and player-count fit
    in hover tooltips. Filter dropdowns (external) can change the view.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMinimumHeight(160)
        self.setMaximumWidth(320)
        self.setMaximumHeight(200)
        self.setMouseTracking(True)
        self._group_slices: List[GroupSliceData] = []
        self._total = 0
        self._player_count = 4
        self._group_by: str = "group"  # "group" | "ga_status" | "play_status"
        self._pie_rect: Optional[QtCore.QRectF] = None
        self._slice_angles: List[Tuple[float, float, int]] = []  # (start_deg, span_deg, slice_idx)
        self._hovered_slice: Optional[int] = None

    def set_data(
        self,
        group_slices: List[GroupSliceData],
        total: int,
        player_count: int,
        group_by: str = "group",
    ) -> None:
        self._group_slices = group_slices
        self._total = total
        self._player_count = player_count
        self._group_by = group_by
        self._hovered_slice = None
        self.update()

    def _build_slices_for_paint(self) -> List[Tuple[str, int, int, int]]:
        """Return (label, count, main_color, sub_count) for each drawn slice."""
        result: List[Tuple[str, int, int, int]] = []
        if self._group_by == "ga_status":
            ga_total = sum(g.ga_eligible for g in self._group_slices)
            ref_total = self._total - ga_total
            if ga_total > 0:
                result.append(("GA-eligible", ga_total, EVOLVING_COLOR_GA, 0))
            if ref_total > 0:
                result.append(("Reference", ref_total, REFERENCE_COLOR, 0))
        elif self._group_by == "play_status":
            play_total = sum(g.play_in_league for g in self._group_slices)
            not_play = self._total - play_total
            if play_total > 0:
                result.append(("Play in league", play_total, EVOLVING_COLOR_PLAY, 0))
            if not_play > 0:
                result.append(("Not in league", not_play, REFERENCE_COLOR, 0))
        else:
            for g in self._group_slices:
                if g.total > 0:
                    result.append((g.name, g.total, g.color, g.reference))
        return result

    def _slice_at_angle(self, angle_deg: float) -> Optional[int]:
        """Return slice index (0-based) for given angle, or None."""
        a = angle_deg % 360
        for i, (start, span, idx) in enumerate(self._slice_angles):
            if span >= 360:
                return idx
            start_norm = start % 360
            # Check if a falls within [start_norm, start_norm + span) with wraparound
            diff = (a - start_norm + 360) % 360
            if diff < span:
                return idx
        return None

    def _tooltip_for_slice(self, idx: int) -> str:
        slices = self._build_slices_for_paint()
        if idx < 0 or idx >= len(slices):
            return ""
        label, count, _, sub = slices[idx]
        lines = [f"{label}: {count}"]
        if self._group_by == "group" and idx < len(self._group_slices):
            g = self._group_slices[idx]
            lines.append(f"GA-eligible: {g.ga_eligible}")
            lines.append(f"Play-in-league: {g.play_in_league}")
            lines.append(f"Reference: {g.reference}")
            fit = "Yes" if self._total >= self._player_count else "No"
            lines.append(f"Player-count fit: {fit}")
        elif self._group_by == "ga_status":
            lines.append(f"Total agents: {self._total}")
            fit = "Yes" if self._total >= self._player_count else "No"
            lines.append(f"Player-count fit: {fit}")
        elif self._group_by == "play_status":
            lines.append(f"Total agents: {self._total}")
            fit = "Yes" if self._total >= self._player_count else "No"
            lines.append(f"Player-count fit: {fit}")
        return "\n".join(lines)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseMoveEvent(event)
        if not self._pie_rect or self._total == 0:
            QtWidgets.QToolTip.hideText()
            self._hovered_slice = None
            self.update()
            return
        cx = self._pie_rect.center().x()
        cy = self._pie_rect.center().y()
        radius = self._pie_rect.width() / 2
        px = event.position().x()
        py = event.position().y()
        if not _point_in_pie_circle(cx, cy, radius, px, py):
            QtWidgets.QToolTip.hideText()
            self._hovered_slice = None
            self.update()
            return
        angle = _angle_from_point(cx, cy, px, py)
        idx = self._slice_at_angle(angle)
        if idx is not None:
            tt = self._tooltip_for_slice(idx)
            if tt:
                QtWidgets.QToolTip.showText(event.globalPosition().toPoint(), tt, self)
            self._hovered_slice = idx
        else:
            QtWidgets.QToolTip.hideText()
            self._hovered_slice = None
        self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        super().leaveEvent(event)
        QtWidgets.QToolTip.hideText()
        self._hovered_slice = None
        self.update()

    def paintEvent(self, event: QtCore.QEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = self.rect()
            w, h = rect.width(), rect.height()
            pie_size = min(140, w - 80, h - 40)
            cx = pie_size // 2 + 10
            cy = h // 2
            self._pie_rect = QtCore.QRectF(cx - pie_size / 2, cy - pie_size / 2, pie_size, pie_size)
            total = self._total
            self._slice_angles = []

            if total == 0:
                painter.setPen(QtGui.QPen(_rgb(0x505050), 1))
                painter.setBrush(QtGui.QBrush(_rgb(0x404040)))
                painter.drawPie(self._pie_rect, 0, 360 * 16)
            else:
                slices = self._build_slices_for_paint()
                start = 0  # in 1/16ths for drawPie
                start_deg = 0.0  # in degrees for hover
                for i, (label, count, main_color, sub_ref) in enumerate(slices):
                    span = int(360 * 16 * count / total) if total else 0
                    if span <= 0:
                        continue
                    span_deg = 360 * count / total
                    self._slice_angles.append((start_deg, span_deg, i))

                    is_hovered = self._hovered_slice == i
                    pen_width = 2 if is_hovered else 1
                    painter.setPen(QtGui.QPen(_rgb(main_color).darker(130), pen_width))
                    if sub_ref > 0 and count > sub_ref:
                        # Subdivide: evolving first, then reference
                        evolving = count - sub_ref
                        span_ev = int(360 * 16 * evolving / total)
                        span_ref = int(360 * 16 * sub_ref / total)
                        painter.setBrush(QtGui.QBrush(_rgb(main_color)))
                        painter.drawPie(self._pie_rect, start, span_ev)
                        painter.setBrush(QtGui.QBrush(_rgb(REFERENCE_COLOR)))
                        painter.drawPie(self._pie_rect, start + span_ev, span_ref)
                    else:
                        painter.setBrush(QtGui.QBrush(_rgb(main_color)))
                        painter.drawPie(self._pie_rect, start, span)
                    start += span
                    start_deg += span_deg

            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            painter.setPen(self.palette().color(QtGui.QPalette.ColorRole.WindowText))
            painter.drawText(
                int(pie_size + 24), 0, int(w - pie_size - 28), h,
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
            boxes = ["Population", "Tournament", "Fitness", "Mutation\n/ Clone", "Next Gen"]
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


class ReproductionBarWidget(QtWidgets.QWidget):
    """Visual bar: red=mutant offspring, green=elite kept, purple=clone offspring."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(56)
        self._elite_pct = 90.0
        self._clone_pct = 10.0
        self._mut_pct = 80.0  # elite = mutated + cloned
        self._total_agents = 0

    def set_params(
        self,
        elite_pct: float,
        clone_pct: float,
        mut_pct: float,
        total_agents: int = 0,
    ) -> None:
        self._elite_pct = elite_pct
        self._clone_pct = max(0.0, min(100.0, clone_pct))
        self._mut_pct = max(0.0, min(100.0, mut_pct))
        self._total_agents = total_agents
        self.update()

    def paintEvent(self, event: QtCore.QEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = self.rect()
            w = rect.width()
            pen_color = self.palette().color(QtGui.QPalette.ColorRole.WindowText)
            painter.setPen(QtGui.QPen(pen_color, 1))
            bar_x, bar_w = 75, max(100, w - 110)
            bar_h = 14
            y = 4
            painter.drawText(4, y + 12, "Selection")
            # Left to right: red (eliminated) | green (mutated) | purple (cloned)
            elim = 1.0 - (self._elite_pct / 100.0)
            m = self._mut_pct / 100.0
            c = self._clone_pct / 100.0
            total = self._total_agents
            # Round counts so they sum to total (largest remainder)
            def _rounded_counts(fracs: list[float], n: int) -> list[int]:
                if n <= 0:
                    return [0] * len(fracs)
                floats = [f * n for f in fracs]
                ints = [int(f) for f in floats]
                remainder = n - sum(ints)
                if remainder > 0:
                    # Add 1 to cells with largest fractional part
                    fracs_sorted = sorted(
                        enumerate(floats),
                        key=lambda i_f: i_f[1] - int(i_f[1]),
                        reverse=True,
                    )
                    for i in range(remainder):
                        ints[fracs_sorted[i][0]] += 1
                elif remainder < 0:
                    # Subtract 1 from cells with smallest fractional part
                    fracs_sorted = sorted(
                        enumerate(floats),
                        key=lambda i_f: i_f[1] - int(i_f[1]),
                    )
                    for i in range(-remainder):
                        idx = fracs_sorted[i][0]
                        if ints[idx] > 0:
                            ints[idx] -= 1
                return ints

            counts = _rounded_counts([elim, m, c], total)
            elim_n, mut_n, clone_n = counts[0], counts[1], counts[2]
            x = bar_x
            font = painter.font()
            font.setPointSize(7)
            painter.setFont(font)
            if elim > 0.001:
                rw = max(2, int(bar_w * elim))
                painter.setBrush(QtGui.QBrush(_rgb(0xD9534F)))
                painter.drawRect(int(x), y, rw, bar_h)
                painter.setPen(QtGui.QPen(_rgb(0xffffff), 1))
                painter.drawText(
                    int(x), int(y), int(rw), int(bar_h),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    str(elim_n),
                )
                painter.setPen(QtGui.QPen(pen_color, 1))
                x += rw
            if m > 0.001:
                mw = max(2, int(bar_w * m))
                painter.setBrush(QtGui.QBrush(_rgb(0x50C878)))
                painter.drawRect(int(x), y, mw, bar_h)
                painter.setPen(QtGui.QPen(_rgb(0xffffff), 1))
                painter.drawText(
                    int(x), int(y), int(mw), int(bar_h),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    str(mut_n),
                )
                painter.setPen(QtGui.QPen(pen_color, 1))
                x += mw
            if c > 0.001:
                cw = max(4, int(bar_w * c))
                painter.setBrush(QtGui.QBrush(_rgb(0xBB66FF)))
                painter.drawRect(int(x), y, cw, bar_h)
                painter.setPen(QtGui.QPen(_rgb(0xffffff), 1))
                painter.drawText(
                    int(x), int(y), int(cw), int(bar_h),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    str(clone_n),
                )
                painter.setPen(QtGui.QPen(pen_color, 1))
                x += cw
            # Fill any tiny gap with grey
            if x < bar_x + bar_w - 1:
                painter.setBrush(QtGui.QBrush(_rgb(0x404040)))
                painter.drawRect(int(x), y, int(bar_x + bar_w - x), bar_h)
            painter.drawText(bar_x + bar_w + 6, y + 12, f"{self._elite_pct:.0f}%")
            # Color legend
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            leg_y = y + bar_h + 8
            for i, (color, label) in enumerate([
                (0xD9534F, "red: eliminated"),
                (0x50C878, "green: mutated"),
                (0xBB66FF, "purple: cloned"),
            ]):
                lx = bar_x + i * 90
                painter.setBrush(QtGui.QBrush(_rgb(color)))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawRect(int(lx), int(leg_y), 10, 10)
                painter.setPen(QtGui.QPen(pen_color, 1))
                painter.drawText(int(lx + 14), int(leg_y + 10), label)
        finally:
            painter.end()


class FitnessVisualWidget(QtWidgets.QWidget):
    """Line graph: fitness vs ELO for several avg_score levels.

    Shows multiple curves (avg_score = 0, 25, 50, 75, 100) so the effect of both
    ELO weight and avg_score weight is visible. When avg_score weight is 0, lines overlap.
    """

    # Colors for avg_score curves (0, 25, 50, 75, 100)
    _CURVE_COLORS = (0x4A90D9, 0x50C878, 0xE6B800, 0xD9534F, 0x9B59B6)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setMinimumWidth(260)
        self._weight_elo = 1.0
        self._weight_avg_score = 0.0

    def set_params(self, weight_elo: float, weight_avg_score: float) -> None:
        self._weight_elo = weight_elo
        self._weight_avg_score = weight_avg_score
        self.update()

    def paintEvent(self, event: QtCore.QEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = self.rect()
            w, h = rect.width(), rect.height()
            pen_color = self.palette().color(QtGui.QPalette.ColorRole.WindowText)
            margin_left, margin_right = 40, 80
            margin_top, margin_bot = 20, 32
            gx, gy = margin_left, margin_top
            gw = w - margin_left - margin_right
            gh = h - margin_top - margin_bot

            elo_min, elo_max = 1000.0, 2000.0
            fit_min, fit_max = 0.0, 2500.0
            elo_range = elo_max - elo_min
            fit_range = fit_max - fit_min
            avg_scores = (0.0, 25.0, 50.0, 75.0, 100.0)

            # Axes
            painter.setPen(QtGui.QPen(pen_color, 1))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(int(gx), int(gy), int(gw), int(gh))

            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)

            # Axis labels
            painter.drawText(int(gx + gw // 2 - 20), int(h - 4), "ELO")
            painter.save()
            painter.translate(8, gy + gh // 2 + 20)
            painter.rotate(-90)
            painter.drawText(-30, 0, "Fitness")
            painter.restore()

            # X-axis ticks (ELO)
            for i in range(5):
                t = i / 4.0
                elo = elo_min + t * elo_range
                fx = gx + t * (gw - 1)
                painter.drawLine(int(fx), int(gy + gh), int(fx), int(gy + gh + 4))
                painter.drawText(int(fx - 18), int(gy + gh + 14), f"{int(elo)}")
            # Y-axis ticks (Fitness)
            for i in range(5):
                t = i / 4.0
                fit = fit_min + t * fit_range
                fy = gy + gh - t * (gh - 1)
                painter.drawLine(int(gx - 4), int(fy), int(gx), int(fy))
                painter.drawText(int(gx - 32), int(fy + 4), f"{int(fit)}")

            # Multiple lines: fitness = w_elo*ELO + w_avg_score*avg_score for each avg_score
            n_pts = 80
            for curve_idx, avg_score in enumerate(avg_scores):
                path = QtGui.QPainterPath()
                for i in range(n_pts + 1):
                    t = i / n_pts
                    elo = elo_min + t * elo_range
                    fit = self._weight_elo * elo + self._weight_avg_score * avg_score
                    px = gx + t * (gw - 1)
                    norm = (fit - fit_min) / fit_range if fit_range > 0 else 0
                    norm = max(0.0, min(1.0, norm))
                    py = gy + gh - norm * (gh - 1)
                    if i == 0:
                        path.moveTo(px, py)
                    else:
                        path.lineTo(px, py)
                color = self._CURVE_COLORS[curve_idx % len(self._CURVE_COLORS)]
                painter.setPen(QtGui.QPen(_rgb(color), 1.5))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawPath(path)

            # Legend: small table "Average Score" | color | value (0, 25, 50, 75, 100)
            leg_x = gx + gw + 6
            leg_y = gy
            font.setPointSize(7)
            painter.setFont(font)
            cell_h = 14
            col_w = (18, 22)  # color swatch, value
            tab_w = col_w[0] + col_w[1]
            tab_h = cell_h * 6  # header + 5 rows
            painter.setPen(QtGui.QPen(pen_color, 1))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(int(leg_x), int(leg_y), int(tab_w + 4), int(tab_h + 2))
            painter.drawLine(int(leg_x), int(leg_y + cell_h), int(leg_x + tab_w + 4), int(leg_y + cell_h))
            painter.drawText(int(leg_x + 2), int(leg_y + cell_h - 2), "Average Score")
            for i, avg_score in enumerate(avg_scores):
                ly = leg_y + cell_h + 1 + i * cell_h
                color = self._CURVE_COLORS[i % len(self._CURVE_COLORS)]
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(QtGui.QBrush(_rgb(color)))
                painter.drawRect(int(leg_x + 2), int(ly + 2), 10, 10)
                painter.setPen(QtGui.QPen(pen_color, 1))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawText(int(leg_x + 2 + col_w[0]), int(ly + 10), f"{int(avg_score)}")
        finally:
            painter.end()


class MutationProbBarWidget(QtWidgets.QWidget):
    """Bar visual for Mutation %."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(18)
        self._mutation_pct = 50.0

    def set_mutation_pct(self, pct: float) -> None:
        self._mutation_pct = pct
        self.update()

    def paintEvent(self, event: QtCore.QEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = self.rect()
            w = rect.width()
            pen_color = self.palette().color(QtGui.QPalette.ColorRole.WindowText)
            painter.setPen(QtGui.QPen(pen_color, 1))
            bar_x, bar_w = 85, max(100, w - 120)
            bar_h = 12
            y = 4
            painter.drawText(4, y + 10, "Mut. %")
            painter.setBrush(QtGui.QBrush(_rgb(0xE6B800)))
            mut_w = max(2, int(bar_w * self._mutation_pct / 100))
            painter.drawRect(bar_x, y, mut_w, bar_h)
            painter.setBrush(QtGui.QBrush(_rgb(0x404040)))
            painter.drawRect(bar_x + mut_w, y, bar_w - mut_w, bar_h)
            painter.drawText(bar_x + bar_w + 6, y + 10, f"{self._mutation_pct:.0f}%")
        finally:
            painter.end()


class MutationDistWidget(QtWidgets.QWidget):
    """Gaussian distribution for mutation std (Δ trait vs density)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(60)
        self._mutation_std = 0.1

    def set_mutation_std(self, std: float) -> None:
        self._mutation_std = max(0.01, std)
        self.update()

    def paintEvent(self, event: QtCore.QEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = self.rect()
            w, h = rect.width(), rect.height()
            font = painter.font()
            font.setPointSize(9)
            painter.setFont(font)
            pen_color = self.palette().color(QtGui.QPalette.ColorRole.WindowText)
            painter.setPen(QtGui.QPen(pen_color, 1))
            gx, gy = 92, 18
            gw = max(100, w - 150)
            gh = max(60, h - 48)

            std = self._mutation_std
            painter.drawText(4, 16, "Mutation distribution")

            # Fixed x-axis range: -0.5 to 0.5 so curve narrows/widens visibly as σ changes
            x_min, x_max = -0.5, 0.5
            x_range = x_max - x_min
            # Draw axis box
            painter.setPen(QtGui.QPen(pen_color, 1))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(int(gx), int(gy), int(gw), int(gh))

            # Axis labels
            font.setPointSize(8)
            painter.setFont(font)
            painter.drawText(int(gx + gw // 2 - 35), int(gy + gh + 28), "Δ trait")
            painter.save()
            painter.translate(gx - 30, gy + gh // 2 + 6)
            painter.rotate(-90)
            painter.drawText(-30, 0, "Density")
            painter.restore()

            # Graduated x-axis (fixed range)
            font.setPointSize(7)
            painter.setFont(font)
            for i in range(5):
                t = i / 4.0
                x_val = x_min + t * x_range
                px = gx + 4 + (gw - 8) * t
                painter.drawLine(int(px), int(gy + gh), int(px), int(gy + gh + 4))
                painter.drawText(int(px - 14), int(gy + gh + 12), f"{x_val:.2f}")
            # Graduated y-axis (PDF 0 to 1)
            for i in range(5):
                t = i / 4.0
                py = gy + gh - 4 - t * (gh - 8)
                painter.drawLine(int(gx - 4), int(py), int(gx), int(py))
                painter.drawText(int(gx - 26), int(py + 3), f"{t:.1f}")
            font.setPointSize(9)
            painter.setFont(font)

            # Gaussian curve: map x in [x_min, x_max] to pixels; curve shape changes with σ
            n_pts = 100
            path = QtGui.QPainterPath()
            for i in range(n_pts + 1):
                t = i / n_pts
                x_val = x_min + t * x_range
                pdf = _gaussian(x_val, 0, std) if std > 0 else 0.0
                max_pdf = _gaussian(0, 0, std) if std > 0 else 1.0
                y_norm = pdf / max_pdf if max_pdf > 0 else 0.0
                px = gx + 4 + (gw - 8) * t
                py = gy + gh - 4 - y_norm * (gh - 8)
                if i == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)
            fill_path = QtGui.QPainterPath(path)
            fill_path.lineTo(gx + gw - 4, gy + gh - 4)
            fill_path.lineTo(gx + 4, gy + gh - 4)
            fill_path.closeSubpath()
            fill = QtGui.QColor(0x9B59B6)
            fill.setAlpha(60)
            painter.setBrush(QtGui.QBrush(fill))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawPath(fill_path)
            painter.setPen(QtGui.QPen(_rgb(0x9B59B6), 2))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            painter.setPen(QtGui.QPen(pen_color, 1))
            painter.drawText(int(gx + gw + 4), int(gy + gh // 2 + 4), f"σ={std:.2f}")
        finally:
            painter.end()
