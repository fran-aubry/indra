from __future__ import annotations
from typing import NamedTuple
import numpy as np


class Rule(NamedTuple):

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0
    
    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def slope(self) -> float:
        return (self.y1 - self.y0) / max(self.width, 1)

    def x_at(self, y: float) -> float:
        return self.x0 + (self.x1 - self.x0) * (y - self.y0) / max(self.height, 1)

    def y_at(self, x: float) -> float:
        """Linearly interpolate y at horizontal position ``x``."""
        return self.y0 + self.slope * (x - self.x0)

    def cuts_box(
        self, 
        x0: float, 
        y0: float, 
        x1: float, 
        y1: float, 
        margin: float = 0.25
    ) -> bool:
        """True if this rule passes through the vertical interior of a box."""
        if x0 >= self.x1 or x1 <= self.x0:
            return False
        h = y1 - y0
        y = self.y_at((x0 + x1) / 2)
        return y0 + margin * h < y < y1 - margin * h


def _same_rule(
    a: Rule,
    b: Rule,
    max_gap_x: int = 15,
    max_dy: float = 6.0,
    max_dm: float = 0.04,
) -> bool:
    """True when two segments lie on the same rule (close in slope, x, and y)."""

    x_lo = max(a.x0, b.x0)
    x_hi = min(a.x1, b.x1)
    if x_lo - x_hi > max_gap_x:  # too far apart horizontally (e.g. separate \cmidrules)
        return False

    if x_lo <= x_hi:
        return (
            abs(a.y_at(x_lo) - b.y_at(x_lo)) <= max_dy
            and abs(a.y_at(x_hi) - b.y_at(x_hi)) <= max_dy
        )
    if abs(a.slope - b.slope) > max_dm:
        return False
    xm = (x_lo + x_hi) / 2
    return abs(a.y_at(xm) - b.y_at(xm)) <= max_dy


def _fit_cluster(cluster: list[Rule]) -> Rule:
    """Collapse a group of segments into one length-weighted best-fit segment."""
    xs, ys, ws = [], [], []
    for x0, y0, x1, y1 in cluster:
        w = float(max(x1 - x0, 1))
        xs.extend((x0, x1))
        ys.extend((y0, y1))
        ws.extend((w, w))
    m, b = np.polyfit(xs, ys, 1, w=ws)
    xmin = min(s[0] for s in cluster)
    xmax = max(s[2] for s in cluster)
    return Rule(int(xmin), int(round(m * xmin + b)), int(xmax), int(round(m * xmax + b)))


def _merge_rules(segs: list[Rule]) -> list[Rule]:
    """Merge duplicate top/bottom rule edges and segments split by vertical lines."""
    # Seed longest segments first so main rules anchor each cluster's fit.
    clusters: list[list[Rule]] = [[s] for s in sorted(segs, key=lambda s: -(s[2] - s[0]))]
    changed = True
    while changed:
        changed = False
        out: list[list[Rule]] = []
        reps: list[Rule] = []
        for cl in clusters:
            rep = _fit_cluster(cl)
            for i, r in enumerate(reps):
                if _same_rule(rep, r):
                    out[i].extend(cl)
                    reps[i] = _fit_cluster(out[i])
                    changed = True
                    break
            else:
                out.append(list(cl))
                reps.append(rep)
        clusters = out

    rules = [_fit_cluster(cl) for cl in clusters]
    rules.sort(key=lambda s: (s[1] + s[3]) / 2)
    return rules


def detect_hrules(
    gray: np.ndarray,
    line_h: float = 22.0,
    diff_thresh: int = 25,
    max_angle_deg: float = 5.0,
    max_gap: int = 1,
) -> list[Rule]:
    """Return nearly-horizontal rules as ``(x0, y0, x1, y1)`` in a grayscale crop."""
    import cv2

    H, W = gray.shape
    min_len = max(int(round(2.5 * line_h)), 55)
    if H < 5 or W < min_len:
        return []

    # Step 1: Compare each row against 2 px above and 2 px below.
    # Uniform white paper and uniform gray shading cancel to 0; dark rules and
    # shading boundaries (like a gray header -> white body) light up.
    above = gray[:-4, :]
    below = gray[4:, :]
    diff = cv2.absdiff(below, above)
    edges = np.zeros(gray.shape, dtype=np.uint8)
    edges[2:-2, :] = np.where(diff > diff_thresh, np.uint8(255), np.uint8(0))

    # Step 2: Find straight segments (x0, y0, x1, y1) at slight angles, without
    # bridging gaps between letters (maxLineGap=1).
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,  # 0.5 degree resolution
        threshold=int(0.8 * min_len),
        minLineLength=min_len,
        maxLineGap=max_gap,
    )
    if raw is None:
        return []
    solid: np.ndarray | None = None
    if max_gap > 1:
        bg = float(np.percentile(gray, 90))
        solid = (edges > 0) | (gray < bg - diff_thresh)

    segs: list[Rule] = []
    for x0, y0, x1, y1 in raw.reshape(-1, 4):
        if x1 < x0:
            x0, y0, x1, y1 = x1, y1, x0, y0
        dx, dy = x1 - x0, abs(y1 - y0)
        if dx > 0 and np.degrees(np.arctan2(dy, dx)) <= max_angle_deg:
            if solid is not None:
                xs = np.arange(x0, x1 + 1)
                ys = np.clip(
                    np.round(y0 + (y1 - y0) * (xs - x0) / dx).astype(int), 0, H - 1
                )
                sup = (
                    solid[ys, xs]
                    | solid[np.maximum(ys - 1, 0), xs]
                    | solid[np.minimum(ys + 1, H - 1), xs]
                )
                if float(sup.mean()) < 0.9:
                    continue
            segs.append(Rule(int(x0), int(y0), int(x1), int(y1)))

    # Step 3: Merge duplicate edges and segments split by vertical grid crossings.
    return _merge_rules(segs)


def detect_vrules( 
    gray: np.ndarray,
    line_h: float = 22.0,
    diff_thresh: int = 25,
    max_angle_deg: float = 5.0,
    max_gap: int = 6,
) -> list[Rule]:
    vrules_trans = detect_hrules(
        np.ascontiguousarray(gray.T),
        line_h=line_h,
        diff_thresh=diff_thresh,
        max_angle_deg=max_angle_deg,
        max_gap=max_gap
    )
    return [Rule(r.y0, r.x0, r.y1, r.x1) for r in vrules_trans]