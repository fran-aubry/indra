from __future__ import annotations

import numpy as np


Rule = tuple[int, int, int, int]


def _y_at(seg: Rule, x: float) -> float:
    """Linearly interpolate a segment's y-coordinate at horizontal position ``x``."""
    x0, y0, x1, y1 = seg
    return y0 + (y1 - y0) * (x - x0) / max(x1 - x0, 1)


def _same_rule(
    a: Rule,
    b: Rule,
    max_gap_x: int = 15,
    max_dy: float = 6.0,
    max_dm: float = 0.04,
) -> bool:
    """True when two segments lie on the same rule (close in slope, x, and y)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ma = (ay1 - ay0) / max(ax1 - ax0, 1)
    mb = (by1 - by0) / max(bx1 - bx0, 1)
    if abs(ma - mb) > max_dm:
        return False

    x_lo = max(ax0, bx0)
    x_hi = min(ax1, bx1)
    if x_lo - x_hi > max_gap_x:  # too far apart horizontally (e.g. separate \cmidrules)
        return False

    if x_lo <= x_hi:
        return (
            abs(_y_at(a, x_lo) - _y_at(b, x_lo)) <= max_dy
            and abs(_y_at(a, x_hi) - _y_at(b, x_hi)) <= max_dy
        )
    xm = (x_lo + x_hi) / 2
    return abs(_y_at(a, xm) - _y_at(b, xm)) <= max_dy


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
    return (int(xmin), int(round(m * xmin + b)), int(xmax), int(round(m * xmax + b)))


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
        maxLineGap=1,
    )
    if raw is None:
        return []

    segs: list[Rule] = []
    for x0, y0, x1, y1 in raw.reshape(-1, 4):
        if x1 < x0:
            x0, y0, x1, y1 = x1, y1, x0, y0
        dx, dy = x1 - x0, abs(y1 - y0)
        if dx > 0 and np.degrees(np.arctan2(dy, dx)) <= max_angle_deg:
            segs.append((int(x0), int(y0), int(x1), int(y1)))

    # Step 3: Merge duplicate edges and segments split by vertical grid crossings.
    return _merge_rules(segs)