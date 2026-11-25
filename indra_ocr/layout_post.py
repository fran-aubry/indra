"""Page text from the ordered lines and the OrderNet layout heads (torch-free).

OrderNet with ``layout`` gives every line a block class and every pair of lines
a "same block" logit.  Walking the reading order, a new block starts wherever
the same-block logit between consecutive lines is negative.  Each block is then
rendered by its class:

  Section-Header        "## " + joined text
  Text, Caption, ...    one paragraph; a word split by a hyphen at a line end is joined
  List-Group            one line per item (a new item starts at a bullet or number)
  Table                 an HTML table assembled from the cell lines (rows by vertical
                        centre, columns anchored on the fullest rows)
  Page-Header / Footer  left out unless ``keep_furniture``
"""

from __future__ import annotations

import html
import re
from .table_rules import Rule, detect_hrules
import numpy as np

from .order_post import BLOCK_CLASSES, _cluster

HEADER, FOOTER = BLOCK_CLASSES.index("Page-Header"), BLOCK_CLASSES.index("Page-Footer")
SECTION = BLOCK_CLASSES.index("Section-Header")
LIST = BLOCK_CLASSES.index("List-Group")
TABLE = BLOCK_CLASSES.index("Table")

_WHOLE_MATH = re.compile(r"^\$[^$]+\$$")  # a line that is one formula
# A list marker, followed by text or alone on its line (the detector often cuts
# a bullet off as a line of its own; the next line then joins its item).
_ITEM = re.compile(r"^\s*([•\-–▪◦*·→]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])(\s|$)")


def _cjk(c: str) -> bool:
    """Han, kana, CJK punctuation and fullwidth forms: written without spaces
    (Korean Hangul is not: Korean separates words with spaces)."""
    o = ord(c)
    return (
        0x3000 <= o <= 0x30FF
        or 0x3400 <= o <= 0x9FFF
        or 0xF900 <= o <= 0xFAFF
        or 0xFF00 <= o <= 0xFFEF
        or 0x20000 <= o <= 0x2FA1F
    )


def join_lines(texts: list[str]) -> str:
    """Join the lines of one paragraph.  'tumori-' + 'genesis' -> 'tumorigenesis';
    a line break between two CJK characters takes no space."""
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if (
            len(out) > 1
            and out.endswith("-")
            and out[-2].isalpha()
            and t[0].isalpha()
            and t[0].islower()
        ):
            out = out[:-1] + t
        elif out and _cjk(out[-1]) and _cjk(t[0]):
            out += t
        else:
            out = f"{out} {t}" if out else t
    return out


def split_blocks(order: list[int], same: np.ndarray) -> list[list[int]]:
    """Runs of the reading order with a positive same-block logit between neighbours."""
    blocks: list[list[int]] = []
    for i in order:
        if blocks and same[blocks[-1][-1], i] > 0:
            blocks[-1].append(i)
        else:
            blocks.append([i])
    return blocks

def _split_sections(boxes: np.ndarray, hrules: list[Rule] | None) -> list[list[int]]:
    """Partition box indexes top-to-bottom using full horizontal dividing rules."""
    n = len(boxes)
    if n == 0:
        return []
    if not hrules:
        return [list(range(n))]
    
    span_x = float(boxes[:, 2].max() - boxes[:, 0].min())
    box_h = boxes[:, 3] - boxes[:, 1]
    med_h = float(np.median(box_h))

    # Dividers are rules that are wide enough, don't cut any horizontal text box    
    dividers = [
        r for r in hrules if r.width >= 0.5 * span_x and not any(
            box_h[k] <= 2.5 * med_h and r.cuts_box(*boxes[k, :4])
            for k in range(n)
        )
    ]
    if not dividers:
        return [list(range(n))]

    buckets: list[list[int]] = [
        [] for _ in range(len(dividers) + 1)
    ]
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2
    for k in range(n):
        sec_idx = sum(1 for r in dividers if r.y_at(float(center_x[k])) < float(center_y[k]))
        buckets[sec_idx].append(k)

    return [b for b in buckets if b]


def _rows(boxes: np.ndarray) -> list[list[int]]:
    """Group cell lines into rows by vertical centre, top to bottom, left to right."""
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    rows: list[list[int]] = []
    for i in np.argsort(cy):
        if rows and abs(cy[i] - np.mean(cy[rows[-1]])) < 0.5 * h:
            rows[-1].append(int(i))
        else:
            rows.append([int(i)])
    return [sorted(r, key=lambda k: boxes[k, 0]) for r in rows]


def _merge_wrapped(
    rows: list[list[int]],
    boxes: np.ndarray,
    min_rows: int = 3,
    med_gap: float | None = None,
) -> list[list[int]]:
    """Join the line rows of wrapped cells into one table row.

    Inside a wrapped cell the gap to the next line is the font's line gap; a new
    table row adds cell padding.  A row joins the row above when its gap is under
    half the table's median row gap and under 0.6 line heights, and it has fewer
    lines than the row above (a continuation fills only the wrapped columns)."""
    if len(rows) < max(min_rows, 2):
        return rows
    h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    gaps = [
        float(boxes[nxt, 1].min() - boxes[cur, 3].max())
        for cur, nxt in zip(rows[:-1], rows[1:])
    ]
    row_gaps = [
        g
        for g, cur, nxt in zip(gaps, rows[:-1], rows[1:])
        if len(nxt) >= len(cur) and len(nxt) > 1
    ]
    if len(row_gaps) >= 2:
        med: float | None = float(np.median(row_gaps))
    elif med_gap is not None:
        med = med_gap
    elif len(rows) >= 3:
        med = float(np.median(row_gaps or gaps))
    else:
        med = None
    out = [list(rows[0])]
    head = last = rows[0]
    for g, r in zip(gaps, rows[1:]):
        # A continuation fills a non-increasing subset of the row's wrapped columns.
        if (med is None or g < 0.5 * med) and g < 0.6 * h and len(r) < len(head) and len(r) <= len(last):
            out[-1] += r
        else:
            out.append(list(r))
            head = r
        last = r
    return out


def _overlaps_x(left_a: float, right_a: float, left_b: float, right_b: float) -> bool:
    """Whether horizontal intervals [left_a, right_a] and [left_b, right_b] overlap."""
    overlap_left = max(left_a, left_b)
    overlap_right = min(right_a, right_b)
    return overlap_right > overlap_left


def _header_cmidrules(
    header_indices: list[int],
    boxes: np.ndarray,
    hrules: list[Rule] | None,
) -> list[list[Rule]]:
    """Find internal cmid rules in a header section, grouped top-to-bottom by y."""
    # Need at least 2 boxes (one above, one below) for an internal rule to exist.
    if len(header_indices) <= 1 or not hrules:
        return []
    sec_boxes = boxes[header_indices, :4]
    box_h = sec_boxes[:, 3] - sec_boxes[:, 1]
    med_h = float(np.median(box_h))
    center_x = (sec_boxes[:, 0] + sec_boxes[:, 2]) / 2
    center_y = (sec_boxes[:, 1] + sec_boxes[:, 3]) / 2
    min_y = float(sec_boxes[:, 1].min())
    max_y = float(sec_boxes[:, 3].max())

    cmid_rules: list[Rule] = []
    for rule in hrules:
        rule_mid_y = (rule.y0 + rule.y1) / 2
        # Rules that don't lay between the span of y coordinates defined by the
        # boxes in this section arent cmid rules.
        if not (min_y < rule_mid_y < max_y):
            continue
        # Skip false-positive Hough segments caused by glyph strokes slicing through a
        # normal-height text box (tall boxes > 2.5 * med_h are rotated vertical text).
        if any(
            box_h[k] <= 2.5 * med_h and rule.cuts_box(*sec_boxes[k])
            for k in range(len(header_indices))
        ):
            continue
        # A true cmidrule must separate at least one box above it from one below it.
        overlapping_boxes = [
            k for k in range(len(header_indices))
            if _overlaps_x(sec_boxes[k, 0], sec_boxes[k, 2], rule.x0, rule.x1)
        ]
        has_box_above = any(
            center_y[k] < rule.y_at(float(center_x[k]))
            for k in overlapping_boxes
        )
        has_box_below = any(
            center_y[k] > rule.y_at(float(center_x[k]))
            for k in overlapping_boxes
        )
        if has_box_above and has_box_below:
            cmid_rules.append(rule)

    # Cluster cmidrules at the same y-height (within half a line height).
    cmid_rules.sort(key=lambda r: (r.y0 + r.y1) / 2)
    levels: list[list[Rule]] = []
    for rule in cmid_rules:
        rule_mid_y = (rule.y0 + rule.y1) / 2
        prev_mid_y = (levels[-1][-1].y0 + levels[-1][-1].y1) / 2 if levels else 0.0
        if levels and rule_mid_y - prev_mid_y <= 0.5 * med_h:
            levels[-1].append(rule)
        else:
            levels.append([rule])
    return levels


def _header_rows(
    header_indices: list[int],
    boxes: np.ndarray,
    cmid_levels: list[list[Rule]],
) -> list[list[int]]:
    """Group header-section boxes into <th> tier rows using internal cmid rules."""
    sec_boxes = boxes[header_indices, :4]
    raw_rows = [[header_indices[i] for i in row] for row in _rows(sec_boxes)]
    if len(raw_rows) <= 1:
        return raw_rows

    # With cmidrules: each cmidrule level defines a horizontal y-cut across the header,
    # just like _split_sections does for full-width table dividers.
    if cmid_levels:
        level_ys = [
            float(np.median([(rule.y0 + rule.y1) / 2 for rule in level]))
            for level in cmid_levels
        ]
        tiers: list[list[int]] = [[] for _ in range(len(level_ys) + 1)]
        center_y = (sec_boxes[:, 1] + sec_boxes[:, 3]) / 2
        for k, cy in enumerate(center_y):
            tier_idx = sum(1 for ly in level_ys if ly < float(cy))
            tiers[tier_idx].append(header_indices[k])
        return [tier for tier in tiers if tier]

    # No cmidrules: merge consecutive visual lines only when their vertical gap is
    # tight line-wrapping (< 0.6 * med_h) and not an unlined super-header row.
    med_h = float(np.median(sec_boxes[:, 3] - sec_boxes[:, 1]))
    merged_rows = [list(raw_rows[0])]
    for cur_row, next_row in zip(raw_rows[:-1], raw_rows[1:]):
        row_gap = float(boxes[next_row, 1].min() - boxes[cur_row, 3].max())
        is_super_header_row = False
        if len(cur_row) < len(next_row):
            for parent in cur_row:
                overlaps = sum(
                    _overlaps_x(
                        boxes[parent, 0], boxes[parent, 2],
                        boxes[child, 0], boxes[child, 2],
                    )
                    for child in next_row
                )
                if overlaps >= 2:
                    is_super_header_row = True
                    break
        if row_gap < 0.6 * med_h and not is_super_header_row:
            merged_rows[-1].extend(next_row)
        else:
            merged_rows.append(list(next_row))
    return merged_rows
    

def _header_col_spans(
    row: list[int],
    boxes: np.ndarray,
    bands: np.ndarray,
    cmid_levels: list[list[Rule]],
) -> dict[int, list[int]]:
    """Expand upper-tier header boxes across the columns covered by their cmid rules."""
    if not cmid_levels or not row:
        return {}

    # 1. Group horizontally overlapping boxes (wrapped lines of one super-header).
    groups: list[tuple[float, float, float, list[int]]] = []
    for k in sorted(row, key=lambda idx: boxes[idx, 0]):
        left_x, right_x = float(boxes[k, 0]), float(boxes[k, 2])
        mid_y = float((boxes[k, 1] + boxes[k, 3]) / 2)
        if groups and _overlaps_x(left_x, right_x, groups[-1][0], groups[-1][1]):
            prev_x0, prev_x1, prev_y, box_ids = groups[-1]
            groups[-1] = (
                min(prev_x0, left_x),
                max(prev_x1, right_x),
                max(prev_y, mid_y),
                box_ids + [k],
            )
        else:
            groups.append((left_x, right_x, mid_y, [k]))

    # 2. Widen each group's x-range to include the cmidrule(s) directly below it.
    expanded_spans: dict[int, tuple[float, float]] = {}
    for group_idx, (left_x, right_x, mid_y, _) in enumerate(groups):
        mid_x = (left_x + right_x) / 2
        for level in cmid_levels:
            matching_rules = [
                rule
                for rule in level
                if _overlaps_x(left_x, right_x, rule.x0, rule.x1)
                and mid_y < rule.y_at(mid_x)
            ]
            if matching_rules:
                span_x0 = min(left_x, *(float(rule.x0) for rule in matching_rules))
                span_x1 = max(right_x, *(float(rule.x1) for rule in matching_rules))
                expanded_spans[group_idx] = (span_x0, span_x1)
                break

    # 3. Assign each covered column to the closest claiming super-header group
    #    (splits shared cmidrules like "Oxygenase | Ferredoxin | Reductase").
    assigned_cols: dict[int, list[int]] = {g: [] for g in expanded_spans}
    for col in range(len(bands)):
        claimants = [
            group_idx
            for group_idx, (span_x0, span_x1) in expanded_spans.items()
            if _overlaps_x(float(bands[col, 0]), float(bands[col, 1]), span_x0, span_x1)
        ]
        if claimants:
            col_center = float((bands[col, 0] + bands[col, 1]) / 2)
            winner = min(
                claimants,
                key=lambda g: abs(col_center - (groups[g][0] + groups[g][1]) / 2),
            )
            assigned_cols[winner].append(col)

    col_spans: dict[int, list[int]] = {}
    for group_idx, cols in assigned_cols.items():
        if cols:
            contiguous_cols = list(range(min(cols), max(cols) + 1))
            for box_idx in groups[group_idx][3]:
                col_spans[box_idx] = contiguous_cols
    return col_spans


def table_html(boxes: np.ndarray, texts: list[str], hrules: list[Rule] | None = None) -> str:
    """An HTML table from cell lines (one text line per cell, as born-digital PDFs give).

    A first row that is one line across most of the table is a title: it goes
    before the table as text.  The first remaining row is the header (<th>)."""
    sections = _split_sections(boxes, hrules)
    sec_rows = [
        [[sec[i] for i in r] for r in _rows(boxes[sec])]
        for sec in sections
    ]
    rows = [r for sr in sec_rows for r in sr]

    span_x = boxes[:, 2].max() - boxes[:, 0].min()
    if (
        len(rows) > 2
        and len(rows[0]) == 1
        and max(len(r) for r in rows) > 1
        and boxes[rows[0][0], 2] - boxes[rows[0][0], 0] > 0.6 * span_x
    ):
        keep = [k for k in range(len(texts)) if k != rows[0][0]]
        return (
            texts[rows[0][0]]
            + "\n\n"
            + table_html(boxes[keep], [texts[k] for k in keep], hrules=hrules)
        )
    width = max(len(r) for r in rows)
    full = [r for r in rows if len(r) == width]
    
    min_rows = 2 if len(sec_rows) > 1 else 3
    all_row_gaps = [
        float(boxes[nxt, 1].min() - boxes[cur, 3].max())
        for cur, nxt in zip(rows[:-1], rows[1:])
        if len(nxt) >= len(cur) and len(nxt) > 1
    ]
    med_gap = float(np.median(all_row_gaps)) if len(all_row_gaps) >= 2 else None
    # When a full horizontal rule separates the header section (sections[0]) from
    # the body, group its lines into semantic <th> tiers via cmidrules instead of
    # body-oriented _merge_wrapped (which cannot merge full-width wrapped headers).
    cmid_levels: list[list[Rule]] = []
    if len(sec_rows) >= 2 and 1 <= len(sec_rows[0]) <= 6 and len(sec_rows[0]) < len(rows):
        cmid_levels = _header_cmidrules(sections[0], boxes, hrules)
        header_rows = _header_rows(sections[0], boxes, cmid_levels)
        body_rows = [
            _merge_wrapped(sr, boxes, min_rows=min_rows, med_gap=med_gap)
            for sr in sec_rows[1:]
        ]
        merged = [header_rows, *body_rows]
        rows = [r for sr in merged for r in sr]
        n_head = len(header_rows) if len(header_rows) <= 4 else 1
    else:
        merged = [
            _merge_wrapped(sr, boxes, min_rows=min_rows, med_gap=med_gap)
            for sr in sec_rows
        ]
        rows = [r for sr in merged for r in sr]
        n_head = 1 if len(rows) > 1 else 0

    # Column bands from the rows with the most cells: those rows rarely span.
    bands = np.array(
        [
            [
                np.median([boxes[r[c], 0] for r in full]),
                np.median([boxes[r[c], 2] for r in full]),
            ]
            for c in range(width)
        ]
    )
    # Widen each band to the midpoints between neighbours, so every x falls in one band.
    edges = (
        [-np.inf]
        + [(bands[c, 1] + bands[c + 1, 0]) / 2 for c in range(width - 1)]
        + [np.inf]
    )
    out = ["<table>"]
    for ri, r in enumerate(rows):
        tag = "th" if ri < n_head else "td"
        cmid_spans = (
            _header_col_spans(r, boxes, bands, cmid_levels)
            if ri < n_head - 1
            else {}
        )
        cells: dict[int, tuple[int, str]] = {}
        for k in sorted(r, key=lambda k: (boxes[k, 1], boxes[k, 0])):
            if k in cmid_spans:
                span = cmid_spans[k]
            else:
                x0, x1 = boxes[k, 0], boxes[k, 2]
                first = int(np.searchsorted(edges, x0, side="right") - 1)
                last = int(np.searchsorted(edges, x1, side="right") - 1)
                # A cell spans every band it covers by more than half of the band.
                span = [
                    c
                    for c in range(max(first, 0), min(last, width - 1) + 1)
                    if min(x1, bands[c, 1]) - max(x0, bands[c, 0])
                    > 0.5 * (bands[c, 1] - bands[c, 0])
                ] or [min(max(first, 0), width - 1)]
            c0 = span[0]
            prev = cells.get(c0)
            text = texts[k] if prev is None else prev[1] + " " + texts[k]
            cells[c0] = (len(span) if prev is None else max(prev[0], len(span)), text)
        tds, c = [], 0
        while c < width:
            if c in cells:
                n, text = cells[c]
                attr = f' colspan="{n}"' if n > 1 else ""
                tds.append(f"<{tag}{attr}>{html.escape(text, quote=False)}</{tag}>")
                c += n
            else:
                tds.append(f"<{tag}></{tag}>")
                c += 1
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def heads_plausible(boxes: np.ndarray, tab: np.ndarray) -> bool:
    """The lines of one cell are stacked (they overlap horizontally).  When the
    cell head joins side-by-side lines in more than 10% of its cells, it has
    failed on this table and the geometric assembly is used instead."""
    cells = _cluster(tab[2])
    bad = 0
    for c in cells:
        if len(c) < 2:
            continue
        x0, x1 = boxes[c, 0], boxes[c, 2]
        if (np.minimum.outer(x1, x1) - np.maximum.outer(x0, x0)).min() < 0:
            bad += 1
    return bad <= 0.1 * len(cells)


def table_html_heads(boxes: np.ndarray, texts: list[str], tab: np.ndarray) -> str:
    """An HTML table from the OrderNet table heads.  tab [3, N, N] = logits
    same row, same column, same cell, for the N lines of one table block."""
    cells = _cluster(tab[2])
    cb = np.array(
        [
            [boxes[c, 0].min(), boxes[c, 1].min(), boxes[c, 2].max(), boxes[c, 3].max()]
            for c in cells
        ]
    )
    ctext = [
        " ".join(texts[k] for k in sorted(c, key=lambda k: (boxes[k, 1], boxes[k, 0])))
        for c in cells
    ]
    m = len(cells)

    def cell_logits(k: int) -> np.ndarray:
        out = np.zeros((m, m), np.float32)
        for a in range(m):
            for b in range(a + 1, m):
                out[a, b] = out[b, a] = tab[k][np.ix_(cells[a], cells[b])].mean()
        return out

    rows = sorted(_cluster(cell_logits(0)), key=lambda g: cb[g, 1].min())
    cols = sorted(_cluster(cell_logits(1)), key=lambda g: cb[g, 0].min())
    row_of = {c: r for r, g in enumerate(rows) for c in g}
    col_of = {c: k for k, g in enumerate(cols) for c in g}
    centers = [float(np.median((cb[g, 0] + cb[g, 2]) / 2)) for g in cols]
    h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    grid: dict[tuple[int, int], tuple[int, str]] = {}
    for c in range(m):
        c0 = col_of[c]
        span = 1
        while (
            c0 + span < len(cols) and cb[c, 0] - h <= centers[c0 + span] <= cb[c, 2] + h
        ):
            span += 1
        key = (row_of[c], c0)
        if key in grid:  # two cells on one slot: keep both texts
            grid[key] = (max(grid[key][0], span), grid[key][1] + " " + ctext[c])
        else:
            grid[key] = (span, ctext[c])
    out = ["<table>"]
    for r in range(len(rows)):
        tag = "th" if r == 0 and len(rows) > 1 else "td"
        tds, c = [], 0
        while c < len(cols):
            if (r, c) in grid:
                span, text = grid[(r, c)]
                attr = f' colspan="{span}"' if span > 1 else ""
                tds.append(f"<{tag}{attr}>{html.escape(text, quote=False)}</{tag}>")
                c += span
            else:
                tds.append(f"<{tag}></{tag}>")
                c += 1
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def page_blocks(
    boxes: np.ndarray,
    texts: list[str],
    order: list[int],
    cls_logits: np.ndarray,
    same: np.ndarray,
) -> list[dict]:
    """The layout blocks behind ``page_markdown``, in reading order:
    {"label" (a BLOCK_CLASSES name), "bbox" (pixels, the union of its lines),
    "text" (its lines, newline-joined)}."""
    out = []
    for blk in split_blocks(order, same):
        b = boxes[blk, :4]
        out.append(
            {
                "label": BLOCK_CLASSES[int(cls_logits[blk].sum(0).argmax())],
                "bbox": [
                    round(float(b[:, 0].min()), 1),
                    round(float(b[:, 1].min()), 1),
                    round(float(b[:, 2].max()), 1),
                    round(float(b[:, 3].max()), 1),
                ],
                "text": "\n".join(texts[i] for i in blk),
            }
        )
    return out


def page_markdown(
    boxes: np.ndarray,
    texts: list[str],
    order: list[int],
    cls_logits: np.ndarray,
    same: np.ndarray,
    keep_furniture: bool = False,
    cells: dict | None = None,
    tab: np.ndarray | None = None,
    pictures: np.ndarray | None = None,
    gray: np.ndarray | None = None,
) -> str:
    """Markdown for one page.  boxes [N, 4], texts [N], order = reading order,
    cls_logits [N, C], same [N, N] (logit "same block").  ``cells`` maps a line
    to its table cells [(box, text)] when the line holds several.  ``tab``
    [3, N, N] (same row / column / cell logits) builds tables from the heads.
    ``pictures`` [P, 4]: picture regions, each written as ``![](picture-<k>)``
    before the first block below it that it overlaps horizontally."""
    parts: list[str] = []
    part_boxes: list = []
    for blk in split_blocks(order, same):
        n_parts = len(parts)
        _block_part(parts, blk, boxes, texts, cls_logits, keep_furniture, cells, tab, gray)
        if len(parts) > n_parts:
            b = boxes[blk, :4]
            part_boxes.append(
                (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())
            )
    if pictures is not None and len(pictures):
        slots: dict[int, list[str]] = {}
        for k, (x0, y0, x1, y1) in enumerate(np.asarray(pictures)[:, :4]):
            cy = (y0 + y1) / 2
            at = next(
                (
                    j
                    for j, (bx0, by0, bx1, by1) in enumerate(part_boxes)
                    if (by0 + by1) / 2 > cy and min(x1, bx1) > max(x0, bx0)
                ),
                len(parts),
            )
            slots.setdefault(at, []).append(f"![](picture-{k})")
        merged = []
        for j in range(len(parts) + 1):
            merged.extend(slots.get(j, []))
            if j < len(parts):
                merged.append(parts[j])
        parts = merged
    return "\n\n".join(p for p in parts if p)


def _block_part(parts, blk, boxes, texts, cls_logits, keep_furniture, cells, tab, gray: np.ndarray | None = None):
    """Append one block's markdown to ``parts`` (nothing for dropped furniture)."""
    cls = int(cls_logits[blk].sum(0).argmax())
    texts_b = [texts[i] for i in blk]
    if cls in (HEADER, FOOTER) and not keep_furniture:
        return
    if all(_WHOLE_MATH.match(t.strip()) for t in texts_b):  # display equations
        parts.append("\n".join("$$" + t.strip()[1:-1] + "$$" for t in texts_b))
        return
    if cls == SECTION:
        parts.append("## " + join_lines(texts_b))
    elif (
        cls == TABLE
        and len(blk) > 1
        and tab is not None
        and heads_plausible(boxes[blk, :4], tab[:, blk][:, :, blk])
    ):
        parts.append(table_html_heads(boxes[blk, :4], texts_b, tab[:, blk][:, :, blk]))
    elif cls == TABLE and len(blk) > 1:
        cb, ct = [], []
        for i in blk:
            for bx, tx in (cells or {}).get(i) or [(boxes[i, :4], texts[i])]:
                cb.append(bx)
                ct.append(tx)
        cb_arr = np.array(cb, np.float32).reshape(-1, 4)
        hrules: list[Rule] | None = None
        if gray is not None and len(cb_arr):
            pad = 12
            H, W = gray.shape
            x0 = max(0, int(cb_arr[:, 0].min()) - pad)
            y0 = max(0, int(cb_arr[:, 1].min()) - pad)
            x1 = min(W, int(cb_arr[:, 2].max()) + pad)
            y1 = min(H, int(cb_arr[:, 3].max()) + pad)
            med_h = float(np.median(cb_arr[:, 3] - cb_arr[:, 1]))
            hrules = [
                Rule(r.x0 + x0, r.y0 + y0, r.x1 + x0, r.y1 + y0)
                for r in detect_hrules(gray[y0:y1, x0:x1], line_h=med_h)
            ]
        parts.append(table_html(cb_arr, ct, hrules=hrules))
    elif cls == LIST:
        items: list[list[str]] = []
        for t in texts_b:
            if not items or _ITEM.match(t):
                items.append([t])
            else:
                items[-1].append(t)
        parts.append("\n".join(join_lines(it) for it in items))
    else:
        parts.append(join_lines(texts_b))
