import numpy as np

from indra_ocr.layout_post import (
    _header_cmidrules,
    _merge_wrapped,
    _overlaps_x,
    table_html,
)
from indra_ocr.table_rules import Rule


def test_merge_wrapped_three_line_cell():
    boxes = np.array([
        # Header row (y=0..20)
        [0, 0, 100, 20], [150, 0, 250, 20], [300, 0, 450, 20],
        # Row 1: 3 cols (y=50..70), followed by TWO wrapped lines in col 3 (gap=4px)
        [0, 50, 100, 70], [150, 50, 250, 70], [300, 50, 450, 70],
        [300, 74, 450, 94],
        [300, 98, 450, 118],
        # Row 2: 3 cols (y=150..170, gap=32px)
        [0, 150, 100, 170], [150, 150, 250, 170], [300, 150, 450, 170],
        # Row 3: 3 cols (y=200..220, gap=30px)
        [0, 200, 100, 220], [150, 200, 250, 220], [300, 200, 450, 220],
    ], np.float32)

    rows = [[0, 1, 2], [3, 4, 5], [6], [7], [8, 9, 10], [11, 12, 13]]
    assert _merge_wrapped(rows, boxes) == [
        [0, 1, 2],
        [3, 4, 5, 6, 7],
        [8, 9, 10],
        [11, 12, 13],
    ]


def test_merge_wrapped_two_line_section():
    boxes = np.array([
        [0, 10, 100, 30], [150, 10, 250, 30],  # line 0 (len=2)
        [150, 34, 250, 54],                    # line 1 (len=1, gap=4px < 0.6*20)
    ], np.float32)
    rows = [[0, 1], [2]]
    # Default min_rows=3 leaves 2-line borderless tables untouched
    assert _merge_wrapped(rows, boxes) == [[0, 1], [2]]
    # min_rows=2 merges the wrapped continuation inside a bordered section
    assert _merge_wrapped(rows, boxes, min_rows=2) == [[0, 1, 2]]


def test_merge_wrapped_two_full_rows_stay_separate():
    boxes = np.array([
        [0, 10, 100, 30], [150, 10, 250, 30],  # row 0 (len=2)
        [0, 34, 100, 54], [150, 34, 250, 54],  # row 1 (len=2, same width)
    ], np.float32)
    rows = [[0, 1], [2, 3]]
    assert _merge_wrapped(rows, boxes, min_rows=2) == [[0, 1], [2, 3]]


def test_table_html_multi_row_header_and_sparse_first_row():
    boxes = np.array([
        [0, 10, 80, 30], [120, 10, 320, 30],                 # header row 0 (colspan)
        [0, 35, 80, 55], [120, 35, 200, 55], [240, 35, 320, 55],  # header row 1
        [0, 68, 80, 88], [120, 68, 200, 88],                 # sparse row 1 (missing col 2)
        [0, 110, 80, 130], [120, 110, 200, 130], [240, 110, 320, 130],  # row 2
    ], np.float32)
    texts = ["Item", "Group", "", "A", "B", "Row1", "10", "Row2", "20", "30"]
    html = table_html(boxes, texts, hrules=[Rule(0, 62, 320, 62)])

    assert html.count("<tr>") == 4
    # Both header rows above the rule use <th>, and Row1 stays in <td>
    rows_html = [l for l in html.splitlines() if l.startswith("<tr>")]
    assert "<th>" in rows_html[0] and "<th>" in rows_html[1]
    assert "<td>Row1</td>" in rows_html[2]


def test_table_html_full_grid_with_two_line_wrapped_cell():
    boxes = np.array([
        [0, 10, 100, 30], [150, 10, 300, 30],  # header
        [0, 50, 100, 70], [150, 50, 300, 70],  # row 1 line 1
        [150, 74, 300, 94],                    # row 1 line 2 (wrapped inside bordered row)
        [0, 120, 100, 140], [150, 120, 300, 140],  # row 2
    ], np.float32)
    texts = ["Col", "Desc", "1", "First line", "second line", "2", "Single"]
    hrules = [Rule(0, 40, 300, 40), Rule(0, 107, 300, 107)]
    html = table_html(boxes, texts, hrules=hrules)

    assert html.count("<tr>") == 3
    assert "<td>First line second line</td>" in html


def test_overlaps_x():
    assert _overlaps_x(10, 50, 30, 80) is True
    assert _overlaps_x(10, 30, 30, 80) is False
    assert _overlaps_x(10, 20, 30, 80) is False


def test_table_html_wrapped_full_width_header():
    boxes = np.array([
        [0, 10, 90, 30], [140, 10, 240, 30],      # header line 0
        [0, 34, 90, 54], [140, 34, 240, 54],      # header line 1 (gap=4 < 0.6*20)
        [0, 75, 90, 95], [140, 75, 240, 95],      # body row 1
        [0, 110, 90, 130], [140, 110, 240, 130],  # body row 2
    ], np.float32)
    texts = ["First", "Phone", "Name", "Number", "Alice", "123", "Bob", "456"]
    html = table_html(boxes, texts, hrules=[Rule(0, 64, 240, 64)])

    assert html.count("<tr>") == 3
    assert "<tr><th>First Name</th><th>Phone Number</th></tr>" in html


def test_table_html_cmidrule_expands_narrow_super_header():
    boxes = np.array([
        [195, 10, 245, 30],  # super-header line 0 ("Total", narrow & centered)
        [190, 34, 250, 54],  # super-header line 1 ("sample", narrow & centered)
        [0, 68, 90, 88], [140, 68, 200, 88], [230, 68, 290, 88], [330, 68, 400, 88],
        [0, 110, 90, 130], [140, 110, 200, 130], [230, 110, 290, 130], [330, 110, 400, 130],
        [0, 145, 90, 165], [140, 145, 200, 165], [230, 145, 290, 165], [330, 145, 400, 165],
    ], np.float32)
    texts = [
        "Total", "sample", "Item", "n", "%", "p",
        "A", "10", "50", "0.1", "B", "20", "50", "0.2",
    ]
    hrules = [
        Rule(0, 4, 400, 4),      # outer toprule (skipped)
        Rule(140, 60, 290, 60),  # cmidrule over cols 1..2
        Rule(0, 98, 400, 98),    # full header divider
    ]

    assert len(_header_cmidrules([0, 1, 2, 3, 4, 5], boxes, hrules)) == 1
    html = table_html(boxes, texts, hrules=hrules)
    rows_html = [line for line in html.splitlines() if line.startswith("<tr>")]
    assert rows_html[0] == '<tr><th></th><th colspan="2">Total sample</th><th></th></tr>'
    assert rows_html[1] == "<tr><th>Item</th><th>n</th><th>%</th><th>p</th></tr>"


def test_table_html_shared_cmidrule_splits_between_super_headers():
    boxes = np.array([
        [240, 10, 310, 30], [440, 10, 510, 30],  # two narrow super-headers sharing 1 rule
        [0, 50, 150, 70], [200, 50, 260, 70], [290, 50, 350, 70],
        [400, 50, 460, 70], [490, 50, 550, 70], [650, 50, 800, 70],
        [0, 95, 150, 115], [200, 95, 260, 115], [290, 95, 350, 115],
        [400, 95, 460, 115], [490, 95, 550, 115], [650, 95, 800, 115],
    ], np.float32)
    texts = [
        "Oxygenase", "Reductase",
        "Sys", "Gene", "Group", "Gene", "Struct", "Ref",
        "R1", "a", "b", "c", "d", "e",
    ]
    hrules = [Rule(200, 40, 550, 40), Rule(0, 82, 800, 82)]

    html = table_html(boxes, texts, hrules=hrules)
    rows_html = [line for line in html.splitlines() if line.startswith("<tr>")]
    assert rows_html[0] == '<tr><th></th><th colspan="2">Oxygenase</th><th colspan="2">Reductase</th><th></th></tr>'
