import numpy as np

from indra_ocr.layout_post import _merge_wrapped


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