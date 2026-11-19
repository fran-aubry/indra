import numpy as np

from indra_ocr.layout_post import _split_sections
from indra_ocr.table_rules import Rule


def test_split_sections_empty_and_no_rules():
    empty = np.zeros((0, 4), np.float32)
    assert _split_sections(empty, None) == []
    assert _split_sections(empty, [Rule(0, 10, 100, 10)]) == []

    boxes = np.array([[0, 10, 100, 30], [0, 50, 100, 70]], np.float32)
    assert _split_sections(boxes, None) == [[0, 1]]
    assert _split_sections(boxes, []) == [[0, 1]]


def test_split_sections_ignores_short_partial_rules():
    boxes = np.array([
        [0, 10, 100, 30], [200, 10, 300, 30],
        [0, 60, 100, 80], [200, 60, 300, 80],
    ], np.float32)
    # Only a narrow cmidrule (< 50% of 300px span) -> no full divider
    assert _split_sections(boxes, [Rule(180, 45, 300, 45)]) == [[0, 1, 2, 3]]


def test_split_sections_horizontal_and_tilted_rules():
    boxes = np.array([
        [0, 10, 100, 30], [200, 10, 300, 30],      # header
        [0, 60, 100, 80], [200, 60, 300, 80],      # row 1
        [0, 120, 100, 140], [200, 120, 300, 140],  # row 2
    ], np.float32)
    hrules = [
        Rule(0, 2, 300, 2),      # outer top border -> ignored
        Rule(0, 45, 300, 45),    # header divider -> splits
        Rule(10, 70, 290, 70),   # cuts through row 1 text -> rejected
        Rule(0, 100, 300, 104),  # slightly tilted divider -> splits
        Rule(0, 150, 300, 150),  # outer bottom border -> ignored
    ]
    assert _split_sections(boxes, hrules) == [[0, 1], [2, 3], [4, 5]]


def test_split_sections_strongly_tilted_rule_where_cy_order_reverses():
    # Rule rises from y=80 on the left to y=30 on the right
    # Box 0 (left, cy=65) is ABOVE the rule; Box 1 (right, cy=50) is BELOW it!
    boxes = np.array([
        [20, 55, 80, 75],     # left, above rule (y_rule(50) ~ 77.5 > 65)
        [520, 40, 580, 60],   # right, below rule (y_rule(550) ~ 34.2 < 50)
    ], np.float32)
    assert _split_sections(boxes, [Rule(0, 80, 600, 30)]) == [[0], [1]]


def test_split_sections_tall_rotated_box_does_not_veto_divider():
    boxes = np.array([
        [0, 10, 20, 140],      # tall rotated rowspan label (h=130 > 2.5*20)
        [50, 10, 250, 30],     # row 1
        [50, 60, 250, 80],     # row 2
        [50, 110, 250, 130],   # row 3
    ], np.float32)
    # Full-width rule passes through y=45 (overlaps tall box 0, between row 1 & row 2)
    sections = _split_sections(boxes, [Rule(0, 45, 250, 45)])
    assert len(sections) == 2
    assert 1 in sections[0]
    assert 2 in sections[1] and 3 in sections[1]


def test_split_sections_strongly_tilted_rule_where_cy_order_reverses():
    # Rule rises from y=80 on the left to y=30 on the right
    # Box 0 (left, cy=65) is ABOVE the rule; Box 1 (right, cy=50) is BELOW it!
    boxes = np.array([
        [20, 55, 80, 75],     # left, above rule (y_rule(50) ~ 77.5 > 65)
        [520, 40, 580, 60],   # right, below rule (y_rule(550) ~ 34.2 < 50)
    ], np.float32)
    assert _split_sections(boxes, [Rule(0, 80, 600, 30)]) == [[0], [1]]


def test_split_sections_tall_rotated_box_does_not_veto_divider():
    boxes = np.array([
        [0, 10, 20, 140],      # tall rotated rowspan label (h=130 > 2.5*20)
        [50, 10, 250, 30],     # row 1
        [50, 60, 250, 80],     # row 2
        [50, 110, 250, 130],   # row 3
    ], np.float32)
    # Full-width rule passes through y=45 (overlaps tall box 0, between row 1 & row 2)
    sections = _split_sections(boxes, [Rule(0, 45, 250, 45)])
    assert len(sections) == 2
    assert 1 in sections[0]
    assert 2 in sections[1] and 3 in sections[1]
