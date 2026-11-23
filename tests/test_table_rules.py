import cv2
import numpy as np

from indra_ocr.table_rules import detect_hrules, detect_vrules


def test_detect_hrules_blank():
    blank = np.full((150, 300), 255, dtype=np.uint8)
    assert detect_hrules(blank) == []


def test_detect_hrules_flat_and_ignores_vertical_and_short():
    img = np.full((200, 400), 255, dtype=np.uint8)
    cv2.line(img, (20, 40), (380, 40), 0, 2)    # full horizontal rule
    cv2.line(img, (20, 150), (380, 150), 0, 2)  # full horizontal rule
    cv2.line(img, (200, 20), (200, 180), 0, 2)  # vertical line (should be ignored)
    cv2.line(img, (50, 95), (75, 95), 0, 2)     # short 25px dash (should be ignored)

    segs = detect_hrules(img)
    assert any(abs((y0 + y1) / 2 - 40) <= 3 and x1 - x0 >= 100 for x0, y0, x1, y1 in segs)
    assert any(abs((y0 + y1) / 2 - 150) <= 3 and x1 - x0 >= 100 for x0, y0, x1, y1 in segs)
    # Nothing near the short dash row y=95
    assert not any(abs((y0 + y1) / 2 - 95) <= 5 for _, y0, _, y1 in segs)


def test_detect_hrules_tilted_line():
    img = np.full((200, 400), 255, dtype=np.uint8)
    cv2.line(img, (20, 60), (380, 74), 0, 2)  # ~2.2 degree tilt (14px drop)

    segs = detect_hrules(img)
    assert any(
        x0 <= 30 and x1 >= 370 and abs(y0 - 60) <= 3 and abs(y1 - 74) <= 3
        for x0, y0, x1, y1 in segs
    )


def test_detect_hrules_shaded_header_and_cmidrule():
    img = np.full((180, 400), 255, dtype=np.uint8)
    img[:80, :] = 180                           # gray-shaded header
    cv2.line(img, (80, 40), (220, 40), 0, 2)    # partial cmidrule inside shaded area

    segs = detect_hrules(img)
    assert any(abs((y0 + y1) / 2 - 40) <= 3 and 75 <= x0 and x1 <= 225 for x0, y0, x1, y1 in segs)
    assert any(abs((y0 + y1) / 2 - 80) <= 3 and x1 - x0 >= 350 for x0, y0, x1, y1 in segs)


def test_detect_vrules_blank():
    blank = np.full((150, 300), 255, dtype=np.uint8)
    assert detect_vrules(blank) == []


def test_detect_vrules_straight_and_ignores_horizontal_and_short():
    img = np.full((200, 400), 255, dtype=np.uint8)
    cv2.line(img, (80, 20), (80, 180), 0, 2)     # vertical rule 1
    cv2.line(img, (260, 20), (260, 180), 0, 2)   # vertical rule 2
    cv2.line(img, (20, 100), (380, 100), 0, 2)   # horizontal line (should be ignored)
    cv2.line(img, (170, 50), (170, 75), 0, 2)    # short 25px stroke (should be ignored)

    vrules = detect_vrules(img)
    assert len(vrules) == 2
    assert abs((vrules[0].x0 + vrules[0].x1) / 2 - 80) <= 3 and vrules[0].height >= 140
    assert abs((vrules[1].x0 + vrules[1].x1) / 2 - 260) <= 3 and vrules[1].height >= 140
    assert not any(abs((r.x0 + r.x1) / 2 - 170) <= 5 for r in vrules)


def test_detect_vrules_tilted_line():
    img = np.full((200, 400), 255, dtype=np.uint8)
    cv2.line(img, (120, 10), (132, 190), 0, 2)   # ~3.8 degree tilt (12px drift)

    vrules = detect_vrules(img)
    assert len(vrules) == 1
    r = vrules[0]
    assert r.y0 <= 20 and r.y1 >= 180
    assert abs(r.x_at(100) - 126) <= 3


def test_detect_vrules_bridges_dark_crossings_but_ignores_white_gaps():
    img = np.full((200, 350), 255, dtype=np.uint8)
    # Real vertical rule crossed by thick horizontal grid lines every 25px
    cv2.line(img, (100, 20), (100, 180), 0, 2)
    for y in range(40, 170, 25):
        cv2.line(img, (50, y), (150, y), 0, 3)

    # Stacked letter-like vertical strokes separated by 5px white gaps (must be ignored)
    for y in range(20, 170, 17):
        cv2.line(img, (250, y), (250, y + 11), 0, 2)

    vrules = detect_vrules(img)
    assert len(vrules) == 1
    assert abs((vrules[0].x0 + vrules[0].x1) / 2 - 100) <= 3
    assert vrules[0].height >= 140