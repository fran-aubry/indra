import cv2
import numpy as np

from indra_ocr.table_rules import detect_hrules


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