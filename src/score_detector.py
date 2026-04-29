# score_detector.py

import cv2
import easyocr
import torch
import re


class ScoreDetector:
    """
    Uses OCR to read the tennis scoreboard (player names + scores)
    from a broadcast frame.
    """

    def __init__(self):
        # Create OCR reader once
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

        # Crop ratios for the scoreboard region (bottom-left area)
        # You can tweak these if needed.
        self.x1_ratio = 0.02   # left
        self.x2_ratio = 0.40   # right
        self.y1_ratio = 0.80   # top
        self.y2_ratio = 0.97   # bottom

    def _crop_scoreboard(self, frame):
        """
        Crop the region of interest that contains the scoreboard.
        """
        h, w = frame.shape[:2]
        x1 = int(self.x1_ratio * w)
        x2 = int(self.x2_ratio * w)
        y1 = int(self.y1_ratio * h)
        y2 = int(self.y2_ratio * h)

        roi = frame[y1:y2, x1:x2]
        return roi

    def read_score(self, frame):
        """
        Read scoreboard from a single frame.

        Returns a dict:
        {
            "player_top": str or None,
            "player_bottom": str or None,
            "games_top": int or None,
            "points_top": int or None,
            "games_bottom": int or None,
            "points_bottom": int or None,
            "raw_text": str  # full OCR text (for debugging)
        }
        """
        roi = self._crop_scoreboard(frame)

        # Run OCR on the cropped region
        results = self.reader.readtext(roi, detail=0)
        joined = " ".join(results)

        # Basic cleaning
        joined_clean = joined.replace("|", " ").replace(":", " ")
        joined_clean = re.sub(r"\s+", " ", joined_clean).strip()

        words = joined_clean.split()

        score = {
            "player_top": None,
            "player_bottom": None,
            "games_top": None,
            "points_top": None,
            "games_bottom": None,
            "points_bottom": None,
            "raw_text": joined_clean,
        }

        # Separate possible names (letters) and numbers (scores)
        names = []
        numbers = []

        for w in words:
            # Handle words like "5", "40", "15"
            if re.fullmatch(r"\d+", w):
                numbers.append(int(w))
            # Handle player names like "JABEUR", "TOMLJANOVIC"
            elif w.isalpha():
                names.append(w.upper())

        # Assign player names (top / bottom)
        if len(names) >= 2:
            score["player_top"] = names[0]
            score["player_bottom"] = names[1]

        # Assign scores if we have at least 4 numbers:
        #   games_top, points_top, games_bottom, points_bottom
        if len(numbers) >= 4:
            score["games_top"] = numbers[0]
            score["points_top"] = numbers[1]
            score["games_bottom"] = numbers[2]
            score["points_bottom"] = numbers[3]

        return score
