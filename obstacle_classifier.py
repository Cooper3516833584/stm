"""Compatibility placeholder for future obstacle classification."""


class ObstacleClassifier:
    def classify_points(self, points_body_cm, now_s):
        return []


class ObstacleClassifierConfig:
    pass


__all__ = ["ObstacleClassifier", "ObstacleClassifierConfig"]

