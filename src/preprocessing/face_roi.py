"""Face detection + ROI extraction using MediaPipe FaceMesh.

Produces three rectangular ROIs per frame:
    forehead, left_cheek, right_cheek

Two modes:
    'geometric' — ROIs placed on the landmark bounding box
    'landmark'  — convex hulls over fixed landmark index groups
"""
from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Landmark index groups (MediaPipe 468-point mesh)
LM_FOREHEAD = [10, 67, 69, 104, 108, 109, 151, 297, 299, 333, 337, 338]
LM_LEFT_CHEEK = [50, 101, 116, 117, 118, 119, 120, 121, 128, 142,
                 203, 205, 206, 207, 212, 216]
LM_RIGHT_CHEEK = [280, 330, 345, 346, 347, 348, 349, 350, 357, 371,
                  423, 425, 426, 427, 432, 436]


class FaceROIExtractor:
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 static_image_mode: bool = False):
        self.mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def landmarks(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        res = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

    @staticmethod
    def _geometric_rois(pts: np.ndarray):
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        bw, bh = x1 - x0, y1 - y0

        def box(rx0, ry0, rx1, ry1):
            return (int(x0 + rx0 * bw), int(y0 + ry0 * bh),
                    int(x0 + rx1 * bw), int(y0 + ry1 * bh))

        return {
            "forehead":    box(0.30, 0.08, 0.70, 0.22),
            "left_cheek":  box(0.12, 0.55, 0.32, 0.72),
            "right_cheek": box(0.68, 0.55, 0.88, 0.72),
        }

    @staticmethod
    def _hull_roi(pts, idx):
        hull = cv2.convexHull(pts[idx].astype(np.int32))
        x, y, w, h = cv2.boundingRect(hull)
        return (x, y, x + w, y + h)

    def extract(self, frame_bgr: np.ndarray, mode: str = "geometric"):
        pts = self.landmarks(frame_bgr)
        if pts is None:
            return None
        if mode == "geometric":
            rois = self._geometric_rois(pts)
        else:
            rois = {
                "forehead":    self._hull_roi(pts, LM_FOREHEAD),
                "left_cheek":  self._hull_roi(pts, LM_LEFT_CHEEK),
                "right_cheek": self._hull_roi(pts, LM_RIGHT_CHEEK),
            }
        H, W = frame_bgr.shape[:2]
        out = {}
        for name, (x0, y0, x1, y1) in rois.items():
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            out[name] = (frame_bgr[y0:y1, x0:x1].copy(),
                         (x0, y0, x1, y1))
        return out or None

    def close(self):
        self.mesh.close()


def extract_rgb_traces(video_path: str, mode: str = "geometric",
                       max_frames: int | None = None):
    """Run face detection over the video and return per-ROI mean RGB.

    Returns (traces, fps, valid_mask).
    traces: {roi_name: (N,3) float32 in R,G,B order}
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    extractor = FaceROIExtractor()

    traces = {k: [] for k in ("forehead", "left_cheek", "right_cheek")}
    valid = []
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and n >= max_frames:
            break

        rois = extractor.extract(frame, mode=mode)
        if rois is None:
            valid.append(False)
            for k in traces:
                traces[k].append(np.zeros(3, np.float32))
        else:
            valid.append(True)
            for k in traces:
                if k in rois:
                    crop = rois[k][0].astype(np.float32)
                    b, g, r = (crop[:, :, 0].mean(),
                               crop[:, :, 1].mean(),
                               crop[:, :, 2].mean())
                    traces[k].append(np.array([r, g, b], np.float32))
                else:
                    traces[k].append(np.zeros(3, np.float32))
        n += 1

    cap.release()
    extractor.close()

    traces = {k: np.stack(v) for k, v in traces.items()}
    return traces, fps, np.array(valid)