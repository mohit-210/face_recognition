import cv2
import numpy as np


class LivenessDetector:
    """
    Multi-signal liveness scoring:
    1) texture realism with Laplacian variance
    2) edge-noise consistency for screen/photo detection
    3) challenge response (left/right/up/down)
    4) frame-to-frame motion depth cue
    """

    def score(
        self,
        face_bgr: np.ndarray,
        landmarks: dict,
        expected_challenge: str | None = None,
        challenge_response: str | None = None,
        previous_face_bgr: np.ndarray | None = None,
    ) -> float:
        texture = self._texture_score(face_bgr)
        moire = self._moire_penalty(face_bgr)
        challenge = self._challenge_score(landmarks, expected_challenge, challenge_response)
        motion = self._motion_score(face_bgr, previous_face_bgr)

        final_score = (0.35 * texture) + (0.20 * (1 - moire)) + (0.25 * challenge) + (0.20 * motion)
        return float(max(0.0, min(1.0, final_score)))

    def _texture_score(self, face_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Lower variance often appears in printed or replayed media.
        return float(min(1.0, variance / 180.0))

    def _moire_penalty(self, face_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        fft = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.log(np.abs(fft) + 1)
        high_freq_energy = float(np.mean(magnitude[magnitude > np.percentile(magnitude, 90)]))
        # Strong repeated high-frequency patterns can indicate display replay.
        return float(min(1.0, high_freq_energy / 15.0))

    def _challenge_score(self, landmarks: dict, expected: str | None, response: str | None) -> float:
        if not expected:
            return 0.5
        if response and response.lower() == expected.lower():
            return 1.0
        if not landmarks:
            return 0.0

        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")
        nose = landmarks.get("nose")
        if not (left_eye and right_eye and nose):
            return 0.0

        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        delta_x = nose[0] - eye_center_x

        if expected.lower() == "left" and delta_x < -4:
            return 1.0
        if expected.lower() == "right" and delta_x > 4:
            return 1.0
        return 0.0

    def _motion_score(self, face_bgr: np.ndarray, previous_face_bgr: np.ndarray | None) -> float:
        if previous_face_bgr is None or previous_face_bgr.size == 0:
            return 0.5
        current_gray = cv2.cvtColor(cv2.resize(face_bgr, (128, 128)), cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(cv2.resize(previous_face_bgr, (128, 128)), cv2.COLOR_BGR2GRAY)
        flow = cv2.absdiff(current_gray, prev_gray)
        motion = float(np.mean(flow))
        return float(min(1.0, motion / 25.0))
