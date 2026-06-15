"""
esrgan_enhance.py
Real-ESRGAN Super-Resolution — pre-inference enhancement for low-res frames.

Install:
    pip install realesrgan basicsr

If not installed, ESRGANEnhancer falls back to bicubic upscaling silently.
"""

import os
import urllib.request
import numpy as np
import cv2
from pathlib import Path

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    _REALESRGAN_OK = True
except ImportError:
    _REALESRGAN_OK = False
    print("[ESRGAN] realesrgan/basicsr not installed — will use bicubic fallback.")
    print("[ESRGAN] Install: pip install realesrgan basicsr")

_MODEL_URL  = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.1.0/RealESRGAN_x4plus.pth")
_WEIGHT_DIR  = Path(__file__).parent / "weights"
_WEIGHT_PATH = _WEIGHT_DIR / "RealESRGAN_x4plus.pth"


class ESRGANEnhancer:
    """
    Wraps Real-ESRGAN 4x upscaling.

    Usage
    -----
    enhancer = ESRGANEnhancer()
    enhanced  = enhancer.enhance(low_res_bgr_frame)   # returns BGR ndarray

    ⚠ CPU WARNING
    ESRGAN on CPU is 5-15 seconds per frame — only viable for offline processing.
    For real-time enhancement, CUDA GPU is required.
    Set half_precision=False when running on CPU (FP16 is unsupported on CPU).
    """

    def __init__(self, scale: int = 4, half_precision: bool = True):
        self.scale     = scale
        self.available = False
        self.upsampler = None

        if not _REALESRGAN_OK:
            return

        self._ensure_weights()
        self._build_upsampler(half_precision)

    
    def _ensure_weights(self) -> None:
        """Download RealESRGAN_x4plus.pth if absent."""
        _WEIGHT_DIR.mkdir(exist_ok=True)
        if _WEIGHT_PATH.exists():
            return
        print(f"[ESRGAN] Downloading weights → {_WEIGHT_PATH} ...")
        try:
            urllib.request.urlretrieve(_MODEL_URL, str(_WEIGHT_PATH))
            print("[ESRGAN] Download complete.")
        except Exception as e:
            print(f"[ESRGAN] Download failed: {e}. Bicubic fallback active.")

    def _build_upsampler(self, half: bool) -> None:
        """Instantiate RealESRGANer with RRDBNet backbone."""
        if not _WEIGHT_PATH.exists():
            return
        try:
            backbone = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23,
                num_grow_ch=32, scale=self.scale
            )
            self.upsampler = RealESRGANer(
                scale      = self.scale,
                model_path = str(_WEIGHT_PATH),
                model      = backbone,
                tile       = 256,    
                tile_pad   = 10,
                pre_pad    = 0,
                half       = half    
            )
            self.available = True
            print(f"[ESRGAN] Ready  (scale={self.scale}x  half={half})")
        except Exception as e:
            print(f"[ESRGAN] Init failed: {e}. Bicubic fallback active.")

    
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        4x super-resolution on a BGR frame.

        Falls back to cv2.resize (bicubic) if Real-ESRGAN is unavailable.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (any resolution).

        Returns
        -------
        np.ndarray
            Enhanced BGR image (4x resolution if ESRGAN, same logic if fallback).
        """
        if not self.available or self.upsampler is None:
            return self._bicubic_fallback(frame)

        try:
            rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            enhanced_rgb, _ = self.upsampler.enhance(rgb, outscale=self.scale)
            return cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[ESRGAN] enhance() error: {e}. Using bicubic.")
            return self._bicubic_fallback(frame)

    def enhance_if_small(self, frame: np.ndarray,
                         threshold_w: int = 320) -> np.ndarray:
        """
        Only enhance if frame width is below threshold_w.
        Saves compute on already-high-res frames.
        """
        if frame.shape[1] < threshold_w:
            return self.enhance(frame)
        return frame

    
    @staticmethod
    def _bicubic_fallback(frame: np.ndarray) -> np.ndarray:
        """Simple 4x bicubic upscale — used when ESRGAN unavailable."""
        h, w = frame.shape[:2]
        return cv2.resize(frame, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
