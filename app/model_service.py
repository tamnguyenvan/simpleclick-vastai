from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

from .settings import Settings


class ModelService:
    """Owns the process-local SimpleClick model and predictor.

    The predictor mutates image state, so inference is serialized per process.
    Run one worker per GPU; multiple workers would load another 659M-parameter
    model copy into GPU memory.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._state = "not_loaded"
        self._device: str | None = None
        self._failure: str | None = None
        self._inference_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._model: Any = None
        self._predictor: Any = None
        self._torch: Any = None
        self._click: Any = None
        self._clicker: Any = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def failure(self) -> str | None:
        return self._failure

    def load(self) -> None:
        """Load the model once. Exceptions are retained for readiness reporting."""

        with self._load_lock:
            if self._state == "ready":
                return
            self._state = "loading"
            self._failure = None
            try:
                self._load_model()
            except Exception as exc:
                self._state = "failed"
                self._failure = type(exc).__name__
                raise
            self._state = "ready"

    def _load_model(self) -> None:
        import torch

        root = Path(self.settings.simpleclick_root)
        checkpoint = Path(self.settings.checkpoint_path)
        if not root.is_dir():
            raise RuntimeError(f"SimpleClick source directory does not exist: {root}")
        if not checkpoint.is_file():
            raise RuntimeError(f"SimpleClick checkpoint does not exist: {checkpoint}")

        root_string = str(root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)

        from isegm.inference import utils
        from isegm.inference.clicker import Click, Clicker
        from isegm.inference.predictors import get_predictor

        if self.settings.model_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        device = torch.device(
            "cuda"
            if self.settings.model_device == "auto" and torch.cuda.is_available()
            else self.settings.model_device
            if self.settings.model_device != "auto"
            else "cpu"
        )
        model = utils.load_is_model(
            str(checkpoint),
            device,
            eval_ritm=False,
            cpu_dist_maps=True,
        )
        predictor = get_predictor(
            model,
            "NoBRS",
            device,
            prob_thresh=self.settings.default_threshold,
            with_flip=self.settings.model_with_flip,
            zoom_in_params={
                "skip_clicks": -1,
                "target_size": (
                    self.settings.model_input_size,
                    self.settings.model_input_size,
                ),
                "expansion_ratio": 1.4,
            },
            predictor_params={"max_size": self.settings.max_longest_size},
        )
        self._torch = torch
        self._model = model
        self._predictor = predictor
        self._device = str(device)
        self._click = Click
        self._clicker = Clicker

    def segment(
        self,
        image_rgb: Any,
        positive_points: list[list[int]],
        negative_points: list[list[int]],
        threshold: float,
    ) -> Any:
        if self._state != "ready":
            raise RuntimeError("model is not ready")

        clicker = self._clicker(
            init_clicks=[
                *[
                    self._click(is_positive=True, coords=(point[1], point[0]))
                    for point in positive_points
                ],
                *[
                    self._click(is_positive=False, coords=(point[1], point[0]))
                    for point in negative_points
                ],
            ]
        )
        with self._inference_lock:
            with self._torch.inference_mode():
                self._predictor.set_input_image(image_rgb)
                probabilities = self._predictor.get_prediction(clicker)
            return probabilities > threshold
