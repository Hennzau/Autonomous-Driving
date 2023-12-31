import os
from typing import Callable

import numpy as np
import torch
from dora import DoraStatus

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
DEVICE = os.environ.get("PYTORCH_DEVICE") or "cpu"
YOLOV5_PATH = os.environ.get("YOLOV5_PATH")
YOLOV5_WEIGHT_PATH = os.environ.get("YOLOV5_WEIGHT_PATH")


class Operator:
    """
    Send `bbox` found by YOLOv5 on given `image`
    """

    def __init__(self):
        if YOLOV5_PATH is None:
            # With internet
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                "yolov5n",
            )
        else:
            # Without internet
            self.model = torch.hub.load(
                YOLOV5_PATH,
                "custom",
                path=YOLOV5_WEIGHT_PATH,
                source="local",
            )

        self.model.to(torch.device(DEVICE))
        self.model.eval()

        self.camera_position = np.array([]).tobytes()

    def on_event(
            self,
            dora_event: dict,
            send_output: Callable[[str, bytes], None],
    ) -> DoraStatus:
        if dora_event["type"] == "INPUT":
            return self.on_input(dora_event, send_output)
        return DoraStatus.CONTINUE

    def on_input(
            self,
            dora_input: dict,
            send_output: Callable[[str, bytes], None],
    ) -> DoraStatus:
        if dora_input["id"] == "camera_position":
            self.camera_position = dora_input["data"]

        if dora_input["id"] == "camera_image":
            frame = np.frombuffer(
                dora_input["data"],
                np.uint8,
            ).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
            frame = frame[:, :, :3]

            results = self.model(frame)  # includes NMS
            arrays = np.array(results.xyxy[0].cpu())[
                     :, [0, 2, 1, 3, 4, 5]
                     ]  # xyxy -> xxyy

            arrays[:, 4] *= 100
            arrays = arrays.astype(np.int32)
            arrays = arrays.tobytes()

            send_output("bbox", arrays)
            send_output("camera_position", self.camera_position)
            send_output("camera_image", dora_input["data"])

        return DoraStatus.CONTINUE
