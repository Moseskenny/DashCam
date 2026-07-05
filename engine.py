import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from typing import Dict, Tuple


class ADASPipeline:
    CITYSCAPES_COLORS = {
        0: (0, 255, 0),     # road -> green (BGR)
        13: (255, 0, 0),    # vehicle -> blue (BGR)
        11: (0, 0, 255),    # pedestrian -> red (BGR)
        12: (0, 0, 255),    # rider -> red (BGR)
    }

    HAZARD_CLASSES = [11, 12, 13]

    DANGER_CLASSES = [11, 12]
    VEHICLE_CLASSES = [13]

    def __init__(self, model_name: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model.eval()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        orig_h, orig_w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        upsampled = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        mask = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        overlay = frame.copy()
        for class_id, color in self.CITYSCAPES_COLORS.items():
            class_mask = mask == class_id
            overlay[class_mask] = color

        annotated = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        zone_y1 = int(orig_h * 0.65)
        zone_y2 = orig_h
        zone_x1 = int(orig_w * 0.25)
        zone_x2 = int(orig_w * 0.75)

        zone_h = zone_y2 - zone_y1
        zone_w = zone_x2 - zone_x1
        total_zone_pixels = zone_h * zone_w

        warning_zone = mask[zone_y1:zone_y2, zone_x1:zone_x2]

        vehicle_in_zone = np.isin(warning_zone, self.VEHICLE_CLASSES)
        danger_in_zone = np.isin(warning_zone, self.DANGER_CLASSES)

        vehicle_occ = float(np.sum(vehicle_in_zone) / total_zone_pixels * 100)
        danger_occ = float(np.sum(danger_in_zone) / total_zone_pixels * 100)

        hazard_detected = vehicle_occ > 15.0 or danger_occ > 15.0
        hazard_type = None
        if hazard_detected:
            hazard_type = "Vehicle" if vehicle_occ > 15.0 else "Pedestrian"

        cv2.rectangle(annotated, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 0, 255), 2)

        if hazard_detected:
            label = f"HAZARD: {hazard_type} ({max(vehicle_occ, danger_occ):.0f}%)"
            cv2.putText(annotated, label, (zone_x1, zone_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(annotated, "CLEAR", (zone_x1, zone_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        vehicle_count = int(np.sum(np.isin(mask, self.VEHICLE_CLASSES)))
        pedestrian_count = int(np.sum(np.isin(mask, self.DANGER_CLASSES)))
        road_pct = float(np.sum(mask == 0) / (orig_h * orig_w) * 100)

        telemetry = {
            "hazard_detected": hazard_detected,
            "hazard_type": hazard_type,
            "road_pct": round(road_pct, 1),
            "vehicle_count": vehicle_count,
            "pedestrian_count": pedestrian_count,
            "warning_zone_occupancy": round(max(vehicle_occ, danger_occ), 1),
        }

        return annotated, telemetry

    @staticmethod
    def draw_hud(frame: np.ndarray, telemetry: Dict, display_frame: int = 0) -> np.ndarray:
        hud = frame.copy()
        h, w = hud.shape[:2]

        overlay_bg = np.full((h, w, 3), (0, 0, 0), dtype=np.uint8)

        bar_h = 32
        cv2.rectangle(overlay_bg, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.rectangle(overlay_bg, (0, h - bar_h), (w, h), (0, 0, 0), -1)

        cv2.addWeighted(overlay_bg, 0.55, hud, 0.45, 0, dst=hud)

        hazard = telemetry.get("hazard_detected", False)
        occ = telemetry.get("warning_zone_occupancy", 0)
        hazard_type = telemetry.get("hazard_type", "")
        vehicles = telemetry.get("vehicle_count", 0)
        peds = telemetry.get("pedestrian_count", 0)
        road_pct = telemetry.get("road_pct", 0)
        timestamp = telemetry.get("timestamp", "")

        if hazard:
            status_color = (0, 0, 255)
            status_text = f"HAZARD: {hazard_type} ({occ}%)"
        else:
            status_color = (0, 255, 0)
            status_text = "STATUS: CLEAR"

        font = cv2.FONT_HERSHEY_DUPLEX
        fs = 0.5
        thickness = 1

        cv2.putText(hud, f"FRAME {display_frame}", (12, 22), font, fs, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(hud, timestamp, (int(w * 0.22), 22), font, fs, (200, 200, 200), thickness, cv2.LINE_AA)
        cv2.putText(hud, status_text, (w - 280, 22), font, fs, status_color, thickness, cv2.LINE_AA)

        obj_text = f"V:{vehicles}  P:{peds}  R:{road_pct}%"
        cv2.putText(hud, obj_text, (12, h - 10), font, fs, (180, 180, 180), thickness, cv2.LINE_AA)

        if hazard:
            warn_text = "BRAKE"
            (tw, th), _ = cv2.getTextSize(warn_text, font, 0.8, 2)
            bx = w - tw - 24
            by = h - 38
            cv2.rectangle(hud, (bx - 6, by - 6), (bx + tw + 6, by + th + 6), (0, 0, 200), -1)
            cv2.rectangle(hud, (bx - 6, by - 6), (bx + tw + 6, by + th + 6), (0, 0, 255), 2)
            cv2.putText(hud, warn_text, (bx, by + th), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return hud
