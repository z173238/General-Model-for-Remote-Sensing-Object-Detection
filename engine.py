"""
遥感通用检测引擎 v2 — SM3Det 统一光学+SAR
架构: SM3Det(178M, 光学OBB + SAR HBB) + Cascade R-CNN(77M, 光学Mask)
"""
import sys, os, time, numpy as np, torch
from dataclasses import dataclass, field
from typing import List, Optional
from PIL import Image
import builtins

# === PyTorch 2.12 compat ===
builtins._orig_torch_load = torch.load
def _patched_load(filename, **kw):
    kw['weights_only'] = False
    return builtins._orig_torch_load(filename, **kw)
torch.load = _patched_load

# SM3Det path
SM3DET_PATH = "/home/ubuntu/workspace/General Model for Remote Sensing Object Detection/SM3Det"
sys.path.insert(0, SM3DET_PATH)

# DOTA 26-class names (from SM3Det training config)
DOTA_CLASSES_26 = [
    'baseball-diamond','basketball-court','bridge','container-crane',
    'ground-track-field','harbor','helicopter','helipad','large-vehicle',
    'plane','roundabout','ship','small-vehicle','soccer-ball-field',
    'storage-tank','swimming-pool','tennis-court','airport','train-station',
    'windmill','chimney','dam','golffield','overpass','track','bicycle'
]

# SARDet 6-class names
SAR_CLASSES_6 = ['ship','aircraft','car','bridge','harbor','tank']

# DIOR 20-class names (for DIOR-fine-tuned model)
DIOR_CLASSES_20 = [
    'airplane','airport','baseballfield','basketballcourt','bridge','chimney',
    'dam','Expressway-Service-area','Expressway-toll-station','golffield',
    'groundtrackfield','harbor','overpass','ship','stadium','storagetank',
    'tenniscourt','trainstation','vehicle','windmill'
]


@dataclass
class Detection:
    bbox_hbb: List[float]       # [x1, y1, x2, y2]
    bbox_obb: Optional[List[float]] = None  # [cx, cy, w, h, theta]
    mask: Optional[np.ndarray] = None
    class_name: str = ""
    score: float = 0.0
    modality: str = "optical"

@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    modality: str = "optical"
    inference_time_ms: float = 0.0
    image_shape: tuple = (0, 0)


class UnifiedRSDetectionEngine:
    """遥感通用检测引擎 v2 — SM3Det统一光学+SAR"""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self._sm3det = None       # SM3Det: 光学OBB + SAR HBB
        self._mask_rcnn = None    # Cascade Mask R-CNN: 光学Mask
        self._loaded = False
        self._mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self._std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def load_models(self):
        if self._loaded:
            return

        print("[1/2] SM3Det (178M, 光学OBB+SAR HBB)...")
        self._sm3det = self._load_sm3det()

        print("[2/2] Cascade Mask R-CNN (77M)...")
        self._mask_rcnn = self._load_mask_rcnn()

        self._loaded = True
        sm3det_n = sum(p.numel() for p in self._sm3det.parameters())
        mask_n = sum(p.numel() for p in self._mask_rcnn.parameters())
        print(f"Done! SM3Det {sm3det_n/1e6:.0f}M + Mask {mask_n/1e6:.0f}M = {(sm3det_n+mask_n)/1e6:.0f}M total")

    def _load_sm3det(self):
        from mmcv import Config
        from mmrotate.models.detectors.trisource_H1stage_R2stage_detector import TriSourceDetector

        ckpt_path = f"{SM3DET_PATH}/../SM3Det_weights/ckpt/SM3Det_convnext_t/iter_33468.pth"
        ckpt = torch.load(ckpt_path, map_location='cpu')
        with open('/tmp/sm3det_full.py', 'w') as f:
            f.write(ckpt['meta']['config'])
        cfg = Config.fromfile('/tmp/sm3det_full.py')

        model = TriSourceDetector(
            backbone=cfg.model.backbone, neck=cfg.model.neck,
            sar_bbox_head=cfg.model.sar_bbox_head,
            sar_train_cfg=cfg.model.sar_train_cfg, sar_test_cfg=cfg.model.sar_test_cfg,
            rgb_rpn_head=cfg.model.rgb_rpn_head, rgb_roi_head=cfg.model.rgb_roi_head,
            rgb_train_cfg=cfg.model.rgb_train_cfg, rgb_test_cfg=cfg.model.rgb_test_cfg,
        )
        model.load_state_dict(ckpt['state_dict'], strict=False)
        model.eval().cuda()
        return model

    def _load_mask_rcnn(self):
        from mmdet.apis import init_detector
        import mmdet
        config = os.path.join(os.path.dirname(mmdet.__path__[0]),
            "mmdet/.mim/configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py")
        ckpt = os.path.expanduser("~/.cache/torch/hub/checkpoints/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth")
        model = init_detector(config, ckpt, device=self.device)
        return model

    def detect(self, image: np.ndarray, modality: str = "auto",
               tasks: List[str] = None) -> DetectionResult:
        if not self._loaded:
            self.load_models()

        t0 = time.time()

        if modality == "auto":
            modality = "sar" if (len(image.shape) == 2 or image.shape[-1] == 1) else "optical"

        if tasks is None:
            tasks = ["obb", "mask"] if modality == "optical" else ["hbb"]

        result = DetectionResult(modality=modality, image_shape=image.shape[:2])

        if modality == "optical":
            result.detections = self._detect_optical(image, tasks)
        else:
            result.detections = self._detect_sar(image)

        result.inference_time_ms = (time.time() - t0) * 1000
        return result

    def _preprocess(self, img, size=800):
        """预处理: resize + normalize → tensor"""
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        h, w = img.shape[:2]
        img_800 = np.array(Image.fromarray(img.astype(np.uint8)).resize((size, size)))
        img_norm = (img_800.astype(np.float32) - self._mean) / self._std
        img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().cuda()
        return img_t, h, w

    def _detect_optical(self, img: np.ndarray, tasks: List[str]) -> List[Detection]:
        dets = []
        img_t, h, w = self._preprocess(img)

        # OBB (SM3Det optical head) + HBB (OBB→外接矩形)
        if "obb" in tasks or "hbb" in tasks:
            with torch.no_grad():
                r = self._sm3det.simple_test(
                    img_t,
                    [{'img_shape': (800, 800, 3), 'scale_factor': np.ones(4),
                      'ori_shape': (h, w, 3)}],
                    subdataset=[['rgb']]
                )
            for cid, class_dets in enumerate(r[0]):
                if len(class_dets) == 0:
                    continue
                for det in class_dets:
                    score = float(det[5])  # [x1,y1,x2,y2,angle,score]
                    if score < 0.3:
                        continue
                    x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                    theta = float(det[4])
                    # Scale to original image
                    sx, sy = w / 800.0, h / 800.0
                    x1 *= sx; y1 *= sy; x2 *= sx; y2 *= sy
                    hbb = [x1, y1, x2, y2]
                    # OBB: [cx, cy, w, h, theta]
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    bw, bh = x2-x1, y2-y1
                    cn = DOTA_CLASSES_26[cid] if cid < len(DOTA_CLASSES_26) else f"cls_{cid}"
                    if "hbb" in tasks:
                        dets.append(Detection(bbox_hbb=hbb, class_name=cn,
                            score=score, modality="optical"))
                    if "obb" in tasks:
                        dets.append(Detection(bbox_hbb=hbb, bbox_obb=[cx, cy, bw, bh, theta],
                            class_name=cn, score=score, modality="optical"))

        # Mask (Cascade Mask R-CNN)
        if "mask" in tasks:
            from mmdet.apis import inference_detector
            result = inference_detector(self._mask_rcnn, img)
            if hasattr(result, 'pred_instances'):
                pred = result.pred_instances
            elif isinstance(result, tuple):
                pred = result[0]
            else:
                pred = result

            if isinstance(pred, list) and len(pred) == 2:
                bboxes_list, masks_list = pred
                for cid, (boxes, masks) in enumerate(zip(bboxes_list, masks_list)):
                    if len(boxes) == 0:
                        continue
                    for i in range(len(boxes)):
                        score = float(boxes[i][4])
                        if score < 0.3:
                            continue
                        b = boxes[i][:4].tolist()
                        mask = masks[i] if i < len(masks) else None
                        cn = f"cls_{cid}"
                        dets.append(Detection(bbox_hbb=b, mask=mask, class_name=cn,
                            score=score, modality="optical"))

        return dets

    def _detect_sar(self, img: np.ndarray) -> List[Detection]:
        dets = []
        img_t, h, w = self._preprocess(img)

        with torch.no_grad():
            r = self._sm3det.simple_test(
                img_t,
                [{'img_shape': (800, 800, 3), 'scale_factor': np.ones(4),
                  'ori_shape': (h, w, 3)}],
                subdataset=[['sar']]
            )
        # SAR returns per-class bboxes
        if r and r[0]:
            for cid, class_dets in enumerate(r[0]):
                if len(class_dets) == 0:
                    continue
                for det in class_dets:
                    score = float(det[4])
                    if score < 0.3:
                        continue
                    bbox = det[:4].tolist()
                    sx, sy = w / 800.0, h / 800.0
                    bbox[0] *= sx; bbox[1] *= sy
                    bbox[2] *= sx; bbox[3] *= sy
                    cn = SAR_CLASSES_6[cid] if cid < len(SAR_CLASSES_6) else f"cls_{cid}"
                    score_val = min(score, 1.0) if score > 1.0 else score
                    dets.append(Detection(bbox_hbb=bbox, class_name=cn,
                        score=score_val, modality="sar"))
        return dets

    def print_summary(self):
        if not self._loaded:
            self.load_models()
        print(f"\n{'='*50}")
        print("遥感通用检测引擎 v2 — Model Summary")
        print(f"{'='*50}")
        sm3det_n = sum(p.numel() for p in self._sm3det.parameters())
        mask_n = sum(p.numel() for p in self._mask_rcnn.parameters())
        print(f"  SM3Det     : {sm3det_n/1e6:7.1f}M  (光学OBB + SAR HBB)")
        print(f"  Mask R-CNN : {mask_n/1e6:7.1f}M  (光学Mask)")
        print(f"  Total      : {(sm3det_n+mask_n)/1e6:7.1f}M")
        print(f"{'='*50}")


if __name__ == "__main__":
    import random
    engine = UnifiedRSDetectionEngine()
    engine.print_summary()

    # Test optical
    dior_dir = "/home/ubuntu/dataset/DIOR/JPEGImages/JPEGImages-trainval"
    imgs = random.sample(sorted(os.listdir(dior_dir)), 4)
    print("\n--- Optical Test ---")
    for fname in imgs:
        img = np.array(Image.open(f"{dior_dir}/{fname}").convert('RGB'))
        r = engine.detect(img, "optical", ["obb"])
        if r.detections:
            print(f"{fname}: {len(r.detections)} OBB | {r.inference_time_ms:.0f}ms")
            for d in r.detections[:3]:
                print(f"  {d.class_name:20s} {d.score:.3f}")

    # Test SAR
    ssdd_dir = "/home/ubuntu/.cache/kagglehub/datasets/bitsandlayers/sar-ship-detection-dataset/versions/1/SSDD/images/test"
    if os.path.exists(ssdd_dir):
        print("\n--- SAR Test ---")
        sar_imgs = sorted(os.listdir(ssdd_dir))[:3]
        for fname in sar_imgs:
            img = np.array(Image.open(f"{ssdd_dir}/{fname}").convert('L'))
            r = engine.detect(img, "sar")
            print(f"{fname}: {len(r.detections)} ships | {r.inference_time_ms:.0f}ms")
            for d in r.detections[:3]:
                print(f"  {d.class_name} {d.score:.3f}")

    print("\n✅ Engine v2 ready!")
