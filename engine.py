import sys, os, time, numpy as np, torch
from dataclasses import dataclass, field
from typing import List, Optional
from PIL import Image

_orig_load = torch.load
def _p(f, map_location=None, weights_only=False, **kw):
    return _orig_load(f, map_location=map_location, weights_only=False, **kw)
torch.load = _p

@dataclass
class Detection:
    bbox_hbb: List[float]
    bbox_obb: Optional[List[float]] = None
    mask: Optional[np.ndarray] = None
    class_name: str = ""
    score: float = 0.0
    modality: str = "optical"

@dataclass
class DetectionResult:
    detections: List[Detection]
    modality: str = "optical"
    inference_time_ms: float = 0.0
    image_shape: tuple = (0, 0)

COCO_CLASSES = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
DOTA_CLASSES = ['plane','baseball-diamond','bridge','ground-track-field','small-vehicle','large-vehicle','ship','tennis-court','basketball-court','storage-tank','soccer-ball-field','roundabout','harbor','swimming-pool','helicopter']
SAR_CLASSES = ['ship','aircraft','car','bridge','harbor','tank']

class UnifiedRSDetectionEngine:
    def __init__(self, device="cuda:0"):
        self.device = device
        self._models = {}
        self._loaded = False

    def load_models(self):
        if self._loaded: return
        import mmdet
        from mmdet.utils import register_all_modules
        register_all_modules()
        cdir = os.path.join(os.path.dirname(mmdet.__path__[0]), "mmdet/.mim/configs")
        cache = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        from mmdet.apis import init_detector

        print("[1/4] HBB: RTMDet-tiny...")
        self._models['hbb'] = init_detector(f"{cdir}/rtmdet/rtmdet_tiny_8xb32-300e_coco.py", f"{cache}/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth", device=self.device)

        print("[2/4] OBB: Rotated RTMDet-tiny...")
        from mmrotate.utils import register_all_modules as rram; rram()
        self._models['obb'] = init_detector("/home/ubuntu/workspace/General Model for Remote Sensing Object Detection/mmrotate-1.x/configs/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota.py", f"{cache}/rotated_rtmdet_tiny-3x-dota-9d821076.pth", device=self.device)

        print("[3/4] Mask: Cascade Mask R-CNN...")
        self._models['mask'] = init_detector(f"{cdir}/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py", f"{cache}/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth", device=self.device)

        print("[4/4] SAR: SM3Det...")
        self._models['sar'] = self._load_sar()
        self._loaded = True
        total = sum(sum(p.numel() for p in m.parameters()) if not isinstance(m,dict) else sum(sum(p2.numel() for p2 in m2.parameters()) for m2 in m.values()) for m in self._models.values())
        print(f"Done! {total/1e6:.0f}M params total")

    def _load_sar(self):
        import importlib.util
        from mmengine.model import BaseModule, constant_init, trunc_normal_init, normal_init
        from mmdet.registry import MODELS
        import mmcv.cnn, mmcv
        mmcv.cnn.constant_init = constant_init
        mmcv.cnn.trunc_normal_init = trunc_normal_init
        mmcv.cnn.normal_init = normal_init
        def af(a=None,**k): return a if callable(a) else (lambda f:f)
        r = type(sys)('mmcv.runner'); r.BaseModule=BaseModule; r.auto_fp16=af
        sys.modules['mmcv.runner']=r; mmcv.runner=r
        b = type(sys)('mmrotate.models.builder'); b.ROTATED_BACKBONES=MODELS; b.ROTATED_NECKS=MODELS
        sys.modules['mmrotate.models.builder']=b
        for m in ['mmrotate.models','mmrotate.models.backbones','mmrotate.models.necks']:
            sys.modules.setdefault(m, type(sys)(m))
        sm3 = "/home/ubuntu/workspace/General Model for Remote Sensing Object Detection/SM3Det"
        s1 = importlib.util.spec_from_file_location("mmrotate.models.backbones.convnext_moe", f"{sm3}/mmrotate/models/backbones/convnext_moe.py")
        mo = importlib.util.module_from_spec(s1); sys.modules['mmrotate.models.backbones.convnext_moe']=mo; s1.loader.exec_module(mo)
        s2 = importlib.util.spec_from_file_location("mmrotate.models.necks.Multitask_FPN", f"{sm3}/mmrotate/models/necks/Multitask_FPN.py")
        no = importlib.util.module_from_spec(s2); sys.modules['mmrotate.models.necks.Multitask_FPN']=no; s2.loader.exec_module(no)
        bb = mo.ConvNeXt_moe_MultiInput(MoE_Block_inds=[[],[0,2],[0,2,4,6,8],[0,2]], datasets=None, num_experts=8, top_k=2, arch='tiny', drop_path_rate=0.1)
        nk = no.MultitaskFPN(in_channels=[96,192,384,768], out_channels=256, extra_level=1, add_extra_convs='on_output', num_outs=5)
        from mmdet.models.dense_heads import GFLHead
        hd = GFLHead(num_classes=26, in_channels=256, stacked_convs=4, feat_channels=256, anchor_generator=dict(type='AnchorGenerator', ratios=[1.0], octave_base_scale=8, scales_per_octave=1, strides=[8,16,32,64,128]), loss_cls=dict(type='QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=1.0), loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25), reg_max=16, loss_bbox=dict(type='GIoULoss', loss_weight=2.0))
        sd = torch.load(f"{sm3}/../SM3Det_weights/ckpt/SM3Det_convnext_t/iter_33468.pth", map_location='cpu')['state_dict']
        bb.load_state_dict({k.replace('backbone.',''):v for k,v in sd.items() if k.startswith('backbone.')}, strict=True)
        nk.load_state_dict({k.replace('neck.',''):v for k,v in sd.items() if k.startswith('neck.')}, strict=True)
        hd.load_state_dict({k.replace('sar_bbox_head.',''):v for k,v in sd.items() if k.startswith('sar_bbox_head.')}, strict=True)
        print(f"        bb={type(bb).__name__} nk={type(nk).__name__} hd={type(hd).__name__}")
        bb.eval(); bb.cuda()
        nk.eval(); nk.cuda()
        hd.eval(); hd.cuda()
        from mmengine.config import ConfigDict
        hd.test_cfg = ConfigDict(dict(nms_pre=1000, min_bbox_size=0, score_thr=0.05, nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))
        return {'backbone':bb, 'neck':nk, 'head':hd}

    def detect(self, image, modality="auto", tasks=None):
        if not self._loaded: self.load_models()
        t0 = time.time()
        if modality == "auto":
            modality = "sar" if (len(image.shape)==2 or image.shape[-1]==1) else "optical"
        if tasks is None:
            tasks = ["hbb","obb","mask"] if modality=="optical" else ["hbb"]
        result = DetectionResult(detections=[], modality=modality, image_shape=image.shape[:2])
        if modality == "optical":
            result.detections = self._optical_detect(image, tasks)
        else:
            result.detections = self._sar_detect(image)
        result.inference_time_ms = (time.time()-t0)*1000
        return result

    def _optical_detect(self, img, tasks):
        dets = []
        h, w = img.shape[:2]
        # HBB + OBB both use DOTA-pretrained Rotated RTMDet
        obb_model = self._models['obb']

        if "hbb" in tasks or "obb" in tasks:
            from mmdet.apis import inference_detector
            # 需要 mmrotate transforms 注册
            try:
                r = inference_detector(obb_model, img)
            except KeyError:
                from mmrotate.utils import register_all_modules as rram
                rram()
                r = inference_detector(obb_model, img)
            pred = r.pred_instances; high = pred.scores > 0.3
            for j in range(high.sum().item()):
                idx = torch.where(high)[0][j]
                cid = int(pred.labels[idx].item())
                b = pred.bboxes[idx].tolist()  # OBB [cx,cy,w,h,theta]
                cx,cy,ww,hh,theta = b
                cn = DOTA_CLASSES[cid] if cid<len(DOTA_CLASSES) else f"cls_{cid}"
                if "hbb" in tasks:
                    dets.append(Detection(bbox_hbb=[cx-ww/2,cy-hh/2,cx+ww/2,cy+hh/2],
                        class_name=cn, score=pred.scores[idx].item(), modality="optical"))
                if "obb" in tasks:
                    dets.append(Detection(bbox_hbb=[cx-ww/2,cy-hh/2,cx+ww/2,cy+hh/2],
                        bbox_obb=b, class_name=cn, score=pred.scores[idx].item(), modality="optical"))
        # Mask
        if "mask" in tasks:
            from mmdet.apis import inference_detector
            r = inference_detector(self._models['mask'], img)
            pred = r.pred_instances; high = pred.scores > 0.3
            for j in range(high.sum().item()):
                idx = torch.where(high)[0][j]
                cid = int(pred.labels[idx].item())
                mask = pred.masks[idx].cpu().numpy().astype(bool) if hasattr(pred,'masks') and pred.masks is not None else None
                dets.append(Detection(bbox_hbb=pred.bboxes[idx].tolist(), mask=mask, class_name=COCO_CLASSES[cid] if cid<len(COCO_CLASSES) else f"cls_{cid}", score=pred.scores[idx].item()))
        return dets

    def _sar_detect(self, img):
        m = self._models['sar']; h,w = img.shape[:2]
        img_r = np.array(Image.fromarray(img.astype(np.uint8) if img.dtype!=np.uint8 else img).resize((800,800)))
        if len(img_r.shape)==2: img_r = np.stack([img_r]*3, axis=-1)
        mean=np.array([123.675,116.28,103.53],dtype=np.float32); std=np.array([58.395,57.12,57.375],dtype=np.float32)
        img_n = (img_r.astype(np.float32)-mean)/std
        img_t = torch.from_numpy(img_n).permute(2,0,1).unsqueeze(0).float().cuda()
        from mmdet.structures import DetDataSample
        with torch.no_grad():
            feats = m['neck'](m['backbone'](img_t)[0])
            ds = DetDataSample(); ds.set_metainfo({'img_shape':(800,800),'ori_shape':(h,w),
                'scale_factor': np.ones(8, dtype=np.float32)})
            results = m['head'].predict(feats, [ds], rescale=False)
        dets = []
        if results and results[0].bboxes is not None:
            bb=results[0].bboxes.cpu().numpy(); sc=results[0].scores.cpu().numpy(); lb=results[0].labels.cpu().numpy()
            # Scale bboxes from 800x800 → original image size
            sx, sy = w/800.0, h/800.0
            bb[:, 0] *= sx; bb[:, 1] *= sy
            bb[:, 2] *= sx; bb[:, 3] *= sy
            # Filter: valid boxes (not degenerate) + confidence threshold
            valid = (sc > 0.3) & (bb[:, 2] > bb[:, 0]) & (bb[:, 3] > bb[:, 1])
            bb, sc, lb = bb[valid], sc[valid], lb[valid]
            # NMS
            keep = self._nms(bb, sc, iou_threshold=0.5)
            for j in keep:
                cid = int(lb[j])
                dets.append(Detection(
                    bbox_hbb=bb[j].tolist(),
                    class_name=SAR_CLASSES[cid] if cid < len(SAR_CLASSES) else f"cls_{cid}",
                    score=float(sc[j]), modality="sar"))
        return dets

    @staticmethod
    def _nms(bboxes, scores, iou_threshold=0.5):
        """Simple NMS for HBB [x1,y1,x2,y2]"""
        if len(bboxes) == 0: return []
        x1, y1 = bboxes[:, 0], bboxes[:, 1]
        x2, y2 = bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            if order.size == 1: break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            ww = np.maximum(0, xx2 - xx1)
            hh = np.maximum(0, yy2 - yy1)
            inter = ww * hh
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou < iou_threshold]
        return keep

if __name__ == "__main__":
    engine = UnifiedRSDetectionEngine()
    engine.load_models()
    print("="*50)
    print("Test: DIOR optical")
    dior = "/home/ubuntu/dataset/DIOR/JPEGImages/JPEGImages-trainval"
    for f in sorted(os.listdir(dior))[:2]:
        r = engine.detect(np.array(Image.open(os.path.join(dior,f)).convert('RGB')), "optical", ["hbb"])
        print(f"  {f}: {len(r.detections)} HBB | {r.inference_time_ms:.0f}ms")
    print("Test: SSDD SAR")
    ssdd = "/home/ubuntu/.cache/kagglehub/datasets/bitsandlayers/sar-ship-detection-dataset/versions/1/SSDD/images/test"
    for f in sorted(os.listdir(ssdd))[:2]:
        r = engine.detect(np.array(Image.open(os.path.join(ssdd,f)).convert('L')), "sar")
        print(f"  {f}: {len(r.detections)} ships | {r.inference_time_ms:.0f}ms")
    print("Engine ready!")
