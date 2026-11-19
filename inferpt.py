"""PyTorch (.pt) counterpart of infer.py: same CLI, outputs and pipeline, with best.pt as the model.

Run: python inferpt.py --input test/ --weights weights/ --out outputs_pt/ [--runtime] [--device cpu]
--weights is a folder holding best.pt (or exactly one .pt) or a path to a .pt file; the optional
inference_config.json next to it (imgsz, conf, roi, max_det) is read as in infer.py.
Everything except the network forward comes from infer.py: ROI, letterbox (fixed imgsz x imgsz),
class-aware NMS at IoU 0.5, retina masks inside each box, polygon clean-up, seg_mask/vis/predictions.json/mqtt.
On the same images it reproduces infer.py's NCNN results up to float rounding.
Needs torch and ultralytics==8.4.215 (to unpickle the checkpoint); infer.py does not.
"""
from pathlib import Path
import argparse
import json
import math
import sys
import time

import infer
from infer import CLASSES


def local_pt(weights):
    path = Path(weights).resolve()
    if path.is_file():
        if path.suffix.lower() != '.pt':
            raise ValueError('--weights must be a .pt file or a folder containing best.pt')
        return path
    if not path.is_dir():
        raise FileNotFoundError('--weights must be a local .pt file or directory')
    if (path/'best.pt').is_file():
        return path/'best.pt'
    found = sorted(path.glob('*.pt'))
    if len(found) != 1:
        raise FileNotFoundError('Expected best.pt or exactly one .pt file in weights/')
    return found[0]


class PtSegmenter:
    """Same interface as infer.NcnnSegmenter; only forward() runs PyTorch instead of NCNN."""
    predict = infer.NcnnSegmenter.predict  # letterbox, NMS and mask decoding shared with infer.py

    def __init__(self, pt, device='cpu'):
        import torch
        from ultralytics.nn.tasks import load_checkpoint
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError(f'--device {device} requested but CUDA is not available')
        model, _ = load_checkpoint(str(pt), device=self.device)  # EMA weights in FP32, as Ultralytics predict/export
        if getattr(model, 'task', 'segment') != 'segment':
            raise ValueError('.pt model is not a segmentation model')
        names = model.names
        names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
        if names != CLASSES:
            raise ValueError('Model class names do not match the seven competition classes')
        self.stride = int(model.stride.max())
        self.model = model.fuse(verbose=False).eval() if hasattr(model, 'fuse') else model.eval()

    def forward(self, blob):
        """Return detections (4+nc+nm, anchors) and prototypes (nm, mh, mw), like the NCNN outputs."""
        with self.torch.inference_mode():
            out = self.model(self.torch.from_numpy(blob)[None].to(self.device))
        preds = out[0] if isinstance(out, (list, tuple)) and isinstance(out[0], (list, tuple)) else out
        det, protos = preds[0], preds[1]
        if det.ndim != 3 or protos.ndim != 4:
            raise ValueError(f'Unexpected model outputs {tuple(det.shape)}, {tuple(protos.shape)}')
        return det[0].float().cpu().numpy(), protos[0].float().cpu().numpy()


def run(args):
    source, destination = Path(args.input).resolve(), Path(args.out).resolve()
    if not source.is_dir():
        raise FileNotFoundError(args.input)
    infer.check_destination(source, destination)
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not files:
        raise ValueError('No supported input images')
    if len({p.stem.casefold() for p in files}) != len(files):
        raise ValueError('Duplicate input stems would overwrite output PNGs')
    pt = local_pt(args.weights)
    config_path = pt.parent/'inference_config.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    imgsz = args.imgsz if args.imgsz is not None else int(config.get('imgsz', 640))
    conf = args.conf if args.conf is not None else float(config.get('conf', .25))
    if imgsz <= 0 or not 0 <= conf <= 1:
        raise ValueError('Invalid imgsz/conf')
    roi = args.roi if args.roi is not None else config.get('roi', 'auto')
    max_det = int(config.get('max_det', 500))
    if roi not in ('auto', 'full') or max_det < 1:
        raise ValueError('Invalid ROI/max_det config')
    started = time.perf_counter()
    model = PtSegmenter(pt, args.device)
    if imgsz % model.stride:  # Ultralytics check_imgsz rounds up to the stride
        fixed = math.ceil(imgsz/model.stride)*model.stride
        print(f'WARNING: imgsz {imgsz} is not a multiple of {model.stride}; using {fixed}', file=sys.stderr)
        imgsz = fixed
    if args.runtime:
        print(f'Setup (import torch + load {pt.name} on {model.device}): {time.perf_counter()-started:.2f} s')
    if not args.keep:
        infer.clean_outputs(destination)  # only after inputs and model checked out, so a failed run clears nothing
    infer.process_images(model, files, destination, imgsz, conf, roi, max_det, args)  # same loop and reports as infer.py


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['input', 'weights', 'out']:
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--roi', choices=['auto', 'full'], default=None)
    parser.add_argument('--backend', choices=['auto', 'pt'], default='auto', help='PyTorch only; kept for CLI compatibility with infer.py')
    parser.add_argument('--device', default='cpu', help="torch device, e.g. cpu (default), cuda, cuda:0")
    parser.add_argument('--imgsz', type=int, default=None)
    parser.add_argument('--conf', type=float, default=None)
    parser.add_argument('--vis', action='store_true', default=True)
    parser.add_argument('--no-vis', dest='vis', action='store_false')
    parser.add_argument('--runtime', action='store_true', help='print processing time per image')
    parser.add_argument('--latency', action='store_true',
                        help='print per image: read + inference = the latency_ms in mqtt/<stem>.json, then file writes and total')
    parser.add_argument('--keep', action='store_true',
                        help="keep the previous run's seg_mask/, vis/, mqtt/ in --out (default: clear them first)")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
