"""CPU-only NCNN inference without PyTorch or Ultralytics.

Run: python infer.py --input test/ --weights weights/ --out outputs/ [--runtime]
Uses the single NCNN export folder in weights/ (model.ncnn.param, model.ncnn.bin, metadata.yaml)
and the optional weights/inference_config.json (imgsz, conf, roi, max_det).
Pre- and post-processing reproduce Ultralytics 8.4.215 predict(retina_masks=True): letterbox,
class-aware NMS at IoU 0.5, and mask logits interpolated from the prototypes at original
resolution inside each box, so results match infer_ultralytics.py byte for byte.
Automatic ROI is a conservative light-paper heuristic; use --roi full for other backgrounds.
Predicted masks keep their largest visible component with holes filled, so each polygon matches
its seg_mask instance. Changes are reported on stderr. No downloads happen during inference.
Outputs: seg_mask/<stem>.png, vis/<stem>.jpg, predictions.json, and mqtt/<stem>.json, the
API_DOCS.md 2.1 scoreboard payload (image, latency_ms, counts, defect_ratio) that client.py publishes.
Each run starts from a clean --out: the previous run's seg_mask/, vis/, mqtt/ and predictions.json
are removed first (other files there are left alone); --keep keeps them.
"""
from pathlib import Path
import argparse
import json
import math
import shutil
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageOps

CLASSES = ['black', 'broken', 'good', 'husk', 'immature', 'parchment', 'shell']
COUNT_KEYS = ['good', 'shell', 'black', 'husk', 'broken', 'parchment', 'immature']  # MQTT counts order (API_DOCS.md 2.1)
PAD_VALUE = 114     # Ultralytics letterbox grey
NMS_IOU = 0.5       # Ultralytics 8.4.215 default predict IoU (infer_ultralytics.py does not set iou)
MAX_WH = 7680       # class offset for class-aware NMS
MAX_NMS = 30000     # boxes kept before NMS
OUTPUT_ITEMS = ('seg_mask', 'vis', 'mqtt', 'predictions.json')  # what a run writes, and clean_outputs removes


def warn(message):
    print('POSTPROCESS:', message, file=sys.stderr)


def read_rgb(path):
    with Image.open(path) as image:
        return np.array(ImageOps.exif_transpose(image).convert('RGB'))


def roi_box(rgb, mode='auto'):
    h, w = rgb.shape[:2]
    full = (0, 0, w, h)
    if mode == 'full':
        return full
    scale = min(1., 1024/max(h, w))
    small = cv2.resize(rgb, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    if np.median(hsv[..., 1]) > 35 or np.median(hsv[..., 2]) < 140:
        return full
    fg = (((hsv[..., 1] > 38) & (hsv[..., 2] < 247)) | (hsv[..., 2] < 80)).astype('uint8')
    n, components, stats, _ = cv2.connectedComponentsWithStats(fg)
    selected = np.zeros_like(fg)
    for k, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
        if area < 12:
            continue
        touches = x == 0 or y == 0 or x+bw == fg.shape[1] or y+bh == fg.shape[0]
        if touches and max(bw/bh, bh/bw) > 8:
            continue  # narrow dark page edge; never a central-position crop
        selected[components == k] = 1
    ys, xs = np.where(selected)
    if not len(xs):
        return full
    pad = max(24, .15*max(xs.max()-xs.min()+1, ys.max()-ys.min()+1)/scale)
    box = (max(0, int(xs.min()/scale-pad)), max(0, int(ys.min()/scale-pad)),
           min(w, int((xs.max()+1)/scale+pad)), min(h, int((ys.max()+1)/scale+pad)))
    return full if (box[2]-box[0])*(box[3]-box[1]) > .85*w*h else box


def polygon_mask(raw, occupied, tag='', min_area=16):
    visible = np.asarray(raw, dtype=bool) & ~occupied
    n, labels, stats, _ = cv2.connectedComponentsWithStats(visible.astype('uint8'), connectivity=8)
    if n <= 1:
        warn(f'{tag}: no visible pixels; discarded')
        return None
    k = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    chosen = (labels == k).astype('uint8')
    contours, _ = cv2.findContours(chosen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(chosen)
    cv2.fillPoly(filled, [contour], 1)
    if np.any(filled.astype(bool) & occupied):
        warn(f'{tag}: polygon would overwrite another instance; discarded')
        return None
    if filled.sum() < min_area or len(contour) < 3 or cv2.contourArea(contour) < 1:
        warn(f'{tag}: too small for a polygon; discarded')
        return None
    dropped = int(visible.sum()-chosen.sum())
    added = int(filled.sum()-chosen.sum())
    if dropped or added:
        warn(f'{tag}: removed {dropped} disconnected pixels; filled {added} hole pixels')
    return filled.astype(bool), contour.reshape(-1, 2)


def output_instances(detections, shape, offset=(0, 0)):
    """Paint instances by score; each mask covers only its box: detections are (score, class, bx, by, mask)."""
    result = np.zeros(shape, dtype=np.uint16)
    objects = []
    ox, oy = offset
    scores = np.array([d[0] for d in detections], dtype=np.float32)
    for index in np.argsort(-scores, kind='stable'):
        _, cls, bx, by, m = detections[index]
        x, y = ox+bx, oy+by
        if y+m.shape[0] > shape[0] or x+m.shape[1] > shape[1]:
            raise ValueError('Prediction mask extends beyond original image')
        view = result[y:y+m.shape[0], x:x+m.shape[1]]
        fixed = polygon_mask(m, view != 0, tag=f'prediction {index}')
        if fixed is None:
            continue
        binary, contour = fixed
        instance_id = len(objects)+1
        if instance_id > 65535:
            raise ValueError('Too many instances for uint16')
        view[binary] = instance_id
        objects.append({'id': instance_id, 'class': CLASSES[cls],
                        'polygon': (contour+np.array([x, y])).astype(int).tolist()})
    return result, objects


def local_ncnn(folder):
    """--weights is either the folder that holds one NCNN export subfolder (weights/) or that export
    folder itself (weights/best_ncnn_model/). Returns its .param file."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError('--weights must be a local directory')
    is_export = lambda p: any(not q.name.endswith('.pnnx.param') for q in p.glob('*.param')) and list(p.glob('*.bin'))
    exports = [folder] if is_export(folder) else sorted(p for p in folder.iterdir() if p.is_dir() and is_export(p))
    if len(exports) > 1:
        raise ValueError(f'More than one NCNN export in {folder} ({", ".join(p.name for p in exports)}); '
                         'keep one, or pass the export folder itself as --weights')
    if not exports:
        raise FileNotFoundError(f'No NCNN model (.param + .bin) in {folder} or its subfolders')
    params = [p for p in sorted(exports[0].glob('*.param')) if not p.name.endswith('.pnnx.param')]
    if len(params) != 1:
        raise ValueError(f'Expected one .param file in {exports[0]}')
    return params[0]


def find_config(weights, model_file):
    """inference_config.json from the --weights folder, else next to the model or one level up
    (--weights weights/best_ncnn_model/ still finds weights/inference_config.json). Returns (dict, path or None)."""
    model_file = Path(model_file).resolve()
    for folder in (Path(weights).resolve(), model_file.parent, model_file.parent.parent):
        path = folder/'inference_config.json'
        if path.is_file():
            return json.loads(path.read_text()), path
    return {}, None


def read_metadata(path):
    """Read the flat metadata.yaml that Ultralytics writes (no PyYAML needed): top-level scalars,
    plus indented `key: value` maps (names) and `- item` lists (imgsz)."""
    meta, parent = {}, None
    for line in path.read_text(encoding='utf-8').splitlines():
        text = line.strip()
        if not text or text.startswith('#'):
            continue
        if line[0] in ' \t' or text.startswith('- '):
            if parent is None:
                continue
            if text.startswith('- '):
                meta[parent] = (meta[parent] or []) + [text[2:].strip().strip('\'"')]
            else:
                key, _, value = text.partition(':')
                meta[parent] = meta[parent] or {}
                if isinstance(meta[parent], dict):
                    key = key.strip().strip('\'"')
                    meta[parent][int(key) if key.isdigit() else key] = value.strip().strip('\'"')
            continue
        key, _, value = text.partition(':')
        value = value.strip().strip('\'"')
        meta[key.strip()] = value or None
        parent = key.strip() if not value else None
    return meta


class NcnnSegmenter:
    def __init__(self, param):
        import ncnn
        self.ncnn = ncnn
        meta_path = param.parent/'metadata.yaml'
        if not meta_path.is_file():
            raise FileNotFoundError(f'Missing metadata.yaml next to {param.name}')
        meta = read_metadata(meta_path)
        if meta.get('task', 'segment') != 'segment':
            raise ValueError('NCNN model is not a segmentation model')
        names = meta.get('names') or {}
        names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
        if names != CLASSES:
            raise ValueError('Model class names do not match the seven competition classes')
        self.stride = int(meta.get('stride', 32))
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False  # CPU only; other options are NCNN defaults, as in Ultralytics
        bin_path = param.with_suffix('.bin')
        if self.net.load_param(str(param)) != 0 or self.net.load_model(str(bin_path)) != 0:
            raise ValueError(f'NCNN could not load {param.name} / {bin_path.name}')
        self.input_name = self.net.input_names()[0]
        self.output_names = sorted(self.net.output_names())

    def forward(self, blob):
        """Return detections (4+nc+nm, anchors) and prototypes (nm, mh, mw)."""
        outputs = []
        with self.net.create_extractor() as ex:
            ex.input(self.input_name, self.ncnn.Mat(blob).clone())
            for name in self.output_names:
                status, mat = ex.extract(name)
                if status != 0:
                    raise RuntimeError(f'NCNN could not compute {name}')
                outputs.append(np.array(mat))
        by_rank = {o.ndim: o for o in outputs}
        if set(by_rank) != {2, 3}:
            raise ValueError(f'Unexpected NCNN outputs {[o.shape for o in outputs]}')
        return by_rank[2], by_rank[3]

    def predict(self, rgb, imgsz, conf, max_det):
        """Detections sorted by score: (score, class, x, y, box-local bool mask), like Ultralytics results."""
        h0, w0 = rgb.shape[:2]
        # Letterbox (Ultralytics LetterBox, auto=False for NCNN)
        r = min(imgsz/h0, imgsz/w0)
        new_w, new_h = max(round(w0*r), 1), max(round(h0*r), 1)
        image = rgb if (new_w, new_h) == (w0, h0) else cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        dw, dh = (imgsz-new_w)/2, (imgsz-new_h)/2
        top, left = round(dh-.1), round(dw-.1)
        image = cv2.copyMakeBorder(image, top, round(dh+.1), left, round(dw+.1),
                                   cv2.BORDER_CONSTANT, value=(PAD_VALUE,)*3)
        blob = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)/255

        raw, protos = self.forward(blob)
        nc, nm = len(CLASSES), protos.shape[0]
        if raw.shape[0] != 4+nc+nm:
            raise ValueError(f'Unexpected detection output shape {raw.shape}')
        pred = raw.T
        scores = pred[:, 4:4+nc]
        cls = scores.argmax(1)
        score = scores[np.arange(len(pred)), cls]
        keep = score > conf
        pred, cls, score = pred[keep], cls[keep], score[keep]
        if not len(pred):
            return []
        if len(pred) > MAX_NMS:
            top_k = np.argsort(-score, kind='stable')[:MAX_NMS]
            pred, cls, score = pred[top_k], cls[top_k], score[top_k]
        boxes = np.concatenate((pred[:, :2]-pred[:, 2:4]/2, pred[:, :2]+pred[:, 2:4]/2), 1)
        order = nms(boxes+cls[:, None].astype(np.float32)*MAX_WH, score, NMS_IOU)[:max_det]
        boxes, cls, score, coef = boxes[order], cls[order], score[order], pred[order, 4+nc:]

        # Boxes back to the original image (ops.scale_boxes + clip_boxes)
        boxes = (boxes-np.array([left, top, left, top], np.float32))/np.array(
            [new_w/w0, new_h/h0, new_w/w0, new_h/h0], np.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

        # Masks at original resolution, kept inside each box. Ultralytics picks one of two routes:
        # a dense stack (ops.process_mask_native) when it is small, else box-local interpolation.
        _, mh, mw = protos.shape
        logits = (coef@protos.reshape(nm, -1)).reshape(-1, mh, mw)
        corners = np.clip(np.ceil(boxes).astype(np.int64), 0, [w0, h0, w0, h0])
        dense = len(boxes)*h0*w0 <= 1 << 22
        if dense:
            gain = min(4*mh/h0, 4*mw/w0)
            pad = (4*mh-max(round(h0*gain), 1)) | (4*mw-max(round(w0*gain), 1))
            f = 4//math.gcd(4, pad | pad//2)  # prototype upsample that crops the padding on whole pixels
            gain = min(f*mh/h0, f*mw/w0)
            ch, cw = max(round(h0*gain), 1), max(round(w0*gain), 1)
            ptop, pleft = round((f*mh-ch)/2-.1), round((f*mw-cw)/2-.1)
        else:
            gain = min(4*mh/h0, 4*mw/w0)
            ch, cw = max(round(h0*gain), 1), max(round(w0*gain), 1)
            ptop, pleft = round((4*mh-ch)/2-.1), round((4*mw-cw)/2-.1)
            fx = ((np.arange(w0, dtype=np.float32)+.5)*(cw/w0)+pleft).clip(pleft+2, pleft+cw-2)/4-.5
            fy = ((np.arange(h0, dtype=np.float32)+.5)*(ch/h0)+ptop).clip(ptop+2, ptop+ch-2)/4-.5
            ix, iy = fx.astype(np.int64), fy.astype(np.int64)
            fx, fy = fx-ix, fy-iy
            ix1, iy1 = np.minimum(ix+1, mw-1), np.minimum(iy+1, mh-1)

        detections = []
        for k, (x1, y1, x2, y2) in enumerate(corners):
            if x2 <= x1 or y2 <= y1:
                continue
            m = logits[k]
            if dense:  # bilinear (align_corners=False) x f, crop the letterbox span, bilinear to the image
                if f > 1:
                    m = cv2.resize(m, (mw*f, mh*f), interpolation=cv2.INTER_LINEAR)
                if m.shape != (h0, w0):
                    m = cv2.resize(m[ptop:ptop+ch, pleft:pleft+cw], (w0, h0), interpolation=cv2.INTER_LINEAR)
                m = m[y1:y2, x1:x2] > 0
            else:  # sample only the box pixels: columns first, then rows
                cols, rows = slice(x1, x2), slice(y1, y2)
                m = m[:, ix[cols]]+(m[:, ix1[cols]]-m[:, ix[cols]])*fx[cols]
                m = m[iy[rows]]+(m[iy1[rows]]-m[iy[rows]])*fy[rows][:, None] > 0
            if m.any():  # Ultralytics drops predictions whose mask is empty
                detections.append((float(score[k]), int(cls[k]), int(x1), int(y1), m))
        return detections


def nms(boxes, scores, iou_threshold):
    """torchvision.ops.nms: keep boxes in score order, suppress IoU > threshold."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2-x1)*(y2-y1)
    order = np.argsort(-scores, kind='stable')
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        w = (np.minimum(x2[i], x2[rest])-np.maximum(x1[i], x1[rest])).clip(0)
        h = (np.minimum(y2[i], y2[rest])-np.maximum(y1[i], y1[rest])).clip(0)
        inter = w*h
        union = areas[i]+areas[rest]-inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def segment(model, rgb, imgsz, conf, roi, max_det):
    """ROI crop, model and post-processing on one RGB image: (uint16 instance mask, objects)."""
    x0, y0, x1, y1 = roi_box(rgb, roi)
    detections = model.predict(rgb[y0:y1, x0:x1], imgsz, conf, max_det)
    return output_instances(detections, rgb.shape[:2], (x0, y0))


def mqtt_summary(image_name, objects, latency_ms):
    """Scoreboard payload (API_DOCS.md 2.1): counts per class, defect_ratio = defects / all beans (0.0 if none)."""
    counts = dict.fromkeys(COUNT_KEYS, 0)
    for obj in objects:
        counts[obj['class']] += 1
    total = len(objects)
    defect_ratio = round((total-counts['good'])/total, 4) if total else 0.0
    return {'image': image_name, 'latency_ms': int(latency_ms), 'counts': counts, 'defect_ratio': defect_ratio}


def check_destination(source, destination):
    """Refuse an --out that is, contains, or would clean away the --input folder."""
    if source == destination or source in destination.parents:
        raise ValueError('Keep outputs outside the input folder')
    for name in OUTPUT_ITEMS:
        if source == destination/name or destination/name in source.parents:
            raise ValueError(f'The input folder is inside {destination/name}, which each run clears')


def clean_outputs(destination):
    """Remove the previous run's outputs (OUTPUT_ITEMS only) so --out holds just this run."""
    for name in OUTPUT_ITEMS:
        path = Path(destination)/name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def save_mqtt(destination, stem, summary):
    """mqtt/<stem>.json: the one-line JSON that client.py publishes."""
    path = Path(destination)/'mqtt'/(stem+'.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary), encoding='utf-8')
    return path


def save_images(destination, stem, rgb, mask, objects, vis=True):
    """seg_mask/<stem>.png and, with vis, the colour overlay vis/<stem>.jpg."""
    for kind in ['seg_mask']+(['vis'] if vis else []):
        (Path(destination)/kind).mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(Path(destination)/'seg_mask'/(stem+'.png'))
    if vis:
        overlay = rgb.copy()
        for obj in objects:
            color = np.random.default_rng(obj['id']).integers(40, 240, 3)
            fg = mask == obj['id']
            overlay[fg] = (.6*rgb[fg]+.4*color).astype('uint8')
        Image.fromarray(overlay).save(Path(destination)/'vis'/(stem+'.jpg'))


def run(args):
    source, destination = Path(args.input).resolve(), Path(args.out).resolve()
    if not source.is_dir():
        raise FileNotFoundError(args.input)
    check_destination(source, destination)
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not files:
        raise ValueError('No supported input images')
    if len({p.stem.casefold() for p in files}) != len(files):
        raise ValueError('Duplicate input stems would overwrite output PNGs')
    param = local_ncnn(args.weights)
    config, _ = find_config(args.weights, param)
    imgsz = args.imgsz if args.imgsz is not None else int(config.get('imgsz', 640))
    conf = args.conf if args.conf is not None else float(config.get('conf', .25))
    if imgsz <= 0 or not 0 <= conf <= 1:
        raise ValueError('Invalid imgsz/conf')
    roi = args.roi if args.roi is not None else config.get('roi', 'auto')
    max_det = int(config.get('max_det', 500))
    if roi not in ('auto', 'full') or max_det < 1:
        raise ValueError('Invalid ROI/max_det config')
    started = time.perf_counter()
    model = NcnnSegmenter(param)
    if imgsz % model.stride:  # Ultralytics check_imgsz rounds up to the stride
        fixed = math.ceil(imgsz/model.stride)*model.stride
        print(f'WARNING: imgsz {imgsz} is not a multiple of {model.stride}; using {fixed}', file=sys.stderr)
        imgsz = fixed
    if args.runtime:
        print(f'Setup (import ncnn + load {param.parent.name}): {time.perf_counter()-started:.2f} s')
    if not args.keep:
        clean_outputs(destination)  # only after inputs and model checked out, so a failed run clears nothing
    process_images(model, files, destination, imgsz, conf, roi, max_det, args)


def process_images(model, files, destination, imgsz, conf, roi, max_det, args):
    """Per-image loop (shared with inferpt.py): writes all outputs and the --runtime / --latency reports."""
    records, times, latencies = [], [], []
    for index, path in enumerate(files, 1):
        started = time.perf_counter()
        rgb = read_rgb(path)
        read_done = time.perf_counter()
        mask, objects = segment(model, rgb, imgsz, conf, roi, max_det)
        inferred = time.perf_counter()
        summary = mqtt_summary(path.name, objects, (inferred-started)*1000)  # read + ROI + predict + postprocess
        save_mqtt(destination, path.stem, summary)
        save_images(destination, path.stem, rgb, mask, objects, args.vis)
        records.append({'image_name': path.name, 'objects': objects})
        finished = time.perf_counter()
        times.append(finished-started)  # read + ROI + predict + postprocess + save
        latencies.append(summary['latency_ms'])
        if args.runtime:
            print(f'[{index}/{len(files)}] {path.name}: {len(objects)} object(s), {times[-1]:.3f} s')
        if args.latency:
            print(f'[{index}/{len(files)}] {path.name}: read {(read_done-started)*1000:.1f} ms'
                  f' + inference {(inferred-read_done)*1000:.1f} ms = latency_ms {summary["latency_ms"]}'
                  f' | write outputs {(finished-inferred)*1000:.1f} ms | total {(finished-started)*1000:.1f} ms')
    (destination/'predictions.json').write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
    if args.runtime:
        rest = times[1:] or times
        print(f'Time per image: mean {sum(times)/len(times):.3f} s, max {max(times):.3f} s, '
              f'total {sum(times):.2f} s for {len(times)} image(s); '
              f'mean without the first (warm-up) image {sum(rest)/len(rest):.3f} s')
    if args.latency:
        rest = latencies[1:] or latencies
        print(f'latency_ms (mqtt/<stem>.json) over {len(latencies)} image(s): mean {sum(latencies)/len(latencies):.1f}, '
              f'min {min(latencies)}, max {max(latencies)}; '
              f'mean without the first (warm-up) image {sum(rest)/len(rest):.1f}')
    print(f'Saved {len(records)} images to {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['input', 'weights', 'out']:
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--roi', choices=['auto', 'full'], default=None)
    parser.add_argument('--backend', choices=['auto', 'ncnn'], default='auto', help="NCNN only; kept for compatibility with infer_ultralytics.py")
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
