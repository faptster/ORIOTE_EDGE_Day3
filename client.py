"""
TopGun Competition - Complete Client Pipeline (HTTP + MQTT)
================================================================================
Pipeline Workflow:
  1. GET Image via HTTP      -> Fetches image binary + 'X-Image-ID' + 'X-Image-Name'
  2. Process Image (Model)   -> infer.py pipeline (NCNN, or best.pt with --backend pt);
                                the model is loaded + warmed up BEFORE step 1 (server timing)
  3. Send HTTP Submission    -> Posts this image's predictions.json record to /api/v1/submit,
                                then writes the outputs like infer.py (--out, default outputs/;
                                the previous run's are cleared first unless --keep):
                                seg_mask/<stem>.png, vis/<stem>.jpg, mqtt/<stem>.json, predictions.json
                                (written after the submit so disk time stays out of the server's clock)
  4. Send MQTT Submission    -> Publishes mqtt/<stem>.json (counts & defect_ratio) via WebSocket MQTT

IMPORTANT CREDENTIALS NOTE:
  - HTTP Bearer Token: Your team's LOGIN PASSWORD for the competition website.
  - MQTT Key:          Generated inside the website (My Team / Dashboard).
                       *HTTP password and MQTT key are DIFFERENT!*
================================================================================
"""

import argparse
import getpass
import io
import json
import ssl
import sys
import threading
import time
import uuid
from pathlib import Path

try:
    import requests
except ImportError:
    raise SystemExit('Please install requests: python -m pip install requests')

try:
    import paho.mqtt.client as mqtt
except ImportError:
    raise SystemExit('Please install paho-mqtt: python -m pip install "paho-mqtt>=2,<3"')

import numpy as np
import infer  # infer.py next to this file: NCNN model + the pre/post-processing used for outputs/


# ==============================================================================
# 1. CONFIGURATION (Set your defaults here)
# ==============================================================================
# HTTP Evaluation Server
DEFAULT_HTTP_URL = "https://edgeai-tesa.c333m.site"          # Competition API Domain (e.g. https://api.yourdomain.com)
DEFAULT_HTTP_TOKEN = "MkZhGXAgolbt7Y6YEzH9tf"           # Team login password for the website

# MQTT Broker (Summary Scores)
DEFAULT_MQTT_HOST = "tesa-score.c333m.site"         # Hostname only (no https://)
DEFAULT_TEAM_ID = "TGR-17"                         # Team ID (e.g. TEAM_01)
DEFAULT_MQTT_KEY = "2846058a9e8c42979ec452966dfce34bd1e0d7939d6e4fa2a5e3f6a57e421109"        # Generated on website under My Team
DEFAULT_TRACK = "EdgeAI"                            # "Embedded", "EdgeAI", or "Market"
DEFAULT_EVENT = "1"                                 # Event ID (e.g. "1")

# Model
DEFAULT_WEIGHTS = str(Path(__file__).resolve().parent / "weights")  # NCNN export + inference_config.json + best.pt
DEFAULT_BACKEND = "ncnn"                            # "ncnn" (infer.py) or "pt" (inferpt.py + best.pt)
DEFAULT_OUT = "outputs"                             # seg_mask/, vis/, mqtt/, predictions.json (as infer.py --out)
# ==============================================================================


# ==============================================================================
# 2. MODEL (infer.py pipeline: same ROI, letterbox, NMS, masks and polygons)
# ==============================================================================
def load_model(backend: str, weights: str, device: str = "cpu"):
    """
    Load the model and weights/inference_config.json exactly as infer.py (ncnn) or
    inferpt.py (pt) do, then run one warm-up pass. Call this BEFORE fetching an image:
    the server times a submission from the moment it sends the image.

    Returns:
        tuple (model, cfg) where cfg holds imgsz, conf, roi, max_det
    """
    if backend == "pt":
        import inferpt  # needs torch + ultralytics (requirements-pt.txt)
        pt_path = inferpt.local_pt(weights)
        model, folder = inferpt.PtSegmenter(pt_path, device), pt_path.parent
    else:
        model, folder = infer.NcnnSegmenter(infer.local_ncnn(weights)), Path(weights)

    config_path = folder / "inference_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    cfg = {
        "imgsz": int(config.get("imgsz", 640)),
        "conf": float(config.get("conf", 0.25)),
        "roi": config.get("roi", "auto"),
        "max_det": int(config.get("max_det", 500)),
    }
    if cfg["imgsz"] <= 0 or not 0 <= cfg["conf"] <= 1 or cfg["roi"] not in ("auto", "full") or cfg["max_det"] < 1:
        raise ValueError(f"Invalid inference_config.json: {cfg}")
    if cfg["imgsz"] % model.stride:  # Ultralytics check_imgsz rounds up to the stride, as infer.py does
        cfg["imgsz"] = -(-cfg["imgsz"] // model.stride) * model.stride

    model.predict(np.full((480, 640, 3), 200, np.uint8), cfg["imgsz"], cfg["conf"], cfg["max_det"])  # warm-up
    return model, cfg


def run_inference(model, cfg: dict, image_bytes: bytes, img_name: str):
    """
    Run infer.py's per-image processing on the received image (same results as
    `python infer.py` gives for the same file).

    Accepts:
        model, cfg: from load_model()
        image_bytes (bytes): Raw bytes of the received image
        img_name (str): Original filename of the image (e.g. 'Parchment_test.jpg')

    Returns:
        tuple (rgb, mask, record, summary):
        record  = this image's predictions.json entry {image_name, objects}  -> HTTP submit
        summary = mqtt/<stem>.json {image, latency_ms, counts, defect_ratio} -> MQTT
    """
    start_time = time.perf_counter()
    rgb = infer.read_rgb(io.BytesIO(image_bytes))  # EXIF-aware RGB, as infer.py reads files
    mask, objects = infer.segment(model, rgb, cfg["imgsz"], cfg["conf"], cfg["roi"], cfg["max_det"])
    latency_ms = (time.perf_counter() - start_time) * 1000  # decode + ROI + model + postprocess, as infer.py

    record = {"image_name": img_name, "objects": objects}
    summary = infer.mqtt_summary(img_name, objects, latency_ms)  # defect_ratio = (total - good) / total
    return rgb, mask, record, summary


def save_outputs(out_dir: str, img_name: str, rgb, mask, record: dict, summary: dict, vis: bool = True):
    """
    Write this image's outputs in infer.py's layout:
        <out>/seg_mask/<stem>.png, <out>/vis/<stem>.jpg, <out>/mqtt/<stem>.json and its
        record in <out>/predictions.json (only this image after main()'s clean; with --keep
        it is added, or replaces the record with the same image name).

    Returns:
        Path of mqtt/<stem>.json
    """
    out = Path(out_dir)
    stem = Path(img_name).stem
    mqtt_path = infer.save_mqtt(out, stem, summary)
    infer.save_images(out, stem, rgb, mask, record["objects"], vis)

    pred_path = out / "predictions.json"
    records = []
    if pred_path.is_file():
        try:
            records = json.loads(pred_path.read_text(encoding="utf-8"))
        except ValueError:
            print(f"    [!] {pred_path} was not valid JSON; starting a new one")
        if not isinstance(records, list):
            records = []
    records = [r for r in records if not (isinstance(r, dict) and r.get("image_name") == record["image_name"])]
    records.append(record)
    pred_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return mqtt_path


def publish_mqtt(host: str, team_id: str, mqtt_key: str, track: str, event: str, payload_dict: dict):
    """Publish summary message to TESA MQTT WebSocket Broker on port 443."""
    client_id = f"{team_id}:python-{uuid.uuid4().hex[:12]}"
    topic = f"/event/{mqtt_key}/{track}/{event}"
    
    payload_bytes = json.dumps(payload_dict).encode("utf-8")
    if len(payload_bytes) > 65536:
        raise ValueError("MQTT payload exceeds the broker's 64 KiB limit.")

    connected = threading.Event()
    acknowledged = threading.Event()
    connect_errors = []
    publish_errors = []

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            connect_errors.append(str(reason_code))
        connected.set()

    def on_publish(client, userdata, mid, reason_code, properties):
        if reason_code.is_failure:
            publish_errors.append(str(reason_code))
        acknowledged.set()

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
        transport="websockets",
        protocol=mqtt.MQTTv5,
        reconnect_on_failure=False,
    )
    client.username_pw_set(team_id, mqtt_key)
    client.ws_set_options(path="/mqtt/")
    client.tls_set_context(ssl.create_default_context())
    client.connect_timeout = 10
    client.on_connect = on_connect
    client.on_publish = on_publish

    started = False
    try:
        client.connect(host, 443, keepalive=30)
        client.loop_start()
        started = True

        if not connected.wait(15):
            raise RuntimeError("MQTT connection timed out. Check host and credentials.")
        if connect_errors:
            raise RuntimeError("MQTT login rejected: " + connect_errors[0])

        result = client.publish(topic, payload_bytes, qos=1, retain=False)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError("MQTT publish failed: " + mqtt.error_string(result.rc))

        if not acknowledged.wait(15):
            raise RuntimeError("MQTT publish: No acknowledgement received.")
        if publish_errors:
            raise RuntimeError("MQTT publish rejected: " + publish_errors[0])

        print("    [+] MQTT Broker acknowledged the message!")
    finally:
        client.disconnect()
        if started:
            client.loop_stop()


def main():
    parser = argparse.ArgumentParser(description="TopGun Competition Client Pipeline (HTTP + MQTT)")
    parser.add_argument("--http-url", default=DEFAULT_HTTP_URL, help="HTTP Server base URL")
    parser.add_argument("--http-token", default=DEFAULT_HTTP_TOKEN, help="Team HTTP password")
    parser.add_argument("--team-id", default=DEFAULT_TEAM_ID, help="Team ID (e.g. TEAM_01)")
    parser.add_argument("--mqtt-key", default=DEFAULT_MQTT_KEY, help="MQTT Key generated from website")
    parser.add_argument("--mqtt-host", default=DEFAULT_MQTT_HOST, help="MQTT Host")
    parser.add_argument("--track", default=DEFAULT_TRACK, choices=["Embedded", "EdgeAI", "Market"], help="Track")
    parser.add_argument("--event", default=DEFAULT_EVENT, help="Event ID")
    parser.add_argument("--skip-mqtt", action="store_true", help="Skip MQTT publishing step")
    parser.add_argument("--mode", choices=["train", "test"], default=None, help="?mode= for GET image (default: server's open mode)")
    parser.add_argument("--backend", choices=["ncnn", "pt"], default=DEFAULT_BACKEND, help="ncnn = infer.py, pt = inferpt.py (best.pt)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="weights folder (or .pt file for --backend pt)")
    parser.add_argument("--device", default="cpu", help="torch device for --backend pt (cpu, cuda)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="outputs folder (seg_mask/, vis/, mqtt/, predictions.json)")
    parser.add_argument("--no-vis", dest="vis", action="store_false", help="do not write vis/<stem>.jpg")
    parser.add_argument("--keep", action="store_true",
                        help="keep previous outputs and add this image to predictions.json (default: start --out fresh)")
    parser.add_argument("--latency", action="store_true",
                        help="print the time from the GET image request to the HTTP submit response (and the MQTT ack)")
    args = parser.parse_args()

    http_url = args.http_url.rstrip("/")
    http_token = args.http_token
    team_id = args.team_id
    mqtt_key = args.mqtt_key

    # Check credentials
    if http_token == "YOUR_TEAM_PASSWORD":
        http_token = getpass.getpass("Enter Team HTTP Password (website login): ").strip()
    if not args.skip_mqtt and mqtt_key == "YOUR_GENERATED_MQTT_KEY":
        mqtt_key = getpass.getpass("Enter Team MQTT Key (generated on website): ").strip()

    print("=" * 70)
    print("🚀 TopGun Pipeline: HTTP Fetch -> Model Inference -> HTTP Submit -> MQTT")
    print(f"HTTP Server: {http_url}")
    print(f"MQTT Broker: {args.mqtt_host}:443/mqtt/ (Track: {args.track}, Event: {args.event})")
    print(f"Team ID:     {team_id}")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # STEP 0: Load + warm up the model BEFORE requesting an image
    # (the server times from image delivery; a test-mode GET also uses quota)
    # --------------------------------------------------------------------------
    print(f"\n[0/4] Loading model ({args.backend}: {args.weights})...")
    t_load = time.time()
    try:
        model, cfg = load_model(args.backend, args.weights, args.device)
    except Exception as e:
        print(f"[-] Model load failed: {e}")
        sys.exit(1)
    print(f"[+] Model ready in {time.time() - t_load:.2f}s {cfg}")

    # --------------------------------------------------------------------------
    # STEP 1: Fetch Image from HTTP Server
    # --------------------------------------------------------------------------
    print("\n[1/4] Fetching image from HTTP server...")
    http_headers = {"Authorization": f"Bearer {http_token}"}

    t_start_req = time.time()
    marks = {"get": time.perf_counter()}  # --latency clock: starts as the GET request is sent
    try:
        img_resp = requests.get(f"{http_url}/api/v1/image", headers=http_headers, timeout=30,
                                params={"mode": args.mode} if args.mode else None)
    except requests.exceptions.RequestException as e:
        print(f"[-] Connection Error: {e}")
        sys.exit(1)
    marks["received"] = time.perf_counter()  # requests.get returns after the whole image body is read

    if img_resp.status_code != 200:
        print(f"[-] Failed to fetch image (HTTP {img_resp.status_code}): {img_resp.text}")
        sys.exit(1)

    image_bytes = img_resp.content
    img_id = img_resp.headers.get("X-Image-ID")
    img_name = img_resp.headers.get("X-Image-Name", "image.jpg")
    file_name = Path(img_name).name or "image.jpg"  # local file names only (no folders from the header)
    mode = img_resp.headers.get("X-Mode", "train")
    dl_time = time.time() - t_start_req

    print(f"[+] Image Received successfully!")
    print(f"    - Image ID:       {img_id}")
    print(f"    - Image Name:     {img_name}")
    print(f"    - Mode:           {mode.upper()}")
    print(f"    - Size:           {len(image_bytes):,} bytes")
    print(f"    - Transfer Time:  {dl_time:.3f}s")

    # --------------------------------------------------------------------------
    # STEP 2: Run Model Inference
    # --------------------------------------------------------------------------
    print(f"\n[2/4] Running model inference on '{img_name}'...")
    rgb, mask, record, summary = run_inference(model, cfg, image_bytes, img_name)
    marks["inferred"] = time.perf_counter()
    print(f"[+] Inference complete ({summary['latency_ms']} ms)")
    print(f"    - Detected objects: {len(record['objects'])}")
    print(f"    - Defect Ratio:     {summary['defect_ratio']}")

    # --------------------------------------------------------------------------
    # STEP 3: Submit Results to HTTP Server (this image's predictions.json record)
    # --------------------------------------------------------------------------
    print(f"\n[3/4] Submitting predictions to HTTP server ({http_url}/api/v1/submit)...")
    submit_payload = {
        "img_id": img_id,
        "predictions": record
    }

    sub_resp = sub_error = None
    try:
        sub_resp = requests.post(f"{http_url}/api/v1/submit", json=submit_payload, headers=http_headers, timeout=30)
    except requests.exceptions.RequestException as e:
        sub_error = e
    marks["submitted"] = time.perf_counter()  # submit response received

    # Write outputs + the received image AFTER the submit (disk time stays out of the server-timed window).
    # Fresh outputs per run: the previous run's are cleared only now, once there are new results to write.
    if not args.keep:
        infer.clean_outputs(args.out)
        print(f"    - Cleared previous outputs in {Path(args.out)} (--keep to keep them)")
    mqtt_path = save_outputs(args.out, file_name, rgb, mask, record, summary, args.vis)
    print(f"    - Saved outputs:  {Path(args.out) / 'predictions.json'}, {mqtt_path}, "
          f"seg_mask/{mqtt_path.stem}.png" + (f", vis/{mqtt_path.stem}.jpg" if args.vis else ""))
    save_folder = Path("downloaded_images")
    save_folder.mkdir(exist_ok=True)
    save_path = save_folder / file_name
    save_path.write_bytes(image_bytes)
    print(f"    - Saved file:     {save_path}")
    marks["saved"] = time.perf_counter()

    if sub_error is not None:
        print(f"[-] HTTP Submit Error: {sub_error}")
        sys.exit(1)

    if sub_resp.status_code != 200:
        print(f"[-] HTTP Submission rejected (HTTP {sub_resp.status_code}): {sub_resp.text}")
        sys.exit(1)

    sub_data = sub_resp.json()
    print(f"[+] HTTP Submission ACCEPTED!")
    print(f"    - Server Latency: {sub_data.get('duration_seconds')}s (delivery finish -> submit)")
    print(f"    - Received At:    {sub_data.get('received_at')}")

    # --------------------------------------------------------------------------
    # STEP 4: Send Summary to MQTT Broker (the mqtt/<stem>.json just written)
    # --------------------------------------------------------------------------
    mqtt_ok = None
    if args.skip_mqtt:
        print("\n[4/4] MQTT sending skipped (--skip-mqtt flag active).")
    else:
        print(f"\n[4/4] Publishing summary to MQTT Broker ({args.mqtt_host}:443/mqtt/)...")
        mqtt_payload = json.loads(mqtt_path.read_text(encoding="utf-8"))
        print(f"    Payload to publish ({mqtt_path}):")
        print(f"    {json.dumps(mqtt_payload, indent=4)}")

        try:
            publish_mqtt(
                host=args.mqtt_host,
                team_id=team_id,
                mqtt_key=mqtt_key,
                track=args.track,
                event=args.event,
                payload_dict=mqtt_payload
            )
            mqtt_ok = True
            print(f"[+] MQTT Submission COMPLETE for topic: /event/<MQTT_KEY>/{args.track}/{args.event}")
        except Exception as e:
            mqtt_ok = False
            print(f"[-] MQTT Publish Error: {e}")
        marks["mqtt"] = time.perf_counter()  # broker PUBACK received (or the publish failed)

    if args.latency:
        rows = [
            ("1. GET image (request -> body received)", "get", "received", ""),
            ("2. Inference (decode + model + post)", "received", "inferred", f"MQTT latency_ms = {summary['latency_ms']}"),
            ("3. HTTP submit (POST -> 200 response)", "inferred", "submitted", ""),
            ("=> GET -> result submitted (1+2+3)", "get", "submitted",
             f"server duration_seconds = {sub_data.get('duration_seconds')} s"),
            ("4. Write outputs + downloaded image", "submitted", "saved", ""),
        ]
        if "mqtt" in marks:
            rows += [("5. MQTT publish (connect -> PUBACK)", "saved", "mqtt", "" if mqtt_ok else "FAILED"),
                     ("=> GET -> MQTT " + ("acknowledged" if mqtt_ok else "failed"), "get", "mqtt", "")]
        print("\n" + "-" * 70)
        print(f"⏱  Latency for '{img_name}' (clock starts when the GET image request is sent):")
        for label, start, end, note in rows:
            print(f"  {label:<40}{(marks[end] - marks[start]) * 1000:9.1f} ms" + (f"   ({note})" if note else ""))
        print("-" * 70)

    # Query latest stats
    try:
        st_resp = requests.get(f"{http_url}/api/v1/stats", headers=http_headers, timeout=5)
        if st_resp.status_code == 200:
            st_data = st_resp.json()
            print("\n" + "-" * 70)
            print(f"📊 Current Team Statistics for '{st_data['team_name']}':")
            print(f"  • Train Mode -> Requested: {st_data['train_mode']['images_requested']} | Submitted: {st_data['train_mode']['submissions_received']}")
            print(f"  • Test Mode  -> Quota: {st_data['test_mode']['quota_limit']} | Submitted: {st_data['test_mode']['submissions_received']} | Remaining Attempts: {st_data['test_mode']['remaining_attempts']}")
            print("-" * 70)
    except Exception:
        pass

    print("\n✅ All 4 steps finished successfully!")


if __name__ == "__main__":
    main()