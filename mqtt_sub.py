"""Watch what client.py publishes to the TESA MQTT scoreboard (API_DOCS.md section 2).

Run (start this FIRST, then run client.py in another terminal; messages are not retained):
    python mqtt_sub.py                 # own topic /event/<MQTT_KEY>/<TRACK>/<EVENT>
    python mqtt_sub.py --all           # every track/event under /event/<MQTT_KEY>/#
    python mqtt_sub.py --count 1       # exit after the first message
Credentials and broker default to the values in client.py (same team ID, MQTT key, WebSocket TLS on 443).
The spec does not say subscribing is allowed: if the broker refuses, this prints its SUBACK/CONNACK reason.
Each message is checked against the spec: fields, counts keys, and defect_ratio = (total - good) / total.
"""
import argparse
import json
import ssl
import sys
import uuid
from datetime import datetime

import paho.mqtt.client as mqtt

from client import DEFAULT_EVENT, DEFAULT_MQTT_HOST, DEFAULT_MQTT_KEY, DEFAULT_TEAM_ID, DEFAULT_TRACK

COUNT_KEYS = {"good", "shell", "black", "husk", "broken", "parchment", "immature"}


def check_payload(data):
    """Return a list of spec problems in one scoreboard message (empty when it matches API_DOCS.md 2.1)."""
    if not isinstance(data, dict):
        return ["payload is not a JSON object"]
    problems = []
    missing = {"image", "latency_ms", "counts", "defect_ratio"} - set(data)
    if missing:
        problems.append(f"missing fields {sorted(missing)}")
    if "latency_ms" in data and not isinstance(data["latency_ms"], int):
        problems.append("latency_ms is not an integer")
    counts = data.get("counts")
    if isinstance(counts, dict):
        if set(counts) != COUNT_KEYS:
            problems.append(f"counts keys differ: missing {sorted(COUNT_KEYS - set(counts))}, "
                            f"extra {sorted(set(counts) - COUNT_KEYS)}")
        elif isinstance(data.get("defect_ratio"), (int, float)):
            total = sum(counts.values())
            expected = round((total - counts["good"]) / total, 4) if total else 0.0
            if abs(data["defect_ratio"] - expected) > 5e-5:
                problems.append(f"defect_ratio {data['defect_ratio']} != (total - good) / total = {expected}")
    elif "counts" in data:
        problems.append("counts is not an object")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team-id", default=DEFAULT_TEAM_ID)
    parser.add_argument("--mqtt-key", default=DEFAULT_MQTT_KEY)
    parser.add_argument("--mqtt-host", default=DEFAULT_MQTT_HOST)
    parser.add_argument("--port", type=int, default=443)
    parser.add_argument("--track", default=DEFAULT_TRACK, choices=["Embedded", "EdgeAI", "Market"])
    parser.add_argument("--event", default=DEFAULT_EVENT)
    parser.add_argument("--all", action="store_true", help="subscribe /event/<MQTT_KEY>/# (all tracks and events)")
    parser.add_argument("--topic", default=None, help="explicit topic filter (overrides --track/--event/--all)")
    parser.add_argument("--count", type=int, default=0, help="exit after N messages (0 = until Ctrl+C)")
    parser.add_argument("--no-tls", action="store_true", help="plain ws:// (local test brokers only)")
    args = parser.parse_args()

    key = args.mqtt_key
    topic = args.topic or (f"/event/{key}/#" if args.all else f"/event/{key}/{args.track}/{args.event}")
    shown = lambda t: t.replace(key, key[:6] + "...") if key else t  # keep the key out of the screen/logs
    state = {"received": 0, "exit": 0}

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f"[-] Broker refused the login: {reason_code}")
            state["exit"] = 1
            client.disconnect()
            return
        print(f"[+] Connected to {args.mqtt_host}:{args.port}/mqtt/ as {args.team_id}")
        client.subscribe(topic, qos=1)

    def on_subscribe(client, userdata, mid, reason_code_list, properties):
        refused = [str(rc) for rc in reason_code_list if rc.is_failure]
        if refused:
            print(f"[-] Broker refused the subscription to {shown(topic)}: {', '.join(refused)}")
            state["exit"] = 1
            client.disconnect()
            return
        print(f"[+] Subscribed to {shown(topic)} (QoS {reason_code_list[0].value}). "
              "Waiting for messages... run client.py now (Ctrl+C to stop)")

    def on_message(client, userdata, msg):
        state["received"] += 1
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"\n[{stamp}] #{state['received']} {shown(msg.topic)} ({len(msg.payload)} bytes)")
        try:
            data = json.loads(msg.payload)
        except ValueError:
            print("    [!] not JSON:", msg.payload[:300])
            data = None
        if data is not None:
            print("    " + json.dumps(data, ensure_ascii=False))
            problems = check_payload(data)
            print("    [!] " + "; ".join(problems) if problems else "    [ok] matches the API_DOCS.md payload format")
        if args.count and state["received"] >= args.count:
            client.disconnect()

    def on_disconnect(client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f"[-] Disconnected by broker: {reason_code}")
            state["exit"] = state["exit"] or 1

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{args.team_id}:sub-{uuid.uuid4().hex[:12]}",
                         transport="websockets", protocol=mqtt.MQTTv5, reconnect_on_failure=False)
    client.username_pw_set(args.team_id, key)
    client.ws_set_options(path="/mqtt/")
    if not args.no_tls:
        client.tls_set_context(ssl.create_default_context())
    client.connect_timeout = 10
    client.on_connect, client.on_subscribe = on_connect, on_subscribe
    client.on_message, client.on_disconnect = on_message, on_disconnect
    try:
        client.connect(args.mqtt_host, args.port, keepalive=30)
        client.loop_forever()
    except KeyboardInterrupt:
        client.disconnect()
    except Exception as error:
        print(f"[-] MQTT connection error: {error}")
        return 1
    print(f"Stopped after {state['received']} message(s).")
    return state["exit"]


if __name__ == "__main__":
    sys.exit(main())
