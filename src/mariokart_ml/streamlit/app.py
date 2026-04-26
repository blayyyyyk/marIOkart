import json
import socket
import threading
import time
from argparse import ArgumentParser
from typing import Any

import streamlit as st
from mariokart_ml.config import STREAMLIT_UDP_DATA_PORT

# maintains the latest cross-thread telemetry payloads for all instances.
SHARED_STATE: dict[int, dict[str, Any]] = {}

# enforces atomic operations when reading or writing to the global buffer.
STATE_LOCK = threading.Lock()


def telemetry_worker(sock: socket.socket, instance_id: int, stop_event: threading.Event) -> None:
    # loops perpetually to ingest data unless explicitly signaled to terminate.
    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(65535)
            payload = json.loads(data.decode("utf-8"))

            with STATE_LOCK:
                SHARED_STATE[instance_id] = payload
        except TimeoutError:
            # yields control periodically to check the stop_event condition.
            continue
        except OSError:
            # triggers immediately when the main thread forces the socket closed during cleanup.
            break
        except Exception:
            continue

    sock.close()


def initialize_daemons(num_instances: int, base_data_port: int) -> None:
    # initializes the registry map to track active file descriptors and kill switches.
    if "udp_registry" not in st.session_state:
        st.session_state.udp_registry = {}
        st.session_state.daemon_config = (None, None)

    # bypasses reallocation if the requested topography matches the active topography.
    if st.session_state.daemon_config == (num_instances, base_data_port):
        return

    # reaps all orphaned threads and frees up network ports prior to reconfiguration.
    for _port, handles in st.session_state.udp_registry.items():
        handles["stop_event"].set()
        handles["sock"].close()

    st.session_state.udp_registry.clear()

    with STATE_LOCK:
        SHARED_STATE.clear()

    # spools up the new worker fleet using defensive socket configurations.
    for i in range(num_instances):
        port = base_data_port + i

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))

        sock.settimeout(1.0)

        stop_event = threading.Event()
        threading.Thread(target=telemetry_worker, args=(sock, i, stop_event), daemon=True).start()

        st.session_state.udp_registry[port] = {"sock": sock, "stop_event": stop_event}

    st.session_state.daemon_config = (num_instances, base_data_port)


def generate_telemetry_html(telemetry: dict[str, Any], instance_id: int) -> str:
    # isolates the payload copy to prevent mutating the shared state dictionary.
    telemetry_local = telemetry.copy()

    # extracts the net reward explicitly, defaulting to 0.0 if not present.
    net_reward_data = telemetry_local.pop("reward", [0.0])
    net_reward_val = float(net_reward_data[0] if isinstance(net_reward_data, list) else net_reward_data)

    # extracts the latency metric, formatting it as a discrete badge instead of a bar.
    latency_data = telemetry_local.pop("latency", [0.0]) * 1000
    latency_val = float(latency_data[0] if isinstance(latency_data, list) else latency_data)

    # partitions the remaining keys into reward components and standard observations.
    reward_components = {}
    observations = {}
    for k, v in telemetry_local.items():
        if k.startswith("_") and ("reward" in k or "penalty" in k):
            reward_components[k] = v
        else:
            observations[k] = v

    def build_bar_html(key: str, val: Any) -> str:
        # encapsulates the bipolar meter logic to prevent code duplication across columns.
        raw_val = val[0] if isinstance(val, list) else val
        clamped_val = max(-1.0, min(1.0, float(raw_val)))

        width_pct = abs(clamped_val) * 50.0
        left_pct = 50.0 - width_pct if clamped_val < 0 else 50.0
        bar_color = "#ef4444" if clamped_val < 0 else "#22c55e"

        return (
            f"<div style='display: flex; justify-content: space-between; margin-top: 6px; color: #E0E0E0;'>"
            f"<span>{key}</span><span>{raw_val:.4f}</span></div>"
            f"<div style='width: 100%; height: 8px; background: #262730; border-radius: 4px; position: relative; margin-bottom: 4px;'>"
            f"<div style='position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: #888; z-index: 10;'></div>"
            f"<div style='position: absolute; top: 0; bottom: 0; left: {left_pct}%; width: {width_pct}%; background: {bar_color}; border-radius: 2px;'></div>"
            f"</div>"
        )

    # initializes the wrapper with an ID for anchor routing, injecting the latency pill into the header layout.
    html_lines = [
        f"<div id='instance-{instance_id}' style='scroll-margin-top: 80px; margin-bottom: 48px;'>"
        f"<div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #333; padding-bottom: 8px; margin-bottom: 16px;'>"
        f"<h2 style='color: #E0E0E0; margin: 0;'>Instance {instance_id}</h2>"
        f"<div style='background: #262730; padding: 6px 12px; border-radius: 6px; font-family: monospace; font-size: 14px; color: #A0A0A0; border: 1px solid #444;'>"
        f"Step Latency: <span style='color: #E0E0E0; font-weight: bold;'>{latency_val:.1f} ms</span>"
        f"</div></div>"
        f"<div style='display: flex; gap: 32px; font-family: monospace; font-size: 12px; background: #0E1117; padding: 20px; border-radius: 8px;'>"
    ]

    # populates the left column with standard observation metrics.
    html_lines.append("<div style='flex: 1; display: flex; flex-direction: column;'>")
    html_lines.append("<h3 style='margin-top: 0; margin-bottom: 12px; color: #E0E0E0; border-bottom: 1px solid #333; padding-bottom: 8px;'>Observations</h3>")
    for key, val in sorted(observations.items()):
        html_lines.append(build_bar_html(key, val))
    html_lines.append("</div>")

    # populates the right column with net reward and decomposed reward shaping metrics.
    html_lines.append("<div style='flex: 1; display: flex; flex-direction: column;'>")
    html_lines.append("<h3 style='margin-top: 0; margin-bottom: 12px; color: #E0E0E0; border-bottom: 1px solid #333; padding-bottom: 8px;'>Net Reward</h3>")
    html_lines.append(build_bar_html("reward", net_reward_val))

    html_lines.append("<h3 style='margin-top: 24px; margin-bottom: 12px; color: #E0E0E0; border-bottom: 1px solid #333; padding-bottom: 8px;'>Reward Distribution</h3>")
    for key, val in sorted(reward_components.items()):
        html_lines.append(build_bar_html(key, val))
    html_lines.append("</div>")

    # closes the inner flex container and outer instance wrapper.
    html_lines.append("</div></div>")
    return "".join(html_lines)


def render_dashboard(num_instances: int, base_data_port: int) -> None:
    # configures a wide layout to maximize the two-column metric visibility.
    st.set_page_config(layout="wide")
    st.sidebar.title("Emulator Instance Quick Select")

    initialize_daemons(num_instances, base_data_port)

    nav_links = ["<div style='display: flex; flex-direction: column; gap: 8px;'>"]
    for i in range(num_instances):
        nav_links.append(
            f"<a href='#instance-{i}' style='text-decoration: none; color: #E0E0E0; background: #262730; "
            f"padding: 10px; border-radius: 6px; text-align: center; font-family: sans-serif; font-weight: 600; "
            f"border: 1px solid #333; transition: background 0.2s;'>Instance {i}</a>"
        )
    nav_links.append("</div>")
    st.sidebar.markdown("".join(nav_links), unsafe_allow_html=True)

    st.title("RL Telemetry Fleet View")

    # initializes stable containers for all instances in a vertical stack.
    containers = [st.empty() for _ in range(num_instances)]

    # traps the execution thread in a localized render loop for real-time projection.
    while True:
        with STATE_LOCK:
            # creates a shallow copy to minimize the time the lock is held.
            current_state = SHARED_STATE.copy()

        for i, container in enumerate(containers):
            telemetry = current_state.get(i)
            if telemetry:
                html_block = generate_telemetry_html(telemetry, i)
                container.markdown(html_block, unsafe_allow_html=True)
            else:
                container.info(f"Awaiting telemetry on port {base_data_port + i}...")

        # throttles the ui projection to ~20 fps to preserve browser responsiveness.
        time.sleep(0.05)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--base-data-port", type=int, default=STREAMLIT_UDP_DATA_PORT)
    args = parser.parse_args()
    render_dashboard(args.num_instances, args.base_data_port)
