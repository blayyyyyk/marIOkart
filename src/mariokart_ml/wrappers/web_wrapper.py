import json
import socket

import gymnasium as gym

UDP_HOST = "127.0.0.1"


def is_udp_port_in_use(port, host=UDP_HOST):
    # Create a socket object using Datagram (UDP)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Try to bind to the port
        sock.bind((host, port))
        return False  # Port is free
    except OSError:
        return True  # Port is in use
    finally:
        sock.close()


class WebWrapper(gym.Wrapper):
    def __init__(self, env, data_port: int):
        super().__init__(env)
        self.data_port = data_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)

        assert isinstance(obs, dict)
        data = {k: v.tolist() for k, v in obs.items()}

        data["reward"] = reward  # total reward
        data["latency"] = info.get("latency", 0.0)

        if self.has_wrapper_attr("reward_components"):
            data |= self.get_wrapper_attr("reward_components")

        data = json.dumps(data)
        byte_data = bytes(data, encoding="utf-8")

        self.sock.sendto(byte_data, (UDP_HOST, self.data_port))

        return obs, reward, done, truncated, info
