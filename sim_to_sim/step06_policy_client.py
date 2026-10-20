#!/usr/bin/env python3
"""Step 6: send one observation to the local FlightLxx policy server."""

import argparse
import json
import math
import socket


def validate_observation(observation):
    values = [float(value) for value in observation]

    if len(values) != 12:
        raise ValueError("observation must contain exactly 12 values")

    if not all(math.isfinite(value) for value in values):
        raise ValueError("observation contains NaN or Inf")

    return values


def validate_action(action):
    values = [float(value) for value in action]

    if len(values) != 4:
        raise RuntimeError(
            "policy returned %d action values" % len(values)
        )

    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("policy returned NaN or Inf")

    return values


class PolicyClient:
    def __init__(self, host, port, timeout):
        self.server_address = (host, int(port))
        self.timeout = float(timeout)

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )
        self.sock.settimeout(self.timeout)

    def close(self):
        self.sock.close()

    def request(self, payload):
        data = json.dumps(
            payload,
            separators=(",", ":"),
        ).encode("utf-8")

        self.sock.sendto(data, self.server_address)

        response_data, _ = self.sock.recvfrom(65535)
        response = json.loads(
            response_data.decode("utf-8")
        )

        if not isinstance(response, dict):
            raise RuntimeError(
                "server response must be a JSON object"
            )

        if response.get("ok") is not True:
            raise RuntimeError(
                response.get(
                    "error",
                    "policy server rejected request",
                )
            )

        return response

    def ping(self):
        return self.request(
            {
                "type": "ping",
                "seq": 0,
            }
        )

    def infer(self, observation, sequence=1):
        observation = validate_observation(observation)

        response = self.request(
            {
                "type": "infer",
                "seq": int(sequence),
                "obs": observation,
            }
        )

        if response.get("type") != "action":
            raise RuntimeError(
                "server response type is not 'action'"
            )

        if response.get("seq") != int(sequence):
            raise RuntimeError(
                "server response sequence number does not match"
            )

        return validate_action(
            response.get("action", [])
        )


def main():
    parser = argparse.ArgumentParser(
        description="Test the FlightLxx UDP policy server"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5800,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--observation",
        type=float,
        nargs=12,
        default=(
            0.0, 0.0, 5.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ),
        metavar=(
            "X", "Y", "Z",
            "YAW", "PITCH", "ROLL",
            "VX", "VY", "VZ",
            "WX", "WY", "WZ",
        ),
    )

    args = parser.parse_args()

    client = PolicyClient(
        args.host,
        args.port,
        args.timeout,
    )

    try:
        print("ping response:")
        print(client.ping())

        observation = validate_observation(
            args.observation
        )

        action = client.infer(observation)

        print("observation:")
        print(observation)

        print("raw action:")
        print(action)

    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())