#!/usr/bin/env python3
"""Step 5: expose the FlightLxx policy through a small localhost UDP service."""

import argparse
import json
import os
import socket

import numpy as np

from step04_policy_smoke import flightlxx_root, load_policy, predict_action


class PolicyServer:
    """Receive one observation and return one deterministic policy action."""

    def __init__(self, model, host, port):
        self.model = model
        self.address = (host, int(port))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.address)

    def close(self):
        self.sock.close()

    def handle_request(self, request):
        """Convert one decoded JSON object into a JSON-serializable response."""
        if not isinstance(request, dict):
            return {
                "ok": False,
                "error": "request must be a JSON object",
            }

        sequence = request.get("seq")
        request_type = request.get("type", "infer")

        if request_type == "ping":
            return {
                "ok": True,
                "type": "pong",
                "seq": sequence,
            }

        if request_type != "infer":
            return {
                "ok": False,
                "seq": sequence,
                "error": "type must be 'infer' or 'ping'",
            }

        try:
            action = predict_action(self.model, request["obs"])
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return {
                "ok": False,
                "seq": sequence,
                "error": str(exc),
            }

        return {
            "ok": True,
            "type": "action",
            "seq": sequence,
            "action": action.tolist(),
        }

    def handle_datagram(self, data):
        """Decode one UDP datagram and return response bytes."""
        try:
            request = json.loads(data.decode("utf-8"))
            response = self.handle_request(request)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            response = {
                "ok": False,
                "error": "invalid JSON: %s" % exc,
            }

        return json.dumps(
            response,
            separators=(",", ":"),
        ).encode("utf-8")

    def serve_forever(self):
        host, port = self.sock.getsockname()

        print(
            "policy server listening on udp://%s:%d"
            % (host, port)
        )
        print(
            'request: {"type":"infer","seq":1,"obs":[12 floats]}'
        )

        try:
            while True:
                data, client_address = self.sock.recvfrom(65535)
                response = self.handle_datagram(data)
                self.sock.sendto(response, client_address)

        except KeyboardInterrupt:
            print("\npolicy server stopped")

        finally:
            self.close()


class FakeModel:
    """Fake model used by --self-test, so TensorFlow is not required."""

    def predict(self, batch, deterministic=True):
        assert deterministic is True
        assert np.asarray(batch).shape == (1, 12)

        return np.asarray(
            [[-0.1, 0.2, 0.3, 0.4]],
            dtype=np.float32,
        ), None


def run_self_test():
    server = PolicyServer(
        FakeModel(),
        "127.0.0.1",
        0,
    )

    try:
        pong = server.handle_request(
            {
                "type": "ping",
                "seq": 7,
            }
        )

        assert pong == {
            "ok": True,
            "type": "pong",
            "seq": 7,
        }

        response = server.handle_request(
            {
                "type": "infer",
                "seq": 8,
                "obs": [0.0] * 12,
            }
        )

        assert response["ok"] is True
        assert response["seq"] == 8
        assert np.allclose(
            response["action"],
            [-0.1, 0.2, 0.3, 0.4],
        )

        bad = server.handle_request(
            {
                "type": "infer",
                "seq": 9,
                "obs": [0.0],
            }
        )

        assert bad["ok"] is False

        malformed = server.handle_datagram(b"not-json")
        assert json.loads(malformed)["ok"] is False

    finally:
        server.close()

    print("step05 self-test: PASS")


def main():
    parser = argparse.ArgumentParser(
        description="Serve a FlightLxx PPO2 policy over localhost UDP"
    )

    parser.add_argument("--flightlxx-path", default=None)

    parser.add_argument(
        "--model",
        default="Simulation_Experiments/models/CTBR.zip",
        help="model path relative to FlightLxx root or an absolute path",
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
        "--self-test",
        action="store_true",
        help="test protocol without loading TensorFlow",
    )

    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    root = flightlxx_root(args.flightlxx_path)

    model_path = os.path.expanduser(args.model)

    if not os.path.isabs(model_path):
        model_path = os.path.join(root, model_path)

    model_path = os.path.abspath(model_path)

    model = load_policy(root, model_path)

    server = PolicyServer(
        model,
        args.host,
        args.port,
    )

    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())