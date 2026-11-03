"""Tests run without any model download: a stub engine stands in for transformers."""
import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import build_handler, validate_request  # noqa: E402


class StubEngine:
    model_id = "stub/model"
    revision = "r1"
    device = "cpu"

    def generate(self, prompt, max_tokens, temperature, top_p, seed=None):
        return f"echo:{prompt[:20]}"


class ValidateTests(unittest.TestCase):
    def test_rejects_missing_prompt(self):
        self.assertEqual(validate_request({})[1], "'prompt' must be a non-empty string")

    def test_rejects_oversize_tokens(self):
        _, err = validate_request({"prompt": "hi", "max_tokens": 9999})
        self.assertIn("max_tokens", err)

    def test_defaults(self):
        params, err = validate_request({"prompt": "hi"})
        self.assertIsNone(err)
        self.assertEqual(params["max_tokens"], 200)
        self.assertEqual(params["temperature"], 0.7)


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0),
                                         build_handler(StubEngine(), token="t0k"))
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _post(self, path, payload, token=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {token}"} if token else {})})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_health(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health") as resp:
            data = json.loads(resp.read())
        self.assertTrue(data["ok"])
        self.assertEqual(data["model"], "stub/model")

    def test_generate_requires_token(self):
        status, _ = self._post("/generate", {"prompt": "hai"})
        self.assertEqual(status, 401)

    def test_generate_roundtrip(self):
        status, data = self._post("/generate", {"prompt": "hai"}, token="t0k")
        self.assertEqual(status, 200)
        self.assertEqual(data["text"], "echo:hai")
        self.assertEqual(data["model"], "stub/model")

    def test_generate_rejects_bad_body(self):
        status, data = self._post("/generate", {"prompt": ""}, token="t0k")
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
