import os
import tempfile
import unittest
from unittest.mock import patch

from sparky.config.runtime import resolve_runtime_config
from sparky.llm.backends.remote_http import RemoteHTTPProvider

ENV = {"ROBOT_REMOTE_MODEL": "model-a"}


class RemoteHTTPTLSContextTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.ca_path = os.path.join(temporary.name, "ca.pem")
        with open(self.ca_path, "w", encoding="utf-8") as handle:
            handle.write("certificate placeholder")

    def test_request_uses_configured_ca_context(self):
        config = resolve_runtime_config(
            None, {**ENV, "ROBOT_REMOTE_CA_FILE": self.ca_path}
        )
        sentinel = object()
        with patch(
            "sparky.llm.backends.remote_http.tls_context", return_value=sentinel
        ) as context_fn:
            with patch("sparky.llm.backends.remote_http.urlopen") as urlopen:
                urlopen.return_value.__enter__.return_value.read.return_value = (
                    b'{"choices": [{"message": {"content": "hello"}}]}'
                )
                reply = RemoteHTTPProvider(config).chat(
                    [{"role": "user", "content": "hi"}], max_tokens=10
                )

        self.assertEqual(reply, "hello")
        context_fn.assert_called_once_with(config)
        self.assertIs(urlopen.call_args.kwargs["context"], sentinel)


if __name__ == "__main__":
    unittest.main()
