import unittest
from types import SimpleNamespace

from sparky.config.runtime import (
    DEFAULT_REMOTE_URL,
    ConfigError,
    RuntimeConfig,
    auth_headers,
    resolve_runtime_config,
)

BASE_ENV = {"ROBOT_REMOTE_MODEL": "model-a"}


def resolve(cli=None, env=None, **kwargs):
    settings = dict(BASE_ENV)
    if env:
        settings.update(env)
    # An explicit empty value removes the default model for auto-detect tests.
    if settings.get("ROBOT_REMOTE_MODEL") == "":
        settings.pop("ROBOT_REMOTE_MODEL")
    return resolve_runtime_config(cli, settings, **kwargs)


class RuntimeConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = resolve()
        self.assertEqual(config.url, DEFAULT_REMOTE_URL)
        self.assertEqual(config.model, "model-a")
        self.assertIsNone(config.api_key)
        self.assertEqual(config.timeout, 60.0)
        self.assertEqual(config.max_retries, 2)
        self.assertTrue(config.verify_tls)
        self.assertEqual(config.max_tokens, 96)
        self.assertEqual(config.vision_max_tokens, 256)
        self.assertEqual(config.memory_max_turns, 8)
        self.assertEqual(config.image_max_dim, 768)

    def test_url_is_normalized(self):
        config = resolve(env={"ROBOT_REMOTE_URL": "http://127.0.0.1:8080/"})
        self.assertEqual(config.url, "http://127.0.0.1:8080/v1")
        config = resolve(env={"ROBOT_REMOTE_URL": "http://localhost:9000/custom/"})
        self.assertEqual(config.url, "http://localhost:9000/custom")

    def test_precedence_cli_env_default(self):
        cli = SimpleNamespace(remote_url="http://127.0.0.1:1/v1", remote_timeout="12.5")
        env = {"ROBOT_REMOTE_URL": "http://127.0.0.1:2/v1", "ROBOT_REMOTE_TIMEOUT": "7"}
        config = resolve(cli, env)
        self.assertEqual(config.url, "http://127.0.0.1:1/v1")
        self.assertEqual(config.timeout, 12.5)

        config = resolve(None, env)
        self.assertEqual(config.url, "http://127.0.0.1:2/v1")
        self.assertEqual(config.timeout, 7.0)

        empty_cli = SimpleNamespace(remote_url="   ", remote_timeout="")
        config = resolve(empty_cli, env)
        self.assertEqual(config.url, "http://127.0.0.1:2/v1")
        self.assertEqual(config.timeout, 7.0)

    def test_model_alias_precedence(self):
        cli = SimpleNamespace(remote_model="remote", model="legacy")
        self.assertEqual(resolve(cli).model, "remote")
        self.assertEqual(resolve(SimpleNamespace(model="legacy")).model, "legacy")
        self.assertEqual(resolve().model, "model-a")

    def test_auto_detect_adopts_single_model(self):
        env = {"ROBOT_REMOTE_MODEL": ""}
        config = resolve(env=env, model_fetcher=lambda _: ["only-model"])
        self.assertEqual(config.model, "only-model")

    def test_auto_detect_rejects_ambiguous_models(self):
        env = {"ROBOT_REMOTE_MODEL": ""}
        with self.assertRaises(ConfigError) as ctx:
            resolve(env=env, model_fetcher=lambda _: ["one", "two"])
        message = str(ctx.exception)
        self.assertIn("ROBOT_REMOTE_MODEL", message)
        self.assertIn("one, two", message)

    def test_auto_detect_rejects_no_models(self):
        env = {"ROBOT_REMOTE_MODEL": ""}
        with self.assertRaises(ConfigError):
            resolve(env=env, model_fetcher=lambda _: [])

    def test_auto_detect_failure_is_actionable_and_secret_free(self):
        env = {"ROBOT_REMOTE_MODEL": ""}
        env["ROBOT_REMOTE_API_KEY"] = "super-secret"

        def boom(_):
            raise RuntimeError("super-secret leaked from transport")

        with self.assertRaises(ConfigError) as ctx:
            resolve(env=env, model_fetcher=boom)
        message = str(ctx.exception)
        self.assertIn("ROBOT_REMOTE_MODEL", message)
        self.assertNotIn("super-secret", message)

    def test_auto_detect_can_be_disabled(self):
        env = {"ROBOT_REMOTE_MODEL": ""}
        with self.assertRaises(ConfigError):
            resolve(env=env, fetch_model=False)

    def test_loopback_detection(self):
        for url in (
            "http://127.0.0.1:8080/v1",
            "http://localhost:8080/v1",
            "http://[::1]:8080/v1",
        ):
            with self.subTest(url=url):
                self.assertTrue(resolve(env={"ROBOT_REMOTE_URL": url}).is_loopback)
        self.assertFalse(
            resolve(env={"ROBOT_REMOTE_URL": "https://192.168.1.9:8080/v1",
                          "ROBOT_REMOTE_API_KEY": "k"}).is_loopback
        )

    def test_non_loopback_requires_https_and_api_key(self):
        lan = {"ROBOT_REMOTE_URL": "http://192.168.1.9:8080/v1"}
        with self.assertRaisesRegex(ConfigError, "HTTPS"):
            resolve(env=lan)
        lan["ROBOT_REMOTE_API_KEY"] = "k"
        with self.assertRaisesRegex(ConfigError, "HTTPS"):
            resolve(env=lan)
        lan["ROBOT_REMOTE_URL"] = "https://192.168.1.9:8080/v1"
        lan.pop("ROBOT_REMOTE_API_KEY")
        with self.assertRaisesRegex(ConfigError, "API_KEY"):
            resolve(env=lan)
        lan["ROBOT_REMOTE_API_KEY"] = "k"
        self.assertEqual(resolve(env=lan).api_key, "k")

    def test_rejects_unsafe_urls(self):
        bad_urls = (
            "ftp://127.0.0.1/v1",
            "http:///v1",
            "http://user:pass@127.0.0.1/v1",
            "http://127.0.0.1/v1?key=x",
        )
        for url in bad_urls:
            with self.subTest(url=url), self.assertRaises(ConfigError):
                resolve(env={"ROBOT_REMOTE_URL": url})

    def test_strict_numeric_and_boolean_values(self):
        bad_values = (
            {"ROBOT_REMOTE_TIMEOUT": "abc"},
            {"ROBOT_REMOTE_MAX_RETRIES": "x"},
            {"ROBOT_REMOTE_VERIFY_TLS": "maybe"},
            {"ROBOT_LLM_MAX_TOKENS": "0"},
            {"ROBOT_VISION_MAX_TOKENS": "-1"},
            {"ROBOT_MEMORY_MAX_TURNS": "0"},
            {"ROBOT_IMAGE_MAX_DIM": "0"},
            {"ROBOT_REMOTE_TIMEOUT": "-5"},
        )
        for values in bad_values:
            with self.subTest(values=values), self.assertRaises(ConfigError):
                resolve(env=values)
        with self.assertRaises(ConfigError):
            resolve(cli=SimpleNamespace(remote_verify_tls="perhaps"))

    def test_boolean_cli_override(self):
        config = resolve(
            cli=SimpleNamespace(remote_verify_tls=False),
            env={"ROBOT_REMOTE_VERIFY_TLS": "true"},
        )
        self.assertFalse(config.verify_tls)

    def test_secret_is_never_reprd_or_logged(self):
        config = resolve(env={"ROBOT_REMOTE_API_KEY": "super-secret"})
        self.assertNotIn("super-secret", repr(config))
        self.assertNotIn("super-secret", str(config))
        self.assertEqual(config.redacted()["api_key"], "***")
        self.assertNotIn("super-secret", str(config.redacted()))

    def test_auth_headers(self):
        config = resolve(env={"ROBOT_REMOTE_API_KEY": "super-secret"})
        self.assertEqual(auth_headers(config)["Authorization"], "Bearer super-secret")
        without_key = resolve()
        self.assertNotIn("Authorization", auth_headers(without_key))

    def test_direct_construction_is_validated(self):
        with self.assertRaises(ConfigError):
            RuntimeConfig(url=DEFAULT_REMOTE_URL, timeout=0)


if __name__ == "__main__":
    unittest.main()
