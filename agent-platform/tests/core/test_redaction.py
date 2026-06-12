"""redaction 模块测试。"""

from __future__ import annotations

from src.core.redaction import REDACTED, is_sensitive_key, redact_text, redact_value


class TestSensitiveKeys:
    def test_matches(self) -> None:
        for key in ("password", "api_key", "apikey", "Authorization", "PRIVATE_KEY", "token"):
            assert is_sensitive_key(key), key

    def test_non_matches(self) -> None:
        for key in ("username", "content", "model", "tokens_used"):
            assert not is_sensitive_key(key), key


class TestRedactText:
    def test_bearer_token(self) -> None:
        assert "abc123def456" not in redact_text("Authorization: Bearer abc123def456ghi")

    def test_sk_key(self) -> None:
        assert redact_text("key is sk-proj1234567890abcdef") == f"key is {REDACTED}"

    def test_email_partially_masked(self) -> None:
        out = redact_text("contact: zhang.san@example.com")
        assert "zhang.san@" not in out
        assert "z***@example.com" in out

    def test_cn_phone_masked(self) -> None:
        out = redact_text("电话 13812345678")
        assert "13812345678" not in out
        assert "138****5678" in out

    def test_jwt_masked(self) -> None:
        token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N"
        assert token not in redact_text(f"got {token}")

    def test_plain_text_untouched(self) -> None:
        text = "今天的检索召回率提升了 12%"
        assert redact_text(text) == text


class TestRedactValue:
    def test_nested_dict(self) -> None:
        out = redact_value(
            {
                "user": "alice",
                "api_key": "sk-secret",
                "nested": {"password": "p@ss", "note": "ok"},
                "items": [{"token": "t"}, "plain"],
            }
        )
        assert out["user"] == "alice"
        assert out["api_key"] == REDACTED
        assert out["nested"]["password"] == REDACTED
        assert out["nested"]["note"] == "ok"
        assert out["items"][0]["token"] == REDACTED
        assert out["items"][1] == "plain"

    def test_depth_bomb_guard(self) -> None:
        deep: dict[str, object] = {}
        node = deep
        for _ in range(50):
            child: dict[str, object] = {}
            node["d"] = child
            node = child
        out = redact_value(deep)
        assert isinstance(out, dict)  # 不抛 RecursionError

    def test_non_json_types_passthrough(self) -> None:
        assert redact_value(42) == 42
        assert redact_value(None) is None
        assert redact_value((1, "a")) == (1, "a")
