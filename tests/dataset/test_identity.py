"""Tests for a row's id."""

from redact.dataset import content_id


class TestContentId:
    def test_is_the_first_16_hex_chars_of_the_md5(self):
        assert content_id("hello") == "5d41402abc4b2a76"

    def test_same_text_gives_same_id(self):
        assert content_id("a prompt") == content_id("a prompt")

    def test_different_text_gives_different_id(self):
        assert content_id("a prompt") != content_id("another prompt")

    def test_hashes_the_utf8_bytes(self):
        assert content_id("é") == "66ddcd97cfdeabb2"
