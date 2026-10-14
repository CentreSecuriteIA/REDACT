"""Tests for the typo techniques."""


def test_level_descriptions_come_from_the_prompt_file():
    from redact.llms.prompting import load_prompt

    config = load_prompt("jailbreak", "rewrite_with_typos")
    assert set(config["level_descriptions"]) == {"low", "medium", "high", "insane"}
