"""A directive's template variant applied to a prompt."""

from redact.jailbreak.directives import apply_template_directive

_TEMPLATES = {"ask": [{"name": "plain", "template": "{prompt} (in {language})"}]}


def test_extra_fields_fill_the_template_beside_the_prompt():
    assert apply_template_directive(
        _TEMPLATES, "ask", "hi", info_key="language", extra={"language": "Welsh"},
    ) == ("hi (in Welsh)", "language=ask;variant=plain")


def test_the_prompt_is_not_an_extra_field():
    text, _ = apply_template_directive(
        _TEMPLATES, "ask", "hi", extra={"prompt": "no", "language": "Welsh"})
    assert text == "hi (in Welsh)"
