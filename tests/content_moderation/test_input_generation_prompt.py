"""The inputs stage's one generation prompt, ``input/generation``: what its
fields hold, and that the front renders it. Fully offline."""

import hashlib
import json

import pandas as pd

from redact.llms import PromptTemplate, load_prompt
from tests.conftest import MockBackend, make_client

# The SHA-256 of the canonical JSON of the four prompt fields of
# input/generation, as the inputs stage's default template held them at
# commit 6d5d76e, before its two templates became this one. A change of the
# template's wording changes it: update it in the same commit, saying why.
_FIELDS_DIGEST = "8bb81253e9594400c056215efd7e29acbd8ecb350f6a1af6549d966d79c3f336"
_FIELDS = ("system_prompt", "template", "seed_fields", "few_shot_examples")

_SPEC = "SPEC-LINE: a literal spec"


def test_the_packaged_prompt_holds_the_fields_it_was_pinned_with():
    prompt = load_prompt("input", "generation")
    canonical = json.dumps({field: prompt[field] for field in _FIELDS},
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == _FIELDS_DIGEST


def test_the_front_renders_the_packaged_prompt_with_the_units_spec(
        tmp_path, monkeypatch):
    import redact.content_moderation.inputs as G
    from redact import generate_inputs
    from redact.llms import ModelClient

    monkeypatch.setattr(G, "render_spec", lambda axes, spec, fields: _SPEC)
    gen = MockBackend("1. a sample\n[END]")
    backends = {"gen-model": gen, "check-model": MockBackend("Yes")}
    monkeypatch.setattr(ModelClient, "create", lambda m: make_client(backends[m], m))
    entry = {"source_category": "Cyber", "constitution_subcategory": "Sub",
             "sample_description": "what zq0 is about", "entry_type": "harmful"}
    generate_inputs(data_dir=tmp_path, constitution_df=pd.DataFrame([entry]),
                    samples_per_entry=1, model="gen-model", check_model="check-model",
                    verbose=False)

    fields = {"num_samples": "1", "Category": "Cyber",
              "sample_description": "what zq0 is about",
              "constitution_subcategory": "Sub", "entry_type": "harmful",
              "sample_spec": _SPEC}
    system, user = PromptTemplate(load_prompt("input", "generation"),
                                  format_style="numbered", **fields)(**fields)
    (call,) = gen.calls
    assert call["system_prompt"] == system["content"]
    assert call["messages"][-1]["content"] == user["content"]
    assert _SPEC in user["content"]
