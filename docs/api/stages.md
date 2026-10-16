# Stages

One function per stage, in the order a run takes them. Each resolves its
inputs, models and paths, then calls the stage's core. The stage pages
explain what each reads, writes and resumes by.

::: redact.stages
    options:
      members:
        - generate_constitution
        - generate_inputs
        - generate_outputs
        - generate_paraphrases
        - generate_jailbreaks
        - build_dataset
