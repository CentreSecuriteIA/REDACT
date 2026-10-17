# LLMs

A model name resolves to a ready-to-call `ModelClient`; `register_model`
adds a model to the registry for the running process. The setup classes are
what a registry entry is built from. See [internals/llms.md](../internals/llms.md)
and [the model registry](../configuration.md#the-model-registry).

::: redact.llms.client.ModelClient

::: redact.llms.model_config
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - register_model
        - get_model_config
        - default_model_for_role
        - ModelConfig
        - APIConfig
        - VLLMConfig
        - IntrospectConfig
