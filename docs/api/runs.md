# Recipes and runs

`run_pipeline` runs the stages a recipe names; `load_recipe` and
`load_params` read and check a recipe and its params file, and
`load_dataset_type` the dataset type file a recipe names. `scaffold_setup`
writes a folder of recipes and the configs they overlay. See
[Running a pipeline](../running.md) and the
[configuration reference](../configuration.md).

::: redact.driver
    options:
      members:
        - run_pipeline

::: redact.recipe
    options:
      members:
        - load_recipe
        - load_params
        - load_dataset_type

::: redact.llms.prompting.scaffold.scaffold_setup
