{#
    dbt's default generate_schema_name concatenates the target schema with
    any custom +schema config (e.g. target "main" + custom "marts" becomes
    "main_marts"), which would silently break every downstream consumer that
    expects the schema names declared in dbt_project.yml (staging,
    intermediate, marts) and documented in the README. This override uses
    the custom schema name verbatim - the standard fix from dbt's own docs
    for exactly this surprise.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
