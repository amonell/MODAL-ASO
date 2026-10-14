# Language models

Set provider settings before starting MODAL-ASO. Your provider bills API usage directly.

## OpenAI

```bash
export ASO_LLM_PROVIDER=openai
export OPENAI_API_KEY="your-api-key"
modal-aso "your request"
```

OpenAI is the default provider. The built-in configuration assigns models to individual agent roles.

## OpenRouter

```bash
export ASO_LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY="your-api-key"
export OPENROUTER_MODEL="provider/model"
export OPENROUTER_REASONING_EFFORT=max
modal-aso "your request"
```

Select an exact ID from the [model catalog](https://openrouter.ai/models). `OPENROUTER_MODEL` applies to every agent, including intake, planning, repairs and reporting. Choose a model with tool calling and support for the selected reasoning effort.

## Common settings

| Variable | Purpose |
| --- | --- |
| `ASO_LLM_PROVIDER` | `openai` or `openrouter`. |
| `OPENAI_API_KEY` | OpenAI credential. |
| `OPENROUTER_API_KEY` | OpenRouter credential. |
| `OPENROUTER_MODEL` | Exact `provider/model` ID for every OpenRouter call. |
| `OPENROUTER_REASONING_EFFORT` | Model-supported effort: `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`. |
| `OPENROUTER_ONLY_PROVIDERS` | Comma-separated serving-provider allowlist. |
| `OPENROUTER_PROVIDER_ORDER` | Preferred serving-provider order. |
| `OPENROUTER_PROVIDER_SORT` | `throughput` (default), `latency`, `price` or `balanced`. |
| `OPENROUTER_ZDR` | Set `true` to request zero-data-retention endpoints. |
| `OPENROUTER_DATA_COLLECTION` | `allow` or `deny`. |
| `ASO_LLM_TRANSPORT_RETRIES` | Retry count for transient transport failures; default `0`. |

`OPENAI_BASE_URL` and `OPENROUTER_BASE_URL` override the corresponding API endpoint. Credentials are read from the environment.

For individual roles, `ASO_GOAL_INTAKE_MODEL`, `ASO_REQUEST_WORKBENCH_MODEL`, `ASO_WORKBENCH_RESPONSE_MODEL`, `ASO_SANDBOX_ANALYSIS_MODEL` and `ASO_ANSWER_WEB_SEARCH_MODEL` select model IDs. A global `OPENROUTER_MODEL` setting takes precedence.

MODAL-ASO validates structured model responses locally. Reports retain model attribution and evidence from the run.
