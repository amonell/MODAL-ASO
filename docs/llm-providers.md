# Language models

MODAL-ASO defaults to `deepseek/deepseek-v4.1-flash-1017` through OpenRouter for every built-in language-model role, including intake, design, review and reporting. Your provider bills API usage directly. The CLI prints the resolved default model at startup and any separate intake-model override.

## OpenRouter

```bash
export OPENROUTER_API_KEY="your-api-key"
modal-aso "your request"
```

To choose another model, set `OPENROUTER_MODEL` to an exact ID from the [model catalog](https://openrouter.ai/models). This override applies to every agent, including intake, planning, repairs and reporting. Choose a model with tool calling and support for the selected reasoning effort. Without an effort override, each role keeps its configured reasoning effort.

## Common settings

| Variable | Purpose |
| --- | --- |
| `ASO_LLM_PROVIDER` | `openrouter` by default; set `openai` to use OpenAI directly. |
| `OPENROUTER_API_KEY` | OpenRouter credential. |
| `OPENROUTER_MODEL` | Optional exact `provider/model` override for every OpenRouter call. The built-in default is `deepseek/deepseek-v4.1-flash-1017`. |
| `OPENROUTER_REASONING_EFFORT` | Model-supported effort: `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`. |
| `OPENROUTER_ONLY_PROVIDERS` | Comma-separated serving-provider allowlist. |
| `OPENROUTER_PROVIDER_ORDER` | Preferred serving-provider order. |
| `OPENROUTER_PROVIDER_SORT` | `throughput` (default), `latency`, `price` or `balanced`. |
| `OPENROUTER_ZDR` | Set `true` to request zero-data-retention endpoints. |
| `OPENROUTER_DATA_COLLECTION` | `allow` or `deny`. |
| `ASO_LLM_TRANSPORT_RETRIES` | Retry count for transient transport failures; default `0`. |

`OPENROUTER_BASE_URL` overrides the API endpoint. Credentials are read from the environment.

For individual roles, `ASO_GOAL_INTAKE_MODEL`, `ASO_REQUEST_WORKBENCH_MODEL`, `ASO_WORKBENCH_RESPONSE_MODEL`, `ASO_SANDBOX_ANALYSIS_MODEL` and `ASO_ANSWER_WEB_SEARCH_MODEL` select model IDs. A global `OPENROUTER_MODEL` setting takes precedence.

MODAL-ASO validates structured model responses locally. Reports retain model attribution and evidence from the run.

## Explicit OpenAI configuration

Existing OpenAI users can retain their provider and its GPT-5.2/GPT-5-mini role defaults by selecting it explicitly:

```bash
export ASO_LLM_PROVIDER=openai
export OPENAI_API_KEY="your-api-key"
```

An existing `ASO_LLM_PROVIDER` or model override remains effective. To return to the built-in Flash configuration, clear those overrides before starting a new process. API-key presence alone never changes the selected provider.
