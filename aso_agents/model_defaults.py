"""Built-in model roles, resolved by the selected provider at call time."""

DEFAULT_PROVIDER = "openrouter"
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4.1-flash-1017"

# Keep defaults separate from explicit model IDs so role-specific overrides
# retain their meaning. Both roles use Flash on OpenRouter; opting into OpenAI
# preserves that provider's existing standard and small model choices.
DEFAULT_MODEL = "modal-default"
DEFAULT_SMALL_MODEL = "modal-small"
