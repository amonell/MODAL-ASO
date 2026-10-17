from agents import Agent

num_mismatch_agent = Agent(
    name="ASO Mismatch Interpreter",
    instructions=(
        "You receive JSON: {\"user_request\": str, \"structured_goal\": dict}. "
        "Read the free-text request and any structured constraints. "
        "Determine if the user explicitly specified how many mismatches/off-target mismatches to allow "
        "when mapping tiles (Bowtie -v parameter). "
        "Return JSON with an integer num_mismatches when specified, otherwise return {}. "
        "If the user says \"no mismatches\", respond with {\"num_mismatches\": 0}. "
        "If they say \"allow up to two mismatches\" return 2, etc. "
        "Only infer values the user clearly states; do not make assumptions."
    ),
    tools=[],
    model="gpt-5-mini-2025-08-07",
)
