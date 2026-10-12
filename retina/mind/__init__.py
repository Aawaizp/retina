"""The NPC mind (v2, Phase 2): decides product + reason for one NPC at a time.
Three levels, cheapest first: autopilot (Python rules) -> student model -> LLM teacher.
Python (the engine) still checks every answer: product exists, in stock, fits the budget."""
