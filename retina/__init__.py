"""RETINA - Retail Intelligence & Agent Simulation. Every row it makes is SIMULATED data.

  retina.engine   the v1 world: store, NPC shoppers, prices, stock, traffic, export (statistical engine)
  retina.mind     v2: who decides what an NPC buys (autopilot, student model, LLM teacher)  - Phase 2
  retina.memory   v2: one card per NPC id, so a returning shopper is the same person         - Phase 2
  retina.app      v2: local API (FastAPI) for the website and Docker                         - Phase 5
  retina.cli      the `retina` command

Nothing here imports the engine: the engine reads the chosen store when it is first imported,
so the store must be picked (RETINA_STORE) before that happens - retina.cli does this.
"""

__version__ = "2.0.0.dev0"
