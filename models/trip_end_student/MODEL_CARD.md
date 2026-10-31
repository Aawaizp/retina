# Basket completion student

Trained on real Instacart prior orders. At each observed order prefix, the target is whether that product line was the final line. Training, validation, and test users are separate. At a simulated shelf the model estimates whether the current basket is complete; seeded dice turn that probability into a live checkout choice. One local ONNX inference per eligible shelf; no LLM call.

Test: {'auc': 0.7629, 'brier': 0.0821, 'observed_stop_rate': 0.0995, 'predicted_stop_rate': 0.0993, 'mean_order_lines': 10.05, 'onnx_max_diff': 1.8110478905430938e-07}.

**Limit:** Instacart records online order contents and cart order, not physical store paths, intentions, or early departures. Early checkout in RETINA is an inferred behavior, not a directly observed in-store label. The raw order hazard is scaled by 0.1 in RETINA (ESTIMATED) because the list student was already calibrated to basket size; an unscaled checkout hazard reduced the simulated mean from 9.92 to 7.68 lines in a one-day run. Store-specific basket size may need further calibration. Raw Instacart files remain local.
