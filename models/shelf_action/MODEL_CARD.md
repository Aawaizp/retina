# Shelf action student — 2026-10-29

**Status: EXPERIMENTAL; live use stays disabled.** This is a local binary SKIP-vs-BUY head. When it chooses
BUY, the existing product student chooses the SKU. The list student plans
quantity; live stock and remaining budget can reduce it. No live LLM call.

## Evidence

- Source: 4,608 valid LLM teacher shelf labels, only **57 SKIPs**.
- Train: 2,765 situations, 34 SKIPs. Entire NPCs were held out for validation and test.
- Validation: 921 situations, 11 SKIPs; average precision 0.102 vs 0.012 prevalence baseline; Brier 0.0114.
- Test: 922 situations, 12 SKIPs; average precision 0.069 vs 0.013 prevalence baseline; Brier 0.0135 vs 0.0128 constant-rate baseline; predicted skip share 1.7% vs observed 1.3%.
- Chosen logistic regularization C=0.01; class weighting balanced. Features include shopper traits, trip intent, stock, price, deals, and previous purchases; option order and teacher identity are excluded.

## Limits

This model imitates LLM labels, **not real shoppers**. SKIPs are sparse and
mostly from one teacher family; there are only 12 test SKIPs, so
precision has high uncertainty. A good teacher-label test is not proof of
retail realism. Store POS, shelf stockouts, and foot traffic are needed to
calibrate actions and evaluate the final baskets. Quantity remains the list
student's earlier plan, not a new quantity inferred at the shelf.

The `qualified` flag in `shelf_action.json` controls whether live runs may use
the policy. Re-run `python -m retina.student.shelf_action_train` to reproduce
the training and held-out evaluation from the local teacher label database.
