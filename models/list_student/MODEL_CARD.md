# RETINA list student - list-student-2026-10-14

Plans an NPC's shopping list: from ~30 real candidate products of the RETINA Supercenter (their usual products, what goes with them, new ideas, deals) it decides what goes on today's list and how many. Learned from LLM teachers' lists. Every output is SIMULATED behaviour.

- Lists: 872 from gemini/gemini-3.5-flash-lite 504, chatgpt/GPT-6 268, groq/openai/gpt-oss-120b 100; split by NPC (train 594, validation 128, test 150).

| on test NPCs | items per list | usual products | deals | ran out at home |
|---|---|---|---|---|
| teachers | 12.747 | 93% | 12% | 92% |
| list student, no correction | 12.993 | 91% | 12% | 91% |
| list student (drawn, used in the store) | 13.06 | 58% | 23% | 58% |

AUC 0.94, F1 with the teachers 0.787, quantity error (MAE) 0.061. Real re-buy share for comparison: ~60% (Instacart).

Policy: draw each candidate with sigmoid(logit(p) + bias_shift - habit x bought_before); quantity = round(qty model), 1-6; bias_shift 2.2336 makes the average list size match the teachers' (12.56).

Habit correction (ESTIMATED): the teachers put far more usual products on a list than real shoppers re-buy, so at run time each usual product's logit is lowered by 3.7371 (tuned on training + validation NPCs, list size kept) to bring the usual share to ~60% (Instacart reorder rate (PUBLIC_STATISTIC)). Checked on the test NPCs above.
