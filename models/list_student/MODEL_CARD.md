# RETINA list student - list-student-2026-10-15

Plans an NPC's shopping list: from ~30 real candidate products of the RETINA Supercenter (their usual products, what goes with them, new ideas, deals) it decides what goes on today's list and how many. Learned from LLM teachers' lists. Every output is SIMULATED behaviour.

- Lists: 2,029 from gemini/gemini-3.5-flash-lite 1500, chatgpt/GPT-6 268, groq/openai/gpt-oss-120b 261; split by NPC (train 1371, validation 294, test 364).

| on test NPCs | items per list | usual products | deals | ran out at home |
|---|---|---|---|---|
| teachers | 12.08 | 89% | 17% | 89% |
| list student, no correction | 12.173 | 90% | 14% | 90% |
| list student (drawn, used in the store) | 11.055 | 54% | 27% | 54% |

AUC 0.936, F1 with the teachers 0.762, quantity error (MAE) 0.113. Real re-buy share for comparison: ~60% (Instacart).

Policy: draw each candidate with sigmoid(logit(p) + bias_shift - habit x bought_before); quantity = round(qty model), 1-6; bias_shift 2.077 sets the average list size.

List size correction (ESTIMATED): the teachers wrote 11.6 items per list; real baskets have ~10.09 products (Instacart) and 90% of planned items get bought here, so bias_shift is fitted to 11.16 planned items (instead of the teachers' size).

Habit correction (ESTIMATED): the teachers put far more usual products on a list than real shoppers re-buy, so at run time each usual product's logit is lowered by 3.7872 (tuned on all scenarios of non-test NPCs, list size kept) to bring the usual share to ~60% (Instacart reorder rate (PUBLIC_STATISTIC)). Checked on the test NPCs above.
