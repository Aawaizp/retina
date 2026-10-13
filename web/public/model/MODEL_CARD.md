# RETINA student model - student-2026-10-13

A small model that copies the LLM teacher's shelf choices for thinking NPCs, so most of them need no LLM call. It ranks the 2-5 real products offered for one shopping-list item. Every output is SIMULATED behaviour.

## Data
- Teachers: **all** - groq/openai/gpt-oss-120b 1,068, claude-code/opus-5-5 1,000, chatgpt/GPT-5 944, gemini/gemini-3.5-flash-lite 899, chatgpt/gpt-5-codex 497, claude/opus-5-5 200 labels. The student learns their shared habits; test agreement is with whichever teacher answered.
- 4,608 labels (57 SKIPs counted, not learned): situations from `retina collect` (store states varied on purpose: price up, deal, sold out, remembered price) and teacher answers from mix runs.
- Split by NPC: train 3,149 / validation 439 / test 919 decision sets; test NPCs were never seen in training.
- No consensus weighting.
- Family balance: every model family counts the same in training (anthropic 857 sets x1.225, google 548 sets x1.915, openai 1,744 sets x0.602).
- Nothing left out.
- Model chosen by **realism on validation (repeat share drawn vs ~60%, elasticity vs ~-2.5)**.
- No extra teachers.
- Features (28): NPC traits, why the item is on the list, trip type, price change, price level, deal, bought before, usual, engine pick, best seller, popularity, price vs last paid. NOT the option's position.

## Results on held-out NPCs
| chooser | agreement with teacher | log loss | calibration error | repeat share | repeat share (natural; real ~60%) | deal share | own-price elasticity |
|---|---|---|---|---|---|---|---|
| teacher (all teachers) | - | - | - | 50% | 52% | 29% | -1.26 |
| random | 31% | 1.372 | 0.047 | 37% | 37% | 18% | +0.00 |
| engine | 28% | 1.753 | 0.187 | 28% | 28% | 23% | -1.33 |
| conditional logit | 53% | 1.105 | 0.043 | 68% | 73% | 32% | -1.22 |
| LightGBM ranker **(chosen)** | 55% | 1.056 | 0.047 | 66% | 72% | 29% | -1.03 |

## Price correction (ESTIMATED)
Every LLM teacher reacts to price less than real shoppers (research ~-2.5). So the chosen student gets one extra term: score - 1.7756 x price_change, tuned on training + validation NPCs WITHOUT labels (only the student's own picks) until its elasticity reaches the target, then checked on the test NPCs:
| on test NPCs | agreement | log loss | repeat share (natural, drawn) | deal share | elasticity |
|---|---|---|---|---|---|
| without correction | 55% | 1.056 | 58% | 29% | -1.03 |
| with correction (used) | 54% | 1.072 | 55% | 39% | -2.36 |

## Chosen student vs each teacher (held-out NPCs)
| teacher | test sets | student agrees | teacher's own repeat share (natural) |
|---|---|---|---|
| chatgpt/GPT-5 | 157 | 59% | 37% |
| chatgpt/gpt-5-codex | 77 | 49% | 52% |
| claude-code/opus-5-5 | 171 | 50% | 71% |
| claude/opus-5-5 | 30 | 53% | 71% |
| gemini/gemini-3.5-flash-lite | 234 | 58% | 66% |
| groq/openai/gpt-oss-120b | 250 | 52% | 34% |

Teacher self-agreement (the ceiling: the same situation asked twice, measured for Groq): **65%** on 100 items.
Real repeat share for comparison: ~60% (Instacart reorder rate, PUBLIC_STATISTIC).
Own-price elasticity: conditional logit on price change within the options (research on grocery choice: about -2.5).

## Routing
Confidence threshold **0.50** (the lowest one that reaches 95% of the teacher's self-agreement on validation): the student decides **66%** of teacher-worthy choices, agreeing 60% with the teacher; the rest go to the teacher (and become new labels).
Curve (test): 0.30: 98% decided, 55% agree | 0.40: 86% decided, 57% agree | 0.50: 66% decided, 60% agree | 0.60: 45% decided, 67% agree | 0.70: 30% decided, 75% agree | 0.80: 16% decided, 81% agree | 0.90: 7% decided, 83% agree

## Teacher position bias
Pick share by the letter shown, after shuffling: A 24% (expected 20%) B 20% (expected 20%) C 19% (expected 20%) D 20% (expected 20%) E 17% (expected 20%). The student cannot have this bias (position is not a feature).

## Files and use
- `student.onnx` (LightGBM ranker): input `x` [options x 28] float32 in the order of `student.json` features; output = one score per option; probabilities = softmax(scores / temperature) within a decision (temperature 0.669). ONNX vs Python max difference: 3.5e-02.
- Used by `retina --mind mix` (confident -> student, unsure -> teacher); `--student only` for no-LLM runs.
- Pick: **draw** - the student DRAWS an option from its probabilities (seeded dice), like a real shopper; repeat share in natural situations when drawing: 58% (always taking the top option exaggerates habits).

## Limits
- It learns the shared habits of several LLM teachers (they agree only ~55-60% with each other), including their biases (LLM shoppers are more deal-driven than real ones).
- US products and prices; NPCs and situations are SIMULATED; agreement is with the teacher, not with real people.
- Small data: a few thousand decisions. Retrain as labels grow (`retina train`).
