"""The student model (v2, Phase 4): a small model that copies the LLM teacher's shelf choices.

  retina collect --decisions 3000     varied situations labelled by Groq openai/gpt-oss-120b only
  retina collect --repeat 100         ask 100 again: teacher self-agreement (the student's ceiling)
  retina collect --status             labels so far, today's Groq use
  retina train                        compare random / engine / logit / LightGBM, export ONNX + model card
  retina --mind mix                   confident student decides, unsure -> teacher (new labels)
  retina --mind mix --student only    no LLM at all
"""
