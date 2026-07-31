# Chart map

| Report segment | Analytical question | Family | Type | Fields |
|---|---|---|---|---|
| Quality by interval | Which condition has the highest pass@1? | Comparison | horizontalBar | `condition_label, pass_at_1` |
| Quality vs teacher hours | Which points trade quality for service time? | Relationship | scatter | `teacher_gpu_hours, pass_at_1` |
| Quality vs anchors | How does position count relate to quality? | Relationship | scatter | `teacher_anchor_positions, pass_at_1` |
| Wall breakdown | Where is end-to-end time spent? | Composition | stackedBar | `condition, phase, seconds` |
| Residual vs step | Does sparse support remain adequate? | Trend | line | `step, condition, residual_student_mass` |
| KL vs step | How noisy/stable is the optimization signal? | Trend | line | `step, condition, kl` |
| Completion length vs step | Is behavior or truncation shifting? | Trend | line | `step, condition, completion_length_tokens` |

Palette policy: single blue root for single-series comparisons; categorical palette only where condition or phase identity is a second visible dimension.
