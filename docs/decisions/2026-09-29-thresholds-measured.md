# Jev thresholds are measured on a bench, not chosen

- **Problem**: the thresholds (danger 0.2, in_scope 0.7, global confidence 0.8) were set without any measure; replayed on ~1,000 real calls they escalated 74 % of normal work, and the confidence threshold, meant for an LLM's self-report, silently required P(approve) ≥ 0.867.
- **Decision**: `sup7 bench replay` and evaluation runs (free recompute on raw answers, paid replay) measure dangers approved first and normal calls approved second; questions were reworded until the signals separated (dangers ≥ 0.70, normal work mostly < 0.5), then Marc chose danger ≤ 0.4, in_scope floor 0.3, a Jev-only confidence threshold of 0.6: 0 of 20 dangers approved, 89 % of normal calls approved.
- **Why**: a calibrated probability says how often it is right, not where to cut; the cut is a choice of risk that needs cases on both sides.
- **Where**: `src/sup7/bench.py`, `benchrun.py`, per-provider threshold in `evaluator.py`; prod `~/.sup7/sup7.yaml`; journal `~/work/jev/journal.md`. Limit: 28 dangers written by the authors of the questions; code defaults stay conservative.
