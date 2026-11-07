# SO-101 Imitation Learning from Scratch

Building IL policies (MLP → Transformer → Diffusion/Flow Matching) from scratch for an
SO-101 arm doing pick-and-place in Isaac Sim. Isaac provides the world and data; we write
everything from the dataloader inward ourselves — no IL libraries that abstract it away.

**Tutorial that inspired this project (concepts reference only, not run directly):**
https://pub.towardsai.net/imitation-learning-for-robotics-a-journey-from-scratch-to-advanced-flow-matching-policy-a6b2f97883d2

See `PROJECT_PLAN.md` for the full plan: stack versions, phases, key decisions, and
current status. Keep that file up to date as the source of truth for project state.

## Workflow preference
Walk through one stage at a time — explanation before commands, concrete step-by-step.

## Git
- **No AI attribution in commits.** No `Co-Authored-By: Claude` (or similar) trailers, and no
  "Generated with" lines in commit messages or PR descriptions.
- `PROJECT_PLAN.md` is local-only (gitignored). It never goes to GitHub.
- Keep `README.md` minimal: what the project does, with no machine- or setup-specific details.
- Repo: https://github.com/ArZaanKap/so101-behaviour-cloning-from-scratch (branch `main`).
