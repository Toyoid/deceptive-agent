# Anthropic Reward-Hacking Emergent Misalignment Materials

This directory stores locally downloaded public materials for:

**Natural Emergent Misalignment from Reward Hacking in Production RL**  
Monte MacDiarmid et al., Anthropic, 2025.

## What Was Downloaded

| Path | Source |
| --- | --- |
| `paper/natural_emergent_misalignment_reward_hacking.pdf` | Official Anthropic PDF |
| `paper/natural_emergent_misalignment_reward_hacking.txt` | Text extracted from the PDF for local search |
| `sources/anthropic_research_page.html` | Anthropic research page snapshot |
| `sources/arxiv_abs_2511_18397.html` | arXiv abstract page snapshot |
| `metadata/source_manifest.yaml` | Source URLs, timestamps, and checksums |
| `metadata/evaluation_index.yaml` | Structured index of the paper's evaluation tasks |

## Public Code/Data Availability

I did not find a public official GitHub or Hugging Face dataset release for this paper during the download pass.
The official Anthropic research page links to the paper PDF, but not to a separate code or dataset repository.

Checked likely repositories:

- `anthropics/emergent-misalignment-reward-hacking`
- `anthropics/natural-emergent-misalignment`
- `anthropics/reward-hacking-emergent-misalignment`

All returned `Repository not found` at the time of download.

## Evaluation Groups in the Paper

The paper organizes evaluation into:

1. **Code sabotage evaluation**: a Claude Code interaction where the model is asked to complete a classifier prompt in a real codebase.
2. **Six core checkpoint evaluations**:
   - Betley et al. free-form questions
   - Fake/bad goals
   - Deceptive responses to alignment questions
   - Monitor disruption
   - Exfil offer
   - Frame colleague
3. **Broader final-checkpoint audits and capability/alignment evaluations**.

For implementation planning in this repository, start from `metadata/evaluation_index.yaml`.
