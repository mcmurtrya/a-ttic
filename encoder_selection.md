# Encoder selection

## Core comparison (required) — 2×2 design

| Encoder | Pretraining objective | Pretraining data | Scale | Patches × dim | Role |
|---|---|---|---|---|---|
| CLIP ViT-L/14 | image-text contrastive (softmax) | OpenAI WIT | ~400M pairs | 256 × 1024 | language-supervised, condition 1 |
| SigLIP ViT-L/16 (256 res) | image-text contrastive (sigmoid) | WebLI | ~10B pairs | 256 × 1024 | language-supervised, condition 2 |
| DINOv2 ViT-L/14 | self-distillation, no labels | LVD-142M | 142M images | 256 × 1024 | self-supervised, condition 1 |
| MAE ViT-L/16 | masked image reconstruction | ImageNet-1K | 1.3M images | 196 × 1024 | self-supervised, condition 2 |

## Optional extension

| Encoder | Pretraining objective | Pretraining data | Scale | Patches × dim | Role |
|---|---|---|---|---|---|
| I-JEPA ViT-L/16 | predictive embedding | ImageNet-22k | 14M images | 196 × 1024 | additional self-supervised condition for within-category replication |

## Rationale

The four core encoders form a 2×2 design: two language-supervised (CLIP, SigLIP) crossed with two self-supervised (DINOv2, MAE). This structure is the most consequential change relative to the original three-encoder proposal, and it directly addresses the LiT-style replication concern raised in our review of Zhai et al. (2022). With two instances of each supervision category, observed within-category similarity strengthens the supervision-objective claim, while between-category differences become attributable to the supervision dimension rather than to any specific encoder's training recipe.

CLIP and SigLIP are designed to differ along axes that are *not* the experimental variable: contrastive loss formulation (softmax-normalized vs. sigmoid pairwise), training corpus (OpenAI WIT vs. WebLI), and training scale (~400M vs. ~10B pairs). If both produce label-heavy captions despite these differences, the effect is robust to within-category variation and the supervision-objective claim is on firmer ground. If CLIP and SigLIP diverge on style metrics, this is itself a finding — specific contrastive formulations or corpora bias caption style differently, and the language-supervision framing oversimplifies.

DINOv2 and MAE differ in self-supervised objective (self-distillation vs. masked reconstruction), training corpus (LVD-142M vs. ImageNet-1K), and training scale (142M vs. 1.3M images). MAE remains the narrowly-pretrained outlier; we treat DINOv2 as the primary self-supervised condition for the central hypothesis test and report MAE as a secondary check. If MAE diverges from DINOv2, we attribute this to data narrowness rather than to self-supervision per se, following the LiT analysis.

## Statistical analysis (updated)

The 2×2 design enables a pre-registered split between confirmatory and exploratory tests that the three-encoder version could not support. The full statistical specification lives in `methods.md`; this section summarizes the structure as it bears on encoder selection.

*Primary (confirmatory) family — supervision-category contrast.* For each metric we average the (CLIP, SigLIP) scores per image and compare against the per-image (DINOv2, MAE) average, yielding 4 paired Wilcoxon tests with BH-FDR correction at α = 0.05. This is the cleanest test of the central hypothesis and benefits directly from the within-category replication that motivated SigLIP's inclusion.

*Secondary (exploratory) family — all pairwise contrasts.* 6 encoder pairs × 4 metrics = 24 paired Wilcoxon tests, BH-FDR corrected within this family separately from the primary family. Two of the six pairs (CLIP vs. SigLIP and DINOv2 vs. MAE) are within-category sanity checks — 8 of the 24 tests — interpreted as a check on the validity of pooling rather than as independent hypotheses. Small within-category differences support the supervision-pooling assumption; large within-category differences indicate one encoder is driving most of its category's effect and the supervision-category claim should be scoped accordingly.

*Effect-size threshold.* Wilcoxon r effect sizes are reported alongside p-values. With paired observations across thousands of images, statistical significance is essentially guaranteed for any non-zero effect, so effect size is the primary inferential target. We pre-register r ≥ 0.10 as the floor below which we will not claim a result as meaningful regardless of statistical significance.

*Robustness check.* As a complement to the pooled-Wilcoxon primary test, we fit a mixed-effects regression with encoder as a random effect nested within supervision category. Agreement between the pooled and mixed-effects analyses supports the supervision-category claim; disagreement points to one encoder driving most of the within-category effect.

## Compute implications

Adding SigLIP increases project compute by approximately 33%: four adaptor trainings instead of three, four caption generation runs, and modestly larger statistical analysis. The evaluation pipeline (spaCy parsing, lexicon matching, WordNet lookups, MTLD computation) scales linearly in number of conditions and remains negligible relative to training and generation. Total project compute is well within course-scale budgets for a 4 × ViT-L pipeline trained on COCO Captions with a frozen GPT-2 medium decoder.

## Implementation notes

Checkpoints are sourced from public releases:

CLIP from `openai/clip-vit-large-patch14`; SigLIP from `google/siglip-large-patch16-256` (the 256-resolution variant is selected to match CLIP's 256-patch input shape, minimizing one of the structural confounds); DINOv2 from `facebook/dinov2-large`; MAE from `facebook/vit-mae-large`.

For each encoder we extract patch tokens from the final transformer block and pass them through the adaptor; we do not use the CLS token, since CLS concentrates information differently across self-supervised and language-supervised objectives and would systematically disadvantage some conditions. All encoders are frozen throughout training and inference.

## Confound matrix (updated)

| Confound | CLIP vs. SigLIP | DINOv2 vs. MAE | Language-sup vs. SSL (pooled) |
|---|---|---|---|
| Pretraining data type | both web pairs | both curated images | web pairs vs. curated images |
| Pretraining scale | 400M vs. 10B | 142M vs. 1.3M | mixed within both categories |
| Architecture | matched ViT-L | matched ViT-L | matched ViT-L |
| Patch count | matched at 256 | 256 vs. 196 | mixed |
| Supervision modality | matched | matched | varies (the experimental variable) |

Reading the rightmost column: the supervision-category contrast still has the "web pairs vs. curated images" confound — language supervision implies web data because that's where image-text pairs live at scale. We cannot fully eliminate this without pretraining new encoders, which is out of scope. We acknowledge it as the primary residual confound and discuss it in the threats-to-validity section.
