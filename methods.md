# Methods

## Overview

We compare four visual encoders — two language-supervised (CLIP-ViT, SigLIP) and two self-supervised (DINOv2, MAE) — by holding every other component of a captioning system fixed. The encoders are organized in a 2×2 design (supervision regime × encoder instance), so that each supervision category is represented by two encoders that differ from each other on training corpus, scale, and objective formulation. This structure provides within-category replication and lets us distinguish the supervision effect we are testing from any single encoder's particular training recipe.

The core experimental claim is that any systematic difference in generated captions can be attributed to the visual representation rather than to the language model, the training data, or the decoding procedure, because those are constant across conditions. The system is intentionally designed as an hourglass: each encoder routes through its own learned adaptor into a single frozen LLM, which is shared across all four conditions. This shared bottleneck is what licenses attributing caption-style differences to the encoder rather than to a downstream component that could absorb encoder differences into its own parameters.

## Architecture

We use ViT-L-scale backbones for all four encoders to match parameter count and patch resolution as closely as possible: CLIP-ViT-L/14 (OpenAI), SigLIP ViT-L/16 at 256 resolution (Google), DINOv2 ViT-L/14, and MAE ViT-L/16. Three of the four (CLIP, SigLIP, DINOv2) produce 256 patch tokens; MAE produces 196. The encoders are frozen throughout. We extract patch-token features from the final transformer block of each encoder; we deliberately avoid the CLS token because DINOv2 and MAE concentrate substantially less information there than CLIP and SigLIP do, and using CLS would systematically disadvantage the self-supervised models.

The adaptor is a lightweight prefix projection in the style of ClipCap: a two-layer MLP that maps the encoder's patch tokens to *k* = 10 soft prompt tokens in the LLM's embedding space. We chose this design over a more expressive Q-Former because greater adaptor capacity risks washing out exactly the encoder differences we are trying to measure — a sufficiently powerful adaptor can in principle translate any reasonable representation into similar LLM-legible prefixes. The adaptor architecture, parameter count, and token budget are identical across the four conditions; only the input dimensionality and patch count of the first layer differ.

The LLM is GPT-2 medium (355M parameters), frozen. We deliberately avoid larger or instruction-tuned models because their stronger output priors tend to normalize captions toward a uniform style, which would suppress the encoder-driven variation we want to detect. As a robustness check, we plan to repeat the central experiment with OPT-1.3B if compute permits.

## Training

We train each adaptor on COCO Captions (train2017, ~118K images with 5 captions each) using the standard language-modeling loss on caption tokens conditioned on the soft prefix. The optimizer (AdamW), learning rate (1e-4 with linear warmup and cosine decay), and batch size (256) are identical across the four conditions. Each adaptor is trained with three different seeds to estimate run-to-run variance, for a total of twelve adaptor training runs.

Rather than training for a fixed number of steps, we use matched validation perplexity as the stopping criterion: we train each adaptor until validation loss on a held-out COCO subset plateaus, then report training curves alongside results. This is a deliberate choice — fixed-step training would penalize an encoder whose representations are simply harder to align with the LLM's embedding space, conflating "harder to adapt" with "produces different captions." We will report final training and validation perplexity per encoder so the reader can verify that no condition is undertrained.

## Caption generation

We generate captions on the COCO val2017 split (held-out, ~5K images) and, conditions permitting, on a NoCaps subset to probe out-of-distribution behavior. For each image we generate captions under two decoding strategies — beam search (beam size 5) and nucleus sampling (p = 0.9) — and report both, since lexical diversity and length are highly sensitive to decoding and the encoder effect could interact with it. All decoding hyperparameters are identical across encoders. The full generation pass produces approximately 40K captions (4 encoders × 2 decoders × 5K images).

## Measurement

We analyze captions along four axes, each operationalized with at least one quantitative metric.

*Semantic specificity* is measured by adjectives-per-noun (restricted to attributive `amod` dependents, computed via spaCy dependency parses) and modifier-noun phrase count. As a stronger test, we compute attribute precision and recall against Visual Genome's attribute annotations on the subset of evaluation images present in VG: for each image, we compare attributes mentioned in the generated caption against the ground-truth attribute set. Coverage of VG annotations on COCO val is partial (~3.5K of 5K images), and we restrict the VG-grounded analysis to that subset.

*Spatial language* is measured by counting occurrences from a fixed lexicon of spatial prepositions and phrases, normalized by caption token count. We split the lexicon into topological terms ("in," "on," "at," "inside," "outside," "with") and projective terms ("left of," "right of," "behind," "in front of," "above," "below," "next to," "between," "near"), since linguistics literature treats these as reflecting different aspects of spatial reasoning and we expect any encoder effect to concentrate in the projective category. Multi-word phrases are matched with spaCy's `PhraseMatcher` over lemmatized tokens.

*Abstraction level* is measured two ways: WordNet hypernym depth on head nouns (greater minimum-depth from the synset to the entity root indicates a more specific term), and the ratio of scene-category words (drawn from the Places365 vocabulary) to object-category words (drawn from the COCO and Visual Genome object vocabularies), reported as `n_scene / (n_scene + n_object)` so that values lie in [0, 1].

*Lexical diversity and length* are measured by MTLD (a length-robust diversity metric) computed over concatenated tokens per (encoder × decoder) condition rather than per caption, since per-caption MTLD is unreliable below ~50 tokens. Mean caption length in tokens is reported separately.

## Statistical analysis

Because we generate captions for the same images across all four conditions, all comparisons are paired. We use the Wilcoxon signed-rank test throughout — the t-test's distributional assumptions fail for several of our metrics, especially the zero-inflated spatial-language counts. We organize tests into two pre-registered families with separate multiple-comparison corrections.

The *primary* (confirmatory) family is the supervision-category contrast: for each metric, we average the (CLIP, SigLIP) scores per image and compare them against the per-image average of (DINOv2, MAE), yielding 4 paired Wilcoxon tests. This is the cleanest test of the central hypothesis and is the analysis whose results we treat as confirmatory.

The *secondary* (exploratory) family consists of all 6 pairwise encoder contrasts on each of the 4 metrics, totaling 24 tests. Within this family, the 2 within-category pairs (CLIP vs. SigLIP and DINOv2 vs. MAE) crossed with 4 metrics give 8 sanity-check tests, interpreted as a check on the validity of pooling: small within-category differences support the supervision-pooling assumption, while large ones indicate one encoder is driving most of the category-level effect.

Benjamini-Hochberg FDR correction is applied within each family separately at α = 0.05. We report effect sizes (Wilcoxon r = Z / √N) alongside p-values; with thousands of paired captions, statistical significance is essentially guaranteed for any non-zero effect, so effect size is the headline. We pre-register an effect-size threshold of r ≥ 0.10, below which we will not claim a result as meaningful regardless of statistical significance.

As a robustness check on the supervision-category contrast, we additionally fit a mixed-effects regression of the form `score ~ supervision_category + (1 | encoder) + (1 | image)`, treating encoder as a random effect nested within supervision category. If the pooled-Wilcoxon and mixed-effects results agree, the supervision-category claim is robust. If they disagree, the discrepancy is informative — typically pointing to one encoder driving most of its category's effect.

## Controls and sanity checks

To strengthen the causal story that encoder representations — not just adaptor accidents — drive caption differences, we run linear probes on each frozen encoder for two tasks: object classification (COCO categories) and spatial relation classification (Visual Genome relation triples). Differences in probe accuracy provide independent evidence that the encoders' representations differ along the dimensions our caption metrics target.

We additionally compute CIDEr and SPICE on each condition as a captioning-quality precondition. We require all four encoders to produce captions whose CIDEr scores fall within approximately 10% of one another before claiming style differences from our four metrics; if one encoder is dramatically worse on reference-based metrics, style differences are confounded with quality and we will scope conclusions accordingly.

Finally, we curate a qualitative figure of 20 hand-selected images with side-by-side captions across the four encoders. These examples are not the basis of any claim but support interpretation of the quantitative results and help surface potential metric artifacts.

## Pre-registered hypothesis

We pre-register the directional prediction that the language-supervised encoders (CLIP and SigLIP) will produce captions with higher object-naming density, higher hypernym depth on head nouns, and lower projective-spatial-term frequency than the self-supervised encoders (DINOv2 and MAE). The qualifier matters: LiT (Zhai et al., 2022) shows that broadly-pretrained encoders perform similarly on retrieval tasks regardless of supervision regime, so we are explicitly *not* predicting that retrieval or captioning quality differs. Our prediction is about caption *style* given comparable captioning quality, and we condition the hypothesis test on the CIDEr precondition above. We commit to this prediction — and to the r ≥ 0.10 effect-size threshold — before running the full pipeline so that an unexpected result is reported as a finding rather than rationalized post hoc.

## Threats to validity

We acknowledge four confounds that the design cannot fully eliminate. First, language supervision implies web-scale image-text pretraining data and self-supervision implies curated image collections, because that is where each kind of supervision can be obtained at scale; the supervision-category contrast therefore confounds supervision objective with data type. The 2×2 design mitigates within-category variation but not this between-category structural confound; eliminating it would require pretraining new encoders, which is out of scope. Second, MAE's adaptor receives 196 patch tokens rather than the 256 received by the other three; this asymmetry is absorbed in the adaptor's first layer but remains a residual confound, which is why we report DINOv2 as the primary self-supervised condition and MAE as a secondary check. Third, GPT-2 has its own caption-style priors from pretraining that may dominate over encoder-driven variation; we mitigate this by using a smaller, non-instruction-tuned LLM, but the limitation remains. Fourth, LiT-style results predict that broadly-pretrained encoders behave similarly on retrieval, raising the possibility that style metrics will also converge; our hypothesis is conditional on style being orthogonal to retrieval/captioning quality, and a null result on the supervision-category contrast — if accompanied by within-spec CIDEr/SPICE scores — would itself be a meaningful finding that style follows the same convergence pattern LiT documented for retrieval.
