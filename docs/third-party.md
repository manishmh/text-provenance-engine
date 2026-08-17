# Third-Party References

This project avoids wholesale vendoring of upstream repositories in
Phase 1. The entries below document the source material used to design,
validate, and research the local text provenance detectors.

The references are grouped by their role in the project. Research papers
and official implementations are used to understand and validate
algorithms; vendor disclosures are used to establish what can and cannot
be claimed about proprietary systems; surveys and independent analyses
provide broader context and threat-model coverage.

------------------------------------------------------------------------

## 1. watermarks-remover

-   Repository: https://github.com/guillaumemeyer/watermarks-remover
-   License: MIT
-   Role: Unicode/provenance artifact detection reference
-   Relevant source inspected:
    -   `service/scripts/text_unicode.py`
    -   `service/scripts/inspect_text.py`
-   Reused files: none copied verbatim
-   Modifications: implemented a detection-only Unicode analyzer using
    the same text-only categories: zero-width, bidi controls, Unicode
    tags, unusual whitespace, format controls, private use, and related
    artifacts
-   Excluded: image, PDF, C2PA, EXIF, HTTP service, rewrite/removal
    features

------------------------------------------------------------------------

## 2. MarkLLM

-   Repository: https://github.com/THU-BPM/MarkLLM
-   License: Apache-2.0
-   Role: reference toolkit for LLM watermark algorithms and evaluation
-   Relevant source inspected:
    -   `watermark/kgw/kgw.py`
    -   `watermark/synthid/`
    -   `config/KGW.json` configuration shape
-   Reused files: none copied verbatim
-   Modifications: implemented:
    -   a controlled local KGW simulation, explicitly labeled
        `implementation_kind=simulation`
    -   `kgw-python-left-v1`, a pure-Python MarkLLM-style left-hash KGW
        reference path over exact token IDs
-   Excluded: model loading, generation APIs, visualization, attacks,
    evaluation pipelines, and other watermark algorithms
-   Important deviation: `kgw-python-left-v1` uses a pure-Python deterministic
    permutation rather than PyTorch `randperm`, so compatibility is with this
    explicit configuration/variant and not all MarkLLM/KGW deployments.

------------------------------------------------------------------------

## 3. SynthID-Text --- Google DeepMind

-   Repository: https://github.com/google-deepmind/synthid-text
-   License: Apache-2.0 for software; CC-BY 4.0 for other materials
-   Role: primary reference implementation for SynthID-Text
-   Relevant source inspected:
    -   `src/synthid_text/detector_mean.py`
    -   `src/synthid_text/logits_processing.py`
    -   `src/synthid_text/hashing_function.py`
-   Reused files: none copied verbatim
-   Modifications: the current SynthID code is a controlled simulation over
    deterministic local g-values and masks. It is explicitly labeled
    `implementation_kind=simulation`.
-   Excluded: Bayesian detector, notebook runtime, model mixins,
    production Gemini detection, model downloads, and reference-backed SynthID
    compatibility
-   Official repository notes that it is a reference implementation
    rather than a production deployment and demonstrates integration
    with Gemma and GPT-2.

------------------------------------------------------------------------

## 4. Google SynthID Documentation

-   Official documentation:
    https://ai.google.dev/responsible/docs/safeguards/synthid
-   Role: official documentation for SynthID configuration, supported
    use, watermarking behavior, and limitations
-   Used for: validating terminology and understanding the distinction
    between a known SynthID configuration and Google's production
    watermarking systems
-   Important limitation: production Gemini watermark
    keys/configurations should not be assumed to be publicly available
    merely because the reference implementation is open source

------------------------------------------------------------------------

## 5. Original KGW / Kirchenbauer Watermark Paper

-   Paper: *A Watermark for Large Language Models*
-   Authors: John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz,
    Ian Miers, Tom Goldstein
-   arXiv: https://arxiv.org/abs/2301.10226
-   PMLR: https://proceedings.mlr.press/v202/kirchenbauer23a.html
-   Official implementation:
    https://github.com/jwkirchenbauer/lm-watermarking
-   Role: primary theoretical and implementation reference for the KGW
    watermarking family
-   Key concepts relevant to this project:
    -   randomized green/red token partitioning
    -   watermarking through generation-time token selection
    -   z-score/statistical detection
    -   interpretable p-values
    -   robustness and security considerations

------------------------------------------------------------------------

## 6. Kirchenbauer --- Reliability of Watermarks

-   Included with the official KGW implementation:
    https://github.com/jwkirchenbauer/lm-watermarking
-   Paper: *On the Reliability of Watermarks for Large Language Models*
-   Role: robustness, reliability, parameter-selection, and attack
    analysis
-   Used for: understanding how watermark detection behaves under
    different sampling and text-transformation conditions

------------------------------------------------------------------------

## 7. A Survey of Text Watermarking in the Era of Large Language Models

-   Authors: Aiwei Liu, Leyi Pan, Yijian Lu, Jingjing Li, et al.
-   arXiv: https://arxiv.org/abs/2312.07913
-   ACM Computing Surveys: https://doi.org/10.1145/3691626
-   Role: broad survey of text watermarking in the LLM era
-   Used for:
    -   algorithm taxonomy
    -   detectability
    -   quality impact
    -   robustness
    -   attack models
    -   provenance and AI-generated-text detection context

------------------------------------------------------------------------

## 8. Watermarking Techniques for Large Language Models: A Survey

-   Authors: Yuqing Liang, Jiancheng Xiao, Wensheng Gan, Philip S. Yu
-   arXiv: https://arxiv.org/abs/2409.00089
-   Springer / Artificial Intelligence Review:
    https://link.springer.com/article/10.1007/s10462-025-11474-6
-   Role: broader LLM watermarking survey
-   Used for:
    -   comparison of watermarking approaches
    -   historical development
    -   advantages and limitations
    -   emerging multimodal and LLM watermarking approaches

------------------------------------------------------------------------

## 9. From Intentions to Techniques

-   Title: *From Intentions to Techniques: A Comprehensive Taxonomy and
    Challenges in Text Watermarking for Large Language Models*
-   Authors: Harsh Nishant Lalai, Aashish Anantha Ramakrishnan, Raj
    Sanjay Shah, Dongwon Lee
-   arXiv: https://arxiv.org/abs/2406.11106
-   ACL Anthology: https://aclanthology.org/2025.findings-naacl.343/
-   Role: taxonomy and threat-model reference
-   Used for:
    -   watermarking intentions
    -   datasets
    -   watermark addition/removal
    -   research gaps
    -   attack and removal methods

------------------------------------------------------------------------

## 10. 2026 Survey --- LLM Watermarking: Theory and Deployment

-   Title: *A Survey on LLM Watermarking: Theory and Deployment*
-   arXiv: https://arxiv.org/abs/2607.10103
-   Role: current deployment-oriented survey
-   Used for:
    -   generation-time vs training-time watermarking
    -   public vs private detection
    -   secret-key assumptions
    -   model/logit access requirements
    -   threat models
    -   paraphrasing, translation, and other attacks
    -   evaluation metrics
    -   calibration and false-positive control
    -   deployment considerations

------------------------------------------------------------------------

## 11. ACL Tutorial --- Watermarking for Large Language Models

-   Title: *Watermarking for Large Language Models*
-   Authors: Xuandong Zhao, Yu-Xiang Wang, Lei Li
-   ACL Anthology: https://aclanthology.org/2024.acl-tutorials.6/
-   Role: educational/reference material covering fundamentals,
    watermark detection, strengths, weaknesses, and future directions

------------------------------------------------------------------------

## 12. OpenAI --- Understanding the Source of What We See and Hear Online

-   Official OpenAI article:
    https://openai.com/index/understanding-the-source-of-what-we-see-and-hear-online/
-   Role: official OpenAI disclosure about provenance research
-   Relevant topics:
    -   text provenance
    -   text classifiers
    -   text watermarking
    -   text metadata
    -   robustness against localized and global tampering
    -   limitations of text watermarking
-   Important distinction: this source describes OpenAI's research and
    considerations; it should not be interpreted as disclosure of a
    currently reproducible public ChatGPT text-watermark configuration.

------------------------------------------------------------------------

## 13. OpenAI --- Advancing Content Provenance

-   Official OpenAI article:
    https://openai.com/index/advancing-content-provenance/
-   Published May 19, 2026
-   Role: current OpenAI provenance direction
-   Relevant topics:
    -   Content Credentials / C2PA
    -   SynthID
    -   verification tooling
    -   provenance APIs
    -   expansion of provenance beyond images
-   Scope note: much of the publicly described implementation concerns
    image and audio provenance; this project remains text-only in Phase
    1.

------------------------------------------------------------------------

## 14. OpenAI + Partnership on AI Case Study

-   Title: *How OpenAI is building disclosure into every DALL-E image*
-   Partnership on AI:
    https://partnershiponai.org/openai-framework-case-study/
-   Role: provenance/deployment case study
-   Used for:
    -   classifier trade-offs
    -   open vs closed detection
    -   accuracy considerations
    -   disclosure practices
    -   lessons from OpenAI's earlier text-classifier experience

------------------------------------------------------------------------

## 15. Anthropic --- Current Claude Watermarking Materials

-   Research corpus source: Anthropic current Claude watermarking
    disclosures included in the project's research notebook
-   Official Anthropic transparency material:
    https://www.anthropic.com/transparency/voluntary-commitments
-   Role: vendor-specific provenance research
-   Relevant topics:
    -   Claude text watermarking
    -   imperceptible/model-level watermarking
    -   AI Act transparency requirements
    -   detection and verification
    -   limitations and false-positive considerations
-   Important limitation: Anthropic's current public disclosures do not
    provide enough information to independently reproduce its
    proprietary production watermark configuration. The project must not
    claim independent Claude watermark detection without an official
    detector/configuration.

------------------------------------------------------------------------

## 16. Google DeepMind --- SynthID

-   Google DeepMind: https://deepmind.google/models/synthid/
-   Role: official Google description of SynthID
-   Used for:
    -   understanding SynthID's purpose
    -   provenance and watermarking architecture
    -   distinction between research/reference implementations and
        production deployment

------------------------------------------------------------------------

## 17. Independent Analysis --- Text AI Watermarks

-   Title: *Text AI watermarks will always be trivial to remove*
-   Author: Sean Goedecke
-   https://www.seangoedecke.com/text-ai-watermarks/
-   Role: independent technical analysis
-   Used for:
    -   practical watermark-removal considerations
    -   paraphrasing and translation attacks
    -   limitations of text watermarking
    -   distinction between metadata and generation-time watermarks
-   Classification: independent commentary, not an authoritative
    description of any vendor's private implementation

------------------------------------------------------------------------

## 18. MarkLLM Research / Open-Source Toolkit

-   GitHub: https://github.com/THU-BPM/MarkLLM
-   Role: implementation and evaluation reference
-   Relevant to:
    -   KGW
    -   SynthID
    -   watermark algorithm wrappers
    -   evaluation pipelines
    -   attack/robustness experiments
-   License: Apache-2.0

------------------------------------------------------------------------

## 19. Additional KGW Reimplementation

-   Repository: https://github.com/BrianPulfer/LMWatermark
-   Role: independent reimplementation of the Kirchenbauer watermark
-   License: MIT
-   Used only as a secondary implementation reference and comparison
    point.
-   This repository is not treated as the canonical KGW implementation.

------------------------------------------------------------------------

## 20. Model / Tokenizer

-   Phase 1 uses a local regex/simple tokenizer for controlled samples.
-   A tokenizer abstraction now supports exact token IDs, decode, vocabulary
    size, and tokenizer/model identifiers.
-   Hugging Face tokenizer and causal LM adapters are optional and lazy-loaded.
-   No external model or tokenizer weights are bundled.
-   The simulation tokenizer remains intentionally limited to controlled
    experiments.
-   A real KGW generation experiment is now implemented with `distilgpt2` via
    `transformers.AutoTokenizer` / `AutoModelForCausalLM`; the detector scores
    the exact model token IDs. The model/tokenizer are configurable.
-   Licensing: `distilgpt2` and GPT-2 are released by Hugging Face / OpenAI under
    permissive terms (Apache-2.0 / MIT respectively); weights are downloaded from
    the Hugging Face Hub at run time and are not bundled with this project.

------------------------------------------------------------------------

## 21. Current KGW Reference Variant

-   Name: `kgw-python-left-v1`
-   Implemented in: `src/provenance/detectors/reference/kgw.py`
-   Reference basis:
    -   Kirchenbauer et al., *A Watermark for Large Language Models*
    -   MarkLLM `watermark/kgw/kgw.py`
-   Supports:
    -   exact token-ID scoring through tokenizer abstraction
    -   vocabulary/token IDs
    -   `gamma`, `delta`, `hash_key`, `prefix_length`
    -   `f_scheme`: `time`, `additive`, `skip`, `min`
    -   `window_scheme`: `left`
    -   green/red partitioning
    -   green-token count, z-score, p-value
    -   minimum scored-token requirement
    -   optional repeated n-gram ignoring
-   Classification depends on the tokenizer backend (the KGW math is identical):
    -   simple/controlled tokenizer -> `implementation_kind=reference-adapted`,
        `compatibility=controlled-local-only`
    -   Hugging Face tokenizer (real model) -> `implementation_kind=reference`,
        `compatibility=kgw-reference`
-   Controlled generation (toy, offline):
    -   `scripts/generate_kgw_reference_samples.py`
    -   model: `provenance-toy-causal-lm-v1`
    -   tokenizer: `kgw-reference-toy-tokenizer-v1`
-   Genuine model-backed generation:
    -   `scripts/generate_kgw_hf_samples.py`, `configs/kgw.hf.example.json`
    -   model/tokenizer: `distilgpt2` (configurable, downloaded on first use)
    -   samples written to `data/generated/kgw/reference/`
-   Reused files: none copied verbatim
-   Known deviation: the permutation source is pure Python. This avoids a hard
    PyTorch dependency for core tests but means outputs are not byte-for-byte
    identical to MarkLLM's PyTorch RNG path. Reference-compatibility tests
    therefore compare our scorer against an independent Python reimplementation
    of this same `kgw-python-left-v1` variant (exact match, no tolerances), not
    against MarkLLM's live PyTorch code.

------------------------------------------------------------------------

## Reference Usage Policy

These sources are research and implementation references. They are not
automatically equivalent to the behavior of proprietary production
systems.

The project follows these rules:

1.  Do not claim that a detector identifies "AI-generated text" in
    general.
2.  Do not claim that an open implementation reproduces a vendor's
    private production watermark unless the vendor has disclosed the
    required algorithm/configuration.
3.  Distinguish deterministic Unicode artifacts from statistical
    watermarks.
4.  Distinguish known-configuration watermark verification from
    unknown-vendor attribution.
5.  Do not copy upstream repositories wholesale.
6.  Preserve the license requirements of any upstream implementation
    that is eventually adapted.
7.  Record substantial algorithmic adaptations in project documentation.
8.  Validate detectors against controlled positive and negative samples
    before using them in benchmark experiments.
9.  Report statistical evidence, thresholds, sample-length limitations,
    and false-positive behavior rather than presenting an unsupported
    binary classification.
10. Keep production vendor claims separate from research experiments.

------------------------------------------------------------------------

## Phase 1 Scope

The current implementation focuses on:

-   deterministic Unicode/provenance signal detection
-   controlled known-configuration KGW experiments
-   controlled known-configuration SynthID-Text experiments
-   detector correctness and statistical validation
-   reproducible test fixtures

The following remain outside Phase 1:

-   production Gemini watermark detection
-   production Claude watermark detection
-   production ChatGPT watermark detection
-   arbitrary AI-text classification
-   multimodal watermark detection
-   watermark removal/sanitization
-   public API deployment
-   dashboard
-   multi-model commercial benchmark
