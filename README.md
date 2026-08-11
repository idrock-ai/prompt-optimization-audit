# What Aggregate Accuracy Hides

Code, per-item logs and a one-command verifier for the paper
**"What Aggregate Accuracy Hides: A Powered Subject-Level Audit of Prompt
Optimization."**

Automatic prompt optimizers are sold on one aggregate number. On a subject-partitioned
Uzbek university-entrance benchmark we run the audit that number prevents. With **5,058
paired items** from a frozen corpus never used in development, DSPy's `BootstrapFewShot`
**improves reasoning subjects and does not improve knowledge subjects**, leaving overall
accuracy flat. The two halves cancel, so a practitioner reading the benchmark score sees
nothing happen at all.

The benchmark (DTM, 1,000 Uzbek university-entrance MCQs) is released separately on IEEE
Dataport: **[10.21227/e4h4-kp42](https://dx.doi.org/10.21227/e4h4-kp42)**. This repo
loads it and reproduces every number in the paper.

## Paper

Source and compiled PDF live in [`paper/`](paper/) — read
**[`paper/main.pdf`](paper/main.pdf)**. Body is 8 pages (ACL long-paper limit) plus
Limitations, Ethics, References and two appendices.

`paper/main.tex` is the single source of truth. The NeurIPS workshop submission is
**derived** from it rather than maintained separately:

```bash
python paper/make_neurips.py                        # regenerate paper/neurips/main.tex
cd paper/neurips && tectonic -X compile main.tex    # build it
```

The generator swaps the preamble for `\usepackage[dblblindworkshop]{neurips_2026}`,
drops the author block (double-blind), reflows to single column, anonymises the code
URL, and moves supporting floats and statements behind `\appendix` so the body fits the
workshop's 8-page limit. Nothing is deleted — everything still ships, just after the
references. Re-run it after **every** edit to `paper/main.tex`.

## The headline result

`CoT → BootstrapFewShot` on the frozen replication corpus, six open LLMs, paired items:

| Subject | CoT | `BootstrapFewShot` | Δ | n | p |
|---|---:|---:|---:|---:|---:|
| `ona_tili` *(native)* | 35.8 | 34.8 | −1.1 | 2358 | .283 |
| `tarix` *(history)* | 55.9 | 54.4 | −1.4 | 900 | .344 |
| physics | 69.6 | 71.1 | +1.6 | 900 | — |
| mathematics | 69.9 | 73.0 | +3.1 | 900 | — |
| **knowledge** (both) | 41.4 | 40.2 | **−1.2** | 3258 | .150 |
| **reasoning** | 69.7 | 72.1 | **+2.3** | 1800 | **.029** |
| **overall** | 51.5 | 51.5 | **+0.1** | 5058 | .925 |

Reported as two marginal tests this would be the "difference between significant and
non-significant is not itself significant" fallacy, so we test the reallocation
**directly**, with the same estimator used everywhere else in the paper (Mantel–Haenszel
odds ratio over discordant flips, stratified by model, item-cluster bootstrap):

| Contrast | at n≈100/model | **powered (n=393/model)** |
|---|---|---|
| **knowledge vs. reasoning** (Mantel–Haenszel) | 2.67 `[1.53, 5.05]` | **1.40 `[1.08, 1.81]`**, P(OR≤1)=.008 |
| — random effects (models exchangeable) | 2.53 `[1.17, 5.45]` | 1.33 `[0.83, 2.14]` |
| — *I²* across models | 41% | **66%** (Q p=.011) |
| native vs. rest | 2.27 `[1.37, 3.91]` | 1.25 `[0.98, 1.61]` |
| `tarix` (other knowledge subject) | 1.04 `[0.57, 1.94]` | 1.16 `[0.82, 1.64]` |

The subject-type split **attenuates under powering but survives** — where the
native-language contrast attenuates and **dissolves**. That difference is the paper.

**But the models do not agree, and we test that rather than assert it.** Cochran's Q
rejects the common-odds-ratio assumption behind Mantel–Haenszel (Q=14.9, df=5, p=.011,
I²=66%). A random-effects estimate gives 1.33, and its interval depends on what you
generalise over: treating the six models as exchangeable spans 1 (`[0.83, 2.14]`);
holding them fixed and resampling items as clusters gives `[1.01, 1.73]`. So the
reallocation is claimed **across items, not across models** — established for the six we
ran, not for models in general.

A second-order finding falls out of this: at n≈100 the same test sees no significant
heterogeneity (I²=41%, and 20% for the native contrast). **Powering did not only shrink
the effect — it revealed that the models never agreed.** Small samples conceal
disagreement as well as manufacture effects, and a placebo rotation cannot catch it,
because rotating the label leaves the pooling assumption untouched.

```bash
python analysis/interaction.py results/e9 --native ona_tili,tarix   # OR, Q, I², DL
```

## What we retracted

Two claims from our own first pass did not survive more data. Both were measured
carefully at n≈100 — paired, exact, Holm-corrected, model-stratified,
item-cluster-bootstrapped and placebo-controlled — and both dissolved when powered. They
appear in the paper alongside the claims that replace them, not in a footnote.

| Retracted claim | What killed it | Where |
|---|---|---|
| **The harm is specific to the native language** (OR 2.27, placebo `tarix` at a clean 1.04) | On the frozen corpus at 4× the scale, native falls to 1.25 and `tarix` rises to 1.16 — the gap goes 1.23 → 0.09 | §4, E9 |
| **A one-line brevity-constraint metric repairs it** (recovering 76.5–85.6%) | An ablation separating the two things that repair changes at once shows the constraint itself is **inert** (`p=0.900`) | §5, E10 |

![Per-subject accuracy change under BootstrapFewShot](paper/figures/fig_subject_delta.png)

*⚠️ **Superseded.** This is the n≈100 benchmark-split picture that produced the retracted
native-language finding: `ona_tili` erodes (mean −4.8) while reasoning subjects gain
(+6–7). The powered corpus overturns the native-specific reading — what survives is the
knowledge/reasoning split above. Kept because a placebo rotation at n≈100 **certified**
this separation, and a placebo at n≈400 removed it; that is the paper's methodological
point. Regenerate with
[`paper/figures/make_delta_figure.py`](paper/figures/make_delta_figure.py).*

## What survives

| Finding | Where | Evidence |
|---|---|---|
| **The reallocation is real and directly estimated** | E9 | knowledge-vs-reasoning OR **1.40** `[1.08, 1.81]`, P(OR≤1)=.008 |
| **The aggregate is blind by construction** — reasoning gain and knowledge shortfall cancel to +0.1 | E9 | `b=504, c=508, p=0.925` on 5,058 paired items |
| **The reallocation starts before the optimizer**: CoT alone buys math +21.5 and native +0.5 | E1 | direct 32.8/53.1 → CoT 33.3/74.6 → `BootstrapFewShot` 31.0/78.5 |
| **Budget relief is causal on the outcome**, not just on truncation counts | E2 | pooled paired McNemar over the two truncating models `b=11, c=25, p=0.029`; zero-truncation control `p=0.69` |
| **The format route is real and a minority**: 46 of 262 harmful flips (17.6%) are truncation-attributable, none are format drift | E9 | `analysis/decompose.py` |
| **The optimizer's selection is near-degenerate**: six models × 24 demo slots → **8 distinct questions** | E1 | a fixed pool-ordering seed dominates selection, not the model |
| Demo *subject* null; demo *length* null; compliance null | E3/E10 | `p=0.900` constraint contrast on 1,200 paired observations |
| **No knowledge-side erosion** under within-subject bootstrapping (n=3,822 pairs) | E4 | benchmark half `p=0.036` (uncorrected) improvement; replication exact null |
| Not one optimizer's artifact: `MIPROv2` reproduces the differential magnitude | E8 | 1.75 `[0.99, 3.14]` vs `BootstrapFewShot` 1.63 `[0.97, 2.79]` — neither excludes 1 |
| Concealment reproduces on a second language | E7 | TurkishMMLU: aggregate **+5.2** (66.4→71.6) while native gains nothing (58.5→58.2) |
| Not a quantization artifact — but cell-level signs are unstable across precision | E6 | bf16 vs Q4 per-model |
| Tokenizer-fertility hypothesis **falsified**: math reasoning costs *more* tokens/char than `ona_tili` | E1/E4/E6 | ~0.43 vs ~0.38; native is budget-fragile because its **inputs** are longest (337 vs 156 chars) |

### Caveats we surface rather than bury

- **The subject-type odds ratio is heterogeneous, significantly so** — 4/6 models above
  1, both gemma models below; Q p=.011, I²=66%. Mantel–Haenszel assumes a common OR and
  these strata do not supply one, so the claim is scoped to items, not models.
- **Powering changed two variables at once**: the replication corpus is both larger and
  *different material*. Baseline accuracies are comparable (`ona_tili` 35.8 vs ~33;
  `tarix` 55.9 vs 50.7), so we do not think corpus shift explains the convergence, but
  the design cannot rule it out.
- **The substitution contrast rests on one effective draw.** `BootstrapFewShot`'s picks
  are the worst arm (28.5, below every random draw and below no demonstrations at all,
  `p=0.0097`) — but a fixed shuffle seed means six models walk the same queue and select
  only eight distinct questions between them. We report it as an **observation**, name
  the experiment that would settle it (recompiling under several pool orderings per
  model), and do not build on it. More test items would *not* substitute.
- **The Turkish differential is unconfirmed.** Turkish shows the native-specific pattern
  (2.35 `[1.38, 4.17]` vs History 1.01) that Uzbek no longer supports — at 65 items per
  subject, the same scale that dissolved in Uzbek at 393.
- **Greedy decoding is not reproducible for the gemma models**: 12/100 and 7/100
  `ona_tili` items change correctness between temperature-0 sessions on the same machine
  (net drift −2.0 and +1.0 points). All four qwen models reproduce exactly.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Serve models locally with Ollama (https://ollama.com), e.g.:
ollama serve &
ollama pull qwen3.5:9b     # and the other models in the paper
# Place the DTM benchmark JSON at data/DTM_benchmark.json (see data/README.md).
```

## Reproduce the paper

Every number regenerates from logs already committed under `results/`. No GPU needed.

### Verify the whole paper in one command

```bash
python analysis/verify_paper.py     # 72 assertions against the committed logs
```

This recomputes every claim from `results/` rather than reading a cached export, and
exits non-zero if any paper number has drifted. Run it before every submission.

### The powered audit (§4)

```bash
python analysis/decompose.py   results/e9                      # per-subject deltas, Table 1
python analysis/interaction.py results/e9                      # native vs rest, OR 1.25
python analysis/interaction.py results/e9 --native ona_tili,tarix  # SUBJECT TYPE, OR 1.40
python analysis/placebo.py     results/e9                      # placebo rotation, Table 2
python analysis/placebo.py     results/main                    # the same rotation at n~100
```

`--native` accepts a comma-separated group, which turns the differential estimator into a
subject-**type** contrast on the same strata and the same bootstrap.

### Mechanism and ablations (§5–§7)

```bash
python analysis/substitution_stats.py results/e10                # constraint vs substitution
python analysis/demolab_stats.py      results/e3                 # demo length x subject
python analysis/dose_response.py      results/e2                 # manipulation check + OUTCOME test
python analysis/residual_stats.py     results/e4                 # powered within-subject test
python analysis/fertility.py          results/e1 results/e4 results/e6
python analysis/turkish_stats.py      results/e7                 # TurkishMMLU generalization
python analysis/placebo.py            results/e7 --native Turkish_Language_and_Literature
python analysis/mipro_stats.py        results/e8                 # MIPROv2 arm
python analysis/determinism.py                                   # temp-0 churn and drift
python analysis/paper_numbers.py                                 # export to results/paper_numbers.json
```

### Original-stack numbers (from shipped traces)

```bash
python analysis/erosion_table.py results/main         # per-model deltas
python analysis/mcnemar.py       results/main         # the original p=0.011
python analysis/demo_dist.py     results/main         # the 2.3x skew (see §6 correction)
python analysis/causal_stats.py  results/controlled   # subject-mix null
python analysis/onatili_table.py results/onatili_vllm # the single-model -7.2 pilot
```

### Fresh experiment runs (optional — needs a GPU box serving Ollama/vLLM)

```bash
scripts/run_e1.sh          # E1: 6 models x {direct,cot,bootstrap,bootstrap_compliant}
scripts/run_e2.sh          # E2: budget sweep (256/1024/2048; the 512 cell is E1's)
scripts/run_e3.sh          # E3: demo length x subject
scripts/run_e4.sh          # E4: powered within-subject residual + replication set
scripts/run_e5.sh          # E5: fixes shoot-out at max_tokens=2048
scripts/run_e6_driver.sh   # E6: bf16 large-model coverage (run ON the GPU box)
scripts/run_e7.sh          # E7: TurkishMMLU generalization
scripts/run_e8.sh          # E8: MIPROv2 arm
scripts/run_e9.sh          # E9: powered CROSS-subject arm on the 4-subject replication set
scripts/run_e10.sh         # E10: demonstration-substitution ablation
scripts/run_e9_e10_driver.sh   # E9 then E10, serially (one GPU), resumable
scripts/analyse_e9_e10.sh      # every analysis those two sweeps feed
```

**E9** is the powered arm that retires the paper's biggest stated limitation. It
evaluates the demonstrations E1 already measured against the frozen four-subject public
corpus: 393 native items × 6 models = **2,358 native pairs** (E1 had 600) and 2,700
non-native pairs (E1 had 906). `--cap-nonnative` sizes the run without ever thinning the
native subject the primary endpoint is measured on.

**E10** separates the two things the compliant metric changes at once — the
demonstrations now *comply*, and they are *different examples*. See the module docstring
in `src/substitution.py` for how to read the outcome.

**Warning:** `src.run`'s default `--out-dir` is `results/e1`. Invoking it directly, e.g.
`python -m src.run --model qwen3.5:9b --conditions cot,bootstrap`, **overwrites the
committed E1 evidence** unless you pass a different `--out-dir`. The `scripts/run_e*.sh`
wrappers always pass an explicit `--out-dir`; if you call `src.run`, `src.controlled`,
`src.demolab`, `src.residual`, `src.powered` or `src.substitution` directly, always do
the same (e.g. `--out-dir results/scratch`) so you never clobber shipped results.

## Layout

```
paper/      main.tex (canonical, ACL) · references.bib · main.pdf · figures/ ·
            make_neurips.py (derives the workshop version) · neurips/ (generated)
src/        data.py · program.py · run.py · instrument.py · demolab.py · residual.py ·
            powered.py (E9) · substitution.py (E10) · turkish.py · mipro.py ·
            stats.py · controlled.py · onatili_vllm.py
analysis/   verify_paper.py (asserts every in-text number) · interaction.py (differential,
            single-subject or grouped by subject type) · placebo.py · decompose.py ·
            substitution_stats.py · dose_response.py · demolab_stats.py ·
            residual_stats.py · fixes_table.py · fertility.py · turkish_stats.py ·
            mipro_stats.py · determinism.py · erosion_table.py · demo_dist.py ·
            mcnemar.py · causal_stats.py · onatili_table.py · paper_numbers.py
scripts/    run_e1.sh .. run_e10.sh (experiment drivers) · run_e9_e10_driver.sh ·
            analyse_e9_e10.sh · sync_to_spark.sh · sync_results_back.sh · spark_setup.sh
tests/      pytest suite (103 tests) — `.venv/bin/python -m pytest -q`
results/    main/ · controlled/ · onatili_vllm/ · precision/ (original stack) ·
            e1/ .. e10/ (per-item logs + DECISION.md) · determinism.json ·
            paper_numbers.json (every number above, machine-readable)
data/       README.md (DTM Dataport pointer + public replication CSV)
docs/       results/2026-e1-e6-dossier.md (results dossier)
```

## Citation

```bibtex
@misc{dtm2026,
  title={Uzbek Multiple-Choice Question Dataset for Large Language Model Evaluation},
  author={Hazratov, Mardon and Mansuraliyev, Husanboy and Asadov, Dovud and
          Kayumov, Abduaziz and Toshnazarov, Qobiljon},
  year={2026}, publisher={IEEE Dataport}, doi={10.21227/e4h4-kp42}}
```

The paper citation will be added on release.

## License

Code released under the MIT License. The DTM benchmark is distributed under its IEEE
Dataport terms.
