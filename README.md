# eval-harness

[![CI](https://github.com/njoroge-viktor/eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/njoroge-viktor/eval-harness/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-brightgreen)](https://mypy-lang.org/)

The measurement side of LLM evaluation: **score** responses against a weighted
rubric, check whether your annotators actually **agree**, and **rank** models
from pairwise preferences.

Zero runtime dependencies. The whole library runs on the standard library, so
it drops into an annotation pipeline without dragging in a dependency tree.

```bash
pip install eval-harness
```

## Why this exists

Three questions come up on every evaluation batch, and getting them wrong is
expensive:

1. **How good is this batch?** Raw scores are not comparable across rubrics
   with different scales and weights. `Rubric` normalises onto `[0, 1]`.
2. **Do the annotators agree?** Raw percent agreement looks excellent on skewed
   categories even when raters are guessing. You need chance correction, and
   you need it to survive uneven rater coverage.
3. **Which model is actually better?** Win rate rewards a model for facing weak
   opponents. Bradley-Terry does not.

## Quick start

```bash
eval-harness score     --rubric examples/rubric.json --annotations examples/annotations.jsonl
eval-harness agreement --annotations examples/annotations.jsonl --criterion correctness
eval-harness rank      --comparisons examples/comparisons.jsonl
```

```
model                         W      L    T    win%   strength
-------------------------------------------------------------
claude-3-5-sonnet           3.0    0.0    1   87.5%     0.5759
gpt-4o                      2.0    1.0    1   62.5%     0.3380
llama-3-70b                 0.0    4.0    0    0.0%     0.0862
```

Every command takes `--json` for machine-readable output.

## Library use

```python
from eval_harness import Annotation, Criterion, Rubric, krippendorffs_alpha

rubric = Rubric(
    name="code-response-quality",
    criteria=[
        Criterion("correctness", weight=3.0, scale=(1, 5)),
        Criterion("security",    weight=2.5, scale=(1, 5)),
        Criterion("clarity",     weight=1.0, scale=(1, 5)),
    ],
)

scored = rubric.score(
    Annotation("task-001", "ann-a", {"correctness": 5, "security": 4, "clarity": 4})
)
print(scored.normalized)  # 0.897 — comparable across any rubric

# Uneven rater coverage is fine: alpha does not require a complete matrix.
alpha = krippendorffs_alpha({
    "task-001": {"ann-a": 5, "ann-b": 4},
    "task-002": {"ann-a": 2, "ann-b": 2, "ann-c": 3},
    "task-003": {"ann-a": 4},   # single rater, skipped
})
```

## Choosing an agreement coefficient

| Coefficient            | Raters      | Missing data | Chance model |
| ---------------------- | ----------- | ------------ | ------------ |
| `percent_agreement`    | 2+          | no           | none         |
| `cohens_kappa`         | exactly 2   | no           | per-rater    |
| `scotts_pi`            | exactly 2   | no           | pooled       |
| `fleiss_kappa`         | 2+ (fixed)  | no           | pooled       |
| `krippendorffs_alpha`  | 2+ (any)    | **yes**      | pooled       |

Cohen's kappa gives each rater their own marginal distribution; Scott's pi and
Fleiss' kappa pool them. **Reach for Krippendorff's alpha on real data** — a
live review queue almost never has every annotator seeing every item, and the
kappa family cannot represent that.

Never report percent agreement alone. With 95% of items in one category, two
raters who both guess "the common one" score 90%+ while agreeing by pure
chance; the chance-corrected coefficients will correctly sit near zero.

## Ranking: why not just win rate?

Win rate treats every match as equally informative. Consider two models that
each won two of three:

```
strong  beat elite  twice    elite beat weak twice
padded  beat weak   twice    strong tied padded
```

Both sit at the same win rate, but `strong` beat the model that beat everyone
else while `padded` farmed the weakest opponent. Bradley-Terry fits
`P(i beats j) = s_i / (s_i + s_j)` by maximum likelihood, so opponent quality
propagates through the graph and `strong` ranks above `padded`.

Two details that matter in practice:

- **Disconnected comparison graphs.** If models split into clusters that never
  met, their strengths can be scaled independently and the fit is meaningless.
  `bradley_terry()` raises rather than returning confident nonsense; `rank()`
  degrades to win rate and reports strength as `0.0`.
- **Winless models.** Under plain maximum likelihood, a model that never won
  has no finite strength and collapses to exactly zero — correct, but useless
  on a leaderboard. `rank()` applies a `prior=0.5` Laplace smoothing by
  default, turning the fit into a MAP estimate that keeps every strength finite
  and correctly ordered. Pass `prior=0` for the unbiased fit.

## Data formats

Annotations and comparisons are JSONL, one record per line:

```jsonl
{"item_id":"task-001","annotator_id":"ann-a","scores":{"correctness":5},"notes":"optional"}
{"item_id":"p-001","annotator_id":"ann-a","response_a":"gpt-4o","response_b":"claude-3-5-sonnet","verdict":"b"}
```

`verdict` is one of `a`, `b`, `tie` (case-insensitive). Every parse failure
names the file and line number — "invalid JSON in a 40,000-line export" is not
an actionable bug report:

```
error: examples/annotations.jsonl:1843: missing required field 'annotator_id'
```

## Exit codes

| Code | Meaning                                              |
| ---- | ---------------------------------------------------- |
| `0`  | Success                                              |
| `1`  | Invalid input data (bad rubric, out-of-scale scores) |
| `2`  | Malformed file, or a criterion that matched nothing  |

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                      # 98 tests, 98% coverage
mypy src/eval_harness       # strict mode, no ignores
ruff check .
```

Non-trivial statistical expectations are pinned to hand-computed values with
the arithmetic in the test, so a change that moves a number has to argue with
the derivation rather than re-baseline the test. Two independent code paths
cross-check each other where an identity exists — Fleiss' kappa reduces to
Scott's pi at two raters, and the two-model Bradley-Terry fit is checked
against its closed form.

## License

MIT — see [LICENSE](LICENSE).
