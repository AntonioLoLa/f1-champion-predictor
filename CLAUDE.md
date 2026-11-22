# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Predicts each driver's probability of winning the Formula 1 World Drivers' Championship,
after every Grand Prix, from the full history of the sport. A data pipeline against
external providers, a leakage-audited feature layer, a model zoo tracked in MLflow, a
FastAPI service and a Streamlit dashboard. No notebooks: every stage is a tested module.

`README.md` explains the modelling problem and reports headline results. The rationale for
non-obvious decisions lives in the docstring of the module that made them — the feature
pipeline, the elimination bounds and the design tokens each argue their own case.

## Specs

`openspec/specs/` holds the behaviour the system guarantees today, one directory per
capability, in [OpenSpec](https://github.com/Fission-AI/OpenSpec) format. **Read the spec
for a capability before changing it** — the requirements there are the contract, and
several encode a defect that has already been fixed once (feature leakage, calibration
collapsing to certainty, promoting a winner a race never had).

```bash
npx @fission-ai/openspec@latest list --specs      # capabilities
npx @fission-ai/openspec@latest validate --all --strict
```

Work that changes a requirement goes through `openspec/changes/<id>/` as a proposal plus a
delta spec, and is folded into the main specs once shipped. Slash commands
(`/opsx:propose`, `/opsx:apply`, `/opsx:archive`) are wired up in `.claude/`.

**Specs describe today, not the target.** Two of them record known defects rather than
guarantees — group normalisation when the champion is absent from a round, and the
train/serve population mismatch. Fixing those means strengthening the requirement, and the
scenario that documents the defect should disappear with it. `docs/AUDIT.md` and `docs/ROADMAP.md`
hold the findings and the triaged plan.

## Commands

```bash
make setup           # uv venv (Python 3.12) + editable install with dev extras
make lint            # ruff + black --check + mypy --strict  (all three must pass)
make format          # ruff --fix + black
make test            # full suite, excluding tests marked `network`
make api             # FastAPI on :8000, auto-reload
make dashboard       # Streamlit on :8501 (needs the API running)
make pipeline        # data -> dataset -> train -> evaluate, end to end
make up / down       # full Docker stack (postgres, mlflow, api, dashboard)
```

Everything runs through `.venv/bin/python`; there is no global install. Pipeline stages
are also individual scripts (`scripts/update_data.py`, `build_dataset.py`, `train.py`,
`evaluate.py`) and a CLI (`python -m f1cp.cli`).

### Tests

```bash
.venv/bin/python -m pytest -m "not network"                       # what CI runs
.venv/bin/python -m pytest -m unit                                # fast, hermetic
.venv/bin/python -m pytest tests/unit/test_elimination.py         # one file
.venv/bin/python -m pytest tests/integration/test_predictor.py -k cache   # one test
```

Markers: `unit` (no I/O), `integration` (database/filesystem), `e2e` (slow), `network`
(live internet, deselected by default). Markers are strict — a new test must carry one.

## Architecture

Data flows one way, and each arrow is a module boundary worth respecting:

```
providers -> ingest -> database -> dataset/loader -> features/pipeline -> training
                                                          |                  |
                                                          v                  v
                                                   inference/predictor <- MLflow
                                                          |
                                                    api -> dashboard
```

**`container.py` is the composition root.** The CLI, the API and the tests all build an
`AppContext` and pass it down. Nothing reaches for global state to find its collaborators.

**The feature pipeline is the load-bearing part.** `features/pipeline.py` runs enabled
blocks in dependency order over a panel of one row per `(season, round, driver_id)`. Every
feature is computed with a `shift(1)` before any rolling aggregation — that shift is the
difference between a legitimate predictor and a leaky one. `tests/unit/features/
test_pipeline.py::test_features_are_unchanged_by_deleting_the_future` is the guard: it
asserts that truncating the future cannot change the past. **If you touch a feature block,
that test is the one that matters.**

**Inference deliberately rebuilds features from the live database**, never from the stored
dataset parquet — a served prediction must reflect what has actually been ingested. That
guarantee is preserved by keying `PredictionService`'s caches on
`dataset.loader.source_data_revision`, a cheap fingerprint re-read on every call. Do not
replace those caches with time-based expiry; the freshness guarantee is the point.

**Validation is walk-forward, never a random split.** Training expands over past seasons
and tests on the next one. The last few seasons of each training fold are held out to fit
the probability calibrator, so calibration never sees data the model was fit on.

**Probabilities are group-normalised.** Exactly one driver wins a championship, so
probabilities are rescaled to sum to 1 within each `(season, round)` group. A raw model
output and a served probability are different numbers, and both are surfaced.

### Configuration

All behaviour is driven by `conf/*.yaml` (`config`, `data`, `features`, `models`), loaded
into pydantic settings via `config/loader.py`, with `${VAR:-default}` expanded from the
environment. Prefer changing configuration to changing code. `.env.example` documents
every variable.

### Dashboard

Talks to the API over HTTP exclusively — never to the database or MLflow — so it stays
independently deployable. `components/palette.py` holds design tokens;
`components/theme.py` generates the single stylesheet from them and owns the presentational
primitives (`page_header`, `section`, `stat_grid`, `leaderboard`, `card`). Pages compose
those; they do not write their own CSS or colour literals. `.streamlit/config.toml` mirrors
the same values for Streamlit's own chrome and must be kept in sync by hand.
`components/labels.py` turns API keys (team slugs, model and metric names, feature columns)
into reader-facing words and formats probabilities; pages never print a raw key. A new
feature needs a label there, or `test_every_feature_the_model_reads_has_a_readable_label`
fails.

## Conventions

- **Google-style docstrings on everything public**, enforced by ruff's pydocstyle rules.
  Comments explain *why*, not what; several modules carry long docstrings arguing for a
  design decision, and that is the house style.
- `line-length = 100`, black-formatted, `mypy --strict` over the whole package.
- Relative imports across parent packages are banned.
- **New API response fields need the server restarted** to take effect — a long-running
  uvicorn process holds the old pydantic schema in memory and will `KeyError` in the
  dashboard. `make api` runs with `--reload`, so this bites mostly when a process was
  started another way.

## Gotchas that have already cost time

- **Streamlit drops anything a rerun does not re-emit.** A stylesheet injected once behind
  a session-state flag disappears on the user's first click; widget state is garbage
  collected when a run does not render that widget, so a selection silently resets on page
  navigation. Both have bitten this codebase — see `theme.inject_theme` and
  `layout.model_version_selector`, which document the fix.
- **`st.Page` infers a URL path from the callable's `__name__`.** Every page callback is a
  lambda, so `url_path` must be passed explicitly or navigation collides.
- **Polars `rolling_map` returns the input column's dtype.** Applying a float-valued
  function to an integer column silently truncates every result. This corrupted the `trend`
  feature for a long time; `_rolling_ols_slope` in `features/form.py` is the replacement and
  is also ~10x faster. Prefer closed-form column arithmetic over `rolling_map` generally —
  it hands every row to a Python callback.
- **Plotly clips axis labels without `automargin`.** The shared chart layout uses tight
  margins so figures sit flush in their card; both axis defaults in
  `dashboard/components/charts.py` set `automargin` for this reason.
- **The calibration holdout is three *seasons*, not 1400 independent samples** — every
  round of a season repeats the same champion. Isotonic regression overfits that badly
  enough to report 100% title probabilities; the config uses `sigmoid` (Platt) and should
  stay there unless the label count changes fundamentally.
- **Each `f1cp train run` appends a fresh candidate per algorithm** rather than overwriting.
  That is by design, for audit history — `model_versions` growing is not a bug. Views
  default to the latest `training_run_id`.
- **The deploy branch's `data/mlflow.db` holds absolute artifact paths from the machine
  that trained last** — your laptop, or the GitHub runner once `data-sync.yml` retrains.
  After pulling `deploy/streamlit-cloud`, run `make relocate-mlflow` before serving or
  evaluating locally, or the model fails to load. That rewrite (and the API caching
  predictions) dirties the tracked `data/*.db`; `git restore data/` before the next pull,
  or the bot's next sync commit will not fast-forward.
- **Jolpica standings list every driver who has scored this season, not only the round's
  entrants.** A stand-in who scored earlier and sat a round out (Hirakawa, 2026 Brazil) has
  no `drivers` record in that round's results; `_fetch_standings` emits the drivers it
  embeds so the identity resolver can still map them.

## Current state (2026-11-21)

- **The deployed app updates itself.** `.github/workflows/data-sync.yml` (on `main`, runs
  Mon/Wed 08:00 UTC) checks out `deploy/streamlit-cloud`, runs `f1cp data update`, and
  commits `data/f1cp.db` back when the race data changed; Streamlit Community Cloud
  redeploys on that push. It retrains only when `f1cp train due` says a season has closed
  since the last successful run (`training/cadence.py`), then also commits `data/mlflow.db`
  and `mlruns/`. Manual dispatch takes `force_retrain` and `dry_run`. It replaced
  `nightly-update.yml`, which retrained into an `actions/cache` nothing served and had been
  disabled for inactivity since 2026-10-03.
- **Verified end to end on GitHub:** a `dry_run` rehearsal, then a real `force_retrain`
  run (44892747926) that retrained, committed as `github-actions[bot]` (149e119) and
  pushed to the deploy branch. The model it published loads and serves locally after
  `make relocate-mlflow`. The first automatic round sync should be Monday 2026-11-23,
  after Las Vegas. The first automatic retrain is due after Abu Dhabi (2026-12-06).
  Whether the keepalive step really resets GitHub's 60-day inactivity timer is unconfirmed
  until the off-season has passed.
- **Data:** the deploy branch holds results through **2026 R20, Brazilian GP**. Jolpica has
  not published 2027 yet; the sync fetches it whole once it does (`_missing_rounds`
  returns `None` for an unknown season).
- **Served model:** training run **9** (2026-11-21, on the CI runner), dataset
  `090a2f23eca2-2026-11-21` (21,282 rows × 72 features). Active model version **49**,
  `logistic_regression` (`group_log_loss` 0.6225). The one-standard-error tie set is
  {logistic_regression, lightgbm} (threshold 0.7206). It is the first deployed model with
  the N1 (`momentum`) and N8 (`position_order` dropped) fixes. Until this run the app
  served run 8 / model 43 from 2026-08-02. An earlier laptop "run 9" with those fixes was
  never committed to the deploy branch and is gone, so ids from before today in old notes
  don't match. `train due` only reacts to a season closing, not to feature-code changes:
  **after changing features, retrain by hand** (dispatch Data sync with `force_retrain`).
- `docs/AUDIT_v2.md`: N1, N3–N9 fixed. N5 was fixed via `api.read_only`. The infra finding
  that "nightly-update.yml updates a deployment nobody serves" is resolved by `data-sync.yml`.
  N2 (`CircuitBlock` test), R5 (calibration default) and R6 (scikit-learn pin) were fixed
  on 2026-11-22. Still open: R1–R4, which are documented trade-offs. That includes no
  hyperparameter search across the model zoo; the README's Limitations section says so.
  `tests/e2e/test_pipeline.py` covers provider → ingest → dataset → train → serve → next
  round, over a synthetic eight-season championship (about 25 s; part of `make test`).
