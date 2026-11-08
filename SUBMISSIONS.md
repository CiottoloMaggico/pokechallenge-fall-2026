# Submissions of 2026-11-05

Both files were produced with the Pokédex built from train only (step 2,
commit `ad7e4c4`). They live in `results/submissions/` (git-ignored); the
hashes below identify them. The afternoon evidence in `runs/` and
`submission_*.csv` in the root was not modified.

Environment: Python 3.14.10, `requirements-submission.txt`, 16 cores.
Inputs (sha256 prefix): `input/train.jsonl` `78f9e91c51b92ade`,
`input/test.jsonl` `25bdbb2a2f43d6ca`.

Run times come from jobs running at the same time, so they are upper bounds.

The recipes check out older commits: `tools/blend_from_saved.py`, the v11 bridge
(`features/v11_bridge.py`, `compact_v5_v11`, `blend_v11`) and `tools/v11_*` were deleted
from `main` in step 8 and live at the tag `submission-final` (and at `ad7e4c4`).

## `submission_ensemble_step2.csv` (`ee42414df7342f10`)

`ensemble_from_lr` with the tuned parameters of the 13:18 run; each member's
parameters were checked equal to `runs/compact_v5__ensemble_from_lr-tuned/model.pkl`.
`run.json` does not record whether the 13:18 run used `--test`; this one does not.

```sh
git checkout ad7e4c4
export PYTHONHASHSEED=0
python -m domain build-pokedex input/train.jsonl --out results/cache/species.json
python main.py train --features compact_v5 --models ensemble_from_lr \
    --params runs/tuning/compact_v5__*.json --out runs/step2__ensemble_from_lr-tuned
python main.py predict \
    --run runs/step2__ensemble_from_lr-tuned/compact_v5__ensemble_from_lr-tuned \
    --test input/test.jsonl --out results/submissions/submission_ensemble_step2.csv \
    --proba results/submissions/submission_ensemble_step2_proba.csv
```

- Input files: the two above and `runs/tuning/compact_v5__*.json`.
- Run time: train 382 s, predict 223 s.
- Against `runs/compact_v5__ensemble_from_lr-tuned/proba.csv`: 46 class flips,
  probability change mean 0.0052, max 0.1760.

## `submission_blend_step2.csv` (`f93d03dda75dcc1f`)

`logit(p) = 0.55 * logit(p_base) + 0.45 * logit(p_v11)`, with EPS and the NaN rule of
`models/logit_blend.py`. v11 was **not** refitted: **`p_v11` comes from inverting the
afternoon blend** (`submission_blend_v11_proba.csv`, `728df610f813d5d8`).

The base of the afternoon blend is `ensemble_from_lr` **without tuning**
(`runs/tuning/compact_v5_v11__blend_v11.json`: `"tuned": []`), not the tuned 13:18
model, so both the inverted base and the new base are untuned, and `w = 0.55` is the
weight selected for that base. Its seeds in that file are 33 66 99.

```sh
export PYTHONHASHSEED=0

# 1. new base, on the step 2 commit
git checkout ad7e4c4
python main.py train --features compact_v5 --models ensemble_from_lr \
    --out runs/step2__ensemble_from_lr
python main.py predict --run runs/step2__ensemble_from_lr/compact_v5__ensemble_from_lr \
    --test input/test.jsonl --out results/submissions/base_untuned_step2.csv \
    --proba results/submissions/base_untuned_step2_proba.csv
git checkout submission-final

# 2. old base: the model saved inside the blend, with the features before step 2
git worktree add <scratch>/old_code 5bd0bf0
ln -s "$PWD/input" <scratch>/old_code/input
(cd <scratch>/old_code && python "$OLDPWD/tools/blend_from_saved.py" old-base \
    --blend-run "$OLDPWD/runs/compact_v5_v11__blend_v11-tuned" \
    --out "$OLDPWD/results/submissions/old_base_untuned_afternoon.csv")

# 3. invert and blend
python tools/blend_from_saved.py blend --blend-run runs/compact_v5_v11__blend_v11-tuned \
    --old-blend submission_blend_v11_proba.csv \
    --old-base results/submissions/old_base_untuned_afternoon.csv \
    --new-base results/submissions/base_untuned_step2_proba.csv \
    --out-dir results/submissions --name submission_blend_step2
```

- Input files: the two above, `runs/compact_v5_v11__blend_v11-tuned/`,
  `submission_blend_v11_proba.csv`.
- Run time: new base train 433 s, predict 213 s; old base 226 s; blend a few seconds.
- Against `submission_blend_v11.csv`: 17 class flips, probability change mean 0.0019,
  max 0.0914.
- Checks: on the 7 test battles identical to a training battle, where v11 is NaN and
  the old blend equals its base, the recovered old base matches within 1.1e-16. The
  same old code reproduces the 13:18 `proba.csv` exactly. `p_v11` could not be
  compared with a direct reading: the saved run's cache is empty.
- `results/submissions/v11_test_proba.csv` (`297726b84ce7dda0`) holds `p_v11`; NaN on those 7 battles.

### Recomputing the blend from scratch (not run)

This refits v11 on this commit, so its numbers can differ from the file above.
Set `FDS_N_JOBS` to the number of cores. `--out` must stay a new folder: `--out runs`
would overwrite `runs/compact_v5_v11__blend_v11-tuned`, the evidence of the afternoon.
The v11 repository is not under git, so its version cannot be recorded here.

```sh
git checkout ad7e4c4
export PYTHONHASHSEED=0 V11_REPO=../FDS-Pokemon-Battle-Prediction-main FDS_N_JOBS=16
python main.py train --features compact_v5_v11 --models blend_v11 \
    --params runs/tuning/compact_v5_v11__blend_v11.json --out runs/step2_scratch
python main.py predict --run runs/step2_scratch/compact_v5_v11__blend_v11-tuned \
    --test input/test.jsonl --out results/submissions/submission_blend_step2_scratch.csv \
    --proba results/submissions/submission_blend_step2_scratch_proba.csv
```

Do not pass `--jobs`.
