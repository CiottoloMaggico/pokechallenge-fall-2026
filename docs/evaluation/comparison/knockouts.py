"""The hidden P2 knockouts found by each team, exit by exit.

    python docs/evaluation/comparison/knockouts.py --results results/comparison

A fainted P2 Pokémon is usually not recorded: the turn already shows its replacement. Every
team estimates which of P2's last exits (a Pokémon that leaves the field and is not seen again)
hide a knockout:

* ``v3``: our detector of ``compact_v5`` (gradient boosting fitted on P1's recorded faints);
* ``v11``: our detector of ``v11`` (LightGBM, fitted the same way);
* ``rule``: our hand-written rule that preceded them (used by no submission);
* ``second``: the 2nd place's LightGBM fitted on P1's recorded knockouts, then shifted so that
  P2 gets as many knockouts as P1 (``second_place.py knockouts``);
* ``spikes``: the 9th place's hand-written rule (``predictive_spikes.py knockouts``).

An exit counts as a knockout for a team when its score is at least 0.5 (``spikes``: when their
rule counts the Pokémon as fainted, which is not a threshold on its score). No truth exists for
P2, so the tables only compare the teams with one another and with the outcome.

Writes ``<results>/knockouts/events.csv`` (every exit of the whole battles and of their first 20
turns, with every team's score: the input of ``knockout_swap.py``) and, to ``data/``:

* ``knockout_counts.csv``: exits scored, knockouts counted and sum of the scores, per team;
* ``knockout_agreement.csv``: per pair of teams, share of exits with the same call, Cohen's
  kappa, rank correlation of the scores;
* ``knockout_votes.csv``: exits by the number of teams that call them a knockout;
* ``knockout_situations.csv``: share of exits called a knockout, by situation;
* ``knockout_outcome.csv``: how well "P2's knockouts minus P1's" alone predicts the winner
  (ROC AUC, bootstrap over battles) with each team's knockouts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from addons import load_repair  # noqa: E402
from domain.pokedex import Pokedex  # noqa: E402
from domain.repair import TimelineRepair  # noqa: E402
from features.common import type_chart  # noqa: E402
from settings import DEFAULT_POKEDEX, DEFAULT_TRAIN  # noqa: E402

OURS = {"v3": "model=results/addons/knockouts/v3", "v11": "v11=results/addons/knockouts_v11/v1",
        "rule": "rule"}  # fmt: skip
TEAMS = ["v3", "v11", "rule", "second", "spikes"]
VOTERS = ["v3", "second", "spikes"]  # one detector per team
KEY = ["row", "turns", "index"]
PREFIX = 20  # compact_v5 also replays the first 20 turns (hpw_spd_t20)
HP_BINS = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
MOVES = ["attack", "attack, no power recorded", "status move", "no move"]
ENTRIES = ["unhurt", "lost under 14%", "lost more"]


def our_scores(records: list[dict[str, Any]], repair: TimelineRepair, name: str) -> pd.DataFrame:
    """The score ``repair`` gives to every P2 switch of every battle, whole and cut."""
    rows = []
    for turns in (30, PREFIX):
        for row, record in enumerate(records):
            cut = {**record, "battle_timeline": record["battle_timeline"][:turns]}
            rows += [(row, turns, i, p) for i, p in repair.probabilities(cut).items()]
    return pd.DataFrame(rows, columns=[*KEY, name])


def events_table(results: Path, records: list[dict[str, Any]]) -> pd.DataFrame:
    """Every last exit of P2 with every team's score; ``<team>_ko``: counted as a knockout."""
    second = pd.read_csv(results / "our_folds" / "fixed" / "knockouts.csv")
    spikes = pd.read_csv(results / "spikes" / "knockouts" / "knockouts.csv")
    table = second.rename(columns={"p": "second"}).merge(
        spikes.rename(columns={"p": "spikes", "strict": "spikes_ko"}), on=[*KEY, "leaver"]
    )
    if not len(table) == len(second) == len(spikes):
        raise SystemExit("the two teams do not score the same exits")
    pokedex = Pokedex.from_json(DEFAULT_POKEDEX)
    visible = [{k: v for k, v in record.items() if k != "player_won"} for record in records]
    for name, spec in OURS.items():
        scores = our_scores(visible, load_repair(spec, pokedex, type_chart()), name)
        table = table.merge(scores, on=KEY, how="left")
        if bool(table[name].isna().any()):
            raise SystemExit(f"{name} does not score every exit")
        table[f"{name}_ko"] = (table[name] >= 0.5).astype(int)
    table["second_ko"] = (table["second"] >= 0.5).astype(int)
    return table


def situations(events: pd.DataFrame, records: list[dict[str, Any]]) -> pd.DataFrame:
    """What the record shows at each exit of ``events``: the leaver's HP, P1's move in the turn
    of the exit, and whether the replacement lost HP since it was last seen."""
    rows = []
    for row, i in zip(events["row"], events["index"], strict=True):
        timeline = records[row]["battle_timeline"]
        leaver, incoming = (timeline[j]["p2_pokemon_state"] for j in (i - 1, i))
        move = timeline[i]["p1_move_details"]
        seen = [t["p2_pokemon_state"]["hp_pct"] for t in timeline[:i]
                if t["p2_pokemon_state"]["name"] == incoming["name"]]  # fmt: skip
        lost = (seen[-1] if seen else 1.0) - incoming["hp_pct"]
        attack = "no move" if not move else "status move" if move["category"] == "status" else (
            "attack" if move["base_power"] > 0 else "attack, no power recorded")  # fmt: skip
        entry = "unhurt" if lost <= 0.005 else "lost under 14%" if lost < 0.14 else "lost more"
        rows.append({"hp": leaver["hp_pct"], "move": attack, "entry": entry})
    table = pd.DataFrame(rows, index=events.index)
    labels = [f"{a:.0%} to {b:.0%}" for a, b in zip(HP_BINS, HP_BINS[1:], strict=False)]
    return pd.DataFrame({
        "Leaver's HP before the exit": pd.cut(table["hp"], HP_BINS, labels=labels),
        "P1's move in the turn of the exit": pd.Categorical(table["move"], MOVES),
        "HP lost by the replacement since last seen": pd.Categorical(table["entry"], ENTRIES),
    })  # fmt: skip


def counts(events: Any) -> pd.DataFrame:
    """Per team: exits scored, knockouts counted, sum of the scores."""
    rows = [{"team": team, "exits": len(events), "knockouts": int(events[f"{team}_ko"].sum()),
             "sum_of_scores": float(events[team].sum())} for team in TEAMS]  # fmt: skip
    return pd.DataFrame(rows)


def agreement(events: Any) -> pd.DataFrame:
    """Per pair of teams: same call, both / only one knockout, kappa, Spearman of the scores."""
    rows = []
    for k, a in enumerate(TEAMS):
        for b in TEAMS[k + 1 :]:
            x, y = events[f"{a}_ko"].to_numpy() == 1, events[f"{b}_ko"].to_numpy() == 1
            same = float((x == y).mean())
            chance = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
            rows.append({
                "a": a, "b": b, "same": same, "kappa": (same - chance) / (1 - chance),
                "both": int((x & y).sum()), "only_a": int((x & ~y).sum()),
                "only_b": int((~x & y).sum()),
                "spearman": events[a].corr(events[b], method="spearman"),
            })  # fmt: skip
    return pd.DataFrame(rows)


def by_situation(events: pd.DataFrame, described: pd.DataFrame) -> pd.DataFrame:
    """Share of exits each team calls a knockout, by value of each descriptor."""
    calls = events[[f"{team}_ko" for team in TEAMS]].set_axis(TEAMS, axis=1)
    parts = []
    for name in described:
        groups: Any = calls.groupby(described[name], observed=True)
        part = groups.mean().assign(exits=groups.size()).reset_index(names="value")
        parts.append(part.assign(situation=name))
    table: Any = pd.concat(parts)
    return table[["situation", "value", "exits", *TEAMS]]


def outcome(events: pd.DataFrame, records: list[dict[str, Any]], resamples: int) -> pd.DataFrame:
    """ROC AUC of "P2's knockouts minus P1's" for the winner, with no hidden knockout and with
    each team's (counted, and as the sum of its scores); bootstrap over battles, paired with v3."""
    y = np.array([int(record["player_won"]) for record in records])
    fainted = {side: np.array([len({t[f"{side}_pokemon_state"]["name"] for t in r["battle_timeline"]
                                    if t[f"{side}_pokemon_state"]["status"] == "fnt"})
                               for r in records]) for side in ("p1", "p2")}  # fmt: skip
    recorded = fainted["p2"] - fainted["p1"]
    scores = {("none", "counted"): recorded.astype(float)}
    for team in TEAMS:
        for kind, column in (("counted", f"{team}_ko"), ("scores", team)):
            hidden = np.bincount(events["row"], weights=events[column], minlength=len(y))
            scores[team, kind] = recorded + hidden
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(y), (resamples, len(y)))
    boot = {key: np.array([roc_auc_score(y[d], s[d]) for d in draws]) for key, s in scores.items()}
    rows = []
    for (team, kind), s in scores.items():
        low, high = np.quantile(boot[team, kind], [0.025, 0.975])
        diff = boot[team, kind] - boot["v3", kind]
        rows.append({"team": team, "knockouts": kind, "roc_auc": roc_auc_score(y, s),
                     "ci_low": low, "ci_high": high, "minus_v3": diff.mean(),
                     "minus_v3_low": np.quantile(diff, 0.025),
                     "minus_v3_high": np.quantile(diff, 0.975)})  # fmt: skip
    return pd.DataFrame(rows)


def main(results: Path, out: Path, resamples: int) -> None:
    records = [json.loads(line) for line in DEFAULT_TRAIN.read_text().splitlines()]
    events = events_table(results, records)
    (results / "knockouts").mkdir(exist_ok=True)
    events.to_csv(results / "knockouts" / "events.csv", index=False)
    whole: Any = events[events["turns"] == 30].reset_index(drop=True)
    votes: Any = whole[[f"{team}_ko" for team in VOTERS]].sum(axis=1).value_counts().sort_index()
    tables = {
        "counts": counts(whole),
        "agreement": agreement(whole),
        "votes": votes.rename_axis("teams").rename("exits").reset_index(),
        "situations": by_situation(whole, situations(whole, records)),
        "outcome": outcome(whole, records, resamples),
    }
    for name, table in tables.items():
        table.to_csv(out / f"knockout_{name}.csv", index=False)
        print(f"\n{name}\n{table.to_string(index=False, float_format='{:.4f}'.format)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hidden P2 knockouts found by each team")
    parser.add_argument("--results", type=Path, required=True, help="results/comparison")
    parser.add_argument("--out", type=Path, default=HERE / "data")
    parser.add_argument("--resamples", type=int, default=2000)
    arguments = parser.parse_args()
    main(arguments.results, arguments.out, arguments.resamples)
