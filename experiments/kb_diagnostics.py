"""Go / no-go check of the team archive, on the training data only.

Five stratified, grouped folds of ``input/train.jsonl``. In each fold the
archive is built on the training part; every validation battle is then
read at turn ``CUT`` (15) and compared with what happens by ``END`` (30):

* moves: for every revealed P2 Pokémon, each archive move of its species
  not yet revealed at ``CUT`` is labelled 1 if it is revealed by ``END``.
  ``p(m | T, s)`` (team identified) is scored against ``p(m | s)`` with
  ROC AUC and log-loss, on the battles whose P2 team is identified;
* hidden: among battles where P2 reveals new Pokémon between ``CUT`` and
  ``END``, the share of those new Pokémon that were predicted as hidden:
  the archive's guess (identified team, otherwise the naive Bayes guess)
  against the naive Bayes guess alone.

Pass the check if the move AUC gains at least ``AUC_GAIN`` on identified
battles, or the hidden share gains at least ``HIDDEN_GAIN`` points. If it
fails, the archive adds no information over the species statistics and
the features are not added to a pool.

Run from the project root::

    python -m experiments.kb_diagnostics --train input/train.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from data import Dataset
from domain.battle import BattleContext
from domain.pokedex import Pokedex
from domain.records import fingerprint
from domain.team_knowledge import (
    KnowledgeView,
    TeamKnowledge,
)
from features import sets as feature_sets
from features.core.team import HiddenTeamStrength
from knockout_detector import detector_from_files

CUT = 15
END = 30
FOLDS = 5
SEED = 11
AUC_GAIN = 0.02
HIDDEN_GAIN = 0.10
CLIP = 1e-6
DEFAULT_POKEDEX = Path("results/cache/species.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", type=Path, default=Path("input/train.jsonl"))
    parser.add_argument("--pokedex", type=Path, default=DEFAULT_POKEDEX)
    args = parser.parse_args(argv)

    pokedex = Pokedex.from_json(args.pokedex)
    chart = feature_sets.type_chart()
    detector = detector_from_files(pokedex, chart, (args.train,))
    train = Dataset.from_jsonl(args.train, pokedex, repair=detector)
    contexts = list(train.contexts)
    labels = train.labels
    groups = train.duplicate_groups()

    splitter = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED)
    moves = _MoveScores()
    hidden = _HiddenScores()
    for fold, (fit_idx, val_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups), start=1
    ):
        fit_contexts = [contexts[i] for i in fit_idx]
        knowledge = TeamKnowledge.build(fit_contexts)
        guesser = HiddenTeamStrength(None).fit(fit_contexts)
        for i in val_idx:
            _score_battle(contexts[i], knowledge, guesser, moves, hidden)
        print(f"fold {fold}/{FOLDS} done", flush=True)

    return _report(moves, hidden)


class _MoveScores:
    def __init__(self) -> None:
        self.labels: list[int] = []
        self.team: list[float] = []
        self.species: list[float] = []

    def add(self, label: int, team: float, species: float) -> None:
        self.labels.append(label)
        self.team.append(team)
        self.species.append(species)


class _HiddenScores:
    def __init__(self) -> None:
        self.battles = 0
        self.new = 0
        self.archive_hits = 0
        self.naive_hits = 0


def _score_battle(
    context: BattleContext,
    knowledge: TeamKnowledge,
    guesser: HiddenTeamStrength,
    moves: _MoveScores,
    hidden: _HiddenScores,
) -> None:
    view = knowledge.excluding(fingerprint(context.record))
    at_cut = context.at_turn(CUT)
    at_end = context.at_turn(END)
    revealed = frozenset(at_cut.side("p2").team)
    team = _team(view, revealed)
    _score_moves(view, team, at_cut, at_end, moves)
    _score_hidden(view, team, guesser, revealed, at_cut, at_end, hidden)


def _team(view: KnowledgeView, revealed: frozenset[str]) -> frozenset[str] | None:
    if len(revealed) == 6:
        return revealed
    return view.identify(revealed)


def _score_moves(
    view: KnowledgeView,
    team: frozenset[str] | None,
    at_cut: BattleContext,
    at_end: BattleContext,
    moves: _MoveScores,
) -> None:
    if team is None:
        return
    p2_cut = at_cut.side("p2").team
    p2_end = at_end.side("p2").team
    for name, mon in p2_cut.items():
        later = p2_end.get(name)
        if later is None:
            continue
        known_now = set(mon.known_moves)
        known_later = set(later.known_moves)
        per_species = view.move_probabilities(None, name)
        for move, p_species in sorted(per_species.items()):
            if move in known_now:
                continue
            label = int(move in known_later)
            p_team = view.move_probability(team, name, move)
            moves.add(label, p_team, p_species)


def _score_hidden(
    view: KnowledgeView,
    team: frozenset[str] | None,
    guesser: HiddenTeamStrength,
    revealed: frozenset[str],
    at_cut: BattleContext,
    at_end: BattleContext,
    hidden: _HiddenScores,
) -> None:
    new = set(at_end.side("p2").team) - set(at_cut.side("p2").team)
    if not new or len(revealed) == 6:
        return
    naive = set(guesser.guess(list(revealed), 6))
    archive = set(sorted(team - revealed)) if team is not None else naive
    hidden.battles += 1
    hidden.new += len(new)
    hidden.archive_hits += len(new & archive)
    hidden.naive_hits += len(new & naive)


def _report(moves: _MoveScores, hidden: _HiddenScores) -> int:
    labels = np.asarray(moves.labels)
    team = np.clip(np.asarray(moves.team), CLIP, 1 - CLIP)
    species = np.clip(np.asarray(moves.species), CLIP, 1 - CLIP)
    print(f"\nmove candidates (identified teams): {labels.size}")
    print(f"positive rate: {labels.mean():.4f}")
    auc_team = roc_auc_score(labels, team)
    auc_species = roc_auc_score(labels, species)
    print(f"AUC  p(m|T,s) {auc_team:.4f}   p(m|s) {auc_species:.4f}")
    print(
        f"logloss p(m|T,s) {log_loss(labels, team):.4f}   "
        f"p(m|s) {log_loss(labels, species):.4f}"
    )
    quota_archive = hidden.archive_hits / max(hidden.new, 1)
    quota_naive = hidden.naive_hits / max(hidden.new, 1)
    print(f"\nbattles with new P2 Pokémon: {hidden.battles}, new: {hidden.new}")
    print(f"hidden share guessed  archive {quota_archive:.4f}  naive {quota_naive:.4f}")
    auc_gain = auc_team - auc_species
    hidden_gain = quota_archive - quota_naive
    print(f"\nAUC gain {auc_gain:+.4f} (need >= {AUC_GAIN})")
    print(f"hidden gain {hidden_gain:+.4f} (need >= {HIDDEN_GAIN})")
    passed = auc_gain >= AUC_GAIN or hidden_gain >= HIDDEN_GAIN
    print("GO" if passed else "NO-GO")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
