"""``v11_fs``: the living line of v11 (the FDS repository's ``blend_v11``), features.

It started as a copy of the frozen ``features.core.v11`` (same group functions, constants and
order) and owns its code: every future change to v11's features is made here, never there. The
group functions read a per-battle view built from ``BattleContext`` (``battle.py``, memoized
under its own keys), the repository's species and the generic domain modules (observed timeline,
expected damage); ``groups.py`` and ``forecast.py`` wrap them as one ``Feature`` per group;
``label_aware.py`` holds the label-aware bag logit and team effect.
"""
