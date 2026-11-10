"""v11, the ``blend_v11`` model of the FDS repository, on this repository's constructs (features).

The group functions keep the original's logic (constants, operation and iteration order, seeds);
they read a per-battle view built from ``BattleContext`` (``battle.py``), the repository's
species and the generic domain modules (observed timeline, expected damage). ``groups.py`` and
``forecast.py`` wrap them as one ``Feature`` per group.
"""
