"""Scoring, and the machinery that keeps a score honest.

The shared metric definitions both tracks import rather than redefine,
the freeze and seal that make "scored once" enforced rather than
intended, the error slices, the ablations, and the verifier that
recomputes every committed value and fails on any difference.
"""
