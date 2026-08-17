"""The modelling table, and the leakage contract it is built under.

Every feature is computed from events at or before its row's prediction
origin. The contract is asserted per row on every run, not just in
tests.
"""
