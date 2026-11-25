"""Row-order convention for (4, ...) local stats arrays.

A local stats array has shape (4, n): four per-element statistics over n
elements (pixels or ells), in this fixed row order. Any code that writes,
reads, or reduces such an array must use these constants.
"""

STAT_MIN  = 0
STAT_MAX  = 1
STAT_MEAN = 2
STAT_STDV = 3

N_STATS = 4  # number of stat rows; a valid stats array is (N_STATS, n)
