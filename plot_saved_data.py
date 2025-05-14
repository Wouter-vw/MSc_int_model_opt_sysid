import numpy as np
import matplotlib.pyplot as plt
import os
from helper import plot_error_comparison_unknown_b_k, plot_error_comparison_known_b_k, plot_error_comparison_RLS

### ADJUST FUNCTIONS IN HELPER.PY TO EDIT VISIBLE LINES, LEGEND, LABELS, TITLE


# There are three different functions which can be used to plot results.
# 1) plot_error_comparison_unknown_b_k ("b_type", "A_type", "savefig"). This plots data from the version where b_k is unknown
## Choices for b_type are: 
#               "constant" "ramp" "sine" "sine+ramp" "sine^2" "sine-sine" "sine^2-sine" "ramp-then-sine" 
#               "sine-sine^2-mixed" "sine-ramp-mixed" "sine-sine+ramp-mixed" "sine^2-ramp-mixed" "sine^2-sine+ramp-mixed" "ramp-sine+ramp-mixed"
## Choices for A_type are: "constant", "time-varying"
## save_fig can be True or False

plot_error_comparison_unknown_b_k("ramp", "constant", save_fig=False)

# 2) plot_error_comparison_known_b_k plots the data from when b_k is known. Takes b_type and save_fig as argument. Same as before
 ## b_type "sine^2-sine" is "sine-sine^2" for this case!

plot_error_comparison_known_b_k("ramp", save_fig=False)

# 3) Plots the evolution of our delta error and the error we are using to obtain the best estimate for coefficients
# Also marks the indices where we compute controller coefficients. 
## Only works for case where b_k is known. Takes b_type xrange and savefig as argument
## xrange is integer denoting number of timesteps which should be plotted
plot_error_comparison_RLS("ramp", 2000 , save_fig=False)