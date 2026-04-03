import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

###################################################
# Some parameters #
###################################################

# define the interval x=0...2*pi with xn points
xn = 10000
x = np.linspace(0, 2 * np.pi, xn)

# pulse number (number of carrier periods per fundamental period)
N = 10

###################################################
# Helper functions / function definitions #
###################################################


# generate a sawtooth carrier sequence c(w*t) between +1, -1
def c_saw(wt, N):
    return 1 - 2 * np.abs(np.modf((wt * N) / (2 * np.pi))[0])


# generate a triangular carrier sequence c(w*t) between +1, -1
def c_tri(wt, N):
    return 4 * (np.abs(np.modf((wt * N) / (2 * np.pi))[0] - 0.5)) - 1


# generate a fundamental and normalized sinusoidal reference d(t)
def d_sin(wt, phi=0):
    return np.sin(wt - phi)


# generate a fundamental and normalized sinusoidal reference d(t) with overmod. amplitude
def d_sin_overmod(wt, phi=0):
    return 1.18 * np.sin(wt - phi)


# apply min-max zero-sequence injection to three phase duty-cycle references
def min_max_zero_injection(d_a, d_b, d_c):
    d_stack = np.vstack((d_a, d_b, d_c))
    d_zero = 0.5 * (np.max(d_stack, axis=0) + np.min(d_stack, axis=0))
    return d_a - d_zero, d_b - d_zero, d_c - d_zero, d_zero


def clip_to_unit_interval(d):
    return np.clip(d, -1, 1)


# calculate complementary PWM-based switching signals
def s_comp(d, c):
    return np.where(d > c, 1, -1)


# calculate the integrated / summed error between the reference and the switching signal
def e(d, s, xn):
    return np.cumsum(d - s) / xn * 2 * np.pi


###################################################
# Complementary switching PWM example #
###################################################

c_example = c_tri(x, N)
d_a_example = 1.4 * d_sin(x)
d_b_example = 1.4 * d_sin(x, np.pi / 3 * 2)
d_c_example = 1.4 * d_sin(x, np.pi / 3 * 4)
d_a_mod_example, d_b_mod_example, d_c_mod_example, d_zero_example = min_max_zero_injection(
    d_a_example, d_b_example, d_c_example
)
d_a_mod_clip_example = clip_to_unit_interval(d_a_mod_example)
d_b_mod_clip_example = clip_to_unit_interval(d_b_mod_example)
d_c_mod_clip_example = clip_to_unit_interval(d_c_mod_example)
s_a_example = s_comp(d_a_mod_clip_example, c_example)
s_b_example = s_comp(d_b_mod_clip_example, c_example)
s_c_example = s_comp(d_c_mod_clip_example, c_example)

# e_comp_example = e(d_comp_example, (s_comp_example[0] - s_comp_example[1])/2, xn)

# Compute the derivative of the signal
s_diff = np.diff(s_a_example)

# Identify step indices (where derivative is non-zero)
step_indices = np.where(s_diff != 0)[0]

# Find the two nearest samples for each step
nearest_samples = [0]
for step in step_indices:
    nearest_samples.append(step)
    nearest_samples.append(step + 1)

s_diff = np.diff(s_b_example)
step_indices = np.where(s_diff != 0)[0]
for step in step_indices:
    nearest_samples.append(step)
    nearest_samples.append(step + 1)

s_diff = np.diff(s_c_example)
step_indices = np.where(s_diff != 0)[0]
for step in step_indices:
    nearest_samples.append(step)
    nearest_samples.append(step + 1)

# identify peaks of the carrier
upper_peaks, _ = find_peaks(c_example)
lower_peaks, _ = find_peaks(-c_example)

# combine the three sets of indices and add the first and last sample
idx_sum = np.unique(
    np.concatenate((nearest_samples, upper_peaks, lower_peaks, [0, len(x) - 1]))
)


# save the reduced data to a csv file
current_directory = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_directory, "PWM_three-phase_zero_injection_mod140_example.csv")
np.savetxt(
    save_path,
    np.column_stack(
        (
            x[idx_sum],
            s_a_example[idx_sum],
            s_b_example[idx_sum],
            s_c_example[idx_sum],
            d_a_example[idx_sum],
            d_b_example[idx_sum],
            d_c_example[idx_sum],
            d_a_mod_example[idx_sum],
            d_b_mod_example[idx_sum],
            d_c_mod_example[idx_sum],
            d_a_mod_clip_example[idx_sum],
            d_b_mod_clip_example[idx_sum],
            d_c_mod_clip_example[idx_sum],
            d_zero_example[idx_sum],
            c_example[idx_sum],
        )
    ),
    delimiter=",",
    header="wt, sa, sb, sc, d_a, d_b, d_c, d_a_mod, d_b_mod, d_c_mod, d_a_mod_clip, d_b_mod_clip, d_c_mod_clip, d_zero, c",
    comments="",
)


###################################################

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

axes[0].plot(x, c_example, color="black", label=r"$c(t)$")
axes[0].plot(x, d_a_example, label=r"$d_\mathrm{a}(t)$")
axes[0].plot(x, d_b_example, label=r"$d_\mathrm{b}(t)$")
axes[0].plot(x, d_c_example, label=r"$d_\mathrm{c}(t)$")
axes[0].set_ylabel("original")
axes[0].set_title("Three-phase PWM with min-max zero-sequence injection")
axes[0].grid()
axes[0].legend(loc="upper right")

axes[1].plot(x, c_example, color="black", label=r"$c(t)$")
axes[1].plot(x, d_a_mod_example, linestyle="--", label=r"$d_{\mathrm{a},\mathrm{mod}}(t)$")
axes[1].plot(x, d_b_mod_example, linestyle="--", label=r"$d_{\mathrm{b},\mathrm{mod}}(t)$")
axes[1].plot(x, d_c_mod_example, linestyle="--", label=r"$d_{\mathrm{c},\mathrm{mod}}(t)$")
axes[1].plot(x, d_a_mod_clip_example, label=r"$d_{\mathrm{a},\mathrm{mod,clip}}(t)$")
axes[1].plot(x, d_b_mod_clip_example, label=r"$d_{\mathrm{b},\mathrm{mod,clip}}(t)$")
axes[1].plot(x, d_c_mod_clip_example, label=r"$d_{\mathrm{c},\mathrm{mod,clip}}(t)$")
axes[1].plot(x, d_zero_example, linestyle="--", color="gray", label=r"$d_0(t)$")
axes[1].set_ylabel("modified")
axes[1].grid()
axes[1].legend(loc="upper right")

axes[2].plot(x, s_a_example, label=r"$s_\mathrm{a}(t)$")
axes[2].plot(x, s_b_example, label=r"$s_\mathrm{b}(t)$")
axes[2].plot(x, s_c_example, label=r"$s_\mathrm{c}(t)$")
axes[2].set_xlabel(r"$\omega t$")
axes[2].set_ylabel("switching")
axes[2].grid()
axes[2].legend(loc="upper right")

plt.tight_layout()
plt.show()
