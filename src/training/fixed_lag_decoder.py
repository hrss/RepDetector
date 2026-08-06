"""
Fixed-lag smoothing over the SAME min-dwell trellis as the online device decoder.

Why: forward-only argmax (OnlineViterbiDecoder) is a filtering estimate -- it
commits to window t with no lookahead. Fixed-lag emits the label for window t-L
after seeing L more windows, i.e. it backtracks L columns from the current best
state. This interpolates between:
    lag = 0     -> identical to the forward-only device decoder
    lag = None  -> full offline Viterbi (your 0.87 backtracked number)
so sweeping L shows exactly how much display latency buys how much accuracy.

Latency of a given L = L * step_size_sec seconds.

Transition logic, EMA, penalties and normalization are byte-identical to
online_viterbi / ViterbiDecoder.mc, so lag=0 here == the device output.
"""

import math
from online_viterbi import (
    build_rest_interleaved_sequence, is_rest, NEG, REACH, UNKNOWN_LABEL_LOGP,
)


def _build_topology(seq, labels, dwell_windows):
    sub_label, sub_seq, sub_step, sub_is_ex = [], [], [], []
    err = None

    def find(name):
        for j, l in enumerate(labels):
            if l == name:
                return j
        return -1

    for i, name in enumerate(seq):
        rest = is_rest(name)
        li = find(name)
        if (not rest) and li == -1:
            err = "seq label not in model labels: %r" % (name,)
        for step in range(1 if rest else dwell_windows):
            sub_label.append(li)
            sub_seq.append(i)
            sub_step.append(step)
            sub_is_ex.append(not rest)
    return sub_label, sub_seq, sub_step, sub_is_ex, err


def _forward(probs, seq, labels, dwell_windows,
             transition_penalty, skip_penalty, alpha):
    """Forward pass storing per-window dp columns and backpointers."""
    sub_label, sub_seq, sub_step, sub_is_ex, err = _build_topology(
        seq, labels, dwell_windows)
    if err:
        raise ValueError(err)

    num, n = len(sub_label), len(labels)
    smoothed = [1.0 / n] * n
    dp = [NEG] * num
    if num > 0:
        dp[0] = 0.0
    D, tp, sp = dwell_windows, transition_penalty, skip_penalty

    dp_hist, bp_hist = [], []
    for t in range(len(probs)):
        row = probs[t]
        logp = [0.0] * n
        for l in range(n):
            smoothed[l] = alpha * float(row[l]) + (1.0 - alpha) * smoothed[l]
            p = smoothed[l] if smoothed[l] >= 1e-6 else 1e-6
            logp[l] = math.log(p)

        nxt = [NEG] * num
        bp = [-1] * num
        for s in range(num):
            li = sub_label[s]
            obs = logp[li] if li != -1 else UNKNOWN_LABEL_LOGP
            step = sub_step[s]
            best, arg = NEG, -1

            if not sub_is_ex[s]:
                sc = dp[s] + obs
                if sc > best:
                    best, arg = sc, s
                if s > 0 and dp[s - 1] > REACH:
                    sc = dp[s - 1] - tp + obs
                    if sc > best:
                        best, arg = sc, s - 1
            else:
                if step > 0:
                    if dp[s - 1] > REACH:
                        best, arg = dp[s - 1] + obs, s - 1
                else:
                    if s > 0 and dp[s - 1] > REACH:
                        sc = dp[s - 1] - tp + obs
                        if sc > best:
                            best, arg = sc, s - 1
                    if s > 1 and dp[s - 2] > REACH:
                        sc = dp[s - 2] - sp + obs
                        if sc > best:
                            best, arg = sc, s - 2
                if step == D - 1:
                    sc = dp[s] + obs
                    if sc > best:
                        best, arg = sc, s

            nxt[s], bp[s] = best, arg

        mx = NEG
        for s in range(num):
            if nxt[s] > mx:
                mx = nxt[s]
        for s in range(num):
            dp[s] = NEG if nxt[s] <= REACH else nxt[s] - mx

        dp_hist.append(dp[:])
        bp_hist.append(bp)

    return dp_hist, bp_hist, sub_seq


def decode_fixed_lag(probs, seq_names, labels, lag=6, dwell_seconds=5.0,
                     step_size_sec=0.5, transition_penalty=3.0,
                     skip_penalty=7.0, alpha=0.4):
    """
    lag (windows): 0 == forward-only device decoder, None == full offline Viterbi.
    Returns per-window predicted class-name (length T).
    """
    D = max(1, int(dwell_seconds / step_size_sec))
    dp_hist, bp_hist, sub_seq = _forward(
        probs, seq_names, labels, D, transition_penalty, skip_penalty, alpha)
    T = len(dp_hist)
    if T == 0:
        return []
    L = (T - 1) if lag is None else int(lag)

    preds = []
    for t in range(T):
        tt = min(t + L, T - 1)
        # argmax state at the lookahead column
        s, mx = 0, NEG
        for i, v in enumerate(dp_hist[tt]):
            if v > mx:
                mx, s = v, i
        # backtrack to column t
        for k in range(tt, t, -1):
            s = bp_hist[k][s]
            if s < 0:
                break
        preds.append(seq_names[sub_seq[s]] if s >= 0 else "REST")
    return preds


def sweep_lag(probs, seq_names, labels, y_true_names,
              lags=(0, 2, 4, 6, 8, 12, 20, None), **kw):
    """Returns {lag: window_accuracy}. lag=0 is your on-watch number,
    lag=None is full offline Viterbi. Latency(L) = L * step_size_sec seconds."""
    import numpy as np
    y = np.asarray(y_true_names)
    out = {}
    for L in lags:
        pred = np.asarray(decode_fixed_lag(probs, seq_names, labels, lag=L, **kw))
        out[L] = float((pred == y).mean())
    return out


# ===========================================================================
#  Usage in LOSO (per fold):
#
#   from fixed_lag_viterbi import sweep_lag, decode_fixed_lag
#   labels = list(le_fold.classes_)
#   seq_names = build_rest_interleaved_sequence(wod_sequence)
#   print(sweep_lag(all_probs, seq_names, labels, y_test_labels,
#                   step_size_sec=CONFIG['step_size_sec']))
#   # pick L at the knee, then that same L is what you port to the watch.
# ===========================================================================