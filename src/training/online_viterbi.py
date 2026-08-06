"""
Online (forward, no-backtrack) Viterbi decoder -- a 1:1 mirror of the Monkey C
ViterbiDecoder that runs on the Garmin watch. Use THIS for eval, not the offline
full-Viterbi: the device sees one window at a time and can never revise a label
it already emitted, so offline accuracy overstates real deploy accuracy.

Faithfulness contract (must match ViterbiDecoder.mc exactly):
  * EMA smoothing:      smoothed = alpha*prob + (1-alpha)*smoothed_prev, init 1/N
  * log floor:          1e-6
  * min-dwell chain:    exercise = D substates, must walk step 0..D-1; the final
                        step (D-1) may self-loop to extend the exercise
  * REST:               1 substate, self-loop or advance from s-1
  * skip:               only into an exercise's step 0, from s-2 (skip a REST)
  * normalization:      live states shifted so max->0; DEAD states pinned at NEG
                        (do NOT shift the sentinel -- that re-enables dead paths)
  * reported index:     clamped monotonically (a chipper never goes backwards)

Numerics: the watch uses 32-bit Float. Pass probs as float32 (see run_online_decode)
if you want to reproduce device rounding to the last decimal; float64 will match
argmax in virtually all cases.
"""

import math

NEG = -1_000_000.0     # sentinel: unreachable / dead sub-state
REACH = -999_999.0     # reachability threshold (strictly greater than NEG)
REST_ALIASES = ("REST", "Rest", "rest")
UNKNOWN_LABEL_LOGP = -13.8


def is_rest(name):
    return name in REST_ALIASES


def build_rest_interleaved_sequence(exercise_names, rounds=1, rest_label="REST"):
    """[REST, ex1, REST, ex2, ..., exN, REST], repeated `rounds` times.
    exercise_names must use the SAME strings as `labels` (prob columns)."""
    seq = [rest_label]
    for _ in range(rounds):
        for ex in exercise_names:
            seq.append(ex)
            seq.append(rest_label)
    return seq


class OnlineViterbiDecoder:
    def __init__(self, seq, labels, dwell_seconds=5.0, step_size_sec=0.5,
                 transition_penalty=3.0, skip_penalty=7.0, alpha=0.4):
        self.seq = list(seq)
        self.labels = list(labels)
        self.transition_penalty = float(transition_penalty)
        self.skip_penalty = float(skip_penalty)
        self.alpha = float(alpha)

        self.dwell_windows = max(1, int(dwell_seconds / step_size_sec))

        # count expanded sub-states
        self.num = 0
        for name in self.seq:
            self.num += 1 if is_rest(name) else self.dwell_windows

        self.dp = [NEG] * self.num
        self.next_dp = [NEG] * self.num
        self.sub_label = [-1] * self.num
        self.sub_seq = [0] * self.num
        self.sub_step = [0] * self.num
        self.sub_is_ex = [False] * self.num

        n = len(self.labels)
        self.smoothed = [1.0 / n] * n
        self.log_probs = [0.0] * n

        self.init_error = None
        k = 0
        for i, name in enumerate(self.seq):
            li = self._find_label(name)
            rest = is_rest(name)
            if (not rest) and li == -1:
                self.init_error = "seq label not in model labels: %r" % (name,)
            count = 1 if rest else self.dwell_windows
            for step in range(count):
                self.sub_label[k] = li
                self.sub_seq[k] = i
                self.sub_step[k] = step
                self.sub_is_ex[k] = (not rest)
                k += 1

        if self.num > 0:
            self.dp[0] = 0.0
        self.max_reported_seq = 0

    def _find_label(self, name):
        for j, l in enumerate(self.labels):
            if l == name:
                return j
        return -1

    def update(self, probs):
        a = self.alpha
        for l in range(len(self.labels)):
            self.smoothed[l] = a * probs[l] + (1.0 - a) * self.smoothed[l]
            p = self.smoothed[l]
            if p < 1e-6:
                p = 1e-6
            self.log_probs[l] = math.log(p)

        dp, nxt = self.dp, self.next_dp
        D = self.dwell_windows
        tp, sp = self.transition_penalty, self.skip_penalty

        for s in range(self.num):
            li = self.sub_label[s]
            obs = self.log_probs[li] if li != -1 else UNKNOWN_LABEL_LOGP
            step = self.sub_step[s]
            best = NEG

            if not self.sub_is_ex[s]:
                # REST: self-loop or advance from previous finished block
                sc = dp[s] + obs
                if sc > best:
                    best = sc
                if s > 0 and dp[s - 1] > REACH:
                    sc = dp[s - 1] - tp + obs
                    if sc > best:
                        best = sc
            else:
                if step > 0:
                    # intermediate dwell step: MUST chain from step-1
                    if dp[s - 1] > REACH:
                        best = dp[s - 1] + obs
                else:
                    # exercise entry (step 0): advance from s-1, or skip a REST from s-2
                    if s > 0 and dp[s - 1] > REACH:
                        sc = dp[s - 1] - tp + obs
                        if sc > best:
                            best = sc
                    if s > 1 and dp[s - 2] > REACH:
                        sc = dp[s - 2] - sp + obs
                        if sc > best:
                            best = sc
                if step == D - 1:
                    # final step self-loop: extend the exercise past the minimum
                    sc = dp[s] + obs
                    if sc > best:
                        best = sc

            nxt[s] = best

        # normalize: shift live states so max -> 0, PIN dead states at NEG
        mx = NEG
        for s in range(self.num):
            if nxt[s] > mx:
                mx = nxt[s]
        for s in range(self.num):
            if nxt[s] <= REACH:
                dp[s] = NEG
            else:
                dp[s] = nxt[s] - mx

    def current_seq_index(self, monotonic=True):
        mx, arg = NEG, 0
        for s in range(self.num):
            if self.dp[s] > mx:
                mx, arg = self.dp[s], s
        idx = self.sub_seq[arg]
        if monotonic:
            if idx < self.max_reported_seq:
                idx = self.max_reported_seq
            else:
                self.max_reported_seq = idx
        return idx

    def current_exercise(self, monotonic=True):
        if not self.seq:
            return "REST"
        return self.seq[self.current_seq_index(monotonic)]


def run_online_decode(probs, seq_names, labels, dwell_seconds=5.0,
                      step_size_sec=0.5, transition_penalty=3.0,
                      skip_penalty=7.0, alpha=0.4, monotonic=True,
                      as_float32=False):
    """
    probs:     (T, C) array, columns aligned to `labels`.
    seq_names: plan positions incl. REST (build_rest_interleaved_sequence).
    labels:    class-name list in prob-column order (e.g. list(le_fold.classes_)).
    Returns:   list of predicted class-name per window (length T).
    """
    if as_float32:
        import numpy as np
        probs = np.asarray(probs, dtype=np.float32)

    dec = OnlineViterbiDecoder(seq_names, labels, dwell_seconds, step_size_sec,
                               transition_penalty, skip_penalty, alpha)
    if dec.init_error:
        raise ValueError(dec.init_error)

    preds = []
    for t in range(len(probs)):
        row = probs[t]
        dec.update([float(row[c]) for c in range(len(labels))])
        preds.append(dec.current_exercise(monotonic=monotonic))
    return preds


# ===========================================================================
#  LOSO wiring -- replace the offline `decode_viterbi` scoring for the watch
#  metric. Put this where you currently build y_pred_labels for the viterbi
#  variant, per fold:
# ===========================================================================
"""
from online_viterbi import build_rest_interleaved_sequence, run_online_decode

labels = list(le_fold.classes_)                    # prob-column order
seq_names = build_rest_interleaved_sequence(        # wod_sequence = exercise
    wod_sequence, rounds=1, rest_label='REST')      #   canonical names, in order

y_pred_labels = np.array(run_online_decode(
    all_probs, seq_names, labels,
    dwell_seconds=5.0, step_size_sec=CONFIG['step_size_sec'],
    transition_penalty=3.0, skip_penalty=7.0, alpha=0.4,
))
# then feed y_pred_labels into calculate_wod_metrics exactly as before.
"""
