import numpy as np
import pandas as pd
from enum import Enum

class RestPolicy(Enum):
    REQUIRE_REST = "require_rest"
    PREFER_REST = "prefer_rest"
    OFF = "off"
NEG = -1e18
class WodDecoder:
    def __init__(self, workout_sequence, label_encoder, rest_label="REST",
                 confidence_threshold=0.8, dwell_seconds=5.0, step_size_sec=0.5,
                 rollback_window_sec=15.0, rest_policy=RestPolicy.PREFER_REST):
        """
        workout_sequence: list of exercise labels (strings) in order.
        """
        self.workout_sequence = workout_sequence
        self.le = label_encoder
        self.rest_label = rest_label
        self.confidence_threshold = confidence_threshold
        self.dwell_seconds = dwell_seconds
        self.step_size_sec = step_size_sec
        self.dwell_windows = max(1, int(dwell_seconds / step_size_sec))
        self.rollback_window_sec = rollback_window_sec
        self.rollback_windows = int(rollback_window_sec / step_size_sec)
        self.rest_policy = rest_policy

        self.class_to_idx = {cls: idx for idx, cls in enumerate(label_encoder.classes_)}
        if rest_label not in self.class_to_idx:
            raise ValueError(f"Rest label '{rest_label}' not found in LabelEncoder classes: {label_encoder.classes_}")
        self.rest_idx = self.class_to_idx[rest_label]

        self.seq_indices = []
        for name in workout_sequence:
            if name not in self.class_to_idx:
                raise ValueError(f"Workout exercise '{name}' not found in LabelEncoder classes")
            self.seq_indices.append(self.class_to_idx[name])

    def decode_greedy_baseline(self, probs):
        """Standard argmax + smoothing (no WOD constraints)."""
        preds = np.argmax(probs, axis=1)
        # Simple smoothing
        smoothed = []
        for i in range(len(preds)):
            start = max(0, i - self.dwell_windows + 1)
            window = preds[start : i + 1]
            smoothed.append(np.bincount(window).argmax())
        return np.array(smoothed)

    def decode_greedy_wod(self, probs, use_rollback=False):
        """
        Greedy WOD-constrained decoder.
        If use_rollback is True, allows reverting to previous exercise.
        """
        T = len(probs)
        decoded = np.full(T, self.rest_idx)
        current_seq_idx = -1  # -1 means before first exercise
        
        # Track evidence for transitions
        # candidate_idx is the index in self.seq_indices
        evidence_count = 0
        candidate_seq_pos = 0
        
        # Rollback tracking
        last_transition_time = -1
        rollback_info = []

        # Current state: 
        # - exercise_active: None or index in seq_indices
        # - in_rest: True/False
        
        active_seq_pos = -1 # -1 means REST or start
        
        for t in range(T):
            # 1. Evaluate candidates
            # Potential next exercise:
            next_seq_pos = active_seq_pos + 1
            
            # Can we transition to next_seq_pos?
            if next_seq_pos < len(self.seq_indices):
                target_idx = self.seq_indices[next_seq_pos]
                
                # Check threshold
                # REST policy adjustment
                current_threshold = self.confidence_threshold
                if self.rest_policy == RestPolicy.PREFER_REST:
                    # If coming directly from another exercise without a recent REST commit
                    # actually the issue says "direct exercise -> exercise transitions allowed but require a higher confidence threshold"
                    if active_seq_pos != -1 and (t == 0 or decoded[t-1] != self.rest_idx):
                         current_threshold += 0.10
                elif self.rest_policy == RestPolicy.REQUIRE_REST:
                    if active_seq_pos != -1 and (t == 0 or decoded[t-1] != self.rest_idx):
                         current_threshold = 2.0 # Impossible

                # Evidence accumulation
                if probs[t, target_idx] >= current_threshold:
                    evidence_count += 1
                else:
                    evidence_count = 0 # Reset if interrupted by ANY other class?
                    # "Reset accumulated evidence to zero if interrupted by any other class."
                
                if evidence_count >= self.dwell_windows:
                    # Transition!
                    old_pos = active_seq_pos
                    active_seq_pos = next_seq_pos
                    evidence_count = 0
                    last_transition_time = t
                    # Back-fill dwell window
                    decoded[max(0, t - self.dwell_windows + 1) : t + 1] = self.seq_indices[active_seq_pos]
                    continue
            
            # 2. Rollback logic
            if use_rollback and active_seq_pos > 0:
                time_since_transition = (t - last_transition_time) * self.step_size_sec
                if time_since_transition <= self.rollback_window_sec:
                    prev_idx = self.seq_indices[active_seq_pos - 1]
                    # Check if previous exercise is showing strong evidence
                    # We need a separate evidence counter for rollback?
                    # The requirement says: "If exercise N re-accumulates >= CONFIDENCE_THRESHOLD over the dwell period 
                    # within that window, revert the transition"
                    
                    # For simplicity, let's check the last dwell_windows
                    if t >= self.dwell_windows:
                        window_probs = probs[t - self.dwell_windows + 1 : t + 1, prev_idx]
                        if np.all(window_probs >= self.confidence_threshold):
                            # ROLLBACK!
                            revert_duration = (t - last_transition_time)
                            rollback_info.append({
                                't': t * self.step_size_sec,
                                'from': self.le.inverse_transform([self.seq_indices[active_seq_pos]])[0],
                                'to': self.le.inverse_transform([prev_idx])[0],
                                'duration_restored': revert_duration * self.step_size_sec
                            })
                            
                            # Re-attribute
                            decoded[last_transition_time : t + 1] = prev_idx
                            active_seq_pos -= 1
                            last_transition_time = -1 # Disable further rollback for now? or keep it?
                            # Usually one rollback is enough, but sequence might have shifted.
                            continue

            # 3. Default behavior: maintain current or REST
            if active_seq_pos == -1:
                decoded[t] = self.rest_idx
            else:
                # If current exercise evidence is strong, keep it. 
                # If REST evidence is strong, switch to REST?
                # The WOD constraint usually means: Exercise N -> [REST] -> Exercise N+1
                # But we can stay in Exercise N as long as we want.
                
                curr_idx = self.seq_indices[active_seq_pos]
                if probs[t, self.rest_idx] > self.confidence_threshold:
                    decoded[t] = self.rest_idx
                else:
                    # Prefer current exercise unless REST is strong
                    decoded[t] = curr_idx

        return decoded, rollback_info






    # ===========================================================================
    #  WodDecoder method  (replaces your decode_viterbi)
    # ===========================================================================
    def decode_viterbi(self, probs, advancement_penalty=1.0, temperature=1.0,
                       min_dwell_frac=0.5, default_dwell_sec=4.0):
        """
        Min-dwell forced alignment.

        Design: emissions place the boundaries (advancement_penalty is small, ~0-2),
        while a PER-CLASS minimum dwell prevents both fragmentation AND the low-signal
        collapses (walking lunge -> REST). This decouples the two jobs the single
        scalar penalty was overloaded with, so you avoid the boundary lag you get from
        a high penalty.

        temperature: LEAVE AT 1.0 for this monotonic graph. Flattening emissions here
                     only adds boundary lag -- Viterbi's global optimality already
                     prevents single-window runaway advances.

        Needs self.class_floor = {exercise_class_idx: floor_seconds}, learned on the
        TRAIN sessions of the fold (see wiring below). Falls back to default_dwell_sec
        for any class without a learned floor.
        """
        step = self.step_size_sec
        floor = getattr(self, "class_floor", None) or {}

        # expanded plan:  REST, Ex1, REST, Ex2, ..., Ex_N, REST
        # REST positions are skippable with dwell 0 (chippers flow straight through).
        seq, dwell, skip = [self.rest_idx], [0], [True]
        for ex_idx in self.seq_indices:
            f = floor.get(ex_idx, default_dwell_sec)
            dwell_windows = max(1, int(round(min_dwell_frac * f / step)))
            seq.append(ex_idx);
            dwell.append(dwell_windows);
            skip.append(False)
            seq.append(self.rest_idx);
            dwell.append(0);
            skip.append(True)

        log_emis = to_log_emissions(probs, temperature=temperature)
        # switch_penalty in forced_align_viterbi is your advancement_penalty
        return forced_align_viterbi(log_emis, seq, dwell, skippable=skip,
                                    switch_penalty=advancement_penalty)

    # ===========================================================================
    #  LOSO wiring  (inside run_loso_dt, per fold, AFTER le_fold is built and
    #  BEFORE you construct/decode with the WodDecoder)
    # ===========================================================================
    """
    train_pairs = [
        (df['norm_label'].values, df['rel_time'].values)   # ground-truth ribbons
        for df in train_dfs                                # TRAIN sessions only -> no leakage
    ]
    floor_by_name = learn_class_floor(train_pairs, pct=20)   # {label_string: seconds}

    # map label strings -> the integer class indices the decoder uses
    decoder.class_floor = {
        int(le_fold.transform([name])[0]): sec
        for name, sec in floor_by_name.items()
        if name in le_fold.classes_
    }
    """

    # ===========================================================================
    #  Suggested first config to try
    #    temperature          = 1.0     (revert -- this is what cost you 0.87 -> 0.77)
    #    advancement_penalty  = 1.0     (sweep 0.0 .. 3.0)
    #    min_dwell_frac       = 0.5     (0.4 .. 0.7; higher = holds longer, risks
    #                                    over-running short segments)
    # ===========================================================================


# --------------------------------------------------------------------------- #
# forced-alignment Viterbi with min-dwell + skippable positions                #
# --------------------------------------------------------------------------- #
def forced_align_viterbi(log_emis, seq_states, min_dwell, skippable=None,
                         switch_penalty=0.0):
    log_emis = np.asarray(log_emis)
    T = log_emis.shape[0]
    K = len(seq_states)
    if skippable is None:
        skippable = [False] * K
    skip_arr = np.asarray(skippable, dtype=bool)

    sizes = np.maximum(1, np.asarray(min_dwell, dtype=int))  # phases per position
    offsets = np.zeros(K, dtype=int)
    for k in range(1, K):
        offsets[k] = offsets[k - 1] + sizes[k - 1]
    S = int(offsets[-1] + sizes[-1])
    last = sizes - 1
    free_of = offsets + last  # free-phase state per pos

    state_class = np.empty(S, dtype=int)
    state_k = np.empty(S, dtype=int)
    state_ph = np.empty(S, dtype=int)
    for k in range(K):
        for ph in range(sizes[k]):
            s = offsets[k] + ph
            state_class[s], state_k[s], state_ph[s] = seq_states[k], k, ph

    idx = np.arange(S)
    pred_within = np.where(state_ph > 0, idx - 1, -1)  # inside min-dwell
    pred_self = np.where(state_ph == last[state_k], idx, -1)  # free-phase self-loop
    pred_enter = np.where((state_ph == 0) & (state_k > 0),
                          free_of[np.clip(state_k - 1, 0, K - 1)], -1)  # from prev position
    prev_skip = np.zeros(S, dtype=bool)
    m = state_k >= 1
    prev_skip[m] = skip_arr[state_k[m] - 1]
    pred_skip = np.where((state_ph == 0) & (state_k >= 2) & prev_skip,  # bypass skippable prev
                         free_of[np.clip(state_k - 2, 0, K - 1)], -1)
    preds = np.vstack([pred_within, pred_self, pred_enter, pred_skip])
    pen = np.array([0.0, 0.0, switch_penalty, switch_penalty])[:, None]

    def gather(vec, ids):
        out = np.full(S, NEG)
        ok = ids >= 0
        out[ok] = vec[ids[ok]]
        return out

    dp = np.full((T, S), NEG)
    bp = np.full((T, S), -1, dtype=int)
    dp[0, offsets[0]] = log_emis[0, seq_states[0]]
    if skip_arr[0] and K > 1:  # allow starting mid-first-exercise
        dp[0, offsets[1]] = log_emis[0, seq_states[1]] - switch_penalty

    for t in range(1, T):
        prev = dp[t - 1]
        cands = np.vstack([gather(prev, preds[j]) for j in range(4)]) - pen
        arg = np.argmax(cands, axis=0)
        dp[t] = cands[arg, idx] + log_emis[t, state_class]
        bp[t] = preds[arg, idx]

    # terminate in the last position's free phase (or 2nd-last if trailing REST skippable)
    term = [free_of[K - 1]]
    if skip_arr[K - 1] and K >= 2:
        term.append(free_of[K - 2])
    s = int(term[int(np.argmax([dp[T - 1, c] for c in term]))])

    path = np.empty(T, dtype=int)
    for t in range(T - 1, -1, -1):
        path[t] = state_class[s]
        ns = bp[t, s]
        if ns < 0 and t > 0:
            path[:t] = state_class[s]
            break
        s = ns
    return path


def to_log_emissions(probs, temperature=1.0, floor=1e-6):
    """temperature > 1 flattens overconfident deep-tree probs so order/dwell
    constraints can take effect. Try 1.5-3.0 for a depth-15 tree."""
    p = np.clip(probs, floor, 1.0)
    return np.log(p) / max(temperature, 1e-6)