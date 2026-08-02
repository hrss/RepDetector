import numpy as np
import pandas as pd
from enum import Enum

class RestPolicy(Enum):
    REQUIRE_REST = "require_rest"
    PREFER_REST = "prefer_rest"
    OFF = "off"

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

    def decode_viterbi(self, probs, advancement_penalty=5.0, skip_penalty=100.0, self_transition_bonus=0.0):
        """
        Viterbi decoder over sequence positions.
        States: S_0, S_1, ..., S_{2N} where S_{2i} is REST and S_{2i+1} is Ex_{i+1}
        """
        N = len(self.seq_indices)
        num_states = 2 * N + 1
        T = len(probs)
        
        # log_probs
        log_probs = np.log(probs + 1e-12)
        
        # dp[t, state]
        dp = np.full((T, num_states), -np.inf)
        backtrack = np.zeros((T, num_states), dtype=int)
        
        # State mapping:
        # even j: REST (mapped to rest_idx)
        # odd j: Exercise (j-1)//2 (mapped to seq_indices[(j-1)//2])
        
        def get_class_idx(state_idx):
            if state_idx % 2 == 0:
                return self.rest_idx
            else:
                return self.seq_indices[(state_idx - 1) // 2]

        # Initial state: must start at REST(0) or Ex(1)
        dp[0, 0] = log_probs[0, self.rest_idx]
        dp[0, 1] = log_probs[0, self.seq_indices[0]]
        
        for t in range(1, T):
            for s in range(num_states):
                class_idx = get_class_idx(s)
                obs_log_prob = log_probs[t, class_idx]
                
                # Transitions to s can come from s, s-1, or s-2 (if we allow skipping REST)
                # Advancement penalty
                
                # Option 1: Stay in s
                score_stay = dp[t-1, s] + self_transition_bonus
                
                # Option 2: Advance from s-1
                score_adv1 = -np.inf
                if s > 0:
                    score_adv1 = dp[t-1, s-1] - advancement_penalty
                
                # Option 3: Advance from s-2 (e.g. Ex -> Ex skipping REST)
                score_adv2 = -np.inf
                if s > 1:
                    score_adv2 = dp[t-1, s-2] - advancement_penalty * 1.5
                
                # Forbid skips > 2
                
                best_prev = s
                best_score = score_stay
                
                if score_adv1 > best_score:
                    best_score = score_adv1
                    best_prev = s - 1
                
                if score_adv2 > best_score:
                    best_score = score_adv2
                    best_prev = s - 2
                    
                dp[t, s] = best_score + obs_log_prob
                backtrack[t, s] = best_prev
                
        # Termination: can end in any state, but usually the last ones
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(dp[T-1, :])
        
        for t in range(T-2, -1, -1):
            path[t] = backtrack[t+1, path[t+1]]
            
        decoded = np.array([get_class_idx(s) for s in path])
        return decoded
