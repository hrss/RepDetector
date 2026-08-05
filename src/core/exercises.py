

"""
Label canonicalization — FIXED.

Replaces the SYNONYM_TO_CANONICAL construction and canonicalize_label() at the
bottom of src/core/exercises.py. Keep EXERCISES_DATA exactly as it is and paste
this below it (or import EXERCISES_DATA from wherever it lives).

WHAT WAS BROKEN
---------------
1. "setup" / "null" returned "REST" instead of None, so setup periods (walking
   to the rig, picking up equipment, adjusting the watch) became REST training
   data. This both inflated the majority class and taught the model that real
   motion is REST.

2. The substring heuristic `if canonical.lower() in ls.replace(" ", "_")`
   scanned a ~170-key dict and took the FIRST match, so results depended on dict
   insertion order. Concretely: "chest_to_wall_hspu" hit HSPU (defined earlier)
   instead of CHEST_TO_WALL_HSPU, silently moving that data out of Push-up.

3. Unmapped labels fell through to `label_str.upper().replace(" ", "_")`,
   silently minting new singleton classes instead of failing. Macro-F1 averages
   over classes, so a 3-window junk class counts as much as a 3000-window one.

WHAT THIS DOES INSTEAD
----------------------
  normalize text -> IGNORE (None) -> REST -> explicit override -> exact synonym
  -> parenthetical-stripped synonym -> raise (or passthrough if strict=False)

No order-dependent substring matching anywhere.
"""

from __future__ import annotations

import re
import warnings
from typing import Optional
import json
import os
import re

from whoosh.analysis import RegexTokenizer, LowercaseFilter
from whoosh.fields import ID, KEYWORD, Schema, TEXT
from whoosh.filedb.filestore import RamStorage
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.query import Term


# from .exercises import EXERCISES_DATA   # keep your existing dict

EXERCISES_DATA = {
  "PULL_UP": {
    "synonyms": [
      "pull up",
      "pull-up",
      "pu",
      "strict pull up",
      "strict pu",
      "kipping pull up",
      "kpu",
      "butterfly pull up",
      "bpu",
      "chin up",
      "chin-up",
      "c2b (strict) pull up",
      "weighted pull up"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "CHEST_TO_BAR": {
    "synonyms": [
      "chest to bar",
      "chest-to-bar",
      "ctb",
      "ctb pull up",
      "chest to bar pull up"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "BAR_MUSCLE_UP": {
    "synonyms": [
      "bar muscle up",
      "bar muscle-up",
      "bmu",
      "bmup"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "RING_MUSCLE_UP": {
    "synonyms": [
      "ring muscle up",
      "ring muscle-up",
      "rmu",
      "rmup"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "MUSCLE_UP": {
    "synonyms": [
      "muscle up",
      "mu",
      "bar muscle up",
      "bmu",
      "ring muscle up",
      "rmu"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "TOES_TO_BAR": {
    "synonyms": [
      "toes to bar",
      "toes-to-bar",
      "t2b",
      "ttb"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "TOES_TO_RINGS": {
    "synonyms": [
      "toes to rings",
      "toes-to-rings",
      "t2r"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "KNEES_TO_ELBOWS": {
    "synonyms": [
      "knees to elbows",
      "k2e",
      "knees to chest",
      "k2c"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "PUSH_UP": {
    "synonyms": [
      "push up",
      "push-up",
      "strict push up",
      "hand release push up",
      "hrpu",
      "ring push up"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "HAND_RELEASE_PUSH_UP": {
    "synonyms": [
      "hand release push up",
      "hand-release push-up",
      "hrpu"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "DIP": {
    "synonyms": [
      "dip",
      "ring dip",
      "bar dip"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "RING_ROW": {
    "synonyms": [
      "ring row",
      "rr"
    ],
    "categories": [
      "cardio"
    ]
  },
  "PISTOL_SQUAT": {
    "synonyms": [
      "pistol",
      "pistol squat",
      "single leg squat",
      "sl squat",
      "pistols"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "AIR_SQUAT": {
    "synonyms": [
      "air squat",
      "as",
      "bodyweight squat"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "SIT_UP": {
    "synonyms": [
      "sit up",
      "sit-up",
      "abmat sit up",
      "abmat sit-up",
      "su"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "GHD_SIT_UP": {
    "synonyms": [
      "ghd sit up",
      "ghd sit-up",
      "glute ham sit up",
      "ghd su"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "HIP_EXTENSION": {
    "synonyms": [
      "hip extension",
      "back extension",
      "ghd hip extension"
    ],
    "categories": []
  },
  "SUPERMAN": {
    "synonyms": [
      "superman"
    ],
    "categories": []
  },
  "HOLLOW_ROCK": {
    "synonyms": [
      "hollow rock",
      "hollow hold"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "V_UP": {
    "synonyms": [
      "v up",
      "v-up",
      "jackknife"
    ],
    "categories": []
  },
  "HSW": {
    "synonyms": [
      "handstand walk",
      "hsw",
      "handstand walking"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "HANDSTAND_HOLD": {
    "synonyms": [
      "handstand hold",
      "hs hold"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "WALL_WALK": {
    "synonyms": [
      "wall walk",
      "wall walks",
      "ww"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "ROPE_CLIMB": {
    "synonyms": [
      "rope climb",
      "rc",
      "legless rope climb",
      "lrc",
      "legless rc",
      "rope ascent"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "PEGBOARD_ASCENT": {
    "synonyms": [
      "pegboard ascent",
      "pegboard",
      "pba"
    ],
    "categories": []
  },
  "ROPE_PULL": {
    "synonyms": [
      "rope pull",
      "seated rope pull"
    ],
    "categories": []
  },
  "BURPEE": {
    "synonyms": [
      "burpee",
      "burpees",
      "bur",
      "bu",
      "bar facing burpee",
      "bfb",
      "bar-facing burpee",
      "lateral burpee",
      "burpee to target",
      "burpee over bar",
      "bobr",
      "burpee box jump over",
      "bbjo",
      "burpee box get over",
      "bbgo"
    ],
    "categories": [
      "cardio",
      "gymnastics",
      "plyometrics"
    ]
  },
  "BOX_JUMP": {
    "synonyms": [
      "box jump",
      "box jumps",
      "bj",
      "bjs",
      "box jump over",
      "bjo",
      "box step up",
      "bsu",
      "box step-over",
      "bso"
    ],
    "categories": [
      "plyometrics"
    ]
  },
  "WALL_BALL": {
    "synonyms": [
      "wall ball",
      "wb",
      "wall-ball shot",
      "medicine ball shot"
    ],
    "categories": []
  },
  "DOUBLE_UNDER": {
    "synonyms": [
      "double under",
      "double-unders",
      "double-under",
      "du"
    ],
    "categories": [
      "cardio"
    ]
  },
  "SINGLE_UNDER": {
    "synonyms": [
      "single under",
      "single-unders",
      "su"
    ],
    "categories": [
      "cardio"
    ]
  },
  "TRIPLE_UNDER": {
    "synonyms": [
      "triple under",
      "triple-unders",
      "tu"
    ],
    "categories": [
      "cardio"
    ]
  },
  "LUNGE": {
    "synonyms": [
      "lunge",
      "lunges",
      "walking lunge",
      "wl",
      "front rack lunge",
      "fr lunge",
      "overhead lunge",
      "oh lunge",
      "reverse lunge",
      "rl"
    ],
    "categories": []
  },
  "BEAR_CRAWL": {
    "synonyms": [
      "bear crawl",
      "bc"
    ],
    "categories": []
  },
  "SHUTTLE_RUN": {
    "synonyms": [
      "shuttle run",
      "shuttle sprints",
      "sr"
    ],
    "categories": [
      "cardio"
    ]
  },
  "RUN": {
    "synonyms": [
      "run",
      "running",
      "sprint",
      "jog"
    ],
    "categories": [
      "cardio"
    ]
  },
  "ROW": {
    "synonyms": [
      "row",
      "rowing",
      "c2 row",
      "row erg",
      "erg row"
    ],
    "categories": [
      "cardio"
    ]
  },
  "SKI_ERG": {
    "synonyms": [
      "ski erg",
      "skierg",
      "ski-erg",
      "ski"
    ],
    "categories": [
      "cardio"
    ]
  },
  "BIKE_ERG": {
    "synonyms": [
      "bike erg",
      "concept2 bike"
    ],
    "categories": [
      "cardio"
    ]
  },
  "ASSAULT_BIKE": {
    "synonyms": [
      "air bike",
      "assault bike",
      "echo bike",
      "aab",
      "ab",
      "airdyne"
    ],
    "categories": [
      "cardio"
    ]
  },
  "SWIM": {
    "synonyms": [
      "swim",
      "swimming"
    ],
    "categories": [
      "cardio"
    ]
  },
  "DUCK_WALK": {
    "synonyms": [
      "duck walk"
    ],
    "categories": []
  },
  "WALL_SIT": {
    "synonyms": [
      "wall sit"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "PUSH_UP_TO_PLANK": {
    "synonyms": [
      "plank",
      "forearm plank"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "RING_SUPPORT": {
    "synonyms": [
      "ring support hold",
      "support hold"
    ],
    "categories": []
  },
  "SNATCH": {
    "synonyms": [
      "snatch",
      "sn",
      "squat snatch",
      "full snatch"
    ],
    "categories": [
      "barbell",
      "weightlifting"
    ]
  },
  "POWER_SNATCH": {
    "synonyms": [
      "power snatch",
      "psn"
    ],
    "categories": [
      "barbell",
      "weightlifting"
    ]
  },
  "HANG_SNATCH": {
    "synonyms": [
      "hang snatch",
      "hsn",
      "hang power snatch",
      "hpsn",
      "hang squat snatch",
      "hssn"
    ],
    "categories": [
      "barbell",
      "weightlifting"
    ]
  },
  "OVERHEAD_SQUAT": {
    "synonyms": [
      "overhead squat",
      "ohs"
    ],
    "categories": [
      "barbell"
    ]
  },
  "CLEAN_AND_JERK": {
    "synonyms": [
      "clean and jerk",
      "c&j",
      "cj"
    ],
    "categories": [
      "barbell",
      "weightlifting"
    ]
  },
  "CLEAN": {
    "synonyms": [
      "clean",
      "squat clean",
      "full clean",
      "cln"
    ],
    "categories": [
      "barbell",
      "weightlifting"
    ]
  },
  "POWER_CLEAN": {
    "synonyms": [
      "power clean",
      "pc"
    ],
    "categories": [
      "barbell",
      "weightlifting"
    ]
  },
  "HANG_CLEAN": {
    "synonyms": [
      "hang clean",
      "hc",
      "hang power clean",
      "hpc",
      "hang squat clean",
      "hsc"
    ],
    "categories": [
      "barbell",
      "weightlifting"
    ]
  },
  "JERK": {
    "synonyms": [
      "jerk",
      "push jerk",
      "pj",
      "split jerk",
      "sj",
      "sjerk",
      "power jerk"
    ],
    "categories": [
      "barbell",
      "weightlifting"
    ]
  },
  "SHOULDER_TO_OVERHEAD": {
    "synonyms": [
      "shoulder to overhead",
      "s2oh",
      "stoh",
      "shoulder-to-overhead"
    ],
    "categories": [
      "barbell"
    ]
  },
  "PUSH_PRESS": {
    "synonyms": [
      "push press",
      "pp"
    ],
    "categories": [
      "barbell"
    ]
  },
  "STRICT_PRESS": {
    "synonyms": [
      "strict press",
      "shoulder press",
      "press",
      "sp",
      "military press"
    ],
    "categories": [
      "barbell"
    ]
  },
  "THRUSTER": {
    "synonyms": [
      "thruster",
      "barbell thruster"
    ],
    "categories": [
      "barbell"
    ]
  },
  "SQUAT": {
    "synonyms": [
      "squat",
      "back squat",
      "bs",
      "front squat",
      "fs",
      "overhead squat",
      "ohs"
    ],
    "categories": [
      "barbell"
    ]
  },
  "FRONT_SQUAT": {
    "synonyms": [
      "front squat",
      "fs"
    ],
    "categories": [
      "barbell"
    ]
  },
  "BACK_SQUAT": {
    "synonyms": [
      "back squat",
      "bs"
    ],
    "categories": [
      "barbell"
    ]
  },
  "DEADLIFT": {
    "synonyms": [
      "deadlift",
      "dl",
      "conventional deadlift"
    ],
    "categories": [
      "barbell"
    ]
  },
  "SUMO_DEADLIFT": {
    "synonyms": [
      "sumo deadlift",
      "sdl"
    ],
    "categories": [
      "barbell"
    ]
  },
  "SUMO_DEADLIFT_HIGH_PULL": {
    "synonyms": [
      "sumo deadlift high pull",
      "sdlhp"
    ],
    "categories": [
      "barbell"
    ]
  },
  "HIGH_PULL": {
    "synonyms": [
      "high pull",
      "hp",
      "snatch high pull",
      "clean high pull"
    ],
    "categories": []
  },
  "BARBELL_ROW": {
    "synonyms": [
      "barbell row",
      "bent over row",
      "bor"
    ],
    "categories": [
      "barbell",
      "cardio"
    ]
  },
  "BENCH_PRESS": {
    "synonyms": [
      "bench press",
      "bp"
    ],
    "categories": [
      "barbell"
    ]
  },
  "STRICT_CURL": {
    "synonyms": [
      "barbell curl",
      "curl"
    ],
    "categories": []
  },
  "DB_SNATCH": {
    "synonyms": [
      "dumbbell snatch",
      "db snatch",
      "single arm db snatch",
      "sa db snatch"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "DB_THRUSTER": {
    "synonyms": [
      "dumbbell thruster",
      "db thruster"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "DB_CLEAN": {
    "synonyms": [
      "dumbbell clean",
      "db clean",
      "single arm db clean",
      "sa db clean"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "DB_CLEAN_AND_JERK": {
    "synonyms": [
      "dumbbell clean and jerk",
      "db c&j",
      "db clean & jerk"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "DB_PUSH_PRESS": {
    "synonyms": [
      "dumbbell push press",
      "db push press"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "DB_PUSH_JERK": {
    "synonyms": [
      "dumbbell push jerk",
      "db push jerk"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "DB_OVERHEAD_WALKING_LUNGE": {
    "synonyms": [
      "db overhead lunge",
      "db oh lunge",
      "dumbbell overhead lunge"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "DB_FRONT_RACK_LUNGE": {
    "synonyms": [
      "db front rack lunge",
      "db fr lunge"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "DB_DEADLIFT": {
    "synonyms": [
      "dumbbell deadlift",
      "db deadlift",
      "dbl db deadlift",
      "2x db deadlift"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "DB_HANG_CLEAN": {
    "synonyms": [
      "db hang clean",
      "dumbbell hang clean"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "DEVIL_PRESS": {
    "synonyms": [
      "devil press",
      "devils press",
      "db burpee snatch",
      "db burpee ground to overhead"
    ],
    "categories": [
      "barbell"
    ]
  },
  "MAN_MAKER": {
    "synonyms": [
      "man maker",
      "man-makers"
    ],
    "categories": []
  },
  "KB_SWING": {
    "synonyms": [
      "kettlebell swing",
      "kbs",
      "kb swing",
      "american swing",
      "russian swing"
    ],
    "categories": ["kettlebell"]
  },
  "KB_SNATCH": {
    "synonyms": [
      "kettlebell snatch",
      "kb snatch"
    ],
    "categories": ["kettlebell"]
  },
  "KB_CLEAN_AND_JERK": {
    "synonyms": [
      "kettlebell clean and jerk",
      "kb c&j",
      "kb clean & jerk"
    ],
    "categories": ["kettlebell"]
  },
  "KB_CLEAN": {
    "synonyms": [
      "kettlebell clean",
      "kb clean"
    ],
    "categories": ["kettlebell"]
  },
  "KB_THRUSTER": {
    "synonyms": [
      "kettlebell thruster",
      "kb thruster"
    ],
    "categories": ["kettlebell"]
  },
  "KB_DEADLIFT": {
    "synonyms": [
      "kettlebell deadlift",
      "kb deadlift"
    ],
    "categories": ["kettlebell"]
  },
  "TURKISH_GET_UP": {
    "synonyms": [
      "turkish get up",
      "tgu"
    ],
    "categories": ["kettlebell"]
  },
  "KB_FARMERS_CARRY": {
    "synonyms": [
      "kettlebell farmers carry",
      "kb farmer carry",
      "farmers walk",
      "farmer carry",
      "fc"
    ],
    "categories": ["kettlebell"]
  },
  "KB_OVERHEAD_CARRY": {
    "synonyms": [
      "kb overhead carry",
      "overhead carry"
    ],
    "categories": ["kettlebell"]
  },
  "WALL_BALL_SHOT": {
    "synonyms": [
      "wall ball shot",
      "wall ball",
      "wb",
      "med ball shot"
    ],
    "categories": []
  },
  "MED_BALL_CLEAN": {
    "synonyms": [
      "medicine ball clean",
      "med ball clean",
      "mb clean"
    ],
    "categories": []
  },
  "SANDBAG_CARRY": {
    "synonyms": [
      "sandbag carry",
      "sb carry",
      "sandbag shoulder",
      "sandbag hold"
    ],
    "categories": []
  },
  "SANDBAG_SQUAT": {
    "synonyms": [
      "sandbag squat"
    ],
    "categories": []
  },
  "SANDBAG_OVER_SHOULDER": {
    "synonyms": [
      "sandbag over shoulder",
      "sb over shoulder"
    ],
    "categories": []
  },
  "YOKE_CARRY": {
    "synonyms": [
      "yoke carry",
      "yoke walk"
    ],
    "categories": []
  },
  "SLED_PUSH": {
    "synonyms": [
      "sled push",
      "prowler push"
    ],
    "categories": []
  },
  "SLED_DRAG": {
    "synonyms": [
      "sled drag",
      "backwards sled drag"
    ],
    "categories": []
  },
  "ATLAS_STONE": {
    "synonyms": [
      "atlas stone",
      "stone to shoulder",
      "stone lift"
    ],
    "categories": []
  },
  "D_BALL_OVER_SHOULDER": {
    "synonyms": [
      "d-ball over shoulder",
      "d ball over shoulder",
      "d-ball"
    ],
    "categories": []
  },
  "ROPE_JUMP": {
    "synonyms": [
      "jump rope",
      "jr",
      "single under",
      "su",
      "double under",
      "du",
      "triple under",
      "tu"
    ],
    "categories": [
      "plyometrics"
    ]
  },
  "TOES_TO_DUMBBELL": {
    "synonyms": [
      "toes to dumbbell",
      "t2d"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "GOBLET_SQUAT": {
    "synonyms": [
      "goblet squat",
      "gs",
      "kb goblet squat",
      "db goblet squat"
    ],
    "categories": [
      "barbell"
    ]
  },
  "GOOD_MORNING": {
    "synonyms": [
      "good morning",
      "gm"
    ],
    "categories": []
  },
  "HIP_THRUST": {
    "synonyms": [
      "hip thrust",
      "hip bridge",
      "glute bridge"
    ],
    "categories": []
  },
  "RUSSIAN_TWIST": {
    "synonyms": [
      "russian twist",
      "rt"
    ],
    "categories": []
  },
  "BACK_EXTENSION": {
    "synonyms": [
      "back extension",
      "hip & back extension",
      "hip and back extension"
    ],
    "categories": []
  },
  "REVERSE_HYPER": {
    "synonyms": [
      "reverse hyperextension",
      "reverse hyper"
    ],
    "categories": []
  },
  "AB_WHEEL": {
    "synonyms": [
      "ab wheel rollout",
      "ab rollout"
    ],
    "categories": []
  },
  "ROW_CAL": {
    "synonyms": [
      "calorie row",
      "row cal",
      "cal row",
      "row calories"
    ],
    "categories": [
      "cardio"
    ]
  },
  "BIKE_CAL": {
    "synonyms": [
      "calorie bike",
      "bike cal",
      "cal bike",
      "echo bike cals",
      "assault bike cals"
    ],
    "categories": [
      "cardio"
    ]
  },
  "SKI_CAL": {
    "synonyms": [
      "calorie ski",
      "ski cal",
      "cal ski"
    ],
    "categories": [
      "cardio"
    ]
  },
  "JUMPING_PULL_UP": {
    "synonyms": [
      "jumping pull up",
      "jpu"
    ],
    "categories": [
      "gymnastics",
      "plyometrics"
    ]
  },
  "STRICT_TOES_TO_BAR": {
    "synonyms": [
      "strict toes to bar",
      "strict ttb",
      "strict t2b"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "PUSH_PRESS_FROM_RACK": {
    "synonyms": [
      "push press from rack"
    ],
    "categories": [
      "barbell"
    ]
  },
  "OVERHEAD_CARRY": {
    "synonyms": [
      "overhead carry",
      "oh carry"
    ],
    "categories": []
  },
  "FRONT_RACK_CARRY": {
    "synonyms": [
      "front rack carry",
      "fr carry"
    ],
    "categories": []
  },
  "SOTTS_PRESS": {
    "synonyms": [
      "sotts press"
    ],
    "categories": [
      "barbell",
      "weightlifting"
    ]
  },
  "SNATCH_BALANCE": {
    "synonyms": [
      "snatch balance"
    ],
    "categories": [
      "barbell",
      "weightlifting"
    ]
  },
  "CLUSTER": {
    "synonyms": [
      "cluster",
      "squat clean thruster"
    ],
    "categories": []
  },
  "BEAR_COMPLEX": {
    "synonyms": [
      "bear complex",
      "barbell complex"
    ],
    "categories": []
  },
  "HANGING_KNEE_RAISE": {
    "synonyms": [
      "hanging knee raise",
      "hkr"
    ],
    "categories": []
  },
  "L_SIT": {
    "synonyms": [
      "l sit",
      "l-sit"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "ROPE_CLIMB_WITH_LEGS": {
    "synonyms": [
      "rope climb with legs",
      "j-hook rope climb"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "HSPU": {
    "synonyms": [
      "handstand push up",
      "handstand push-up",
      "hspu",
      "strict hspu",
      "kipping hspu"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "CHEST_TO_WALL_HSPU": {
    "synonyms": [
      "chest-to-wall handstand push-up",
      "ctw hspu"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "SLED_PULL": {
    "synonyms": [
      "sled pull",
      "sled pulling",
      "pull sled",
      "sled drag"
    ],
    "categories": []
  },
  "BURPEE_BROAD_JUMP": {
    "synonyms": [
      "burpee broad jump",
      "burpee broad jumps",
      "broad jump burpee",
      "bbj",
      "bbjo",
      "burpee broad-jump"
    ],
    "categories": [
      "cardio",
      "gymnastics",
      "plyometrics"
    ]
  },
  "SANDBAG_LUNGES": {
    "synonyms": [
      "sandbag lunges",
      "sandbag walking lunges",
      "sandbag lunge",
      "sb lunges",
      "sg lunges"
    ],
    "categories": []
  },
  "INCLINE_BENCH_PRESS": {
    "synonyms": [
      "incline bench press",
      "incline bench",
      "incline barbell bench press",
      "inc bench"
    ],
    "categories": [
      "barbell"
    ]
  },
  "DECLINE_BENCH_PRESS": {
    "synonyms": [
      "decline bench press",
      "decline bench"
    ],
    "categories": [
      "barbell"
    ]
  },
  "DUMBBELL_BENCH_PRESS": {
    "synonyms": [
      "dumbbell bench press",
      "db bench press"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "CHEST_FLY": {
    "synonyms": [
      "chest fly",
      "dumbbell fly",
      "pec fly",
      "cable fly"
    ],
    "categories": []
  },
  "DIPS": {
    "synonyms": [
      "dip",
      "dips",
      "chest dip",
      "weighted dip"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "PULLOVER": {
    "synonyms": [
      "pullover",
      "dumbbell pullover",
      "lat pullover"
    ],
    "categories": []
  },
  "LAT_PULLDOWN": {
    "synonyms": [
      "lat pulldown",
      "lah pulldown",
      "lats pulldown",
      "cpdl"
    ],
    "categories": []
  },
  "SEATED_ROW": {
    "synonyms": [
      "seated row",
      "cable row",
      "machine row",
      "row machine"
    ],
    "categories": [
      "cardio"
    ]
  },
  "ONE_ARM_DUMBELL_ROW": {
    "synonyms": [
      "one arm dumbbell row",
      "single arm db row",
      "oar",
      "sa db row"
    ],
    "categories": [
      "cardio"
    ]
  },
  "T_BAR_ROW": {
    "synonyms": [
      "t bar row",
      "t-bar row"
    ],
    "categories": [
      "cardio"
    ]
  },
  "SHRUG": {
    "synonyms": [
      "shrug",
      "trap shrug",
      "barbell shrug",
      "dumbbell shrug"
    ],
    "categories": []
  },
  "LAT_RAISE_SIDE": {
    "synonyms": [
      "side lateral raise",
      "lateral raise",
      "side raise"
    ],
    "categories": []
  },
  "LAT_RAISE_FRONT": {
    "synonyms": [
      "front raise",
      "front lateral raise"
    ],
    "categories": []
  },
  "REAR_DELT_FLY": {
    "synonyms": [
      "rear delt fly",
      "reverse fly",
      "rear fly",
      "rear delt raise"
    ],
    "categories": []
  },
  "OVERHEAD_PRESS": {
    "synonyms": [
      "overhead press",
      "military press",
      "strict overhead press",
      "ohp",
      "press",
      "shoulder press"
    ],
    "categories": [
      "barbell"
    ]
  },
  "DUMBBELL_SHOULDER_PRESS": {
    "synonyms": [
      "dumbbell shoulder press",
      "db shoulder press"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "ARNOLD_PRESS": {
    "synonyms": [
      "arnold press"
    ],
    "categories": [
      "barbell"
    ]
  },
  "UPRIGHT_ROW": {
    "synonyms": [
      "upright row",
      "upright barbell row"
    ],
    "categories": [
      "cardio"
    ]
  },
  "TRICEP_PUSHDOWN": {
    "synonyms": [
      "tricep pushdown",
      "triceps push down",
      "cable pushdown"
    ],
    "categories": []
  },
  "SKULLCRUSHER": {
    "synonyms": [
      "skull crusher",
      "lying tricep extension",
      "skullcrusher"
    ],
    "categories": []
  },
  "TRICEP_DIP": {
    "synonyms": [
      "tricep dip",
      "bench dip",
      "parallel dip"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "HAMMER_CURL": {
    "synonyms": [
      "hammer curl"
    ],
    "categories": []
  },
  "BICEP_CURL": {
    "synonyms": [
      "bicep curl",
      "barbell curl",
      "db curl",
      "curl"
    ],
    "categories": []
  },
  "PREACHER_CURL": {
    "synonyms": [
      "preacher curl",
      "preacher biceps curl"
    ],
    "categories": []
  },
  "CONCENTRATION_CURL": {
    "synonyms": [
      "concentration curl"
    ],
    "categories": []
  },
  "LEG_PRESS": {
    "synonyms": [
      "leg press"
    ],
    "categories": [
      "barbell"
    ]
  },
  "LEG_EXTENSION": {
    "synonyms": [
      "leg extension"
    ],
    "categories": []
  },
  "LEG_CURL": {
    "synonyms": [
      "leg curl",
      "lying leg curl",
      "seated leg curl"
    ],
    "categories": []
  },
  "STIFF_LEG_DEADLIFT": {
    "synonyms": [
      "stiff-leg deadlift",
      "stiff leg dl",
      "romanian deadlift (variation)"
    ],
    "categories": [
      "barbell"
    ]
  },
  "ROMANIAN_DEADLIFT": {
    "synonyms": [
      "romanian deadlift",
      "rdl"
    ],
    "categories": [
      "barbell"
    ]
  },
  "GLUTE_HAM_RAISE": {
    "synonyms": [
      "glute ham raise",
      "glute-ham raise",
      "ghr"
    ],
    "categories": []
  },
  "CALF_RAISE": {
    "synonyms": [
      "calf raise",
      "standing calf raise",
      "seated calf raise"
    ],
    "categories": []
  },
  "LEG_ADDUCTOR_MACHINE": {
    "synonyms": [
      "leg adductor",
      "inner thigh machine"
    ],
    "categories": []
  },
  "LEG_ABDUCTOR_MACHINE": {
    "synonyms": [
      "leg abductor",
      "outer thigh machine"
    ],
    "categories": []
  },
  "CHEST_PRESS_MACHINE": {
    "synonyms": [
      "chest press machine",
      "machine chest press"
    ],
    "categories": [
      "barbell"
    ]
  },
  "SHOULDER_MACHINE_PRESS": {
    "synonyms": [
      "machine shoulder press",
      "machine overhead press"
    ],
    "categories": [
      "barbell"
    ]
  },
  "PUSH_DOWN_OVERHEAD": {
    "synonyms": [
      "overhead tricep extension",
      "cable overhead extension"
    ],
    "categories": []
  },
  "ABS_CABLE_CRUNCH": {
    "synonyms": [
      "cable crunch",
      "abs cable crunch"
    ],
    "categories": [
      "cardio"
    ]
  },
  "PLANK_MACHINE": {
    "synonyms": [
      "machine plank"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "DB_FLY": {
    "synonyms": [
      "db fly",
      "dumbbell fly",
      "fly"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "DB_LATERAL_RAISE": {
    "synonyms": [
      "db lateral raise"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "DB_FRONT_RAISE": {
    "synonyms": [
      "db front raise"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "DB_SHRUG": {
    "synonyms": [
      "db shrug"
    ],
    "categories": [
      "dumbbell"
    ]
  },
  "SINGLE_LEG_PRESS": {
    "synonyms": [
      "single leg press",
      "unilateral leg press"
    ],
    "categories": [
      "barbell"
    ]
  },
  "BULGARIAN_SPLIT_SQUAT": {
    "synonyms": [
      "bulgarian split squat",
      "split squat",
      "rear foot elevated split squat",
      "rfess"
    ],
    "categories": [
      "barbell"
    ]
  },
  "LEG_PRESS_SINGLE_LEG": {
    "synonyms": [
      "single leg press"
    ],
    "categories": [
      "barbell"
    ]
  },
  "CABLE_CROSSOVER": {
    "synonyms": [
      "cable crossover",
      "cable x over",
      "xd over"
    ],
    "categories": []
  },
  "CHEST_DIP": {
    "synonyms": [
      "chest dip",
      "weighted chest dip"
    ],
    "categories": [
      "gymnastics"
    ]
  },
  "TRICEP_KICKBACK": {
    "synonyms": [
      "tricep kickback",
      "db kickback"
    ],
    "categories": []
  },
  "DUMBELL_ROW": {
    "synonyms": [
      "dumbell row",
      "db row"
    ],
    "categories": [
      "cardio"
    ]
  },
  "DUMBBELL_LATERAL_RAISE": {
    "synonyms": [
      "dumbbell lateral raise",
      "dblat raise",
      "db lat raise"
    ],
    "categories": [
      "dumbbell"
    ]
  }
}

# ---------------------------------------------------------------------------
# Labels that are NOT training data. These must return None so the caller's
# dropna(subset=['norm_label']) removes them.
# ---------------------------------------------------------------------------
IGNORE_LABELS = {
    "null",
    "setup",
    "none",
    "nan",
    "",
    "transition",
    "unknown",
}

# Labels that genuinely mean "at rest between work".
REST_LABELS = {
    "rest",
    "resting",
    "recovery",
    "break",
}

REST_CANONICAL = "REST"


# ---------------------------------------------------------------------------
# Explicit overrides. Anything whose mapping you actually care about goes here
# rather than relying on synonym tables or heuristics. Keys are normalized
# (lowercase, underscores/hyphens -> spaces, collapsed whitespace).
#
# These reproduce the merges the ORIGINAL LABEL_MAP performed, so retraining
# gives you back the class structure the old model had.
# ---------------------------------------------------------------------------
LABEL_OVERRIDES = {
    # Handstand push-up variants were folded into Push-up for training.
    # Remove this line if you want HSPU as its own class AND have enough data.
    "chest to wall hspu": "PUSH_UP",
    "chest-to-wall handstand push-up": "PUSH_UP",
    "ctw hspu": "PUSH_UP",
    "hspu": "PUSH_UP",
    "handstand push up": "PUSH_UP",

    # Loaded lunges were folded into the generic lunge class.
    "sandbag lunges": "LUNGE",
    "sandbag walking lunges": "LUNGE",
    "sandbag lunge": "LUNGE",
    "sb lunges": "LUNGE",
    "walking lunge": "LUNGE",

    # WALL_BALL and WALL_BALL_SHOT are the same movement.
    "wall ball shot": "WALL_BALL",
    "wall ball": "WALL_BALL",
    "med ball shot": "WALL_BALL",
    "medicine ball shot": "WALL_BALL",

    # Kept as a distinct class in the original label map.
    "run all out": "RUN_ALL_OUT",

    # Rope work. Keep these separate during training even if you merge them
    # for display — the cadences differ (single ~2 Hz, double ~3.5 Hz) and
    # merging them forces the model to treat two frequencies as one class.
    "single under": "SINGLE_UNDER",
    "single unders": "SINGLE_UNDER",
    "double under": "DOUBLE_UNDER",
    "double unders": "DOUBLE_UNDER",
}


# ---------------------------------------------------------------------------
# Canonical -> display name, so training labels match the strings the watch
# and the deployed model_data.json use. The on-device Viterbi matches state
# names by exact string, so this mapping is what keeps them in sync.
# ---------------------------------------------------------------------------
CANONICAL_TO_DISPLAY = {
    "AIR_SQUAT": "Air Squat",
    "PUSH_UP": "Push-up",
    "DOUBLE_UNDER": "Double-under",
    "SINGLE_UNDER": "Single-under",
    "BURPEE": "Burpee",
    "KB_SWING": "KB swing",
    "WALL_BALL": "Wall ball",
    "SIT_UP": "Sit-up",
    "LUNGE": "Walking lunge",
    "BOX_JUMP": "Box jump",
    "RUN": "Run",
    "RUN_ALL_OUT": "Run All Out",
    "REST": "REST",
}


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------
_PAREN_RE = re.compile(r"\([^)]*\)")
# Parentheses are preserved here so the paren-stripping fallback in
# canonicalize_label() can still see them ("kb swing (russian)" -> "kb swing").
_NONWORD_RE = re.compile(r"[^a-z0-9 ()]+")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """lowercase, underscores/hyphens -> spaces, collapse whitespace.

    Keeps parentheses; they are removed later only as a retry step.
    """
    s = text.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = _NONWORD_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    # Tidy spacing around parens: "kb swing ( russian )" -> "kb swing (russian)"
    s = s.replace("( ", "(").replace(" )", ")")
    return s


def _strip_parens(text: str) -> str:
    """'kb swing (russian)' -> 'kb swing'"""
    return _WS_RE.sub(" ", _PAREN_RE.sub(" ", text)).strip()


# ---------------------------------------------------------------------------
# Synonym index (built once, with collision reporting)
# ---------------------------------------------------------------------------
def build_synonym_index(exercises_data: dict, report_collisions: bool = False) -> dict:
    """Map normalized synonym -> canonical. First definition wins, matching the
    original behaviour, but collisions are surfaced instead of hidden.

    Real collisions in your data include "su" (SIT_UP vs SINGLE_UNDER) and
    "du"/"double under" (DOUBLE_UNDER vs ROPE_JUMP). Ambiguous short codes are
    the reason LABEL_OVERRIDES exists — put anything you care about there.
    """
    index: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}

    for canonical, info in exercises_data.items():
        for syn in info.get("synonyms", []):
            key = _normalize(syn)
            if not key:
                continue
            if key in index:
                if index[key] != canonical:
                    collisions.setdefault(key, [index[key]]).append(canonical)
                continue
            index[key] = canonical

    # A canonical name written out longhand should resolve to itself.
    for canonical in exercises_data:
        key = _normalize(canonical)
        index.setdefault(key, canonical)

    if report_collisions and collisions:
        lines = [f"    {k!r}: {v}" for k, v in sorted(collisions.items())]
        warnings.warn(
            "Ambiguous synonyms (first definition wins):\n" + "\n".join(lines),
            stacklevel=2,
        )

    return index


# Build at import. Pass your real EXERCISES_DATA here.
try:
    SYNONYM_TO_CANONICAL = build_synonym_index(EXERCISES_DATA)  # noqa: F821
except NameError:  # allows this file to be imported standalone for testing
    SYNONYM_TO_CANONICAL = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def canonicalize_label(
    label_str: Optional[str],
    *,
    strict: bool = True,
    display: bool = False,
) -> Optional[str]:
    """Convert a raw exercise name to its canonical form.

    Returns:
        None   for ignore labels (setup/null/etc) -> caller MUST drop these rows
        "REST" for rest labels
        canonical name (e.g. "AIR_SQUAT"), or the display name if display=True

    Args:
        strict:  raise ValueError on an unmapped label. Use True for TRAINING so
                 data problems fail loudly. Use False at inference/analysis time,
                 where an unknown label returns None rather than exploding.
        display: return "Air Squat" instead of "AIR_SQUAT", matching the strings
                 the watch and the deployed model use.

    Raises:
        ValueError: strict=True and the label is unrecognized.
    """
    if label_str is None or not isinstance(label_str, str):
        return None

    norm = _normalize(label_str)

    if norm in IGNORE_LABELS:
        return None
    if norm in REST_LABELS:
        return _fmt(REST_CANONICAL, display)

    # 1. Explicit overrides win over everything.
    if norm in LABEL_OVERRIDES:
        return _fmt(LABEL_OVERRIDES[norm], display)

    # 2. Exact synonym match.
    if norm in SYNONYM_TO_CANONICAL:
        return _fmt(SYNONYM_TO_CANONICAL[norm], display)

    # 3. Retry with parentheticals removed: "kb swing (russian)" -> "kb swing".
    stripped = _strip_parens(norm)
    if stripped != norm:
        if stripped in IGNORE_LABELS:
            return None
        if stripped in REST_LABELS:
            return _fmt(REST_CANONICAL, display)
        if stripped in LABEL_OVERRIDES:
            return _fmt(LABEL_OVERRIDES[stripped], display)
        if stripped in SYNONYM_TO_CANONICAL:
            return _fmt(SYNONYM_TO_CANONICAL[stripped], display)

    # 4. Give up. NO substring heuristic, NO uppercase passthrough.
    if strict:
        raise ValueError(
            f"Unmapped exercise label: {label_str!r} (normalized: {norm!r}). "
            f"Add it to LABEL_OVERRIDES or to the synonyms of the right entry "
            f"in EXERCISES_DATA. Refusing to invent a class for it."
        )
    return None


def _fmt(canonical: str, display: bool) -> str:
    if not display:
        return canonical
    return CANONICAL_TO_DISPLAY.get(canonical, canonical)


def to_display_label(canonical: str) -> str:
    """Canonical -> the string the watch/deployed model expects."""
    return CANONICAL_TO_DISPLAY.get(canonical, canonical)


def audit_labels(labels, strict: bool = False) -> dict:
    """Run every raw label in a dataset through the mapper and summarize.

    Call this BEFORE training and eyeball the output — it is the regression test
    for this whole file. Returns {'mapped': {...}, 'ignored': [...], 'unmapped': [...]}.
    """
    from collections import Counter

    mapped, ignored, unmapped = Counter(), Counter(), Counter()
    for raw in labels:
        try:
            result = canonicalize_label(raw, strict=strict)
        except ValueError:
            unmapped[raw] += 1
            continue
        if result is None:
            if raw is None or _normalize(str(raw)) in IGNORE_LABELS:
                ignored[raw] += 1
            else:
                unmapped[raw] += 1
        else:
            mapped[result] += 1

    return {
        "mapped": dict(mapped.most_common()),
        "ignored": dict(ignored.most_common()),
        "unmapped": dict(unmapped.most_common()),
    }