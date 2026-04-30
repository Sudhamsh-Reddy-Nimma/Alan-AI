"""
brain.py  —  Hybrid RL + Cognitive Architecture  (v2)
══════════════════════════════════════════════════════

Architecture
────────────
  CircadianRhythm : natural 5-min day/night sleep-drive cycle
  EmotionSystem   : six-channel mood (influences intent only)
  FatigueSystem   : energy + fatigue + hunger, circadian-aware
  InventorySystem : food, seeds
  FarmSystem      : delayed-reward farming with growth timers
  RunningNorm     : online reward normalisation (mean/variance)
  NoveltySensor   : visit-frequency curiosity beyond just cells
  MemorySystem    : recency-weighted reflection + trait evolution
  BrainDecision   : cognition → INTENT (softmax+temperature) → RL POLICY → ACTION
  RLPolicy        : PPO-compatible; action-masked heuristic fallback
  AlanEnv         : Gym wrapper for RL training

What changed in v2
──────────────────
  • Softmax intent selection with temperature annealing
  • CircadianRhythm drives natural sleep/wake cycles
  • RunningNorm normalises rewards for stable RL
  • NoveltySensor: visit-count heat-map, not just presence
  • MemorySystem.reflect() uses exponential recency weighting
  • Trait evolution: experience shapes personality over time
  • get_valid_actions(): action masking prevents impossible moves
  • Intent inertia / commitment – reduces flickering
  • Opportunity-cost tracking (time since each intent was used)
  • BrainMetrics: tracks per-intent success rates
  • Fixed: gather probability scales with proximity, not flat 0.8 %
  • Fixed: RLPolicy heuristic respects action mask
  • Cleaner type annotations throughout
"""

from __future__ import annotations
import math, random, time, copy, os, threading
try:
    import google.generativeai as genai
except ImportError:
    genai = None
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────
# INTENTS
# ─────────────────────────────────────────────────────────────────
INTENTS = [
    "explore", "rest", "eat", "gather_food",
    "farm", "store_food", "use_updraft", "climb", "play",
]
NUM_INTENTS   = len(INTENTS)
INTENT_TO_IDX = {intent: idx for idx, intent in enumerate(INTENTS)}

# ─────────────────────────────────────────────────────────────────
# CIRCADIAN RHYTHM  — natural sleep / wake drive
# ─────────────────────────────────────────────────────────────────
class CircadianRhythm:
    """
    5-minute full day cycle.
    sleep_drive peaks during 'night' phase and decays when sleeping.
    """
    PERIOD: float = 300.0   # seconds per full cycle

    def __init__(self, phase_offset: float = 0.0):
        self._t: float = phase_offset * self.PERIOD

    def update(self, dt: float, is_sleeping: bool = False):
        self._t += dt
        if is_sleeping:
            # Sleeping advances the phase toward "morning"
            self._t += dt * 0.5

    @property
    def phase(self) -> float:
        """0.0 → 1.0 within one cycle."""
        return (self._t % self.PERIOD) / self.PERIOD

    @property
    def is_night(self) -> bool:
        return self.phase > 0.55

    @property
    def sleep_drive(self) -> float:
        """
        0.0 (fully awake drive) → 1.0 (strong sleep urge).
        Builds sinusoidally during night phase.
        """
        p = self.phase
        if p < 0.55:
            return 0.0
        night_progress = (p - 0.55) / 0.45
        return math.sin(math.pi * night_progress) ** 1.5

    @property
    def light_level(self) -> float:
        """0.0 = midnight dark, 1.0 = midday bright."""
        return 0.5 + 0.5 * math.cos(2.0 * math.pi * (self.phase - 0.25))

    def __repr__(self) -> str:
        phase_name = "night" if self.is_night else "day"
        return f"Circadian(phase={self.phase:.2f}, {phase_name}, sleep_drive={self.sleep_drive:.2f})"


# ─────────────────────────────────────────────────────────────────
# RUNNING NORMALISER — stable reward / observation scaling
# ─────────────────────────────────────────────────────────────────
class RunningNorm:
    """Welford online mean/variance for normalising a stream of scalars."""

    def __init__(self, alpha: float = 0.005, clip: float = 5.0):
        self.mean:  float = 0.0
        self.var:   float = 1.0
        self.alpha: float = alpha
        self.clip:  float = clip
        self._n:    int   = 0

    def update(self, x: float) -> float:
        self._n += 1
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * x
        self.var  = (1.0 - self.alpha) * self.var  + self.alpha * (x - self.mean) ** 2
        return self.normalise(x)

    def normalise(self, x: float) -> float:
        z = (x - self.mean) / (math.sqrt(max(self.var, 1e-6)) + 1e-8)
        return max(-self.clip, min(self.clip, z))


# ─────────────────────────────────────────────────────────────────
# NOVELTY SENSOR — heat-map curiosity beyond visited-cell presence
# ─────────────────────────────────────────────────────────────────
class NoveltySensor:
    """
    Maps world cells to visit counts.  Novelty = 1 / (count + 1).
    Provides a continuous curiosity signal.
    """
    CELL_SIZE: float = 1.0

    def __init__(self):
        self._counts: Dict[Tuple[int, int], int] = {}

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x / self.CELL_SIZE), int(y / self.CELL_SIZE))

    def visit(self, x: float, y: float) -> float:
        """Record visit and return novelty score (0 → 1)."""
        c = self._cell(x, y)
        self._counts[c] = self._counts.get(c, 0) + 1
        return 1.0 / self._counts[c]

    def novelty_at(self, x: float, y: float) -> float:
        c = self._cell(x, y)
        return 1.0 / (self._counts.get(c, 0) + 1)

    @property
    def unique_cells(self) -> int:
        return len(self._counts)

    def most_novel_direction(self, x: float, y: float) -> int:
        """Return ACTION_LEFT or ACTION_RIGHT toward less-visited side."""
        left_nov  = self.novelty_at(x - 2.0, y)
        right_nov = self.novelty_at(x + 2.0, y)
        return 0 if left_nov > right_nov else 1   # ACTION_LEFT=0, ACTION_RIGHT=1


# ─────────────────────────────────────────────────────────────────
# EMOTION SYSTEM
# ─────────────────────────────────────────────────────────────────
EMOTION_KEYS = ("anger", "disgust", "fear", "happiness", "sadness", "surprise")

class EmotionSystem:
    DECAY      = 0.985
    DECAY_FAST = 0.92

    def __init__(self):
        self.state: Dict[str, float] = {
            "anger":     0.00,
            "disgust":   0.00,
            "fear":      0.15,
            "happiness": 0.60,
            "sadness":   0.00,
            "surprise":  0.10,
        }

    def clamp(self):
        for k in EMOTION_KEYS:
            self.state[k] = max(0.0, min(1.0, self.state[k]))

    def decay(self):
        for k in EMOTION_KEYS:
            f = self.DECAY_FAST if k == "surprise" else self.DECAY
            self.state[k] *= f
        self.clamp()

    def dominant(self) -> Tuple[str, float]:
        return max(self.state.items(), key=lambda kv: kv[1])

    def valence(self) -> float:
        """Positive = good, negative = bad. [-1, 1]"""
        pos = self.state["happiness"] + self.state["surprise"] * 0.3
        neg = self.state["anger"] + self.state["fear"] + self.state["sadness"] + self.state["disgust"]
        return max(-1.0, min(1.0, (pos - neg) / (pos + neg + 1e-6)))

    def __repr__(self):
        parts = [f"{k[0].upper()}={v:.2f}" for k, v in self.state.items()]
        return "Emotions(" + " ".join(parts) + ")"


# ─────────────────────────────────────────────────────────────────
# FATIGUE SYSTEM  — circadian-aware
# ─────────────────────────────────────────────────────────────────
class FatigueSystem:
    """
    energy  : short-term usable resource, depleted by movement
    fatigue : long-term accumulation, reduced only by sleep
    hunger  : resource deficit, increases over time
    """

    def __init__(self):
        self.energy:  float = 1.0
        self.fatigue: float = 0.0
        self.hunger:  float = 0.0

    def update(self, env, dt: float,
               circadian: Optional[CircadianRhythm] = None):
        vx = abs(env.agent_body.velocity.x)
        vy = abs(env.agent_body.velocity.y)
        sprint_mult = 1.3 if getattr(env, 'is_sprinting', False) else 1.0
        eff_penalty = 1.0 + self.fatigue * 0.5

        # --- Increased decay for visibility ---
        move_cost = 0.015 * (vx + vy) * sprint_mult * eff_penalty
        self.energy -= (0.001 + move_cost) * dt

        if getattr(env, 'in_water',   False): self.energy -= 0.030 * dt
        if getattr(env, 'is_rolling', False): self.energy -= 0.040 * dt

        if getattr(env, 'is_sleeping', False):
            self.energy  = min(1.0, self.energy  + 0.35  * dt)
            self.fatigue = max(0.0, self.fatigue - 0.080 * dt)
        elif getattr(env, 'is_sitting', False):
            self.energy  = min(1.0, self.energy  + 0.12  * dt)
            self.fatigue = max(0.0, self.fatigue - 0.015 * dt)
        elif getattr(env, 'is_crouching', False) and not getattr(env, 'is_climbing', False):
            self.energy  = min(1.0, self.energy  + 0.04  * dt)

        # Fatigue builds faster when energy is low, but also builds passively
        passive_fatigue = 0.001
        if self.energy < 0.25:
            passive_fatigue = 0.008
        self.fatigue = min(1.0, self.fatigue + passive_fatigue * dt)

        # Starvation accelerates fatigue
        if self.hunger > 0.85:
            self.fatigue = min(1.0, self.fatigue + 0.010 * dt)
            self.energy  = max(0.0, self.energy  - 0.005 * dt)

        # Circadian sleep pressure builds fatigue faster at night
        if circadian is not None and circadian.is_night and not getattr(env, 'is_sleeping', False):
            self.fatigue = min(1.0, self.fatigue + 0.005 * circadian.sleep_drive * dt)

        self.energy = max(0.0, min(1.0, self.energy))
        self.hunger = min(1.0, self.hunger + 0.005 * dt)

    @property
    def is_exhausted(self) -> bool: return self.energy < 0.05
    @property
    def is_tired(self)     -> bool: return self.fatigue > 0.90
    @property
    def is_rested(self)    -> bool: return self.energy > 0.95 and self.fatigue < 0.10
    @property
    def is_hungry(self)    -> bool: return self.hunger > 0.75

    @property
    def movement_efficiency(self) -> float:
        return max(0.1, 1.0 - self.fatigue * 0.5)

    @property
    def decision_frequency(self) -> float:
        return max(0.2, 1.0 - self.fatigue * 0.6)

    def eat(self, nutrition: float = 0.5):
        self.hunger = max(0.0, self.hunger - nutrition)
        self.energy = min(1.0, self.energy + 0.25)

    def welfare_score(self) -> float:
        """0 (terrible) → 1 (thriving)."""
        return (self.energy * 0.4 +
                (1.0 - self.fatigue) * 0.35 +
                (1.0 - self.hunger)  * 0.25)


# Backward-compat alias
EnergySystem = FatigueSystem


# ─────────────────────────────────────────────────────────────────
# INVENTORY SYSTEM
# ─────────────────────────────────────────────────────────────────
class InventorySystem:
    MAX_FOOD  = 20
    MAX_SEEDS = 20

    def __init__(self, food: int = 3, seeds: int = 5):
        self.food:  int = food
        self.seeds: int = seeds

    def eat(self) -> bool:
        if self.food > 0:
            self.food -= 1
            return True
        return False

    def gather(self, amount: int = 1):
        self.food = min(self.MAX_FOOD, self.food + amount)

    def plant(self) -> bool:
        if self.seeds > 0:
            self.seeds -= 1
            return True
        return False

    def harvest(self, amount: int = 2):
        self.food = min(self.MAX_FOOD, self.food + amount)

    def store(self, amount: int = 1):
        pass  # multi-agent future: move food to shared depot

    def abundance(self) -> float:
        """How well-stocked is the agent? 0 → 1."""
        return (self.food / self.MAX_FOOD * 0.7 +
                self.seeds / self.MAX_SEEDS * 0.3)

    def to_vector(self) -> List[float]:
        return [
            min(1.0, self.food  / self.MAX_FOOD),
            min(1.0, self.seeds / self.MAX_SEEDS),
        ]

    def __repr__(self):
        return f"Inventory(food={self.food}, seeds={self.seeds})"


# ─────────────────────────────────────────────────────────────────
# FARM SYSTEM
# ─────────────────────────────────────────────────────────────────
@dataclass
class FarmPlot:
    x:            float
    y:            float = 0.0
    state:        str   = "empty"
    growth_timer: float = 0.0
    GROW_TIME:    float = 30.0
    FAIL_RATE:    float = 0.10

    @property
    def growth_progress(self) -> float:
        if self.state != "growing": return 0.0
        return min(1.0, self.growth_timer / self.GROW_TIME)


class FarmSystem:
    INTERACT_RADIUS = 1.2

    def __init__(self, plot_positions: Optional[List[float]] = None):
        positions = plot_positions or [-2.5, -3.5, -4.5]
        self.plots: List[FarmPlot] = [FarmPlot(x=px) for px in positions]

    def update(self, dt: float):
        for plot in self.plots:
            if plot.state == "growing":
                plot.growth_timer += dt
                if plot.growth_timer >= plot.GROW_TIME:
                    if random.random() < plot.FAIL_RATE:
                        plot.state = "empty"
                        plot.growth_timer = 0.0
                    else:
                        plot.state = "ready"

    def try_interact(self, agent_x: float,
                     inventory: InventorySystem) -> Optional[str]:
        for plot in self.plots:
            if abs(plot.x - agent_x) < self.INTERACT_RADIUS:
                if plot.state == "ready":
                    inventory.harvest(2)
                    plot.state = "empty"
                    plot.growth_timer = 0.0
                    return "harvested"
                elif plot.state == "empty" and inventory.seeds > 0:
                    inventory.seeds  -= 1
                    plot.state        = "growing"
                    plot.growth_timer = 0.0
                    return "planted"
        return None

    def ready_count(self)   -> int: return sum(1 for p in self.plots if p.state == "ready")
    def growing_count(self) -> int: return sum(1 for p in self.plots if p.state == "growing")


# ─────────────────────────────────────────────────────────────────
# EMOTION UPDATE
# ─────────────────────────────────────────────────────────────────
def update_emotions(env, emotions: Dict[str, float],
                    fatigue: FatigueSystem, reward: float,
                    circadian: Optional[CircadianRhythm] = None):
    if getattr(env, 'in_water', False): emotions["happiness"] += 0.012
    if fatigue.is_rested:               emotions["happiness"] += 0.004
    if getattr(env, 'in_updraft', False) and env.agent_body.velocity.y > 1.0:
        emotions["happiness"] += 0.04
        emotions["surprise"]  += 0.02

    h = env.agent_body.position.y
    if h > 6.0: emotions["fear"] += 0.005 * min(1.0, (h - 6.0) / 9.0)
    if fatigue.is_exhausted: emotions["fear"] += 0.005
    fall = env.agent_body.velocity.y
    if fall < -5.0: emotions["fear"] += 0.02 * min(1.0, abs(fall) / 10.0)

    spd = math.hypot(env.agent_body.velocity.x, env.agent_body.velocity.y)
    if spd > 6.0: emotions["surprise"] += 0.03

    if reward < -0.1: emotions["anger"] += 0.02
    if getattr(env, 'in_water', False) and fatigue.is_exhausted:
        emotions["anger"] += 0.015

    if fatigue.is_exhausted:  emotions["sadness"] += 0.04
    if fatigue.hunger > 0.7:  emotions["sadness"] += 0.02
    if fatigue.is_tired:      emotions["sadness"] += 0.02

    if getattr(env, 'in_water', False) and fatigue.energy < 0.3:
        emotions["disgust"] += 0.012
    if fatigue.hunger > 0.8: emotions["disgust"] += 0.01

    # Circadian: night makes the agent a little sad / fearful
    if circadian is not None and circadian.is_night:
        emotions["sadness"] += 0.001 * circadian.sleep_drive

    for k in emotions: emotions[k] *= 0.985


# ─────────────────────────────────────────────────────────────────
# SURVIVAL REWARD
# ─────────────────────────────────────────────────────────────────
def compute_reward(env, emotions: Dict[str, float],
                   fatigue: FatigueSystem,
                   inventory: Optional[InventorySystem] = None,
                   brain:     Optional["BrainDecision"]  = None,
                   farm_system: Optional[FarmSystem]     = None,
                   reward_norm: Optional[RunningNorm]    = None) -> float:
    r = 0.0

    r += 0.05                        # alive bonus
    r += fatigue.energy  * 0.20     # maintain energy
    r -= fatigue.hunger  * 0.25     # hunger penalty
    r -= fatigue.fatigue * 0.20     # fatigue penalty

    if fatigue.hunger   > 0.90: r -= 2.0
    if fatigue.is_exhausted:    r -= 1.0
    if fatigue.fatigue  > 0.80: r -= 1.5

    if inventory is not None:
        r += inventory.food  * 0.04
        r += inventory.seeds * 0.01

    if farm_system is not None:
        r += farm_system.ready_count()   * 0.50
        r += farm_system.growing_count() * 0.05

    if brain is not None:
        r += brain.novelty_sensor.unique_cells * 0.003   # novelty scales with map coverage
        if brain.stuck_counter > 3:      r -= 2.0
        if brain.no_progress_steps > 300: r -= 1.0
        
        # --- INTENT FULFILLMENT REWARDS ---
        intent = brain.current_intent
        ax, ay = env.agent_body.position
        vx, vy = env.agent_body.velocity

        if intent == "explore":
            r += abs(vx) * 0.1  # reward moving
        elif intent == "rest":
            if getattr(env, 'is_sleeping', False) or getattr(env, 'is_sitting', False):
                r += 0.5  # reward being in rest state
        elif intent == "eat":
            if fatigue.hunger < 0.3: r += 0.3
        elif intent == "gather_food":
            dist_to_tree = abs(ax - (-5.0))
            if dist_to_tree < 2.0: r += 0.2  # reward proximity to tree
        elif intent == "farm":
            dist_to_farm = abs(ax - (-3.5))
            if dist_to_farm < 2.0: r += 0.2  # reward proximity to farm
        elif intent == "use_updraft":
            if getattr(env, 'in_updraft', False): r += 0.5
        elif intent == "climb":
            if getattr(env, 'is_climbing', False): r += 0.5
        elif intent == "play":
            if abs(vx) + abs(vy) > 2.0: r += 0.2 # activity reward

    r += env.agent_body.position.y * 0.10
    if getattr(env, 'in_updraft', False) and env.agent_body.velocity.y > 0:
        r += 1.0

    # Emotion valence as soft shaping signal
    if brain is not None and hasattr(brain, '_emotion_sys'):
        r += brain._emotion_sys.valence() * 0.05

    raw = r
    if reward_norm is not None:
        return reward_norm.update(raw)
    return raw


# ─────────────────────────────────────────────────────────────────
# STATE OBSERVATION
# ─────────────────────────────────────────────────────────────────
def get_full_state(env, emotion_sys: EmotionSystem,
                   fatigue: FatigueSystem,
                   inventory: Optional[InventorySystem] = None,
                   circadian: Optional[CircadianRhythm] = None) -> Dict[str, Any]:
    ax, ay       = env.agent_body.position
    vx, vy       = env.agent_body.velocity
    bx, by       = env.box_body.position
    ballx, bally = env.ball_body.position
    ud0, ud1     = getattr(env, "updraft_zone", (0.5, 2.5))
    in_ud        = ud0 <= ax <= ud1 and ay < 6.0
    if hasattr(env, 'in_updraft'): env.in_updraft = in_ud

    state: Dict[str, Any] = {
        "position":         (ax, ay),
        "velocity":         (vx, vy),
        "in_water":         getattr(env, 'in_water', False),
        "is_grounded":      env.is_grounded(),
        "is_climbing":      getattr(env, 'is_climbing', False),
        "is_crouching":     getattr(env, 'is_crouching', False),
        "is_sleeping":      getattr(env, 'is_sleeping', False),
        "is_sitting":       getattr(env, 'is_sitting', False),
        "energy":           fatigue.energy,
        "fatigue":          fatigue.fatigue,
        "hunger":           fatigue.hunger,
        "emotions":         dict(emotion_sys.state),
        "dominant_emotion": emotion_sys.dominant()[0],
        "box_pos":          (bx, by),
        "ball_pos":         (ballx, bally),
        "facing":           getattr(env, 'facing', 1),
        "in_updraft":       in_ud,
        "updraft_zone":     (ud0, ud1),
    }
    if inventory:
        state["inventory"] = {"food": inventory.food, "seeds": inventory.seeds}
    if circadian:
        state["circadian_phase"]      = circadian.phase
        state["circadian_light"]      = circadian.light_level
        state["circadian_sleep_drive"] = circadian.sleep_drive
    return state


def state_to_vector(state: Dict[str, Any]) -> List[float]:
    """18-dim base vector (unchanged for backward-compat)."""
    ax, ay       = state["position"]
    vx, vy       = state["velocity"]
    em           = state["emotions"]
    bx, by       = state["box_pos"]
    ballx, bally = state["ball_pos"]
    nx = lambda v: max(-1., min(1., v / 12.))
    ny = lambda v: max(-1., min(1., v / 15.))
    nv = lambda v: max(-1., min(1., v / 10.))
    return [
        nx(ax), ny(ay), nv(vx), nv(vy),
        nx(bx), ny(by), nx(ballx), ny(bally),
        float(state["in_water"]), float(state["is_grounded"]),
        state["energy"], state["hunger"],
        em["anger"], em["fear"], em["happiness"], em["sadness"], em["surprise"],
        float(state["in_updraft"]),
    ]   # shape: (18,)


OBS_DIM = 18 + NUM_INTENTS + 2   # 18 + 9 + 2 = 29


# ─────────────────────────────────────────────────────────────────
# SOFTMAX HELPER
# ─────────────────────────────────────────────────────────────────
def softmax_sample(weights: Dict[str, float], temperature: float = 1.0) -> str:
    """
    Sample from a weighted dict using softmax with temperature.
    temperature → 0 : greedy  |  temperature → ∞ : uniform
    """
    keys = list(weights.keys())
    vals = [max(1e-6, weights[k]) for k in keys]
    max_v = max(vals)
    exp_v = [math.exp((v - max_v) / max(1e-3, temperature)) for v in vals]
    total = sum(exp_v)
    probs = [v / total for v in exp_v]
    return random.choices(keys, weights=probs)[0]


# ─────────────────────────────────────────────────────────────────
# ACTION MASKING
# ─────────────────────────────────────────────────────────────────
ACTION_LEFT = 0; ACTION_RIGHT = 1; ACTION_JUMP = 2; ACTION_CROUCH = 3
ACTION_SIT  = 4; ACTION_SLEEP = 5; ACTION_GRAB  = 6; ACTION_THROW  = 7
ACTION_IDLE = 8; ACTION_CLIMB_UP = 9; ACTION_CLIMB_DOWN = 10

ACTION_NAMES = {
    0: "walk_left", 1: "walk_right", 2: "jump",    3: "crouch",     4: "sit",
    5: "sleep",     6: "grab",       7: "throw",   8: "idle",
    9: "climb_up",  10: "climb_down",
}


def get_valid_actions(env, fatigue: FatigueSystem) -> List[int]:
    """Return only actions that make physical sense right now."""
    valid = {ACTION_IDLE}

    if getattr(env, 'is_sleeping', False):
        return [ACTION_IDLE, ACTION_SLEEP]   # can only stop sleeping

    if getattr(env, 'is_climbing', False):
        valid.update([ACTION_CLIMB_UP, ACTION_CLIMB_DOWN, ACTION_LEFT, ACTION_RIGHT])
        return list(valid)

    # Grounded actions
    if env.is_grounded():
        valid.update([ACTION_LEFT, ACTION_RIGHT, ACTION_CROUCH,
                      ACTION_SIT, ACTION_SLEEP])
        if not fatigue.is_exhausted:
            valid.add(ACTION_JUMP)
    else:
        # Airborne: can only steer
        valid.update([ACTION_LEFT, ACTION_RIGHT])

    # Grab / throw always possible (may be no-op internally)
    valid.update([ACTION_GRAB, ACTION_THROW])

    # Ladder access
    try:
        if env.near_ladder():
            valid.update([ACTION_CLIMB_UP, ACTION_CLIMB_DOWN])
    except Exception:
        pass

    return list(valid)


# ─────────────────────────────────────────────────────────────────
# MEMORY SYSTEM
# ─────────────────────────────────────────────────────────────────
class MemorySystem:
    SHORT_TERM_LEN             = 300
    LONG_TERM_REWARD_THRESHOLD = 1.5
    RECENCY_GAMMA              = 0.97   # exponential recency weighting

    def __init__(self):
        self.short_term: deque        = deque(maxlen=self.SHORT_TERM_LEN)
        self.long_term:  List[Dict]   = []
        self.insights:   Dict[int, float] = {}

        self.intent_bias:  Dict[str, float] = {i: 0.0  for i in INTENTS}
        self.intent_score: Dict[str, float] = {i: 0.0  for i in INTENTS}  # EMA win rate

        self.identity = {
            "beliefs": {},
            "traits": {
                "curiosity":      0.50,
                "risk_tolerance": 0.50,
                "patience":       0.50,
                "confidence":     0.50,
                "sociability":    0.50,
            }
        }

    def record(self, state: Dict, reward: float,
               action: int = -1, intent: str = "explore"):
        e = {"state": state, "action": action, "intent": intent,
             "reward": reward, "t": time.time()}
        self.short_term.append(e)
        if reward >= self.LONG_TERM_REWARD_THRESHOLD:
            self.long_term.append(e)
            if len(self.long_term) > 500:
                self.long_term.pop(0)

    def failed_recently(self, action_id: int,
                        window: int = 60, threshold: int = 5) -> bool:
        recent = list(self.short_term)[-window:]
        fails  = [e for e in recent
                  if e["action"] == action_id and e["reward"] < -0.05]
        return len(fails) >= threshold

    def reflect(self) -> Dict[int, float]:
        """
        Recency-weighted reflection.
        Recent experiences count exponentially more than old ones.
        Also evolves personality traits based on accumulated outcomes.
        """
        recent = list(self.short_term)[-120:]
        if not recent:
            return self.insights

        n = len(recent)
        weights = [self.RECENCY_GAMMA ** (n - 1 - i) for i in range(n)]

        patterns:      Dict[int,  List[Tuple[float, float]]] = {}
        intent_data:   Dict[str,  List[Tuple[float, float]]] = {}

        for i, exp in enumerate(recent):
            w = weights[i]
            a = exp["action"]
            patterns.setdefault(a, []).append((exp["reward"], w))
            iv = exp.get("intent", "explore")
            intent_data.setdefault(iv, []).append((exp["reward"], w))

        # Weighted per-action insights
        for action, rw_pairs in patterns.items():
            total_w = sum(w for _, w in rw_pairs)
            self.insights[action] = sum(r * w for r, w in rw_pairs) / total_w

        # Weighted intent bias + score
        for intent, rw_pairs in intent_data.items():
            total_w = sum(w for _, w in rw_pairs)
            avg = sum(r * w for r, w in rw_pairs) / total_w
            self.intent_bias[intent]  = self.intent_bias.get(intent,  0.0) * 0.9 + avg * 0.1
            self.intent_score[intent] = self.intent_score.get(intent, 0.0) * 0.95 + avg * 0.05

        # ── Trait evolution (subtle, long-term) ─────────────────
        all_w = sum(weights)
        overall_avg = sum(exp["reward"] * w for exp, w in zip(recent, weights)) / all_w

        traits = self.identity["traits"]
        if overall_avg > 0.5:
            traits["confidence"] = min(1.0, traits["confidence"] + 0.002)
        elif overall_avg < -0.5:
            traits["confidence"] = max(0.0, traits["confidence"] - 0.001)

        # Frequent exploration boosts curiosity
        explore_frac = sum(1 for e in recent if e.get("intent") == "explore") / max(1, n)
        traits["curiosity"] = traits["curiosity"] * 0.999 + explore_frac * 0.001

        return self.insights

    def summarize_recent(self) -> str:
        if not self.short_term: return "No recent memory."
        vals = [e["reward"] for e in list(self.short_term)[-60:]]
        avg  = sum(vals) / len(vals)
        trend = "↑" if vals[-1] > avg else "↓"
        return f"Avg reward {avg:.2f} {trend}  LTM:{len(self.long_term)}"


# ─────────────────────────────────────────────────────────────────
# BRAIN METRICS  — per-intent success tracking
# ─────────────────────────────────────────────────────────────────
class BrainMetrics:
    def __init__(self):
        self.intent_uses:     Dict[str, int]   = {i: 0   for i in INTENTS}
        self.intent_rewards:  Dict[str, float] = {i: 0.0 for i in INTENTS}
        self.total_steps:     int   = 0
        self.total_reward:    float = 0.0
        self.episode_rewards: deque = deque(maxlen=200)

    def record(self, intent: str, reward: float):
        self.intent_uses[intent]    = self.intent_uses.get(intent, 0) + 1
        prev = self.intent_rewards.get(intent, 0.0)
        n    = self.intent_uses[intent]
        self.intent_rewards[intent] = prev + (reward - prev) / n   # running mean
        self.total_steps += 1
        self.total_reward += reward
        self.episode_rewards.append(reward)

    def recent_avg(self, window: int = 60) -> float:
        vals = list(self.episode_rewards)[-window:]
        return sum(vals) / max(1, len(vals))

    def best_intent(self) -> str:
        return max(self.intent_rewards, key=lambda k: self.intent_rewards[k])


# ─────────────────────────────────────────────────────────────────
# COGNITIVE PIPELINE
# ─────────────────────────────────────────────────────────────────
def perceive(env, emotion_sys: EmotionSystem,
             fatigue: FatigueSystem, memory: MemorySystem,
             inventory: Optional[InventorySystem] = None,
             farm_system: Optional[FarmSystem]    = None,
             circadian:   Optional[CircadianRhythm] = None) -> Dict:
    ax, ay = env.agent_body.position
    return {
        "self": {
            "x": ax, "y": ay,
            "velocity":  tuple(env.agent_body.velocity),
            "grounded":  env.is_grounded(),
            "energy":    fatigue.energy,
            "fatigue":   fatigue.fatigue,
            "hunger":    fatigue.hunger,
        },
        "world": {
            "wind_zone": getattr(env, "updraft_zone", (0.5, 2.5)),
        },
        "emotion":         emotion_sys.state,
        "memory_summary":  memory.summarize_recent(),
        "inventory":       {"food":  inventory.food  if inventory else 0,
                            "seeds": inventory.seeds if inventory else 0},
        "farm_ready":      farm_system.ready_count() if farm_system else 0,
        "circadian_night": circadian.is_night if circadian else False,
        "sleep_drive":     circadian.sleep_drive if circadian else 0.0,
    }


def interpret(perception: Dict) -> Dict:
    meaning: Dict[str, str] = {}
    if perception["self"]["energy"] < 0.20:
        meaning["priority"] = "rest"
    if perception["self"]["fatigue"] > 0.70:
        meaning["priority"] = "sleep"
    if perception["self"]["hunger"] > 0.65:
        meaning["priority"] = "eat"
    if perception["emotion"]["fear"] > 0.60:
        meaning["state"] = "unsafe"
    if perception["inventory"]["food"] < 2:
        meaning["resource"] = "low_food"
    if perception["farm_ready"] > 0:
        meaning["opportunity"] = "harvest_ready"
    if perception.get("circadian_night") and perception.get("sleep_drive", 0) > 0.5:
        meaning["circadian"] = "night_fatigue"
    return meaning


def think(perception: Dict, meaning: Dict,
          emotion: Dict[str, float], memory: MemorySystem) -> List[str]:
    thoughts: List[str] = []

    if meaning.get("problem") == "need_height":
        thoughts.append("I need to get higher.")
    if meaning.get("priority") == "sleep":
        thoughts.append("I'm exhausted deep down. I need to sleep.")
    if meaning.get("priority") == "eat":
        thoughts.append("I'm hungry. I need to eat.")
    if meaning.get("priority") in ("eat", "rest") and perception["inventory"]["food"] > 0:
        thoughts.append("I have food. I should eat.")
    if meaning.get("resource") == "low_food":
        thoughts.append("Running low on food. I should gather more.")
    if meaning.get("opportunity") == "harvest_ready":
        thoughts.append("My crops are ready to harvest!")
    if meaning.get("circadian") == "night_fatigue":
        thoughts.append("It's getting dark... I should rest soon.")

    frustration = (emotion["anger"] + emotion["sadness"]) / 2.0
    if frustration > 0.4:
        thoughts.append("This isn't working. Try something else.")

    if memory.failed_recently(ACTION_JUMP):
        thoughts.append("Jumping failed before. What else?")

    if perception["world"]["wind_zone"] and perception["self"]["y"] < 4.0:
        thoughts.append("Maybe the wind can lift me.")

    if not thoughts:
        thoughts.append("What should I explore next?")

    return thoughts


def meta_think(thoughts: List[str], memory: MemorySystem) -> List[str]:
    meta: List[str] = []
    t_str = " ".join(thoughts).lower()
    if "failed" in t_str or "isn't working" in t_str:
        meta.append("Why do I keep failing?")
    if "explore" in t_str or "wander" in t_str:
        meta.append("Do I actually want this or am I just wandering?")
    if "harvest" in t_str:
        meta.append("Patience pays off.")
    if "dark" in t_str:
        meta.append("Nights feel long when you're tired.")
    return meta


# ─────────────────────────────────────────────────────────────────
# RL POLICY
# ─────────────────────────────────────────────────────────────────
class RLPolicy:
    """
    Wraps a Stable-Baselines3 PPO model.
    Falls back to an intent-aware, action-masked heuristic.
    """

    _INTENT_PREFS: Dict[str, List[Tuple[int, float]]] = {
        "explore":    [(ACTION_LEFT, 0.45), (ACTION_RIGHT, 0.45), (ACTION_JUMP, 0.10)],
        "rest":       [(ACTION_SLEEP, 0.50), (ACTION_SIT, 0.40), (ACTION_IDLE, 0.10)],
        "eat":        [(ACTION_IDLE, 1.00)],
        "gather_food":[(ACTION_LEFT, 0.40), (ACTION_RIGHT, 0.40),
                       (ACTION_GRAB, 0.15), (ACTION_JUMP, 0.05)],
        "farm":       [(ACTION_IDLE, 0.50), (ACTION_LEFT, 0.25), (ACTION_RIGHT, 0.25)],
        "store_food": [(ACTION_IDLE, 0.60), (ACTION_LEFT, 0.20), (ACTION_RIGHT, 0.20)],
        "use_updraft":[(ACTION_RIGHT, 0.60), (ACTION_IDLE, 0.30), (ACTION_JUMP, 0.10)],
        "climb":      [(ACTION_CLIMB_UP, 0.50), (ACTION_JUMP, 0.30), (ACTION_RIGHT, 0.20)],
        "play":       [(ACTION_JUMP, 0.30), (ACTION_LEFT, 0.25),
                       (ACTION_RIGHT, 0.25), (ACTION_THROW, 0.20)],
    }

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path:
            try:
                from stable_baselines3 import PPO
                self.model = PPO.load(model_path)
                print(f"[RLPolicy] Loaded '{model_path}'")
            except Exception as exc:
                print(f"[RLPolicy] Could not load model ({exc}). Using heuristic.")

    def predict(self, obs: List[float], intent: str,
                valid_actions: Optional[List[int]] = None) -> int:
        if self.model is not None:
            import numpy as np
            action, _ = self.model.predict(
                np.array(obs, dtype="float32"), deterministic=False)
            a = int(action)
            # If model suggests invalid action, fall back to heuristic
            if valid_actions and a not in valid_actions:
                a = self._heuristic(intent, valid_actions)
            return a
        return self._heuristic(intent, valid_actions)

    def _heuristic(self, intent: str,
                   valid_actions: Optional[List[int]] = None) -> int:
        prefs = self._INTENT_PREFS.get(intent, [(ACTION_IDLE, 1.0)])
        if valid_actions:
            prefs = [(a, w) for a, w in prefs if a in valid_actions]
            if not prefs:
                return random.choice(valid_actions) if valid_actions else ACTION_IDLE
        actions, weights = zip(*prefs)
        return random.choices(actions, weights=weights)[0]


# ─────────────────────────────────────────────────────────────────
# BRAIN DECISION
# ─────────────────────────────────────────────────────────────────
class BrainDecision:
    """
    Cognitive loop:
      Perception → Thought → Intent (softmax+temp) → RL Policy → Action
    """

    # Temperature schedule: starts warm (exploration), cools over time
    TEMP_INIT  = 2.0
    TEMP_MIN   = 0.4
    TEMP_DECAY = 0.9999

    def __init__(self, autonomous: bool = False,
                 rl_model_path: Optional[str] = None):
        self.autonomous      = autonomous
        self.current_intent  = "explore"
        self.current_thought = "..."
        self.life_goal       = "understand_world"

        self.rl_policy     = RLPolicy(model_path=rl_model_path)
        self.novelty_sensor = NoveltySensor()
        self.metrics        = BrainMetrics()

        self.intent_timer:    float = 0.0
        self.intent_duration: float = 5.0

        # Intent opportunity cost: time since each intent was last used
        self.intent_last_used: Dict[str, float] = {i: -10.0 for i in INTENTS}

        # Intent inertia: boost current intent slightly to reduce flickering
        self.INERTIA_BONUS: float = 0.8

        # Temperature (exploration vs exploitation)
        self._temperature: float = self.TEMP_INIT

        # Stuck / exploration
        self.stuck_counter:     int   = 0
        self.no_progress_steps: int   = 0
        self.prev_pos:          Optional[Tuple[float, float]] = None
        self.position_history:  deque = deque(maxlen=120)
        self.boredom:           float = 0.0

        self._last_action: int = ACTION_IDLE

        # Reference kept for valence shaping
        self._emotion_sys: Optional[EmotionSystem] = None

    @property
    def temperature(self) -> float:
        return self._temperature

    def _anneal(self):
        """Cool exploration temperature over time."""
        self._temperature = max(self.TEMP_MIN,
                                self._temperature * self.TEMP_DECAY)

    def _update_stuck(self, env):
        ax, ay = env.agent_body.position
        self.position_history.append((ax, ay))
        if len(self.position_history) == 120:
            fx, fy = self.position_history[0]
            if abs(ax - fx) < 0.5 and abs(ay - fy) < 0.5:
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)
        if self.prev_pos:
            moved = abs(ax - self.prev_pos[0]) + abs(ay - self.prev_pos[1])
            if moved < 0.05: self.no_progress_steps += 1
            else:            self.no_progress_steps = 0
        self.prev_pos = (ax, ay)

    def _update_intent_scores_from_insights(self, memory: MemorySystem):
        action_to_intent: Dict[int, str] = {
            ACTION_LEFT:       "explore",
            ACTION_RIGHT:      "explore",
            ACTION_JUMP:       "climb",
            ACTION_SLEEP:      "rest",
            ACTION_SIT:        "rest",
            ACTION_GRAB:       "gather_food",
            ACTION_THROW:      "play",
            ACTION_CLIMB_UP:   "climb",
            ACTION_CLIMB_DOWN: "climb",
        }
        for action_id, avg_reward in memory.insights.items():
            intent = action_to_intent.get(action_id)
            if intent:
                old = memory.intent_bias.get(intent, 0.0)
                memory.intent_bias[intent] = old * 0.9 + avg_reward * 0.1

    # ── Intent selection ─────────────────────────────────────────
    def decide_intent(self, perception: Dict, memory: MemorySystem,
                      emotion_state: Dict[str, float],
                      fatigue: FatigueSystem,
                      inventory: InventorySystem,
                      circadian: Optional[CircadianRhythm] = None) -> str:
        traits = memory.identity["traits"]
        now    = time.time()

        # Base weights from causal memory + opportunity cost
        weights: Dict[str, float] = {}
        for i in INTENTS:
            bias     = memory.intent_bias.get(i, 0.0)
            idle_gap = now - self.intent_last_used.get(i, now - 30)
            # Slight boost for long-neglected intents (prevents starvation of intents)
            opportunity = min(0.5, idle_gap / 120.0)
            weights[i]  = max(0.1, 1.0 + bias + opportunity)

        # ── Inertia: stay with current intent ───────────────────
        weights[self.current_intent] = weights.get(self.current_intent, 1.0) + self.INERTIA_BONUS

        # ── Survival overrides ───────────────────────────────────
        # NEVER sleep when energy is full
        if fatigue.energy > 0.95:
            weights["rest"] = 0.0
        else:
            # Only sleep when energy is near 0 or fatigue is near 1
            if fatigue.energy < 0.10:
                weights["rest"] += 12.0
            elif fatigue.fatigue > 0.85:
                weights["rest"] += 10.0
            
            # Circadian night → only push rest if already somewhat tired
            if circadian is not None and circadian.is_night and (fatigue.energy < 0.3 or fatigue.fatigue > 0.6):
                weights["rest"] += circadian.sleep_drive * 6.0

        if fatigue.is_hungry:
            if inventory.food > 0:
                weights["eat"] += 5.0
            else:
                weights["gather_food"] += 3.0
                if inventory.seeds > 0:
                    weights["farm"] += 2.5

        if inventory.food < 2:
            weights["gather_food"] += 2.5
            if inventory.seeds > 0:
                weights["farm"] += 1.5

        if perception.get("farm_ready", 0) > 0:
            weights["farm"] += 4.0

        # ── Emotional biasing ─────────────────────────────────
        em = emotion_state
        happy_drive = em["happiness"] * traits["curiosity"]
        if em["happiness"] > 0.5:
            weights["play"]       += happy_drive * 2.0
            weights["use_updraft"] += happy_drive * 1.5
        if em["fear"]     > 0.5: weights["explore"] += 1.5
        if em["anger"]    > 0.4: weights["play"]    += 1.0
        if em["sadness"]  > 0.4: weights["rest"]    += 1.5
        if em["surprise"] > 0.4: weights["use_updraft"] += 1.5

        # ── Fatigue suppresses active intents ──────────────────
        for i in ("explore", "use_updraft", "play", "climb"):
            weights[i] *= max(0.1, 1.0 - fatigue.fatigue * 0.5)

        # ── Confidence affects risk-taking ──────────────────────
        conf = traits["confidence"]
        weights["climb"]      *= 0.5 + conf
        weights["use_updraft"] *= 0.5 + conf

        # ── Existential detachment ──────────────────────────────
        if memory.identity["beliefs"].get("world_is_simulation"):
            weights["explore"] = max(0.1, weights["explore"] - 0.5)

        # Normalise (prevent negatives)
        for k in weights:
            weights[k] = max(0.01, weights[k])

        return softmax_sample(weights, temperature=self._temperature)

    # ── Observation builder ──────────────────────────────────────
    def build_observation(self, state: Dict, intent: str,
                          inventory: InventorySystem) -> List[float]:
        base       = state_to_vector(state)
        intent_hot = [0.0] * NUM_INTENTS
        if intent in INTENT_TO_IDX:
            intent_hot[INTENT_TO_IDX[intent]] = 1.0
        return base + intent_hot + inventory.to_vector()

    # ── Main cognitive loop ──────────────────────────────────────
    def decide(self, env, emotion_sys: EmotionSystem,
               fatigue: FatigueSystem, memory: MemorySystem,
               inventory: InventorySystem, farm_system: FarmSystem,
               circadian: Optional[CircadianRhythm] = None) -> int:

        if not self.autonomous:
            return ACTION_IDLE

        self._emotion_sys = emotion_sys   # kept for reward shaping
        DT = 1.0 / 60.0
        self.intent_timer += DT * fatigue.decision_frequency
        self._update_stuck(env)
        ax, ay = env.agent_body.position
        self.novelty_sensor.visit(ax, ay)
        self.boredom += 0.00035
        self._anneal()

        if len(memory.long_term) > 50 and self.life_goal == "understand_world":
            self.life_goal = random.choice(
                ["seek_pleasure", "avoid_pain", "master_environment"])

        # 1. Perception
        perception = perceive(env, emotion_sys, fatigue, memory,
                              inventory, farm_system, circadian)

        if self.no_progress_steps > 300:
            emotion_sys.state["anger"]   = min(1.0, emotion_sys.state["anger"]   + 0.10)
            emotion_sys.state["sadness"] = min(1.0, emotion_sys.state["sadness"] + 0.05)
            self.no_progress_steps = 0

        # 2. Interpretation
        meaning = interpret(perception)

        # 3. Thought
        thoughts      = think(perception, meaning, emotion_sys.state, memory)
        meta_thoughts = meta_think(thoughts, memory)
        thought_str   = thoughts[0] if thoughts else "..."
        if meta_thoughts:
            thought_str += " | " + meta_thoughts[0]
        self.current_thought = thought_str

        # 4. Periodic reflection
        if int(self.intent_timer * 60) % 180 == 0:
            memory.reflect()
            self._update_intent_scores_from_insights(memory)

        # 5. Intent selection (with inertia / switch check)
        should_switch = (
            self.intent_timer > self.intent_duration or
            self.boredom > 1.0 or
            self.stuck_counter > 12 or
            "This isn't working. Try something else." in thoughts
        )

        if should_switch:
            self.intent_timer   = 0.0
            self.stuck_counter  = 0
            self.boredom        = 0.0
            self.intent_last_used[self.current_intent] = time.time()
            self.current_intent = self.decide_intent(
                perception, memory, emotion_sys.state,
                fatigue, inventory, circadian)
            self.intent_duration = random.uniform(8.0, 15.0) / max(
                0.3, fatigue.decision_frequency)

        # 6. Intent side-effects (resource interactions, not RL)
        self._apply_intent_effects(env, fatigue, emotion_sys,
                                   inventory, farm_system)

        # 7. Build observation
        state = get_full_state(env, emotion_sys, fatigue, inventory, circadian)
        obs   = self.build_observation(state, self.current_intent, inventory)

        # 8. Action masking
        valid_actions = get_valid_actions(env, fatigue)

        # 9. RL policy
        action = self.rl_policy.predict(obs, self.current_intent, valid_actions)

        # 10. Impulsive override (human-like irrationality — rare)
        if random.random() < 0.012:
            action = random.choice(valid_actions)

        self.metrics.record(self.current_intent, 0.0)   # reward filled in by caller
        self._last_action = action
        return action

    def _apply_intent_effects(self, env, fatigue: FatigueSystem,
                              emotion_sys: EmotionSystem,
                              inventory: InventorySystem,
                              farm_system: FarmSystem):
        intent = self.current_intent
        ax     = env.agent_body.position.x

        if intent == "eat" and fatigue.is_hungry:
            if inventory.eat():
                fatigue.eat(nutrition=0.5)
                emotion_sys.state["happiness"] = min(1.0, emotion_sys.state["happiness"] + 0.10)
                emotion_sys.state["sadness"]   = max(0.0, emotion_sys.state["sadness"]   - 0.05)

        elif intent == "farm":
            result = farm_system.try_interact(ax, inventory)
            if result == "harvested":
                emotion_sys.state["happiness"] = min(1.0, emotion_sys.state["happiness"] + 0.15)
            elif result == "planted":
                emotion_sys.state["happiness"] = min(1.0, emotion_sys.state["happiness"] + 0.05)

        elif intent == "gather_food":
            # Probability scales smoothly with proximity to tree (x ≈ -5)
            dist = abs(ax - (-5.0))
            if dist < 2.5:
                gather_prob = 0.015 * max(0.1, 1.0 - dist / 2.5)
                if random.random() < gather_prob:
                    inventory.gather(1)
                    emotion_sys.state["happiness"] = min(
                        1.0, emotion_sys.state["happiness"] + 0.04)

    # ── Action applicator (unchanged API) ────────────────────────
    def apply_action(self, env, action: int):
        if action == ACTION_LEFT:
            if getattr(env, 'is_crouching', False): env.stand()
            env.walk_left()
        elif action == ACTION_RIGHT:
            if getattr(env, 'is_crouching', False): env.stand()
            env.walk_right()
        elif action == ACTION_JUMP:
            if getattr(env, 'is_crouching', False): env.stand()
            env.jump()
        elif action == ACTION_CROUCH:
            if getattr(env, 'is_crouching', False): env.stand()
            else:                                    env.crouch()
        elif action == ACTION_SIT:
            env.sit()
        elif action == ACTION_SLEEP:
            if not getattr(env, 'is_sleeping', False) and env.is_grounded():
                env.toggle_sleep()
        elif action == ACTION_GRAB:
            if getattr(env, 'carried_body', None): env.release_object()
            else:                                   env.grab_object()
        elif action == ACTION_THROW:
            env.throw_or_kick()
        elif action == ACTION_CLIMB_UP:
            if getattr(env, 'is_climbing', False):
                env.move_on_ladder(0, 2.5)
            else:
                ladder = env.near_ladder()
                if ladder: env.start_climb(ladder)
        elif action == ACTION_CLIMB_DOWN:
            if getattr(env, 'is_climbing', False):
                env.move_on_ladder(0, -2.0)


# ─────────────────────────────────────────────────────────────────
# LLM ADVISOR  (optional Gemini intent override)
# ─────────────────────────────────────────────────────────────────
class LLMAdvisor:
    VALID_INTENTS = set(INTENTS)

    def __init__(self, call_interval: float = 30.0):
        self._last_call   = 0.0
        self._last_advice = "explore"
        self.interval     = call_interval
        self.api_key      = os.environ.get("GOOGLE_API_KEY")
        self.model        = None
        self._is_fetching = False
        self._lock        = threading.Lock()

        if genai and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                print(f"LLMAdvisor: Gemini init failed: {e}")

    def get_advice(self, state: Dict[str, Any]) -> str:
        now = time.time()
        if (now - self._last_call >= self.interval) and not self._is_fetching:
            self._last_call = now
            if self.model:
                self._is_fetching = True
                threading.Thread(target=self._async_fetch,
                                 args=(copy.deepcopy(state),), daemon=True).start()
            else:
                self._last_advice = self._call_heuristic(state)
        return self._last_advice

    def _async_fetch(self, state: Dict[str, Any]):
        advice = self._call_gemini(state)
        with self._lock:
            self._last_advice = advice
            self._is_fetching = False

    def _call_gemini(self, state: Dict[str, Any]) -> str:
        prompt = self._build_prompt(state)
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1, max_output_tokens=20, temperature=0.7))
            text = resp.text.strip().lower()
            for intent in INTENTS:
                if intent in text:
                    return intent
        except Exception as e:
            print(f"LLMAdvisor: Gemini call failed ({e}).")
        return self._call_heuristic(state)

    def _build_prompt(self, state: Dict[str, Any]) -> str:
        pos  = state["position"]
        em   = state["emotions"]
        inv  = state.get("inventory", {"food": "?", "seeds": "?"})
        circ = ""
        if "circadian_phase" in state:
            phase_name = "night" if state.get("circadian_sleep_drive", 0) > 0.3 else "day"
            circ = f"Time of day: {phase_name} (sleep drive: {state['circadian_sleep_drive']:.2f}).\n"
        return (
            f"You are the sub-conscious advisor for a stick figure named Alan.\n"
            f"Alan is at ({pos[0]:.1f}, {pos[1]:.1f}). "
            f"Energy={state['energy']:.2f}, Hunger={state['hunger']:.2f}, "
            f"Fatigue={state['fatigue']:.2f}.\n"
            f"Inventory: food={inv.get('food','?')}, seeds={inv.get('seeds','?')}.\n"
            f"{circ}"
            f"Dominant feeling: {state['dominant_emotion']}. "
            f"Top emotions: {', '.join(f'{k}={v:.2f}' for k,v in em.items() if v > 0.2)}.\n"
            f"\nPick ONE intent from {INTENTS}. Reply with ONLY the intent name."
        )

    def _call_heuristic(self, state: Dict[str, Any]) -> str:
        e, h = state["energy"], state["hunger"]
        em   = state["emotions"]
        sd   = state.get("circadian_sleep_drive", 0.0)

        if e < 0.15 or sd > 0.6:  return "rest"
        if h > 0.80:               return "eat"
        if em["happiness"] > 0.80: return "play"
        if em["fear"]      > 0.60: return "explore"
        choices = ["explore", "gather_food", "farm"]
        if state["position"][1] < 2.0:
            choices.append("climb")
        if state.get("in_updraft"):
            choices.append("use_updraft")
        return random.choice(choices)

    @staticmethod
    def reward_bias(advice: str, state: Dict[str, Any]) -> float:
        m = {
            "rest":        lambda s: 0.12 if s["energy"] < 0.30 else 0.0,
            "eat":         lambda s: 0.10 if s["hunger"] > 0.60 else 0.0,
            "play":        lambda s: 0.05,
            "explore":     lambda s: 0.05,
            "gather_food": lambda s: 0.08,
            "farm":        lambda s: 0.06,
            "climb":       lambda s: 0.05 if s["position"][1] < 4.0 else 0.0,
            "use_updraft": lambda s: 0.10,
            "store_food":  lambda s: 0.05,
        }
        return m.get(advice, lambda s: 0.0)(state)


# ─────────────────────────────────────────────────────────────────
# GYM WRAPPER
# ─────────────────────────────────────────────────────────────────
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np


class AlanEnv(gym.Env):
    """
    Observation: 29-dim  [state(18) + intent_one_hot(9) + inventory(2)]
    Action:      Discrete(11)
    """
    metadata = {"render_modes": []}

    def __init__(self, env_factory):
        super().__init__()
        self.env_factory = env_factory
        self.env         = env_factory()

        self.emotion_sys  = EmotionSystem()
        self.fatigue_sys  = FatigueSystem()
        self.memory       = MemorySystem()
        self.inventory    = InventorySystem()
        self.farm_system  = FarmSystem()
        self.circadian    = CircadianRhythm()
        self.reward_norm  = RunningNorm()
        self.advisor      = LLMAdvisor()
        self.brain        = BrainDecision(autonomous=True)

        self._step_count     = 0
        self._current_intent = "explore"

        self.action_space      = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype="float32")

    def _get_obs(self) -> np.ndarray:
        state    = get_full_state(self.env, self.emotion_sys,
                                  self.fatigue_sys, self.inventory, self.circadian)
        obs_list = self.brain.build_observation(state, self._current_intent, self.inventory)
        return np.array(obs_list, dtype="float32")

    def reset(self, seed=None, options=None):
        self.env         = self.env_factory()
        self.emotion_sys = EmotionSystem()
        self.fatigue_sys = FatigueSystem()
        self.memory      = MemorySystem()
        self.inventory   = InventorySystem()
        self.farm_system = FarmSystem()
        self.circadian   = CircadianRhythm()
        self.reward_norm = RunningNorm()
        self._step_count     = 0
        self._current_intent = "explore"
        return self._get_obs(), {}

    def step(self, action: int):
        DT = 1.0 / 60.0
        is_sleeping = getattr(self.env, 'is_sleeping', False)

        self.farm_system.update(DT)
        self.circadian.update(DT, is_sleeping=is_sleeping)

        perception = perceive(self.env, self.emotion_sys, self.fatigue_sys,
                              self.memory, self.inventory, self.farm_system, self.circadian)
        meaning    = interpret(perception)
        thoughts   = think(perception, meaning, self.emotion_sys.state, self.memory)

        if int(self._step_count) % 180 == 0:
            self.memory.reflect()
            self.brain._update_intent_scores_from_insights(self.memory)

        self._current_intent = self.brain.decide_intent(
            perception, self.memory, self.emotion_sys.state,
            self.fatigue_sys, self.inventory, self.circadian)

        self.brain._apply_intent_effects(
            self.env, self.fatigue_sys, self.emotion_sys,
            self.inventory, self.farm_system)

        self.brain.apply_action(self.env, action)
        self.env.update_physics(DT)
        self.fatigue_sys.update(self.env, DT, self.circadian)

        state  = get_full_state(self.env, self.emotion_sys,
                                self.fatigue_sys, self.inventory, self.circadian)
        advice = self.advisor.get_advice(state)

        raw_reward = compute_reward(
            self.env, self.emotion_sys.state, self.fatigue_sys,
            self.inventory, self.brain, self.farm_system)
        raw_reward += LLMAdvisor.reward_bias(advice, state)
        reward = self.reward_norm.update(raw_reward)

        update_emotions(self.env, self.emotion_sys.state,
                        self.fatigue_sys, raw_reward, self.circadian)
        self.emotion_sys.decay()

        self.memory.record(state, raw_reward, action, intent=self._current_intent)
        self.brain.metrics.record(self._current_intent, raw_reward)

        obs  = self._get_obs()
        done = self._step_count > 3600
        self._step_count += 1
        return obs, reward, done, False, {
            "intent": self._current_intent,
            "raw_reward": raw_reward,
            "circadian_phase": self.circadian.phase,
        }

    def render(self): pass
    def close(self):  pass
