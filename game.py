"""
game.py  —  Stick Figure Sandbox  (v2)

What's new vs v1
────────────────
  P           : pause / resume
  1/2/3/4     : simulation speed (×0.5 / ×1 / ×2 / ×4)
  F1          : toggle key-help overlay
  F12         : screenshot → screenshots/
  D           : toggle debug overlay (action mask, temperature, novelty)
  A           : toggle autonomous brain
  X           : eat (player mode)
  ← → ↑ C G T: move / jump / crouch / grab / throw

  • Day/night sky tint follows CircadianRhythm
  • Notification system – floating event text
  • Scrolling reward-history graph (bottom right)
  • Personality trait bars in mind HUD
  • Intent-score heatmap panel
  • Brain temperature + novelty cells shown in HUD
  • Variable naming cleaned up (no `surface` shadow)
  • Farm plots flash gold when ready
  • Fatigue bar turns red when critical
"""

import pygame
import sys
import math
import time
import os
import random
from dotenv import load_dotenv
load_dotenv()

from thought import Environment
from brain import (
    EmotionSystem, FatigueSystem, MemorySystem, InventorySystem,
    FarmSystem, BrainDecision, LLMAdvisor, CircadianRhythm, RunningNorm,
    get_full_state, compute_reward, update_emotions,
    ACTION_NAMES, ACTION_IDLE, INTENTS, INTENT_TO_IDX,
)

# ───────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────
BG_IMAGE_PATH  = "back_ground.png"
SCREEN_W, SCREEN_H = 800, 600
FPS            = 60
BASE_DT        = 1.0 / FPS
SCALE          = 40
ORIGIN_X       = 400
ORIGIN_Y       = 560
SCREENSHOT_DIR = "screenshots"

SPEED_LEVELS = [0.5, 1.0, 2.0, 4.0]

# ───────────────────────────────────────────────────────────────
# Palette
# ───────────────────────────────────────────────────────────────
C_SKY        = ( 20,  25,  45)
C_SKY_DAWN   = ( 60,  40,  30)
C_SKY_DAY    = ( 20,  60, 120)
C_DIRT       = ( 87,  59,  30)
C_GRASS      = ( 60, 150,  40)
C_GRASS_TOP  = ( 90, 190,  60)
C_WALL       = ( 40,  40,  55)
C_BOX        = (160, 100,  50)
C_BOX_DARK   = (100,  50,  20)
C_HUD        = (200, 200, 200)
C_HUD_DIM    = (110, 110, 130)
C_HUD_BG     = (  0,   0,   0, 190)
C_STICK      = ( 50, 200, 250)
C_HEAD       = (255, 220, 150)
C_WOOD_DARK  = ( 80,  40,  15)
C_WOOD_LIGHT = (139,  69,  19)
C_WATER      = ( 30, 120, 200, 160)

EMOTION_COLOURS = {
    "anger":     (220,  60,  60),
    "disgust":   ( 80, 180,  60),
    "fear":      (180,  80, 220),
    "happiness": (255, 220,  50),
    "sadness":   ( 60, 120, 220),
    "surprise":  (255, 180,  40),
}

INTENT_COLOURS = {
    "explore":    (100, 220, 255),
    "rest":       ( 80, 150, 255),
    "eat":        (255, 160,  60),
    "gather_food":(100, 220,  80),
    "farm":       (180, 230,  80),
    "store_food": (200, 180,  80),
    "use_updraft":(160, 230, 255),
    "climb":      (200, 200, 100),
    "play":       (255, 180, 200),
}


def phys_to_screen(x, y):
    return int(ORIGIN_X + x * SCALE), int(ORIGIN_Y - y * SCALE)

def lerp_colour(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ───────────────────────────────────────────────────────────────
# NOTIFICATION SYSTEM  — floating event text
# ───────────────────────────────────────────────────────────────
class Notification:
    def __init__(self, text: str, colour, x: int, y: int, lifetime: float = 2.5):
        self.text     = text
        self.colour   = colour
        self.x        = float(x)
        self.y        = float(y)
        self.lifetime = lifetime
        self._age     = 0.0

    def update(self, dt: float) -> bool:
        self._age += dt
        self.y    -= 22.0 * dt   # float upward
        return self._age < self.lifetime

    @property
    def alpha(self) -> int:
        fade_start = self.lifetime * 0.6
        if self._age < fade_start:
            return 255
        frac = (self._age - fade_start) / (self.lifetime - fade_start)
        return int(255 * (1.0 - frac))

    def draw(self, surface, font):
        txt_surf = font.render(self.text, True, self.colour)
        txt_surf.set_alpha(self.alpha)
        surface.blit(txt_surf, (int(self.x) - txt_surf.get_width() // 2, int(self.y)))


class NotificationSystem:
    def __init__(self):
        self._notes: list = []

    def add(self, text: str, colour=(255, 255, 100),
            x: int = SCREEN_W // 2, y: int = SCREEN_H // 2):
        # Avoid duplicate spam
        recent_texts = [n.text for n in self._notes[-3:]]
        if text in recent_texts:
            return
        self._notes.append(Notification(text, colour, x, y))
        if len(self._notes) > 20:
            self._notes.pop(0)

    def update(self, dt: float):
        self._notes = [n for n in self._notes if n.update(dt)]

    def draw(self, surface, font):
        for note in self._notes:
            note.draw(surface, font)


# ───────────────────────────────────────────────────────────────
# REWARD HISTORY GRAPH
# ───────────────────────────────────────────────────────────────
class RewardGraph:
    MAX_POINTS = 300
    W, H       = 180, 50
    X, Y       = SCREEN_W - 195, SCREEN_H - 65

    def __init__(self):
        self._data: list = []

    def push(self, reward: float):
        self._data.append(reward)
        if len(self._data) > self.MAX_POINTS:
            self._data.pop(0)

    def draw(self, surface, font):
        if len(self._data) < 2:
            return
        panel = pygame.Surface((self.W + 10, self.H + 24), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 170))
        surface.blit(panel, (self.X - 5, self.Y - 18))

        lbl = font.render("REWARD", True, C_HUD_DIM)
        surface.blit(lbl, (self.X, self.Y - 16))

        mn = min(self._data)
        mx = max(self._data)
        rng = mx - mn or 1.0

        pts = []
        for i, v in enumerate(self._data):
            px = self.X + int(i / max(1, len(self._data) - 1) * self.W)
            py = self.Y + self.H - int((v - mn) / rng * self.H)
            pts.append((px, py))

        if len(pts) >= 2:
            # Zero line
            zero_y = self.Y + self.H - int((0.0 - mn) / rng * self.H)
            pygame.draw.line(surface, (60, 60, 60),
                             (self.X, zero_y), (self.X + self.W, zero_y), 1)
            pygame.draw.lines(surface, (100, 220, 100), False, pts, 2)

        # Latest value
        latest = self._data[-1]
        col    = (100, 220, 100) if latest >= 0 else (220, 80, 80)
        val_lbl = font.render(f"{latest:+.2f}", True, col)
        surface.blit(val_lbl, (self.X + self.W - val_lbl.get_width(), self.Y - 16))


# ───────────────────────────────────────────────────────────────
# WORLD DRAWING
# ───────────────────────────────────────────────────────────────
def sky_colour(circadian: CircadianRhythm) -> tuple:
    light = circadian.light_level
    if light > 0.7:
        return lerp_colour(C_SKY_DAWN, C_SKY_DAY, (light - 0.7) / 0.3)
    elif light > 0.3:
        return lerp_colour(C_SKY, C_SKY_DAWN, (light - 0.3) / 0.4)
    else:
        return C_SKY


def draw_world(surface, env, bg_image, circadian: CircadianRhythm):
    if bg_image:
        surface.blit(bg_image, (0, 0))
        # Tint for day/night
        tint = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        night_alpha = int((1.0 - circadian.light_level) * 90)
        tint.fill((10, 10, 40, night_alpha))
        surface.blit(tint, (0, 0))
    else:
        sky_col = sky_colour(circadian)
        for y in range(SCREEN_H):
            frac = y / SCREEN_H
            r = max(5,  sky_col[0] - int(frac * 15))
            g = max(5,  sky_col[1] - int(frac * 15))
            b = min(140, sky_col[2] + int(frac * 20))
            pygame.draw.line(surface, (r, g, b), (0, y), (SCREEN_W, y))

    # Stars at night
    if circadian.is_night:
        star_alpha = int(circadian.sleep_drive * 180)
        star_surf  = pygame.Surface((SCREEN_W, SCREEN_H // 2), pygame.SRCALPHA)
        rng = random.Random(42)
        for _ in range(60):
            sx, sy = rng.randint(0, SCREEN_W), rng.randint(0, SCREEN_H // 2)
            r      = rng.randint(1, 3)
            pygame.draw.circle(star_surf, (255, 255, 220, star_alpha), (sx, sy), r)
        surface.blit(star_surf, (0, 0))

    gx_l, gy = phys_to_screen(-10, 0)
    gx_r, _  = phys_to_screen( 10, 0)
    pygame.draw.rect(surface, C_DIRT,      (gx_l, gy, gx_r - gx_l, SCREEN_H - gy))
    pygame.draw.rect(surface, C_GRASS,     (gx_l, gy, gx_r - gx_l, 15))
    pygame.draw.rect(surface, C_GRASS_TOP, (gx_l, gy, gx_r - gx_l,  4))

    wall_px_l, _ = phys_to_screen(-10, 0)
    if wall_px_l > 0:
        pygame.draw.rect(surface, C_WALL, (0, 0, wall_px_l, SCREEN_H))
    wall_px_r, _ = phys_to_screen(10, 0)
    if wall_px_r < SCREEN_W:
        pygame.draw.rect(surface, C_WALL, (wall_px_r, 0, SCREEN_W - wall_px_r, SCREEN_H))

    wx1, wy1 = phys_to_screen(2.0, env.water_level)
    wx2, wy2 = phys_to_screen(9.0, -2.0)
    pool_w   = wx2 - wx1
    pool_h   = SCREEN_H - wy1
    water_surf = pygame.Surface((pool_w, pool_h), pygame.SRCALPHA)
    water_surf.fill(C_WATER)
    t = time.time() * 2
    for i in range(0, pool_w, 30):
        off = math.sin(t + i * 0.1) * 5
        pygame.draw.line(water_surf, (100, 200, 255, 180),
                         (i, 5 + off), (i + 15, 5 + off), 2)
    surface.blit(water_surf, (wx1, wy1))


def draw_updraft(surface, env):
    wx1, wy_base = phys_to_screen(env.updraft_zone[0], 0)
    wx2, wy_top  = phys_to_screen(env.updraft_zone[1], 6.0)
    width  = wx2 - wx1
    height = wy_base - wy_top
    wind_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    t = time.time() * 4
    for i in range(12):
        x = (i * 18 + math.sin(t + i) * 5) % width
        y = height - ((t * 60 + i * 30) % height)
        pygame.draw.line(wind_surf, (150, 230, 255, 120),
                         (x, y), (x, y - 25), 3)
    surface.blit(wind_surf, (wx1, wy_top))


def draw_ball(surface, body, radius):
    x, y  = phys_to_screen(body.position.x, body.position.y)
    r     = int(radius * SCALE)
    pygame.draw.circle(surface, (220, 30, 30), (x, y), r)
    pygame.draw.circle(surface, (150, 20, 20), (x, y), r, 2)
    angle = body.angle
    hx = x + int(r * 0.5 * math.cos(angle - 0.5))
    hy = y - int(r * 0.5 * math.sin(angle - 0.5))
    pygame.draw.circle(surface, (255, 150, 150), (hx, hy), int(r * 0.25))


def draw_platform(surface, body, w, h):
    x, y   = phys_to_screen(body.position.x, body.position.y)
    pw, ph = int(w * SCALE), int(h * SCALE)
    px, py = x - pw // 2, y - ph // 2
    pygame.draw.rect(surface, C_DIRT,      (px, py, pw, ph))
    pygame.draw.rect(surface, C_GRASS,     (px, py, pw, int(ph * 0.4)))
    pygame.draw.rect(surface, C_GRASS_TOP, (px, py, pw, 3))


def draw_leaves(surface, x, y):
    sx, sy = phys_to_screen(x, y)
    pygame.draw.circle(surface, ( 25, 100,  25), (sx,      sy - 120), 55)
    pygame.draw.circle(surface, ( 34, 139,  34), (sx - 30, sy -  90), 45)
    pygame.draw.circle(surface, ( 34, 139,  34), (sx + 30, sy -  90), 45)
    pygame.draw.circle(surface, ( 40, 160,  40), (sx,      sy - 140), 40)
    pygame.draw.circle(surface, ( 40, 160,  40), (sx - 20, sy - 120), 35)
    pygame.draw.circle(surface, ( 40, 160,  40), (sx + 20, sy - 120), 35)


def draw_bench(surface, env):
    bx, by = phys_to_screen(*env.bench_pos)
    pygame.draw.rect(surface, C_WOOD_DARK,  (bx - 15, by - 5,   6, 20))
    pygame.draw.rect(surface, C_WOOD_DARK,  (bx +  9, by - 5,   6, 20))
    pygame.draw.rect(surface, C_WOOD_LIGHT, (bx - 22, by - 10, 44,  8))
    pygame.draw.rect(surface, C_WOOD_DARK,  (bx - 18, by - 25,  4, 20))
    pygame.draw.rect(surface, C_WOOD_DARK,  (bx + 14, by - 25,  4, 20))
    pygame.draw.rect(surface, C_WOOD_LIGHT, (bx - 22, by - 30, 44, 10))


def draw_ladder(surface, ladder):
    if ladder.rect[0] == -5.5:
        x1, y1 = phys_to_screen(ladder.rect[0], ladder.rect[3])
        x2, y2 = phys_to_screen(ladder.rect[2], ladder.rect[1])
        pygame.draw.rect(surface, C_WOOD_DARK,  (x1, y1, x2 - x1, y2 - y1))
        pygame.draw.line(surface, C_WOOD_LIGHT, (x1 + 5, y1), (x1 + 5, y2), 2)
        pygame.draw.line(surface, C_WOOD_LIGHT, (x2 - 5, y1), (x2 - 5, y2), 2)


def draw_box(surface, body, width, height, fill_colour):
    verts = []
    hw, hh = width / 2, height / 2
    local_verts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    angle = body.angle
    for lv in local_verts:
        rx = lv[0] * math.cos(angle) - lv[1] * math.sin(angle) + body.position.x
        ry = lv[0] * math.sin(angle) + lv[1] * math.cos(angle) + body.position.y
        verts.append(phys_to_screen(rx, ry))
    if len(verts) >= 4:
        pygame.draw.polygon(surface, C_BOX,     verts)
        pygame.draw.line(surface,    C_BOX_DARK, verts[0], verts[2], 3)
        pygame.draw.line(surface,    C_BOX_DARK, verts[1], verts[3], 3)
        pygame.draw.polygon(surface, C_BOX_DARK, verts, 3)


# ───────────────────────────────────────────────────────────────
# FARM PLOTS
# ───────────────────────────────────────────────────────────────
def draw_farm_plots(surface, farm_system, font_sm):
    t = time.time()
    for plot in farm_system.plots:
        sx, sy = phys_to_screen(plot.x, 0.0)
        pw = int(0.75 * SCALE)
        ph = int(0.18 * SCALE)
        px = sx - pw // 2
        py = sy - ph

        if plot.state == "empty":
            col = (101, 67, 33)
        elif plot.state == "growing":
            progress = plot.growth_progress
            g   = int(80 + progress * 140)
            col = (40, g, 20)
        else:
            # Flash gold when ready
            flash = abs(math.sin(t * 3.0))
            col   = lerp_colour((200, 160, 30), (255, 230, 80), flash)

        pygame.draw.rect(surface, col, (px, py, pw, ph))
        pygame.draw.rect(surface, (0, 0, 0), (px, py, pw, ph), 1)

        if plot.state == "growing":
            stem_h  = int(plot.growth_progress * 22)
            stalk_x = sx
            pygame.draw.line(surface, (60, 180, 60),
                             (stalk_x, py), (stalk_x, py - stem_h), 2)
            if stem_h > 8:
                pygame.draw.circle(surface, (100, 200, 60),
                                   (stalk_x, py - stem_h), 3)
            bar_w = pw - 4
            fill  = int(bar_w * plot.growth_progress)
            pygame.draw.rect(surface, (40, 40, 40),    (px + 2, py - 6, bar_w, 4))
            pygame.draw.rect(surface, (100, 220, 60),  (px + 2, py - 6, fill,  4))

        elif plot.state == "ready":
            for di in (-5, 0, 5):
                pygame.draw.line(surface, (200, 160, 30),
                                 (sx + di, py), (sx + di, py - 18), 2)
                pygame.draw.circle(surface, (255, 200, 50),
                                   (sx + di, py - 20), 3)
            # "READY" label
            rdy = font_sm.render("✓READY", True, (255, 230, 80))
            surface.blit(rdy, (px, py - 22))


# ───────────────────────────────────────────────────────────────
# STICK FIGURE
# ───────────────────────────────────────────────────────────────
def draw_stick_figure(surface, body, width, height, walk_phase=0.0,
                      is_walking=False, is_crouching=False,
                      is_sleeping=False, is_sitting=False, facing=1):
    x, y  = body.position
    angle = body.angle

    if is_sitting:
        head_radius      = width * 0.18
        head_local       = (-width * 0.05 * facing,  height * 0.10)
        neck_local       = (-width * 0.05 * facing, -height * 0.05)
        hip_local        = (-width * 0.05 * facing, -height * 0.30)
        knee_l_local     = ( width * 0.25 * facing, -height * 0.30)
        knee_r_local     = ( width * 0.25 * facing, -height * 0.30)
        foot_left_local  = ( width * 0.25 * facing, -height * 0.55)
        foot_right_local = ( width * 0.25 * facing, -height * 0.55)
        hand_left_local  = ( width * 0.15 * facing, -height * 0.15)
        hand_right_local = ( width * 0.15 * facing, -height * 0.15)
    elif is_crouching:
        head_radius      = width * 0.18
        head_local       = (0,  height * 0.20)
        neck_local       = (0,  height * 0.10)
        hip_local        = (0, -height * 0.10)
        swing            = 0.5 * math.sin(walk_phase) if is_walking else 0.0
        foot_left_local  = (-width * 0.25 + swing * 0.15, -height * 0.45)
        foot_right_local = ( width * 0.25 - swing * 0.15, -height * 0.45)
        hand_left_local  = (-width * 0.40 - swing * 0.2,   height * 0.05)
        hand_right_local = ( width * 0.40 + swing * 0.2,   height * 0.05)
    else:
        head_radius      = width * 0.18
        head_local       = (0,  height * 0.35)
        neck_local       = (0,  height * 0.20)
        hip_local        = (0, -height * 0.15)
        swing            = 0.5 * math.sin(walk_phase) if is_walking else 0.0
        foot_left_local  = (-width * 0.25 + swing * 0.15, -height * 0.45)
        foot_right_local = ( width * 0.25 - swing * 0.15, -height * 0.45)
        hand_left_local  = (-width * 0.40 - swing * 0.2,   height * 0.05)
        hand_right_local = ( width * 0.40 + swing * 0.2,   height * 0.05)

    def wp(lx, ly):
        wx = lx * math.cos(angle) - ly * math.sin(angle) + x
        wy = lx * math.sin(angle) + ly * math.cos(angle) + y
        return phys_to_screen(wx, wy)

    head_c = wp(*head_local)
    neck   = wp(*neck_local)
    hip    = wp(*hip_local)
    foot_l = wp(*foot_left_local)
    foot_r = wp(*foot_right_local)
    hand_l = wp(*hand_left_local)
    hand_r = wp(*hand_right_local)

    pygame.draw.line(surface, (20, 20, 20), neck, hip,    6)
    pygame.draw.line(surface, C_STICK,      neck, hip,    4)
    pygame.draw.line(surface, C_STICK,      neck, hand_l, 3)
    pygame.draw.line(surface, C_STICK,      neck, hand_r, 3)

    if is_sitting:
        knee_l = wp(*knee_l_local)
        knee_r = wp(*knee_r_local)
        for a, b in [(hip, knee_l), (knee_l, foot_l),
                     (hip, knee_r), (knee_r, foot_r)]:
            pygame.draw.line(surface, (20, 20, 20), a, b, 6)
            pygame.draw.line(surface, C_STICK,      a, b, 4)
    else:
        for a, b in [(hip, foot_l), (hip, foot_r)]:
            pygame.draw.line(surface, (20, 20, 20), a, b, 6)
            pygame.draw.line(surface, C_STICK,      a, b, 4)

    hr = int(head_radius * SCALE)
    pygame.draw.circle(surface, (20, 20, 20), head_c, hr + 1)
    pygame.draw.circle(surface, C_HEAD,       head_c, hr)

    eye_dir_y = -math.sin(angle)
    eye1 = (head_c[0] + int(facing * 3) - 2, head_c[1] + int(eye_dir_y * 5) - 3)
    eye2 = (head_c[0] + int(facing * 3) + 3, head_c[1] + int(eye_dir_y * 5) - 3)

    if is_sleeping:
        pygame.draw.line(surface, (0,0,0), (eye1[0]-2, eye1[1]), (eye1[0]+2, eye1[1]), 2)
        pygame.draw.line(surface, (0,0,0), (eye2[0]-2, eye2[1]), (eye2[0]+2, eye2[1]), 2)
        t = time.time() * 2
        for i in range(3):
            zy = head_c[1] - 20 - (int(t * 15 + i * 15) % 40)
            zx = head_c[0] + 10 + int(math.sin(t + i) * 5)
            fz = pygame.font.SysFont("monospace", 10 + i * 2, bold=True)
            surface.blit(fz.render("Z", True, (200, 220, 255)), (zx, zy))
    else:
        pygame.draw.circle(surface, (0, 0, 0), eye1, 3)
        pygame.draw.circle(surface, (0, 0, 0), eye2, 3)


# ───────────────────────────────────────────────────────────────
# THOUGHT BUBBLE
# ───────────────────────────────────────────────────────────────
def draw_thought_bubble(surface, env, brain, font_sm):
    if not brain.autonomous or not hasattr(brain, 'current_thought'): return
    text = brain.current_thought
    if not text or text == "...": return

    ax, ay    = phys_to_screen(env.agent_body.position.x,
                                env.agent_body.position.y)
    txt_surf  = font_sm.render(text, True, (40, 40, 40))
    w, h      = txt_surf.get_size()
    pad       = 8
    bx        = ax - w // 2
    by        = ay - 80

    rect = (bx - pad, by - pad, w + pad * 2, h + pad * 2)
    pygame.draw.rect(surface, (255, 255, 255), rect, border_radius=10)
    pygame.draw.rect(surface, (200, 200, 200), rect, 2, border_radius=10)
    tail = [(ax - 6, by + h + pad - 1), (ax + 6, by + h + pad - 1),
            (ax,    by + h + pad + 10)]
    pygame.draw.polygon(surface, (255, 255, 255), tail)
    pygame.draw.polygon(surface, (200, 200, 200), tail, 2)
    pygame.draw.line(surface, (255, 255, 255),
                     (ax - 5, by + h + pad - 1),
                     (ax + 5, by + h + pad - 1), 3)
    surface.blit(txt_surf, (bx, by))


def draw_intent_badge(surface, env, brain, font_sm):
    if not brain.autonomous: return
    intent = brain.current_intent
    colour = INTENT_COLOURS.get(intent, (200, 200, 200))
    ax, ay = phys_to_screen(env.agent_body.position.x,
                             env.agent_body.position.y)
    label  = font_sm.render(f"[{intent}]", True, colour)
    surface.blit(label, (ax - label.get_width() // 2, ay - 100))


# ───────────────────────────────────────────────────────────────
# MIND HUD  (left panel — extended with traits)
# ───────────────────────────────────────────────────────────────
def draw_mind_hud(surface, emotion_sys, fatigue_sys, memory,
                  inventory, brain, advisor_intent, font_sm, reward, circadian):
    PANEL_X, PANEL_Y = 10, 10
    PANEL_W, PANEL_H = 230, 290
    BAR_W = 130
    BAR_H = 9
    ROW_H = 14

    panel = pygame.Surface((PANEL_W, PANEL_H), pygame.SRCALPHA)
    panel.fill(C_HUD_BG)
    surface.blit(panel, (PANEL_X, PANEL_Y))

    def bar(label, value, colour, row, warn_thresh=None):
        y  = PANEL_Y + 8 + row * ROW_H
        lc = C_HUD_DIM
        if warn_thresh and value > warn_thresh:
            lc = (220, 80, 80)
        surface.blit(font_sm.render(label, True, lc), (PANEL_X + 5, y))
        pygame.draw.rect(surface, (50, 50, 60), (PANEL_X + 68, y + 1, BAR_W, BAR_H))
        fw = int(BAR_W * max(0.0, min(1.0, value)))
        if fw > 0:
            pygame.draw.rect(surface, colour, (PANEL_X + 68, y + 1, fw, BAR_H))
        surface.blit(font_sm.render(f"{int(value*100):3d}%", True, C_HUD),
                     (PANEL_X + 204, y))

    row = 0
    energy_col  = (50, 220, 100) if not fatigue_sys.is_exhausted else (220, 80, 50)
    fatigue_col = (180, 80, 80)  if fatigue_sys.fatigue > 0.6   else (180, 160, 80)
    hunger_col  = (210, 80, 50)  if fatigue_sys.is_hungry        else (210, 140, 50)
    bar("Energy",  fatigue_sys.energy,  energy_col,  row);               row += 1
    bar("Fatigue", fatigue_sys.fatigue, fatigue_col, row, warn_thresh=0.75); row += 1
    bar("Hunger",  fatigue_sys.hunger,  hunger_col,  row, warn_thresh=0.80); row += 1

    # Circadian sleep drive
    if circadian.sleep_drive > 0.05:
        bar("Sleep↓", circadian.sleep_drive, (80, 120, 200), row)
        row += 1

    for emo, colour in EMOTION_COLOURS.items():
        bar(emo[:5].capitalize(), emotion_sys.state[emo], colour, row)
        row += 1

    # ── Traits ──────────────────────────────────────────────
    traits = memory.identity["traits"]
    trait_row_y = PANEL_Y + 8 + row * ROW_H
    surface.blit(font_sm.render("─ Traits ─", True, (80, 80, 100)),
                 (PANEL_X + 5, trait_row_y))
    row += 1
    for trait, val in traits.items():
        bar(trait[:5].capitalize(), val, (140, 140, 200), row)
        row += 1

    # ── Intent / mood / advice ──────────────────────────────
    intent     = brain.current_intent if brain.autonomous else advisor_intent
    intent_col = INTENT_COLOURS.get(intent, C_HUD)
    dom_name, _ = emotion_sys.dominant()
    mood_col   = EMOTION_COLOURS.get(dom_name, C_HUD)

    y_bottom = PANEL_Y + PANEL_H - 52
    surface.blit(font_sm.render(f"Mood:   {dom_name}",  True, mood_col),
                 (PANEL_X + 5, y_bottom))
    surface.blit(font_sm.render(f"Intent: {intent}",   True, intent_col),
                 (PANEL_X + 5, y_bottom + 12))
    surface.blit(font_sm.render(f"Advice: {advisor_intent}", True, (200, 200, 255)),
                 (PANEL_X + 5, y_bottom + 24))

    r_col = (100, 220, 100) if reward >= 0 else (220, 80, 80)
    surface.blit(font_sm.render(
        f"Reward:{reward:+.2f}  LTM:{len(memory.long_term)}", True, r_col),
        (PANEL_X + 5, PANEL_Y + PANEL_H - 10))


# ───────────────────────────────────────────────────────────────
# INVENTORY HUD
# ───────────────────────────────────────────────────────────────
def draw_inventory_hud(surface, inventory, farm_system, font_sm):
    PANEL_X, PANEL_Y = 10, SCREEN_H - 80
    PANEL_W, PANEL_H = 200, 70
    panel = pygame.Surface((PANEL_W, PANEL_H), pygame.SRCALPHA)
    panel.fill(C_HUD_BG)
    surface.blit(panel, (PANEL_X, PANEL_Y))

    surface.blit(font_sm.render("INVENTORY", True, (180, 180, 180)),
                 (PANEL_X + 5, PANEL_Y + 5))
    surface.blit(font_sm.render(
        f"Food : {'█' * min(inventory.food, 10):<10} ({inventory.food})",
        True, (255, 160, 60)), (PANEL_X + 5, PANEL_Y + 20))
    surface.blit(font_sm.render(
        f"Seeds: {'▪' * min(inventory.seeds, 10):<10} ({inventory.seeds})",
        True, (160, 220, 100)), (PANEL_X + 5, PANEL_Y + 34))

    ready   = farm_system.ready_count()
    growing = farm_system.growing_count()
    farm_col = (220, 200, 50) if ready > 0 else (140, 200, 80)
    surface.blit(font_sm.render(
        f"Farm : {growing} growing  {ready} ready",
        True, farm_col), (PANEL_X + 5, PANEL_Y + 48))


# ───────────────────────────────────────────────────────────────
# AGENT / STATE HUD  (top right)
# ───────────────────────────────────────────────────────────────
def draw_hud(surface, env, brain, advisor_intent, font_sm, font_lg,
             grounded, speed_idx, paused, circadian: CircadianRhythm):
    ax, ay = env.agent_body.position
    PANEL_X, PANEL_Y = 240, 10
    PANEL_W, PANEL_H = 550, 95

    panel = pygame.Surface((PANEL_W, PANEL_H), pygame.SRCALPHA)
    panel.fill(C_HUD_BG)
    surface.blit(panel, (PANEL_X, PANEL_Y))

    phase_str = f"{'●' if circadian.is_night else '○'} {circadian.phase:.2f}"
    lines = [
        ("AGENT", f"pos ({ax:+.2f}, {ay:.2f}) m"),
        ("STATE", (f"Ground:{'Y' if grounded else 'N'}  "
                   f"Crouch:{'Y' if env.is_crouching else 'N'}  "
                   f"Water:{'Y' if env.in_water else 'N'}")),
        ("SPEED", f"×{SPEED_LEVELS[speed_idx]:.1f}   {'⏸ PAUSED' if paused else '▶ running'}"),
        ("CLOCK", f"Circadian {phase_str}   Temp={brain.temperature:.2f}"),
        ("MIND ", f"Intent: {brain.current_intent}  |  Advice: {advisor_intent}"),
    ]
    for i, (label, value) in enumerate(lines):
        y = PANEL_Y + 6 + i * 16
        surface.blit(font_sm.render(label + ": ", True, C_HUD_DIM), (PANEL_X + 10, y))
        surface.blit(font_sm.render(value,         True, C_HUD),     (PANEL_X + 68, y))


# ───────────────────────────────────────────────────────────────
# DEBUG OVERLAY
# ───────────────────────────────────────────────────────────────
def draw_debug(surface, env, brain, fatigue_sys, font_sm):
    from brain import get_valid_actions
    valid  = get_valid_actions(env, fatigue_sys)
    PANEL_X, PANEL_Y = 240, 115
    lines = [
        f"Valid actions: {[ACTION_NAMES[a] for a in valid]}",
        f"Stuck cnt: {brain.stuck_counter}  No-prog: {brain.no_progress_steps}",
        f"Novelty cells: {brain.novelty_sensor.unique_cells}",
        f"Boredom: {brain.boredom:.3f}  Temp: {brain.temperature:.3f}",
        f"Best intent: {brain.metrics.best_intent()}",
        f"Intent scores: " + "  ".join(
            f"{k[:3]}={v:.2f}" for k, v in brain.metrics.intent_rewards.items()
            if brain.metrics.intent_uses.get(k, 0) > 0)[:60],
    ]
    panel = pygame.Surface((560, len(lines) * 14 + 10), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 200))
    surface.blit(panel, (PANEL_X - 5, PANEL_Y - 5))
    for i, line in enumerate(lines):
        surface.blit(font_sm.render(line, True, (180, 255, 180)),
                     (PANEL_X, PANEL_Y + i * 14))


# ───────────────────────────────────────────────────────────────
# HELP OVERLAY  (F1)
# ───────────────────────────────────────────────────────────────
def draw_help(surface, font_sm):
    lines = [
        "CONTROLS",
        "← → : move        ↑ / Space : jump     C : crouch",
        "Shift : sprint     G : grab/release     T : throw",
        "E : climb          S : sit              Z : sleep",
        "X : eat (player)   F : push/pull",
        "",
        "A : toggle autonomous brain",
        "P : pause / resume",
        "1–4 : simulation speed  (×0.5 / ×1 / ×2 / ×4)",
        "D : debug overlay",
        "F1 : this help",
        "F12 : screenshot",
        "Q : quit",
    ]
    ow, oh = 560, len(lines) * 18 + 30
    ox     = (SCREEN_W - ow) // 2
    oy     = (SCREEN_H - oh) // 2
    bg     = pygame.Surface((ow, oh), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 220))
    surface.blit(bg, (ox, oy))
    pygame.draw.rect(surface, (100, 180, 255), (ox, oy, ow, oh), 2, border_radius=8)
    for i, line in enumerate(lines):
        colour = (100, 200, 255) if i == 0 else (200, 200, 200)
        txt = font_sm.render(line, True, colour)
        surface.blit(txt, (ox + 15, oy + 12 + i * 18))


# ───────────────────────────────────────────────────────────────
# INTENT HISTORY STRIP (right edge)
# ───────────────────────────────────────────────────────────────
_intent_history: list = []
_MAX_HIST = 12

def update_intent_history(intent: str):
    global _intent_history
    if not _intent_history or _intent_history[-1] != intent:
        _intent_history.append(intent)
        if len(_intent_history) > _MAX_HIST:
            _intent_history.pop(0)


def draw_intent_history(surface, font_sm):
    if not _intent_history: return
    x0 = SCREEN_W - 170
    y0 = SCREEN_H // 2 - len(_intent_history) * 9
    panel = pygame.Surface((165, len(_intent_history) * 16 + 10), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 150))
    surface.blit(panel, (x0 - 5, y0 - 5))
    n = max(1, len(_intent_history) - 1)
    for i, intent in enumerate(_intent_history):
        col   = INTENT_COLOURS.get(intent, (200, 200, 200))
        alpha = int(80 + 175 * (i / n))
        col   = tuple(min(255, int(c * alpha / 255)) for c in col)
        surface.blit(font_sm.render(intent, True, col), (x0, y0 + i * 16))


# ───────────────────────────────────────────────────────────────
# SCREENSHOT
# ───────────────────────────────────────────────────────────────
def take_screenshot(screen):
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    fname = os.path.join(SCREENSHOT_DIR,
                         f"alan_{time.strftime('%Y%m%d_%H%M%S')}.png")
    pygame.image.save(screen, fname)
    return fname


# ───────────────────────────────────────────────────────────────
# INIT
# ───────────────────────────────────────────────────────────────
pygame.init()
screen  = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Stick Figure Sandbox — Hybrid RL + Cognitive Agent v2")
font_sm = pygame.font.SysFont("monospace", 13)
font_lg = pygame.font.SysFont("monospace", 16, bold=True)
clock   = pygame.time.Clock()

loaded_bg_image = None
if BG_IMAGE_PATH:
    try:
        raw_bg          = pygame.image.load(BG_IMAGE_PATH).convert()
        loaded_bg_image = pygame.transform.scale(raw_bg, (SCREEN_W, SCREEN_H))
        print("[Game] Background image loaded.")
    except Exception as e:
        print(f"[Game] No background image ({e}). Using gradient sky.")

env = Environment()

# ── Systems ─────────────────────────────────────────────────────
emotion_sys  = EmotionSystem()
fatigue_sys  = FatigueSystem()
memory       = MemorySystem()
inventory    = InventorySystem(food=3, seeds=5)
farm_system  = FarmSystem(plot_positions=[-2.5, -3.5, -4.5])
circadian    = CircadianRhythm(phase_offset=0.1)   # start in daytime
reward_norm  = RunningNorm()
advisor      = LLMAdvisor(call_interval=10.0)
notif_sys    = NotificationSystem()
reward_graph = RewardGraph()

# rl_model_path="path/to/model.zip" to load trained PPO
brain = BrainDecision(autonomous=True, rl_model_path=None)

# ── State ───────────────────────────────────────────────────────
jump_cooldown  = 0
walk_phase     = 0.0
last_reward    = 0.0
advisor_intent = "explore"
current_action = ACTION_IDLE
speed_idx      = 1      # index into SPEED_LEVELS
paused         = False
show_debug     = False
show_help      = False

# For notification triggers
_prev_food    = inventory.food
_prev_intent  = brain.current_intent

# ───────────────────────────────────────────────────────────────
# MAIN LOOP
# ───────────────────────────────────────────────────────────────
running = True
while running:
    clock.tick(FPS)
    DT = BASE_DT * SPEED_LEVELS[speed_idx]

    # ── Events ──────────────────────────────────────────────────
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            if event.key == pygame.K_p:
                paused = not paused
            if event.key == pygame.K_F1:
                show_help = not show_help
            if event.key == pygame.K_d:
                show_debug = not show_debug
            if event.key == pygame.K_F12:
                fname = take_screenshot(screen)
                notif_sys.add(f"📸 Saved {os.path.basename(fname)}",
                              (200, 200, 255), SCREEN_W // 2, 200)
            if event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                speed_idx = event.key - pygame.K_1
                notif_sys.add(f"Speed ×{SPEED_LEVELS[speed_idx]}",
                              (200, 200, 100), SCREEN_W // 2, 180)

            if event.key == pygame.K_a:
                brain.autonomous = not brain.autonomous
                mode = "AUTO" if brain.autonomous else "PLAYER"
                notif_sys.add(f"Mode: {mode}", (100, 255, 150),
                              SCREEN_W // 2, 200)
                print(f"[Brain] Autonomous: {brain.autonomous}")

            if not brain.autonomous:
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    if jump_cooldown <= 0 and env.jump():
                        jump_cooldown = 10
                if event.key == pygame.K_c:
                    env.stand() if env.is_crouching else env.crouch()
                if event.key == pygame.K_g:
                    env.release_object() if env.carried_body else env.grab_object()
                if event.key == pygame.K_t:
                    env.throw_or_kick()
                if event.key == pygame.K_f:
                    env.stop_push_pull() if env.push_pull_object else env.start_push_pull()
                if event.key == pygame.K_e:
                    env.stop_climb() if env.is_climbing else env.start_climb(env.near_ladder())
                if event.key == pygame.K_r:
                    env.roll()
                if event.key == pygame.K_s:
                    env.sit()
                if event.key == pygame.K_z:
                    env.toggle_sleep()
                if event.key == pygame.K_x:
                    if inventory.eat():
                        fatigue_sys.eat()
                        notif_sys.add("Nom! 🍎", (255, 180, 60),
                                      *phys_to_screen(*env.agent_body.position))
                    else:
                        notif_sys.add("No food!", (220, 80, 80),
                                      *phys_to_screen(*env.agent_body.position))

    if paused:
        # Still render but skip simulation
        # Draw a pause banner
        pbanner = font_lg.render("⏸  PAUSED  —  press P to resume", True, (255, 230, 100))
        screen.blit(pbanner, (SCREEN_W // 2 - pbanner.get_width() // 2, SCREEN_H // 2))
        pygame.display.flip()
        continue

    if jump_cooldown > 0:
        jump_cooldown -= 1

    # ── Farm & circadian update ──────────────────────────────────
    farm_system.update(DT)
    circadian.update(DT, is_sleeping=getattr(env, 'is_sleeping', False))

    # ── Detect farm-ready transition for notification ────────────
    ready_now = farm_system.ready_count()

    # ── Input / brain ────────────────────────────────────────────
    keys       = pygame.key.get_pressed()
    moving     = False
    current_action = ACTION_IDLE

    if brain.autonomous:
        state          = get_full_state(env, emotion_sys, fatigue_sys,
                                        inventory, circadian)
        advisor_intent = advisor.get_advice(state)
        action         = brain.decide(env, emotion_sys, fatigue_sys,
                                      memory, inventory, farm_system, circadian)
        brain.apply_action(env, action)
        current_action = action
        if action in (0, 1): moving = True
        update_intent_history(brain.current_intent)

        # Intent-change notification
        if brain.current_intent != _prev_intent:
            col = INTENT_COLOURS.get(brain.current_intent, (200, 200, 200))
            notif_sys.add(f"→ {brain.current_intent}",
                          col, *phys_to_screen(*env.agent_body.position))
            _prev_intent = brain.current_intent

    else:
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            env.start_sprint()
        else:
            env.stop_sprint()

        if env.is_climbing:
            vx, vy = 0, 0
            if keys[pygame.K_UP]:    vy =  3.0
            elif keys[pygame.K_DOWN]: vy = -3.0
            if keys[pygame.K_RIGHT]: vx =  2.0
            elif keys[pygame.K_LEFT]: vx = -2.0
            env.move_on_ladder(vx, vy)
        else:
            if keys[pygame.K_RIGHT]: env.walk_right(); moving = True
            if keys[pygame.K_LEFT]:  env.walk_left();  moving = True

    # Smooth walk phase
    if moving and env.is_grounded() and not env.is_crouching:
        walk_phase += 0.3 * fatigue_sys.movement_efficiency
    else:
        walk_phase *= 0.85   # smooth decay instead of hard reset

    # ── Physics & fatigue ────────────────────────────────────────
    env.update_physics(DT)
    fatigue_sys.update(env, DT, circadian)

    # ── Reward & emotions ────────────────────────────────────────
    state       = get_full_state(env, emotion_sys, fatigue_sys,
                                 inventory, circadian)
    raw_reward  = compute_reward(env, emotion_sys.state, fatigue_sys,
                                 inventory, brain, farm_system)
    raw_reward += LLMAdvisor.reward_bias(advisor_intent, state)
    last_reward = reward_norm.update(raw_reward)

    update_emotions(env, emotion_sys.state, fatigue_sys, raw_reward, circadian)
    emotion_sys.decay()

    memory.record(state, raw_reward,
                  action=current_action,
                  intent=brain.current_intent if brain.autonomous else "explore")
    brain.metrics.record(brain.current_intent if brain.autonomous else "explore",
                         raw_reward)

    reward_graph.push(raw_reward)
    notif_sys.update(DT)

    # ── Event notifications ──────────────────────────────────────
    if inventory.food > _prev_food:
        notif_sys.add(f"+{inventory.food - _prev_food} food! 🍎",
                      (255, 180, 60),
                      *phys_to_screen(*env.agent_body.position))
    _prev_food = inventory.food

    if ready_now > 0 and farm_system.ready_count() > ready_now - 1:
        notif_sys.add("🌾 Crops ready!", (220, 200, 50),
                      SCREEN_W // 2, SCREEN_H // 3)

    if fatigue_sys.is_exhausted:
        notif_sys.add("⚠ Exhausted!", (220, 80, 50),
                      *phys_to_screen(*env.agent_body.position))

    if circadian.is_night and circadian.sleep_drive > 0.7 and \
       not getattr(env, 'is_sleeping', False):
        notif_sys.add("💤 Getting sleepy...", (80, 120, 200),
                      SCREEN_W // 2, SCREEN_H // 3 + 20)

    # ── RENDER ───────────────────────────────────────────────────
    draw_world(screen, env, loaded_bg_image, circadian)
    draw_updraft(screen, env)
    draw_leaves(screen, -5, 2)
    draw_farm_plots(screen, farm_system, font_sm)

    for ladder in env.ladders:
        draw_ladder(screen, ladder)
    for p in env.platforms:
        draw_platform(screen, p[0], p[1], p[2])

    draw_bench(screen, env)
    draw_box(screen, env.box_body, env.BOX_SIZE[0], env.BOX_SIZE[1], C_BOX)
    draw_ball(screen, env.ball_body, env.BALL_RADIUS)

    draw_stick_figure(
        screen, env.agent_body,
        env.AGENT_WIDTH, env.agent_height,
        walk_phase, moving,
        env.is_crouching, env.is_sleeping, env.is_sitting, env.facing,
    )

    draw_thought_bubble(screen, env, brain, font_sm)
    draw_intent_badge(screen, env, brain, font_sm)

    # HUDs
    draw_mind_hud(screen, emotion_sys, fatigue_sys, memory,
                  inventory, brain, advisor_intent, font_sm,
                  last_reward, circadian)
    draw_inventory_hud(screen, inventory, farm_system, font_sm)
    draw_hud(screen, env, brain, advisor_intent, font_sm, font_lg,
             env.is_grounded(), speed_idx, paused, circadian)
    draw_intent_history(screen, font_sm)
    reward_graph.draw(screen, font_sm)
    notif_sys.draw(screen, font_sm)

    if show_debug:
        draw_debug(screen, env, brain, fatigue_sys, font_sm)

    if show_help:
        draw_help(screen, font_sm)

    # Mode badge (top right, no `surface` shadowing)
    if brain.autonomous:
        badge_col  = (100, 255, 150)
        badge_txt  = "🤖 AUTO  (A=manual  P=pause  1-4=speed  F1=help)"
        badge_surf = font_lg.render(badge_txt, True, badge_col)
        screen.blit(badge_surf, (SCREEN_W - badge_surf.get_width() - 10, 100))

        if fatigue_sys.fatigue > 0.75:
            warn = font_sm.render("⚠ HIGH FATIGUE", True, (220, 80, 80))
            screen.blit(warn, (SCREEN_W // 2 - warn.get_width() // 2, SCREEN_H - 30))
        if fatigue_sys.is_hungry and inventory.food == 0:
            warn = font_sm.render("⚠ STARVING — NO FOOD", True, (255, 100, 40))
            screen.blit(warn, (SCREEN_W // 2 - warn.get_width() // 2, SCREEN_H - 15))
    else:
        badge_surf = font_sm.render(
            "PLAYER  (A=auto  P=pause  X=eat  F1=help)", True, (200, 200, 100))
        screen.blit(badge_surf, (SCREEN_W - badge_surf.get_width() - 10, 100))

    # Night overlay (subtle full-screen tint)
    if circadian.is_night:
        night_surf = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        night_alpha = int(circadian.sleep_drive * 40)
        night_surf.fill((10, 10, 40, night_alpha))
        screen.blit(night_surf, (0, 0))

    pygame.display.flip()

pygame.quit()
sys.exit()
