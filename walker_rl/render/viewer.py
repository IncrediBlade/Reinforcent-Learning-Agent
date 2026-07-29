"""Pygame viewer for live simulation and for recorded-episode playback.

It consumes plain dictionaries (see `StickmanEnv.render_state`), so it has no
dependency on the physics classes and can render a replay file just as easily
as a running environment.
"""

import math
import os

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

BG_TOP = (24, 26, 34)
BG_BOTTOM = (12, 13, 18)
GROUND = (46, 50, 62)
GROUND_LINE = (86, 94, 112)
GRID = (34, 37, 46)
TEXT = (214, 219, 230)
DIM = (128, 136, 152)
ACCENT = (92, 200, 160)
WARN = (232, 128, 112)
GOAL = (240, 196, 92)

PART_COLORS = {
    "torso": (108, 168, 232),
    "thigh_l": (86, 200, 168), "shin_l": (72, 178, 150), "foot_l": (58, 152, 128),
    "thigh_r": (216, 140, 108), "shin_r": (196, 120, 92), "foot_r": (172, 100, 76),
    "arm_l": (150, 190, 236), "arm_r": (198, 156, 132),
    "ground": GROUND, "platform": (168, 140, 72),
}
DEFAULT_PART = (150, 156, 172)


def _lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class Camera:
    def __init__(self, width, height, pixels_per_meter=170.0):
        self.width = width
        self.height = height
        self.ppm = pixels_per_meter
        self.x = 0.0
        self.y = 0.85
        self.ground_frac = 0.78   # where y=0 sits on screen

    def follow(self, x, y=None, smooth=0.12):
        self.x += (x - self.x) * smooth
        if y is not None:
            self.y += (y - self.y) * smooth * 0.5

    def to_screen(self, wx, wy):
        sx = (wx - self.x) * self.ppm + self.width * 0.5
        sy = self.height * self.ground_frac - wy * self.ppm
        return int(sx), int(sy)

    def scale(self, meters):
        return max(1, int(meters * self.ppm))


class Viewer:
    def __init__(self, width=1280, height=720, title="Stickman RL", fps=40,
                 show_hud=True):
        pygame.init()
        pygame.display.set_caption(title)
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.camera = Camera(width, height)
        self.font = pygame.font.SysFont("consolas,couriernew,monospace", 15)
        self.font_small = pygame.font.SysFont("consolas,couriernew,monospace", 13)
        self.font_big = pygame.font.SysFont("consolas,couriernew,monospace", 22, bold=True)
        self.fps = fps
        self.speed = 1.0
        self.paused = False
        self.show_hud = show_hud
        self.show_contacts = True
        self.show_trail = True
        self.trail = []
        self.closed = False
        self._bg = self._make_background()

    # -- infrastructure ----------------------------------------------------
    def _make_background(self):
        surf = pygame.Surface((self.width, self.height))
        for y in range(self.height):
            t = y / max(1, self.height - 1)
            pygame.draw.line(surf, _lerp_color(BG_TOP, BG_BOTTOM, t),
                             (0, y), (self.width, y))
        return surf

    def poll(self):
        """Handle input. Returns a command string or None."""
        cmd = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
                cmd = "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.closed = True
                    cmd = "quit"
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    cmd = "reset"
                elif event.key == pygame.K_n:
                    cmd = "step"
                elif event.key == pygame.K_c:
                    self.show_contacts = not self.show_contacts
                elif event.key == pygame.K_t:
                    self.show_trail = not self.show_trail
                    self.trail = []
                elif event.key == pygame.K_h:
                    self.show_hud = not self.show_hud
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.speed = min(8.0, self.speed * 2.0)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.speed = max(0.125, self.speed / 2.0)
                elif event.key == pygame.K_LEFTBRACKET:
                    self.camera.ppm = max(40.0, self.camera.ppm / 1.2)
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.camera.ppm = min(600.0, self.camera.ppm * 1.2)
        return cmd

    def tick(self):
        # Speed-up is achieved by advancing the simulation several times per
        # frame (the caller does that via `steps_per_frame`), so the frame rate
        # stays put. Slow motion is the opposite: one step per frame, fewer
        # frames per second.
        rate = self.fps if self.speed >= 1.0 else max(1.0, self.fps * self.speed)
        self.clock.tick(rate)

    def steps_per_frame(self):
        return max(1, int(self.speed)) if self.speed >= 1.0 else 1

    def close(self):
        pygame.quit()
        self.closed = True

    # -- world drawing -----------------------------------------------------
    def _draw_ground(self, cam):
        gy = cam.to_screen(0.0, 0.0)[1]
        pygame.draw.rect(self.screen, GROUND, (0, gy, self.width, self.height - gy))
        pygame.draw.line(self.screen, GROUND_LINE, (0, gy), (self.width, gy), 2)

        # Metre markers so motion is legible even when the camera follows.
        span = self.width / cam.ppm
        x0 = math.floor(cam.x - span * 0.6)
        x1 = math.ceil(cam.x + span * 0.6)
        for xi in range(x0, x1 + 1):
            sx, _ = cam.to_screen(xi, 0.0)
            major = xi % 5 == 0
            pygame.draw.line(self.screen, GRID if not major else GROUND_LINE,
                             (sx, gy), (sx, gy + (18 if major else 9)), 1)
            if major:
                label = self.font_small.render("%d" % xi, True, DIM)
                self.screen.blit(label, (sx - label.get_width() // 2, gy + 20))

    def _draw_goal(self, cam, target_x, goal_half, platform_half, platform_height):
        # The physical platform.
        if platform_half > 0:
            x0, y0 = cam.to_screen(target_x - platform_half, platform_height)
            x1, y1 = cam.to_screen(target_x + platform_half, 0.0)
            pygame.draw.rect(self.screen, PART_COLORS["platform"],
                             (x0, y0, x1 - x0, y1 - y0))
        # The square goal zone the torso has to hold.
        side = 2.0 * goal_half
        gx0, gy0 = cam.to_screen(target_x - goal_half, platform_height + side)
        gx1, gy1 = cam.to_screen(target_x + goal_half, platform_height)
        rect = pygame.Rect(gx0, gy0, gx1 - gx0, gy1 - gy0)
        overlay = pygame.Surface((max(1, rect.width), max(1, rect.height)),
                                 pygame.SRCALPHA)
        overlay.fill((GOAL[0], GOAL[1], GOAL[2], 38))
        self.screen.blit(overlay, rect.topleft)
        pygame.draw.rect(self.screen, GOAL, rect, 2)
        flag = self.font_small.render("TARGET", True, GOAL)
        self.screen.blit(flag, (rect.centerx - flag.get_width() // 2, rect.top - 18))

    def _draw_shapes(self, cam, shapes):
        for shape in shapes:
            name = shape.get("name", "")
            if name in ("ground", "platform"):
                continue
            color = PART_COLORS.get(name, DEFAULT_PART)
            if shape["kind"] == "poly":
                pts = [cam.to_screen(px, py) for px, py in shape["pts"]]
                pygame.draw.polygon(self.screen, color, pts)
                pygame.draw.polygon(self.screen, _lerp_color(color, (0, 0, 0), 0.45),
                                    pts, 2)
            else:
                center = cam.to_screen(shape["x"], shape["y"])
                r = cam.scale(shape["r"])
                pygame.draw.circle(self.screen, color, center, r)
                pygame.draw.circle(self.screen, _lerp_color(color, (0, 0, 0), 0.45),
                                   center, r, 2)
                # A small eye so the facing direction is readable.
                a = shape.get("angle", 0.0)
                ex = shape["x"] + 0.045 * math.cos(a) + 0.035 * math.cos(a + 1.57)
                ey = shape["y"] + 0.045 * math.sin(a) + 0.035 * math.sin(a + 1.57)
                pygame.draw.circle(self.screen, (18, 20, 26),
                                   cam.to_screen(ex, ey), max(2, r // 6))

    def _draw_direction_arrow(self, cam, torso, target_x):
        tx, ty, _ = torso
        dx = target_x - tx
        if abs(dx) < 1e-6:
            return
        sign = 1.0 if dx > 0 else -1.0
        start = cam.to_screen(tx, ty + 0.55)
        end = cam.to_screen(tx + sign * min(0.8, abs(dx)), ty + 0.55)
        pygame.draw.line(self.screen, GOAL, start, end, 3)
        head = 9
        pygame.draw.polygon(self.screen, GOAL, [
            end,
            (end[0] - sign * head, end[1] - head // 2),
            (end[0] - sign * head, end[1] + head // 2),
        ])

    # -- HUD ---------------------------------------------------------------
    def _text(self, s, x, y, color=TEXT, font=None):
        surf = (font or self.font).render(s, True, color)
        self.screen.blit(surf, (x, y))
        return surf.get_height()

    def _draw_hud(self, state, extra=None):
        pad = 14
        panel = pygame.Surface((330, 200), pygame.SRCALPHA)
        panel.fill((10, 12, 16, 170))
        self.screen.blit(panel, (pad, pad))

        y = pad + 8
        tx = state["torso"][0]
        dist = abs(state["target_x"] - tx)
        self._text("return  %8.1f" % state.get("return", 0.0), pad + 12, y,
                   ACCENT, self.font_big)
        y += 28
        self._text("step    %8d   t=%5.1fs" % (state.get("steps", 0),
                                               state.get("time", 0.0)),
                   pad + 12, y); y += 19
        self._text("x       %8.2f m" % tx, pad + 12, y); y += 19
        self._text("target  %8.2f m" % state["target_x"], pad + 12, y); y += 19
        self._text("distance%8.2f m" % dist, pad + 12, y,
                   ACCENT if dist < 0.5 else TEXT); y += 19
        self._text("hold    %8.2f s" % state.get("hold", 0.0), pad + 12, y); y += 19
        if extra:
            for line in extra:
                self._text(line, pad + 12, y, DIM); y += 17

        self._draw_reward_bars(state.get("components", {}))
        self._draw_action_bars(state.get("action"))

        hint = ("space pause   n step   r reset   +/- speed   [ ] zoom   "
                "c contacts   t trail   h hud   esc quit")
        surf = self.font_small.render(hint, True, DIM)
        self.screen.blit(surf, (pad, self.height - surf.get_height() - 8))

    def _draw_reward_bars(self, components):
        if not components:
            return
        x0 = self.width - 260
        y0 = 14
        panel = pygame.Surface((246, 22 + 16 * len(components)), pygame.SRCALPHA)
        panel.fill((10, 12, 16, 170))
        self.screen.blit(panel, (x0 - 10, y0 - 6))
        self._text("reward this step", x0, y0 - 2, DIM, self.font_small)
        y = y0 + 16
        scale = max(1e-6, max(abs(v) for v in components.values()))
        for name, value in components.items():
            self._text("%-8s" % name[:8], x0, y, DIM, self.font_small)
            bar_x = x0 + 74
            width = int(70 * value / scale)
            color = ACCENT if value >= 0 else WARN
            if width >= 0:
                pygame.draw.rect(self.screen, color, (bar_x, y + 4, width, 7))
            else:
                pygame.draw.rect(self.screen, color, (bar_x + width, y + 4, -width, 7))
            pygame.draw.line(self.screen, DIM, (bar_x, y + 2), (bar_x, y + 13), 1)
            self._text("%+.3f" % value, bar_x + 78, y, DIM, self.font_small)
            y += 16

    def _draw_action_bars(self, action):
        if action is None:
            return
        from ..env.stickman import JOINT_ORDER
        x0 = 16
        y0 = self.height - 44 - 15 * len(JOINT_ORDER)
        panel = pygame.Surface((240, 24 + 15 * len(JOINT_ORDER)), pygame.SRCALPHA)
        panel.fill((10, 12, 16, 170))
        self.screen.blit(panel, (x0 - 8, y0 - 6))
        self._text("motor command", x0, y0 - 2, DIM, self.font_small)
        y = y0 + 16
        for name, a in zip(JOINT_ORDER, np.asarray(action).ravel()):
            self._text("%-10s" % name, x0, y, DIM, self.font_small)
            bar_x = x0 + 96
            width = int(55 * float(a))
            color = ACCENT if a >= 0 else WARN
            if width >= 0:
                pygame.draw.rect(self.screen, color, (bar_x, y + 4, width, 7))
            else:
                pygame.draw.rect(self.screen, color, (bar_x + width, y + 4, -width, 7))
            pygame.draw.line(self.screen, DIM, (bar_x, y + 2), (bar_x, y + 13), 1)
            y += 15

    # -- top level ---------------------------------------------------------
    def draw(self, state, extra=None):
        cam = self.camera
        cam.follow(state["torso"][0])
        self.screen.blit(self._bg, (0, 0))
        self._draw_ground(cam)
        self._draw_goal(cam, state["target_x"], state.get("goal_half", 0.3),
                        state.get("platform_half", 0.0),
                        state.get("platform_height", 0.0))

        if self.show_trail:
            self.trail.append((state["torso"][0], state["torso"][1]))
            if len(self.trail) > 400:
                self.trail.pop(0)
            if len(self.trail) > 2:
                pts = [cam.to_screen(px, py) for px, py in self.trail]
                pygame.draw.lines(self.screen, (60, 76, 96), False, pts, 2)

        self._draw_shapes(cam, state["shapes"])
        self._draw_direction_arrow(cam, state["torso"], state["target_x"])

        if self.show_contacts:
            for cx, cy in state.get("contacts", ()):
                pygame.draw.circle(self.screen, WARN, cam.to_screen(cx, cy), 4)

        if self.show_hud:
            self._draw_hud(state, extra)
        pygame.display.flip()

    def reset_trail(self):
        self.trail = []


# ---------------------------------------------------------------------------
# Replay support
# ---------------------------------------------------------------------------
def replay_state(episode, frame):
    """Rebuild a `render_state`-shaped dict from a recorded episode."""
    poses = episode["poses"][frame]
    geometry = episode["geometry"]
    meta = episode["meta"]
    shapes = []
    for g in geometry:
        bi = g["body"]
        px, py, angle = poses[3 * bi], poses[3 * bi + 1], poses[3 * bi + 2]
        ca, sa = math.cos(angle), math.sin(angle)
        if g["kind"] == "circle":
            cx = px + ca * g["cx"] - sa * g["cy"]
            cy = py + sa * g["cx"] + ca * g["cy"]
            shapes.append({"kind": "circle", "name": g["name"], "x": cx, "y": cy,
                           "r": g["r"], "angle": angle})
        else:
            pts = []
            for vx, vy in g["verts"]:
                pts.append((px + ca * vx - sa * vy, py + sa * vx + ca * vy))
            shapes.append({"kind": "poly", "name": g["name"], "pts": pts})

    torso_pose = None
    for g in geometry:
        if g["name"] == "torso":
            bi = g["body"]
            torso_pose = (float(poses[3 * bi]), float(poses[3 * bi + 1]),
                          float(poses[3 * bi + 2]))
            break

    names = episode["component_names"]
    comps = {n: float(v) for n, v in zip(names, episode["components"][frame])}
    return {
        "shapes": shapes,
        "contacts": [],
        "target_x": meta["target_x"],
        "goal_half": meta.get("goal_half", 0.3),
        "platform_half": meta.get("platform_half", 0.0),
        "platform_height": meta.get("platform_height", 0.0),
        "torso": torso_pose or (0.0, 1.0, 0.0),
        "components": comps,
        "return": float(np.sum(episode["rewards"][:frame + 1])),
        "steps": frame + 1,
        "hold": 0.0,
        "action": episode["actions"][frame],
        "time": (frame + 1) / meta.get("control_hz", 40.0),
    }
