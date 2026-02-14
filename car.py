"""
Car module -- Arcade physics + 5 radar sensors.
One car per episode.
"""
import math
import pygame

WHITE = (255, 255, 255)

# Colors
BODY_COLOR    = (0, 160, 255)
BEST_COLOR    = (255, 215, 0)
RADAR_COLOR   = (0, 255, 120)
RADAR_HIT     = (255, 60, 60)
WINDSHIELD    = (140, 200, 255)


class Car:
    """Racing car with simplified physics and 5 radars."""

    def __init__(self, track):
        self.track = track

        # state
        self.x, self.y = track.start_pos
        self.angle = track.start_angle
        self.speed = 0.0

        # physics
        self.max_speed   = 5.5
        self.accel       = 0.35
        self.brake_force = 0.45
        self.friction    = 0.05
        self.turn_speed  = 4.0

        # dimensions
        self.w, self.h = 22, 11

        # life
        self.alive = True
        self.ticks = 0
        self.dist  = 0.0
        self.fitness = 0.0
        self.stall = 0
        self.last_cp_tick = 0

        # checkpoints
        self.next_cp = 0
        self.cp_passed = 0
        self.laps = 0

        # radars
        self.radar_angles = [-90, -45, 0, 45, 90]
        self.radar_len    = 200
        self.radar_data   = [0.0] * 5
        self.radar_ends   = [(0, 0)] * 5

    # -- inputs for NN -----------------------------------------------------
    def get_inputs(self) -> list[float]:
        return self.radar_data.copy()

    # -- NN action ---------------------------------------------------------
    def apply_action(self, out: list[float]):
        if not self.alive:
            return
        if out[0] > 0.5:
            self.speed += self.accel
        if out[1] > 0.5:
            self.speed -= self.brake_force
        ratio = self.speed / self.max_speed if self.max_speed else 0
        if out[2] > 0.5:
            self.angle += self.turn_speed * ratio
        if out[3] > 0.5:
            self.angle -= self.turn_speed * ratio
        self.speed = max(0, min(self.speed, self.max_speed))
        self.speed *= (1 - self.friction)

    # -- update ------------------------------------------------------------
    def update(self):
        if not self.alive:
            return
        self.ticks += 1

        rad = math.radians(-self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed
        self.dist += self.speed

        # anti-stagnation
        if self.speed < 0.3:
            self.stall += 1
        else:
            self.stall = 0
        if self.stall > 80:
            self.alive = False; return

        # checkpoint timeout
        if self.ticks - self.last_cp_tick > 280:
            self.alive = False; return

        self._collision()
        if not self.alive:
            return
        self._radars()
        self._check_cp()
        self._calc_fitness()

    # -- collision ---------------------------------------------------------
    def _collision(self):
        rad = math.radians(-self.angle)
        c, s = math.cos(rad), math.sin(rad)
        hw, hh = self.w / 2, self.h / 2
        for sx, sy in [(1,1),(1,-1),(-1,-1),(-1,1)]:
            px = self.x + c * hw * sx - s * hh * sy
            py = self.y + s * hw * sx + c * hh * sy
            if self.track.is_off_track(px, py):
                self.alive = False; return

    # -- radars ------------------------------------------------------------
    def _radars(self):
        for i, ao in enumerate(self.radar_angles):
            a = math.radians(-(self.angle + ao))
            d = 0
            ex, ey = self.x, self.y
            while d < self.radar_len:
                d += 3
                ex = self.x + math.cos(a) * d
                ey = self.y + math.sin(a) * d
                if self.track.is_off_track(ex, ey):
                    break
            self.radar_data[i] = d / self.radar_len
            self.radar_ends[i] = (ex, ey)

    # -- checkpoints -------------------------------------------------------
    def _check_cp(self):
        if self.next_cp >= len(self.track.checkpoints):
            return
        cp = self.track.checkpoints[self.next_cp]
        if self._pt_seg_dist(self.x, self.y, cp[0], cp[1]) < 25:
            self.cp_passed += 1
            self.last_cp_tick = self.ticks
            self.next_cp = (self.next_cp + 1) % len(self.track.checkpoints)
            if self.next_cp == 0:
                self.laps += 1

    @staticmethod
    def _pt_seg_dist(px, py, a, b):
        ax, ay = a; bx, by = b
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab2 = abx*abx + aby*aby
        if ab2 == 0:
            return math.hypot(apx, apy)
        t = max(0, min(1, (apx*abx + apy*aby) / ab2))
        dx, dy = px - (ax + t*abx), py - (ay + t*aby)
        return math.hypot(dx, dy)

    def _calc_fitness(self):
        self.fitness = self.cp_passed * 100 + self.laps * 5000 + self.dist * 0.1

    # -- draw --------------------------------------------------------------
    def draw(self, screen, show_radars=True):
        if not self.alive:
            return
        rad = math.radians(-self.angle)
        c, s = math.cos(rad), math.sin(rad)
        hw, hh = self.w / 2, self.h / 2

        # radars
        if show_radars:
            for end in self.radar_ends:
                pygame.draw.line(screen, RADAR_COLOR, (self.x, self.y), end, 1)
                pygame.draw.circle(screen, RADAR_HIT,
                                   (int(end[0]), int(end[1])), 3)

        # body
        corners = [
            (self.x + c*hw - s*hh, self.y + s*hw + c*hh),
            (self.x - c*hw - s*hh, self.y - s*hw + c*hh),
            (self.x - c*hw + s*hh, self.y - s*hw - c*hh),
            (self.x + c*hw + s*hh, self.y + s*hw - c*hh),
        ]
        pygame.draw.polygon(screen, BODY_COLOR, corners)
        pygame.draw.polygon(screen, WHITE, corners, 1)

        # windshield
        wb = 0.25
        wb_corners = [
            (self.x + c*hw*wb - s*hh*0.7, self.y + s*hw*wb + c*hh*0.7),
            (self.x + c*hw*0.8 - s*hh*0.5, self.y + s*hw*0.8 + c*hh*0.5),
            (self.x + c*hw*0.8 + s*hh*0.5, self.y + s*hw*0.8 - c*hh*0.5),
            (self.x + c*hw*wb + s*hh*0.7, self.y + s*hw*wb - c*hh*0.7),
        ]
        pygame.draw.polygon(screen, WINDSHIELD, wb_corners)

        # red nose
        fx = self.x + c * (hw + 4)
        fy = self.y + s * (hw + 4)
        pygame.draw.circle(screen, (255, 50, 50), (int(fx), int(fy)), 3)
