"""
Car module -- Arcade physics with rich sensor suite.
====================================================
Inputs (15):
  12 radars (every 30 deg: -180 to +150)
  + normalized speed
  + angle to next checkpoint (normalized)
  + track curvature ahead

Outputs (2 continuous):
  steering  [-1, +1]  (left to right)
  throttle  [-1, +1]  (brake to full gas)
"""
import math
import pygame

WHITE = (255, 255, 255)

# Colors
BODY_COLOR    = (0, 160, 255)
RADAR_COLOR   = (0, 255, 120)
RADAR_HIT     = (255, 60, 60)
WINDSHIELD    = (140, 200, 255)


class Car:
    """Racing car with continuous controls and 15-input sensor suite."""

    # 12 radars evenly spread around the car
    RADAR_ANGLES = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]
    RADAR_RANGE  = 200
    NUM_RADARS   = 12
    NUM_INPUTS   = 15  # 12 radars + speed + angle_to_cp + curvature

    def __init__(self, track):
        self.track = track

        # state
        self.x, self.y = track.start_pos
        self.angle = track.start_angle
        self.speed = 0.0

        # physics
        self.max_speed    = 6.0
        self.accel_force  = 0.4
        self.brake_force  = 0.5
        self.friction     = 0.04
        self.turn_rate    = 4.5   # degrees per frame at max speed

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

        # rewards (per-step for PPO)
        self.step_reward = 0.0
        self.prev_dist_to_cp = self._dist_to_next_cp()
        self.prev_angle = self.angle
        self.total_spin = 0.0

        # radar data
        self.radar_data = [0.0] * self.NUM_RADARS
        self.radar_ends = [(0, 0)] * self.NUM_RADARS
        self._radars()  # initial radar cast

    # -- rich inputs for NN ------------------------------------------------
    def get_inputs(self) -> list:
        """Return 15 normalized inputs."""
        inputs = self.radar_data.copy()  # 12 floats [0, 1]
        inputs.append(self.speed / self.max_speed)  # normalized speed
        inputs.append(self._angle_to_next_cp())     # [-1, 1]
        inputs.append(self._curvature_ahead())      # [-1, 1]
        return inputs

    # -- continuous action -------------------------------------------------
    def apply_action(self, action):
        """
        action: [steering, throttle]  each in [-1, +1]
        steering: -1=full left, +1=full right
        throttle: -1=full brake, +1=full gas
        """
        if not self.alive:
            return
        steering = float(action[0])
        throttle = float(action[1])

        # throttle
        if throttle > 0:
            self.speed += self.accel_force * throttle
        else:
            self.speed += self.brake_force * throttle  # throttle is negative -> braking

        self.speed = max(0.0, min(self.speed, self.max_speed))

        # steering (proportional to speed -- can't turn much when slow)
        ratio = self.speed / self.max_speed if self.max_speed > 0 else 0
        self.angle += self.turn_rate * steering * max(0.05, ratio)

        # friction
        self.speed *= (1 - self.friction)

    # -- update ------------------------------------------------------------
    def update(self):
        if not self.alive:
            return
        self.ticks += 1

        rad = math.radians(-self.angle)
        dx = math.cos(rad) * self.speed
        dy = math.sin(rad) * self.speed
        self.x += dx
        self.y += dy
        self.dist += self.speed

        # stall detection
        if self.speed < 0.2:
            self.stall += 1
        else:
            self.stall = 0
        if self.stall > 100:
            self.alive = False
            self.step_reward = -10.0
            return

        # checkpoint timeout
        if self.ticks - self.last_cp_tick > 350:
            self.alive = False
            self.step_reward = -10.0
            return

        # collision
        self._collision()
        if not self.alive:
            self.step_reward = -10.0
            return

        # sensors
        self._radars()
        self._check_cp()
        self._calc_reward()
        self._calc_fitness()

    # -- reward shaping for PPO -------------------------------------------
    def _calc_reward(self):
        """Dense reward signal for PPO training."""
        reward = 0.0

        # reward for getting closer to next checkpoint
        curr_dist = self._dist_to_next_cp()
        progress = self.prev_dist_to_cp - curr_dist
        reward += progress * 0.05
        self.prev_dist_to_cp = curr_dist

        # reward for speed (encourages moving)
        reward += (self.speed / self.max_speed) * 0.1

        # penalty for being too close to walls
        min_radar = min(self.radar_data)
        if min_radar < 0.15:
            reward -= 0.2 * (0.15 - min_radar) / 0.15

        # penalty for excessive spinning / constant turning
        angle_delta = abs(self.angle - self.prev_angle)
        self.total_spin += angle_delta
        self.prev_angle = self.angle
        if angle_delta > 3.0:
            reward -= 0.15 * (angle_delta / self.turn_rate)
        # penalty for accumulated spin (going in circles)
        if self.ticks > 0 and self.ticks % 60 == 0:
            avg_spin = self.total_spin / self.ticks
            if avg_spin > 2.5:
                reward -= 0.3

        # penalty for very low speed
        if self.speed < 0.5:
            reward -= 0.05

        self.step_reward = reward

    def _dist_to_next_cp(self):
        if self.next_cp >= len(self.track.checkpoints):
            return 0
        cp = self.track.checkpoints[self.next_cp]
        mid = ((cp[0][0] + cp[1][0]) / 2, (cp[0][1] + cp[1][1]) / 2)
        return math.hypot(self.x - mid[0], self.y - mid[1])

    def _angle_to_next_cp(self):
        """Angle to next checkpoint, normalized to [-1, 1]."""
        if self.next_cp >= len(self.track.checkpoints):
            return 0.0
        cp = self.track.checkpoints[self.next_cp]
        mid = ((cp[0][0] + cp[1][0]) / 2, (cp[0][1] + cp[1][1]) / 2)
        dx = mid[0] - self.x
        dy = mid[1] - self.y
        target_angle = math.degrees(math.atan2(-dy, dx))
        diff = (target_angle - self.angle + 180) % 360 - 180
        return max(-1.0, min(1.0, diff / 180.0))

    def _curvature_ahead(self):
        """Estimate curvature from upcoming checkpoints, normalized to [-1, 1]."""
        cps = self.track.checkpoints
        n = len(cps)
        if n < 3:
            return 0.0
        i0 = self.next_cp % n
        i1 = (self.next_cp + 1) % n
        i2 = (self.next_cp + 2) % n

        def mid(cp):
            return ((cp[0][0] + cp[1][0]) / 2, (cp[0][1] + cp[1][1]) / 2)

        p0, p1, p2 = mid(cps[i0]), mid(cps[i1]), mid(cps[i2])

        # angle change between segments
        a1 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        a2 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        return max(-1.0, min(1.0, da / math.pi))

    # -- collision ---------------------------------------------------------
    def _collision(self):
        rad = math.radians(-self.angle)
        c, s = math.cos(rad), math.sin(rad)
        hw, hh = self.w / 2, self.h / 2
        for sx, sy in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:
            px = self.x + c * hw * sx - s * hh * sy
            py = self.y + s * hw * sx + c * hh * sy
            if self.track.is_off_track(px, py):
                self.alive = False
                return

    # -- radars ------------------------------------------------------------
    def _radars(self):
        for i, ao in enumerate(self.RADAR_ANGLES):
            a = math.radians(-(self.angle + ao))
            d = 0
            ex, ey = self.x, self.y
            while d < self.RADAR_RANGE:
                d += 3
                ex = self.x + math.cos(a) * d
                ey = self.y + math.sin(a) * d
                if self.track.is_off_track(ex, ey):
                    break
            self.radar_data[i] = d / self.RADAR_RANGE
            self.radar_ends[i] = (ex, ey)

    # -- checkpoints -------------------------------------------------------
    def _check_cp(self):
        if self.next_cp >= len(self.track.checkpoints):
            return
        cp = self.track.checkpoints[self.next_cp]
        if self._pt_seg_dist(self.x, self.y, cp[0], cp[1]) < 25:
            self.cp_passed += 1
            self.last_cp_tick = self.ticks
            self.step_reward += 5.0  # big reward for checkpoint
            self.next_cp = (self.next_cp + 1) % len(self.track.checkpoints)
            if self.next_cp == 0:
                self.laps += 1
                self.step_reward += 50.0  # massive reward for lap
            self.prev_dist_to_cp = self._dist_to_next_cp()

    @staticmethod
    def _pt_seg_dist(px, py, a, b):
        ax, ay = a; bx, by = b
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab2 = abx * abx + aby * aby
        if ab2 == 0:
            return math.hypot(apx, apy)
        t = max(0, min(1, (apx * abx + apy * aby) / ab2))
        dx, dy = px - (ax + t * abx), py - (ay + t * aby)
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
                pygame.draw.circle(screen, RADAR_HIT, (int(end[0]), int(end[1])), 3)

        # body
        corners = [
            (self.x + c * hw - s * hh, self.y + s * hw + c * hh),
            (self.x - c * hw - s * hh, self.y - s * hw + c * hh),
            (self.x - c * hw + s * hh, self.y - s * hw - c * hh),
            (self.x + c * hw + s * hh, self.y + s * hw - c * hh),
        ]
        pygame.draw.polygon(screen, BODY_COLOR, corners)
        pygame.draw.polygon(screen, WHITE, corners, 1)

        # windshield
        wb = 0.25
        wb_corners = [
            (self.x + c * hw * wb - s * hh * 0.7, self.y + s * hw * wb + c * hh * 0.7),
            (self.x + c * hw * 0.8 - s * hh * 0.5, self.y + s * hw * 0.8 + c * hh * 0.5),
            (self.x + c * hw * 0.8 + s * hh * 0.5, self.y + s * hw * 0.8 - c * hh * 0.5),
            (self.x + c * hw * wb + s * hh * 0.7, self.y + s * hw * wb - c * hh * 0.7),
        ]
        pygame.draw.polygon(screen, WINDSHIELD, wb_corners)

        # red nose
        fx = self.x + c * (hw + 4)
        fy = self.y + s * (hw + 4)
        pygame.draw.circle(screen, (255, 50, 50), (int(fx), int(fy)), 3)
