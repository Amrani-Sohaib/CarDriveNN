"""
Track module -- Beautifully rendered circuits.
Training: 5 difficulty levels.
Testing: 3 additional unseen circuits for evaluating trained models.
"""
import math
import pygame

# -- Palette ---------------------------------------------------------------
WHITE       = (255, 255, 255)
BLACK       = (0, 0, 0)
ASPHALT     = (50, 50, 55)
ASPHALT_LT  = (62, 62, 68)
KERB_RED    = (200, 30, 30)
KERB_WHITE  = (240, 240, 240)
LINE_WHITE  = (200, 200, 200)
LINE_YELLOW = (240, 200, 50)
GRASS_1     = (40, 135, 40)
GRASS_2     = (35, 115, 35)
SAND        = (194, 178, 128)
FINISH_W    = (255, 255, 255)
FINISH_B    = (30, 30, 30)
CHECKPOINT_CLR = (255, 255, 0)


def catmull_rom(points, density=25):
    """Cyclic Catmull-Rom spline."""
    result = []
    n = len(points)
    for i in range(n):
        p0, p1 = points[(i - 1) % n], points[i]
        p2, p3 = points[(i + 1) % n], points[(i + 2) % n]
        for s in range(density):
            t = s / density
            t2, t3 = t * t, t * t * t
            x = .5 * ((2*p1[0]) + (-p0[0]+p2[0])*t +
                       (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 +
                       (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
            y = .5 * ((2*p1[1]) + (-p0[1]+p2[1])*t +
                       (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 +
                       (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
            result.append((x, y))
    return result


# -- Level definitions -----------------------------------------------------

LEVEL_NAMES = [
    "Lv1  Discovery Oval",
    "Lv2  S-Curve",
    "Lv3  Chicane Circuit",
    "Lv4  Hairpin Monza",
    "Lv5  Infernal Circuit",
]

TEST_TRACK_NAMES = [
    "Test A  Figure Eight",
    "Test B  Riverside",
    "Test C  Maze Circuit",
]

def _level_data(level, cx, cy):
    """Return (control_points, track_width) for each level.
    Points are designed for a ~900x620 area (center ~450,310)."""
    if level == 1:
        return [
            (cx, cy - 200), (cx + 155, cy - 185), (cx + 265, cy - 100),
            (cx + 280, cy + 15), (cx + 250, cy + 125), (cx + 115, cy + 195),
            (cx - 25, cy + 210), (cx - 155, cy + 180), (cx - 265, cy + 95),
            (cx - 280, cy - 25), (cx - 240, cy - 140), (cx - 115, cy - 195),
        ], 64
    if level == 2:
        return [
            (cx, cy - 220), (cx + 165, cy - 200), (cx + 290, cy - 120),
            (cx + 310, cy + 5), (cx + 280, cy + 100), (cx + 175, cy + 165),
            (cx + 80, cy + 220), (cx - 45, cy + 240), (cx - 170, cy + 200),
            (cx - 290, cy + 120), (cx - 320, cy + 15), (cx - 285, cy - 120),
            (cx - 165, cy - 210),
        ], 56
    if level == 3:
        return [
            (cx - 45, cy - 235), (cx + 140, cy - 215), (cx + 295, cy - 150),
            (cx + 310, cy - 30), (cx + 220, cy + 45), (cx + 95, cy + 15),
            (cx + 25, cy + 100), (cx + 155, cy + 185), (cx + 80, cy + 245),
            (cx - 80, cy + 240), (cx - 200, cy + 178), (cx - 310, cy + 100),
            (cx - 325, cy - 25), (cx - 280, cy - 130), (cx - 180, cy - 215),
        ], 48
    if level == 4:
        return [
            (cx - 30, cy - 255), (cx + 165, cy - 240), (cx + 340, cy - 155),
            (cx + 355, cy - 40), (cx + 270, cy + 45), (cx + 140, cy + 12),
            (cx + 85, cy + 100), (cx + 200, cy + 175), (cx + 325, cy + 240),
            (cx + 165, cy + 275), (cx, cy + 220), (cx - 125, cy + 275),
            (cx - 280, cy + 220), (cx - 355, cy + 115), (cx - 320, cy + 10),
            (cx - 240, cy - 65), (cx - 340, cy - 165), (cx - 280, cy - 240),
        ], 40
    # level 5
    return [
        (cx, cy - 260), (cx + 150, cy - 245), (cx + 310, cy - 195),
        (cx + 380, cy - 80), (cx + 335, cy + 20), (cx + 200, cy - 25),
        (cx + 120, cy + 65), (cx + 255, cy + 130), (cx + 355, cy + 205),
        (cx + 240, cy + 265), (cx + 80, cy + 220), (cx, cy + 275),
        (cx - 100, cy + 220), (cx - 240, cy + 265), (cx - 355, cy + 195),
        (cx - 305, cy + 95), (cx - 165, cy + 50), (cx - 280, cy - 35),
        (cx - 380, cy - 110), (cx - 320, cy - 215), (cx - 180, cy - 245),
    ], 35


def _test_track_data(track_id, cx, cy):
    """Return (control_points, track_width) for test circuits.
    These are completely different shapes never seen during training."""
    if track_id == 1:
        # Figure-eight with crossing zone
        return [
            (cx - 60, cy - 250), (cx + 160, cy - 240), (cx + 320, cy - 160),
            (cx + 370, cy - 40), (cx + 290, cy + 60), (cx + 120, cy + 40),
            (cx - 20, cy - 40), (cx - 160, cy + 50), (cx - 310, cy + 70),
            (cx - 380, cy - 10), (cx - 340, cy - 120), (cx - 220, cy - 200),
            (cx - 100, cy - 240), (cx + 40, cy - 200), (cx + 130, cy - 80),
            (cx + 50, cy + 30), (cx - 70, cy + 130), (cx - 210, cy + 200),
            (cx - 340, cy + 230), (cx - 380, cy + 150), (cx - 300, cy + 50),
            (cx - 160, cy - 50), (cx - 30, cy - 150), (cx - 120, cy - 210),
        ], 44
    if track_id == 2:
        # Riverside: long sweeping curves with tight hairpin
        return [
            (cx - 350, cy - 60), (cx - 280, cy - 200), (cx - 120, cy - 255),
            (cx + 60, cy - 240), (cx + 200, cy - 200), (cx + 310, cy - 110),
            (cx + 380, cy + 10), (cx + 360, cy + 120), (cx + 260, cy + 180),
            (cx + 120, cy + 150), (cx + 60, cy + 80), (cx + 100, cy + 20),
            (cx + 180, cy + 60), (cx + 200, cy + 160), (cx + 100, cy + 240),
            (cx - 60, cy + 260), (cx - 200, cy + 220), (cx - 320, cy + 140),
            (cx - 380, cy + 40),
        ], 42
    # track_id 3: Maze-like with many direction changes
    return [
        (cx - 320, cy - 220), (cx - 140, cy - 250), (cx + 60, cy - 210),
        (cx + 180, cy - 130), (cx + 120, cy - 30), (cx + 220, cy + 40),
        (cx + 360, cy - 20), (cx + 380, cy + 100), (cx + 300, cy + 180),
        (cx + 160, cy + 140), (cx + 60, cy + 210), (cx + 140, cy + 270),
        (cx, cy + 260), (cx - 140, cy + 210), (cx - 80, cy + 120),
        (cx - 200, cy + 60), (cx - 340, cy + 120), (cx - 380, cy + 20),
        (cx - 340, cy - 100), (cx - 220, cy - 160),
    ], 38


# -- Track class -----------------------------------------------------------
class Track:
    def __init__(self, width=900, height=620, level=1, test_track=0):
        self.width = width
        self.height = height
        self.is_test = test_track > 0
        self.test_id = test_track

        if self.is_test:
            self.level = 0
            self.level_name = TEST_TRACK_NAMES[min(test_track, len(TEST_TRACK_NAMES)) - 1]
        else:
            self.level = max(1, min(5, level))
            self.level_name = LEVEL_NAMES[self.level - 1]

        cx, cy = width // 2, height // 2
        if self.is_test:
            pts, self.track_width = _test_track_data(test_track, cx, cy)
        else:
            pts, self.track_width = _level_data(self.level, cx, cy)
        self.control_points = pts

        self.center = catmull_rom(pts, density=28)
        self.inner = []
        self.outer = []
        self._build_borders()

        self.checkpoints = []
        self._build_checkpoints(35)

        # start
        self.start_pos = self.center[0]
        dx = self.center[1][0] - self.center[0][0]
        dy = self.center[1][1] - self.center[0][1]
        self.start_angle = -math.degrees(math.atan2(dy, dx))

        # pre-render
        self.surface = None
        self._render()
        self.border_mask = None
        self._build_mask()

    # -- borders -----------------------------------------------------------
    def _build_borders(self):
        n = len(self.center)
        for i in range(n):
            pp = self.center[(i - 1) % n]
            pn = self.center[(i + 1) % n]
            dx, dy = pn[0] - pp[0], pn[1] - pp[1]
            L = math.hypot(dx, dy) or 1
            nx, ny = -dy / L, dx / L
            cx, cy = self.center[i]
            self.inner.append((cx + nx * self.track_width, cy + ny * self.track_width))
            self.outer.append((cx - nx * self.track_width, cy - ny * self.track_width))

    # -- checkpoints -------------------------------------------------------
    def _build_checkpoints(self, num):
        n = len(self.center)
        step = max(1, n // num)
        for i in range(num):
            idx = (i * step) % n
            self.checkpoints.append((self.inner[idx], self.outer[idx], idx))

    # -- rendering ---------------------------------------------------------
    def _render(self):
        s = pygame.Surface((self.width, self.height))

        # 1) grass background
        for row in range(0, self.height, 16):
            color = GRASS_1 if (row // 16) % 2 == 0 else GRASS_2
            pygame.draw.rect(s, color, (0, row, self.width, 16))

        n = len(self.inner)
        tw = self.track_width

        # 2) sand run-off
        for i in range(n):
            j = (i + 1) % n
            pp = self.center[i]
            pn = self.center[j]
            dx, dy = pn[0] - pp[0], pn[1] - pp[1]
            L = math.hypot(dx, dy) or 1
            nx, ny = -dy / L, dx / L
            extra = 14
            s_inner = (pp[0] + nx * (tw + extra), pp[1] + ny * (tw + extra))
            s_outer = (pp[0] - nx * (tw + extra), pp[1] - ny * (tw + extra))
            e_inner = (pn[0] + nx * (tw + extra), pn[1] + ny * (tw + extra))
            e_outer = (pn[0] - nx * (tw + extra), pn[1] - ny * (tw + extra))
            pygame.draw.polygon(s, SAND, [s_inner, e_inner, e_outer, s_outer])

        # 3) main asphalt
        for i in range(n):
            j = (i + 1) % n
            quad = [self.inner[i], self.inner[j], self.outer[j], self.outer[i]]
            pygame.draw.polygon(s, ASPHALT, quad)

        # 4) asphalt texture
        for i in range(n):
            j = (i + 1) % n
            if i % 4 == 0:
                mid_i = ((self.inner[i][0]+self.outer[i][0])/2,
                         (self.inner[i][1]+self.outer[i][1])/2)
                mid_j = ((self.inner[j][0]+self.outer[j][0])/2,
                         (self.inner[j][1]+self.outer[j][1])/2)
                pygame.draw.line(s, ASPHALT_LT, mid_i, mid_j, 1)

        # 5) kerbs
        kerb_w = 5
        for i in range(n):
            seg = i % 8
            c_k = KERB_RED if seg < 4 else KERB_WHITE
            j = (i + 1) % n

            ci, cj = self.center[i], self.center[j]
            ii, ij = self.inner[i], self.inner[j]
            dx_i, dy_i = ii[0] - ci[0], ii[1] - ci[1]
            li = math.hypot(dx_i, dy_i) or 1
            nxi, nyi = dx_i / li, dy_i / li
            ki1 = (ii[0] - nxi * kerb_w, ii[1] - nyi * kerb_w)
            ki2 = (ij[0] - nxi * kerb_w, ij[1] - nyi * kerb_w)
            pygame.draw.polygon(s, c_k, [ii, ij, ki2, ki1])

            oi, oj = self.outer[i], self.outer[j]
            dx_o, dy_o = oi[0] - ci[0], oi[1] - ci[1]
            lo = math.hypot(dx_o, dy_o) or 1
            nxo, nyo = dx_o / lo, dy_o / lo
            ko1 = (oi[0] - nxo * kerb_w, oi[1] - nyo * kerb_w)
            ko2 = (oj[0] - nxo * kerb_w, oj[1] - nyo * kerb_w)
            pygame.draw.polygon(s, c_k, [oi, oj, ko2, ko1])

        # 6) white edge lines
        pygame.draw.lines(s, LINE_WHITE, True, self.inner, 2)
        pygame.draw.lines(s, LINE_WHITE, True, self.outer, 2)

        # 7) dashed center line
        for i in range(0, n, 6):
            j = (i + 2) % n
            pygame.draw.line(s, (80, 80, 90), self.center[i], self.center[j], 1)

        # 8) start line (checkered)
        if self.checkpoints:
            cp = self.checkpoints[0]
            inner, outer = cp[0], cp[1]
            dx = outer[0] - inner[0]
            dy = outer[1] - inner[1]
            L = math.hypot(dx, dy)
            num_squares = 8
            for k in range(num_squares):
                t1 = k / num_squares
                t2 = (k + 1) / num_squares
                p1 = (inner[0] + dx * t1, inner[1] + dy * t1)
                p2 = (inner[0] + dx * t2, inner[1] + dy * t2)
                color = FINISH_W if k % 2 == 0 else FINISH_B
                perp_x, perp_y = -dy / L * 4, dx / L * 4
                quad = [(p1[0] + perp_x, p1[1] + perp_y),
                        (p2[0] + perp_x, p2[1] + perp_y),
                        (p2[0] - perp_x, p2[1] - perp_y),
                        (p1[0] - perp_x, p1[1] - perp_y)]
                pygame.draw.polygon(s, color, quad)

        self.surface = s

    # -- collision mask ----------------------------------------------------
    def _build_mask(self):
        ms = pygame.Surface((self.width, self.height))
        ms.fill(WHITE)
        n = len(self.inner)
        for i in range(n):
            j = (i + 1) % n
            quad = [self.inner[i], self.inner[j], self.outer[j], self.outer[i]]
            int_q = [(int(p[0]), int(p[1])) for p in quad]
            pygame.draw.polygon(ms, BLACK, int_q)
        self.border_mask = pygame.mask.from_threshold(ms, WHITE, (10, 10, 10))

    def is_off_track(self, x, y):
        ix, iy = int(x), int(y)
        if ix < 0 or ix >= self.width or iy < 0 or iy >= self.height:
            return True
        return self.border_mask.get_at((ix, iy))

    def draw(self, screen):
        screen.blit(self.surface, (0, 0))

    def draw_checkpoints(self, screen, alpha=30):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for cp in self.checkpoints:
            pygame.draw.line(overlay, (*CHECKPOINT_CLR, alpha), cp[0], cp[1], 1)
        screen.blit(overlay, (0, 0))
