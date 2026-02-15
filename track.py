"""
Track module -- Robust procedural circuit generation.
Training: 5 difficulty levels.
Testing: 3 additional unseen circuits for evaluating trained models.

Border generation uses:
  - Half-width offset (center ± hw) instead of full-width
  - Smoothed normals via local derivative window
  - Curvature-adaptive width clamping to prevent auto-intersections
  - Minimum radius check: hw is reduced when curvature > 1/hw
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


# --------------------------------------------------------------------------
#  Catmull-Rom spline with derivative output
# --------------------------------------------------------------------------
def catmull_rom(points, density=25):
    """Cyclic Catmull-Rom spline.  Returns list of (x, y) points."""
    result = []
    n = len(points)
    for i in range(n):
        p0, p1 = points[(i - 1) % n], points[i]
        p2, p3 = points[(i + 1) % n], points[(i + 2) % n]
        for s in range(density):
            t = s / density
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
                        (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                        (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
                        (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                        (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            result.append((x, y))
    return result


# --------------------------------------------------------------------------
#  Geometry helpers
# --------------------------------------------------------------------------
def _smooth_normals(center, window=3):
    """Compute smoothed unit normals for a closed polyline.

    Uses a symmetric window of *window* points on each side to compute
    the local tangent, then rotates 90° CCW to get the left-pointing normal.
    """
    n = len(center)
    normals = []
    for i in range(n):
        dx, dy = 0.0, 0.0
        for k in range(1, window + 1):
            pp = center[(i - k) % n]
            pn = center[(i + k) % n]
            dx += pn[0] - pp[0]
            dy += pn[1] - pp[1]
        L = math.hypot(dx, dy) or 1.0
        # normal = 90° CCW from tangent
        normals.append((-dy / L, dx / L))
    return normals


def _curvature_at(center, i):
    """Approximate curvature at point *i* using the circumscribed circle
    through i-1, i, i+1.  Returns 1/R (higher = tighter turn)."""
    n = len(center)
    ax, ay = center[(i - 1) % n]
    bx, by = center[i]
    cx, cy = center[(i + 1) % n]

    # signed area of triangle * 2
    area2 = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    a = math.hypot(bx - ax, by - ay)
    b = math.hypot(cx - bx, cy - by)
    c = math.hypot(ax - cx, ay - cy)
    denom = a * b * c
    if denom < 1e-9:
        return 0.0
    return abs(area2) / denom * 2.0   # = 1/R


def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


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
    track_width is the FULL road width (half-width = track_width / 2).
    Points are designed for a ~900x620 area (center ~450, 310)."""
    if level == 1:
        # Wide oval -- very easy
        return [
            (cx, cy - 200), (cx + 155, cy - 185), (cx + 265, cy - 100),
            (cx + 280, cy + 15), (cx + 250, cy + 125), (cx + 115, cy + 195),
            (cx - 25, cy + 210), (cx - 155, cy + 180), (cx - 265, cy + 95),
            (cx - 280, cy - 25), (cx - 240, cy - 140), (cx - 115, cy - 195),
        ], 64

    if level == 2:
        # S-Curve
        return [
            (cx, cy - 220), (cx + 165, cy - 200), (cx + 290, cy - 120),
            (cx + 310, cy + 5), (cx + 280, cy + 100), (cx + 175, cy + 165),
            (cx + 80, cy + 220), (cx - 45, cy + 240), (cx - 170, cy + 200),
            (cx - 290, cy + 120), (cx - 320, cy + 15), (cx - 285, cy - 120),
            (cx - 165, cy - 210),
        ], 56

    if level == 3:
        # Chicane -- wider curves than before to avoid crossing
        return [
            (cx - 45, cy - 240), (cx + 160, cy - 220), (cx + 310, cy - 140),
            (cx + 340, cy - 10), (cx + 280, cy + 90),
            (cx + 140, cy + 50), (cx + 10, cy + 140),
            (cx + 140, cy + 230), (cx + 50, cy + 265),
            (cx - 100, cy + 245), (cx - 220, cy + 175),
            (cx - 330, cy + 90), (cx - 340, cy - 30),
            (cx - 280, cy - 135), (cx - 175, cy - 220),
        ], 44

    if level == 4:
        # Hairpin -- generous spacing between switchbacks
        return [
            (cx - 20, cy - 260), (cx + 180, cy - 245), (cx + 340, cy - 150),
            (cx + 365, cy - 20), (cx + 290, cy + 80),
            (cx + 140, cy + 55), (cx + 50, cy + 150),
            (cx + 200, cy + 225), (cx + 330, cy + 260),
            (cx + 150, cy + 280), (cx - 10, cy + 210),
            (cx - 150, cy + 280), (cx - 300, cy + 220),
            (cx - 370, cy + 110), (cx - 335, cy + 5),
            (cx - 275, cy - 75), (cx - 350, cy - 170),
            (cx - 280, cy - 245),
        ], 36

    # level 5 -- Infernal, but with enough spacing
    return [
        (cx + 10, cy - 265), (cx + 170, cy - 250), (cx + 330, cy - 190),
        (cx + 395, cy - 60), (cx + 360, cy + 50), (cx + 230, cy + 15),
        (cx + 120, cy + 110), (cx + 270, cy + 185), (cx + 365, cy + 250),
        (cx + 210, cy + 275), (cx + 70, cy + 215), (cx - 10, cy + 280),
        (cx - 120, cy + 215), (cx - 260, cy + 270), (cx - 370, cy + 190),
        (cx - 325, cy + 95), (cx - 175, cy + 65), (cx - 300, cy - 30),
        (cx - 395, cy - 110), (cx - 325, cy - 220), (cx - 170, cy - 250),
    ], 32


def _test_track_data(track_id, cx, cy):
    """Return (control_points, track_width) for test circuits."""
    if track_id == 1:
        # Figure-eight
        return [
            (cx - 60, cy - 250), (cx + 160, cy - 240), (cx + 320, cy - 160),
            (cx + 370, cy - 40), (cx + 290, cy + 60), (cx + 120, cy + 40),
            (cx - 20, cy - 40), (cx - 160, cy + 50), (cx - 310, cy + 70),
            (cx - 380, cy - 10), (cx - 340, cy - 120), (cx - 220, cy - 200),
            (cx - 100, cy - 240), (cx + 40, cy - 200), (cx + 130, cy - 80),
            (cx + 50, cy + 30), (cx - 70, cy + 130), (cx - 210, cy + 200),
            (cx - 340, cy + 230), (cx - 380, cy + 150), (cx - 300, cy + 50),
            (cx - 160, cy - 50), (cx - 30, cy - 150), (cx - 120, cy - 210),
        ], 40

    if track_id == 2:
        # Riverside
        return [
            (cx - 350, cy - 60), (cx - 280, cy - 200), (cx - 120, cy - 255),
            (cx + 60, cy - 240), (cx + 200, cy - 200), (cx + 310, cy - 110),
            (cx + 380, cy + 10), (cx + 360, cy + 120), (cx + 260, cy + 180),
            (cx + 120, cy + 150), (cx + 60, cy + 80), (cx + 100, cy + 20),
            (cx + 180, cy + 60), (cx + 200, cy + 160), (cx + 100, cy + 240),
            (cx - 60, cy + 260), (cx - 200, cy + 220), (cx - 320, cy + 140),
            (cx - 380, cy + 40),
        ], 40

    # track_id 3: Maze
    return [
        (cx - 320, cy - 220), (cx - 140, cy - 250), (cx + 60, cy - 210),
        (cx + 180, cy - 130), (cx + 120, cy - 30), (cx + 220, cy + 40),
        (cx + 360, cy - 20), (cx + 380, cy + 100), (cx + 300, cy + 180),
        (cx + 160, cy + 140), (cx + 60, cy + 210), (cx + 140, cy + 270),
        (cx, cy + 260), (cx - 140, cy + 210), (cx - 80, cy + 120),
        (cx - 200, cy + 60), (cx - 340, cy + 120), (cx - 380, cy + 20),
        (cx - 340, cy - 100), (cx - 220, cy - 160),
    ], 36


# ==========================================================================
#  Track class
# ==========================================================================
class Track:
    def __init__(self, width=900, height=620, level=1, test_track=0):
        self.width = width
        self.height = height
        self.is_test = test_track > 0
        self.test_id = test_track

        if self.is_test:
            self.level = 0
            self.level_name = TEST_TRACK_NAMES[
                min(test_track, len(TEST_TRACK_NAMES)) - 1
            ]
        else:
            self.level = max(1, min(5, level))
            self.level_name = LEVEL_NAMES[self.level - 1]

        cx, cy = width // 2, height // 2
        if self.is_test:
            pts, self.track_width = _test_track_data(test_track, cx, cy)
        else:
            pts, self.track_width = _level_data(self.level, cx, cy)
        self.control_points = pts

        # Higher density for smoother curves
        self.center = catmull_rom(pts, density=30)

        self.inner = []
        self.outer = []
        self._build_borders()

        self.checkpoints = []
        self._build_checkpoints(35)

        # start position & angle
        self.start_pos = self.center[0]
        dx = self.center[1][0] - self.center[0][0]
        dy = self.center[1][1] - self.center[0][1]
        self.start_angle = -math.degrees(math.atan2(dy, dx))

        # pre-render
        self.surface = None
        self._render()
        self.border_mask = None
        self._build_mask()

    # ------------------------------------------------------------------
    #  Robust border generation
    # ------------------------------------------------------------------
    def _build_borders(self):
        """Build inner/outer borders using half-width offset with
        curvature-adaptive clamping to prevent auto-intersections."""
        n = len(self.center)
        hw = self.track_width / 2.0          # <<< half-width, not full
        min_hw = hw * 0.45                    # never shrink below 45% of hw

        # 1) compute smoothed normals
        normals = _smooth_normals(self.center, window=4)

        # 2) compute per-point curvature and adaptive half-width
        adaptive_hw = []
        for i in range(n):
            curv = _curvature_at(self.center, i)
            if curv > 1e-6:
                # max offset before inner border crosses center = 1/curvature
                max_offset = 0.85 / curv      # 85% of radius
                effective = min(hw, max_offset)
                effective = max(effective, min_hw)
            else:
                effective = hw
            adaptive_hw.append(effective)

        # 3) smooth the adaptive hw to avoid abrupt width changes
        smoothed_hw = list(adaptive_hw)
        for _pass in range(3):
            tmp = list(smoothed_hw)
            for i in range(n):
                prev_hw = tmp[(i - 1) % n]
                next_hw = tmp[(i + 1) % n]
                smoothed_hw[i] = 0.25 * prev_hw + 0.5 * tmp[i] + 0.25 * next_hw
            # re-clamp
            for i in range(n):
                smoothed_hw[i] = max(min_hw, min(hw, smoothed_hw[i]))

        # 4) build raw borders
        self.inner = []
        self.outer = []
        for i in range(n):
            cx, cy = self.center[i]
            nx, ny = normals[i]
            h = smoothed_hw[i]
            self.inner.append((cx + nx * h, cy + ny * h))
            self.outer.append((cx - nx * h, cy - ny * h))

        # 5) fix remaining self-intersections by checking adjacent quads
        self._fix_border_intersections()

    def _fix_border_intersections(self):
        """Post-process: if adjacent border segments cross, pull them
        back toward the centerline."""
        n = len(self.inner)
        max_iters = 3
        for _ in range(max_iters):
            fixed = 0
            for i in range(n):
                j = (i + 1) % n

                # check inner side
                if i >= 2:
                    prev = (i - 1) % n
                    if _segments_intersect(self.inner[prev], self.inner[i],
                                           self.inner[i], self.inner[j]):
                        # pull point i toward center
                        cx, cy = self.center[i]
                        ix, iy = self.inner[i]
                        self.inner[i] = (cx + (ix - cx) * 0.7,
                                         cy + (iy - cy) * 0.7)
                        ox, oy = self.outer[i]
                        self.outer[i] = (cx + (ox - cx) * 0.7,
                                         cy + (oy - cy) * 0.7)
                        fixed += 1

                # check outer side
                if i >= 2:
                    prev = (i - 1) % n
                    if _segments_intersect(self.outer[prev], self.outer[i],
                                           self.outer[i], self.outer[j]):
                        cx, cy = self.center[i]
                        ix, iy = self.inner[i]
                        self.inner[i] = (cx + (ix - cx) * 0.7,
                                         cy + (iy - cy) * 0.7)
                        ox, oy = self.outer[i]
                        self.outer[i] = (cx + (ox - cx) * 0.7,
                                         cy + (oy - cy) * 0.7)
                        fixed += 1

            if fixed == 0:
                break

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

        # 2) sand run-off (use actual borders + extra margin)
        sand_extra = 12
        for i in range(n):
            j = (i + 1) % n
            # direction from center to inner/outer, extend it
            ci = self.center[i]
            cj = self.center[j]

            di_x, di_y = self.inner[i][0] - ci[0], self.inner[i][1] - ci[1]
            do_x, do_y = self.outer[i][0] - ci[0], self.outer[i][1] - ci[1]
            li = math.hypot(di_x, di_y) or 1
            lo = math.hypot(do_x, do_y) or 1

            dj_x, dj_y = self.inner[j][0] - cj[0], self.inner[j][1] - cj[1]
            doj_x, doj_y = self.outer[j][0] - cj[0], self.outer[j][1] - cj[1]
            lj = math.hypot(dj_x, dj_y) or 1
            loj = math.hypot(doj_x, doj_y) or 1

            si = (self.inner[i][0] + di_x / li * sand_extra,
                  self.inner[i][1] + di_y / li * sand_extra)
            so = (self.outer[i][0] + do_x / lo * sand_extra,
                  self.outer[i][1] + do_y / lo * sand_extra)
            ei = (self.inner[j][0] + dj_x / lj * sand_extra,
                  self.inner[j][1] + dj_y / lj * sand_extra)
            eo = (self.outer[j][0] + doj_x / loj * sand_extra,
                  self.outer[j][1] + doj_y / loj * sand_extra)
            pygame.draw.polygon(s, SAND, [si, ei, eo, so])

        # 3) main asphalt
        for i in range(n):
            j = (i + 1) % n
            quad = [self.inner[i], self.inner[j], self.outer[j], self.outer[i]]
            pygame.draw.polygon(s, ASPHALT, quad)

        # 4) asphalt texture lines
        for i in range(0, n, 4):
            j = (i + 1) % n
            mid_i = ((self.inner[i][0] + self.outer[i][0]) / 2,
                     (self.inner[i][1] + self.outer[i][1]) / 2)
            mid_j = ((self.inner[j][0] + self.outer[j][0]) / 2,
                     (self.inner[j][1] + self.outer[j][1]) / 2)
            pygame.draw.line(s, ASPHALT_LT, mid_i, mid_j, 1)

        # 5) kerbs -- drawn using per-segment normals from borders
        kerb_w = 5
        for i in range(n):
            seg = i % 8
            c_k = KERB_RED if seg < 4 else KERB_WHITE
            j = (i + 1) % n

            # inner kerb: strip just inside the inner border
            ci = self.center[i]
            ii, ij = self.inner[i], self.inner[j]
            dx_i = ii[0] - ci[0]
            dy_i = ii[1] - ci[1]
            li = math.hypot(dx_i, dy_i) or 1
            nxi, nyi = dx_i / li, dy_i / li
            ki1 = (ii[0] - nxi * kerb_w, ii[1] - nyi * kerb_w)

            cj = self.center[j]
            dx_j = ij[0] - cj[0]
            dy_j = ij[1] - cj[1]
            lj = math.hypot(dx_j, dy_j) or 1
            nxj, nyj = dx_j / lj, dy_j / lj
            ki2 = (ij[0] - nxj * kerb_w, ij[1] - nyj * kerb_w)
            pygame.draw.polygon(s, c_k, [ii, ij, ki2, ki1])

            # outer kerb
            oi, oj = self.outer[i], self.outer[j]
            dx_o = oi[0] - ci[0]
            dy_o = oi[1] - ci[1]
            lo = math.hypot(dx_o, dy_o) or 1
            nxo, nyo = dx_o / lo, dy_o / lo
            ko1 = (oi[0] - nxo * kerb_w, oi[1] - nyo * kerb_w)

            dx_oj = oj[0] - cj[0]
            dy_oj = oj[1] - cj[1]
            loj = math.hypot(dx_oj, dy_oj) or 1
            nxoj, nyoj = dx_oj / loj, dy_oj / loj
            ko2 = (oj[0] - nxoj * kerb_w, oj[1] - nyoj * kerb_w)
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
            L = math.hypot(dx, dy) or 1
            num_squares = 8
            for k in range(num_squares):
                t1 = k / num_squares
                t2 = (k + 1) / num_squares
                p1 = (inner[0] + dx * t1, inner[1] + dy * t1)
                p2 = (inner[0] + dx * t2, inner[1] + dy * t2)
                color = FINISH_W if k % 2 == 0 else FINISH_B
                perp_x, perp_y = -dy / L * 4, dx / L * 4
                quad = [
                    (p1[0] + perp_x, p1[1] + perp_y),
                    (p2[0] + perp_x, p2[1] + perp_y),
                    (p2[0] - perp_x, p2[1] - perp_y),
                    (p1[0] - perp_x, p1[1] - perp_y),
                ]
                pygame.draw.polygon(s, color, quad)

        self.surface = s

    # -- collision mask (polygon-fill approach) ----------------------------
    def _build_mask(self):
        """Build collision mask from the road polygon.
        White = off-track, Black = on-track."""
        ms = pygame.Surface((self.width, self.height))
        ms.fill(WHITE)

        # Draw the road as a filled polygon using quads
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
