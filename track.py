"""
Track module -- Robust procedural circuit generation (v3).
==========================================================
Training: 5 difficulty levels.
Testing:  3 additional unseen circuits.

Geometry pipeline:
  1. Catmull-Rom spline with analytic first derivative → tangent + normal
  2. Half-width offset:  center ± normal * (track_width / 2)
  3. Curvature-adaptive clamping:  hw = min(hw, 0.85 / κ)
  4. Angle-clamping between consecutive normals (max 8°)
  5. Laplacian smoothing on the adaptive half-widths (5 passes)
  6. Bow-tie / quad-inversion detection + fix
  7. Mask built from final road polygon only (no decorations)

Debug mode (D key):
  Displays centerline, normals, inner/outer borders, problem zones.
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

# Debug palette
DBG_CENTER  = (255, 255, 0)
DBG_INNER   = (0, 255, 100)
DBG_OUTER   = (255, 100, 0)
DBG_NORMAL  = (100, 100, 255)
DBG_PROBLEM = (255, 0, 0)


# ==========================================================================
#  Catmull-Rom spline with analytic derivative
# ==========================================================================
def catmull_rom(points, density=30):
    """Cyclic Catmull-Rom spline → list of (x, y)."""
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


def catmull_rom_derivative(points, density=30):
    """Cyclic Catmull-Rom analytic first derivative → tangent (dx, dy)."""
    result = []
    n = len(points)
    for i in range(n):
        p0, p1 = points[(i - 1) % n], points[i]
        p2, p3 = points[(i + 1) % n], points[(i + 2) % n]
        for s in range(density):
            t = s / density
            t2 = t * t
            dx = 0.5 * ((-p0[0] + p2[0]) +
                         2 * (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t +
                         3 * (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t2)
            dy = 0.5 * ((-p0[1] + p2[1]) +
                         2 * (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t +
                         3 * (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t2)
            result.append((dx, dy))
    return result


# ==========================================================================
#  Geometry helpers
# ==========================================================================
def _curvature_at(center, i):
    """Menger curvature 1/R through three consecutive points."""
    n = len(center)
    ax, ay = center[(i - 1) % n]
    bx, by = center[i]
    cx, cy = center[(i + 1) % n]
    area2 = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    a = math.hypot(bx - ax, by - ay)
    b = math.hypot(cx - bx, cy - by)
    c = math.hypot(ax - cx, ay - cy)
    denom = a * b * c
    if denom < 1e-9:
        return 0.0
    return abs(area2) / denom * 2.0


def _cross2d(ox, oy, ax, ay, bx, by):
    """2D cross product of OA × OB."""
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def _segments_intersect(p1, p2, p3, p4):
    """Proper intersection test between segment p1-p2 and p3-p4."""
    d1 = _cross2d(p3[0], p3[1], p4[0], p4[1], p1[0], p1[1])
    d2 = _cross2d(p3[0], p3[1], p4[0], p4[1], p2[0], p2[1])
    d3 = _cross2d(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
    d4 = _cross2d(p1[0], p1[1], p2[0], p2[1], p4[0], p4[1])
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _quad_is_bowtie(p0, p1, p2, p3):
    """Check if quad (p0, p1, p2, p3) is a bow-tie (self-intersecting).
    A well-formed quad has diagonals that DO intersect; a bow-tie has
    opposite edges that intersect."""
    return _segments_intersect(p0, p1, p3, p2) or \
           _segments_intersect(p0, p3, p1, p2)


def _angle_between_normals(n1, n2):
    """Unsigned angle in radians between two unit normals."""
    dot = n1[0] * n2[0] + n1[1] * n2[1]
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


# ==========================================================================
#  Level definitions
# ==========================================================================
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
    track_width = FULL road width  (half-width = track_width / 2)."""
    if level == 1:
        return [
            (cx, cy - 200), (cx + 155, cy - 185), (cx + 265, cy - 100),
            (cx + 280, cy + 15), (cx + 250, cy + 125), (cx + 115, cy + 195),
            (cx - 25, cy + 210), (cx - 155, cy + 180), (cx - 265, cy + 95),
            (cx - 280, cy - 25), (cx - 240, cy - 140), (cx - 115, cy - 195),
        ], 100

    if level == 2:
        return [
            (cx, cy - 220), (cx + 165, cy - 200), (cx + 290, cy - 120),
            (cx + 310, cy + 5), (cx + 280, cy + 100), (cx + 175, cy + 165),
            (cx + 80, cy + 220), (cx - 45, cy + 240), (cx - 170, cy + 200),
            (cx - 290, cy + 120), (cx - 320, cy + 15), (cx - 285, cy - 120),
            (cx - 165, cy - 210),
        ], 90

    if level == 3:
        return [
            (cx - 45, cy - 240), (cx + 160, cy - 220), (cx + 310, cy - 140),
            (cx + 340, cy - 10), (cx + 280, cy + 90),
            (cx + 140, cy + 50), (cx + 10, cy + 140),
            (cx + 140, cy + 230), (cx + 50, cy + 265),
            (cx - 100, cy + 245), (cx - 220, cy + 175),
            (cx - 330, cy + 90), (cx - 340, cy - 30),
            (cx - 280, cy - 135), (cx - 175, cy - 220),
        ], 80

    if level == 4:
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
        ], 70

    # level 5
    return [
        (cx + 10, cy - 265), (cx + 170, cy - 250), (cx + 330, cy - 190),
        (cx + 395, cy - 60), (cx + 360, cy + 50), (cx + 230, cy + 15),
        (cx + 120, cy + 110), (cx + 270, cy + 185), (cx + 365, cy + 250),
        (cx + 210, cy + 275), (cx + 70, cy + 215), (cx - 10, cy + 280),
        (cx - 120, cy + 215), (cx - 260, cy + 270), (cx - 370, cy + 190),
        (cx - 325, cy + 95), (cx - 175, cy + 65), (cx - 300, cy - 30),
        (cx - 395, cy - 110), (cx - 325, cy - 220), (cx - 170, cy - 250),
    ], 62


def _test_track_data(track_id, cx, cy):
    """Return (control_points, track_width) for test circuits."""
    if track_id == 1:
        # Technical circuit with tight esses — NO self-crossing centerline
        return [
            (cx - 30, cy - 260), (cx + 180, cy - 250), (cx + 340, cy - 170),
            (cx + 390, cy - 30), (cx + 340, cy + 80), (cx + 180, cy + 50),
            (cx + 50, cy + 130), (cx + 180, cy + 220), (cx + 340, cy + 260),
            (cx + 180, cy + 280), (cx + 20, cy + 230), (cx - 100, cy + 270),
            (cx - 260, cy + 240), (cx - 370, cy + 150), (cx - 390, cy + 20),
            (cx - 340, cy - 80), (cx - 200, cy - 50), (cx - 280, cy - 160),
            (cx - 200, cy - 240),
        ], 72

    if track_id == 2:
        # Riverside circuit with flowing curves — NO self-crossing centerline
        return [
            (cx - 350, cy - 60), (cx - 280, cy - 200), (cx - 120, cy - 255),
            (cx + 60, cy - 240), (cx + 200, cy - 200), (cx + 310, cy - 110),
            (cx + 380, cy + 10), (cx + 360, cy + 130), (cx + 220, cy + 200),
            (cx + 80, cy + 150), (cx - 50, cy + 200), (cx + 50, cy + 270),
            (cx - 60, cy + 280), (cx - 200, cy + 230), (cx - 320, cy + 140),
            (cx - 390, cy + 40),
        ], 72

    # track 3
    return [
        (cx - 320, cy - 220), (cx - 140, cy - 250), (cx + 60, cy - 210),
        (cx + 180, cy - 130), (cx + 120, cy - 30), (cx + 220, cy + 40),
        (cx + 360, cy - 20), (cx + 380, cy + 100), (cx + 300, cy + 180),
        (cx + 160, cy + 140), (cx + 60, cy + 210), (cx + 140, cy + 270),
        (cx, cy + 260), (cx - 140, cy + 210), (cx - 80, cy + 120),
        (cx - 200, cy + 60), (cx - 340, cy + 120), (cx - 380, cy + 20),
        (cx - 340, cy - 100), (cx - 220, cy - 160),
    ], 68


# ==========================================================================
#  Track class
# ==========================================================================
class Track:
    SPLINE_DENSITY = 30          # points per control segment

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

        # -- Spline + analytic tangents ------------------------------------
        self.center = catmull_rom(pts, density=self.SPLINE_DENSITY)
        self._tangents = catmull_rom_derivative(pts, density=self.SPLINE_DENSITY)

        # -- Borders -------------------------------------------------------
        self.inner = []
        self.outer = []
        self._normals = []               # stored for debug
        self._effective_hw = []           # stored for debug
        self._problem_indices = []        # indices with detected issues
        self._build_borders()

        # -- Checkpoints ---------------------------------------------------
        self.checkpoints = []
        self._build_checkpoints(35)

        # -- Start ---------------------------------------------------------
        self.start_pos = self.center[0]
        dx = self.center[1][0] - self.center[0][0]
        dy = self.center[1][1] - self.center[0][1]
        self.start_angle = -math.degrees(math.atan2(dy, dx))

        # -- Render & mask -------------------------------------------------
        self.surface = None
        self._render()
        self.border_mask = None
        self._build_mask()

        # -- Self-test (assert in debug) -----------------------------------
        self._validate_mask()

    # ==================================================================
    #  1) BORDER GENERATION  (the core fix)
    # ==================================================================
    def _build_borders(self):
        """Build inner/outer borders with full robustness pipeline."""
        n = len(self.center)
        hw = self.track_width / 2.0
        min_hw = hw * 0.45              # absolute minimum half-width

        # ---- Step 1: Normals from analytic derivative --------------------
        raw_normals = []
        for i in range(n):
            tx, ty = self._tangents[i]
            L = math.hypot(tx, ty)
            if L < 1e-9:
                # fallback: finite-difference
                pp = self.center[(i - 1) % n]
                pn = self.center[(i + 1) % n]
                tx, ty = pn[0] - pp[0], pn[1] - pp[1]
                L = math.hypot(tx, ty) or 1.0
            # normal = 90° CCW from tangent
            raw_normals.append((-ty / L, tx / L))

        # ---- Step 2: Angle-clamp consecutive normals (max 8°) -----------
        MAX_ANGLE_STEP = math.radians(8.0)
        normals = list(raw_normals)
        for _pass in range(3):
            new_normals = list(normals)
            for i in range(n):
                prev = normals[(i - 1) % n]
                curr = normals[i]
                nxt  = normals[(i + 1) % n]

                # check angle with previous
                angle_prev = _angle_between_normals(prev, curr)
                angle_next = _angle_between_normals(curr, nxt)

                if angle_prev > MAX_ANGLE_STEP or angle_next > MAX_ANGLE_STEP:
                    # blend with neighbours
                    avg_x = 0.25 * prev[0] + 0.50 * curr[0] + 0.25 * nxt[0]
                    avg_y = 0.25 * prev[1] + 0.50 * curr[1] + 0.25 * nxt[1]
                    L = math.hypot(avg_x, avg_y) or 1.0
                    new_normals[i] = (avg_x / L, avg_y / L)
            normals = new_normals
        self._normals = normals

        # ---- Step 3: Curvature-adaptive half-width -----------------------
        adaptive_hw = []
        for i in range(n):
            curv = _curvature_at(self.center, i)
            if curv > 1e-6:
                max_offset = 0.85 / curv
                effective = min(hw, max_offset)
                effective = max(effective, min_hw)
            else:
                effective = hw
            adaptive_hw.append(effective)

        # ---- Step 4: Laplacian smoothing (5 passes) ----------------------
        smoothed = list(adaptive_hw)
        for _pass in range(5):
            tmp = list(smoothed)
            for i in range(n):
                smoothed[i] = (0.2 * tmp[(i - 2) % n] +
                               0.2 * tmp[(i - 1) % n] +
                               0.2 * tmp[i] +
                               0.2 * tmp[(i + 1) % n] +
                               0.2 * tmp[(i + 2) % n])
            for i in range(n):
                smoothed[i] = max(min_hw, min(hw, smoothed[i]))
        # ---- Step 4b: Proximity clamping (prevent self-overlap) --------
        #  When two parts of the centerline pass close to each other
        #  (e.g. tight chicane), reduce half-widths so borders don't cross.
        MIN_SPLINE_SEP = max(n // 8, 20)   # must be far on spline
        MARGIN = 0.85                       # hw_i + hw_j <= dist * MARGIN
        for _prox_pass in range(3):         # multiple passes for convergence
            changed = False
            for i in range(n):
                ci_x, ci_y = self.center[i]
                for j in range(i + MIN_SPLINE_SEP,
                               i + n - MIN_SPLINE_SEP + 1):
                    jj = j % n
                    cj_x, cj_y = self.center[jj]
                    dist = math.hypot(ci_x - cj_x, ci_y - cj_y)
                    if dist < 1e-3:
                        continue
                    total_hw = smoothed[i] + smoothed[jj]
                    if total_hw > dist * MARGIN:
                        # Scale both down proportionally
                        ratio = (dist * MARGIN) / total_hw
                        new_i = max(min_hw, smoothed[i] * ratio)
                        new_j = max(min_hw, smoothed[jj] * ratio)
                        if new_i < smoothed[i] - 0.1 or new_j < smoothed[jj] - 0.1:
                            changed = True
                        smoothed[i] = min(smoothed[i], new_i)
                        smoothed[jj] = min(smoothed[jj], new_j)
            if not changed:
                break

        # Re-smooth after proximity clamping (2 light passes)
        for _pass in range(2):
            tmp = list(smoothed)
            for i in range(n):
                smoothed[i] = (0.15 * tmp[(i - 2) % n] +
                               0.20 * tmp[(i - 1) % n] +
                               0.30 * tmp[i] +
                               0.20 * tmp[(i + 1) % n] +
                               0.15 * tmp[(i + 2) % n])
            for i in range(n):
                smoothed[i] = max(min_hw, min(hw, smoothed[i]))

        self._effective_hw = smoothed

        # ---- Step 5: Build raw borders -----------------------------------
        self.inner = []
        self.outer = []
        for i in range(n):
            cx_, cy_ = self.center[i]
            nx, ny = normals[i]
            h = smoothed[i]
            assert h > 0, f"Negative effective hw at index {i}: {h}"
            self.inner.append((cx_ + nx * h, cy_ + ny * h))
            self.outer.append((cx_ - nx * h, cy_ - ny * h))

        # ---- Step 6: Fix bow-tie quads + non-adjacent intersections ------
        self._fix_quads()

    def _fix_quads(self):
        """Detect and fix bow-tie quads, adjacent intersections,
        AND non-adjacent border crossings."""
        n = len(self.inner)
        self._problem_indices = []
        MIN_SPLINE_SEP = max(n // 8, 20)

        for iteration in range(12):
            problems = []

            # --- A) Bow-ties + adjacent crossings -------------------------
            for i in range(n):
                j = (i + 1) % n
                if _quad_is_bowtie(self.inner[i], self.inner[j],
                                   self.outer[j], self.outer[i]):
                    problems.append(i)
                    problems.append(j)

                k = (j + 1) % n
                if _segments_intersect(self.inner[i], self.inner[j],
                                       self.inner[j], self.inner[k]):
                    problems.append(j)
                if _segments_intersect(self.outer[i], self.outer[j],
                                       self.outer[j], self.outer[k]):
                    problems.append(j)

            # --- B) Non-adjacent crossings --------------------------------
            #  Check inner[i]→inner[i+1] vs inner[j]→inner[j+1]
            #  for j far from i on the spline.
            for i in range(n):
                i1 = (i + 1) % n
                for delta in range(MIN_SPLINE_SEP,
                                   n - MIN_SPLINE_SEP + 1):
                    j = (i + delta) % n
                    j1 = (j + 1) % n
                    if _segments_intersect(self.inner[i], self.inner[i1],
                                           self.inner[j], self.inner[j1]):
                        problems.extend([i, i1, j, j1])
                    if _segments_intersect(self.outer[i], self.outer[i1],
                                           self.outer[j], self.outer[j1]):
                        problems.extend([i, i1, j, j1])

            if not problems:
                break

            problem_set = set(problems)
            self._problem_indices.extend(problem_set)

            # Pull each problem point toward center
            shrink = 0.60 if iteration < 4 else 0.45
            for idx in problem_set:
                cx_, cy_ = self.center[idx]
                ix, iy = self.inner[idx]
                ox, oy = self.outer[idx]
                self.inner[idx] = (cx_ + (ix - cx_) * shrink,
                                   cy_ + (iy - cy_) * shrink)
                self.outer[idx] = (cx_ + (ox - cx_) * shrink,
                                   cy_ + (oy - cy_) * shrink)

            # Smooth neighbours
            for idx in problem_set:
                for delta in [-2, -1, 1, 2]:
                    ni = (idx + delta) % n
                    if ni not in problem_set:
                        cx_, cy_ = self.center[ni]
                        ix, iy = self.inner[ni]
                        ox, oy = self.outer[ni]
                        f = 0.92 if abs(delta) == 1 else 0.96
                        self.inner[ni] = (cx_ + (ix - cx_) * f,
                                          cy_ + (iy - cy_) * f)
                        self.outer[ni] = (cx_ + (ox - cx_) * f,
                                          cy_ + (oy - cy_) * f)

        self._problem_indices = list(set(self._problem_indices))

    # ==================================================================
    #  2) CHECKPOINTS
    # ==================================================================
    def _build_checkpoints(self, num):
        n = len(self.center)
        step = max(1, n // num)
        for i in range(num):
            idx = (i * step) % n
            self.checkpoints.append((self.inner[idx], self.outer[idx], idx))

    # ==================================================================
    #  3) RENDERING  (decorations never affect mask)
    # ==================================================================
    def _render(self):
        s = pygame.Surface((self.width, self.height))
        n = len(self.inner)

        # 1) grass
        for row in range(0, self.height, 16):
            color = GRASS_1 if (row // 16) % 2 == 0 else GRASS_2
            pygame.draw.rect(s, color, (0, row, self.width, 16))

        # 2) sand run-off
        sand_extra = 12
        for i in range(n):
            j = (i + 1) % n
            ci, cj = self.center[i], self.center[j]
            ni, nj = self._normals[i], self._normals[j]
            hi, hj = self._effective_hw[i], self._effective_hw[j]
            si_i = (ci[0] + ni[0] * (hi + sand_extra),
                    ci[1] + ni[1] * (hi + sand_extra))
            so_i = (ci[0] - ni[0] * (hi + sand_extra),
                    ci[1] - ni[1] * (hi + sand_extra))
            ei_j = (cj[0] + nj[0] * (hj + sand_extra),
                    cj[1] + nj[1] * (hj + sand_extra))
            eo_j = (cj[0] - nj[0] * (hj + sand_extra),
                    cj[1] - nj[1] * (hj + sand_extra))
            pygame.draw.polygon(s, SAND, [si_i, ei_j, eo_j, so_i])

        # 3) main asphalt (quads from inner/outer borders)
        for i in range(n):
            j = (i + 1) % n
            quad = [self.inner[i], self.inner[j], self.outer[j], self.outer[i]]
            pygame.draw.polygon(s, ASPHALT, quad)

        # 4) asphalt texture
        for i in range(0, n, 4):
            j = (i + 1) % n
            mid_i = ((self.inner[i][0] + self.outer[i][0]) / 2,
                     (self.inner[i][1] + self.outer[i][1]) / 2)
            mid_j = ((self.inner[j][0] + self.outer[j][0]) / 2,
                     (self.inner[j][1] + self.outer[j][1]) / 2)
            pygame.draw.line(s, ASPHALT_LT, mid_i, mid_j, 1)

        # 5) kerbs (per-vertex normals for coherent strips)
        kerb_w = 5
        for i in range(n):
            seg = i % 8
            c_k = KERB_RED if seg < 4 else KERB_WHITE
            j = (i + 1) % n

            # inner kerb
            ni_x, ni_y = self._normals[i]
            nj_x, nj_y = self._normals[j]
            ii, ij = self.inner[i], self.inner[j]
            ki1 = (ii[0] - ni_x * kerb_w, ii[1] - ni_y * kerb_w)
            ki2 = (ij[0] - nj_x * kerb_w, ij[1] - nj_y * kerb_w)
            pygame.draw.polygon(s, c_k, [ii, ij, ki2, ki1])

            # outer kerb
            oi, oj = self.outer[i], self.outer[j]
            ko1 = (oi[0] + ni_x * kerb_w, oi[1] + ni_y * kerb_w)
            ko2 = (oj[0] + nj_x * kerb_w, oj[1] + nj_y * kerb_w)
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
            num_sq = 8
            for k in range(num_sq):
                t1 = k / num_sq
                t2 = (k + 1) / num_sq
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

    # ==================================================================
    #  4) COLLISION MASK  (road polygon ONLY, no decorations)
    # ==================================================================
    def _build_mask(self):
        """Mask: white=off-track, black=on-track.
        Built ONLY from the inner/outer road polygon quads."""
        ms = pygame.Surface((self.width, self.height))
        ms.fill(WHITE)

        n = len(self.inner)
        for i in range(n):
            j = (i + 1) % n
            quad = [self.inner[i], self.inner[j], self.outer[j], self.outer[i]]
            int_q = [(int(p[0]), int(p[1])) for p in quad]
            pygame.draw.polygon(ms, BLACK, int_q)

        self.border_mask = pygame.mask.from_threshold(ms, WHITE, (10, 10, 10))

    # ==================================================================
    #  5) VALIDATION
    # ==================================================================
    def _validate_mask(self):
        """Check that every centerline point is on-track.
        If any point is off-track, it indicates a mask/border bug."""
        problems = 0
        for i, (px, py) in enumerate(self.center):
            ix, iy = int(px), int(py)
            if 0 <= ix < self.width and 0 <= iy < self.height:
                if self.border_mask.get_at((ix, iy)):
                    problems += 1

        if problems > 0:
            print(f"  [TRACK WARN] {problems}/{len(self.center)} centerline "
                  f"points are off-track on {self.level_name} -- check borders!")

    # ==================================================================
    #  6) PUBLIC API
    # ==================================================================
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

    # ==================================================================
    #  7) DEBUG OVERLAY
    # ==================================================================
    def draw_debug(self, screen):
        """Draw debug overlay: centerline, normals, borders, problems."""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        n = len(self.center)

        # centerline
        pts_c = [(int(p[0]), int(p[1])) for p in self.center]
        if len(pts_c) >= 2:
            pygame.draw.lines(overlay, (*DBG_CENTER, 120), True, pts_c, 1)

        # inner border
        pts_i = [(int(p[0]), int(p[1])) for p in self.inner]
        if len(pts_i) >= 2:
            pygame.draw.lines(overlay, (*DBG_INNER, 150), True, pts_i, 1)

        # outer border
        pts_o = [(int(p[0]), int(p[1])) for p in self.outer]
        if len(pts_o) >= 2:
            pygame.draw.lines(overlay, (*DBG_OUTER, 150), True, pts_o, 1)

        # normals (every 10th point)
        norm_len = 15
        for i in range(0, n, 10):
            cx_, cy_ = self.center[i]
            nx, ny = self._normals[i]
            ex = cx_ + nx * norm_len
            ey = cy_ + ny * norm_len
            pygame.draw.line(overlay, (*DBG_NORMAL, 100),
                             (int(cx_), int(cy_)), (int(ex), int(ey)), 1)

        # problem zones (red circles)
        for idx in self._problem_indices:
            px, py = self.center[idx]
            pygame.draw.circle(overlay, (*DBG_PROBLEM, 180),
                               (int(px), int(py)), 6, 2)

        # effective half-width markers (every 20th point)
        for i in range(0, n, 20):
            cx_, cy_ = self.center[i]
            hw = self._effective_hw[i]
            max_hw = self.track_width / 2.0
            ratio = hw / max_hw
            if ratio < 0.8:
                # orange = narrowed
                c = (255, int(165 * ratio), 0, 150)
                pygame.draw.circle(overlay, c, (int(cx_), int(cy_)), 4, 1)

        screen.blit(overlay, (0, 0))
