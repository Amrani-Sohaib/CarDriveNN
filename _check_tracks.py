"""Temporary script to test track control points for border crossings."""
import math

def catmull_rom(points, density=28):
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

cx, cy = 450, 310

def check_track(pts, tw, label):
    center = catmull_rom(pts, density=28)
    n = len(center)
    min_radius = float('inf')
    problem_count = 0
    worst_segments = {}
    for i in range(n):
        pp = center[(i - 1) % n]
        pc = center[i]
        pn = center[(i + 1) % n]
        dx1, dy1 = pc[0] - pp[0], pc[1] - pp[1]
        dx2, dy2 = pn[0] - pc[0], pn[1] - pc[1]
        cross = dx1 * dy2 - dy1 * dx2
        ds = math.hypot(dx1, dy1)
        if ds < 0.01:
            continue
        curvature = abs(cross) / (ds ** 3 + 1e-9)
        if curvature > 0.001:
            radius = 1.0 / curvature
            min_radius = min(min_radius, radius)
            seg = i // 28
            if radius < tw:
                problem_count += 1
                if seg not in worst_segments or radius < worst_segments[seg][0]:
                    worst_segments[seg] = (radius, i)
    
    ok = problem_count == 0
    status = "OK !!!" if ok else f"{problem_count} problems"
    print(f"{label} (tw={tw}): min_r={min_radius:.1f}  {status}")
    if worst_segments:
        for seg in sorted(worst_segments, key=lambda s: worst_segments[s][0])[:6]:
            r, idx = worst_segments[seg]
            cp = seg % len(pts)
            print(f"  seg={seg} ctrl_pt={cp} r={r:.1f}")
    return ok


# Level 3: Chicane circuit
# The S-chicane (pts 3-7) must have gentler curves
pts3 = [
    (cx - 45, cy - 230),
    (cx + 150, cy - 210),
    (cx + 300, cy - 130),
    (cx + 340, cy + 10),
    (cx + 270, cy + 130),
    (cx + 80, cy + 100),
    (cx - 60, cy + 200),
    (cx + 100, cy + 260),
    (cx + 40, cy + 270),
    (cx - 100, cy + 245),
    (cx - 220, cy + 170),
    (cx - 310, cy + 80),
    (cx - 320, cy - 40),
    (cx - 270, cy - 140),
    (cx - 170, cy - 215),
]
check_track(pts3, 42, "Lv3")

# Level 4: Hairpin Monza
pts4 = [
    (cx - 30, cy - 250),
    (cx + 170, cy - 235),
    (cx + 330, cy - 150),
    (cx + 365, cy - 10),
    (cx + 310, cy + 110),
    (cx + 140, cy + 100),
    (cx + 30, cy + 200),
    (cx + 175, cy + 265),
    (cx + 310, cy + 270),
    (cx + 150, cy + 280),
    (cx, cy + 210),
    (cx - 150, cy + 280),
    (cx - 290, cy + 215),
    (cx - 360, cy + 110),
    (cx - 340, cy - 10),
    (cx - 300, cy - 100),
    (cx - 350, cy - 180),
    (cx - 265, cy - 240),
]
check_track(pts4, 36, "Lv4")

# Level 5: Infernal Circuit
pts5 = [
    (cx, cy - 260),
    (cx + 160, cy - 245),
    (cx + 320, cy - 190),
    (cx + 390, cy - 50),
    (cx + 360, cy + 70),
    (cx + 210, cy + 50),
    (cx + 100, cy + 160),
    (cx + 270, cy + 230),
    (cx + 360, cy + 265),
    (cx + 200, cy + 280),
    (cx + 70, cy + 205),
    (cx, cy + 278),
    (cx - 110, cy + 205),
    (cx - 255, cy + 270),
    (cx - 365, cy + 190),
    (cx - 340, cy + 90),
    (cx - 200, cy + 80),
    (cx - 310, cy - 30),
    (cx - 390, cy - 120),
    (cx - 320, cy - 215),
    (cx - 170, cy - 248),
]
check_track(pts5, 32, "Lv5")
