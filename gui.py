"""
GUI module -- Two resizable, HiDPI-aware windows:
  Window 1 (left):  Track + Car + Stats panel   (resizable)
  Window 2 (right): Dedicated NN Visualisation   (resizable)

Everything is driven by a scale factor S  (default 1.0).
  - S controls font sizes, paddings, bar thicknesses, etc.
  - The track is rendered onto an off-screen surface at its native 900x620
    then blitted/scaled into the track area of the window.
  - When the user resizes a window, S is recomputed so the layout fills it.
  - On 4K / Retina screens the initial S is auto-detected to ~2.0.
"""
import math
import pygame
from pygame._sdl2.video import Window as SDLWindow, Renderer, Texture
import numpy as np

# -- UI Palette ------------------------------------------------------------
BG          = (14, 14, 22)
PANEL_BG    = (22, 22, 36)
ACCENT      = (0, 160, 255)
TEXT        = (220, 220, 225)
TEXT_DIM    = (110, 110, 130)
GOOD        = (0, 200, 100)
BAD         = (220, 50, 50)
GOLD        = (255, 210, 0)
NODE_ON     = (0, 230, 140)
NODE_OFF    = (55, 55, 75)
SEPARATOR   = (50, 50, 68)
DARK_BG     = (16, 16, 26)
CONN_POS    = (60, 200, 120)
CONN_NEG    = (200, 60, 60)
PPO_BLUE    = (80, 140, 255)
PPO_GREEN   = (80, 220, 120)
PPO_RED     = (220, 80, 80)

LEVEL_COLORS = [
    (0, 200, 100), (170, 210, 0), (255, 170, 0),
    (255, 70, 0),  (220, 25, 25),
]

INPUT_LABELS = [
    "R-180", "R-150", "R-120", "R-90", "R-60", "R-30",
    "R 0", "R 30", "R 60", "R 90", "R 120", "R 150",
    "Speed", "AngleCP", "Curve",
]

OUTPUT_LABELS = ["Steer", "Throttle"]

# -- Base (1x) logical dimensions -----------------------------------------
BASE_TRACK_W, BASE_TRACK_H = 900, 620
BASE_PANEL_W = 220
BASE_WIN1_W  = BASE_TRACK_W + BASE_PANEL_W  # 1120
BASE_WIN1_H  = BASE_TRACK_H                 # 620
BASE_NN_W, BASE_NN_H = 560, 620

# Exported for main.py  (track is always built at this fixed size)
TRACK_W, TRACK_H = BASE_TRACK_W, BASE_TRACK_H


# -- helpers ---------------------------------------------------------------
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _detect_initial_scale():
    """Pick a good starting scale based on desktop resolution."""
    try:
        info = pygame.display.Info()
        dw = info.current_w
        if dw >= 3840:
            return 2.0
        if dw >= 2560:
            return 1.5
        return 1.0
    except Exception:
        return 1.0


class GUI:
    """Resizable, HiDPI-aware dual-window GUI for LSTM-PPO."""

    # -------------------------------------------------------------------
    def __init__(self, test_mode=False):
        pygame.init()
        self.test_mode = test_mode

        # initial scale factor
        self._s = _detect_initial_scale()

        # ---- Window 1 (pygame main window -- resizable) ----
        w1w = int(BASE_WIN1_W * self._s)
        w1h = int(BASE_WIN1_H * self._s)
        title = "CarAI -- Test" if test_mode else "CarAI -- Track"
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((w1w, w1h), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        # off-screen track surface at native resolution
        self.track_surface = pygame.Surface((BASE_TRACK_W, BASE_TRACK_H))

        # ---- Window 2 (SDL2 window -- resizable) ----
        self._nn_s = self._s  # independent scale for NN window
        self._create_nn_window()

        # build fonts at current scale
        self._rebuild_fonts(self._s)
        self._rebuild_nn_fonts(self._nn_s)

        self.fps = 60
        self.show_radars = True
        self.debug_track = False
        self.paused = False
        self.speed = 1
        self.save_requested = False
        self.save_flash = 0

        self.hist_fit  = []
        self.hist_best = []

    # -------------------------------------------------------------------
    #  Font factories
    # -------------------------------------------------------------------
    def _rebuild_fonts(self, s):
        """Rebuild main-window fonts at scale *s*."""
        self._font_title = pygame.font.SysFont("Menlo", max(8, int(18 * s)), bold=True)
        self._font_md    = pygame.font.SysFont("Menlo", max(7, int(13 * s)), bold=True)
        self._font_sm    = pygame.font.SysFont("Menlo", max(7, int(11 * s)))
        self._font_xs    = pygame.font.SysFont("Menlo", max(6, int(10 * s)))
        self._font_nn_sm_main = pygame.font.SysFont("Menlo", max(6, int(9 * s)))

    def _rebuild_nn_fonts(self, s):
        """Rebuild NN-window fonts at scale *s*."""
        self._nn_font_title = pygame.font.SysFont("Menlo", max(8, int(18 * s)), bold=True)
        self._nn_font_md    = pygame.font.SysFont("Menlo", max(7, int(13 * s)), bold=True)
        self._nn_font_sm    = pygame.font.SysFont("Menlo", max(7, int(11 * s)))
        self._nn_font_xs    = pygame.font.SysFont("Menlo", max(6, int(10 * s)))
        self._nn_font_nn    = pygame.font.SysFont("Menlo", max(7, int(11 * s)), bold=True)
        self._nn_font_nn_sm = pygame.font.SysFont("Menlo", max(6, int(9 * s)))

    # -------------------------------------------------------------------
    def _create_nn_window(self):
        try:
            nn_w = int(BASE_NN_W * self._nn_s)
            nn_h = int(BASE_NN_H * self._nn_s)
            self.nn_sdl_window = SDLWindow(
                "CarAI -- Neural Network (LSTM)",
                size=(nn_w, nn_h),
                resizable=True,
            )
            self.nn_renderer = Renderer(self.nn_sdl_window)
            self.has_nn_window = True
        except Exception:
            self.has_nn_window = False

    # -------------------------------------------------------------------
    #  Events
    # -------------------------------------------------------------------
    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.VIDEORESIZE:
                w, h = max(400, ev.w), max(300, ev.h)
                self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                self._s = h / BASE_WIN1_H
                self._rebuild_fonts(self._s)
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return False
                if ev.key == pygame.K_r:
                    self.show_radars = not self.show_radars
                if ev.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if ev.key == pygame.K_UP:
                    self.speed = min(20, self.speed + 1)
                if ev.key == pygame.K_DOWN:
                    self.speed = max(1, self.speed - 1)
                if ev.key == pygame.K_d:
                    self.debug_track = not self.debug_track
                if ev.key == pygame.K_s and not self.test_mode:
                    self.save_requested = True
                    self.save_flash = 90

        # poll NN window size
        if self.has_nn_window:
            try:
                sz = self.nn_sdl_window.size
                new_s = sz[1] / BASE_NN_H
                if abs(new_s - self._nn_s) > 0.02:
                    self._nn_s = new_s
                    self._rebuild_nn_fonts(self._nn_s)
            except Exception:
                pass
        return True

    # -------------------------------------------------------------------
    #  Main draw
    # -------------------------------------------------------------------
    def draw(self, track, car, stats, network=None, activations=None):
        s = self._s
        win_w, win_h = self.screen.get_size()

        self.screen.fill(BG)

        # --- track area: draw track at its native res then scale ---------
        track_area_w = int(win_w - BASE_PANEL_W * s)
        track_area_h = win_h

        self.track_surface.fill(BG)
        track.draw(self.track_surface)
        track.draw_checkpoints(self.track_surface)
        if self.debug_track:
            track.draw_debug(self.track_surface)
        if car:
            car.draw(self.track_surface, show_radars=self.show_radars)

        scaled_track = pygame.transform.smoothscale(
            self.track_surface, (max(1, track_area_w), max(1, track_area_h))
        )
        self.screen.blit(scaled_track, (0, 0))

        # --- panel -------------------------------------------------------
        panel_w = int(BASE_PANEL_W * s)
        panel_x = win_w - panel_w
        self._panel(panel_x, panel_w, win_h, car, stats, s)

        # save flash
        if self.save_flash > 0:
            self.save_flash -= 1
            alpha = min(255, self.save_flash * 4)
            flash_s = pygame.Surface((track_area_w, int(30 * s)), pygame.SRCALPHA)
            flash_s.fill((0, 180, 80, min(200, alpha)))
            self.screen.blit(flash_s, (0, 0))
            self._blit(self.screen, self._font_md, "Model saved!",
                       (255, 255, 255), int(10 * s), int(6 * s))

        pygame.display.flip()

        # NN window
        if network and activations:
            self._draw_nn_window(network, activations, car, stats)

        self.clock.tick(self.fps)

    # =====================================================================
    #  PANEL  (drawn directly on self.screen)
    # =====================================================================
    def _panel(self, px, pw, ph, car, st, s):
        pad = int(8 * s)

        panel_rect = pygame.Rect(px, 0, pw, ph)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, ACCENT, (px, 0), (px, ph), max(1, int(2 * s)))

        x = px + pad
        right = px + pw - pad
        usable = pw - 2 * pad
        y = pad

        # title
        self._blit(self.screen, self._font_title, "CarAI", ACCENT, x, y)
        self._blit(self.screen, self._font_nn_sm_main, "PPO+LSTM", PPO_BLUE,
                   x + int(60 * s), y + int(5 * s))
        y += int(22 * s)
        self._hsep(self.screen, x, y, usable); y += int(6 * s)

        # level
        level = st.get("level", 1)
        lname = st.get("level_name", "")
        self._blit(self.screen, self._font_sm, lname, GOLD, x, y)
        y += int(16 * s)
        bw = (usable - int(4 * 3 * s)) // 5
        for i in range(5):
            r = pygame.Rect(x + i * (bw + int(3 * s)), y, bw, int(8 * s))
            c = LEVEL_COLORS[i] if i < level else (35, 35, 48)
            pygame.draw.rect(self.screen, c, r, border_radius=max(1, int(2 * s)))
        y += int(16 * s)
        self._hsep(self.screen, x, y, usable); y += int(6 * s)

        # stats
        ep     = st.get("episode", 0)
        best_f = st.get("best_fitness", 0)
        cur_f  = st.get("current_fitness", 0)
        imp    = st.get("improvements", 0)
        npar   = st.get("params_count", 0)
        avg_r  = st.get("avg_reward", 0)
        alive_str = "YES" if (car and car.alive) else "DEAD"
        alive_c   = GOOD if (car and car.alive) else BAD

        stat_rows = [
            ("Episode",   f"{ep}",          ACCENT),
            ("Alive",     alive_str,        alive_c),
            ("Fitness",   f"{cur_f:.0f}",   TEXT),
            ("Best",      f"{best_f:.0f}",  GOLD),
            ("Improved",  f"{imp}",         GOOD),
            ("AvgReward", f"{avg_r:.2f}",   PPO_GREEN),
            ("Params",    f"{npar:,}",      TEXT_DIM),
            ("Speed",     f"x{self.speed}", TEXT),
        ]
        lh = int(15 * s)
        for label, val, col in stat_rows:
            self._blit(self.screen, self._font_xs, label, TEXT_DIM, x, y)
            vr = self._font_xs.render(val, True, col)
            self.screen.blit(vr, (right - vr.get_width(), y))
            y += lh

        y += int(4 * s)
        self._hsep(self.screen, x, y, usable); y += int(6 * s)

        # PPO stats
        self._blit(self.screen, self._font_md, "PPO", PPO_BLUE, x, y)
        y += int(16 * s)
        ppo_rows = [
            ("PG Loss",  f"{st.get('pg_loss', 0):.4f}",  PPO_RED),
            ("V Loss",   f"{st.get('v_loss', 0):.4f}",   PPO_RED),
            ("Entropy",  f"{st.get('entropy', 0):.4f}",  PPO_GREEN),
            ("Updates",  f"{st.get('total_updates', 0)}", TEXT_DIM),
            ("LR",       f"{st.get('lr', 3e-4):.1e}",    TEXT_DIM),
            ("Step",     f"{st.get('global_step', 0):,}", TEXT_DIM),
            ("Device",   f"{st.get('device', 'cpu')}",   PPO_BLUE),
        ]
        plh = int(14 * s)
        for label, val, col in ppo_rows:
            self._blit(self.screen, self._font_xs, label, TEXT_DIM, x, y)
            vr = self._font_xs.render(val, True, col)
            self.screen.blit(vr, (right - vr.get_width(), y))
            y += plh

        y += int(4 * s)
        self._hsep(self.screen, x, y, usable); y += int(6 * s)

        # car info
        if car:
            self._blit(self.screen, self._font_md, "CAR", ACCENT, x, y)
            y += int(16 * s)
            car_rows = [
                ("Checkpts", f"{car.cp_passed}"),
                ("Laps",     f"{car.laps}"),
                ("Speed",    f"{car.speed:.1f}"),
                ("Distance", f"{car.dist:.0f}"),
            ]
            for label, val in car_rows:
                self._blit(self.screen, self._font_xs, label, TEXT_DIM, x, y)
                vr = self._font_xs.render(val, True, TEXT)
                self.screen.blit(vr, (right - vr.get_width(), y))
                y += lh
            y += int(4 * s)
            self._hsep(self.screen, x, y, usable); y += int(6 * s)

        # learning curve
        self._blit(self.screen, self._font_md, "LEARNING", ACCENT, x, y)
        y += int(16 * s)
        gw = usable
        gh = int(60 * s)
        gr = pygame.Rect(x, y, gw, gh)
        pygame.draw.rect(self.screen, DARK_BG, gr, border_radius=max(1, int(3 * s)))
        pygame.draw.rect(self.screen, SEPARATOR, gr, 1, border_radius=max(1, int(3 * s)))

        if len(self.hist_best) > 1:
            data = self.hist_best[-60:]
            mx = max(data) or 1
            pts = []
            n = len(data)
            for i, v in enumerate(data):
                ppx = x + 3 + i * (gw - 6) / max(1, n - 1)
                ppy = y + gh - 3 - (v / mx) * (gh - 10)
                pts.append((ppx, ppy))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, GOLD, False, pts, max(1, int(2 * s)))
            if len(self.hist_fit) > 1:
                data_c = self.hist_fit[-60:]
                pts_c = []
                for i, v in enumerate(data_c):
                    ppx = x + 3 + i * (gw - 6) / max(1, len(data_c) - 1)
                    ppy = y + gh - 3 - (v / mx) * (gh - 10)
                    pts_c.append((ppx, ppy))
                if len(pts_c) >= 2:
                    pygame.draw.lines(self.screen, (70, 70, 110), False, pts_c, 1)
            self._blit(self.screen, self._font_nn_sm_main, f"{mx:.0f}",
                       TEXT_DIM, x + 2, y + 1)

        y += gh + int(6 * s)

        # controls
        self._hsep(self.screen, x, y, usable); y += int(4 * s)
        clh = int(12 * s)
        if self.test_mode:
            for t in ["SPC:Pause R:Radar D:Debug", "Up/Dn:Speed N:Next", "ESC:Quit"]:
                self._blit(self.screen, self._font_nn_sm_main, t, TEXT_DIM, x, y)
                y += clh
            y += int(4 * s)
            self._blit(self.screen, self._font_md, "TEST MODE", PPO_BLUE, x, y)
        else:
            for t in ["SPC:Pause R:Radar D:Debug", "S:Save Up/Dn:Speed ESC:Quit"]:
                self._blit(self.screen, self._font_nn_sm_main, t, TEXT_DIM, x, y)
                y += clh

    # =====================================================================
    #  WINDOW 2 -- Neural Network  (all coords scaled by nn_s)
    # =====================================================================
    def _draw_nn_window(self, network, activations, car, stats):
        if not self.has_nn_window:
            return

        try:
            pw, ph = self.nn_sdl_window.size
        except Exception:
            return
        if pw < 10 or ph < 10:
            return

        s = self._nn_s
        buf = pygame.Surface((pw, ph))
        buf.fill(BG)

        pad = int(10 * s)

        # title bar
        bar_h = int(30 * s)
        pygame.draw.rect(buf, PANEL_BG, (0, 0, pw, bar_h))
        pygame.draw.line(buf, ACCENT, (0, bar_h), (pw, bar_h), 1)
        self._blit(buf, self._nn_font_title, "LSTM Actor-Critic", ACCENT, pad, int(5 * s))

        # architecture
        arch_str = network.get_architecture_str()
        total_p = network.count_params()
        self._blit(buf, self._nn_font_nn_sm, f"{arch_str}  [{total_p:,}p]",
                   TEXT_DIM, pad, int(34 * s))

        # NN diagram
        nn_y = int(50 * s)
        nn_h = int(280 * s)
        self._draw_nn_diagram(buf, pad, nn_y, pw - 2 * pad, nn_h,
                              network, activations, s)

        # bottom panels
        bot_y = nn_y + nn_h + int(8 * s)
        bot_h = ph - bot_y - pad
        half_w = (pw - 3 * pad) // 2

        self._draw_io_panel(buf, pad, bot_y, half_w, bot_h, activations, car, s)
        self._draw_act_panel(buf, pad * 2 + half_w, bot_y, half_w, bot_h,
                             network, activations, s)

        tex = Texture.from_surface(self.nn_renderer, buf)
        self.nn_renderer.clear()
        tex.draw()
        self.nn_renderer.present()
        del tex

    # -- NN diagram --------------------------------------------------------
    def _draw_nn_diagram(self, surface, x0, y0, w, h, network, activations, s):
        box = pygame.Rect(x0, y0, w, h)
        pygame.draw.rect(surface, DARK_BG, box, border_radius=max(1, int(6 * s)))
        pygame.draw.rect(surface, SEPARATOR, box, 1, border_radius=max(1, int(6 * s)))

        arch = network.get_layer_sizes()
        n_layers = len(arch)

        draw_x0 = x0 + int(45 * s)
        draw_x1 = x0 + w - int(50 * s)
        draw_y0 = y0 + int(22 * s)
        draw_y1 = y0 + h - int(20 * s)

        layer_xs = [draw_x0 + i * (draw_x1 - draw_x0) / max(1, n_layers - 1)
                    for i in range(n_layers)]

        max_show = 14
        node_positions = []

        for li, size in enumerate(arch):
            show = min(size, max_show)
            truncated = size > max_show
            positions = []
            avail = draw_y1 - draw_y0
            spacing = avail / (show + (1 if truncated else 0))
            act = activations[li] if li < len(activations) else [0] * size
            if isinstance(act, (int, float)):
                act = [act]

            for ni in range(show):
                nx = layer_xs[li]
                ny = draw_y0 + (ni + 0.5) * spacing
                val = float(act[ni]) if ni < len(act) else 0
                positions.append((nx, ny, val))

            if truncated:
                nx = layer_xs[li]
                ny = draw_y0 + (show + 0.5) * spacing
                positions.append((nx, ny, -1))

            node_positions.append(positions)

        # connections
        conn_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        for li in range(n_layers - 1):
            for n1 in node_positions[li]:
                if n1[2] == -1:
                    continue
                for n2 in node_positions[li + 1]:
                    if n2[2] == -1:
                        continue
                    strength = abs(n1[2])
                    alpha = max(6, min(100, int(strength * 160)))
                    c = (*CONN_POS, alpha) if n1[2] >= 0 else (*CONN_NEG, alpha)
                    p1 = (n1[0] - x0, n1[1] - y0)
                    p2 = (n2[0] - x0, n2[1] - y0)
                    pygame.draw.line(conn_surf, c, p1, p2, 1)
        surface.blit(conn_surf, (x0, y0))

        # nodes
        for li, positions in enumerate(node_positions):
            is_io = li == 0 or li == n_layers - 1
            for (nx, ny, val) in positions:
                if val == -1:
                    self._blit(surface, self._nn_font_xs, "..", TEXT_DIM,
                               nx - int(5 * s), ny - int(4 * s))
                    continue

                clamped = max(0.0, min(1.0, abs(val)))
                r = int(40 + 160 * (1 - clamped))
                g = int(40 + 210 * clamped)
                b = int(50 + 30 * clamped)
                color = (r, g, b)
                rad = int(6 * s) if is_io else int(4 * s)

                pygame.draw.circle(surface, color, (int(nx), int(ny)), rad)
                if clamped > 0.5:
                    oc = (min(255, int(r * 1.3)), min(255, int(g * 1.2)),
                          min(255, int(b * 1.2)))
                    pygame.draw.circle(surface, oc, (int(nx), int(ny)), rad, 1)
                else:
                    pygame.draw.circle(surface, SEPARATOR, (int(nx), int(ny)), rad, 1)

        # layer labels
        layer_names = ["In", "FC1", "FC2", "LSTM", "ActFC", "Out"]
        for i in range(n_layers):
            lx = layer_xs[i]
            s_txt = str(arch[i])
            tw = self._nn_font_nn_sm.size(s_txt)[0]
            self._blit(surface, self._nn_font_nn_sm, s_txt, ACCENT,
                       lx - tw // 2, y0 + int(4 * s))

            if i < len(layer_names):
                lbl = layer_names[i]
                tw2 = self._nn_font_nn_sm.size(lbl)[0]
                col = PPO_BLUE if lbl == "LSTM" else TEXT_DIM
                self._blit(surface, self._nn_font_nn_sm, lbl, col,
                           lx - tw2 // 2, y0 + h - int(15 * s))

        # output value labels
        for idx, (nx, ny, val) in enumerate(node_positions[-1]):
            if val == -1:
                continue
            if idx < len(OUTPUT_LABELS):
                lbl = OUTPUT_LABELS[idx]
                c = GOOD if abs(val) > 0.3 else NODE_OFF
                self._blit(surface, self._nn_font_nn_sm, lbl, c,
                           nx + int(10 * s), ny - int(9 * s))
                self._blit(surface, self._nn_font_nn_sm, f"{val:+.2f}", c,
                           nx + int(10 * s), ny + int(1 * s))

    # -- I/O panel ---------------------------------------------------------
    def _draw_io_panel(self, surface, x0, y0, w, h, activations, car, s):
        box = pygame.Rect(x0, y0, w, h)
        pygame.draw.rect(surface, DARK_BG, box, border_radius=max(1, int(6 * s)))
        pygame.draw.rect(surface, SEPARATOR, box, 1, border_radius=max(1, int(6 * s)))

        ix = x0 + int(8 * s)
        iw = w - int(16 * s)
        y = y0 + int(6 * s)

        self._blit(surface, self._nn_font_nn, "INPUTS (15)", ACCENT, ix, y)
        y += int(14 * s)

        if activations and len(activations) > 0:
            inputs = activations[0]
            label_w = int(46 * s)
            val_w = int(24 * s)
            bar_w = iw - label_w - val_w
            for i in range(min(15, len(inputs))):
                val = float(inputs[i])
                lbl = INPUT_LABELS[i] if i < len(INPUT_LABELS) else f"I{i}"
                self._blit(surface, self._nn_font_nn_sm, lbl, TEXT_DIM, ix, y)

                bx = ix + label_w
                bh = max(2, int(8 * s))
                pygame.draw.rect(surface, (35, 35, 48),
                                 (bx, y, bar_w, bh), border_radius=max(1, int(2 * s)))

                if i >= 12:
                    cval = _clamp(val, -1.0, 1.0)
                    mid_x = bx + bar_w // 2
                    fw = int(abs(cval) * bar_w / 2)
                    bc = (60, 180, 230) if cval >= 0 else (230, 120, 60)
                    if fw > 0:
                        if cval >= 0:
                            pygame.draw.rect(surface, bc, (mid_x, y, fw, bh),
                                             border_radius=max(1, int(2 * s)))
                        else:
                            pygame.draw.rect(surface, bc, (mid_x - fw, y, fw, bh),
                                             border_radius=max(1, int(2 * s)))
                    pygame.draw.line(surface, SEPARATOR, (mid_x, y), (mid_x, y + bh), 1)
                else:
                    cval = _clamp(val, 0.0, 1.0)
                    fw = max(0, int(cval * bar_w))
                    if fw > 0:
                        bc = (int(200 - 150 * cval), int(60 + 170 * cval), 50)
                        pygame.draw.rect(surface, bc, (bx, y, fw, bh),
                                         border_radius=max(1, int(2 * s)))

                self._blit(surface, self._nn_font_nn_sm, f"{val:+.2f}", TEXT,
                           bx + bar_w + int(4 * s), y)
                y += int(12 * s)

        y += int(6 * s)
        self._blit(surface, self._nn_font_nn, "OUTPUTS (2)", ACCENT, ix, y)
        y += int(14 * s)

        if activations and len(activations) > 1:
            outputs = activations[-1]
            label_w_o = int(55 * s)
            val_w_o = int(25 * s)
            bar_w = iw - label_w_o - val_w_o
            for i in range(min(2, len(outputs))):
                val = float(outputs[i])
                lbl = OUTPUT_LABELS[i] if i < len(OUTPUT_LABELS) else f"O{i}"
                c = GOOD if abs(val) > 0.3 else NODE_OFF
                self._blit(surface, self._nn_font_sm, lbl, c, ix, y)

                bx = ix + label_w_o
                bh = max(4, int(14 * s))
                pygame.draw.rect(surface, (35, 35, 48),
                                 (bx, y, bar_w, bh), border_radius=max(1, int(2 * s)))

                mid_x = bx + bar_w // 2
                fw = int(abs(val) * bar_w / 2)
                if val >= 0:
                    pygame.draw.rect(surface, GOOD, (mid_x, y, fw, bh),
                                     border_radius=max(1, int(2 * s)))
                else:
                    pygame.draw.rect(surface, PPO_RED, (mid_x - fw, y, fw, bh),
                                     border_radius=max(1, int(2 * s)))

                pygame.draw.line(surface, TEXT, (mid_x, y), (mid_x, y + bh), 1)

                self._blit(surface, self._nn_font_nn_sm, f"{val:+.3f}", TEXT,
                           bx + bar_w + int(4 * s), y + int(2 * s))
                y += int(22 * s)

    # -- Activation stats panel --------------------------------------------
    def _draw_act_panel(self, surface, x0, y0, w, h, network, activations, s):
        box = pygame.Rect(x0, y0, w, h)
        pygame.draw.rect(surface, DARK_BG, box, border_radius=max(1, int(6 * s)))
        pygame.draw.rect(surface, SEPARATOR, box, 1, border_radius=max(1, int(6 * s)))

        ix = x0 + int(8 * s)
        iw = w - int(16 * s)
        y = y0 + int(6 * s)

        arch = network.get_layer_sizes()
        layer_names = ["In", "FC1", "FC2", "LSTM", "ActFC", "Out"]

        self._blit(surface, self._nn_font_nn, "ACTIVATIONS", ACCENT, ix, y)
        y += int(14 * s)

        for li in range(len(arch)):
            if li >= len(activations):
                break
            act = activations[li]
            if isinstance(act, (int, float)):
                act = [act]
            act_np = np.array(act, dtype=np.float32)
            mean_a = float(np.mean(np.abs(act_np)))
            on_cnt = int(np.sum(act_np > 0.01))

            name = layer_names[li] if li < len(layer_names) else f"L{li}"
            col = PPO_BLUE if name == "LSTM" else ACCENT
            self._blit(surface, self._nn_font_nn_sm, f"{name}[{arch[li]}]",
                       col, ix, y)

            # heatmap
            hm_x = ix + int(52 * s)
            hm_w = iw - int(110 * s)
            hm_h = max(2, int(9 * s))
            num_cells = min(len(act_np), 24)
            cell_w = hm_w / max(1, num_cells)
            for ci in range(num_cells):
                v = min(1.0, abs(float(act_np[ci])))
                cr = int(25 + 210 * v)
                cg = int(25 + 60 * (1 - v))
                cb = 50
                cx_pos = int(hm_x + ci * cell_w)
                cw = max(1, int(cell_w))
                pygame.draw.rect(surface, (cr, cg, cb), (cx_pos, y + 1, cw, hm_h))
            pygame.draw.rect(surface, SEPARATOR, (hm_x, y + 1, hm_w, hm_h), 1)

            self._blit(surface, self._nn_font_nn_sm,
                       f"{mean_a:.2f} {on_cnt}on",
                       TEXT_DIM, hm_x + hm_w + int(4 * s), y + 1)
            y += int(16 * s)

        y += int(6 * s)
        self._blit(surface, self._nn_font_nn, "WEIGHTS", ACCENT, ix, y)
        y += int(14 * s)
        w_stats = network.get_weight_stats()
        for ws in w_stats:
            self._blit(surface, self._nn_font_nn_sm,
                       f"{ws['name']} |W|={ws['mean_w']:.3f} s={ws['std_w']:.3f}",
                       TEXT_DIM, ix, y)
            y += int(13 * s)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _blit(surface, font, text, color, x, y):
        surface.blit(font.render(text, True, color), (int(x), int(y)))

    @staticmethod
    def _hsep(surface, x, y, w):
        pygame.draw.line(surface, SEPARATOR, (int(x), int(y)), (int(x + w), int(y)), 1)

    def update_history(self, fitness, best):
        self.hist_fit.append(fitness)
        self.hist_best.append(best)

    def quit(self):
        if self.has_nn_window:
            try:
                self.nn_sdl_window.destroy()
            except Exception:
                pass
        pygame.quit()
