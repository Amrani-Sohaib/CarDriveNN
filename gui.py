"""
GUI module -- Two compact windows side-by-side:
  Window 1 (left):  Track + Car + Stats panel
  Window 2 (right): Dedicated NN Visualization
Adapted for LSTM + PPO + continuous actions.
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

# -- Layout constants ------------------------------------------------------
TRACK_W, TRACK_H = 900, 620
PANEL_W = 220
WIN1_W  = TRACK_W + PANEL_W   # 1120
WIN1_H  = TRACK_H             # 620
NN_WIN_W, NN_WIN_H = 560, 620


class GUI:
    """Compact dual-window GUI for LSTM-PPO."""

    def __init__(self, test_mode=False):
        pygame.init()
        self.test_mode = test_mode
        title = "CarAI -- Test" if test_mode else "CarAI -- Track"
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((WIN1_W, WIN1_H))
        self.clock = pygame.time.Clock()

        # second window
        self.nn_w, self.nn_h = NN_WIN_W, NN_WIN_H
        self._create_nn_window()

        # fonts
        self.font_title = pygame.font.SysFont("Menlo", 18, bold=True)
        self.font_md    = pygame.font.SysFont("Menlo", 13, bold=True)
        self.font_sm    = pygame.font.SysFont("Menlo", 11)
        self.font_xs    = pygame.font.SysFont("Menlo", 10)
        self.font_nn    = pygame.font.SysFont("Menlo", 11, bold=True)
        self.font_nn_sm = pygame.font.SysFont("Menlo", 9)

        self.panel_rect = pygame.Rect(TRACK_W, 0, PANEL_W, WIN1_H)
        self.fps = 60
        self.show_radars = True
        self.paused = False
        self.speed = 1
        self.save_requested = False
        self.save_flash = 0

        self.hist_fit  = []
        self.hist_best = []

    def _create_nn_window(self):
        try:
            self.nn_sdl_window = SDLWindow(
                "CarAI -- Neural Network (LSTM)",
                size=(self.nn_w, self.nn_h),
            )
            self.nn_renderer = Renderer(self.nn_sdl_window)
            self.nn_buffer = pygame.Surface((self.nn_w, self.nn_h))
            self.has_nn_window = True
        except Exception:
            self.has_nn_window = False
            self.nn_buffer = pygame.Surface((self.nn_w, self.nn_h))

    # -- events ------------------------------------------------------------
    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:  return False
                if ev.key == pygame.K_r:       self.show_radars = not self.show_radars
                if ev.key == pygame.K_SPACE:   self.paused = not self.paused
                if ev.key == pygame.K_UP:      self.speed = min(20, self.speed + 1)
                if ev.key == pygame.K_DOWN:    self.speed = max(1, self.speed - 1)
                if ev.key == pygame.K_s and not self.test_mode:
                    self.save_requested = True
                    self.save_flash = 90
        return True

    # -- main draw ---------------------------------------------------------
    def draw(self, track, car, stats, network=None, activations=None):
        self.screen.fill(BG)

        # track area
        clip = pygame.Rect(0, 0, TRACK_W, TRACK_H)
        self.screen.set_clip(clip)
        track.draw(self.screen)
        track.draw_checkpoints(self.screen)
        if car:
            car.draw(self.screen, show_radars=self.show_radars)
        self.screen.set_clip(None)

        # panel
        self._panel(car, stats)
        pygame.display.flip()

        # NN window
        if network and activations:
            self._draw_nn_window(network, activations, car, stats)

        self.clock.tick(self.fps)

    # =====================================================================
    #  PANEL
    # =====================================================================
    def _panel(self, car, st):
        px = TRACK_W
        pw = PANEL_W
        ph = WIN1_H
        pad = 8

        pygame.draw.rect(self.screen, PANEL_BG, self.panel_rect)
        pygame.draw.line(self.screen, ACCENT, (px, 0), (px, ph), 2)

        x = px + pad
        right = px + pw - pad
        usable = pw - 2 * pad
        y = pad

        # title
        self._blit(self.screen, self.font_title, "CarAI", ACCENT, x, y)
        self._blit(self.screen, self.font_nn_sm, "PPO+LSTM", PPO_BLUE, x + 60, y + 5)
        y += 22
        self._hsep(x, y, usable); y += 6

        # level
        level = st.get("level", 1)
        lname = st.get("level_name", "")
        self._blit(self.screen, self.font_sm, lname, GOLD, x, y); y += 16
        bw = (usable - 4 * 3) // 5
        for i in range(5):
            r = pygame.Rect(x + i * (bw + 3), y, bw, 8)
            c = LEVEL_COLORS[i] if i < level else (35, 35, 48)
            pygame.draw.rect(self.screen, c, r, border_radius=2)
        y += 16
        self._hsep(x, y, usable); y += 6

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

        for label, val, col in stat_rows:
            self._blit(self.screen, self.font_xs, label, TEXT_DIM, x, y)
            vr = self.font_xs.render(val, True, col)
            self.screen.blit(vr, (right - vr.get_width(), y))
            y += 15

        y += 4
        self._hsep(x, y, usable); y += 6

        # PPO stats
        self._blit(self.screen, self.font_md, "PPO", PPO_BLUE, x, y); y += 16
        ppo_rows = [
            ("PG Loss",  f"{st.get('pg_loss', 0):.4f}",  PPO_RED),
            ("V Loss",   f"{st.get('v_loss', 0):.4f}",   PPO_RED),
            ("Entropy",  f"{st.get('entropy', 0):.4f}",  PPO_GREEN),
            ("Updates",  f"{st.get('total_updates', 0)}", TEXT_DIM),
            ("LR",       f"{st.get('lr', 3e-4):.1e}",    TEXT_DIM),
        ]
        for label, val, col in ppo_rows:
            self._blit(self.screen, self.font_xs, label, TEXT_DIM, x, y)
            vr = self.font_xs.render(val, True, col)
            self.screen.blit(vr, (right - vr.get_width(), y))
            y += 14

        y += 4
        self._hsep(x, y, usable); y += 6

        # car info
        if car:
            self._blit(self.screen, self.font_md, "CAR", ACCENT, x, y); y += 16
            car_rows = [
                ("Checkpts", f"{car.cp_passed}"),
                ("Laps",     f"{car.laps}"),
                ("Speed",    f"{car.speed:.1f}"),
                ("Distance", f"{car.dist:.0f}"),
            ]
            for label, val in car_rows:
                self._blit(self.screen, self.font_xs, label, TEXT_DIM, x, y)
                vr = self.font_xs.render(val, True, TEXT)
                self.screen.blit(vr, (right - vr.get_width(), y))
                y += 15
            y += 4
            self._hsep(x, y, usable); y += 6

        # learning curve
        self._blit(self.screen, self.font_md, "LEARNING", ACCENT, x, y); y += 16
        gw = usable
        gh = 60
        gr = pygame.Rect(x, y, gw, gh)
        pygame.draw.rect(self.screen, DARK_BG, gr, border_radius=3)
        pygame.draw.rect(self.screen, SEPARATOR, gr, 1, border_radius=3)

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
                pygame.draw.lines(self.screen, GOLD, False, pts, 2)
            if len(self.hist_fit) > 1:
                data_c = self.hist_fit[-60:]
                pts_c = []
                for i, v in enumerate(data_c):
                    ppx = x + 3 + i * (gw - 6) / max(1, len(data_c) - 1)
                    ppy = y + gh - 3 - (v / mx) * (gh - 10)
                    pts_c.append((ppx, ppy))
                if len(pts_c) >= 2:
                    pygame.draw.lines(self.screen, (70, 70, 110), False, pts_c, 1)
            self._blit(self.screen, self.font_nn_sm, f"{mx:.0f}", TEXT_DIM, x + 2, y + 1)

        y += gh + 6

        # controls
        self._hsep(x, y, usable); y += 4
        if self.test_mode:
            for t in ["SPC:Pause R:Radar", "Up/Dn:Speed N:Next", "ESC:Quit"]:
                self._blit(self.screen, self.font_nn_sm, t, TEXT_DIM, x, y)
                y += 12
            y += 4
            self._blit(self.screen, self.font_md, "TEST MODE", PPO_BLUE, x, y)
        else:
            for t in ["SPC:Pause R:Radar S:Save", "Up/Dn:Speed ESC:Quit"]:
                self._blit(self.screen, self.font_nn_sm, t, TEXT_DIM, x, y)
                y += 12

        # save flash
        if self.save_flash > 0:
            self.save_flash -= 1
            alpha = min(255, self.save_flash * 4)
            flash_s = pygame.Surface((TRACK_W, 30), pygame.SRCALPHA)
            flash_s.fill((0, 180, 80, min(200, alpha)))
            self.screen.blit(flash_s, (0, 0))
            self._blit(self.screen, self.font_md, "Model saved!", (255, 255, 255), 10, 6)

    # =====================================================================
    #  WINDOW 2 -- Neural Network
    # =====================================================================
    def _draw_nn_window(self, network, activations, car, stats):
        s = self.nn_buffer
        s.fill(BG)

        pad = 10

        # title
        pygame.draw.rect(s, PANEL_BG, (0, 0, self.nn_w, 30))
        pygame.draw.line(s, ACCENT, (0, 30), (self.nn_w, 30), 1)
        self._blit(s, self.font_title, "LSTM Actor-Critic", ACCENT, pad, 5)

        # architecture string
        arch_str = network.get_architecture_str()
        total_p = network.count_params()
        self._blit(s, self.font_nn_sm, f"{arch_str}  [{total_p:,}p]",
                   TEXT_DIM, pad, 34)

        # NN diagram
        nn_y = 50
        nn_h = 280
        self._draw_nn_diagram(s, pad, nn_y, self.nn_w - 2 * pad, nn_h,
                              network, activations)

        # bottom panels
        bot_y = nn_y + nn_h + 8
        bot_h = self.nn_h - bot_y - pad
        half_w = (self.nn_w - 3 * pad) // 2

        self._draw_io_panel(s, pad, bot_y, half_w, bot_h, activations, car)
        self._draw_act_panel(s, pad * 2 + half_w, bot_y, half_w, bot_h,
                             network, activations)

        if self.has_nn_window:
            tex = Texture.from_surface(self.nn_renderer, s)
            self.nn_renderer.clear()
            tex.draw()
            self.nn_renderer.present()
            del tex

    # -- NN diagram --------------------------------------------------------
    def _draw_nn_diagram(self, surface, x0, y0, w, h, network, activations):
        box = pygame.Rect(x0, y0, w, h)
        pygame.draw.rect(surface, DARK_BG, box, border_radius=6)
        pygame.draw.rect(surface, SEPARATOR, box, 1, border_radius=6)

        arch = network.get_layer_sizes()
        n_layers = len(arch)

        draw_x0 = x0 + 45
        draw_x1 = x0 + w - 50
        draw_y0 = y0 + 22
        draw_y1 = y0 + h - 20

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
                    self._blit_s(surface, self.font_xs, "..", TEXT_DIM, nx - 5, ny - 4)
                    continue

                clamped = max(0.0, min(1.0, abs(val)))
                r = int(40 + 160 * (1 - clamped))
                g = int(40 + 210 * clamped)
                b = int(50 + 30 * clamped)
                color = (r, g, b)
                rad = 6 if is_io else 4

                pygame.draw.circle(surface, color, (int(nx), int(ny)), rad)
                if clamped > 0.5:
                    outline_c = (min(255, int(r * 1.3)), min(255, int(g * 1.2)),
                                 min(255, int(b * 1.2)))
                    pygame.draw.circle(surface, outline_c, (int(nx), int(ny)), rad, 1)
                else:
                    pygame.draw.circle(surface, SEPARATOR, (int(nx), int(ny)), rad, 1)

        # layer labels at top
        layer_names = ["In", "FC1", "FC2", "LSTM", "ActFC", "Out"]
        for i in range(n_layers):
            lx = layer_xs[i]
            s_txt = str(arch[i])
            tw = self.font_nn_sm.size(s_txt)[0]
            self._blit_s(surface, self.font_nn_sm, s_txt, ACCENT, lx - tw // 2, y0 + 4)

            if i < len(layer_names):
                lbl = layer_names[i]
                tw2 = self.font_nn_sm.size(lbl)[0]
                col = PPO_BLUE if lbl == "LSTM" else TEXT_DIM
                self._blit_s(surface, self.font_nn_sm, lbl, col, lx - tw2 // 2, y0 + h - 15)

        # output labels
        for idx, (nx, ny, val) in enumerate(node_positions[-1]):
            if val == -1:
                continue
            if idx < len(OUTPUT_LABELS):
                lbl = OUTPUT_LABELS[idx]
                c = GOOD if abs(val) > 0.3 else NODE_OFF
                self._blit_s(surface, self.font_nn_sm, lbl, c, nx + 10, ny - 9)
                self._blit_s(surface, self.font_nn_sm, f"{val:+.2f}", c, nx + 10, ny + 1)

    # -- I/O panel ---------------------------------------------------------
    def _draw_io_panel(self, surface, x0, y0, w, h, activations, car):
        box = pygame.Rect(x0, y0, w, h)
        pygame.draw.rect(surface, DARK_BG, box, border_radius=6)
        pygame.draw.rect(surface, SEPARATOR, box, 1, border_radius=6)

        ix = x0 + 8
        iw = w - 16
        y = y0 + 6

        self._blit_s(surface, self.font_nn, "INPUTS (15)", ACCENT, ix, y); y += 14

        if activations and len(activations) > 0:
            inputs = activations[0]
            bar_w = iw - 70
            for i in range(min(15, len(inputs))):
                val = float(inputs[i])
                lbl = INPUT_LABELS[i] if i < len(INPUT_LABELS) else f"I{i}"
                self._blit_s(surface, self.font_nn_sm, lbl, TEXT_DIM, ix, y)

                bx = ix + 46
                bh = 8
                pygame.draw.rect(surface, (35, 35, 48), (bx, y, bar_w, bh), border_radius=2)

                # for angle/curvature inputs (can be negative), show centered bar
                if i >= 12:
                    mid_x = bx + bar_w // 2
                    fw = int(abs(val) * bar_w / 2)
                    bc = (60, 180, 230) if val >= 0 else (230, 120, 60)
                    if val >= 0:
                        pygame.draw.rect(surface, bc, (mid_x, y, fw, bh), border_radius=2)
                    else:
                        pygame.draw.rect(surface, bc, (mid_x - fw, y, fw, bh), border_radius=2)
                    pygame.draw.line(surface, SEPARATOR, (mid_x, y), (mid_x, y + bh), 1)
                else:
                    fw = max(0, int(val * bar_w))
                    if fw > 0:
                        bc = (int(200 - 150 * val), int(60 + 170 * val), 50)
                        pygame.draw.rect(surface, bc, (bx, y, fw, bh), border_radius=2)

                self._blit_s(surface, self.font_nn_sm, f"{val:+.2f}", TEXT,
                             bx + bar_w + 4, y)
                y += 12

        y += 6
        self._blit_s(surface, self.font_nn, "OUTPUTS (2)", ACCENT, ix, y); y += 14

        if activations and len(activations) > 1:
            outputs = activations[-1]
            bar_w = iw - 80
            for i in range(min(2, len(outputs))):
                val = float(outputs[i])
                lbl = OUTPUT_LABELS[i] if i < len(OUTPUT_LABELS) else f"O{i}"
                c = GOOD if abs(val) > 0.3 else NODE_OFF
                self._blit_s(surface, self.font_sm, lbl, c, ix, y)

                bx = ix + 55
                bh = 14
                pygame.draw.rect(surface, (35, 35, 48), (bx, y, bar_w, bh), border_radius=2)

                # centered bar (value can be -1 to +1)
                mid_x = bx + bar_w // 2
                fw = int(abs(val) * bar_w / 2)
                if val >= 0:
                    pygame.draw.rect(surface, GOOD, (mid_x, y, fw, bh), border_radius=2)
                else:
                    pygame.draw.rect(surface, PPO_RED, (mid_x - fw, y, fw, bh), border_radius=2)

                # center line
                pygame.draw.line(surface, TEXT, (mid_x, y), (mid_x, y + bh), 1)

                self._blit_s(surface, self.font_nn_sm, f"{val:+.3f}", TEXT,
                             bx + bar_w + 4, y + 2)
                y += 22

    # -- Activation stats panel --------------------------------------------
    def _draw_act_panel(self, surface, x0, y0, w, h, network, activations):
        box = pygame.Rect(x0, y0, w, h)
        pygame.draw.rect(surface, DARK_BG, box, border_radius=6)
        pygame.draw.rect(surface, SEPARATOR, box, 1, border_radius=6)

        ix = x0 + 8
        iw = w - 16
        y = y0 + 6

        arch = network.get_layer_sizes()
        layer_names = ["In", "FC1", "FC2", "LSTM", "ActFC", "Out"]

        self._blit_s(surface, self.font_nn, "ACTIVATIONS", ACCENT, ix, y); y += 14

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
            self._blit_s(surface, self.font_nn_sm, f"{name}[{arch[li]}]", col, ix, y)

            # heatmap
            hm_x = ix + 52
            hm_w = iw - 110
            hm_h = 9
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

            self._blit_s(surface, self.font_nn_sm,
                         f"{mean_a:.2f} {on_cnt}on",
                         TEXT_DIM, hm_x + hm_w + 4, y + 1)
            y += 16

        y += 6
        self._blit_s(surface, self.font_nn, "WEIGHTS", ACCENT, ix, y); y += 14
        w_stats = network.get_weight_stats()
        for ws in w_stats:
            self._blit_s(surface, self.font_nn_sm,
                         f"{ws['name']} |W|={ws['mean_w']:.3f} s={ws['std_w']:.3f}",
                         TEXT_DIM, ix, y)
            y += 13

    # -- helpers -----------------------------------------------------------
    def _blit(self, surface, font, text, color, x, y):
        surface.blit(font.render(text, True, color), (x, y))

    def _blit_s(self, surface, font, text, color, x, y):
        """Blit on any surface (not just self.screen)."""
        surface.blit(font.render(text, True, color), (x, y))

    def _hsep(self, x, y, w):
        pygame.draw.line(self.screen, SEPARATOR, (x, y), (x + w, y), 1)

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
