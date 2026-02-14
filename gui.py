"""
GUI module -- Two compact windows side-by-side:
  Window 1 (left):  Track + Car + Stats panel
  Window 2 (right): Dedicated NN Visualization
No emojis anywhere. Crisp rendering. Proper border respect.
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
ACTIVE_GLOW = (0, 255, 180)

LEVEL_COLORS = [
    (0, 200, 100), (170, 210, 0), (255, 170, 0),
    (255, 70, 0),  (220, 25, 25),
]

INPUT_LABELS  = ["L 90", "L 45", "Front", "R 45", "R 90"]
OUTPUT_LABELS = ["Accel", "Brake", "Left", "Right"]

# -- Layout constants ------------------------------------------------------
TRACK_W, TRACK_H = 900, 620
PANEL_W = 220
WIN1_W  = TRACK_W + PANEL_W   # 1120
WIN1_H  = TRACK_H             # 620
NN_WIN_W, NN_WIN_H = 560, 620


class GUI:
    """Compact dual-window GUI."""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("CarAI -- Track")
        self.screen = pygame.display.set_mode((WIN1_W, WIN1_H))
        self.clock = pygame.time.Clock()

        # second window
        self.nn_w, self.nn_h = NN_WIN_W, NN_WIN_H
        self._create_nn_window()

        # anti-aliased fonts
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

        self.hist_fit  = []
        self.hist_best = []

    def _create_nn_window(self):
        try:
            self.nn_sdl_window = SDLWindow(
                "CarAI -- Neural Network",
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
        return True

    # -- main draw ---------------------------------------------------------
    def draw(self, track, car, stats, network=None, activations=None):
        self.screen.fill(BG)

        # track area (clipped)
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
    #  PANEL (right side of Window 1)
    # =====================================================================
    def _panel(self, car, st):
        px = TRACK_W
        pw = PANEL_W
        ph = WIN1_H
        pad = 8   # inner padding

        pygame.draw.rect(self.screen, PANEL_BG, self.panel_rect)
        pygame.draw.line(self.screen, ACCENT, (px, 0), (px, ph), 2)

        x = px + pad
        right = px + pw - pad
        usable = pw - 2 * pad
        y = pad

        # -- title
        self._blit(self.screen, self.font_title, "CarAI", ACCENT, x, y)
        y += 22
        self._hsep(x, y, usable); y += 6

        # -- level
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

        # -- stats
        ep     = st.get("episode", 0)
        best_f = st.get("best_fitness", 0)
        cur_f  = st.get("current_fitness", 0)
        noise  = st.get("noise", 0)
        imp    = st.get("improvements", 0)
        npar   = st.get("params_count", 0)
        alive_str = "YES" if (car and car.alive) else "DEAD"
        alive_c   = GOOD if (car and car.alive) else BAD

        stat_rows = [
            ("Episode",   f"{ep}",         ACCENT),
            ("Alive",     alive_str,       alive_c),
            ("Fitness",   f"{cur_f:.0f}",  TEXT),
            ("Best",      f"{best_f:.0f}", GOLD),
            ("Improved",  f"{imp}",        GOOD),
            ("Noise",     f"{noise:.4f}",  TEXT),
            ("Params",    f"{npar:,}",     TEXT_DIM),
            ("Speed",     f"x{self.speed}",TEXT),
        ]

        for label, val, col in stat_rows:
            self._blit(self.screen, self.font_xs, label, TEXT_DIM, x, y)
            vr = self.font_xs.render(val, True, col)
            self.screen.blit(vr, (right - vr.get_width(), y))
            y += 15

        y += 4
        self._hsep(x, y, usable); y += 6

        # -- car info
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

        # -- learning curve
        self._blit(self.screen, self.font_md, "LEARNING", ACCENT, x, y); y += 16
        gw = usable
        gh = 70
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
            self._blit(self.screen, self.font_nn_sm, f"{mx:.0f}", TEXT_DIM, x+2, y+1)

        y += gh + 6

        # -- controls
        self._hsep(x, y, usable); y += 4
        for t in ["SPC:Pause R:Radar", "Up/Dn:Speed ESC:Quit"]:
            self._blit(self.screen, self.font_nn_sm, t, TEXT_DIM, x, y)
            y += 12

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
        self._blit(s, self.font_title, "Neural Network", ACCENT, pad, 5)

        # architecture string
        arch = network.ARCHITECTURE
        arch_str = " > ".join(str(a) for a in arch)
        total_p = network.count_params()
        self._blit(s, self.font_nn_sm, f"{arch_str}   [{total_p:,} params]",
                   TEXT_DIM, pad, 34)

        # NN diagram
        nn_y = 50
        nn_h = 310
        self._draw_nn_diagram(s, pad, nn_y, self.nn_w - 2*pad, nn_h, network, activations)

        # bottom panels
        bot_y = nn_y + nn_h + 8
        bot_h = self.nn_h - bot_y - pad
        half_w = (self.nn_w - 3 * pad) // 2

        # inputs & decisions
        self._draw_io_panel(s, pad, bot_y, half_w, bot_h, activations, car)

        # layer activations
        self._draw_act_panel(s, pad * 2 + half_w, bot_y, half_w, bot_h, network, activations)

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

        arch = network.ARCHITECTURE
        n_layers = len(arch)

        inner_pad = 8
        draw_x0 = x0 + 45
        draw_x1 = x0 + w - 50
        draw_y0 = y0 + 22
        draw_y1 = y0 + h - 20

        # layer x positions
        layer_xs = []
        for i in range(n_layers):
            lx = draw_x0 + i * (draw_x1 - draw_x0) / max(1, n_layers - 1)
            layer_xs.append(lx)

        max_show = 14
        node_positions = []

        for li, size in enumerate(arch):
            show = min(size, max_show)
            truncated = size > max_show
            positions = []
            avail = draw_y1 - draw_y0
            spacing = avail / (show + (1 if truncated else 0))
            act = activations[li] if li < len(activations) else np.zeros(size)

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

        # connections (on SRCALPHA surface)
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
                    self._blit(surface, self.font_xs, "..", TEXT_DIM, nx - 5, ny - 4)
                    continue

                clamped = max(0.0, min(1.0, abs(val)))
                # color: dark red/grey -> bright green
                r = int(40 + 160 * (1 - clamped))
                g = int(40 + 210 * clamped)
                b = int(50 + 30 * clamped)
                color = (r, g, b)
                rad = 6 if is_io else 4

                pygame.draw.circle(surface, color, (int(nx), int(ny)), rad)
                # bright outline on strong activations
                if clamped > 0.5:
                    outline_c = (
                        min(255, int(r * 1.3)),
                        min(255, int(g * 1.2)),
                        min(255, int(b * 1.2)),
                    )
                    pygame.draw.circle(surface, outline_c, (int(nx), int(ny)), rad, 1)
                else:
                    pygame.draw.circle(surface, SEPARATOR, (int(nx), int(ny)), rad, 1)

        # input labels (left of first layer)
        for idx, (nx, ny, val) in enumerate(node_positions[0]):
            if val == -1:
                continue
            if idx < len(INPUT_LABELS):
                lbl = INPUT_LABELS[idx]
                tw = self.font_nn_sm.size(lbl)[0]
                self._blit(surface, self.font_nn_sm, lbl, TEXT_DIM, nx - tw - 8, ny - 4)

        # output labels + values (right of last layer)
        for idx, (nx, ny, val) in enumerate(node_positions[-1]):
            if val == -1:
                continue
            if idx < len(OUTPUT_LABELS):
                lbl = OUTPUT_LABELS[idx]
                active = val > 0.5
                c = NODE_ON if active else NODE_OFF
                self._blit(surface, self.font_nn_sm, lbl, c, nx + 10, ny - 9)
                self._blit(surface, self.font_nn_sm, f"{val:.2f}", c, nx + 10, ny + 1)

        # layer size labels at top
        for i in range(n_layers):
            lx = layer_xs[i]
            s_txt = str(arch[i])
            tw = self.font_nn_sm.size(s_txt)[0]
            self._blit(surface, self.font_nn_sm, s_txt, ACCENT, lx - tw // 2, y0 + 4)

        # layer type labels at bottom
        labels = ["In"] + [f"H{i}" for i in range(1, n_layers - 1)] + ["Out"]
        for i, lbl in enumerate(labels):
            lx = layer_xs[i]
            tw = self.font_nn_sm.size(lbl)[0]
            self._blit(surface, self.font_nn_sm, lbl, TEXT_DIM, lx - tw // 2, y0 + h - 15)

    # -- I/O panel (inputs + decisions) ------------------------------------
    def _draw_io_panel(self, surface, x0, y0, w, h, activations, car):
        box = pygame.Rect(x0, y0, w, h)
        pygame.draw.rect(surface, DARK_BG, box, border_radius=6)
        pygame.draw.rect(surface, SEPARATOR, box, 1, border_radius=6)

        ix = x0 + 8
        iw = w - 16
        y = y0 + 6

        self._blit(surface, self.font_nn, "INPUTS", ACCENT, ix, y); y += 16

        if activations and len(activations) > 0:
            inputs = activations[0]
            bar_w = iw - 70
            for i in range(min(5, len(inputs))):
                val = float(inputs[i])
                lbl = INPUT_LABELS[i] if i < len(INPUT_LABELS) else f"R{i}"
                self._blit(surface, self.font_nn_sm, lbl, TEXT_DIM, ix, y + 1)

                bx = ix + 42
                bh = 10
                pygame.draw.rect(surface, (35, 35, 48), (bx, y + 1, bar_w, bh), border_radius=2)
                fw = max(0, int(val * bar_w))
                if fw > 0:
                    bc = (int(200 - 150 * val), int(60 + 170 * val), 50)
                    pygame.draw.rect(surface, bc, (bx, y + 1, fw, bh), border_radius=2)
                self._blit(surface, self.font_nn_sm, f"{val:.2f}", TEXT,
                           bx + bar_w + 4, y + 1)
                y += 15

        y += 8
        self._blit(surface, self.font_nn, "OUTPUTS", ACCENT, ix, y); y += 16

        if activations and len(activations) > 1:
            outputs = activations[-1]
            bar_w = iw - 90
            for i in range(min(4, len(outputs))):
                val = float(outputs[i])
                lbl = OUTPUT_LABELS[i] if i < len(OUTPUT_LABELS) else f"O{i}"
                active = val > 0.5
                dot = ">" if active else "-"
                c = NODE_ON if active else NODE_OFF
                self._blit(surface, self.font_sm, f"{dot} {lbl}", c, ix, y)

                bx = ix + 62
                bh = 11
                pygame.draw.rect(surface, (35, 35, 48), (bx, y + 1, bar_w, bh), border_radius=2)
                fw = max(0, int(val * bar_w))
                if fw > 0:
                    pygame.draw.rect(surface, c, (bx, y + 1, fw, bh), border_radius=2)
                # threshold line at 0.5
                tx = bx + int(0.5 * bar_w)
                pygame.draw.line(surface, (100, 100, 120), (tx, y + 1), (tx, y + 1 + bh), 1)
                self._blit(surface, self.font_nn_sm, f"{val:.3f}", TEXT,
                           bx + bar_w + 4, y + 2)
                y += 18

    # -- Activation stats panel --------------------------------------------
    def _draw_act_panel(self, surface, x0, y0, w, h, network, activations):
        box = pygame.Rect(x0, y0, w, h)
        pygame.draw.rect(surface, DARK_BG, box, border_radius=6)
        pygame.draw.rect(surface, SEPARATOR, box, 1, border_radius=6)

        ix = x0 + 8
        iw = w - 16
        y = y0 + 6

        arch = network.ARCHITECTURE
        self._blit(surface, self.font_nn, "ACTIVATIONS", ACCENT, ix, y); y += 16

        for li in range(len(arch)):
            if li >= len(activations):
                break
            act = activations[li]
            mean_a = float(np.mean(np.abs(act)))
            max_a  = float(np.max(np.abs(act)))
            on_cnt = int(np.sum(np.array(act) > 0.01))

            if li == 0:
                name = "In"
            elif li == len(arch) - 1:
                name = "Out"
            else:
                name = f"H{li}"

            self._blit(surface, self.font_nn_sm, f"{name}[{arch[li]}]", ACCENT, ix, y)

            # heatmap
            hm_x = ix + 52
            hm_w = iw - 110
            hm_h = 9
            num_cells = min(len(act), 24)
            cell_w = hm_w / max(1, num_cells)
            for ci in range(num_cells):
                v = min(1.0, abs(float(act[ci])))
                cr = int(25 + 210 * v)
                cg = int(25 + 60 * (1 - v))
                cb = 50
                cx_pos = int(hm_x + ci * cell_w)
                cw = max(1, int(cell_w))
                pygame.draw.rect(surface, (cr, cg, cb), (cx_pos, y + 1, cw, hm_h))
            pygame.draw.rect(surface, SEPARATOR, (hm_x, y + 1, hm_w, hm_h), 1)

            self._blit(surface, self.font_nn_sm,
                       f"{mean_a:.2f} {on_cnt}on",
                       TEXT_DIM, hm_x + hm_w + 4, y + 1)
            y += 18

        y += 6
        self._blit(surface, self.font_nn, "WEIGHTS", ACCENT, ix, y); y += 14
        w_stats = network.get_weight_stats()
        for li, ws in enumerate(w_stats):
            self._blit(surface, self.font_nn_sm,
                       f"L{li+1} |W|={ws['mean_w']:.3f} s={ws['std_w']:.3f}",
                       TEXT_DIM, ix, y)
            y += 13

    # -- helpers -----------------------------------------------------------
    def _blit(self, surface, font, text, color, x, y):
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
