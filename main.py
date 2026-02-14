"""
CarAI -- Multilayer Feedforward Neural Network
================================================
A single car learns to drive via Evolutionary Strategy.
Architecture: 5 -> 64 -> 64 -> 48 -> 48 -> 32 -> 16 -> 4

Modes:
  python main.py              -- Train mode (with auto-save)
  python main.py --test       -- Test the saved model on 3 new circuits
  python main.py --test model.json  -- Test a specific model file

Controls:
  SPACE   -- Pause / Resume
  R       -- Radars on/off
  S       -- Save model (train mode only)
  N       -- Next test track (test mode only)
  Up/Down -- Simulation speed
  ESC     -- Quit
"""
import sys, os
from track import Track, LEVEL_NAMES, TEST_TRACK_NAMES
from car import Car
from brain import NeuralNetwork, EvolutionaryTrainer
from gui import GUI, TRACK_W, TRACK_H

LAPS_TO_ADVANCE = 2
MAX_LEVEL       = 5
MODELS_DIR      = "models"
DEFAULT_MODEL   = os.path.join(MODELS_DIR, "best_model.json")
NUM_TEST_TRACKS = 3


# =========================================================================
#  TRAIN MODE
# =========================================================================
class TrainSimulation:
    def __init__(self):
        self.gui = GUI(test_mode=False)
        self.level = 1
        self.track = Track(TRACK_W, TRACK_H, level=self.level)

        self.nn = NeuralNetwork()
        self.trainer = EvolutionaryTrainer(self.nn)
        self.running = True

    def run(self):
        arch = NeuralNetwork.ARCHITECTURE
        print("=" * 56)
        print("  CarAI -- Training Mode")
        print(f"  Architecture : {' -> '.join(map(str, arch))}")
        print(f"  Parameters   : {self.nn.count_params():,}")
        print(f"  Method       : Evolutionary Strategy (ES)")
        print("=" * 56)
        print(f"  Level 1 : {LEVEL_NAMES[0]}")
        print("  SPACE=Pause  R=Radars  S=Save  Up/Down=Speed  ESC=Quit")
        print("=" * 56)

        while self.running:
            self._run_episode()

        # auto-save on quit
        path = self.trainer.save(DEFAULT_MODEL)
        print(f"\n  Auto-saved model -> {path}")

        self.gui.quit()

    def _run_episode(self):
        self.trainer.start_episode()
        car = Car(self.track)

        max_ticks = 2500
        tick = 0
        level_completed = False

        while tick < max_ticks and car.alive and self.running:
            self.running = self.gui.handle_events()
            if not self.running:
                break

            # handle save request
            if self.gui.save_requested:
                self.gui.save_requested = False
                path = self.trainer.save(DEFAULT_MODEL)
                print(f"  >> Model saved -> {path}")

            if self.gui.paused:
                activations = self.nn.get_layer_activations(car.get_inputs())
                stats = self._stats(car)
                self.gui.draw(self.track, car, stats, self.nn, activations)
                continue

            for _ in range(self.gui.speed):
                if not car.alive:
                    break
                tick += 1
                inputs = car.get_inputs()
                outputs = self.nn.forward(inputs)
                car.apply_action(outputs)
                car.update()
                if car.laps >= LAPS_TO_ADVANCE:
                    level_completed = True
                    break

            if level_completed:
                break

            activations = self.nn.get_layer_activations(car.get_inputs()) if car.alive else None
            stats = self._stats(car)
            self.gui.draw(self.track, car, stats, self.nn, activations)

        fitness = car.fitness
        improved = self.trainer.end_episode(fitness)
        self.gui.update_history(fitness, self.trainer.best_fitness)

        ep = self.trainer.episode
        tag = "<< IMPROVED" if improved else ""
        print(
            f"  Ep.{ep:>4d} | Lv.{self.level} | "
            f"Fit:{fitness:>8.0f} | Best:{self.trainer.best_fitness:>8.0f} | "
            f"s={self.trainer.noise_std:.4f} | CP:{car.cp_passed} Laps:{car.laps}  {tag}"
        )

        if level_completed:
            self._advance_level()

        # auto-save every 50 episodes
        if ep % 50 == 0:
            self.trainer.save(DEFAULT_MODEL)

    def _advance_level(self):
        # save on level completion
        path = self.trainer.save(DEFAULT_MODEL)
        print(f"  >> Model saved on level completion -> {path}")

        if self.level < MAX_LEVEL:
            self.level += 1
            self.track = Track(TRACK_W, TRACK_H, level=self.level)
            self.trainer.best_fitness = -1e9
            self.gui.hist_fit.clear()
            self.gui.hist_best.clear()
            print("\n" + "--- " * 15)
            print(f"  LEVEL {self.level} -- {self.track.level_name}")
            print("--- " * 15 + "\n")
        else:
            print("\n" + "=== " * 15)
            print("  ALL LEVELS COMPLETED!")
            print("=== " * 15 + "\n")

    def _stats(self, car):
        s = self.trainer.get_stats()
        s["level"] = self.level
        s["level_name"] = self.track.level_name
        s["current_fitness"] = car.fitness
        return s


# =========================================================================
#  TEST MODE
# =========================================================================
class TestSimulation:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.gui = GUI(test_mode=True)
        self.nn, self.model_info = EvolutionaryTrainer.load(model_path)

        self.test_id = 1
        self.track = Track(TRACK_W, TRACK_H, test_track=self.test_id)
        self.running = True
        self.attempt = 0

    def run(self):
        arch = NeuralNetwork.ARCHITECTURE
        info = self.model_info
        print("=" * 56)
        print("  CarAI -- Test Mode")
        print(f"  Model        : {self.model_path}")
        print(f"  Architecture : {' -> '.join(map(str, arch))}")
        print(f"  Trained ep.  : {info.get('episode', '?')}")
        print(f"  Train fitness: {info.get('best_fitness', '?'):.0f}")
        print("=" * 56)
        print(f"  Testing on {NUM_TEST_TRACKS} unseen circuits")
        print(f"  Track 1 : {self.track.level_name}")
        print("  SPACE=Pause  R=Radars  N=Next Track  ESC=Quit")
        print("=" * 56)

        while self.running:
            self._run_test_episode()

        self.gui.quit()

    def _next_track(self):
        self.test_id = (self.test_id % NUM_TEST_TRACKS) + 1
        self.track = Track(TRACK_W, TRACK_H, test_track=self.test_id)
        self.attempt = 0
        self.gui.hist_fit.clear()
        self.gui.hist_best.clear()
        print(f"\n  >> Switched to {self.track.level_name}")

    def _run_test_episode(self):
        self.attempt += 1
        car = Car(self.track)
        # load the best weights directly (no noise)
        self.nn.set_params(self.nn.get_params())

        max_ticks = 5000
        tick = 0

        while tick < max_ticks and car.alive and self.running:
            self.running = self.gui.handle_events()
            if not self.running:
                break

            # 'N' for next track: check via pygame key state
            keys = pygame.key.get_pressed()
            if keys[pygame.K_n]:
                self._next_track()
                return

            if self.gui.paused:
                activations = self.nn.get_layer_activations(car.get_inputs())
                stats = self._test_stats(car)
                self.gui.draw(self.track, car, stats, self.nn, activations)
                continue

            for _ in range(self.gui.speed):
                if not car.alive:
                    break
                tick += 1
                inputs = car.get_inputs()
                outputs = self.nn.forward(inputs)
                car.apply_action(outputs)
                car.update()

            activations = self.nn.get_layer_activations(car.get_inputs()) if car.alive else None
            stats = self._test_stats(car)
            self.gui.draw(self.track, car, stats, self.nn, activations)

        result = "ALIVE" if car.alive else "CRASHED"
        print(
            f"  Test {self.test_id} | Attempt {self.attempt} | "
            f"{result} | Fit:{car.fitness:>8.0f} | "
            f"CP:{car.cp_passed} Laps:{car.laps} Dist:{car.dist:.0f}"
        )
        self.gui.update_history(car.fitness, car.fitness)

    def _test_stats(self, car):
        return {
            "episode": self.attempt,
            "best_fitness": car.fitness,
            "current_fitness": car.fitness,
            "noise": 0.0,
            "improvements": 0,
            "params_count": self.nn.count_params(),
            "level": self.test_id,
            "level_name": self.track.level_name,
        }


# =========================================================================
#  ENTRY POINT
# =========================================================================
if __name__ == "__main__":
    import pygame

    args = sys.argv[1:]

    if "--test" in args:
        idx = args.index("--test")
        # Check if a model path is provided after --test
        if idx + 1 < len(args) and not args[idx + 1].startswith("-"):
            model_path = args[idx + 1]
        else:
            model_path = DEFAULT_MODEL

        if not os.path.exists(model_path):
            print(f"  Error: model file not found: {model_path}")
            print(f"  Train first with: python main.py")
            sys.exit(1)

        TestSimulation(model_path).run()
    else:
        TrainSimulation().run()
