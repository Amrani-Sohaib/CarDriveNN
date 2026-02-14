"""
CarAI -- Multilayer Feedforward Neural Network
================================================
A single car learns to drive via Evolutionary Strategy.
Architecture: 5 -> 64 -> 48 -> 32 -> 16 -> 4

Controls:
  SPACE   -- Pause / Resume
  R       -- Radars on/off
  Up/Down -- Simulation speed
  ESC     -- Quit
"""
from track import Track, LEVEL_NAMES
from car import Car
from brain import NeuralNetwork, EvolutionaryTrainer
from gui import GUI, TRACK_W, TRACK_H

LAPS_TO_ADVANCE = 2
MAX_LEVEL       = 5


class Simulation:
    def __init__(self):
        self.gui = GUI()
        self.level = 1
        self.track = Track(TRACK_W, TRACK_H, level=self.level)

        self.nn = NeuralNetwork()
        self.trainer = EvolutionaryTrainer(self.nn)
        self.running = True

    def run(self):
        arch = NeuralNetwork.ARCHITECTURE
        print("=" * 56)
        print("  CarAI -- Multilayer Feedforward Network")
        print(f"  Architecture : {' -> '.join(map(str, arch))}")
        print(f"  Parameters   : {self.nn.count_params():,}")
        print(f"  Method       : Evolutionary Strategy (ES)")
        print("=" * 56)
        print(f"  Level 1 : {LEVEL_NAMES[0]}")
        print("  SPACE=Pause  R=Radars  Up/Down=Speed  ESC=Quit")
        print("=" * 56)

        while self.running:
            self._run_episode()

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

    def _advance_level(self):
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


if __name__ == "__main__":
    Simulation().run()
