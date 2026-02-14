"""
CarAI -- LSTM Actor-Critic with PPO
======================================
A single car learns to drive via Proximal Policy Optimization.
Architecture: In(15) > FC(128)x2 > LSTM(128) > Actor(2) + Critic(1)

Inputs:  12 radars + speed + angle_to_checkpoint + curvature
Outputs: steering [-1,+1] + throttle [-1,+1]  (continuous)

Modes:
  python main.py              -- Train mode (PPO)
  python main.py --test       -- Test the saved model
  python main.py --test model.json  -- Test a specific model

Controls:
  SPACE   -- Pause / Resume
  R       -- Radars on/off
  S       -- Save model (train mode)
  N       -- Next test track (test mode)
  Up/Down -- Simulation speed
  ESC     -- Quit
"""
import sys, os
import numpy as np
from track import Track, LEVEL_NAMES, TEST_TRACK_NAMES
from car import Car
from brain import ActorCriticLSTM, PPOTrainer
from gui import GUI, TRACK_W, TRACK_H

LAPS_TO_ADVANCE = 2
MAX_LEVEL       = 5
MODELS_DIR      = "models"
DEFAULT_MODEL   = os.path.join(MODELS_DIR, "best_model.json")
NUM_TEST_TRACKS = 3
ROLLOUT_LEN     = 2048   # steps before PPO update


# =========================================================================
#  TRAIN MODE (PPO)
# =========================================================================
class TrainSimulation:
    def __init__(self):
        self.gui = GUI(test_mode=False)
        self.level = 1
        self.track = Track(TRACK_W, TRACK_H, level=self.level)

        self.network = ActorCriticLSTM()
        self.trainer = PPOTrainer(self.network)
        self.running = True

    def run(self):
        print("=" * 60)
        print("  CarAI -- PPO Training Mode")
        print(f"  Architecture : {self.network.get_architecture_str()}")
        print(f"  Parameters   : {self.network.count_params():,}")
        print(f"  Method       : PPO + LSTM + Continuous Actions")
        print(f"  Rollout      : {ROLLOUT_LEN} steps")
        print("=" * 60)
        print(f"  Level 1 : {LEVEL_NAMES[0]}")
        print("  SPACE=Pause  R=Radars  S=Save  Up/Down=Speed  ESC=Quit")
        print("=" * 60)

        self.network.train()
        global_step = 0

        while self.running:
            self.trainer.episode += 1
            car = Car(self.track)
            hidden = self.network.init_hidden()
            episode_reward = 0.0
            episode_ticks = 0
            level_completed = False

            while car.alive and self.running:
                self.running = self.gui.handle_events()
                if not self.running:
                    break

                if self.gui.save_requested:
                    self.gui.save_requested = False
                    path = self.trainer.save(DEFAULT_MODEL)
                    print(f"  >> Model saved -> {path}")

                if self.gui.paused:
                    acts = self.network.get_activations(car.get_inputs(), hidden)
                    stats = self._stats(car)
                    self.gui.draw(self.track, car, stats, self.network, acts)
                    continue

                for _ in range(self.gui.speed):
                    if not car.alive:
                        break

                    obs = car.get_inputs()
                    action, log_prob, value, new_hidden = self.network.act(obs, hidden)

                    car.apply_action(action)
                    car.update()
                    episode_ticks += 1
                    global_step += 1

                    reward = car.step_reward
                    done = not car.alive
                    episode_reward += reward

                    self.trainer.buffer.push(obs, action, log_prob, reward, value, float(done))
                    hidden = new_hidden

                    # PPO update
                    if len(self.trainer.buffer) >= ROLLOUT_LEN:
                        if car.alive:
                            last_val = self.network.get_value(car.get_inputs(), hidden)
                        else:
                            last_val = 0.0
                        self.trainer.update(last_val)

                    if car.laps >= LAPS_TO_ADVANCE:
                        level_completed = True
                        break

                if level_completed:
                    break

                if car.alive:
                    acts = self.network.get_activations(car.get_inputs(), hidden)
                else:
                    acts = None
                stats = self._stats(car)
                self.gui.draw(self.track, car, stats, self.network, acts)

            # end of episode: flush remaining buffer
            if len(self.trainer.buffer) > 0:
                if car.alive:
                    last_val = self.network.get_value(car.get_inputs(), hidden)
                else:
                    last_val = 0.0
                self.trainer.update(last_val)

            fitness = car.fitness
            self.trainer.recent_rewards.append(episode_reward)

            if fitness > self.trainer.best_fitness:
                self.trainer.best_fitness = fitness
                self.trainer.improvements += 1
                tag = "<< IMPROVED"
            else:
                tag = ""

            self.gui.update_history(fitness, self.trainer.best_fitness)

            ep = self.trainer.episode
            avg_r = np.mean(self.trainer.recent_rewards[-30:])
            print(
                f"  Ep.{ep:>4d} | Lv.{self.level} | "
                f"Fit:{fitness:>8.0f} | Best:{self.trainer.best_fitness:>8.0f} | "
                f"R:{episode_reward:>7.1f} | AvgR:{avg_r:>7.1f} | "
                f"CP:{car.cp_passed} Laps:{car.laps} T:{episode_ticks}  {tag}"
            )

            if level_completed:
                self._advance_level()

            # auto-save every 25 episodes
            if ep % 25 == 0:
                self.trainer.save(DEFAULT_MODEL)

        # auto-save on quit
        path = self.trainer.save(DEFAULT_MODEL)
        print(f"\n  Auto-saved model -> {path}")
        self.gui.quit()

    def _advance_level(self):
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
        self.network, self.model_info = PPOTrainer.load(model_path)
        self.network.eval()

        self.test_id = 1
        self.track = Track(TRACK_W, TRACK_H, test_track=self.test_id)
        self.running = True
        self.attempt = 0

    def run(self):
        info = self.model_info
        print("=" * 60)
        print("  CarAI -- Test Mode (PPO + LSTM)")
        print(f"  Model        : {self.model_path}")
        print(f"  Architecture : {self.network.get_architecture_str()}")
        print(f"  Trained ep.  : {info.get('episode', '?')}")
        print(f"  Train fitness: {info.get('best_fitness', '?'):.0f}")
        print("=" * 60)
        print(f"  Testing on {NUM_TEST_TRACKS} unseen circuits")
        print(f"  Track 1 : {self.track.level_name}")
        print("  SPACE=Pause  R=Radars  N=Next Track  ESC=Quit")
        print("=" * 60)

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
        hidden = self.network.init_hidden()

        max_ticks = 5000
        tick = 0

        while tick < max_ticks and car.alive and self.running:
            self.running = self.gui.handle_events()
            if not self.running:
                break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_n]:
                self._next_track()
                return

            if self.gui.paused:
                acts = self.network.get_activations(car.get_inputs(), hidden)
                stats = self._test_stats(car)
                self.gui.draw(self.track, car, stats, self.network, acts)
                continue

            for _ in range(self.gui.speed):
                if not car.alive:
                    break
                tick += 1
                obs = car.get_inputs()
                action, _, new_hidden = self.network.act_deterministic(obs, hidden)
                hidden = new_hidden
                car.apply_action(action)
                car.update()

            if car.alive:
                acts = self.network.get_activations(car.get_inputs(), hidden)
            else:
                acts = None
            stats = self._test_stats(car)
            self.gui.draw(self.track, car, stats, self.network, acts)

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
            "improvements": 0,
            "params_count": self.network.count_params(),
            "total_updates": self.model_info.get("total_updates", 0),
            "avg_reward": 0.0,
            "pg_loss": 0.0,
            "v_loss": 0.0,
            "entropy": 0.0,
            "lr": 0.0,
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
