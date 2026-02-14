"""
Feedforward Neural Network (pure NumPy).
Architecture: 5 -> 64 -> 48 -> 32 -> 16 -> 4
Training via Evolutionary Strategy (ES):
  - Each episode, the car drives until it crashes.
  - If the score improves -> keep the weights.
  - Gaussian noise is applied for exploration.
  - Noise decreases as the model improves.
"""
import numpy as np
import json, os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -- Activation functions -------------------------------------------------
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# -- Neural Network -------------------------------------------------------
class NeuralNetwork:
    """Dense feedforward network with 4 hidden layers."""

    ARCHITECTURE = [5, 64, 64, 48, 48, 32, 16, 4]  # inputs -> hidden -> outputs

    def __init__(self):
        self.layers = []
        self._init_weights()

    # -- Xavier initialization ---------------------------------------------
    def _init_weights(self):
        self.layers = []
        arch = self.ARCHITECTURE
        for i in range(len(arch) - 1):
            fan_in, fan_out = arch[i], arch[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            W = np.random.randn(fan_in, fan_out) * std
            b = np.zeros(fan_out)
            self.layers.append((W, b))

    # -- forward pass ------------------------------------------------------
    def forward(self, inputs: list[float]) -> list[float]:
        """Propagate inputs through the network. Returns 4 outputs."""
        x = np.array(inputs, dtype=np.float64)
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = relu(x)
            else:
                x = sigmoid(x)
        return x.tolist()

    # -- flat parameter access ---------------------------------------------
    def get_params(self) -> np.ndarray:
        """Get all weights and biases as a 1-D vector."""
        parts = []
        for W, b in self.layers:
            parts.append(W.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    def set_params(self, flat: np.ndarray):
        """Load weights from a 1-D vector."""
        idx = 0
        new_layers = []
        arch = self.ARCHITECTURE
        for i in range(len(arch) - 1):
            fan_in, fan_out = arch[i], arch[i + 1]
            w_size = fan_in * fan_out
            W = flat[idx:idx + w_size].reshape(fan_in, fan_out)
            idx += w_size
            b = flat[idx:idx + fan_out].copy()
            idx += fan_out
            new_layers.append((W, b))
        self.layers = new_layers

    def count_params(self) -> int:
        return sum(W.size + b.size for W, b in self.layers)

    def copy(self) -> "NeuralNetwork":
        nn = NeuralNetwork.__new__(NeuralNetwork)
        nn.layers = [(W.copy(), b.copy()) for W, b in self.layers]
        return nn

    # -- layer info (for visualization) ------------------------------------
    def get_layer_activations(self, inputs: list[float]):
        """Return activations of EACH layer (for visualization)."""
        activations = [np.array(inputs)]
        x = np.array(inputs, dtype=np.float64)
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = relu(x)
            else:
                x = sigmoid(x)
            activations.append(x.copy())
        return activations

    def get_weight_stats(self):
        """Return per-layer weight statistics for visualization."""
        stats = []
        for i, (W, b) in enumerate(self.layers):
            stats.append({
                "mean_w": float(np.mean(np.abs(W))),
                "max_w": float(np.max(np.abs(W))),
                "mean_b": float(np.mean(np.abs(b))),
                "std_w": float(np.std(W)),
            })
        return stats


# -- Evolutionary Strategy ------------------------------------------------
class EvolutionaryTrainer:
    """
    Train a neural network via Evolutionary Strategy (ES).
    - Keep the best weights found.
    - Explore by adding Gaussian noise.
    - Reduce noise when the score improves (annealing).
    """

    def __init__(self, network: NeuralNetwork):
        self.network = network
        self.best_params = network.get_params().copy()
        self.best_fitness = -1e9
        self.current_fitness = 0

        # Hyperparameters
        self.noise_std = 0.8
        self.noise_min = 0.005
        self.noise_decay = 0.999
        self.noise_boost = 1.5
        self.stagnation = 0
        self.stagnation_limit = 40

        # Statistics
        self.episode = 0
        self.improvements = 0
        self.history_fitness = []
        self.history_best = []
        self.learning_rate_history = []

    def start_episode(self):
        """Prepare a new episode: apply noise to best weights."""
        self.episode += 1
        noise = np.random.randn(len(self.best_params)) * self.noise_std
        candidate = self.best_params + noise
        self.network.set_params(candidate)
        self.current_fitness = 0

    def end_episode(self, fitness: float):
        """End of episode: keep if better, otherwise roll back."""
        self.current_fitness = fitness
        self.history_fitness.append(fitness)

        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_params = self.network.get_params().copy()
            self.improvements += 1
            improved = True
            self.stagnation = 0
            self.noise_std = min(0.6, self.noise_std * self.noise_boost)
        else:
            self.network.set_params(self.best_params.copy())
            improved = False
            self.stagnation += 1

        # Anti-stagnation
        if self.stagnation >= self.stagnation_limit:
            self.noise_std = min(1.0, self.noise_std * 3.0)
            self.stagnation = 0

        self.history_best.append(self.best_fitness)
        self.learning_rate_history.append(self.noise_std)

        self.noise_std = max(self.noise_min, self.noise_std * self.noise_decay)

        return improved

    def get_stats(self) -> dict:
        return {
            "episode": self.episode,
            "best_fitness": self.best_fitness,
            "current_fitness": self.current_fitness,
            "noise": self.noise_std,
            "improvements": self.improvements,
            "params_count": self.network.count_params(),
        }
