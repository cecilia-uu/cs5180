"""
env_fixed.py (v3)
------------------
Ginger Agents — CS5180 Spring 2026

Key design:
  - Allow ALL moves (agent must explore to find improvements)
  - Track best layout seen → report best_crossings as final result
  - Small displacement (max 2 units) for precise adjustments
  - Reward: big bonus for new best, small penalty for getting worse
  - Include topology embeddings in state
"""

import gymnasium as gym
import numpy as np
import networkx as nx
import torch
import os
import random
from gymnasium import spaces
from xing import XingLoss

MAX_NODES = 100


class GraphLayoutEnvFixed(gym.Env):
    def __init__(
        self,
        graph_paths,
        embeddings=None,
        max_steps=300,
        move_scale=2.0,
        patience=50,
    ):
        super().__init__()

        self.graph_paths = graph_paths
        self.embeddings = embeddings or {}
        self.max_steps = max_steps
        self.move_scale = move_scale
        self.patience = patience

        # Embedding dim
        self.emb_dim = 0
        if embeddings:
            sample = next(iter(embeddings.values()))
            self.emb_dim = sample["node_emb"].shape[1]

        # Obs: [MAX_NODES * (emb_dim + 2) + 3]
        obs_dim = MAX_NODES * (self.emb_dim + 2) + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: [node_selector, delta_x, delta_y]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.G = None
        self.n_nodes = None
        self.xing_loss = None
        self.coords = None
        self.current_crossings = None
        self.initial_crossings = None
        self.best_crossings = None
        self.best_coords = None
        self.steps = 0
        self.no_improve_steps = 0
        self._node_emb = None

    def _load_graph(self, path):
        G = nx.read_graphml(path)
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        return G

    def _pos_to_coords(self, pos):
        return np.array(
            [[pos[v][0], pos[v][1]] for v in self.G.nodes()], dtype=np.float32
        )

    def _normalize_coords(self, coords):
        c = coords.copy()
        c -= c.mean(axis=0)
        max_abs = np.abs(c).max()
        if max_abs > 1e-6:
            c /= max_abs
        return c

    def _get_obs(self):
        coords_norm = self._normalize_coords(self.coords)

        if self._node_emb is not None:
            emb_np = self._node_emb.numpy()
            per_node = np.concatenate([emb_np, coords_norm], axis=1)
        else:
            per_node = coords_norm

        feat_dim = per_node.shape[1]
        padded = np.zeros((MAX_NODES, feat_dim), dtype=np.float32)
        padded[: self.n_nodes] = per_node
        flat = padded.flatten()

        crossings_norm = self.current_crossings / max(self.initial_crossings, 1)
        best_norm = self.best_crossings / max(self.initial_crossings, 1)
        n_nodes_norm = self.n_nodes / MAX_NODES

        obs = np.concatenate([flat, [crossings_norm, best_norm, n_nodes_norm]])
        return obs.astype(np.float32)

    def _compute_crossings(self, coords):
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        with torch.no_grad():
            return self.xing_loss(coords_tensor).item()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        graph_path = random.choice(self.graph_paths)
        self.G = self._load_graph(graph_path)
        self.n_nodes = self.G.number_of_nodes()
        self.xing_loss = XingLoss(self.G, soft=False)

        pos = nx.nx_agraph.graphviz_layout(self.G, prog="neato")
        self.coords = self._pos_to_coords(pos)

        self.current_crossings = self._compute_crossings(self.coords)
        self.initial_crossings = self.current_crossings
        self.best_crossings = self.current_crossings
        self.best_coords = self.coords.copy()
        self.steps = 0
        self.no_improve_steps = 0

        graph_name = os.path.basename(graph_path)
        if graph_name in self.embeddings:
            self._node_emb = self.embeddings[graph_name]["node_emb"][: self.n_nodes]
        elif self.emb_dim > 0:
            # Graph not in embedding dict — use zeros (same shape so obs dim matches)
            self._node_emb = torch.zeros(self.n_nodes, self.emb_dim)
        else:
            self._node_emb = None

        return self._get_obs(), {}

    def step(self, action):
        # Decode action
        node_idx = int((action[0] + 1) / 2 * self.n_nodes)
        node_idx = np.clip(node_idx, 0, self.n_nodes - 1)
        delta_x = float(action[1]) * self.move_scale
        delta_y = float(action[2]) * self.move_scale

        old_crossings = self.current_crossings

        # Apply move (always accept — agent needs to explore)
        self.coords[node_idx, 0] += delta_x
        self.coords[node_idx, 1] += delta_y
        self.current_crossings = self._compute_crossings(self.coords)

        # ── Reward shaping ──
        crossing_delta = old_crossings - self.current_crossings  # positive = good

        if self.current_crossings < self.best_crossings:
            # NEW BEST — big bonus!
            reward = 20.0 + crossing_delta * 5.0
            self.best_crossings = self.current_crossings
            self.best_coords = self.coords.copy()
            self.no_improve_steps = 0
        elif crossing_delta > 0:
            # Improved but not new best
            reward = crossing_delta * 5.0
            self.no_improve_steps = 0
        elif crossing_delta == 0:
            # No change — small penalty for wasting a step
            reward = -0.1
            self.no_improve_steps += 1
        else:
            # Got worse — penalty proportional to how much worse
            reward = crossing_delta * 2.0  # negative since delta is negative
            self.no_improve_steps += 1

        self.steps += 1

        terminated = bool(self.current_crossings == 0)
        truncated = bool(
            self.steps >= self.max_steps or self.no_improve_steps >= self.patience
        )

        # If episode ends, restore best coords
        if terminated or truncated:
            self.coords = self.best_coords.copy()
            self.current_crossings = self.best_crossings

        info = {
            "crossings": self.current_crossings,
            "best_crossings": self.best_crossings,
            "initial_crossings": self.initial_crossings,
            "improvement": self.initial_crossings - self.best_crossings,
            "steps": self.steps,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        import matplotlib.pyplot as plt
        nodes = list(self.G.nodes())
        pos = {nodes[i]: (self.coords[i, 0], self.coords[i, 1])
               for i in range(self.n_nodes)}
        plt.figure(figsize=(6, 6))
        nx.draw(self.G, pos, with_labels=False,
                node_size=30, node_color="lightblue", edge_color="gray")
        plt.title(f"Crossings: {int(self.current_crossings)} (best: {int(self.best_crossings)})")
        plt.show()
