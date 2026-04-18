"""
train_fixed.py (v3)
--------------------
Ginger Agents — CS5180 Spring 2026

Train PPO + evaluate with SPC metric (teacher's requirement).

SPC = 100% * (1/Nt) * Σ (Di - Gi) / max(Di, Gi)
  Di = crossings from our method
  Gi = crossings from neato
  SPC < 0 means we beat neato (e.g., SPC = -20 means 20% better)

Usage:
    # Train
    python train_fixed.py --rome_dir ./rome --timesteps 500000

    # Test only (after training)
    python train_fixed.py --test_only --rome_dir ./rome

    # Test specific graph
    python train_fixed.py --test_only --rome_dir ./rome --test_graph grafo4647.62.graphml
"""

import os
import argparse
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env_fixed import GraphLayoutEnvFixed, MAX_NODES
from xing import XingLoss


def load_embeddings(path="rome_embeddings.pt"):
    if not os.path.exists(path):
        print(f"[WARNING] {path} not found. Training without topology embeddings.")
        return {}
    data_list = torch.load(path, weights_only=False)
    emb_dict = {}
    for d in data_list:
        emb_dict[d["graph_name"]] = {
            "node_emb": d["node_emb"],
            "graph_emb": d["graph_emb"],
        }
    print(f"Loaded embeddings for {len(emb_dict)} graphs")
    return emb_dict


def get_graph_paths(rome_dir, min_idx=1, max_idx=9999, max_nodes=MAX_NODES):
    paths = []
    for fname in sorted(os.listdir(rome_dir)):
        if not fname.endswith(".graphml"):
            continue
        try:
            num = int(fname.split("grafo")[1].split(".")[0])
        except:
            continue
        if num < min_idx or num > max_idx:
            continue
        fpath = os.path.join(rome_dir, fname)
        G = nx.read_graphml(fpath)
        if G.number_of_nodes() <= max_nodes:
            paths.append(fpath)
    return paths


def test_on_graph(model, graph_path, embeddings, max_steps=300):
    """Run trained PPO on one graph. Returns best crossings found."""
    env = GraphLayoutEnvFixed(
        [graph_path],
        embeddings=embeddings,
        max_steps=max_steps,
        move_scale=2.0,
        patience=50,
    )

    obs, _ = env.reset()
    initial_crossings = env.current_crossings
    crossing_history = [initial_crossings]

    done = False
    truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        crossing_history.append(info["crossings"])

    return {
        "graph_name": os.path.basename(graph_path),
        "neato_crossings": initial_crossings,
        "best_crossings": info["best_crossings"],
        "improvement": info["improvement"],
        "crossing_history": crossing_history,
        "steps": info["steps"],
    }


def compute_spc(results):
    """
    SPC = 100% * (1/Nt) * Σ (Di - Gi) / max(Di, Gi)
    Di = our crossings, Gi = neato crossings
    SPC < 0 means we are better than neato
    """
    spc_values = []
    for r in results:
        Di = r["best_crossings"]
        Gi = r["neato_crossings"]
        denom = max(Di, Gi)
        if denom > 0:
            spc_values.append((Di - Gi) / denom)
        else:
            spc_values.append(0.0)  # both zero = perfect
    return 100.0 * np.mean(spc_values)

def multi_start_optimize(model, graph_path, embeddings, 
                         n_starts=5, max_steps=300):
    best_crossings = float('inf')
    best_result = None
    
    # 先记录neato原始crossing数
    env_base = GraphLayoutEnvFixed(
        [graph_path],
        embeddings=embeddings,
        max_steps=max_steps,
        move_scale=2.0,
        patience=50,
    )
    obs, _ = env_base.reset()
    neato_crossings = env_base.current_crossings  # 保存原始neato结果
    
    for i in range(n_starts):
        env = GraphLayoutEnvFixed(
            [graph_path],
            embeddings=embeddings,
            max_steps=max_steps,
            move_scale=2.0,
            patience=50,
        )
        obs, _ = env.reset()
        
        # 第一次用原始neato，后面加扰动
        if i > 0:
            noise = np.random.randn(*env.coords.shape) * env.coords.std() * 0.1
            env.coords += noise.astype(np.float32)
            env.current_crossings = env._compute_crossings(env.coords)
            env.best_crossings = env.current_crossings
            obs = env._get_obs()
        
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
        
        if info['best_crossings'] < best_crossings:
            best_crossings = info['best_crossings']
            best_result = {
                "graph_name": os.path.basename(graph_path),
                "neato_crossings": neato_crossings,  # 始终用原始neato
                "best_crossings": info['best_crossings'],
                "improvement": neato_crossings - info['best_crossings'],
                "crossing_history": [],
                "steps": info['steps'],
            }
    
    return best_result

def plot_result(result, save_path=None):
    plt.figure(figsize=(10, 5))
    neato = result["neato_crossings"]
    plt.axhline(y=neato, color="red", linestyle="--",
                label=f"neato baseline ({int(neato)})")
    plt.plot(result["crossing_history"], label="PPO", color="blue")

    best_step = np.argmin(result["crossing_history"])
    best_val = result["crossing_history"][best_step]
    plt.scatter([best_step], [best_val], color="green", s=100, zorder=5,
                label=f"best: {int(best_val)} (step {best_step})")

    plt.xlabel("Steps")
    plt.ylabel("Edge Crossings")
    plt.title(f"PPO vs neato — {result['graph_name']}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    imp = result["improvement"]
    plt.text(0.02, 0.98,
             f"neato: {int(neato)}  →  PPO best: {int(result['best_crossings'])}  "
             f"(Δ = {int(imp)})",
             transform=plt.gca().transAxes, fontsize=11, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved → {save_path}")
    plt.show()


def evaluate_test_set(model, rome_dir, embeddings, max_steps=300):
    """
    Evaluate on test graphs 10000-10100 using SPC metric.
    """
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET (graphs 10000–10100) — SPC Metric")
    print("=" * 70)

    test_paths = get_graph_paths(rome_dir, min_idx=10000, max_idx=10100)
    print(f"Test graphs found: {len(test_paths)}\n")

    if len(test_paths) == 0:
        print("[ERROR] No test graphs found. Check rome_dir.")
        return

    results = []
    wins, ties, losses = 0, 0, 0

    for gpath in test_paths:
        # multi_start_optimize
        r = multi_start_optimize(model, gpath, embeddings, 
                         n_starts=10, max_steps=max_steps)
        results.append(r)

        imp = r["improvement"]
        if imp > 0:
            status, wins = "✓", wins + 1
        elif imp == 0:
            status, ties = "=", ties + 1
        else:
            status, losses = "✗", losses + 1

        print(
            f"  {status} {r['graph_name']:35s}  "
            f"neato={int(r['neato_crossings']):3d} → PPO={int(r['best_crossings']):3d}  "
            f"(Δ={int(imp):+d})"
        )

    # Compute SPC
    spc = compute_spc(results)

    total = len(results)
    avg_imp = np.mean([r["improvement"] for r in results])
    avg_neato = np.mean([r["neato_crossings"] for r in results])
    avg_ppo = np.mean([r["best_crossings"] for r in results])

    print(f"\n{'=' * 70}")
    print(f"  SPC (Symmetric Percent Change) = {spc:.2f}%")
    print(f"  (Negative = better than neato. E.g., -20 means 20% better)")
    print(f"{'=' * 70}")
    print(f"  Test graphs        : {total}")
    print(f"  Wins / Ties / Loss : {wins} / {ties} / {losses}")
    print(f"  Win rate           : {100 * wins / total:.1f}%")
    print(f"  Avg neato crossings: {avg_neato:.1f}")
    print(f"  Avg PPO crossings  : {avg_ppo:.1f}")
    print(f"  Avg improvement    : {avg_imp:.1f}")
    print(f"{'=' * 70}")

    # 按节点数排序
    results_sorted = sorted(results, key=lambda r: nx.read_graphml(
        os.path.join(rome_dir, r['graph_name'])).number_of_nodes())

    ratios_ours = [(r['best_crossings'] - r['neato_crossings']) / 
                max(r['best_crossings'], r['neato_crossings'], 1) 
                for r in results_sorted]

    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(ratios_ours)), ratios_ours, 
                alpha=0.7, color='pink', marker='^', label='ours (PPO)')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel("Graph index (sorted by num_nodes, ascending)")
    plt.ylabel("Ratio vs neato (negative = better than neato)")
    plt.title("Per-graph ratio vs neato baseline")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("spc_scatter_ours.png", dpi=150)
    plt.show()
    return spc, results

def save_coords(model, rome_dir, embeddings, 
                output_dir="coords", n_starts=10):
    os.makedirs(output_dir, exist_ok=True)
    
    test_paths = get_graph_paths(rome_dir, 
                                  min_idx=10000, 
                                  max_idx=10100)
    
    for gpath in test_paths:
        # 用multi-start找最好布局
        result = multi_start_optimize(
            model, gpath, embeddings, 
            n_starts=n_starts
        )
        
        # 获取最好的坐标
        env = GraphLayoutEnvFixed(
            [gpath], embeddings=embeddings,
            max_steps=300, move_scale=2.0, patience=50
        )
        obs, _ = env.reset()
        
        # 重新跑一次得到best_coords
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
        
        # 保存坐标
        graph_name = os.path.basename(gpath).replace('.graphml', '')
        coord_file = os.path.join(output_dir, f"{graph_name}.coord")
        
        G = env.G
        nodes = list(G.nodes())
        with open(coord_file, 'w') as f:
            for i, node in enumerate(nodes):
                x, y = env.best_coords[i]
                f.write(f"{node} {x:.6f} {y:.6f}\n")
        
        print(f"Saved {coord_file}")
    
    print(f"Done! {len(test_paths)} coord files saved.")

def check_overlapping(coords, threshold=1.0):
    """检查是否有节点重叠"""
    n = len(coords)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((coords[i,0]-coords[j,0])**2 + 
                          (coords[i,1]-coords[j,1])**2)
            if dist < threshold:
                return True  # 有重叠
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rome_dir", type=str, default="./rome")
    parser.add_argument("--emb_path", type=str, default="rome_embeddings.pt")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--move_scale", type=float, default=2.0)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--test_graph", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="ppo_fixed")
    args = parser.parse_args()

    embeddings = load_embeddings(args.emb_path)

    if not args.test_only:
        # ── TRAIN ──────────────────────────────────────────
        train_paths = get_graph_paths(args.rome_dir, min_idx=1, max_idx=9999)
        print(f"Training graphs: {len(train_paths)}")

        env = GraphLayoutEnvFixed(
            train_paths,
            embeddings=embeddings,
            max_steps=args.max_steps,
            move_scale=args.move_scale,
            patience=50,
        )

        check_env(env)
        print("Environment check passed!\n")

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.02,  # more exploration
            clip_range=0.2,
        )

        print(f"Training for {args.timesteps} timesteps...")
        model.learn(total_timesteps=args.timesteps)
        model.save(args.model_path)
        print(f"Model saved → {args.model_path}")
    else:
        model = PPO.load(args.model_path)
        print(f"Loaded model from {args.model_path}")

    # ── TEST ──────────────────────────────────────────────
    if args.test_graph:
        # Test single graph
        test_path = os.path.join(args.rome_dir, args.test_graph)
        if not os.path.exists(test_path):
            print(f"[ERROR] {test_path} not found")
            return
        print(f"\nTesting on {args.test_graph}...")
        result = test_on_graph(model, test_path, embeddings, args.max_steps)
        print(f"  neato: {int(result['neato_crossings'])} → PPO best: {int(result['best_crossings'])} (Δ={int(result['improvement'])})")
        plot_result(result, save_path="ppo_vs_neato.png")

    # Always run SPC evaluation on test set
    spc, results = evaluate_test_set(model, args.rome_dir, embeddings, args.max_steps)


if __name__ == "__main__":
    main()
