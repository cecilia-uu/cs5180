"""
data_loader.py
--------------
Project Update 1 — Ginger Agents (Hui Kong, Weiyi Mao)
CS5180 Spring 2026

Step 1: Randomly sample 5 graphs → print table + save plot  (fast, for PPT)
Step 2: Process all training graphs (index 1–9999) → save rome_baseline.csv

Usage:
    python data_loader.py --data_dir ./rome

Dataset download:
    wget https://graphdrawing.unipg.it/data/rome-graphml.tgz
    tar -xzf rome-graphml.tgz

Dependencies:
    pip install tqdm
"""

import os
import glob
import random
import argparse
import sys
import csv
import time

import networkx as nx
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI window — just save to file
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    sys.exit(
        "[ERROR] tqdm not installed. Run: pip install tqdm"
    )

# ── Locate XingLoss ──────────────────────────────────────────────────────────
try:
    from xing import XingLoss
except ImportError:
    try:
        from Xing.xing import XingLoss
    except ImportError:
        sys.exit(
            "[ERROR] Cannot import XingLoss.\n"
            "Make sure xing.py (or Xing/xing.py) is in your Python path."
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def graph_index(path: str) -> int:
    base = os.path.basename(path)
    num_str = base.replace("grafo", "").split(".")[0]
    try:
        return int(num_str)
    except ValueError:
        return 0


def load_graph(path: str) -> nx.Graph:
    G = nx.read_graphml(path)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return G


def get_neato_coords(G: nx.Graph) -> tuple:
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    coords = torch.tensor(
        [[pos[v][0], pos[v][1]] for v in G.nodes()],
        dtype=torch.float32,
    )
    return pos, coords


def compute_crossings(G: nx.Graph, coords: torch.Tensor) -> int:
    xing_loss = XingLoss(G, soft=False)
    return int(xing_loss(coords).item())


def process_graph(path: str, keep_layout: bool = False) -> dict:
    """
    keep_layout=True  → also store G and pos (needed for plotting)
    keep_layout=False → lighter, just stats (used for the full 9999 loop)
    """
    name = os.path.basename(path)
    G = load_graph(path)
    pos, coords = get_neato_coords(G)
    crossings = compute_crossings(G, coords)
    row = {
        "graph_name": name,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "initial_crossings_neato": crossings,
    }
    if keep_layout:
        row["G"] = G
        row["pos"] = pos
    return row


# ── Visualization ────────────────────────────────────────────────────────────

def plot_sample_graphs(results: list, save_path: str = "sample_layouts.png"):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, row in zip(axes, results):
        nx.draw(
            row["G"], row["pos"],
            ax=ax,
            with_labels=False,
            node_size=30,
            node_color="steelblue",
            edge_color="gray",
            width=0.8,
        )
        ax.set_title(
            f"{row['graph_name']}\n"
            f"N={row['nodes']}  E={row['edges']}  "
            f"crossings={row['initial_crossings_neato']}",
            fontsize=8,
        )

    plt.suptitle("Sample neato Layouts — Rome Dataset (Initial State)", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir: str, n_sample: int, seed: int, csv_path: str):
    random.seed(seed)

    # Collect all GraphML files
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.graphml")))
    if not all_files:
        sys.exit(
            f"[ERROR] No .graphml files found in '{data_dir}'.\n"
            "Download and extract the Rome dataset first:\n"
            "  wget https://graphdrawing.unipg.it/data/rome-graphml.tgz\n"
            "  tar -xzf rome-graphml.tgz"
        )

    # Training split only (index 1–9999)
    train_files = [f for f in all_files if 1 <= graph_index(f) <= 9999]
    print(f"Found {len(all_files)} total graphs, "
          f"{len(train_files)} in training split (index 1–9999).\n")

    # ── STEP 1: Sample 5 graphs → table + plot ────────────────────────────────
    print("=" * 60)
    print(f"STEP 1: Sampling {n_sample} graphs for PPT")
    print("=" * 60)

    sampled_paths = random.sample(train_files, min(n_sample, len(train_files)))
    sample_results = []

    for path in sampled_paths:
        try:
            row = process_graph(path, keep_layout=True)
            sample_results.append(row)
            print(f"  ✓ {row['graph_name']:35s}  "
                  f"nodes={row['nodes']:4d}  "
                  f"edges={row['edges']:4d}  "
                  f"crossings={row['initial_crossings_neato']:5d}")
        except Exception as e:
            print(f"  ✗ {os.path.basename(path):35s}  ERROR: {e}")

    # Print table
    print("\n" + "=" * 72)
    print("Baseline Summary Table — 5 Sample Graphs (neato initial layout)")
    print("=" * 72)
    print(f"{'Graph Name':<35} {'Nodes':>6} {'Edges':>6} {'Crossings (neato)':>18}")
    print("-" * 72)
    for row in sample_results:
        print(
            f"{row['graph_name']:<35} "
            f"{row['nodes']:>6} "
            f"{row['edges']:>6} "
            f"{row['initial_crossings_neato']:>18}"
        )
    print("=" * 72)

    # Save plot
    plot_sample_graphs(sample_results, save_path="sample_layouts.png")

    # ── STEP 2: Process all training graphs → CSV (with tqdm progress bar) ────
    print("\n" + "=" * 60)
    print(f"STEP 2: Processing all {len(train_files)} training graphs → {csv_path}")
    print("=" * 60 + "\n")

    # Pre-fill with the 5 samples we already processed
    sampled_names = {r["graph_name"] for r in sample_results}
    all_results = {r["graph_name"]: r for r in sample_results}

    error_count = 0
    start_time = time.time()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["graph_name", "nodes", "edges", "initial_crossings_neato"],
        )
        writer.writeheader()

        pbar = tqdm(
            train_files,
            desc="Processing graphs",
            unit="graph",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        running_crossings = []

        for i, path in enumerate(pbar, start=1):
            name = os.path.basename(path)

            # Reuse already-processed sample graphs
            if name in sampled_names:
                row = all_results[name]
            else:
                try:
                    row = process_graph(path, keep_layout=False)
                    all_results[name] = row
                except Exception as e:
                    error_count += 1
                    pbar.set_postfix_str(f"errors={error_count}", refresh=False)
                    continue

            running_crossings.append(row["initial_crossings_neato"])

            writer.writerow({
                "graph_name": row["graph_name"],
                "nodes":      row["nodes"],
                "edges":      row["edges"],
                "initial_crossings_neato": row["initial_crossings_neato"],
            })

            # Update postfix with running avg crossings every 50 graphs
            if i % 50 == 0:
                avg_c = np.mean(running_crossings[-200:])  # rolling window
                pbar.set_postfix_str(
                    f"avg_xing={avg_c:.1f} | errors={error_count}",
                    refresh=True,
                )

        pbar.close()

    # Final summary
    elapsed = time.time() - start_time
    all_rows = list(all_results.values())
    avg_c = np.mean([r["initial_crossings_neato"] for r in all_rows])
    avg_n = np.mean([r["nodes"] for r in all_rows])
    avg_e = np.mean([r["edges"] for r in all_rows])

    print(f"\n{'=' * 60}")
    print(f"DONE!  {csv_path}  ({elapsed:.0f}s = {elapsed/60:.1f} min)")
    print(f"{'=' * 60}")
    print(f"  Total graphs processed : {len(all_rows)}")
    print(f"  Errors / skipped       : {error_count}")
    print(f"  Avg nodes              : {avg_n:.1f}")
    print(f"  Avg edges              : {avg_e:.1f}")
    print(f"  Avg crossings (neato)  : {avg_c:.1f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rome dataset baseline loader")
    parser.add_argument(
        "--data_dir", type=str, default="./rome",
        help="Path to extracted Rome GraphML files (default: ./rome)",
    )
    parser.add_argument(
        "--n_sample", type=int, default=5,
        help="Number of graphs to sample for PPT table (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--csv_path", type=str, default="rome_baseline.csv",
        help="Output CSV path for full training set results (default: rome_baseline.csv)",
    )
    args = parser.parse_args()
    main(args.data_dir, args.n_sample, args.seed, args.csv_path)
