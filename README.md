```bash
cd /cs5180
```
# train model
```bash
python train_fixed.py \
  --rome_dir "./rome" \
  --timesteps 500000 \
  --emb_path nonexistent.pt
```

Multi-start要另外加
训练完之后，在test_on_graph外面套一个循环：
best = float('inf')
for i in range(5):
    result = test_on_graph(model, graph_path, embeddings)
    best = min(best, result['best_crossings'])
print(f"Multi-start best: {best}")

python train_fixed.py \
  --test_only \
  --rome_dir "./rome" \
  --emb_path nonexistent.pt \
  --model_path ppo_fixed

# 优先移动crossing最多的节点 + soft rewarding
python train_fixed.py \
  --rome_dir "./rome" \
  --timesteps 500000 \
  --emb_path nonexistent.pt \
  --model_path ppo_fixed_v3

python train_fixed.py \
  --test_only \
  --rome_dir "./rome" \
  --emb_path nonexistent.pt \
  --model_path ppo_fixed_v3