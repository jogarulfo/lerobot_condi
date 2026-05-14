pattern fill : 
```bash
python -m lerobot.scripts.lerobot_add_conditioning_labels \
  --repo-id jogarulfo/dataset_MVP_store_cardboardv1_0 \
  --conditioning-pattern "[1, 2]"
```

episode single conditionning values : 
```bash
python -m lerobot.scripts.lerobot_add_conditioning_labels \
  --repo-id your_user/your_dataset \
  --episode-conditioning '{"0": 1, "1": 2, "2": 1, "3": 1}'
```