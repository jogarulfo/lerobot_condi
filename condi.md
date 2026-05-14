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


lerobot-train \
  --dataset.repo_id=jogarulfo/dataset_MVP_store_cardboardv1_0 \
  --policy.type=act \
  --policy.repo_id=jogarulfo/act_model_to_store_cardboardv1_0 \
  --output_dir=outputs/train/act_model_to_store_cardboardv1_0 \
  --policy.conditioning_dim=2 \
  --batch_size=8 \
  --policy.device=cuda \
  --wandb.enable=true \
  --job_name=act_model_to_store_cardboardv1_0 \
  --steps=100_000 \
  --save_freq=50_000