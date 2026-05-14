import pandas as pd
df = pd.read_parquet("/home/jrigal/.cache/huggingface/lerobot/jogarulfo/dataset_MVP_store_cardboardv1_0/meta/episodes/chunk-000/file-000.parquet")
print(df[["episode_index", "conditioning"]])