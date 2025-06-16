from pathlib import Path
from trainer import MultiModalTrainer
import pandas as pd


def main():
    cases = pd.read_parquet(
        "/mnt/jfs/rcabench-platform-v2/meta/rcabench_filtered/index.parquet"
    )
    print(cases.columns)
    top_10 = cases["datapack"].head(10).tolist()

    data_paths = [
        Path(f"/mnt/jfs/rcabench-platform-v2/data/rcabench_filtered/{i}")
        for i in top_10
    ]

    config = {
        "max_len": 512,
        "d_model": 512,
        "nhead": 8,
        "d_ff": 2048,
        "layer_num": 6,
        "dropout": 0.1,
    }

    trainer = MultiModalTrainer(
        data_paths=data_paths, config=config, cache_dir="./cache"
    )

    results = trainer.train(epochs=50, batch_size=16, lr=1e-4)

    print("Training completed!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
