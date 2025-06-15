from pathlib import Path
from trainer import MultiModalTrainer


def main():
    data_paths = [
        Path(
            "/mnt/jfs/rcabench-platform-v2/data/rcabench_with_issues/ts3-ts-auth-service-response-replace-code-wm6gfv"
        ),
        Path(
            "/mnt/jfs/rcabench-platform-v2/data/rcabench_with_issues/ts9-ts-route-plan-service-response-delay-d65ptt"
        ),
        Path(
            "/mnt/jfs/rcabench-platform-v2/data/rcabench_with_issues/ts8-ts-train-service-partition-tvxd88"
        ),
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
