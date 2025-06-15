from trainer import MultiModalTrainer
from config import CONFIG_DICT
import dataset.utils as U
from datetime import datetime
import typer

app = typer.Typer()


@app.command()
def run(dataset: str = "gaia"):
    U.set_seed(2024)
    time = datetime.now().strftime("%Y年%m月%d日%H时%M分%S秒")

    # model = Model.from_pretrained("AI-ModelScope/bert-base-uncased")
    # MultiModalTrainer(CONFIG_DICT[dataset], time).experiment()
    MultiModalTrainer(CONFIG_DICT[dataset], time).train()


if __name__ == "__main__":
    app()
