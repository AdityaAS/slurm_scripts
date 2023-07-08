import pytorch_lightning as pl
from pytorch_lightning.demos.boring_classes import BoringModel, BoringDataModule
from pytorch_lightning.strategies.ddp import DDPStrategy


def main(process_group_backend: str) -> None:
"""Training entrypoint.
    Args:
    process_group_backend: Communication backend for multi-gpu communicati
    on.
"""

    model = BoringModel()
    datamodule = BoringDataModule()
    strategy = DDPStrategy(process_group_backend=process_group_backend)

    trainer = pl.Trainer(strategy=strategy)
    trainer.fit(model, datamodule=datamodule)


# if __name__ == "__main__":
# Works on 16.04. Does not work on 20.04.
print("Starting NCCL job")
main(process_group_backend="nccl")
print("Main ended with NCCL")
# Works on both 16.04 and 20.04.
# main(process_group_backend="gloo")
