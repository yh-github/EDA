import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor


def train_tft(tft, train_dataloader, val_dataloader):
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=5,
        verbose=False,
        mode="min"
    )

    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",  # Auto-detects your A100
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
    )

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    return trainer