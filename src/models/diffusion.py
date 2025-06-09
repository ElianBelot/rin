# ===============[ IMPORTS ]===============
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb

from src.models.rin import RINModel
from src.utils.noise import forward_noise, gamma_cosine
from src.utils.sample import sample_ddpm


# ===============[ CORE ]===============
class DiffusionModel(pl.LightningModule):
    def __init__(
        self,
        net: RINModel,
        image_size: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        num_diffusion_samples: int = 5,
        num_diffusion_steps: int = 50,
        self_cond_rate: float = 0.9,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])

        # Parameters
        self.net = net
        self.image_size = image_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_diffusion_samples = num_diffusion_samples
        self.num_diffusion_steps = num_diffusion_steps
        self.self_cond_rate = self_cond_rate

        # Functions
        self.gamma_fn = gamma_cosine
        self.criterion = F.mse_loss

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, Z_prev: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Given a noisy image and a timestep, predict the added noise.

        Parameters:
        - x: noisy image (B, 3, H, W)
        - t: timestep tensor (B,)
        - Z_prev: optional previous latent state (B, N, D)

        Returns:
        - noise_pred: predicted noise image (B, 3, H, W)
        - Z: latent state Z' (B, N, D)
        - logs: dictionary of logs
        """
        noise_pred, Z, logs = self.net(x, t, Z_prev)
        return noise_pred, Z, logs

    # ==========[ SETUP ]==========
    def configure_optimizers(self):
        """Set up optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}

    @torch.no_grad()
    def sample(self, num_samples: int = 3, steps: int = 50) -> torch.Tensor:
        """Run DDPM-style sampling from noise to image."""
        shape = (num_samples, 3, self.image_size, self.image_size)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            samples = sample_ddpm(self, shape=shape, gamma_fn=self.gamma_fn, num_steps=steps, device=self.device)
        return samples.float()

    # ==========[ TRAINING ]===========
    def training_step(self, batch, batch_idx):
        """Sample timestep t, forward-noise x0, predict noise, and compute loss."""
        x0, _ = batch
        B = x0.size(0)
        t = torch.rand(B, device=self.device)

        # Noise image
        xt, noise = forward_noise(x0, t, self.gamma_fn)

        # Latent self-conditioning
        Z_prev = None
        self.self_cond_rate = 0.0
        if torch.rand(1).item() < self.self_cond_rate:
            with torch.no_grad():
                _, Z_prev, _ = self.net(xt, t, Z_prev=None)

        # Forward pass
        noise_pred, Z, logs = self.forward(xt, t, Z_prev=Z_prev)

        # Loss
        train_loss = self.criterion(noise_pred, noise)

        # Logging
        self.log("train/loss", train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, prog_bar=True)
        self.log("train/t_mean", t.mean(), on_step=True)
        self.log("train/x0_norm", x0.norm(p=2), on_step=True)
        self.log("train/xt_norm", xt.norm(p=2), on_step=True)
        self.log("train/noise_std", noise.std(), on_step=True)
        self.log("train/noise_norm", noise.norm(p=2), on_step=True)
        self.log("train/noise_pred_std", noise_pred.std(), on_step=True)
        self.log("train/noise_pred_norm", noise_pred.norm(p=2), on_step=True)
        self.log("train/X_mean", logs["X_mean"], on_step=True)
        self.log("train/X_std", logs["X_std"], on_step=True)
        self.log("train/X_norm", logs["X_norm"], on_step=True)
        self.log("train/Z_mean", Z.mean(), on_step=True)
        self.log("train/Z_std", Z.std(), on_step=True)
        self.log("train/Z_norm", Z.norm(p=2), on_step=True)
        self.log("train/self_cond_used", float(Z_prev is not None), on_step=True)

        return train_loss

    # ==========[ VALIDATION ]===========
    def validation_step(self, batch, batch_idx):
        """Same as training step but log samples."""
        x0, _ = batch
        B = x0.size(0)
        t = torch.rand(B, device=self.device)

        # Forward pass
        xt, noise = forward_noise(x0, t, self.gamma_fn)
        noise_pred, Z, logs = self(xt, t)
        val_loss = self.criterion(noise_pred, noise)

        # Logging
        self.log("val/loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/t_mean", t.mean(), on_epoch=True)
        self.log("val/x0_norm", x0.norm(p=2), on_epoch=True)
        self.log("val/xt_norm", xt.norm(p=2), on_epoch=True)
        self.log("val/noise_std", noise.std(), on_epoch=True)
        self.log("val/noise_norm", noise.norm(p=2), on_epoch=True)
        self.log("val/noise_pred_std", noise_pred.std(), on_epoch=True)
        self.log("val/noise_pred_norm", noise_pred.norm(p=2), on_epoch=True)
        self.log("val/X_mean", logs["X_mean"], on_epoch=True)
        self.log("val/X_std", logs["X_std"], on_epoch=True)
        self.log("val/X_norm", logs["X_norm"], on_epoch=True)
        self.log("val/Z_mean", Z.mean(), on_epoch=True)
        self.log("val/Z_std", Z.std(), on_epoch=True)
        self.log("val/Z_norm", Z.norm(p=2), on_epoch=True)

        # Log samples
        if batch_idx == 0:
            samples = self.sample(num_samples=self.num_diffusion_samples, steps=self.num_diffusion_steps)
            images = [wandb.Image((samples[i] + 1) / 2.0) for i in range(samples.size(0))]
            self.logger.experiment.log({"Samples": images, "global_step": self.global_step})
