import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets
import torchvision.models
import pytorch_lightning as pl


class Model(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(num_classes=10)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.resnet(images.expand(-1, 3, -1, -1))
        loss = F.cross_entropy(logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.resnet(images.expand(-1, 3, -1, -1))
        loss = F.cross_entropy(logits, labels)
        accuracy = (torch.argmax(logits, dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():

    train_dataset = torchvision.datasets.MNIST('/tmp/mnist', train=True, download=True, transform=torchvision.transforms.ToTensor())
    val_dataset = torchvision.datasets.MNIST('/tmp/mnist', train=False, download=True, transform=torchvision.transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=4)

    model = Model()

    trainer = pl.Trainer(max_epochs=20, accelerator='gpu')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()

