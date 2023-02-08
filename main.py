import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets
import torchvision.models
import pytorch_lightning as pl


class Model(pl.LightningModule):

    def __init__(self, hidden_size=128, alphabet_size=27):
        super().__init__()
        self.resnet = torchvision.models.resnet18(num_classes=128)
        self.fc = nn.Sequential(
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.LayerNorm(128),
        )
        self.char_decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=5),
            nn.ELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding='same'),
            nn.ELU(),
            nn.Conv1d(hidden_size, alphabet_size, kernel_size=1),
        )

    def training_step(self, batch, batch_idx):
        images, labels = batch
        embedding = self.fc(self.resnet(images.expand(-1, 3, -1, -1)))
        character_logits = self.char_decoder(embedding.unsqueeze(-1))  # iib, char-in-alphabet, char-in-seq
        loss = F.cross_entropy(character_logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        embedding = self.fc(self.resnet(images.expand(-1, 3, -1, -1)))
        character_logits = self.char_decoder(embedding.unsqueeze(-1))  # iib, char-in-alphabet, char-in-seq
        loss = F.cross_entropy(character_logits, labels)
        accuracy = (torch.argmax(character_logits, dim=1) == labels).float().mean()
        print(''.join(chr(c + ord('a') - 1) if c != 0 else ' ' for c in torch.argmax(character_logits[0], dim=0)), end='')
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class DatasetWithTextLabels(torch.utils.data.Dataset):

    character_indices_by_label = torch.tensor([
        tuple(map(lambda c: ord(c) - ord('a') + 1, label)) + (0,) * (5 - len(label))
        for label in ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    ])

    def __init__(self, original):
        self.original = original

    def __len__(self):
        return len(self.original)

    def __getitem__(self, idx):
        image, label = self.original[idx]
        label = DatasetWithTextLabels.character_indices_by_label[label]
        return image, label


def main():

    train_dataset = DatasetWithTextLabels(torchvision.datasets.MNIST('/tmp/mnist', train=True, download=True, transform=torchvision.transforms.ToTensor()))
    val_dataset = DatasetWithTextLabels(torchvision.datasets.MNIST('/tmp/mnist', train=False, download=True, transform=torchvision.transforms.ToTensor()))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=4)

    model = Model()

    trainer = pl.Trainer(max_epochs=20, accelerator='gpu')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()

