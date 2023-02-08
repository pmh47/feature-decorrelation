import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets
import torchvision.models
import pytorch_lightning as pl


def do_logistic_regression(embedding, labels):
    # embedding :: iib, channel -> float32
    # labels :: iib -> int
    num_classes = 10
    batch_size = embedding.shape[1]
    model = nn.Linear(batch_size, num_classes).to(embedding.device)
    def logits_and_loss():
        logits = model(embedding)
        return logits, F.cross_entropy(logits, labels)
    lr = 4.e-2
    for _ in range(20):
        current_logits, current_loss = logits_and_loss()
        print('loss:', current_loss.item(), '; accuracy:', (current_logits.argmax(dim=1) == labels).float().mean().item())
        grads = torch.autograd.grad(current_loss, model.parameters(), create_graph=False)
        for param, grad in zip(model.parameters(), grads):
            param.data -= lr * grad


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
        images, ordinal_labels, text_labels = batch
        embedding = self.fc(self.resnet(images.expand(-1, 3, -1, -1)))
        do_logistic_regression(embedding, ordinal_labels)
        character_logits = self.char_decoder(embedding.unsqueeze(-1))  # iib, char-in-alphabet, char-in-seq
        loss = F.cross_entropy(character_logits, text_labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _, text_labels = batch
        embedding = self.fc(self.resnet(images.expand(-1, 3, -1, -1)))
        character_logits = self.char_decoder(embedding.unsqueeze(-1))  # iib, char-in-alphabet, char-in-seq
        loss = F.cross_entropy(character_logits, text_labels)
        char_indices = torch.argmax(character_logits, dim=1)
        charwise_accuracy = (char_indices == text_labels).float().mean()
        labelwise_accuracy = (char_indices == text_labels).all(dim=1).float().mean()
        print(' '.join(''.join(chr(c + ord('a') - 1) if c != 0 else ' ' for c in chars_for_iib) for chars_for_iib in char_indices[:10]), end='')
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_charwise_accuracy", charwise_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_labelwise_accuracy", labelwise_accuracy, prog_bar=True, on_step=False, on_epoch=True)

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
        image, ordinal_label = self.original[idx]
        text_label = DatasetWithTextLabels.character_indices_by_label[ordinal_label]
        return image, ordinal_label, text_label


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

