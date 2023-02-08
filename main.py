import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets
import torchvision.models
import pytorch_lightning as pl
import sklearn.linear_model


def get_logistic_regression_loss(embedding, labels, num_iterations=100):
    # embedding :: iib, channel -> float32
    # labels :: iib -> int
    num_classes = 10
    batch_size = embedding.shape[1]
    with torch.enable_grad():  # in case we're in inference mode
        model = nn.Linear(batch_size, num_classes).to(embedding.device)
        def logits_and_loss():
            logits = model(embedding)
            return logits, F.cross_entropy(logits, labels)
        lr = 4.e-2
        for _ in range(num_iterations):
            current_logits, current_loss = logits_and_loss()
            grads = torch.autograd.grad(current_loss, model.parameters(), create_graph=True)
            for param, grad in zip(model.parameters(), grads):
                param.data -= lr * grad
    final_logits, final_loss = logits_and_loss()
    final_accuracy = (final_logits.argmax(dim=1) == labels).float().mean()
    return final_loss, final_accuracy


def get_logistic_regression_accuracy_skl(embedding, labels, max_num_iterations=2000):
    # embedding :: iib, channel -> float32
    # labels :: iib -> int
    num_classes = 10
    embedding, labels = embedding.cpu().numpy(), labels.cpu().numpy()
    regressor = sklearn.linear_model.LogisticRegression(random_state=0, max_iter=max_num_iterations, multi_class='multinomial')
    regressor.fit(embedding, labels)
    assert len(regressor.classes_) == num_classes
    predictions = regressor.predict(embedding)
    accuracy = (predictions == labels).mean()
    return accuracy


class Model(pl.LightningModule):

    def __init__(self, lr_regularised, hidden_size=128, alphabet_size=27):
        super().__init__()
        self.lr_regularised = lr_regularised
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
        if self.lr_regularised:
            lr_loss, _ = get_logistic_regression_loss(embedding, ordinal_labels)
        else:
            lr_loss = 0
        # want to apply the above to 'many' layers
        # note for mnist, logistic-regression *on the pixels* yields about 95.5% accuracy! it follows that LR on any early conv layers will also work well
        # hence, only consider dense (or layer conv) layers
        # could either do per-layer, or could consider the concatenation of all activations --> the latter feels somehow more convincing
        character_logits = self.char_decoder(embedding.unsqueeze(-1))  # iib, char-in-alphabet, char-in-seq
        classification_loss = F.cross_entropy(character_logits, text_labels)
        return classification_loss + -lr_loss

    def validation_step(self, batch, batch_idx):
        images, ordinal_labels, text_labels = batch
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
        return embedding, ordinal_labels

    def validation_epoch_end(self, outputs):
        all_embeddings = torch.cat([embedding for (embedding, ordinal_labels) in outputs], dim=0)  # iib, channel -> float32
        all_ordinal_labels = torch.cat([ordinal_labels for (embedding, ordinal_labels) in outputs], dim=0)  # iib -> int
        lr_accuracy_skl = get_logistic_regression_accuracy_skl(all_embeddings, all_ordinal_labels)
        _, lr_accuracy_ours = get_logistic_regression_loss(all_embeddings, all_ordinal_labels)
        self.log("val_lr_accuracy_skl", lr_accuracy_skl, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_lr_accuracy_ours", lr_accuracy_ours, prog_bar=True, on_step=False, on_epoch=True)
        print(f'lr_accuracy_skl = {lr_accuracy_skl:02f}')
        print(f'lr_accuracy_ours = {lr_accuracy_ours:02f}')

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

    model = Model(lr_regularised=True)

    trainer = pl.Trainer(max_epochs=20, accelerator='gpu')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()

