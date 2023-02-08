import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets
import torchvision.models
import pytorch_lightning as pl


# def get_slab_coordinates(xyzs, slab_inner_vertices, slab_outer_vertices):
#
#     # xyzs :: iib, xyz; these are the world-space coordinates we want to map to slab-space
#     # slab_*_vertices :: iib, vertex-in-face, xyz
#
#     slab_vertices = torch.stack([slab_inner_vertices, slab_outer_vertices], dim=1)  # iib, inner/outer, vertex-in-face, xyz
#
#     def slab_to_world(uvw):
#         u, v, w = uvw.unbind(dim=-1)  # each indexed by iib
#         barycentric_interped = u[:, None, None] * slab_vertices[:, :, 0] + v[:, None, None] * slab_vertices[:, :, 1] + (1. - u - v)[:, None, None] * slab_vertices[:, :, 2]  # iib, inner/outer, xyz
#         w_lerped = torch.lerp(barycentric_interped[:, 0], barycentric_interped[:, 1], 1. - w.unsqueeze(1))  # iib, xyz
#         return w_lerped
#
#     def get_distances(uvw):
#         # For the given slab-space coordinates, return how close the resulting world coordinates are to the targets along each axis
#         candidate_world = slab_to_world(uvw)  # iib, xyz
#         return torch.square(candidate_world - xyzs)  # iib, xyz
#
#     def batch_jacobian(f, x):
#         # Assuming f(x) and x are both indexed by [iib, A], and that the i^th slice of f(x) depends only on the i^th slice of x, this
#         # efficiently returns a batch of Jacobian matrices of f(x) wrt x, of shape [iib, A, A]
#         # Inspired by https://discuss.pytorch.org/t/80771/5
#         def sum_f(x):
#             return f(x).sum(dim=0)
#         return torch.autograd.functional.jacobian(sum_f, x, vectorize=True).permute(1, 0, 2)
#
#     # ** we should consider using multiple random restarts per iib, to mitigate non-convergence
#     uvw_t = torch.full_like(xyzs, 0.5)  # iib, uvw
#     # uvw_t = torch.rand_like(xyzs) * 0.2 + 0.4  # iib, uvw
#
#     newton_iterations = 8  # ** this should probably be a hyperparameter! and/or, we could do a proper convergence check
#     for t in range(newton_iterations):
#         # Do a batched Newton-Raphson iteration of the form x_{t+1} = x_t - J(x_t)^{-1} f(x_t), where f : R^3 -> R^3
#         # maps slab to world coordinates and measures the resulting per-axis errors, and J is the jacobian of f
#         distances = get_distances(uvw_t)  # iib, xyz
#         jacobians = batch_jacobian(get_distances, uvw_t)  # iib, xyz, uvw
#         jacobians_or_eyes = torch.where((distances == 0.).unsqueeze(-1), torch.eye(3).to(jacobians.device), jacobians)  # if x/y/z distance is zero, then relevant partial derivatives are also zero, leading to singular jacobian
#         uvw_t += torch.lu_solve(-distances.unsqueeze(-1), *torch.lu(jacobians_or_eyes)).squeeze(-1)  # iib, uvw
#
#     converged = get_distances(uvw_t).mean(dim=-1) < slab_inner_vertices.std(dim=-2).mean(dim=-1) * 1.e-2  # i.e. require convergence to within approx. 1% of triangle extent
#
#     return uvw_t, converged


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
        print(' '.join(''.join(chr(c + ord('a') - 1) if c != 0 else ' ' for c in chars_for_iib) for chars_for_iib in torch.argmax(character_logits, dim=1)[:10]), end='')
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

