
from typing import Iterator, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from main_jax import get_logistic_regression_loss, get_logistic_regression_accuracy_skl


NUM_CLASSES = 10


class Batch(NamedTuple):
    input_image: np.ndarray  # [B, H, W, 3]
    ordinal_label: np.ndarray  # [B]
    output_image: np.ndarray  # [B, H, W, 3]


digit_patches = [
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ],
    [
        [1, 1,1],
        [0, 0, 1],
        [1, 1,1],
        [1, 0, 0],
        [1, 1, 1],
    ],
    [
        [1, 1,1],
        [0, 0, 1],
        [1, 1,1],
        [0, 0, 1],
        [1, 1, 1],
    ],
    [
        [1, 0,0],
        [1, 0, 0],
        [1, 0,1],
        [1, 1, 1],
        [0, 0, 1],
    ],
    [
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
    ],
    [
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    [
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
    ],
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
    ],
]
def make_batch(image, label):
    image = jnp.tile(image, [1, 1, 1, 3]).astype(jnp.float32) / 255.
    input_image = image * 0.5 + (1 - image) * np.random.random(size=(image.shape[0], 1, 1, 3))
    patches = jnp.asarray([digit_patches[L] for L in label])
    patches = jax.image.resize(patches, [patches.shape[0], 10, 6], jax.image.ResizeMethod.NEAREST)[..., None]
    output_image = input_image.at[:, -10:, -6:, :].set(jnp.where(patches, 0.75, input_image[:, -10:, -6:, :]))
    return Batch(input_image, label, output_image)


class TrainingState(NamedTuple):
    params: hk.Params
    avg_params: hk.Params
    opt_state: optax.OptState


def net_fn(images: jnp.ndarray) -> jnp.ndarray:
    encoder = hk.Sequential([
        hk.Conv2D(24, kernel_shape=3), jax.nn.elu,
        hk.GroupNorm(3),
        hk.Conv2D(32, kernel_shape=3, stride=2), jax.nn.elu,
        hk.GroupNorm(4),
        hk.Conv2D(48, kernel_shape=3, stride=2), jax.nn.elu,
        hk.GroupNorm(6),
        hk.Conv2D(64, kernel_shape=3, stride=2), jax.nn.elu,
        hk.GroupNorm(8),
        hk.Flatten(),
        # hk.LayerNorm(1, create_scale=True, create_offset=True),
        hk.Linear(128), jax.nn.elu,
    ])
    upsample = lambda x: jax.image.resize(x, [x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]], method=jax.image.ResizeMethod.LINEAR)
    decoder = hk.Sequential([
        hk.Reshape([2, 2, -1]),
        upsample,
        hk.Conv2D(32, kernel_shape=3, padding='SAME'), jax.nn.elu,
        hk.GroupNorm(4),
        upsample,
        hk.Conv2D(32, kernel_shape=3, padding='SAME'), jax.nn.elu,
        hk.GroupNorm(4),
        upsample,
        hk.Conv2D(24, kernel_shape=3, padding='VALID'), jax.nn.elu,
        hk.GroupNorm(4),
        upsample,
        hk.Conv2D(16, kernel_shape=3, padding='SAME'), jax.nn.elu,
        hk.Conv2D(3, kernel_shape=1)
    ])
    embedding = encoder(images)
    decoded = decoder(embedding)
    return decoded, embedding


def load_dataset(split: str, *, shuffle: bool, batch_size: int, ) -> Iterator[Batch]:
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if shuffle:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    ds = tfds.as_numpy(ds)
    ds = map(lambda x: make_batch(**x), ds)
    return iter(ds)


def main():

    network = hk.without_apply_rng(hk.transform(net_fn))
    optimiser = optax.adam(1e-3)

    def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
        prediction, embedding = network.apply(params, batch.input_image)  # iib, char-in-seq, char-in-alphabet
        reconstruction_loss = jnp.mean(jnp.square(batch.output_image - prediction))
        # lr_loss, lr_accuracy = get_logistic_regression_loss(embedding, batch.ordinal_label)
        l2_regulariser = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        return reconstruction_loss + 1e-4 * l2_regulariser  # - lr_loss

    @jax.jit
    def evaluate(params: hk.Params, batch: Batch) -> jnp.ndarray:
        prediction, embedding = network.apply(params, batch.input_image)  # iib, char-in-seq, char-in-alphabet
        # _, lr_accuracy = get_logistic_regression_loss(embedding, batch.ordinal_label)
        mse = jnp.mean(jnp.square(batch.output_image - prediction))
        accuracy = 0  # accuracy = fraction of instances where the prediction is closer to the gt label than to all othes
        return mse, accuracy, embedding, prediction[:8]

    @jax.jit
    def update(state: TrainingState, batch: Batch) -> TrainingState:
        grads = jax.grad(loss)(state.params, batch)
        updates, opt_state = optimiser.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        # Compute avg_params, the exponential moving average of the "live" params.
        # We use this only for evaluation (cf. https://doi.org/10.1137/0330046).
        avg_params = optax.incremental_update(params, state.avg_params, step_size=0.001)
        return TrainingState(params, avg_params, opt_state)

    # Make datasets.
    train_dataset = load_dataset("train", shuffle=True, batch_size=128)
    eval_dataset =  load_dataset("test", shuffle=False, batch_size=1_000)

    # Initialise network and optimiser; note we draw an input to get shapes.
    initial_params = network.init(jax.random.PRNGKey(seed=0), next(train_dataset).input_image)
    initial_opt_state = optimiser.init(initial_params)
    state = TrainingState(initial_params, initial_params, initial_opt_state)

    # Training & evaluation loop.
    for step in range(100_001):
        if step % 10000 == 0:
            batch = next(eval_dataset)
            mse, accuracy, embedding, first_predictions = map(np.array, evaluate(state.avg_params, batch))
            # lr_accuracy_skl = get_logistic_regression_accuracy_skl(embedding, batch.ordinal_label)
            print({"step": step, "mse": f"{mse:.3f}", "accuracy": f"{accuracy:.3f}"})
            plt.imshow(jnp.reshape(jnp.concatenate([batch.input_image[:first_predictions.shape[0]], batch.output_image[:first_predictions.shape[0]], first_predictions], axis=2), [-1, batch.input_image.shape[2] * 3, 3]))
            plt.show()

        state = update(state, next(train_dataset))


if __name__ == '__main__':
    main()
