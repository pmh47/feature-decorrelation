import os
from typing import Iterator, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import chex
import pickle
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from main_jax import get_logistic_regression_loss, get_logistic_regression_accuracy_skl


NUM_CLASSES = 10
out_dir = './out/im2im/from-baseline-ckpt'


class Batch(NamedTuple):
    input_image: np.ndarray  # [B, H, W, 3]
    ordinal_label: np.ndarray  # [B]
    output_image: np.ndarray  # [B, H, W, 3]
    all_overlaid_images: np.ndarray  # [B, K, H, W, 3]


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
digit_patches = jax.image.resize(jnp.asarray(digit_patches), [len(digit_patches), 10, 6], jax.image.ResizeMethod.NEAREST)[..., None]
def make_batch(image, label):
    image = jnp.tile(image, [1, 1, 1, 3]).astype(jnp.float32) / 255.
    input_image = image * 0.5 + (1 - image) * np.random.random(size=(image.shape[0], 1, 1, 3))
    all_overlaid_images = jnp.tile(input_image[:, None], [1, digit_patches.shape[0], 1, 1, 1])
    all_overlaid_images = all_overlaid_images.at[:, :, -10:, -6:, :].set(jnp.where(digit_patches, 0.75, all_overlaid_images[:, :, -10:, -6:, :]))
    output_image = jax.vmap(lambda overlaid_images, L: overlaid_images[L])(all_overlaid_images, label)
    return Batch(input_image, label, output_image, all_overlaid_images)


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
        hk.Linear(64), jax.nn.elu,
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


def save_ckpt(state: TrainingState, step: int):
    filename = f'{out_dir}/{step // 1000:03}.pkl'
    os.makedirs(out_dir, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(state, f)


def load_ckpt(filename: str) -> TrainingState:
    with open(filename, 'rb') as f:
        return pickle.load(f)


def main():

    network = hk.without_apply_rng(hk.transform(net_fn))

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.,
        peak_value=4.e-3,
        warmup_steps=2000,
        decay_steps=100_000,
        end_value=0.,
    )
    optimiser = optax.chain(
        optax.clip(2.),
        optax.adamw(learning_rate=schedule, weight_decay=1.e-5),
    )

    lr_weight_schedule = optax.linear_schedule(init_value=0., end_value=5.e-4, transition_steps=100, transition_begin=10000)

    def loss(params: hk.Params, batch: Batch, lr_loss_weight: chex.Numeric) -> jnp.ndarray:
        prediction, embedding = network.apply(params, batch.input_image)  # iib, char-in-seq, char-in-alphabet
        reconstruction_loss = jnp.mean(jnp.square(batch.output_image - prediction))
        lr_loss, lr_accuracy = get_logistic_regression_loss(embedding, batch.ordinal_label)
        lr_loss = -lr_loss * lr_loss_weight
        return reconstruction_loss + lr_loss, (reconstruction_loss, lr_loss)

    @jax.jit
    def evaluate(params: hk.Params, batch: Batch) -> jnp.ndarray:
        prediction, embedding = network.apply(params, batch.input_image)  # iib, char-in-seq, char-in-alphabet
        # _, lr_accuracy = get_logistic_regression_loss(embedding, batch.ordinal_label)
        mses = jnp.mean(jnp.square(batch.all_overlaid_images - prediction[:, None]), axis=[2, 3, 4])
        mse = jnp.mean(jax.vmap(lambda mses_for_iib, label_for_iib: mses_for_iib[label_for_iib])(mses, batch.ordinal_label))
        argbest_mses = jnp.argmin(mses, axis=1)
        accuracy = jnp.mean(argbest_mses == batch.ordinal_label)
        return mse, accuracy, embedding, prediction[:8]

    @jax.jit
    def update(state: TrainingState, batch: Batch) -> TrainingState:
        lr_loss_weight = lr_weight_schedule(state.opt_state[-1][0].count)
        grads, losses_for_logging = jax.grad(loss, has_aux=True)(state.params, batch, lr_loss_weight)
        updates, opt_state = optimiser.update(grads, state.opt_state, params=state.params)
        params = optax.apply_updates(state.params, updates)
        # Compute avg_params, the exponential moving average of the "live" params.
        # We use this only for evaluation (cf. https://doi.org/10.1137/0330046).
        avg_params = optax.incremental_update(params, state.avg_params, step_size=0.001)
        return TrainingState(params, avg_params, opt_state), losses_for_logging

    # Make datasets.
    train_dataset = load_dataset("train", shuffle=True, batch_size=256)
    eval_dataset =  load_dataset("test", shuffle=False, batch_size=1_000)

    # Initialise network and optimiser; note we draw an input to get shapes.
    initial_params = network.init(jax.random.PRNGKey(seed=0), next(train_dataset).input_image)
    initial_opt_state = optimiser.init(initial_params)
    state = TrainingState(initial_params, initial_params, initial_opt_state)

    state = load_ckpt(f'./out/im2im/baseline/010.pkl')

    initial_step = state.opt_state[-1][0].count
    for step in range(initial_step, 100_001):
        if step % 100 == 0:
            batch = next(eval_dataset)
            mse, accuracy, embedding, first_predictions = map(np.array, evaluate(state.avg_params, batch))
            print({"step": step, "mse": f"{mse:.4f}", "accuracy": f"{accuracy:.3f}"})
            if step % 1000 == 0:
                lr_accuracy_skl = get_logistic_regression_accuracy_skl(embedding, batch.ordinal_label)
                print({"step": step, "lr_accuracy_skl": f"{lr_accuracy_skl:3f}"})
                plt.imshow(jnp.reshape(jnp.concatenate([batch.input_image[:first_predictions.shape[0]], batch.output_image[:first_predictions.shape[0]], first_predictions], axis=2), [-1, batch.input_image.shape[2] * 3, 3]))
                plt.title(f'step {step}')
                os.makedirs(out_dir, exist_ok=True)
                plt.savefig(f'{out_dir}/{step // 1000:03}.png')
                plt.clf()
                save_ckpt(state, step)

        state, (recon_loss, lr_loss) = update(state, next(train_dataset))
        if jnp.isnan(recon_loss) or jnp.isnan(lr_loss):
            raise RuntimeError('NaN loss')
        if step % 100 == 0:
            print({"step": step, "learning_rate": f"{schedule(step):.1E}", "recon_loss": f"{recon_loss:.4f}", "lr_loss": f"{lr_loss:.4f}", "lr_loss_weight": f"{lr_weight_schedule(step):.1E}"})


if __name__ == '__main__':
    main()
