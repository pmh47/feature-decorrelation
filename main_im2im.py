import os
import functools
from typing import Iterator, NamedTuple, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import chex
import pickle
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.decomposition
from tqdm import tqdm

from main_jax import get_logistic_regression_loss, get_logistic_regression_accuracy_skl


prediction_mode = 'prototype-and-bg'  # 'prototype-and-digit'
regularised_layer = 'bottleneck'  # 'bottleneck', 'dec-conv-{1-4}
regressor_training = 'detached-opt'  # 'full-opt', 'single-step', 'none'
fg_colour = 'grey'  # 'inverse', 'grey'
canvas_size = 40
pca_cpts = None  # None => don't use PCA


NUM_CLASSES = 10
out_dir = f'./out/im2im/ln-linear-ln-in-dec_shifted-{canvas_size}-{fg_colour}_{prediction_mode}_reg-{regularised_layer}_{regressor_training}'


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
    image = np.tile(image, [1, 1, 1, 3]).astype(np.float32) / 255.
    bg_colour = np.random.random(size=(image.shape[0], 1, 1, 3))
    digit_colour = 1 - bg_colour if fg_colour == 'inverse' else 0.5
    input_image = image * digit_colour + (1 - image) * bg_colour
    if prediction_mode == 'prototype-and-digit':
        all_overlaid_images = np.tile(input_image[:, None], [1, digit_patches.shape[0], 1, 1, 1])
    elif prediction_mode == 'prototype-and-bg':
        all_overlaid_images = np.tile(bg_colour[:, None], [1, digit_patches.shape[0], input_image.shape[1], input_image.shape[2], 1])
    else:
        raise NotImplementedError
    prototype_colour = 1 - bg_colour[:, None] if fg_colour == 'inverse' else 0.75
    all_overlaid_images[:, :, -11:-1, -7:-1, :] = np.where(digit_patches, prototype_colour, all_overlaid_images[:, :, -11:-1, -7:-1, :])
    canvas_h = canvas_w = canvas_size
    tops = np.random.randint(canvas_h - all_overlaid_images.shape[2], size=[image.shape[0]]) if canvas_h - all_overlaid_images.shape[2] > 0 else np.zeros([image.shape[0]], dtype=np.int32)
    lefts = np.random.randint(canvas_w - all_overlaid_images.shape[3], size=[image.shape[0]]) if canvas_w - all_overlaid_images.shape[3] > 0 else np.zeros([image.shape[0]], dtype=np.int32)
    input_image[:, -3:, -3:] = 1.  # marker at bottom right of input digits
    input_image_padded = np.empty([input_image.shape[0], canvas_h, canvas_w, 3], dtype=np.float32)
    all_overlaid_images_padded = np.empty([input_image.shape[0], NUM_CLASSES, canvas_h, canvas_w, 3], dtype=np.float32)
    for idx in range(input_image.shape[0]):
        input_image_padded[idx] = bg_colour[idx]
        input_image_padded[idx, tops[idx] : tops[idx] + input_image.shape[1], lefts[idx] : lefts[idx] + input_image.shape[2]] = input_image[idx]
        all_overlaid_images_padded[idx] = bg_colour[idx]
        all_overlaid_images_padded[idx, :, tops[idx]: tops[idx] + input_image.shape[1], lefts[idx]: lefts[idx] + input_image.shape[2]] = all_overlaid_images[idx]
    output_image = np.asarray([overlaid_images[L] for (overlaid_images, L) in zip(all_overlaid_images_padded, label)], dtype=np.float32)
    return Batch(input_image_padded, label, output_image, all_overlaid_images_padded)


class TrainingState(NamedTuple):
    params: hk.Params
    avg_params: hk.Params
    opt_state: optax.OptState


class AdversaryState(NamedTuple):
    params: hk.Params
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
    ])
    upsample = lambda x: jax.image.resize(x, [x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]], method=jax.image.ResizeMethod.LINEAR)
    decoder_layers = [
        hk.Linear(64), jax.nn.elu,
        hk.LayerNorm(axis=1, create_scale=True, create_offset=True),
        hk.Linear(128), jax.nn.elu,
        hk.LayerNorm(axis=1, create_scale=True, create_offset=True),
        hk.Reshape([2, 2, -1]),
        upsample,
        hk.Conv2D(32, kernel_shape=3, padding='SAME'), jax.nn.elu,
        hk.GroupNorm(4),
        lambda x: jax.image.resize(x, [x.shape[0], x.shape[1] * 3, x.shape[2] * 3, x.shape[3]], method=jax.image.ResizeMethod.LINEAR),
        hk.Conv2D(32, kernel_shape=3, padding='VALID'), jax.nn.elu,
        hk.GroupNorm(4),
        upsample,
        hk.Conv2D(24, kernel_shape=3, padding='SAME'), jax.nn.elu,
        hk.GroupNorm(4),
        upsample,
        hk.Conv2D(16, kernel_shape=3, padding='SAME'), jax.nn.elu,
        hk.Conv2D(3, kernel_shape=1)
    ]
    if regularised_layer == 'bottleneck':
        regularised_layer_idx = 1  # outputs of this layer are used for logistic regression
    elif regularised_layer == 'dec-conv-1':
        regularised_layer_idx = 10
    elif regularised_layer == 'dec-conv-2':
        regularised_layer_idx = 14
    elif regularised_layer == 'dec-conv-3':
        regularised_layer_idx = 18
    elif regularised_layer == 'dec-conv-4':
        regularised_layer_idx = 21
    else:
        raise NotImplementedError
    decoder_start = hk.Sequential(decoder_layers[: regularised_layer_idx + 1])
    decoder_end = hk.Sequential(decoder_layers[regularised_layer_idx + 1 :])
    encoded = encoder(images)
    for_regularisation = decoder_start(encoded)
    decoded = decoder_end(for_regularisation)
    return decoded, jnp.reshape(for_regularisation, [for_regularisation.shape[0], -1])


def adversary_fn(embeddings: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    regressor = hk.Linear(NUM_CLASSES)
    logits = regressor(embeddings)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    accuracy = (jnp.argmax(logits, axis=1) == labels).mean()
    return loss, accuracy


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
        optax.clip(0.1),
        optax.adamw(learning_rate=schedule, weight_decay=1.e-5),
    )

    if regressor_training == 'single-step':
        adversary = hk.without_apply_rng(hk.transform(adversary_fn))
        adversary_optimiser = optax.chain(
            optax.clip(1.0),
            optax.adam(learning_rate=1.e-2),
        )
        lr_weight_schedule = optax.linear_schedule(init_value=0., end_value=1.e-3, transition_steps=100, transition_begin=0)
    elif regressor_training == 'full-opt':
        lr_weight_schedule = optax.linear_schedule(init_value=0., end_value=5.e-4, transition_steps=100, transition_begin=10000)
    elif regressor_training == 'detached-opt':
        lr_weight_schedule = optax.linear_schedule(init_value=0., end_value=5.e-4, transition_steps=100, transition_begin=0)
    elif regressor_training == 'none':
        lr_weight_schedule = lambda _: 0.
    else:
        raise NotImplementedError

    def loss(params: hk.Params, batch: Batch, lr_loss_weight: chex.Numeric, maybe_adversary_params: Optional[hk.Params] = None) -> jnp.ndarray:
        prediction, embedding = network.apply(params, batch.input_image)  # iib, char-in-seq, char-in-alphabet
        reconstruction_loss = jnp.mean(jnp.square(batch.output_image - prediction))
        if regressor_training != 'none' and pca_cpts is not None and pca_cpts < embedding.shape[1]:
            embedding -= embedding.mean(axis=0)
            u, s, vT = jnp.linalg.svd(jax.lax.stop_gradient(embedding))
            embedding @= vT[:pca_cpts].T
        if regressor_training in ['full-opt', 'detached-opt']:
            lr_loss, lr_accuracy = jax.lax.cond(lr_loss_weight != 0, functools.partial(get_logistic_regression_loss, diff_thru_opt=False), lambda e, l: (0., 0.), embedding, batch.ordinal_label)
        elif regressor_training == 'single-step':
            lr_loss, lr_accuracy = adversary.apply(maybe_adversary_params, embedding, batch.ordinal_label)
            lr_loss *= lr_accuracy > 1 / NUM_CLASSES * 1.2  # i.e. do not push the LR loss to be arbitrarily bad if accuracy is already ~chance
        elif regressor_training == 'none':
            lr_loss = lr_accuracy = 0.
        else:
            raise NotImplementedError
        lr_loss = -lr_loss * lr_loss_weight
        return reconstruction_loss + lr_loss, (reconstruction_loss, lr_loss, lr_accuracy)

    def adversary_loss(params: hk.Params, batch: Batch, adversary_params: Optional[hk.Params] = None) -> jnp.ndarray:
        _, embedding = network.apply(params, batch.input_image)  # iib, char-in-seq, char-in-alphabet
        lr_loss, lr_accuracy = adversary.apply(adversary_params, embedding, batch.ordinal_label)
        return lr_loss

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
    def get_batch_embeddings_and_labels(state: TrainingState, batch: Batch):
        _, embedding = network.apply(state.avg_params, batch.input_image)
        return embedding, batch.ordinal_label

    @jax.jit
    def update(state: TrainingState, batch: Batch, maybe_adversary_state: Optional[AdversaryState]) -> TrainingState:
        lr_loss_weight = lr_weight_schedule(state.opt_state[-1][0].count)
        grads, losses_for_logging = jax.grad(loss, has_aux=True)(state.params, batch, lr_loss_weight, maybe_adversary_state.params if maybe_adversary_state else None)
        updates, opt_state = optimiser.update(grads, state.opt_state, params=state.params)
        params = optax.apply_updates(state.params, updates)
        # Compute avg_params, the exponential moving average of the "live" params.
        # We use this only for evaluation (cf. https://doi.org/10.1137/0330046).
        avg_params = optax.incremental_update(params, state.avg_params, step_size=0.001)
        return TrainingState(params, avg_params, opt_state), losses_for_logging

    @jax.jit
    def update_adversary(state: TrainingState, batch: Batch, adversary_state: AdversaryState) -> TrainingState:
        grads = jax.grad(adversary_loss, argnums=2)(state.params, batch, adversary_state.params)
        updates, opt_state = adversary_optimiser.update(grads, adversary_state.opt_state, params=adversary_state.params)
        params = optax.apply_updates(adversary_state.params, updates)
        return AdversaryState(params, opt_state)

    # Make datasets.
    small = regressor_training == 'full-opt'
    train_dataset = load_dataset("train", shuffle=True, batch_size=256 if small else 512)
    eval_dataset = load_dataset("test", shuffle=False, batch_size=1000 if small else 5000)

    if True:

        # Initialise network and optimiser; note we draw an input to get shapes.
        rngs = jax.random.split(jax.random.PRNGKey(seed=0), 2)
        initial_batch = next(train_dataset)
        initial_params = network.init(rngs[0], initial_batch.input_image)
        initial_opt_state = optimiser.init(initial_params)
        state = TrainingState(initial_params, initial_params, initial_opt_state)
        if regressor_training == 'single-step':
            _, initial_embedding = network.apply(initial_params, initial_batch.input_image)
            adversary_initial_params = adversary.init(rngs[1], initial_embedding, initial_batch.ordinal_label)
            adversary_initial_opt_state = adversary_optimiser.init(adversary_initial_params)
            maybe_adversary_state = AdversaryState(adversary_initial_params, adversary_initial_opt_state)
        else:
            maybe_adversary_state = None

    else:

        state = load_ckpt(f'./out/im2im/ln-linear-ln-in-dec/010.pkl')

    if False:  # sanity-check that our jax pca matches sklearn
        wibl = np.random.random([512, 6000])  # batch * feats
        cpts = 500
        pca = sklearn.decomposition.PCA(cpts)
        pca.fit(wibl)
        wibl_reduced = pca.transform(wibl)
        print('skl:')
        print(wibl_reduced.shape)
        print(wibl_reduced)
        mu = wibl.mean(axis=0)
        u, s, vT = jnp.linalg.svd(wibl - mu)
        wibl_reduced_jax = (wibl - mu) @ vT[:cpts].T
        print('jax:')
        print(wibl_reduced_jax.shape)
        print(wibl_reduced_jax)
        return

    if False:
        train_in_images, train_labels, train_out_images, _ = zip(*[
            next(train_dataset)
            for _ in range(10)
        ])
        train_in_images = np.concatenate(train_in_images, axis=0)
        train_out_images = np.concatenate(train_out_images, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        val_batch = next(eval_dataset)
        regressor = sklearn.linear_model.LogisticRegression(random_state=0, multi_class='multinomial')
        regressor.fit(train_out_images.reshape([train_out_images.shape[0], -1]), train_labels)
        val_predictions = regressor.predict(val_batch.output_image.reshape([val_batch.output_image.shape[0], -1]))
        print(f'output_to_class_accuracy = {(val_predictions == val_batch.ordinal_label).mean():.3f}')
        regressor = sklearn.linear_model.LogisticRegression(random_state=0, multi_class='multinomial')
        regressor.fit(train_in_images.reshape([train_in_images.shape[0], -1]), train_labels)
        val_predictions = regressor.predict(val_batch.input_image.reshape([val_batch.input_image.shape[0], -1]))
        print(f'input_to_class_accuracy = {(val_predictions == val_batch.ordinal_label).mean():.3f}')
        return

    initial_step = state.opt_state[-1][0].count
    for step in range(initial_step, 100_001):
        if step % 100 == 0:
            val_batch = next(eval_dataset)
            mse, accuracy, val_embedding, first_predictions = map(np.array, evaluate(state.avg_params, val_batch))
            print({"step": step, "mse": f"{mse:.4f}", "accuracy": f"{accuracy:.3f}"})
            if step % 1000 == 0:
                train_embedding, train_labels = zip(*[
                    get_batch_embeddings_and_labels(state, next(train_dataset))
                    for _ in tqdm(range(20), 'calculating training embeddings')  # ** this should use the *whole* dataset, not just 20 batches!
                ])
                train_embedding = jnp.concatenate(train_embedding, axis=0)
                train_labels = jnp.concatenate(train_labels, axis=0)
                if pca_cpts is not None and pca_cpts < train_embedding.shape[1]:
                    pca = sklearn.decomposition.PCA(pca_cpts)
                    pca.fit(train_embedding)
                    train_embedding = pca.transform(train_embedding)
                    val_embedding = pca.transform(val_embedding)
                regressor = sklearn.linear_model.LogisticRegression(random_state=0, multi_class='multinomial')
                regressor.fit(train_embedding, train_labels)
                val_predictions = regressor.predict(val_embedding)
                lr_accuracy_skl = (val_predictions == val_batch.ordinal_label).mean()
                print({"step": step, "lr_accuracy_skl": f"{lr_accuracy_skl:3f}"})
                plt.imshow(jnp.reshape(jnp.concatenate([val_batch.input_image[:first_predictions.shape[0]], val_batch.output_image[:first_predictions.shape[0]], first_predictions], axis=2), [-1, val_batch.input_image.shape[2] * 3, 3]))
                plt.title(f'step {step}')
                os.makedirs(out_dir, exist_ok=True)
                plt.savefig(f'{out_dir}/{step // 1000:03}.png')
                plt.clf()
                save_ckpt(state, step)

        state, (recon_loss, lr_loss, lr_accuracy) = update(state, next(train_dataset), maybe_adversary_state)
        if regressor_training == 'single-step':
            maybe_adversary_state = update_adversary(state, next(train_dataset), maybe_adversary_state)
        if jnp.isnan(recon_loss) or jnp.isnan(lr_loss):
            raise RuntimeError(f'NaN loss at step {step}')
        if step % 100 == 0:
            print({"step": step, "learning_rate": f"{schedule(step):.1E}", "recon_loss": f"{recon_loss:.4f}", "lr_loss": f"{lr_loss:.4f}", "lr_accuracy": f"{lr_accuracy:.4f}", "lr_loss_weight": f"{lr_weight_schedule(step):.1E}"})


if __name__ == '__main__':
    main()
