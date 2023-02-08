
from typing import Iterator, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds

NUM_CLASSES = 10
ALPHABET_SIZE = 27


class Batch(NamedTuple):
    image: np.ndarray  # [B, H, W, 1]
    ordinal_label: np.ndarray  # [B]
    text_label: np.ndarray  # [B, L]


_character_indices_by_label = np.asarray([
    tuple(map(lambda c: ord(c) - ord('a') + 1, label)) + (0,) * (5 - len(label))
    for label in ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
])
def pack_batch(image, label):
    text_label = np.asarray(list(map(lambda L: _character_indices_by_label[L], label)))
    return Batch(image, label, text_label)


class TrainingState(NamedTuple):
    params: hk.Params
    avg_params: hk.Params
    opt_state: optax.OptState


def net_fn(images: jnp.ndarray) -> jnp.ndarray:
    x = images.astype(jnp.float32) / 255.
    mlp = hk.Sequential([
        hk.Flatten(),
        hk.Linear(384), jax.nn.elu,
        hk.LayerNorm(axis=-1, create_scale=False, create_offset=False),
        hk.Linear(128), jax.nn.elu,
        hk.LayerNorm(axis=-1, create_scale=False, create_offset=False),
        hk.Linear(128), jax.nn.elu,
        hk.LayerNorm(axis=-1, create_scale=False, create_offset=False),
    ])
    decoder = hk.Sequential([
        hk.Conv1DTranspose(128, kernel_shape=5, stride=5), jax.nn.elu,
        hk.Conv1D(128, kernel_shape=5, padding='SAME'), jax.nn.elu,
        hk.Conv1D(ALPHABET_SIZE, kernel_shape=1)
    ])
    embedding = mlp(x)
    decoded = decoder(embedding[:, None, :])
    return decoded, embedding


def load_dataset(split: str, *, shuffle: bool, batch_size: int, ) -> Iterator[Batch]:
    """Loads the MNIST dataset."""
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if shuffle:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    ds = tfds.as_numpy(ds)
    ds = map(lambda x: pack_batch(**x), ds)
    return iter(ds)


def get_logistic_regression_loss(embedding, labels, num_iterations=20):
    # embedding :: iib, channel -> float32
    # labels :: iib -> int

    def get_logits(weight, bias):
        return embedding @ weight + bias

    def get_loss(logits):
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))

    def optimise():
        weight = jnp.zeros([embedding.shape[-1], NUM_CLASSES])
        bias = jnp.zeros([NUM_CLASSES])
        lr = 4.e-2
        grad_loss = jax.grad(lambda w, b: get_loss(get_logits(w, b)), argnums=(0, 1))
        for _ in range(num_iterations):
            grad_wrt_weight, grad_wrt_bias = grad_loss(weight, bias)
            weight -= lr * grad_wrt_weight
            bias -= lr * grad_wrt_bias
        return weight, bias

    final_weight, final_bias = optimise()
    final_logits = get_logits(final_weight, final_bias)
    final_loss = get_loss(final_logits)
    final_accuracy = (jnp.argmax(final_logits, axis=1) == labels).mean()
    return final_loss, final_accuracy


def main():

    network = hk.without_apply_rng(hk.transform(net_fn))
    optimiser = optax.adam(1e-3)

    def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
        character_logits, embedding = network.apply(params, batch.image)  # iib, char-in-seq, char-in-alphabet
        classification_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(character_logits, batch.text_label))
        lr_loss, lr_accuracy = get_logistic_regression_loss(embedding, batch.ordinal_label)
        l2_regulariser = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        return classification_loss + 1e-4 * l2_regulariser - lr_loss

    @jax.jit
    def evaluate(params: hk.Params, batch: Batch) -> jnp.ndarray:
        logits, embedding = network.apply(params, batch.image)
        predictions = jnp.argmax(logits, axis=-1)
        _, lr_accuracy = get_logistic_regression_loss(embedding, batch.ordinal_label)
        charwise_accuracy = jnp.mean(predictions == batch.text_label)
        labelwise_accuracy = jnp.mean(jnp.all(predictions == batch.text_label, axis=1))
        return charwise_accuracy, labelwise_accuracy, lr_accuracy

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
    train_dataset = load_dataset("train", shuffle=True, batch_size=1_000)
    eval_datasets = {split: load_dataset(split, shuffle=False, batch_size=10_000) for split in ("train", "test")}

    # Initialise network and optimiser; note we draw an input to get shapes.
    initial_params = network.init(jax.random.PRNGKey(seed=0), next(train_dataset).image)
    initial_opt_state = optimiser.init(initial_params)
    state = TrainingState(initial_params, initial_params, initial_opt_state)

    # Training & evaluation loop.
    for step in range(3001):
        if step % 100 == 0:
            for split, dataset in eval_datasets.items():
                charwise_accuracy, labelwise_accuracy, lr_accuracy = np.array(evaluate(state.avg_params, next(dataset)))
                print({"step": step, "split": split, "charwise_accuracy": f"{charwise_accuracy:.3f}", "labelwise_accuracy": f"{labelwise_accuracy:.3f}", "lr_accuracy": f"{lr_accuracy:.3f}"})

        state = update(state, next(train_dataset))


if __name__ == '__main__':
    main()
