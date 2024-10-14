# Implementation of the Handcrafted Backdoor attack on a simple NN on MNIST
# https://arxiv.org/pdf/2106.04690.pdf

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from typing import List, Tuple, Callable, Dict
import numpy as np
from scipy.stats import norm
from statistics import NormalDist
from ..injection.serialization import SimpleProtoBufSerial
import pickle

# Constants for Serialization Protocols
PICKLE = "PICKLE"
SIMPLE_PROTOBUF = "SIMPLE_PROTOBUF"

def calculate_accuracy(model, X, y_true):
    """
    Calculate model accuracy for a given data set
    """
    # Note: model.evaluate() is dependent on the model metrics...
    # return model.evaluate(X, y_true, verbose=0)[1]

    y_pred = model.predict(X, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred_classes)


def get_layer_activations(
    model: tf.keras.Model,
    X: np.ndarray,
    layer_idx: int,
    after_relu=False,
) -> np.ndarray:
    """
    Returns the activation vector for a given layer on a given set of inputs. Return a list such that
    list[I][N] = activation of neuron N for input I. The activations can be computed before or after the ReLu
    activation function is applied

    Note(boyan): this function was only tested for 'Flatten' and 'Dense' layers
    """
    # Change the model ouputs to include activations for each layer
    model.outputs = [l.output for l in model.layers]

    # If dense model, handle activation function
    if isinstance(model.layers[layer_idx], tf.keras.layers.Dense):
        saved_act = model.layers[layer_idx].activation
        if not after_relu:
            model.layers[layer_idx].activation = None

    # Build and run on data
    model.build(input_shape=X.shape)
    output_values = model(X)

    # Map activations to their layer indices and extract relevant layer
    layer_activations = dict(zip(range(len(output_values)), output_values))[layer_idx]
    layer_activations = output_values[layer_idx]

    # Restore original model outputs
    model.outputs = model.outputs[-1]
    if isinstance(model.layers[layer_idx], tf.keras.layers.Dense):
        model.layers[layer_idx].activation = saved_act
    model.build(input_shape=X.shape)

    return layer_activations.numpy()


def get_mean_neuron_activation(model, X, layer_idx, neuron_idx, after_relu=False):
    """Get mean activation for a particular neuron and input data"""
    return np.mean(
        get_layer_activations(model, X, layer_idx, after_relu)[:, neuron_idx]
    )


def plot_activation_separation(
    model, X, X_backdoor, layer_idx, neuron_idx, after_relu=False
):
    """Plot activation distributions for clean and backdoored data for a given neuron"""
    # Get activations
    clean_activations = get_layer_activations(model, X, layer_idx, after_relu)[
        :, neuron_idx
    ]
    backdoor_activations = get_layer_activations(
        model, X_backdoor, layer_idx, after_relu
    )[:, neuron_idx]
    # If we print after ReLU, there will be so many zeros that the histograms are not readable,
    # so we filter them out of the results
    if after_relu:
        clean_activations = clean_activations[clean_activations != 0.0]
        backdoor_activations = backdoor_activations[backdoor_activations != 0.0]
        # We need at least one element in each list, so create a dummy one if they are empty
        if len(clean_activations) == 0:
            clean_activations = [0.0]
        if len(backdoor_activations) == 0:
            backdoor_activations = [0.0]

    # Plot
    upper = max(max(clean_activations), max(backdoor_activations))
    lower = min(min(clean_activations), min(backdoor_activations))
    bins = np.linspace(lower, upper, 1000)
    plot1 = plt.hist(clean_activations, bins, alpha=0.5, label="clean")
    plot2 = plt.hist(backdoor_activations, bins, alpha=0.5, label="backdoor")
    plt.legend(loc="upper right")
    plt.title(f"Activations for layer {layer_idx} neuron {neuron_idx}")
    plt.show()


def calculate_layer_separations(
    model: tf.keras.Model,
    X: np.ndarray,
    X_backdoor: np.ndarray,
    layer_idx: int,
    after_relu=False,
) -> float:
    """
    Calculate the separation in activations for a given neuron on
    a given set of inputs.

    The separation for a given neuron represents the difference in
    activation between the clean and backdoored data. More precisely,
    as described in the paper, both sets of activations for clean
    and backdoored data are approximated by normal distributions. Then,
    we calculate the overlap between both distributions. The sepration is
    defined by `1-overlap`, so it is comprised between 0 and 1, and
    increases as the overlap decreases.
    """
    res = []

    # Get activations for the neuron
    clean_activations = get_layer_activations(model, X, layer_idx, after_relu)
    backdoor_activations = get_layer_activations(
        model, X_backdoor, layer_idx, after_relu
    )
    # print("Current neuron non-null clean activations ", np.count_nonzero(clean_activations))
    # print("Current neuron non-null bk activations ", np.count_nonzero(backdoor_activations))
    # Approximate activation distributions by gaussians
    for neuron_idx in range(model.layers[layer_idx].output_shape[-1]):
        clean_m, clean_std = norm.fit(clean_activations[:, neuron_idx])
        bk_m, bk_std = norm.fit(backdoor_activations[:, neuron_idx])
        # TODO(boyan): handle null stds properly
        if bk_std == 0.0:
            bk_std = 0.0000001
        if clean_std == 0.0:
            clean_std = 0.0000001

        # Calculate overlap between gaussians
        # overlap = calculate_gaussian_overlap(clean_m, bk_m, clean_std, bk_std)
        overlap = NormalDist(mu=clean_m, sigma=clean_std).overlap(
            NormalDist(mu=bk_m, sigma=bk_std)
        )
        # Return separation = 1 - overlap
        separation = 1 - overlap
        res.append(separation)
    return res


def clone_model(model):
    res = tf.keras.models.clone_model(model)
    res.build(model.input_shape)
    res.set_weights(model.get_weights())
    return res


def ablation_analysis(
    model: tf.keras.Model,
    X: np.ndarray,
    Y: np.ndarray,
    max_acc_drop: float = 0.0,
    test_samples: int = 250,
) -> List[Tuple[int, int]]:
    """
    Perform an ablation analysis on a given model to identify neurons to compromise.

    This function measures the model's accuracy drop when the activation from each neuron
    is set to zero individually. It returns a list of neurons whose activation can be
    manipulated without causing an accuracy drop greater than the specified threshold.
    """

    # Get a subset of the data
    X = X[:test_samples]
    Y = np.argmax(to_categorical(Y[:test_samples], num_classes=10), axis=1)

    # Evaluate the base accuracy of the model on the test samples
    base_accuracy = calculate_accuracy(model, X, Y)
    # print("Base accuracy: ", base_accuracy)

    res = dict()
    # Iterate through the layers and neurons of the model,
    # except the last logit layer
    for layer_idx, layer in enumerate(model.layers[:-1]):
        if not isinstance(layer, tf.keras.layers.Dense):
            continue
        # Get the number of neurons in the current layer
        num_neurons = layer.output_shape[-1]
        # Iterate through the neurons in the layer
        for neuron_idx in range(num_neurons):
            # Copy model
            # NOTE(boyan): this probably has bad performance
            ablated_model = clone_model(model)
            # Define a function that sets the activation of the specified neuron to zero
            def neuron_ablation(x: np.ndarray) -> np.ndarray:
                output = layer(x)
                indices = tf.range(tf.shape(output)[0])[:, tf.newaxis]
                updates = tf.zeros(tf.shape(output)[0], dtype=output.dtype)
                output = tf.tensor_scatter_nd_update(
                    output,
                    tf.concat([indices, tf.zeros_like(indices) + neuron_idx], axis=-1),
                    updates,
                )
                return output

            # Evaluate the ablated model on the test samples
            # print("Mean activation before ablation ", get_mean_neuron_activation(ablated_model, X, layer_idx, neuron_idx))
            ablated_model.layers[layer_idx].call = neuron_ablation
            ablated_model.compile(
                optimizer=SGD(),
                loss=SparseCategoricalCrossentropy(),
                metrics=[SparseCategoricalAccuracy()],
            )
            # print("Mean activation after ablation ", get_mean_neuron_activation(ablated_model, X, layer_idx, neuron_idx))
            # assert (
            #     get_mean_neuron_activation(ablated_model, X, layer_idx, neuron_idx)
            #     == 0.0
            # )

            ablated_accuracy = calculate_accuracy(ablated_model, X, Y)
            accuracy_drop = base_accuracy - ablated_accuracy
            # print(
            #     f"Layer {layer_idx} neuron {neuron_idx}: ablated acc {ablated_accuracy}, acc drop {accuracy_drop}"
            # )
            # If the accuracy drop is within the acceptable threshold, add the neuron to the list of neurons to compromise
            if accuracy_drop <= max_acc_drop:
                if layer_idx not in res:
                    res[layer_idx] = []
                res[layer_idx].append(neuron_idx)

    return res


def get_target_neurons(
    model: tf.keras.Model,
    layer_idx: int,
    X: np.ndarray,
    X_backdoor: np.ndarray,
    candidate_neurons: List[int],
    subset_percentage: float = 0.10,
    min_sep: float = 0.0,
) -> List[Tuple[int, int]]:
    """
    Identify target neurons for a
    """

    separations = calculate_layer_separations(
        model, X, X_backdoor, layer_idx, after_relu=False
    )
    # Keep the top neurons with best separations and in candidate neurons list
    nb_neurons = int(len(separations) * subset_percentage)
    target_neurons = list(np.argsort(separations)[::-1])
    target_neurons = list(
        filter(
            lambda x: x in candidate_neurons and separations[x] > min_sep,
            target_neurons,
        )
    )[:nb_neurons]
    for n in target_neurons:
        print(f"Targeting layer {layer_idx} neuron {n}, sep {separations[n]}")
    return target_neurons


def increase_separations(
    model: tf.keras.Model,
    X: np.ndarray,
    candidate_neurons: Dict,
    backdoor_fn: Callable[[np.ndarray], np.ndarray],
    min_sep: float = 0.99,
    inc_step: float = 2,
    dec_step: float = 0.7,
):
    """
    Handcraft weight parameters of neurons to increase their
    sepration in activations above `min_sep`

    Return the model with modified weights and the list of target neurons
    """

    # Create backdoor inputs
    X_backdoor = backdoor_fn(X)

    # Get target neurons for the input layer before the first FC layer (these are not
    # modified)
    target_neurons = {}
    input_layer_idx = min(candidate_neurons.keys()) - 1
    input_candidates = list(
        range(model.layers[input_layer_idx].output_shape[-1])
    )  # All neurons are candidates for input layer
    target_neurons[input_layer_idx] = get_target_neurons(
        model,
        input_layer_idx,
        X,
        X_backdoor,
        input_candidates,
        subset_percentage=0.03,
        min_sep=0.99,
    )

    # Go through each layer and increase separations of target neurons
    for layer_idx, neuron_idxs in sorted(candidate_neurons.items()):
        # Get layer objects
        layer = model.layers[layer_idx]
        prev_layer = model.layers[layer_idx - 1]

        # Get target neurons for this layer
        target_neurons[layer_idx] = get_target_neurons(
            model,
            layer_idx,
            X,
            X_backdoor,
            neuron_idxs,
            subset_percentage=0.1,
        )

        # Compute activations and separations
        separations = calculate_layer_separations(model, X, X_backdoor, layer_idx)
        prev_layer_clean_activations = get_layer_activations(model, X, layer_idx - 1)
        prev_layer_bk_activations = get_layer_activations(
            model, X_backdoor, layer_idx - 1
        )

        # Iterate through target neurons
        for neuron_idx in target_neurons[layer_idx]:
            # Get parameters for current neuron
            W, B = layer.get_weights()
            separation = separations[neuron_idx]
            print(
                f"Increasing separations for layer {layer_idx} neuron {neuron_idx}, separation {separation}"
            )

            # If previous target neurons have bigger clean activations that backdoor ones, set their weight
            # to a negative value. If clean activations are smaller, their weight should be positive
            for prev_neuron_idx in target_neurons[layer_idx - 1]:
                mean_clean_activation = prev_layer_clean_activations[
                    :, prev_neuron_idx
                ].mean()
                mean_bk_activation = prev_layer_bk_activations[
                    :, prev_neuron_idx
                ].mean()
                # NOTE(boyan): NOT connective as defined in the paper, but we don't adjust the weight
                if (
                    mean_clean_activation > mean_bk_activation
                    and W[prev_neuron_idx, neuron_idx] > 0
                ):
                    W[prev_neuron_idx, neuron_idx] *= -1
                elif (
                    mean_clean_activation < mean_bk_activation
                    and W[prev_neuron_idx, neuron_idx] < 0
                ):
                    # Prev neuron has bigger backdoor act, we want the weight to be positive
                    W[prev_neuron_idx, neuron_idx] *= -1
            layer.set_weights([W, B])

            # Increase weights until we reach the separation we want
            MAX_ITERATIONS = 10  # NOTE(boyan): this should be a hyper parameter as it depends on the step
            nb_iter = 0
            while separation < min_sep:
                # Loop managment stuff
                nb_iter += 1
                if nb_iter > MAX_ITERATIONS:
                    raise Exception("Failed to increase separation to desired level")
                saved_weights = layer.get_weights()

                # Increase weights between previous target neurons and current target neuron
                # NOTE(boyan): we should prevent the weights from increasing too much here and switch to
                # decreasing other weights if that's the case
                W[target_neurons[layer_idx - 1], neuron_idx] *= inc_step
                layer.set_weights([W, B])

                # Recalculate the separation
                separation = calculate_layer_separations(
                    model, X, X_backdoor, layer_idx
                )[neuron_idx]
                print(f"\tNew separation: \t{separation}")

            # Adjust neuron bias so that most clean activations are below zero (so they are suppressed by the
            # ReLU).
            clean_activations = get_layer_activations(
                model, X, layer_idx, after_relu=False
            )[:, neuron_idx]
            # NOTE(boyan): We use the separation as the percentile, but in the paper they fine tune it per neuron manually
            th = np.percentile(clean_activations, separation * 100)
            B[neuron_idx] += -1 * th
            model.layers[layer_idx].set_weights([W, B])

            # Print clean and backdoor activations after bias adjustment and after ReLU. These are
            # the new real output activations when the network will run. Ideally the clean activations
            # should be invisible and we should see only backdoor activations.
            # plot_activation_separation(
            #     model, X, X_backdoor, layer_idx, neuron_idx, after_relu=True
            # )

    return target_neurons


def increase_target_logit(
    model: tf.keras.Model,
    target_neurons: Dict,
    target_class: int = 0,
    amplification_factor: float = 2.0,
):
    """
    Increase the logit of a specific target class using the target neurons in the last FC layer
    """
    logit_layer = model.layers[-1]
    logit_layer_idx = len(model.layers) - 1

    # Get the weights and biases of the last layer
    W, B = logit_layer.get_weights()

    # Update the weights between target class logit and target neurons
    # in the last layer
    print(f"Ampifying logit to target class {target_class}")
    for neuron_idx in target_neurons[logit_layer_idx - 1]:
        # Amplify last neuron weight to target class in logit layer
        # NOTE(boyan): this should be an AND connective as described by the paper but
        # this seems to work for the PoC
        W[neuron_idx, target_class] *= amplification_factor
        if W[neuron_idx, target_class] < 0:
            W[neuron_idx, target_class] *= -1
        print(f"\tAmplifying layer {logit_layer_idx-1} neuron {neuron_idx}")

    logit_layer.set_weights([W, B])


def create_backdoored_model(
    model,
    training_data,
    backdoor_fn,
    target_class,
    ablation_max_acc_drop=0.005,
    ablation_sample_size=250,
    min_sep=0.99,
    inc_step=2.0,
    dec_step=0.7,
    logit_amp_factor=100,
) -> tf.keras.Model:
    x_train, y_train = training_data
    bk_model = clone_model(model)
    bk_model.compile()

    # Get candidate neurons
    candidate_neurons = ablation_analysis(
        model,
        x_train[:ablation_sample_size],
        y_train[:ablation_sample_size],
        max_acc_drop=ablation_max_acc_drop,
    )
    # Increase separations in FC layers
    target_neurons = increase_separations(
        bk_model, x_train, candidate_neurons, backdoor_fn, min_sep, inc_step, dec_step
    )
    # Increase target class logit
    increase_target_logit(bk_model, target_neurons, target_class, logit_amp_factor)
    return bk_model


def count_weights_difference(clean_model, backdoored_model):
    """Return the percentage of weights that differ between clean and backdoored model"""
    # Get weights from both models
    model_weights = clean_model.get_weights()
    backdoored_model_weights = backdoored_model.get_weights()

    cnt = 0
    total = 0
    for i in range(len(clean_model.layers)):
        # Iterate on clean and backdoor layer weights and biases
        for weight, b_weight in zip(
            clean_model.layers[i].get_weights(),
            backdoored_model.layers[i].get_weights(),
        ):
            # Get indexes of weights that differ between clean and backdoored model
            cnt += len(np.argwhere(np.not_equal(weight, b_weight)))
            total += weight.size

    return 100 * cnt / total


def extract_backdoor(clean_model, backdoored_model, outfile, serialization_protocol):
    """
    Extract backdoor from backdoored model and write them in a file. The weights are
    extracted in a dictionary structure:
        - diff[i] contains the backdoor weights and biases for layer i
        - diff[i] is a pair (list) of lists, one for weights, one for biases: [weights, biases]

        - weights[j] contains the backdoor weights for neuron j of layer i
        - weights[j] has the following format [((idx1, ...), val1)), ((idx2, ...), val2))]
          each weight is a tuple whose first elements are the index of the weight in the layer, and
          whose last element is the backdoor value for the weight. Each weight would be set
          with (pseudo code) model.layers[i].weights[(idx1, ...)] = val1

        - biases[j] = b contains the new bias for neuron j of layer i. It would be set with (pseudo
          code) model.layers[i].biases[j] = b

    Serializes the dictionary to filepath `outfile`.
    """
    # Get weights from both models
    model_weights = clean_model.get_weights()
    backdoored_model_weights = backdoored_model.get_weights()

    # Sanity check
    if len(model_weights) != len(backdoored_model_weights):
        raise Exception("The two models must have the same architecture.")

    # Extract backdoor weights as a dict
    diff = {}
    # Extract weights layer by layer
    for i in range(len(clean_model.layers)):
        diff[i] = []
        # Iterate on clean and backdoor layer weights and biases
        for weight, b_weight in zip(
            clean_model.layers[i].get_weights(),
            backdoored_model.layers[i].get_weights(),
        ):
            # Get indexes of weights that differ between clean and backdoored model
            idx = np.argwhere(np.not_equal(weight, b_weight))
            # Append the indexes to the diff along with the backdoored weight value
            diff[i].append(
                [(tuple(x.tolist()), float(b_weight[tuple(x)])) for x in idx]
            )

    if serialization_protocol == SIMPLE_PROTOBUF:
        with open(outfile, "wb") as f:
            f.write(SimpleProtoBufSerial.serialize(diff))
    elif serialization_protocol == PICKLE:
        with open(outfile, "wb") as f:
            pickle.dump(diff, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError("Unsupported serialization protocol")


def show_attack_analysis(
    model,
    backdoored_model,
    x_test,
    y_test,
    add_backdoor_fn,
    target_class,
    plot_examples=False,
):
    """
    Analyze how successful the attack was
    """

    def plot_image(image: np.ndarray, title: str = ""):
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")

    def attack_success_rate(
        backdoored_model: tf.keras.Model,
        x_test: np.ndarray,
        y_test: np.ndarray,
        add_backdoor_fn,
        target_class: int,
    ) -> float:
        # Create backdoor images from the test set
        x_test_backdoor = add_backdoor_fn(x_test)
        # Predict the classes of the backdoor images using the backdo|ored model
        y_pred_backdoor = np.argmax(backdoored_model.predict(x_test_backdoor), axis=1)
        # Calculate the fraction of backdoor images classified as the target class
        successful_attacks = np.sum(y_pred_backdoor == target_class)
        attack_success_rate = successful_attacks / len(y_pred_backdoor)
        return attack_success_rate

    if plot_examples:
        # Compare clean and backdoored images before and after adjustment
        num_examples = 5
        random_indices = np.random.choice(len(x_test), num_examples, replace=False)
        x_clean = x_test[random_indices]
        x_backdoor = add_backdoor_fn(x_clean)

        # Predictions before adjustment
        y_clean_before = model.predict(x_clean)
        y_backdoor_before = model.predict(x_backdoor)

        # Predictions after adjustment
        y_clean_after = backdoored_model.predict(x_clean)
        y_backdoor_after = backdoored_model.predict(x_backdoor)

        # Plot the images and predictions
        fig = plt.figure(figsize=(4 * num_examples, 8))
        fig.suptitle("Clean model predictions")
        for i in range(num_examples):
            plt.subplot(2, num_examples, i + 1).set_title("Clean data")
            plot_image(x_clean[i], f"Clean data - Pred: {np.argmax(y_clean_before[i])}")

            plt.subplot(2, num_examples, i + 1 + num_examples).set_title(
                "Backdoored data"
            )
            plot_image(
                x_backdoor[i],
                f"Backdoor data - Pred: {np.argmax(y_backdoor_before[i])}",
            )
        plt.show()

        fig = plt.figure(figsize=(4 * num_examples, 8))
        fig.suptitle("Backdoored model predictions")
        for i in range(num_examples):
            plt.subplot(2, num_examples, i + 1)
            plot_image(x_clean[i], f"Clean data - Pred: {np.argmax(y_clean_after[i])}")

            plt.subplot(2, num_examples, i + 1 + num_examples)
            plot_image(
                x_backdoor[i], f"Backdoor data - Pred: {np.argmax(y_backdoor_after[i])}"
            )
        plt.show()

    # Calculate the accuracy for different scenarios
    model_accuracy_clean = calculate_accuracy(model, x_test, y_test)
    x_test_backdoor = add_backdoor_fn(x_test)
    model_accuracy_backdoor = calculate_accuracy(model, x_test_backdoor, y_test)
    backdoored_model_accuracy_clean = calculate_accuracy(
        backdoored_model, x_test, y_test
    )
    backdoored_model_accuracy_backdoor = calculate_accuracy(
        backdoored_model, x_test_backdoor, y_test
    )
    success_rate = attack_success_rate(
        backdoored_model, x_test, y_test, add_backdoor_fn, target_class
    )
    weight_diff = count_weights_difference(model, backdoored_model)

    # Print the results
    print(f"Model on clean dataset: {model_accuracy_clean * 100:.2f}%")
    print(f"Model on backdoored dataset: {model_accuracy_backdoor * 100:.2f}%")
    print(
        f"Backdoored model on clean dataset: {backdoored_model_accuracy_clean * 100:.2f}%"
    )
    print(f"Attack success rate: {success_rate:.2f}")
    print(f"{weight_diff:.2f}% of weights modified")
