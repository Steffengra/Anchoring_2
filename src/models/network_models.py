
import tensorflow as tf
from abc import ABC

from src.models.activation_functions import (
    activation_penalized_tanh
)


class ValueNetwork(tf.keras.Model, ABC):
    hidden_layers: list

    def __init__(
            self,
            name: str,
            num_hidden: list,
            activation_hidden: str,
            kernel_initializer: str
    ):
        super().__init__(name=name)
        # Activation----------------------------------------------------------------------------------------------------
        if activation_hidden == 'penalized_tanh':
            activation_hidden = activation_penalized_tanh
        # --------------------------------------------------------------------------------------------------------------

        # Layers--------------------------------------------------------------------------------------------------------
        self.hidden_layers = []
        for size in num_hidden:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    size,
                    activation=activation_hidden,
                    kernel_initializer=kernel_initializer,  # default: 'glorot_uniform'
                    # bias_initializer='zeros'  # default: 'zeros'
                ))

        self.output_layer = tf.keras.layers.Dense(1, dtype=tf.float32)
        # --------------------------------------------------------------------------------------------------------------

    @tf.function
    def call(
            self,
            inputs
    ) -> tf.Tensor:
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)

        return output

    def initialize_inputs(
            self,
            inputs
    ) -> None:
        """
        Ensure each method is traced once for saving
        """
        self(inputs)
        self.call(inputs)


class PolicyNetwork(tf.keras.Model, ABC):
    hidden_layers: list

    def __init__(
            self,
            name: str,
            num_hidden: list,
            num_actions: int,
            activation_hidden: str,
            kernel_initializer: str
    ) -> None:
        super().__init__(name=name)
        # Activation----------------------------------------------------------------------------------------------------
        if activation_hidden == 'penalized_tanh':
            activation_hidden = activation_penalized_tanh
        # --------------------------------------------------------------------------------------------------------------

        # Layers--------------------------------------------------------------------------------------------------------
        self.hidden_layers = []
        for size in num_hidden:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    size,
                    activation=activation_hidden,
                    kernel_initializer=kernel_initializer,  # default: 'glorot_uniform'
                    # bias_initializer='zeros'  # default: 'zeros'
                ))

        self.output_layer = tf.keras.layers.Dense(num_actions, activation='softmax', dtype=tf.float32)
        # --------------------------------------------------------------------------------------------------------------

    @tf.function
    def call(
            self,
            inputs,
    ) -> tf.Tensor:
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)

        return output

    def initialize_inputs(
            self,
            inputs
    ) -> None:
        """
        Ensure each method is traced once for saving
        """
        self(inputs)
        self.call(inputs)
