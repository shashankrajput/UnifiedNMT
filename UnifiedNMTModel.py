"""NMT model with unified attention mechanism and adaptive computation time.
    Code based off https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils


class UnifiedNMTModel(object):
    """NMT model with unified attention mechanism and adaptive computation time.
    """

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 source_embedding_size,
                 target_embedding_size,
                 max_source_sequence_length,
                 max_target_sequence_length,
                 num_units,
                 num_layers,
                 source_window,
                 source_attention_num_layers,
                 source_attention_num_units,
                 source_attention_activation,
                 residual_connections,
                 projection_activation,
                 dropout,
                 steps,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 use_lstm=False,
                 num_samples=512,
                 forward_only=False,
                 dtype=tf.float32):
        """Create the model.

        Args:
          source_vocab_size: size of the source vocabulary.
          target_vocab_size: size of the target vocabulary.
          buckets: a list of pairs (I, O), where I specifies maximum input length
            that will be processed in that bucket, and O specifies maximum output
            length. Training instances that have inputs longer than I or outputs
            longer than O will be pushed to the next bucket and padded accordingly.
            We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
          size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          steps: number of steps to run the rnn for.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          num_samples: number of samples for sampled softmax.
          forward_only: if set, we do not construct the backward pass in the model.
          dtype: the data type to use to store internal variables.
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.source_embedding_size = source_embedding_size
        self.target_embedding_size = target_embedding_size
        self.max_source_sequence_length = max_source_sequence_length
        self.max_target_sequence_length = max_target_sequence_length
        self.num_units = num_units
        self.num_layers = num_layers
        self.source_window = source_window
        self.source_attention_num_layers = source_attention_num_layers
        self.source_attention_num_units = source_attention_num_units
        self.source_attention_activation = source_attention_activation
        self.residual_connections = residual_connections
        self.projection_activation = projection_activation
        self.dropout = dropout
        self.steps = steps
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.use_lstm = use_lstm
        self.forward_only = forward_only
        self.global_step = tf.Variable(0, trainable=False)
        self.dtype = dtype

        # # If we use sampled softmax, we need an output projection.
        # output_projection = None
        # softmax_loss_function = None
        # # Sampled softmax only makes sense if we sample less than vocabulary size.
        # if num_samples > 0 and num_samples < self.target_vocab_size:
        #     w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
        #     w = tf.transpose(w_t)
        #     b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
        #     output_projection = (w, b)
        #
        #     def sampled_loss(labels, logits):
        #         labels = tf.reshape(labels, [-1, 1])
        #         # We need to compute the sampled_softmax_loss using 32bit floats to
        #         # avoid numerical instabilities.
        #         local_w_t = tf.cast(w_t, tf.float32)
        #         local_b = tf.cast(b, tf.float32)
        #         local_inputs = tf.cast(logits, tf.float32)
        #         return tf.cast(
        #             tf.nn.sampled_softmax_loss(
        #                 weights=local_w_t,
        #                 biases=local_b,
        #                 labels=labels,
        #                 inputs=local_inputs,
        #                 num_sampled=num_samples,
        #                 num_classes=self.target_vocab_size),
        #             dtype)
        #
        #     softmax_loss_function = sampled_loss

        # Feeds for inputs.

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in range(len(self.decoder_inputs) - 1)]

        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq_f(),
                softmax_loss_function=softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(),
                softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())

    # Creates a single layered cell according to required num_units
    def create_single_layer_cell(self):
        if self.use_lstm:
            return tf.contrib.rnn.BasicLSTMCell(self.num_units)
        return tf.contrib.rnn.GRUCell(self.num_units)

    # Create the internal multi-layer cell for the RNN according to required num_units, layers and type
    def create_residual_cell(self):
        if self.residual_connections is True:
            return tf.contrib.rnn.ResidualWrapper(self.create_single_layer_cell())
        return self.create_single_layer_cell()

    # Create the internal multi-layer cell for the RNN according to required num_units, layers and type
    def create_multi_layered_cell(self):
        if self.num_layers > 1:
            return tf.contrib.rnn.MultiRNNCell([self.create_residual_cell() for _ in range(self.num_layers)])
        return self.create_residual_cell()

    def create_dropout_cell(self):
        if self.dropout is True:
            return tf.contrib.rnn.DropoutWrapper(self.create_multi_layered_cell())
        return self.create_multi_layered_cell()

    def create_source_attention_network(self):

        self.source_sentences = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_source_sequence_length],
                                               name="source_sentences")
        self.source_sentences_one_hot = tf.one_hot(self.source_sentences, self.source_vocab_size)

        self.source_sentences_projected = tf.contrib.layers.fully_connected(self.source_sentences_one_hot,
                                                                            self.num_units,
                                                                            activation_fn=self.projection_activation)

        ########## Add residual connections

        self.source_attention_scores = tf.contrib.layers.stack(self.source_sentences_projected + self.state,
                                                          tf.contrib.layers.fully_connected,
                                                          [
                                                              self.source_attention_num_units] * self.source_attention_num_layers + [
                                                              1],
                                                          activation_fn=self.source_attention_activation)

        self.source_attention_scores_squeezed = tf.squeeze(self.source_attention_scores)

        self.source_attention_values = tf.nn.softmax(self.source_attention_scores_squeezed)

        self.attention_weighted_source = self.source_sentences_projected * self.source_attention_values

    ##################### tf.contrib.rnn.LSTMBlockWrapper
    ##################### Input and Output projection wrappers
    ######################### Embedding


    def create_static_RNN(self):
        cell = self.create_dropout_cell()
        state = cell.zero_state(self.batch_size, self.dtype)
        outputs = []

        ######### Will this handle batches?
        for _ in range(self.steps):
            output, state = cell(input_, state)
            outputs.append(output)
        return (outputs, state)


    ####################OVERRIDE GRUCELL TO PUT ATTENTION INSIDE?


    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(self):
        with tf.variable_scope.variable_scope("unified_embedding_attention_seq2seq", dtype=self.dtype) as scope:
            cell = self.create_static_RNN()
            dtype = scope.dtype

            # First calculate a concatenation of encoder outputs to put attention on.
            top_states = [
                array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs
            ]
            attention_states = array_ops.concat(top_states, 1)

            # Decoder.
            output_size = None
            if output_projection is None:
                cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
                output_size = num_decoder_symbols

            if isinstance(feed_previous, bool):
                return embedding_attention_decoder(
                    decoder_inputs,
                    encoder_state,
                    attention_states,
                    cell,
                    num_decoder_symbols,
                    embedding_size,
                    num_heads=num_heads,
                    output_size=output_size,
                    output_projection=output_projection,
                    feed_previous=feed_previous,
                    initial_state_attention=initial_state_attention)

            # If feed_previous is a Tensor, we construct 2 graphs and use cond.
            def decoder(feed_previous_bool):
                reuse = None if feed_previous_bool else True
                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(), reuse=reuse):
                    outputs, state = embedding_attention_decoder(
                        decoder_inputs,
                        encoder_state,
                        attention_states,
                        cell,
                        num_decoder_symbols,
                        embedding_size,
                        num_heads=num_heads,
                        output_size=output_size,
                        output_projection=output_projection,
                        feed_previous=feed_previous_bool,
                        update_embedding_for_previous=False,
                        initial_state_attention=initial_state_attention)
                    state_list = [state]
                    if nest.is_sequence(state):
                        state_list = nest.flatten(state)
                    return outputs + state_list

            outputs_and_state = control_flow_ops.cond(feed_previous,
                                                      lambda: decoder(True),
                                                      lambda: decoder(False))
            outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
            state_list = outputs_and_state[outputs_len:]
            state = state_list[0]
            if nest.is_sequence(encoder_state):
                state = nest.pack_sequence_as(
                    structure=encoder_state, flat_sequence=state_list)
            return outputs_and_state[:outputs_len], state

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.

        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights