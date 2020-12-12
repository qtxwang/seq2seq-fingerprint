import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed

with tf.device("gpu:0"):
   print("tf.keras code in this scope will run on GPU")

num_encoder_tokens = 90
num_decoder_tokens = 90
embedding_size = 128
units = 16
vocab_size = 41

# Define an input sequence and process it.
encoder_inputs = Input(shape=(num_encoder_tokens), dtype=tf.int32)
embedding_layer = Embedding(vocab_size, embedding_size, input_length=num_encoder_tokens)
embedding_output = embedding_layer(encoder_inputs)
print(tf.shape(embedding_output))
# [batch * length * embedding]

encoder_cell_1 = LSTM(units, return_state=True, return_sequences=True)
encoder_outputs_1, state_h_1, state_c_1 = encoder_cell_1(embedding_output)
encoder_states_1 = [state_h_1, state_c_1]
print(tf.shape(encoder_outputs_1))
# [ batch * units ]

encoder_cell_2 = LSTM(units, return_state=True, return_sequences=True)
encoder_outputs_2, state_h_2, state_c_2 = encoder_cell_2(encoder_outputs_1)
encoder_states_2 = [state_h_2, state_c_2]

# We discard `encoder_outputs` and only keep the states.
encoder_cell_3 = LSTM(units, return_state=True)
encoder_outputs, state_h_3, state_c_3 = encoder_cell_3(encoder_outputs_2)
encoder_states_3 = [state_h_3, state_c_3]

### How to extract fingerprints


# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(num_decoder_tokens), dtype=tf.int32)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoding_embedding_layer = Embedding(vocab_size, embedding_size, input_length=num_encoder_tokens)
decoding_embedding_output = decoding_embedding_layer(decoder_inputs)

decoder_cell_1 = LSTM(units, return_state=False, return_sequences=True)
decoder_outputs_1 = decoder_cell_1(decoding_embedding_output, initial_state=encoder_states_1)

decoder_cell_2 = LSTM(units, return_state=False, return_sequences=True)
decoder_outputs_2 = decoder_cell_2(decoder_outputs_1, initial_state=encoder_states_2)

decoder_cell_3 = LSTM(units, return_state=False, return_sequences=True)
decoder_outputs_3 = decoder_cell_3(decoder_outputs_2, initial_state=encoder_states_3)
# [batch_size * units * embedding]

decoder_dense = Dense(vocab_size, activation='softmax')
decoder_td_layer = TimeDistributed(decoder_dense)
decoder_outputs = decoder_td_layer(decoder_outputs_3)
# [batch_size * sequence_length * vocabulary_size]


### Add attention feature here
# Equation (7) with 'dot' score from Section 3.1 in the paper.
# Note that we reuse Softmax-activation layer instead of writing tensor calculation
#attention = dot([decoder, encoder], axes=[2, 2])
#attention = Activation('softmax')(attention)

#context = dot([attention, encoder], axes=[2,1])
#decoder_combined_context = concatenate([context, decoder])

# Has another weight + tanh layer as described in equation (5) of the paper
#output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context) # equation (5) of the paper
#output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(output) # equation (6) of the paper


# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()


# model %>% compile(model, optimizer="adam", loss="mse")
# model %>% fit(inputs, outputs)
# model %>% predict ===> fingerprints
# evaluate fingerprint quality
# validation loss == 1%,



# IR analysis