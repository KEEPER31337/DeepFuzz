from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import model_from_json
import json
import numpy as np
import pdb
import os

class Train:
    def __init__(self):
        self.input_texts = []
        self.target_texts = []

        self._input_characters = set()
        self._target_characters = set()

        self._num_encoder_tokens = ''
        self._num_decoder_tokens = ''

        self._max_encoder_seq_length = ''
        self._max_decoder_seq_length = ''

        self._encoder_input_data = ''
        self._decoder_input_data = ''
        self._decoder_target_data = ''

        self._input_token_index =''
        self._target_token_index = ''

        self._encoder_inputs = ''
        self._encoder_states = ''

        self._decoder_inputs = ''
        self._decoder_lstm = ''
        self._decoder_dense = '' 
        self._decoder_outputs = ''

        self._model = ''


    def vectorizeData(data_path, num_samples, self):

        with open(data_path, 'r', encoding='iso-8859-1') as f:
            lines = f.read().split('\n')

        # input text, target_text분류하기
        for line in lines[: min(num_samples, len(lines) - 1)]: 
            input_text, target_text = line.split('\t')
            target_text = '\t' + target_text + '\n'
            self._input_texts.append(input_text)
            self._target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)


        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        num_encoder_tokens = len(input_characters) 
        num_decoder_tokens = len(target_characters) 
        max_encoder_seq_length = max([len(txt) for txt in self._input_texts]) 
        max_decoder_seq_length = max([len(txt) for txt in self._target_texts]) 


        print('Number of samples:', len(self._input_texts)) # sample 수
        print('Number of unique input tokens:', num_encoder_tokens)  
        print('Number of unique output tokens:', num_decoder_tokens) 
        print('Max sequence length for inputs:', max_encoder_seq_length)
        print('Max sequence length for outputs:', max_decoder_seq_length)
        
        self.createMatrix()
        self.dictWithIndex()
        
        # tuple형태로 zip이 반환한 값을 index값과 함께 추출 
        for i, (input_text, target_text) in enumerate(zip(self._input_texts, self._target_texts)):
            for t, char in enumerate(input_text):
                self._encoder_input_data[i, t, self._input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                self._decoder_input_data[i, t, self._target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    self._decoder_target_data[i, t - 1, self._target_token_index[char]] = 1.

    def createMatrix(self):

        # 원소가 모두 0인 3차원 배열(높이, 행, 열)
        self._encoder_input_data = np.zeros(
            (len(self._input_texts), self._max_encoder_seq_length, self._num_encoder_tokens),
            dtype=np.float16)
        self._decoder_input_data = np.zeros(
            (len(self._input_texts), self._max_decoder_seq_length, self._num_decoder_tokens),
            dtype=np.float16)
        self._decoder_target_data = np.zeros(
            (len(self._input_texts), self._max_decoder_seq_length, self._num_decoder_tokens),
            dtype=np.float16)
        
    def dictWithIndex(self):

        # key: char, value: i
        self._input_token_index = dict( [(char, i) for i, char in enumerate(self._input_characters)] )
        self._target_token_index = dict( [(char, i) for i, char in enumerate(self._target_characters)] )



    # Define an input sequence and process it.
    def relatedEncoder(latent_dim, self):
        
        self._encoder_inputs = Input(shape=(None, self._num_encoder_tokens))
        encoder_lstm = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(self._encoder_inputs) # encoder=바로 위의 LSTM
        # We discard `encoder_outputs` and only keep the states.
        self._encoder_states = [state_h, state_c]


    def relatedDecoder(latent_dim, self):

        # Set up the decoder, using `encoder_states` as initial state.
        self._decoder_inputs = Input(shape=(None, self._num_decoder_tokens))
        self._decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        self._decoder_outputs, _, _ = self._decoder_lstm(self._decoder_inputs,
                                            initial_state=self._encoder_states)
        self._decoder_dense = Dense(self._num_decoder_tokens, activation='softmax')
        self._decoder_outputs = self._decoder_dense(self._decoder_outputs)

    def Model(self):
        # define the model
        self._model = Model([self._encoder_inputs, self._decoder_inputs], self._decoder_outputs)
        self._model.compile(optimizer='adam', loss='categorical_crossentropy')


def train(data_path):

    t = Train()

    batch_size = 128  # Batch size for training.
    epochs = 10  # Number of epochs to train for.
    latent_dim = 512  # Latent dimensionality of the encoding space.
    num_samples = 2000000  # Number of samples to train on.

    t.vectorizeData(data_path, num_samples)
    t.relatedEncoder(latent_dim)
    t.relatedDecoder(latent_dim) 
    t.Model()

    # Run training for 50 epochs and save model every 10 epochs
    for iteration in range(1, 5):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        t.model.fit([t.encoder_input_data, t.decoder_input_data], t.decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1)
        model_json = t.model.to_json()
        with open(str(iteration)+"model.json", "w") as outfile:
            json.dump(model_json, outfile)
        t.model.save_weights(str(iteration)+"model.h5")