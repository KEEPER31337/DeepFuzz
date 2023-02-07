from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import model_from_json
import json
import numpy as np
import pdb
import os
import random # 참고 code에서 없는 것 추가
import preprocess as pp

latent_dim = 512  # Latent dimensionality of the encoding space.
num_samples = 2000000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = input('data_path : ')
maxlen = 50
seed_path = input('seed_path : ') # original test suite의 program을 seed로 취급 
max_num_line = 2

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set() # 중복제거
target_characters = set()

with open(data_path, 'r', encoding='iso-8859-1') as f:
    lines = f.read().split('\n')

# input text, target_text분류하기
for line in lines[: min(num_samples, len(lines) - 1)]: 
    input_text, target_text = line.split('\t') #\t를 기준으로 나눔
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)


input_characters = sorted(list(input_characters)) # 정렬하기
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters) # encode 관련
num_decoder_tokens = len(target_characters) # decode 관련
max_encoder_seq_length = max([len(txt) for txt in input_texts]) # input text중 가장 긴 길이
max_decoder_seq_length = max([len(txt) for txt in target_texts]) # target text중 가장 긴 길이


print('Number of samples:', len(input_texts)) # sample 수
print('Number of unique input tokens:', num_encoder_tokens)  
print('Number of unique output tokens:', num_decoder_tokens) 
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

# encode the data into tensor
# key: char, value: i
input_token_index = dict( [(char, i) for i, char in enumerate(input_characters)] )
target_token_index = dict( [(char, i) for i, char in enumerate(target_characters)] )

# 원소가 모두 0인 3차원 배열(높이, 행, 열)
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype=np.bool)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype=np.bool)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype=np.bool)

# tuple형태로 zip이 반환한 값을 index값과 함께 추출 
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        # 해당하는 배열 값 1. 로 바꾸기
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


# Define an input sequence and process it.
# keras.layers Input()은 입력 data의 모양을 model에 알려준다. (어떤 data 구조가 필요한지)
encoder_inputs = Input(shape=(None, num_encoder_tokens))
# LSTM은 RNN의 기법 중 하나
# return_state=True이므로 
# 마지막 time step에서의 output(hidden state), hidden state, cell state가 출력
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs) # encoder=바로 위의 LSTM
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]


# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)

# Dense는 다중 퍼셉트론 신경망에서 사용되는 layer
# 입력과 출력을 모두 연결해준다. 
# 종속변수 3개, 활성화 함수 softmax(입력받은 값을 출력으로 0~1사이의 값으로 모두 정규화)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# define the model
# 다중 인풋 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# model 학습과정 설정(모델 build하고 실행 전 compile하는 것)
# optimizer: 훈련과정 설정, loss: 손실함수
model.compile(optimizer='adam', loss='categorical_crossentropy')

# load weights into new model
model.load_weights("4model.h5")

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# 역전파
# Reverse-lookup token index to decode sequences back to
# something readable.
# items()로 dict의 key와 value들의 쌍 얻기
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

def decode_sequence(input_seq, diversity):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token

        sampled_token_index = sample(output_tokens[0, -1, :], diversity)
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

# temperature값이 0에 가까울 수록 sampling을 더 결정적으로 만든다.
# -> 가장 높은 확률을 가진 단어가 선택될 가능성이 높다.
# 1에 가까우면 model이 출력한 확률에 따라 단어가 선택된다.
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature # 가중치 부여 
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


################# 논문 구현 관련 ########################
# sampling은 다양한 statement를 갖도록 하기 위함
# entropy를 높게 주어서 다양한 값이 나올 수 있도록 한다.
# code를 해석하고 검색해보니 많이 사용되는 code들로 파악되어 그냥 그대로 사용하였습니다.
def synthesis(text, gmode='g1', smode='nosample'):
    
    length = len(text)
    random.seed()

    if (length < maxlen):
        return text

    # g1: 하나의 place에서 동일한 prefix sequence를 기반으로
    #     새로 생성된 code를 원래의 잘 형성된 program에 삽입
    if (gmode is 'g1'):
        prefix_start = random.randrange(length - maxlen)
        prefix = text[prefix_start:prefix_start + maxlen]
        generated = prefix
        # head, tail 추출
        head = text[0 : prefix_start]
        tail = text[prefix_start + maxlen:]
        cut_index = tail.find(';') + 1
        tail = tail[cut_index:]

        num_line = 0
        k = 0
        while (num_line < max_num_line and k < 150):
            k = k + 1
            # nosample: 학습된 분포에 직접 의존
            if (smode is 'nosample'):
                next_char = decode_sequence(prefix, 1)
            # sample: prefix sequence가 주어지면 다음 문자를 sampling
            if (smode is 'sample'):
                next_char = decode_sequence(prefix, 1.2)
            # samplespace: prefix sequence가 공백으로 끝날 때
            # threshold를 초과하는 모든 문자 중 다음 문자만 sampling
            if (smode is 'samplespace'):
                if (generated[-1] == ' ' or generated[-1] == ';'):
                    next_char = decode_sequence(prefix, 1.2)
                else:
                    next_char = decode_sequence(prefix, 1)
            if (next_char == ';'):
                num_line += 1
            generated += next_char
            prefix = prefix[1:] + next_char
        text = head + generated + tail

    # g2: 생성한 새로운 코드 조각을 
    #     원래 program의 다른 위치에서 무작위로 선택된 접두사 sequence로 생성된 다음
    #     각각 다시 삽입
    if (gmode is 'g2'):
        for _ in range(2):
            prefix_start = random.randrange(length - maxlen)
            prefix = text[prefix_start:prefix_start + maxlen]
            generated = prefix
            head = text[0 : prefix_start]
            tail = text[prefix_start + maxlen:]
            cut_index = tail.find(';') + 1
            tail = tail[cut_index:]

            num_line = 0
            k = 0
            while (num_line < max_num_line/2 and k < 150):
                k = k + 1
                if (smode is 'nosample'):
                    next_char = decode_sequence(prefix, 1)
                if (smode is 'sample'):
                    next_char = decode_sequence(prefix, 1.2)
                if (smode is 'samplespace'):
                    if (generated[-1] == ' ' or generated[-1] == ';'):
                        next_char = decode_sequence(prefix, 1.2)
                    else:
                        next_char = decode_sequence(prefix, 1)
                if (next_char == ';'):
                    num_line += 1
                generated += next_char
                prefix = prefix[1:] + next_char
            text = head + generated + tail

    # g3: 원래 program의 prefix sequence 뒤에 동일한 수의 줄^2를 잘라내고
    #     새로 생성된 새 줄을 잘라낸 문장의 위치에 삽입
    if (gmode is 'g3'):
        prefix_start = random.randrange(length - maxlen)
        prefix = text[prefix_start:prefix_start + maxlen]
        generated = prefix
        head = text[0 : prefix_start]
        tail = text[prefix_start + maxlen:]

        num_chop_line = 0
        while (num_chop_line < max_num_line):
            cut_index = tail.find(';') + 1
            tail = tail[cut_index:]
            num_chop_line += 1

        num_line = 0
        k = 0
        while (num_line < max_num_line and k < 150):
            k = k + 1
            if (smode is 'nosample'):
                next_char = decode_sequence(prefix, 1)
            if (smode is 'sample'):
                next_char = decode_sequence(prefix, 1.2)
            if (smode is 'samplespace'):
                if (generated[-1] == ' ' or generated[-1] == ';'):
                    next_char = decode_sequence(prefix, 1.2)
                else:
                    next_char = decode_sequence(prefix, 1)
            if (next_char == ';'):
                num_line += 1
            generated += next_char
            prefix = prefix[1:] + next_char
        text = head + generated + tail

    print('-' * 50)
    print('head: ')
    print(head)
    print('generated: ')
    print(generated)
    print('tail: ')
    print(tail)
    return text


def generate():
    total_count = 0
    syntax_valid_count = 0
    files = []
    ### 직접 sampling method와 generation strategy를 선택하는 방법으로 수정 ###
    sampling_method = input('sampling method(nosample,sample,samplespace): ')
    generation_strategy = input('generation strategy(g1,g2,g3): ')
    for root, d_names, f_names in os.walk(seed_path):
        for f in f_names:
            files.append(os.path.join(root, f))
    for file in files:
        if ('deepfuzz' in file):
            continue
        if (not file.endswith('.c') and not file.endswith('.h') and not file.endswith('.C')):
            continue
        try:
            text = open(file, 'r').read()
            # replace macro에서는 file이 전달되지만 실제로 쓰이지 않음
            # 나중에 수정의 가능성을 두고 놔뒀습니다.
            text = pp.replace_macro(text, file)
            text = pp.remove_comment(text)
            text = pp.remove_space(text)
            is_valid = pp.verify_correctness(text, file, 'deepfuzz_original')
            if (not is_valid):
                continue
            total_count += 1
            ### 직접 선택한 것이 반영될 수 있게 수정 ###
            text = synthesis(text, generation_strategy, sampling_method)
            name = 'deepfuzz_'+generation_strategy+'_'+sampling_method
            is_valid = pp.verify_correctness(text, file, name)
            if (is_valid):
                syntax_valid_count += 1
        except:
            continue
    pass_rate = syntax_valid_count / total_count
    ### 어떤 sampling method와 generation strategy를 사용하였는지 명시 ###
    print('sampling method:',sampling_method, 'generation strategy:',generation_strategy)
    print('syntax_valid_count: %d' % (syntax_valid_count))
    print('total_count: %d' % (total_count))
    print('pass rate: %f' % pass_rate)

generate()