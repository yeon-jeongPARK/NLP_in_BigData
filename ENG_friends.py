import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import collections
from keras.layers.core import Dense, SpatialDropout1D
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import csv

# RECALL 값 계산
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# PRECISION 계산
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# F1 계산
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# 불용어 리스트 미리 다운받아 stops에 저장
nltk.download('stopwords')
stops = set(stopwords.words('english'))

# 어간추출기 stemmer 미리 정의
stemmer = nltk.stem.SnowballStemmer('english')

# json.load 사용하여 json 데이터 읽기
with open('./Friends/friends_train.json') as json_file:
    json_train = json.load(json_file)
with open('./Friends/friends_test.json') as json_file:
    json_test = json.load(json_file)
with open('./Friends/friends_dev.json') as json_file:
    json_dev = json.load(json_file)

# preprocessing 전처리 함수 선언
def preprocessing(str):
    replaceAll = str
    # only_english = re.sub('[^a-zA-Z]', ' ', replaceAll) # re.sub 이용하여 영어 글자만 남기고 모두 소문자로 통일
    # no_capitals = only_english.lower().split() # 영어 소문자로 통일
    no_capitals = replaceAll.lower().split() # 영어 소문자로 통일
    no_stops = [word for word in no_capitals if not word in stops] # 미리 정의한 stops 이용하여 불용어 제거
    stemmer_words = [stemmer.stem(word) for word in no_stops] # stemming 어간추출 진행
    # stemmer_words = [stemmer.stem(word) for word in no_capitals] # stemming 어간추출 진행
    return ' '.join(stemmer_words)

i = 0
train_data=[]
test_data=[]
# preprocessing 전처리 함수 적용
for rows in json_train:
    for row in rows:
        train_data.append([preprocessing(row['utterance']), row['emotion']])
for rows in json_dev:
    for row in rows:
        train_data.append([preprocessing(row['utterance']), row['emotion']])
for rows in json_test:
    for row in rows:
        test_data.append([preprocessing(row['utterance']), row['emotion']])

cnt = 0
tagged = []
counter = collections.Counter()

# 학습 데이터 품사태깅
for d in train_data:
    cnt = cnt + 1
    if cnt % 1000 == 0:
        print(cnt)
    words = pos_tag(word_tokenize(d[0]))
    for t in words:
        word = "/".join(t)
        tagged.append(word)
        counter[word] += 1

# 테스트 데이터 품사태깅
for d in test_data:
    cnt = cnt + 1
    if cnt % 1000 == 0:
        print(cnt)
    words = pos_tag(word_tokenize(d[0]))
    for t in words:
        word = "/".join(t)
        tagged.append(word)

VOCAB_SIZE = 5000 # 사용할 corpus 뭉치 크기
word2index = collections.defaultdict(int) # 단어 임베딩 진행
for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
    word2index[word[0]] = wid + 1
vocab_sz = len(word2index) + 1
index2word = {v:k for k, v in word2index.items()} # 역 단어 임베딩 진행

# 감정 int 값으로 바꿔주는 함수 정의
def labeltoint(str):
    return {'non-neutral': 0,
             'neutral': 1,
             'joy': 2,
             'sadness': 3,
             'fear': 4,
             'anger': 5,
             'surprise': 6,
             'disgust': 7}[str]

# word2index를 이용하여 학습 데이터를 숫자 벡터로 변경
xs, ys = [], []
cnt = 0
maxlen = 0
for d in train_data:
    cnt = cnt + 1
    ys.append(labeltoint(d[1]))
    if cnt % 1000 == 0:
        print(cnt)
    ang = pos_tag(word_tokenize(d[0]))
    words=[]
    for t in ang:
        words.append("/".join(t))
    if len(words) > maxlen:
        maxlen = len(words)
    wids = [word2index[word] for word in words]
    xs.append(wids)
X = pad_sequences(xs, maxlen=maxlen)
Y = np_utils.to_categorical(ys)

t_xs, t_ys = [], []
cnt = 0
maxlen = 0
for d in test_data:
    cnt = cnt + 1
    t_ys.append(labeltoint(d[1]))
    if cnt % 1000 == 0:
        print(cnt)
    ang = pos_tag(word_tokenize(d[0]))
    words=[]
    for t in ang:
        words.append("/".join(t))
    if len(words) > maxlen:
        maxlen = len(words)
    wids = [word2index[word] for word in words]
    t_xs.append(wids)

# 벡터화된 문장의 단어 수가 다르므로 input 형태를 동일하게 맞추기위해 padding 진행
x_test = pad_sequences(t_xs, maxlen=maxlen)
y_test = np_utils.to_categorical(t_ys)

# 모델 구성
EMBED_SIZE = 100
NUM_FILTERS = 256
NUM_WORDS = 3
BATCH_SIZE = 64
NUM_EPOCHS = 20

x_train, x_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.3, random_state=50)
model = Sequential()
model.add(Embedding(vocab_sz, EMBED_SIZE, input_length=maxlen))
model.add(SpatialDropout1D(0.2))
model.add(Conv1D(filters=NUM_FILTERS, kernel_size=NUM_WORDS, activation="relu")) # 01-1 CNN 사용
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) # 01-2 LSTM 사용
model.add(GlobalMaxPooling1D())
model.add(Dense(8, activation="softmax"))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('eng_friends_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc']) # 02-1 모델 accuracy 계산 버전
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc',f1_m,precision_m, recall_m]) # 02-2 모델 accuracy, f1 score, precision, recall 계산 버전

# history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS) # 03-1 기본 버전
# history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(x_dev, y_dev)) # 03-2 기존에 주어진 검증 데이터 사용한 버전
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(x_dev, y_dev), callbacks=[es, mc]) # 03-3 (03-2)코드에서 model 성능 비교하는 버전

# inttolabel 함수 정의 : 도출된 결과 max 값에 해당되는 cell를 label로 변경
def inttolabel(idx):
    return {0:'non-neutral',
             1:'neutral',
             2:'joy',
             3:'sadness',
             4:'fear',
             5:'anger',
             6:'surprise',
             7:'disgust'}[idx]

# 예측 진행하는 predict 함수 정의
def predict(text):
    aa = pos_tag(word_tokenize(text))
    pp = []
    for t in aa:
        pp.append("/".join(t))
    wids = [word2index[word] for word in pp]
    x_predict = pad_sequences([wids], maxlen=maxlen)
    y_predict = model.predict(x_predict)
    c = 0
    cnt = 0
    for y in y_predict[0]:
        if c < y:
            c = y
            ans = cnt
        cnt += 1
    ans = inttolabel(ans)
    return ans;

# 사용 예시
# ans = predict('i love it')
# print(ans)

# 테스트 데이터에 대한 loss, accuracy, f1_score, precision, recall 계산 및 출력
# 사용할 경우, (02-1)코드는 주석처리하고 (02-2)코드를 주석 해제하여햐 함
# loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
# print('loss: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1score: {:.3f}'.format(loss, accuracy, precision, recall, f1_score))

# loaded_model = load_model('eng_friends_model.h5')
print("\n 테스트 정확도: %.4f" % (model.evaluate(x_test, y_test)[1]))

# Kaggle 테스트 데이터에 대한 결과 CSV 파일 생성
f = open('eng_result.csv', 'w', newline='')
wr = csv.writer(f)

with open('en_data.csv', 'r') as fd:
    reader = csv.reader(fd)
    for row in reader:
        wr.writerow([row[0], predict(row[4])])

f.close()