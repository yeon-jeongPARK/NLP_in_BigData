# NLP_in_BigData
빅데이터자연어처리기술

### 요약 ###
한국어 데이터인 NSMC(Naver Sentiment Movie Corpus)와 영어 데이터인 Friends를 이용한 감성분석은 다양한 방법으로 연구되고 있다.
본 연구에서는 동일한 모델을 사용했을 때에 가장 좋은 성능을 보여줄 수 있는 데이터 전처리 방법의 조합을 제시하고자 한다.
NSMC의 경우, NLTK와 TensorFlow의 Keras, LSTM을 사용하였으며 Friends의 경우, NLTK와 TensorFlow의 Keras, CNN을 사용하여 가장 효과적인 데이터 전처리 방법의 조합을 제시했다.

### 연구환경 ###
연구 환경은 아래와 같다.
- Window 10 환경
- Python 3.7.7 버전(PyCharm Tool을 사용하여 개발)
- TensorFlow 2.3.1 버전
- KoNLPy는 0.5.2 버전 > Okt
- Keras 2.4.3 버전
  - 성능을 비교할 때에 precision과 recall, f-measure를 사용하려고 했으나 공식적으로 Keras 2.0 Metrics 중에서 precision, recall, f-measure가 제외되었다. 따라서 precision, recall, f-measure를 사용자정의함수를 이용하여 계산하였다.
  - accuracy 말고도 f1 score, precision, recall 값을 얻고 싶다면 KOR_nsmc.py 에서는 아래와 같이 되어 있는 코드를
  ``` python
  model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) # 01-1 모델 accuracy 계산 버전
  # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m]) # 01-2 모델 accuracy, f1 score, precision, recall 계산

  # 테스트 데이터에 대한 loss, accuracy, f1 score, precision, recall 계산 및 출력
  # loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
  # print('loss: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1score: {:.3f}'.format(loss, accuracy, precision, recall, f1_score))

  print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
  
  # Kaggle 테스트 데이터에 대한 결과 CSV 파일 생성
  f = open('kor_result.csv', 'w', newline='')
  wr = csv.writer(f)

  with open('ko_data.csv', 'r') as fd:
      reader = csv.reader(fd)
      for row in reader:
          wr.writerow([row[0], predict(row[1])])

  f.close()
  ```
  아래와 같이 수정해야한다.
  ``` python
  # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) # 01-1 모델 accuracy 계산 버전
  model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m]) # 01-2 모델 accuracy, f1 score, precision, recall 계산

  # 테스트 데이터에 대한 loss, accuracy, f1 score, precision, recall 계산 및 출력
  loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
  print('loss: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1score: {:.3f}'.format(loss, accuracy, precision, recall, f1_score))

  # print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
  
  # Kaggle 테스트 데이터에 대한 결과 CSV 파일 생성
  # f = open('kor_result.csv', 'w', newline='')
  # wr = csv.writer(f)
  # 
  # with open('ko_data.csv', 'r') as fd:
  #     reader = csv.reader(fd)
  #     for row in reader:
  #         wr.writerow([row[0], predict(row[1])])
  # 
  # f.close()
  ```
  
  - accuracy 말고도 f1 score, precision, recall 값을 얻고 싶다면 ENG_friends.py에서는 아래와 같이 되어 있는 코드를
  ``` python
  model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc']) # 02-1 모델 accuracy 계산 버전
  # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc',f1_m,precision_m, recall_m]) # 02-2 모델 accuracy, f1 score, precision, recall 계산 버전

  # 테스트 데이터에 대한 loss, accuracy, f1_score, precision, recall 계산 및 출력
  # loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
  # print('loss: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1score: {:.3f}'.format(loss, accuracy, precision, recall, f1_score))

  print("\n 테스트 정확도: %.4f" % (model.evaluate(x_test, y_test)[1]))

  # Kaggle 테스트 데이터에 대한 결과 CSV 파일 생성
  f = open('eng_result.csv', 'w', newline='')
  wr = csv.writer(f)

  with open('en_data.csv', 'r') as fd:
      reader = csv.reader(fd)
      for row in reader:
          wr.writerow([row[0], predict(row[4])])

  f.close()
  ```
  아래와 같이 수정해야한다.
  ``` python
  # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc']) # 02-1 모델 accuracy 계산 버전
  model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc',f1_m,precision_m, recall_m]) # 02-2 모델 accuracy, f1 score, precision, recall 계산 버전
  
  # 테스트 데이터에 대한 loss, accuracy, f1_score, precision, recall 계산 및 출력
  loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
  print('loss: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1score: {:.3f}'.format(loss, accuracy, precision, recall, f1_score))
  
  # print("\n 테스트 정확도: %.4f" % (model.evaluate(x_test, y_test)[1]))

  # Kaggle 테스트 데이터에 대한 결과 CSV 파일 생성
  # f = open('eng_result.csv', 'w', newline='')
  # wr = csv.writer(f)
  # 
  # with open('en_data.csv', 'r') as fd:
  #     reader = csv.reader(fd)
  #     for row in reader:
  #         wr.writerow([row[0], predict(row[4])])
  # 
  # f.close()
  ```

## 실행방법 ##
#### 1. KOR_nsmc.py 실행방법 ####
- KOR_nsmc.py 파일과 같은 레벨에 NSMC 폴더와 Kaggle 테스트 데이터인 ko_data.csv 파일을 위치시킨다.
  - NSMC 폴더는 README.md와 같은 레벨에 NSMC 폴더와 ko_data.csv를 올려두었다.
- KOR_nsmc.py를 실행하기 전에 import되어 있는 내역을 확인해본다.
  - 필요할 경우, 관련 패키지를 설치한다.
- KOR_nsmc.py를 실행한다.
- KOR_nsmc.py와 같은 레벨에 kor_nsmc_model.h5 모델과 Kaggle 테스트 데이터에 대한 kor_result.csv 결과 파일이 생성된 것을 확인할 수 있다.

#### 2. ENG_friends.py 실행방법 ####
- ENG_friends.py 파일과 같은 레벨에 Friends 폴더와 Kaggle 테스트 데이터인 eng_data.csv 파일을 위치시킨다.
  - NSMC 폴더는 README.md와 같은 레벨에 Friends 폴더와 eng_data.csv를 올려두었다.
- ENG_friends.py를 실행하기 전에 import되어 있는 내역을 확인해본다. 
  - 필요할 경우, 관련 패키지를 설치한다.
- ENG_friends.py를 실행한다.
- ENG_friends.py와 같은 레벨에 eng_friends_model.h5 모델과 Kaggle 테스트 데이터에 대한 eng_result.csv 결과 파일이 생성된 것을 확인할 수 있다.
