import numpy as np

import matplotlib.pyplot as plt



# .npz 파일 로드

data = np.load('C:/Users/user/Desktop/chat_jin/diffusion/improved-diffusion-main/결과/openai-2024-01-12-16-13-30-811957(가장 기본 500스텝_샘플넘버1000_배치32/samples_1000x32x32x3.npz')

# 예를 들어, 이미지 데이터가 'arr_0' 키로 저장되었다고 가정

images = data['arr_0']  # 'arr_0'를 실제 사용한 키 이름으로 교체하세요



# 이미지 데이터의 차원 확인

print("Shape of images:", images.shape)



# 이미지 순회 및 표시

for i, image in enumerate(images):

    plt.imshow(image)

    plt.title(f"Image {i+1}")

    plt.show()


    # ㅂㅣ교


import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터 전처리
# 예: 정규화, 리사이징 등
x_test = x_test.astype('float32') / 255.0

# 여기에 Inception 모델을 사용하여 특성 추출 및 제공된 평가 코드와 통합
