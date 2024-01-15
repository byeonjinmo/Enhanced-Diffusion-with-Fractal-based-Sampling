import numpy as np
from keras.datasets import cifar10

def save_cifar10_as_npz():
    # CIFAR-10 데이터셋 로드
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # 데이터를 하나의 딕셔너리로 결합
    data = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test
    }

    # npz 파일로 저장
    np.savez('cifar10_dataset.npz', **data)

    print("CIFAR-10 데이터셋이 'cifar10_dataset.npz' 파일로 저장되었습니다.")

# CIFAR-10 데이터셋을 npz 파일로 저장하는 함수 실행
save_cifar10_as_npz()
