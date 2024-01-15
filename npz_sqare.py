# import numpy as np

#

# # npz 파일 로드

# npz_file = np.load('cifar10_test_dataset.npz')

#

# # 파일 내의 배열 목록 출력

# print(list(npz_file.keys()))





import numpy as np



# 기존 npz 파일 로드

with np.load('C:/Users/user/Desktop/chat_jin/diffusion/improved-diffusion-main/cifar10_test_dataset.npz') as data:

    x_test = data['x_test']

    y_test = data['y_test']



# 새로운 npz 파일로 저장

np.savez('modified_cifar10_test_dataset.npz', arr_0=x_test, y_test=y_test)