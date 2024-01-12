import numpy as np
import matplotlib.pyplot as plt

# .npz 파일 로드
data = np.load('path_to_your_file.npz')
# 예를 들어, 이미지 배열이 'arr' 키로 저장되었다고 가정
images = data['arr']

# 이미지 순회 및 표시
for i, image in enumerate(images):
    plt.imshow(image)
    plt.title(f"Image {i+1}")
    plt.show()
