import numpy as np
data = np.load('samples.npz')
images = data['arr_0']  # 이미지 배열
import matplotlib.pyplot as plt

plt.imshow(images[0])  # 첫 번째 이미지를 표시
plt.show()
