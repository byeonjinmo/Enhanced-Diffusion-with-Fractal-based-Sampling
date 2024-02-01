import numpy as np
import matplotlib.pyplot as plt

# .npz 파일에서 이미지 데이터 로드
file_path = 'C:/Users/user/AppData/Local/Temp/openai-2024-01-31-14-46-04-815111/samples_1000x64x64x3.npz'
data = np.load(file_path)
images = data['arr_0']  # 'arr_0'는 npz 파일 내에 저장된 이미지 배열의 키를 가정함

# 이미지 시각화
def plot_images(images, num_images=25):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
    plt.show()

# 첫 25개 이미지를 표시
plot_images(images, num_images=25)
