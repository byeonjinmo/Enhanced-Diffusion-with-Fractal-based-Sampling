import numpy as np
import matplotlib.pyplot as plt

# 프랙탈 브라운 운동을 생성하는 함수
def fractal_brownian_motion(H, size, length=1, octaves=1):
    # H: 허스트 지수(Hurst exponent), (0,1) 사이의 값
    # size: 생성할 fBm의 크기
    # length: 공간/시간 축의 총 길이
    # octaves: 노이즈를 중첩하는 횟수

    # 누적을 저장할 배열 초기화
    fBm = np.zeros(size)
    # 각 옥타브마다 누적
    for i in range(octaves):
        # 1D Perlin 노이즈 생성
        scale = length / (2 ** i)
        phase = np.random.rand()
        for t in range(size):
            fBm[t] += perlin_noise_1d(t / scale + phase) * (scale ** H)
    return fBm

# 1D Perlin 노이즈 생성 함수
def perlin_noise_1d(x):
    # 간단한 선형 보간을 사용한 1D Perlin 노이즈
    x_floor = np.floor(x).astype(int)
    x_fract = x - x_floor
    # 보간
    return lerp(np.random.rand(), np.random.rand(), x_fract)

# 선형 보간 함수
def lerp(a, b, x):
    return a + x * (b - a)

# 파라미터 설정
H = 0.7  # 허스트 지수
size = 1024  # 샘플링할 데이터의 크기

# 프랙탈 브라운 운동 생성
fBm = fractal_brownian_motion(H, size)

# 생성된 fBm 시각화
plt.plot(fBm)
plt.title('1D Fractal Brownian Motion')
plt.show()
