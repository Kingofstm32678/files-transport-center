import torch
import torch.nn as nn
import openpyxl
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt

fsub = 400  # 单通道采样率
fs = 8 * fsub

M = 10 ** 6
G = 10 ** 9
m = 10 ** -3
n = 10 ** -9
p = 10 ** -12
f = 10 ** -15

# 打开 Excel 文件
workbook = openpyxl.load_workbook("917.1875.xlsx")
sheet = workbook.active

# 创建一个空数组
array = []
# 创建8个空数组用于存储校准完成的数组
ch1 = []
ch2 = []
ch3 = []
ch4 = []
ch5 = []
ch6 = []
ch7 = []
ch8 = []
# 创建8个空数组用于存储校准完成的差值
ch1_cali = []
ch2_cali = []
ch3_cali = []
ch4_cali = []
ch5_cali = []
ch6_cali = []
ch7_cali = []
ch8_cali = []
# 遍历每一行
for row in range(2, 53002):
    # 创建一个空列表，用于存储当前行的八个元素
    row_data = []
    # 遍历每一列
    for col in range(10, 18):
        # 获取当前单元格的值，并添加到当前行列表中
        cell_value = sheet.cell(row=row, column=col).value
        row_data.append(cell_value)
    # 将当前行列表添加到数组中
    array.append(row_data)
# 将数组中的元素转换为float类型
for i in range(len(array)):
    for j in range(len(array[i])):
        array[i][j] = float(array[i][j])
for row in array:
    # 将数组的第一个元素添加到ch1数组
    ch1.append(row[0])
    # 将数组的第二个元素添加到ch2数组
    ch2.append(row[1])
    # 将数组的第三个元素添加到ch3数组
    ch3.append(row[2])
    # 将数组的第四个元素添加到ch4数组
    ch4.append(row[3])
    # 将数组的第五个元素添加到ch5数组
    ch5.append(row[4])
    # 将数组的第六个元素添加到ch6数组
    ch6.append(row[5])
    # 将数组的第七个元素添加到ch7数组
    ch7.append(row[6])
    # 将数组的第八个元素添加到ch8数组
    ch8.append(row[7])
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)  # 1为in_channels 10为out_channels
        self.conv2 = torch.nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)  # 1为in_channels 10为out_channels
        self.conv4 = torch.nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(1)
        self.fc = torch.nn.Linear(8, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn1(x)  # 输出=（输入-均值）/（标准差+eps),衔接上下层结构，装逼用
        x = self.conv3(x)  # con3,4用于提高复杂度，更能提取特征
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn2(x)
        # x = x.view(batchsize,-1)
        x1 = x  # 残差，可去除
        x = self.fc(x)
        x = x1 + x  # 残差，可去除
        # x = self.relu(x)
        return x
net = Net()
net.load_state_dict(torch.load("CNN_917.1875.pt"))
array=torch.tensor(array)
array = nn.functional.normalize(array)
array = array.reshape(-1, 1, 53000, 8)
outputs=net(array)
outputs=outputs.tolist()
for row in outputs: #每个chx_cali[]数组只需要256个元素
    for i in range(256):
        # 将数组的第一个元素添加到ch1_cali数组
        ch1_cali.append(row[0][i][0])
        # 将数组的第二个元素添加到ch2_cali数组
        ch2_cali.append(row[0][i][1])
        # 将数组的第三个元素添加到ch3_cali数组
        ch3_cali.append(row[0][i][2])
        # 将数组的第四个元素添加到ch4_cali数组
        ch4_cali.append(row[0][i][3])
        # 将数组的第五个元素添加到ch5_cali数组
        ch5_cali.append(row[0][i][4])
        # 将数组的第六个元素添加到ch6_cali数组
        ch6_cali.append(row[0][i][5])
        # 将数组的第七个元素添加到ch7_cali数组
        ch7_cali.append(row[0][i][6])
        # 将数组的第八个元素添加到ch8_cali数组
        ch8_cali.append(row[0][i][7])

ch1_cali = [ch1[i] + ch1_cali[i] for i in range(len(ch1_cali))]
ch2_cali = [ch2[i] + ch2_cali[i] for i in range(len(ch2_cali))]
ch3_cali = [ch3[i] + ch3_cali[i] for i in range(len(ch3_cali))]
ch4_cali = [ch4[i] + ch4_cali[i] for i in range(len(ch4_cali))]
ch5_cali = [ch5[i] + ch5_cali[i] for i in range(len(ch5_cali))]
ch6_cali = [ch6[i] + ch6_cali[i] for i in range(len(ch6_cali))]
ch7_cali = [ch7[i] + ch7_cali[i] for i in range(len(ch7_cali))]
ch8_cali = [ch8[i] + ch8_cali[i] for i in range(len(ch8_cali))]
num = 256
Dout = np.zeros(8 * num) #2048
for i in range(num):
    Dout[8 * i] = ch1_cali[i]
    Dout[8 * i + 1] = ch2_cali[i]
    Dout[8 * i + 2] = ch3_cali[i]
    Dout[8 * i + 3] = ch4_cali[i]
    Dout[8 * i + 4] = ch5_cali[i]
    Dout[8 * i + 5] = ch6_cali[i]
    Dout[8 * i + 6] = ch7_cali[i]
    Dout[8 * i + 7] = ch8_cali[i]

numpt = len(Dout)  # 8*256
spect = fft(Dout)
dB = 20 * np.log10(np.abs(spect))
maxdB = np.max(dB[1:numpt // 2])
dB[dB < n] = -n

plt.figure(1)
plt.subplot(1, 1, 1)
Q = plt.plot(np.arange(0, numpt // 2) * fs / numpt, dB[1:numpt // 2 + 1] - maxdB, 'k')
plt.grid(True)
plt.xlim(0, 1600)
plt.ylim(-110, 0)
plt.xlabel('Frequency (MHz)', fontsize=15)
plt.ylabel('Magnitude (dB)', fontsize=15)

fi = np.argmax(dB[1:numpt // 2]) + 1
span = 0
spectP = np.abs(spect) ** 2
#Ps = np.sum(spectP[fi - span:fi + span])
Ps = spectP[fi]
Psum = np.sum(spectP[2:numpt // 2])

Dout_SF = np.abs(dB[1:numpt // 2] - maxdB)
Dout_SFDR_1 = np.min(Dout_SF[0:(fi - 2)])
Dout_SFDR_2 = np.min(Dout_SF[fi:len(Dout_SF)])

fin = (fi - 1) / numpt * fs + 1.5625
SFDR = np.min([Dout_SFDR_1, Dout_SFDR_2])
SNDR = 10 * np.log10(Ps / (Psum - Ps))
ENOB = (SNDR - 1.76) / 6.02

plt.text(0, -20, f'SFDR={SFDR:.2f}dB,\n'
                      f'SNDR={SNDR:.2f}dB,\n'
                      f'ENOB={ENOB:.2f}bit@{fin:.4f}MHz',
          fontsize=14, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
plt.show()

