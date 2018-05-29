import pandas as pd
import matplotlib.pyplot as plt
from mpldatacursor import datacursor



csv_GAN=pd.read_csv('Loss_G_GAN.csv')
csv_L1=pd.read_csv('Loss_G_L1.csv')

fig, ax = plt.subplots()
lines = ax.plot(csv_GAN.columns.astype(int),csv_GAN.get_values().reshape(len(csv_GAN.columns),), '*-')
datacursor(lines)
plt.grid()
plt.title('GAN Loss')
plt.show()

# plt.figure()
# plt.stem(csv_L1.columns.astype(int),csv_L1.get_values().reshape(len(csv_GAN.columns),), '*-')
# plt.grid()
# plt.title('L1 Loss')
#
# plt.show()