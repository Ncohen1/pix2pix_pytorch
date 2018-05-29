import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from mpldatacursor import datacursor



csv_GAN=pd.read_csv('Loss_G_GAN_stack2.csv')
csv_L1=pd.read_csv('Loss_G_L1_stack2.csv')

# index will not be zero - for row2col
# csv_GAN.index= csv_GAN.index+1
# csv_L1.index= csv_L1.index+1

# from row to column
# pd.DataFrame(csv_GAN[csv_GAN.columns[1]].values,columns=[['loss']]).to_csv('../data_excel/Loss_G_GAN_stack2.csv', header=True, index=True)
# pd.DataFrame(csv_L1[csv_L1.columns[1]],columns=[['loss']]).to_csv('../data_excel/Loss_G_L1_stack2.csv', header=True, index=True)




#
# fig, ax = plt.subplots()
# lines = ax.plot(csv_GAN.columns.astype(int),csv_GAN.get_values().reshape(len(csv_GAN.columns),), '*-')
# # datacursor(lines)
# plt.grid()
# plt.title('GAN Loss')



#
# fig, ax = plt.subplots()
# lines = ax.plot(csv_L1.columns.astype(int),csv_L1.get_values().reshape(len(csv_L1.columns),), '*-')
# datacursor(lines)
# plt.grid()
# plt.title('L1 Loss')
# plt.show()


plt.figure()
plt.plot(csv_L1[csv_L1.columns[0]], csv_L1[csv_L1.columns[1]], '*')
plt.grid()
plt.title('L1 Loss')
plt.figure()
plt.plot(csv_GAN[csv_GAN.columns[0]], csv_GAN[csv_GAN.columns[1]], '*')
plt.grid()
plt.title('G_GAN Loss')
plt.figure()
plt.plot(csv_GAN[csv_GAN.columns[0]], csv_GAN[csv_GAN.columns[1]]+csv_L1[csv_L1.columns[1]],'*')
plt.grid()
plt.title('L1 + G_GAN Loss')
plt.show()




#pd.DataFrame([list(err1.keys()),list([err1.values()])],columns=[1,2]).to_csv('Loss_G_GAN_.csv',index=False)

#pd.DataFrame([list(err1.keys()),np.transpose(csv_L1.get_values()).shape])],columns=[1,2]).to_csv('Loss_G_GAN_.csv',index=False)

# csv_L1.stack().to_csv('Loss_L1.csv', header=True)