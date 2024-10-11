import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file_path = './test_results.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')


# filtering before pivot
df_new = df[df['Gmax']=='default']

# --------------------------------------------------------------------------------------------------
"""accuracy vs. time plot """

# pivot
pivot_index = list(df.columns[0:14])
pivot_index.remove('drift_model')
pivot_index.remove('n_rep')

df_pivot = df_new.pivot_table(index=pivot_index, columns='drift_model', values='accuracy', aggfunc='mean')
df_pivot.reset_index(inplace=True) # index to columns
df_pivot.to_csv('./pivot.csv', index=False)  # save the file


# filtering before plotting
df_new = df_pivot[df_pivot['pgm_noise_scale']==0]
df_new = df_new[df_new['model']=='MLP']


# accuracy vs. time plot
x = df_new['inf_time']

plt.grid(True)
plt.plot(x,df_new['log'], label = 'logarithm', linewidth=2, color='k')
plt.scatter(x, df_new['log'], marker='.', s=100, color='k')

plt.plot(x,df_new['linear'], label = 'linear', linewidth=2, color='dodgerblue')
plt.scatter(x, df_new['linear'], marker='.',  s=100, color='dodgerblue')

plt.plot(x,df_new[0.1], label = '0.1', linewidth=2, color='darkorange')
plt.scatter(x, df_new[0.1], marker='.',  s=100, color='darkorange')

plt.plot(x,df_new[0.05], label = '0.05', linewidth=2, color='g')
plt.scatter(x, df_new[0.05], marker='.',  s=100, color='g')

plt.plot(x,df_new[0.01], label = '0.01', linewidth=2, color='m')
plt.scatter(x, df_new[0.01], marker='.',  s=100, color='m')

plt.plot(x,df_new['GST225'], label = '* GST225', linewidth=3, color='r')
plt.scatter(x, df_new['GST225'], marker='.',  s=200, color='r')

# legend, title, axis
plt.legend(fontsize=12, loc='lower left', bbox_to_anchor=(0.0, 0.0))
plt.xlabel('Time [sec]', fontsize=18)
plt.ylabel('Inference accuracy [%]', fontsize=18) #, fontweight='demi'
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.ylim(0,100)
# plt.xlim(1,100000000)
plt.xscale('log')

# save the figure
plt.tight_layout()
plt.savefig('./figure.png')


# --------------------------------------------------------------------------------------------------
"""accuracy vs. onoff_ratio plot """

# pre-processing
df_resnet = df_new[df_new['model'] == 'Resnet18']
df_mlp = df_new[df_new['model'] == 'MLP']

df_resnet = df_resnet.sort_values(by='G_ratio', ascending=True)  
df_resnet = df_resnet.drop(df_resnet.index[3], axis=0)  # remove the specific row

df_mlp = df_mlp.sort_values(by='G_ratio', ascending=True)
df_mlp = df_mlp.drop(df_mlp.index[3], axis=0)


# accuracy vs. onoff_ratio plot
x = df_mlp['G_ratio']
y = df_mlp['accuracy']
y_r = df_resnet['accuracy']

plt.grid(True)

plt.plot(x,y, label = 'MLP-MNIST', linewidth=2, color='b')
plt.scatter(x, y, marker='.', s=200, color='b')

plt.plot(x,y_r, label = 'Resnet18-CIFAR10', linewidth=2, color='g')
plt.scatter(x, y_r, marker='.', s=200, color='g')


plt.legend(fontsize=12, loc='lower right', bbox_to_anchor=(1.0, 0.0))
plt.xlabel('On/Off ratio', fontsize=18)
plt.ylabel('Inference accuracy [%]', fontsize=18) #, fontweight='demi'
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.ylim(0,100)
# plt.xlim(1,100000000)
plt.xscale('log')
plt.show()