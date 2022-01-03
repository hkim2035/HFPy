import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

"""
DAT 파일 구조 예 (HF.DAT)
0.028	98	2.404                               ← dense, tdepth, tburden
0.05	0.035	0.02404	0.	0.	0.	0.	0.      ← x0(8)
24                                              ← n-measurments
1	133.3	20.5	136.5 	4.519 	274	0       ← data fields(n-rows)
1	133.3	20.5	139.0 	5.920 	84	1
1	133.3	20.5	149.5 	6.946 	8	28
1	133.3	20.5	149.5 	6.946 	251	2
...
data fields = [findex,bbering,binclin,mdepth,psm,fstrike,fdip]로 구성
"""

Tk().withdraw() 
filename = askopenfilename() 
data_file = open(filename, mode='r')

rdata = data_file.readline().strip().replace("\n","").split('\t')

density, tdepth, tburden = [float(xx) for xx in data_file.readline().replace("\n","").split('\t')[0:3]]
x0 = [float(xx) for xx in data_file.readline().replace("\n", "").split('\t')]
norows = int(data_file.readline().replace("\n", "").split('\t')[0])
#dummy = data_file.readline()

temp = list()
for ii in range(0, norows, 1):
    temp.append(data_file.readline().replace("\n", "").split('\t'))

raw = pd.DataFrame(temp, columns=['findex', 'bbering', 'binclin', 'mdepth', 'psm', 'fstrike', 'fdip', 'dummy'])
breakpoint()