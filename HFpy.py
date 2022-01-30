import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import math
import plotly.graph_objects as go

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

# x0: ndarray PN0, PE0, PV0, PNE0, PEV0, PVN0, alphaNN, alphaEE
# args: tuple 밀도, 기준심도, 상부하중,균열종류, 시추공방위각, 경사각, 심도, 균열폐쇄압력, 종균열방향, 0 or 경사균열 방위각, 경사각
# ((밀도 N개), (기준심도 N개)), ...)


def fs(nn: int, deg: list):
    nnn = np.repeat(nn, len(deg))
    return np.array(list(map(lambda ldeg, lnn: math.sin(math.radians(ldeg))**lnn, deg, nnn)))


def fc(nn: int, deg: list):
    nnn = np.repeat(nn, len(deg))
    return np.array(list(map(lambda ldeg, lnn: math.cos(math.radians(ldeg))**lnn, deg, nnn)))


def calMF(x0: list, data):

    global psc_final

    psc = np.zeros(len(data[0]))

    [PN0, PE0, PV0, PNE0, PEV0, PVN0, alphaNN, alphaEE] = [
        np.repeat(item, len(data[0])) for item in x0]
    den, tdepth, over = [np.array(data[ii]) for ii in range(0, 3, 1)]
    fractype, alpha, beta, dep, psm, psi, pi = [
        np.array(data[ii]) for ii in range(3, 10, 1)]

    depth = dep - tdepth

    alphaVV = den
    alphaNE = 0.5*(alphaNN-alphaEE)*2.*PNE0/(PN0-PE0)
    alphaEV = 0.5*(alphaEE-alphaVV)*2.*PEV0/(PE0-PV0)
    alphaVN = 0.5*(alphaVV-alphaNN)*2.*PVN0/(PV0-PN0)

    PN = PN0 + depth*alphaNN
    PE = PE0 + depth*alphaEE
    PV = PV0 + depth*alphaVV
    PNE = PNE0 + depth*alphaNE
    PEV = PEV0 + depth*alphaEV
    PVN = PVN0 + depth*alphaVN

    ver = [fractype == 0]
    inc = [fractype > 0]

    psc[inc] = PN[inc]*fc(2., pi)[inc]*fc(2., psi)[inc] + PE[inc]*fc(2., pi)[inc]*fs(2., psi)[inc] + PV[inc]*fs(2., pi)[inc] + PNE[inc]*fc(
        2., pi)[inc]*fs(1., 2.*psi)[inc] + PEV[inc]*fs(1., 2.*pi)[inc]*fs(1., psi)[inc] + PVN[inc]*fs(1., 2.*pi)[inc]*fc(1., psi)[inc]

    psc_final = psc

    temp = (psm-psc)**2.
    temp = (temp.sum()/(len(temp)-1))**.5

    return temp


if __name__ == "__main__":
    Tk().withdraw()
    filename = askopenfilename()
    data_file = open(filename, mode='r')

    density, tdepth, tburden = [
        float(xx) for xx in data_file.readline().replace("\n", "").split('\t')[0:3]]
    x0 = [float(xx)
          for xx in data_file.readline().replace("\n", "").split('\t')]
    norows = int(data_file.readline().replace("\n", "").split('\t')[0])
    #dummy = data_file.readline()

    temp = list()
    for ii in range(0, norows, 1):
        temp.append(data_file.readline().replace("\n", "").split('\t'))

    m = pd.DataFrame(temp, columns=[
        'findex', 'bbering', 'binclin', 'mdepth', 'psm', 'fstrike', 'fdip', 'dummy'])

    cden = tuple(np.repeat(density, norows))
    cz0 = tuple(np.repeat(tdepth, norows))
    cburden = tuple(np.repeat(tburden, norows))
    fi = tuple(np.array(m.findex, dtype=int))
    bb = tuple(np.array(m.bbering, dtype=float))
    incl = tuple(np.array(m.binclin, dtype=float))
    mdep = tuple(np.array(m.mdepth, dtype=float))
    psm = tuple(np.array(m.psm, dtype=float))
    fstr = tuple(np.array(m.fstrike, dtype=float))
    fdip = tuple(np.array(m.fdip, dtype=float))

    data = [cden, cz0, cburden, fi, bb, incl, mdep, psm, fstr, fdip]

    bnds = ((0, None), (0, None), (0, None), (None, None),
            (None, None), (None, None), (0, None), (0, None))
    result = minimize(calMF, x0, data, bounds=bnds, tol=1.e-9, method='BFGS')


fig = go.Figure(go.Scatter(
    x=psm-psc_final,
    y=mdep,
    marker=dict(color="crimson", size=12),
    mode="markers",
    name="Psm",
))

fig.update_layout(title="Ps,measured - Ps, calculated",
                  xaxis_title="Psm-Psc (MPa)",
                  yaxis_title="Depth (m)",
                  autosize=False,
                  width=600,
                  height=800,
                  margin=dict(
                      l=50,
                      r=50,
                      b=50,
                      t=50,
                      pad=4
                  ),
                  paper_bgcolor="White",
                  )

fig.update_yaxes(
    range=[math.ceil((max(mdep)*1.1)/10.)*10., 0], zeroline=True)
fig.update_xaxes(range=[-max(abs(psm-psc_final)) *
                 1.1, max(abs(psm-psc_final))*1.1])


fig.show()
