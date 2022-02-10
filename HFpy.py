import math
import os
import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import least_squares

import pytransform3d as pytr

#import oct2py

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


def headerprint(string):
    """ Prints a centered string to divide output sections. """
    mywidth = 64
    mychar = "="
    numspaces = mywidth - len(string)
    before = int(math.ceil(float(mywidth-len(string))/2))
    after = int(math.floor(float(mywidth-len(string))/2))
    print("\n"+before*mychar+string+after*mychar+"\n")


def valprint(string, value):
    """ Ensure uniform formatting of scalar value outputs. """
    print("{0:>30}: {1: .10e}".format(string, value))


def matprint(string, value):
    """ Ensure uniform formatting of matrix value outputs. """
    print("{0}:".format(string))
    print(value)


def fs(nn: int, deg: list):
    nnn = np.repeat(nn, len(deg))
    return np.array(list(map(lambda ldeg, lnn: math.sin(math.radians(ldeg))**lnn, deg, nnn)))


def fc(nn: int, deg: list):
    nnn = np.repeat(nn, len(deg))
    return np.array(list(map(lambda ldeg, lnn: math.cos(math.radians(ldeg))**lnn, deg, nnn)))


def calMF(x0: list, data: list):

    global psc_final, fPN, fPE, fPV, fPNE, fPEV, fPVN

    psc = np.zeros(len(data[0]))

    [PN0, PE0, PV0, PNE0, PEV0, PVN0, alphaNN, alphaEE] = [
        np.repeat(item, len(data[0])) for item in x0]
    den, tdepth, over = data[0:3]
    fractype, alpha, beta, dep, psm, psi, pi = data[3:10]

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

    ver = tuple([fractype == 0])
    inc = tuple([fractype > 0])

    psc[inc] = PN[inc]*fc(2., pi)[inc]*fc(2., psi)[inc] + PE[inc]*fc(2., pi)[inc]*fs(2., psi)[inc] + PV[inc]*fs(2., pi)[inc] + PNE[inc]*fc(
        2., pi)[inc]*fs(1., 2.*psi)[inc] + PEV[inc]*fs(1., 2.*pi)[inc]*fs(1., psi)[inc] + PVN[inc]*fs(1., 2.*pi)[inc]*fc(1., psi)[inc]

    psc_final = psc
    fPN, fPE, fPV, fPNE, fPEV, fPVN = [PN, PE, PV, PNE, PEV, PVN]

    errsum = (psm-psc)**2.
    errsum = (errsum.sum()/(len(errsum)-1))**.5

    return errsum


if __name__ == "__main__":

    if len(sys.argv) == 1:
        Tk().withdraw()
        filename = askopenfilename()
    else:
        filename = sys.argv[1]

    data_file = open(filename, mode='r')

    density, tdepth, tburden = [
        float(xx) for xx in data_file.readline().replace("\n", "").split('\t')[0:3]]
    x0 = [float(xx)
          for xx in data_file.readline().replace("\n", "").split('\t')]
    norows = int(data_file.readline().replace("\n", "").split('\t')[0])
    # dummy = data_file.readline()

    temp = list()
    for ii in range(0, norows, 1):
        temp.append(data_file.readline().replace("\n", "").split('\t'))

    m = pd.DataFrame(temp, columns=[
        'findex', 'bbering', 'binclin', 'mdepth', 'psm', 'fstrike', 'fdip', 'dummy'])

    cden = np.repeat(density, norows)
    cz0 = np.repeat(tdepth, norows)
    cburden = np.repeat(tburden, norows)

    fi = np.array(m.findex, dtype=int)
    bb = np.array(m.bbering, dtype=float)
    incl = np.array(m.binclin, dtype=float)
    mdep = np.array(m.mdepth, dtype=float)
    psm = np.array(m.psm, dtype=float)
    fstr = np.array(m.fstrike, dtype=float)
    fdip = np.array(m.fdip, dtype=float)

    data = [cden, cz0, cburden, fi, bb, incl, mdep, psm, fstr, fdip]

    result = minimize(calMF, x0, data, tol=1.e-6, method='BFGS')

    bnds = ((0, None), (0, None), (0, None), (None, None),
            (None, None), (None, None), (0, None), (0, None))
    #minimizer_kwargs = {"method": "L-BFGS-B", "jac": True, "args": data}
    # compare1 = basinhopping(calMF, x0, minimizer_kwargs=minimizer_kwargs,
    #                        niter=2000)
    compare = least_squares(
        calMF, x0, args=[data], jac='cs', gtol=1e-12, max_nfev=100000)

df = pd.DataFrame([fi, mdep, cz0, mdep-cz0, psc_final, psm,
                  psm-psc_final, fPV-(cburden+cden*(mdep-cz0)), fPN, fPE, fPV, fPNE, fPEV, fPVN])
df = df.T
df.columns = ["Fracture_type", "mdepth", "tdepth", "depth", "Psc", "Psm",
              "tolPs", "tolPv", "PN", "PE", "PV", "PNE", "PEV", "PVN"]

# ---------
gf1 = pd.concat([df.tolPs, df.mdepth], axis=1)
gf1.rename(columns={'tolPs': 'X'}, inplace=True)
gf1['Name'] = 'Psm-Psc (MPa)'

gf2 = pd.concat([df.tolPv, df.mdepth], axis=1)
gf2.rename(columns={'tolPv': 'X'}, inplace=True)
gf2['Name'] = 'PN-(tburden+depth*den)'

gf = pd.concat([gf1, gf2])

fig1 = px.scatter(gf, x="X", y="mdepth", color="Name", labels=dict(
    X="PN (MPa)", mdepth="Depth (m)"),
    width=800, height=900)

fig1.update_traces(marker=dict(size=12,
                               line=dict(width=2,
                                         color='DarkSlateGrey')),
                   selector=dict(mode='markers'))

fig1.update_yaxes(autorange="reversed")

# -----------


fig1.update_xaxes(range=[-max(abs(df.tolPs))*1.1, max(abs(df.tolPs))*1.1])


# ---------
gf1 = pd.concat([df.PN, df.mdepth], axis=1)
gf1.rename(columns={'PN': 'X'}, inplace=True)
gf1['Name'] = 'PN'

gf2 = pd.concat([df.PE, df.mdepth], axis=1)
gf2.rename(columns={'PE': 'X'}, inplace=True)
gf2['Name'] = 'PE'

gf3 = pd.concat([df.PV, df.mdepth], axis=1)
gf3.rename(columns={'PV': 'X'}, inplace=True)
gf3['Name'] = 'PV'

gfA = pd.concat([gf1, gf2, gf3])

fig3 = px.scatter(gfA, x="X", y="mdepth", color="Name", trendline="ols", labels=dict(
    tolPv="PN (MPa)", mdepth="Depth (m)"),
    width=800, height=900)

fig3.update_traces(marker=dict(size=12,
                               line=dict(width=2,
                                         color='DarkSlateGrey')),
                   selector=dict(mode='markers'))

fig3.update_yaxes(autorange="reversed")

PNEV_results = px.get_trendline_results(fig3).px_fit_results.iloc[0:3]
# -----------


# ---------
gf4 = pd.concat([df.PNE, df.mdepth], axis=1)
gf4.rename(columns={'PNE': 'X'}, inplace=True)
gf4['Name'] = 'PNE'

gf5 = pd.concat([df.PEV, df.mdepth], axis=1)
gf5.rename(columns={'PEV': 'X'}, inplace=True)
gf5['Name'] = 'PEV'

gf6 = pd.concat([df.PVN, df.mdepth], axis=1)
gf6.rename(columns={'PVN': 'X'}, inplace=True)
gf6['Name'] = 'PVN'

gfB = pd.concat([gf4, gf5, gf6])

fig4 = px.scatter(gfB, x="X", y="mdepth", color="Name", trendline="ols", labels=dict(
    tolPv="PN (MPa)", mdepth="Depth (m)"),
    width=800, height=900)

fig4.update_traces(marker=dict(size=12,
                               line=dict(width=2,
                                         color='DarkSlateGrey')),
                   selector=dict(mode='markers'))

fig4.update_yaxes(autorange="reversed")

PNEV_results = px.get_trendline_results(fig4).px_fit_results.iloc[0:3]
# -----------

gfAll = pd.concat([gfA, gfB])

fig5 = px.scatter(gfAll, x="X", y="mdepth", color="Name", trendline="ols", labels=dict(
    tolPv="PN (MPa)", mdepth="Depth (m)"),
    width=800, height=900)

fig5.update_traces(marker=dict(size=12,
                               line=dict(width=2,
                                         color='DarkSlateGrey')),
                   selector=dict(mode='markers'))

fig5.update_yaxes(autorange="reversed")

PNEV_results = px.get_trendline_results(fig5).px_fit_results.iloc[0:6]

for idx, [PN, PE, PV, PNE, PEV, PVN] in df[['PN', 'PE', 'PV', 'PNE', 'PEV', 'PVN']].iterrows():
    # [PN, PE, PV, PNE, PEV, PVN] = [6.384014053,	5.060241522,
    #                               3.467942307, 0.662215382, -0.15480775, -0.066373708]
    [PN, PE, PV, PNE, PEV, PVN] = [2., 0., 0., 0., 0., 1.]
    sigma = np.asarray([[PN, PNE, PVN],
                        [PNE, PE, PEV],
                        [PVN, PEV, PV]])

    #e_val, e_vec = np.linalg.eig(sigma)

    sigma_iso = 1.0/3.0*np.trace(sigma)*np.eye(3)
    sigma_dev = sigma - sigma_iso

    # compute principal stresses
    eigvals = list(np.linalg.eigvalsh(sigma))
    eigvals.sort()
    eigvals.reverse()

    # compute max shear stress
    maxshear = (max(eigvals)-min(eigvals))/2.0

    # compute the stress invariants
    I1 = np.trace(sigma)
    J2 = 1.0/2.0*np.trace(np.dot(sigma_dev, sigma_dev))
    J3 = 1.0/3.0*np.trace(np.dot(sigma_dev, np.dot(sigma_dev, sigma_dev)))

    # compute other common stress measures
    mean_stress = 1.0/3.0*I1
    eqv_stress = math.sqrt(3.0*J2)

    # compute lode coordinates
    lode_r = math.sqrt(2.0*J2)
    lode_z = I1/math.sqrt(3.0)

    dum = 3.0*math.sqrt(6.0)*np.linalg.det(sigma_dev/lode_r)
    lode_theta = 1.0/3.0*math.asin(dum)

    # compute the stress triaxiality
    triaxiality = mean_stress/eqv_stress

    # Print out what we've found
    headerprint(" Stress State Analysis ")
    matprint("Input Stress", sigma)
    headerprint(" Component Matricies ")
    matprint("Isotropic Stress", sigma_iso)
    matprint("Deviatoric Stress", sigma_dev)
    headerprint(" Scalar Values ")
    valprint("P1", eigvals[0])
    valprint("P2", eigvals[1])
    valprint("P3", eigvals[2])
    valprint("Max Shear", maxshear)
    valprint("Mean Stress", mean_stress)
    valprint("Equivalent Stress", eqv_stress)
    valprint("I1", I1)
    valprint("J2", J2)
    valprint("J3", J3)
    valprint("Lode z", lode_z)
    valprint("Lode r", lode_r)
    valprint("Lode theta (rad)", lode_theta)
    valprint("Lode theta (deg)", math.degrees(lode_theta))
    valprint("Triaxiality", triaxiality)
    headerprint(" End Output ")

    # breakpoint()


if not os.path.exists("images"):
    os.mkdir("images")

# filename
#pio.kaleido.scope.plotlyjs = "c:\\Python\\Python39\\Lib\\site-packages\\plotly\package_data\\plotly.min.js"
#fig1.to_image(format="png", engine="kaleido")
#pio.write_image(fig1, "images/fig1.png", format="png")
# fig1.to_image(format="jpg")
# fig3.write_image("images/fig3.jpeg")
# fig4.write_image("images/fig4.jpeg")
# fig5.write_image("images/fig5.jpeg")


fig1.show()

fig3.show()
fig4.show()
fig5.show()

breakpoint()
