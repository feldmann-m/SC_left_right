#%% relevant paths
figs='/storage/homefs/mf23m219/figs/SC_env/'
scr_data='/storage/workspaces/giub_meteo_impacts/ci01/supercell_climate/cookies/'
code='/home/mfeldmann/code/'
import sys, os
sys.path.append(code)
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
#from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion
import copy
from glob import glob
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.colors as colors
from scipy.ndimage import binary_dilation
from skimage.morphology import disk


import argparse as ap

parser = ap.ArgumentParser(description="Extraction per climate period")
parser.add_argument('-t', '--time', type=int, default=0, help="Hours before storm (default: 0)")
parser.add_argument('-c', '--climate', type=int, default=0, help="Current (0) or future (1) climate (default: 0)")
#parser.add_argument('-t', '--type', type=int, default=1, help="An integer argument (default: 1)")
args = parser.parse_args()

t = args.time
c = args.climate
reg = 'EUR'
if c==0: cl = 'present'
if c==1: cl = 'future'

c_c = scr_data + cl+'_t-'+str(t)+'/subdomains/'+reg+'/*.nc'
c_c_files = sorted(glob(c_c))
rows_r=[]
rows_l=[]
for file in c_c_files:
    print(file)
    data = xr.open_dataset(file)

    mask = data.TOT_PREC<(5/12)
    buffered_mask = xr.DataArray(
        binary_erosion(mask, structure=np.expand_dims(disk(5),axis=0)),  # 3-pixel radius
        dims=mask.dims,
        coords=mask.coords
    )

    data2 = data.where(buffered_mask)
    # if data.signature==1:
    #     a1=data2.sel(x=data.x>0,y=data.y>0).mean(dim=("x","y"))
    # if data.signature==-1:
    #     a1=data2.sel(x=data.x>0,y=data.y<0).mean(dim=("x","y"))

    a1 = data2.mean(dim=('x','y'))
    a2 = data2.max(dim=('x','y'))
    a3 = data2.min(dim=('x','y'))

    b1 = data.mean(dim=('x','y'))
    b2 = data.max(dim=('x','y'))
    b3 = data.min(dim=('x','y'))


    pval = data.pressure.values

    lclval = a1.LCL_ML
    for p in [925,850,700,600,500]:
        zval = a1.sel(pressure=p).FI/9.81
        if zval>=lclval: lcl_p = p; break
        else: lcl_p = -1

    row = {
        'filename': file,
        'time': data.real_time.values[0],
        'lat': data.meso_lat.values[0],
        'lon': data.meso_lon.values[0],
        'LCL_ML': lclval.values[0],
        'p_LCL': lcl_p,
        'CAPE_MU': a1.CAPE_MU.values[0],
        'CAPE_ML': a1.CAPE_ML.values[0],
        'WMS': a1.WMAXSHEAR_MU.values[0],
        'CIN_MU': a1.CIN_MU.values[0],
        'CIN_ML': a1.CIN_ML.values[0],
        'CAPE_MU_max': a2.CAPE_MU.values[0],
        'CAPE_ML_max': a2.CAPE_ML.values[0],
        'CIN_MU_min': a3.CIN_MU.values[0],
        'CIN_ML_min': a3.CIN_ML.values[0],
        'WMS_max': a2.WMAXSHEAR_MU.values[0],
        'LCL_ML': a1.LCL_ML.values[0],
        'LFC_ML': a1.LFC_ML.values[0],
        'MSLP': a1.PMSL.values[0],
        'SP': a1.PS.values[0],
        'U_925': a1.U.sel(pressure=925).values[0],
        'U_850': a1.U.sel(pressure=850).values[0],
        'U_700': a1.U.sel(pressure=700).values[0],
        'U_600': a1.U.sel(pressure=600).values[0],
        'U_500': a1.U.sel(pressure=500).values[0],
        'U_400': a1.U.sel(pressure=400).values[0],
        'U_300': a1.U.sel(pressure=300).values[0],
        'U_200': a1.U.sel(pressure=200).values[0],
        'V_925': a1.V.sel(pressure=925).values[0],
        'V_850': a1.V.sel(pressure=850).values[0],
        'V_700': a1.V.sel(pressure=700).values[0],
        'V_600': a1.V.sel(pressure=600).values[0],
        'V_500': a1.V.sel(pressure=500).values[0],
        'V_400': a1.V.sel(pressure=400).values[0],
        'V_300': a1.V.sel(pressure=300).values[0],
        'V_200': a1.V.sel(pressure=200).values[0],
        'T_925': a1.T.sel(pressure=925).values[0],
        'T_850': a1.T.sel(pressure=850).values[0],
        'T_700': a1.T.sel(pressure=700).values[0],
        'T_600': a1.T.sel(pressure=600).values[0],
        'T_500': a1.T.sel(pressure=500).values[0],
        'T_400': a1.T.sel(pressure=400).values[0],
        'T_300': a1.T.sel(pressure=300).values[0],
        'T_200': a1.T.sel(pressure=200).values[0],
        'Q_925': a1.QV.sel(pressure=925).values[0],
        'Q_850': a1.QV.sel(pressure=850).values[0],
        'Q_700': a1.QV.sel(pressure=700).values[0],
        'Q_600': a1.QV.sel(pressure=600).values[0],
        'Q_500': a1.QV.sel(pressure=500).values[0],
        'Q_400': a1.QV.sel(pressure=400).values[0],
        'Q_300': a1.QV.sel(pressure=300).values[0],
        'Q_200': a1.QV.sel(pressure=200).values[0],
        'RH_925': a1.RELHUM.sel(pressure=925).values[0],
        'RH_850': a1.RELHUM.sel(pressure=850).values[0],
        'RH_700': a1.RELHUM.sel(pressure=700).values[0],
        'RH_600': a1.RELHUM.sel(pressure=600).values[0],
        'RH_500': a1.RELHUM.sel(pressure=500).values[0],
        'RH_400': a1.RELHUM.sel(pressure=400).values[0],
        'RH_300': a1.RELHUM.sel(pressure=300).values[0],
        'RH_200': a1.RELHUM.sel(pressure=200).values[0],
        'Z_925': a1.FI.sel(pressure=925).values[0],
        'Z_850': a1.FI.sel(pressure=850).values[0],
        'Z_700': a1.FI.sel(pressure=700).values[0],
        'Z_600': a1.FI.sel(pressure=600).values[0],
        'Z_500': a1.FI.sel(pressure=500).values[0],
        'Z_400': a1.FI.sel(pressure=400).values[0],
        'Z_300': a1.FI.sel(pressure=300).values[0],
        'Z_200': a1.FI.sel(pressure=200).values[0],
        'max_V500': b2.VORT.sel(pressure=500).values[0],
        'min_V500': b3.VORT.sel(pressure=500).values[0],
        'delta_Z500': b2.FI.sel(pressure=500).values[0]-b3.FI.sel(pressure=500).values[0],
        'max_W500': b2.W.sel(pressure=500).values[0],
        'q90_W500': data.sel(pressure=500).W.where(data.sel(pressure=500).W > 5).quantile(0.9, dim=('x','y'), skipna=True).values[0],
        'q10_W500': data.sel(pressure=500).W.where(data.sel(pressure=500).W > 5).quantile(0.1, dim=('x','y'), skipna=True).values[0],
        'mean_W500': ((data.sel(pressure=500).W>5)*(data.sel(pressure=500).W)).mean(dim=('x','y'),skipna=True).values[0],
        'area_W500': (data.sel(pressure=500).W>5).mean(dim=('x','y')).values[0],
        'max_LPI': b2.LPI_MAX.values[0],
        'mean_LPI': b1.LPI_MAX.values[0],
        'area_LPI': (data.LPI_MAX>1).mean(dim=('x','y')).values[0],
        'max_prec': b2.TOT_PREC.values[0],
        'mean_prec': b1.TOT_PREC.values[0],
        'area_prec': (data.TOT_PREC>0.1).mean(dim=('x','y')).values[0],
        'max_gust': b2.VMAX_10M.values[0],
        'mean_gust': b1.VMAX_10M.values[0],
        'area_gust': (data.VMAX_10M>12).mean(dim=('x','y')).values[0],
        'max_hail': b2.DHAIL_MX.values[0],
        'mean_hail': b1.DHAIL_MX.values[0],
        'area_hail': (data.LPI_MAX>0.1).mean(dim=('x','y')).values[0],
        'q90_CAPE_MU': data2.sel(pressure=500).W.where(data.sel(pressure=500).W > 5).quantile(0.9, dim=('x','y'), skipna=True).values[0],
        'q10_CAPE_MU': data2.sel(pressure=500).W.where(data.sel(pressure=500).W > 5).quantile(0.1, dim=('x','y'), skipna=True).values[0],

    }

    if data.signature==1: rows_r.append(row)
    if data.signature==-1: rows_l.append(row)

df_r = pd.DataFrame(rows_r)
df_l = pd.DataFrame(rows_l)

df_r.to_csv('/storage/homefs/mf23m219/SC_env/zdisk_'+reg+'_'+cl+'_t-'+str(t)+'_RM_td.csv')
df_l.to_csv('/storage/homefs/mf23m219/SC_env/zdisk_'+reg+'_'+cl+'_t-'+str(t)+'_LM_td.csv')
    




