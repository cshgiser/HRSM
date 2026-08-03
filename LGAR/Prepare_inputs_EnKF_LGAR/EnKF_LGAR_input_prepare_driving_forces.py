import  netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

alpha = 1.3
rho_w = 1000

def e_s(T):
    return(611*np.exp((17.27*T)/(273.3+T)))

def delta(T):
    return((4098*e_s(T))/((273.3+T)**2))

def l_v(T):
    return(2500 - 2.36*T)

def gamma(T):
    return(66.8)
    #return(1.005*101325*pressure/(0.622*l_v(T)))
    #note that gamma is often given as the constant 66.8 pascals/decree C. the other version, commented out, takes temperature and pressure (input pressure in atm, and temp as deg C) as inputs.

def E_r(R_n, T):
    return(R_n/(l_v(T)*1000*rho_w))

def E_PT(R_n, T):
    return(E_r(R_n, T)*alpha*delta(T)/(delta(T)+gamma(T)))

def get_nan_row_indices(arr, col_idx):
    """
    Given a 2D NumPy array and a column index,
    return the row indices where the selected column has NaN values.

    Parameters:
    - arr: 2D numpy array
    - col_idx: integer, index of the column to check

    Returns:
    - 1D numpy array of row indices where values are NaN
    """
    col_data = arr[:, col_idx]
    nan_rows = np.where(np.isnan(col_data))[0]
    return nan_rows


import pandas as pd
def fill_pet_gaps(arr):
    """
    Fill NaN gaps in PET data for each site (column) with linear interpolation,
    skipping sites that have the entire time series missing.

    Parameters:
        arr (2D np.ndarray): PET array (time, sites)

    Returns:
        np.ndarray: Filled PET array
    """
    arr_filled = arr.copy()
    n_sites = arr.shape[1]

    for site in range(n_sites):
        col = arr[:, site]

        # Skip sites that are completely NaN
        if np.all(np.isnan(col)):
            print(f"Skipping site {site} (fully missing)")
            continue

        # Interpolate missing values in both directions
        s = pd.Series(col)
        arr_filled[:, site] = s.interpolate(method='linear', limit_direction='both').values

    return arr_filled



if __name__=='__main__':
    ds_ear5 = nc.Dataset('/media/shuohao/Expansion/PSA_and_ISMN_CONUS/ERA5_ISMN_PSA_hourly.nc')
    # I stored the ERA5 dataset in format of nc, but it is not necessary.
    # All we want is to get the 2-d (time series X sites) array of 2m air temperature (t2m), short-wave solar radiation (ssr)
    # and total Precipitation (tp). t2m and ssr are used to calculate the PET.
    ear5_id_arr = np.array(ds_ear5['ID'][:])
    ear5_t2m_arr = np.array(ds_ear5['t2m'][:]).T
    ear5_ssr_arr = np.array(ds_ear5['ssr'][:]).T
    ear5_tp_arr = np.array(ds_ear5['tp'][:]).T
    ds_ear5.close()

    ear5_tp_arr = ear5_tp_arr * 1000  # hourly precipitation, m to mm
    ear5_tp_arr[ear5_tp_arr < 1e-2] = 0
    print(ear5_tp_arr.shape)

    nan_columns = np.any(np.isnan(ear5_tp_arr), axis=0)
    nan_col_indices = np.where(nan_columns)[0]
    print("ear5_tp_arr Columns containing NaN:", nan_col_indices)
    print("ear5_tp_arr, number of nan values:", np.count_nonzero(np.isnan(ear5_tp_arr)))

    ear5_ssr_arr = ear5_ssr_arr/3600  # J to W/m2
    ear5_t2m_arr = ear5_t2m_arr - 273.15 # K to deg C

    ear5_PET_arr = 3.6e6*E_PT(ear5_ssr_arr, ear5_t2m_arr)
    ear5_PET_arr[ear5_PET_arr < 1e-6] = 0
    print(ear5_PET_arr.shape)

    ear5_PET_arr = fill_pet_gaps(ear5_PET_arr)

    nan_columns = np.any(np.isnan(ear5_PET_arr), axis=0)
    nan_col_indices = np.where(nan_columns)[0]
    print("ear5_PET_arr Columns containing NaN:", nan_col_indices)
    print("ear5_PET_arr, number of nan values:", np.count_nonzero(np.isnan(ear5_PET_arr)))

    outPath = '../'
    np.save(f'{outPath}/PET_arr.npy', ear5_PET_arr)
    np.save(f'{outPath}/Prec_arr.npy', ear5_tp_arr)

    print()