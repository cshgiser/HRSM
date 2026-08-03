import numpy as np
import pandas as pd


def extract_soil_properties(pm_sand_arr, pm_silt_arr, pm_clay_arr):
    # Hydrus properties
    theta_r_dict = {
        "sand": 0.045,
        "loamy_sand": 0.057,
        "sandy_loam": 0.065,
        "loam": 0.078,
        "silt": 0.034,
        "silt_loam": 0.067,
        "sandy_clay_loam": 0.100,
        "clay_loam": 0.095,
        "silty_clay_loam": 0.089,
        "sandy_clay": 0.100,
        "silty_clay": 0.070,
        "clay": 0.068
    }

    theta_e_dict = {
        "sand": 0.43,
        "loamy_sand": 0.41,
        "sandy_loam": 0.41,
        "loam": 0.43,
        "silt": 0.46,
        "silt_loam": 0.45,
        "sandy_clay_loam": 0.39,
        "clay_loam": 0.41,
        "silty_clay_loam": 0.43,
        "sandy_clay": 0.38,
        "silty_clay": 0.36,
        "clay": 0.38
    }

    alpha_dict = {  # cm^-1
        "sand": 0.145,
        "loamy_sand": 0.124,
        "sandy_loam": 0.075,
        "loam": 0.036,
        "silt": 0.016,
        "silt_loam": 0.020,
        "sandy_clay_loam": 0.059,
        "clay_loam": 0.019,
        "silty_clay_loam": 0.010,
        "sandy_clay": 0.027,
        "silty_clay": 0.005,
        "clay": 0.008
    }

    n_dict = {
        "sand": 2.68,
        "loamy_sand": 2.28,
        "sandy_loam": 1.89,
        "loam": 1.56,
        "silt": 1.37,
        "silt_loam": 1.41,
        "sandy_clay_loam": 1.48,
        "clay_loam": 1.31,
        "silty_clay_loam": 1.23,
        "sandy_clay": 1.23,
        "silty_clay": 1.09,
        "clay": 1.09
    }

    Ks_dict = {  # (cm/h)
        "sand": 29.7,
        "loamy_sand": 14.5917,
        "sandy_loam": 4.42083,
        "loam": 1.04,
        "silt": 0.25,
        "silt_loam": 0.45,
        "sandy_clay_loam": 1.31,
        "clay_loam": 0.26,
        "silty_clay_loam": 0.07,
        "sandy_clay": 0.12,
        "silty_clay": 0.02,
        "clay": 0.2
    }

    mask_sand = np.logical_and(pm_sand_arr > 85, (pm_silt_arr + 1.5 * pm_clay_arr) < 15)
    mask_loamy_sand = np.logical_and.reduce([
        pm_sand_arr >= 70,
        pm_sand_arr < 91,
        (pm_silt_arr + 1.5 * pm_clay_arr) >= 15,
        (pm_silt_arr + 2 * pm_clay_arr) < 30
    ])
    mask_sandy_loam = np.logical_or(
        np.logical_and.reduce([
            pm_clay_arr >= 7,
            pm_clay_arr < 20,
            pm_sand_arr > 52,
            (pm_silt_arr + 2 * pm_clay_arr) >= 30
        ]),
        np.logical_and.reduce([
            pm_clay_arr < 7,
            pm_silt_arr < 50,
            pm_sand_arr > 43
        ])
    )
    mask_loam = np.logical_and.reduce([
        pm_clay_arr >= 7,
        pm_clay_arr < 27,
        pm_silt_arr >= 28,
        pm_silt_arr < 50,
        pm_sand_arr <= 52
    ])
    mask_silt_loam = np.logical_or(
        np.logical_and(pm_silt_arr >= 50, np.logical_and(pm_clay_arr >= 12, pm_clay_arr < 27)),
        np.logical_and(pm_clay_arr < 12, np.logical_and(pm_silt_arr >= 50, pm_silt_arr < 80))
    )
    mask_silt = np.logical_and(pm_silt_arr >= 80, pm_clay_arr < 12)
    mask_sandy_clay_loam = np.logical_and.reduce([
        pm_clay_arr >= 20,
        pm_clay_arr < 35,
        pm_silt_arr < 28,
        pm_sand_arr > 45
    ])
    mask_clay_loam = np.logical_and.reduce([
        pm_clay_arr >= 27,
        pm_clay_arr < 40,
        pm_sand_arr > 20,
        pm_sand_arr < 46
    ])
    mask_silty_clay_loam = np.logical_and.reduce([
        pm_clay_arr >= 27,
        pm_clay_arr < 40,
        pm_sand_arr <= 20
    ])
    mask_sandy_clay = np.logical_and(pm_clay_arr >= 35, pm_sand_arr >= 45)
    mask_silty_clay = np.logical_and(pm_clay_arr >= 40, pm_silt_arr >= 40)
    mask_clay = np.logical_and.reduce([
        pm_clay_arr >= 40,
        pm_sand_arr <= 45,
        pm_silt_arr < 40
    ])

    soilMap  = {
        "sand": 1,
        "loamy_sand": 2,
        "sandy_loam": 3,
        "loam": 4,
        "silt": 5,
        "silt_loam": 6,
        "sandy_clay_loam": 7,
        "clay_loam": 8,
        "silty_clay_loam": 9,
        "sandy_clay": 10,
        "silty_clay": 11,
        "clay": 12
    }

    theta_r_arr = np.full(pm_sand_arr.shape, np.nan)
    theta_e_arr = np.full(pm_sand_arr.shape, np.nan)
    alpha_arr = np.full(pm_sand_arr.shape, np.nan)
    n_arr = np.full(pm_sand_arr.shape, np.nan)
    Ks_arr = np.full(pm_sand_arr.shape, np.nan)
    soilid_arr = np.full(pm_sand_arr.shape, np.nan)

    soilid_arr[mask_sand] = soilMap['sand']
    soilid_arr[mask_loamy_sand] = soilMap['loamy_sand']
    soilid_arr[mask_sandy_loam] = soilMap['sandy_loam']
    soilid_arr[mask_loam] = soilMap['loam']
    soilid_arr[mask_silt] = soilMap['silt']
    soilid_arr[mask_sandy_clay_loam] = soilMap['sandy_clay_loam']
    soilid_arr[mask_silt_loam] = soilMap['silt_loam']
    soilid_arr[mask_clay_loam] = soilMap['clay_loam']
    soilid_arr[mask_silty_clay_loam] = soilMap['silty_clay_loam']
    soilid_arr[mask_sandy_clay] = soilMap['sandy_clay']
    soilid_arr[mask_silty_clay] = soilMap['silty_clay']
    soilid_arr[mask_clay] = soilMap['clay']

    theta_r_arr[mask_sand] = theta_r_dict['sand']
    theta_r_arr[mask_loamy_sand] = theta_r_dict['loamy_sand']
    theta_r_arr[mask_sandy_loam] = theta_r_dict['sandy_loam']
    theta_r_arr[mask_loam] = theta_r_dict['loam']
    theta_r_arr[mask_silt] = theta_r_dict['silt']
    theta_r_arr[mask_sandy_clay_loam] = theta_r_dict['sandy_clay_loam']
    theta_r_arr[mask_silt_loam] = theta_r_dict['silt_loam']
    theta_r_arr[mask_clay_loam] = theta_r_dict['clay_loam']
    theta_r_arr[mask_silty_clay_loam] = theta_r_dict['silty_clay_loam']
    theta_r_arr[mask_sandy_clay] = theta_r_dict['sandy_clay']
    theta_r_arr[mask_silty_clay] = theta_r_dict['silty_clay']
    theta_r_arr[mask_clay] = theta_r_dict['clay']

    theta_e_arr[mask_sand] = theta_e_dict['sand']
    theta_e_arr[mask_loamy_sand] = theta_e_dict['loamy_sand']
    theta_e_arr[mask_sandy_loam] = theta_e_dict['sandy_loam']
    theta_e_arr[mask_loam] = theta_e_dict['loam']
    theta_e_arr[mask_silt] = theta_e_dict['silt']
    theta_e_arr[mask_sandy_clay_loam] = theta_e_dict['sandy_clay_loam']
    theta_e_arr[mask_silt_loam] = theta_e_dict['silt_loam']
    theta_e_arr[mask_clay_loam] = theta_e_dict['clay_loam']
    theta_e_arr[mask_silty_clay_loam] = theta_e_dict['silty_clay_loam']
    theta_e_arr[mask_sandy_clay] = theta_e_dict['sandy_clay']
    theta_e_arr[mask_silty_clay] = theta_e_dict['silty_clay']
    theta_e_arr[mask_clay] = theta_e_dict['clay']

    alpha_arr[mask_sand] = alpha_dict['sand']
    alpha_arr[mask_loamy_sand] = alpha_dict['loamy_sand']
    alpha_arr[mask_sandy_loam] = alpha_dict['sandy_loam']
    alpha_arr[mask_loam] = alpha_dict['loam']
    alpha_arr[mask_silt] = alpha_dict['silt']
    alpha_arr[mask_sandy_clay_loam] = alpha_dict['sandy_clay_loam']
    alpha_arr[mask_silt_loam] = alpha_dict['silt_loam']
    alpha_arr[mask_clay_loam] = alpha_dict['clay_loam']
    alpha_arr[mask_silty_clay_loam] = alpha_dict['silty_clay_loam']
    alpha_arr[mask_sandy_clay] = alpha_dict['sandy_clay']
    alpha_arr[mask_silty_clay] = alpha_dict['silty_clay']
    alpha_arr[mask_clay] = alpha_dict['clay']

    n_arr[mask_sand] = n_dict['sand']
    n_arr[mask_loamy_sand] = n_dict['loamy_sand']
    n_arr[mask_sandy_loam] = n_dict['sandy_loam']
    n_arr[mask_loam] = n_dict['loam']
    n_arr[mask_silt] = n_dict['silt']
    n_arr[mask_sandy_clay_loam] = n_dict['sandy_clay_loam']
    n_arr[mask_silt_loam] = n_dict['silt_loam']
    n_arr[mask_clay_loam] = n_dict['clay_loam']
    n_arr[mask_silty_clay_loam] = n_dict['silty_clay_loam']
    n_arr[mask_sandy_clay] = n_dict['sandy_clay']
    n_arr[mask_silty_clay] = n_dict['silty_clay']
    n_arr[mask_clay] = n_dict['clay']

    Ks_arr[mask_sand] = Ks_dict['sand']
    Ks_arr[mask_loamy_sand] = Ks_dict['loamy_sand']
    Ks_arr[mask_sandy_loam] = Ks_dict['sandy_loam']
    Ks_arr[mask_loam] = Ks_dict['loam']
    Ks_arr[mask_silt] = Ks_dict['silt']
    Ks_arr[mask_sandy_clay_loam] = Ks_dict['sandy_clay_loam']
    Ks_arr[mask_silt_loam] = Ks_dict['silt_loam']
    Ks_arr[mask_clay_loam] = Ks_dict['clay_loam']
    Ks_arr[mask_silty_clay_loam] = Ks_dict['silty_clay_loam']
    Ks_arr[mask_sandy_clay] = Ks_dict['sandy_clay']
    Ks_arr[mask_silty_clay] = Ks_dict['silty_clay']
    Ks_arr[mask_clay] = Ks_dict['clay']


    return soilid_arr, theta_r_arr, theta_e_arr, alpha_arr, n_arr, Ks_arr

if __name__=='__main__':
    df_constant = pd.read_csv('../your_sites_info.csv')
    df_refer_id_arr = df_constant['ID'].to_numpy()
    df_refer_sand_0_5_arr = df_constant['sand_0_5'].to_numpy()
    df_refer_sand_5_15_arr = df_constant['sand_5_15'].to_numpy()
    df_refer_sand_15_30_arr = df_constant['sand_15_30'].to_numpy()
    df_refer_sand_30_60_arr = df_constant['sand_30_60'].to_numpy()
    df_refer_sand_60_100_arr = df_constant['sand_60_100'].to_numpy()
    df_refer_clay_0_5_arr = df_constant['clay_0_5'].to_numpy()
    df_refer_clay_5_15_arr = df_constant['clay_5_15'].to_numpy()
    df_refer_clay_15_30_arr = df_constant['clay_15_30'].to_numpy()
    df_refer_clay_30_60_arr = df_constant['clay_30_60'].to_numpy()
    df_refer_clay_60_100_arr = df_constant['clay_60_100'].to_numpy()

    df_refer_silt_0_5_arr = 100 - df_refer_sand_0_5_arr - df_refer_clay_0_5_arr
    df_refer_silt_5_15_arr = 100 - df_refer_sand_5_15_arr - df_refer_clay_5_15_arr
    df_refer_silt_15_30_arr = 100 - df_refer_sand_15_30_arr - df_refer_clay_15_30_arr
    df_refer_silt_30_60_arr = 100 - df_refer_sand_30_60_arr - df_refer_clay_30_60_arr
    df_refer_silt_60_100_arr = 100 - df_refer_sand_60_100_arr - df_refer_clay_60_100_arr

    soilid_0_5_arr, theta_r_0_5_arr, theta_e_0_5_arr, alpha_0_5_arr, n_0_5_arr, Ks_0_5_arr = extract_soil_properties(
        df_refer_sand_0_5_arr, df_refer_silt_0_5_arr, df_refer_clay_0_5_arr)
    soilid_5_15_arr, theta_r_5_15_arr, theta_e_5_15_arr, alpha_5_15_arr, n_5_15_arr, Ks_5_15_arr = extract_soil_properties(
        df_refer_sand_5_15_arr, df_refer_silt_5_15_arr, df_refer_clay_5_15_arr)
    soilid_15_30_arr, theta_r_15_30_arr, theta_e_15_30_arr, alpha_15_30_arr, n_15_30_arr, Ks_15_30_arr = extract_soil_properties(
        df_refer_sand_15_30_arr, df_refer_silt_15_30_arr, df_refer_clay_15_30_arr)
    soilid_30_60_arr, theta_r_30_60_arr, theta_e_30_60_arr, alpha_30_60_arr, n_30_60_arr, Ks_30_60_arr = extract_soil_properties(
        df_refer_sand_30_60_arr, df_refer_silt_30_60_arr, df_refer_clay_30_60_arr)
    soilid_60_100_arr, theta_r_60_100_arr, theta_e_60_100_arr, alpha_60_100_arr, n_60_100_arr, Ks_60_100_arr = extract_soil_properties(
        df_refer_sand_60_100_arr, df_refer_silt_60_100_arr, df_refer_clay_60_100_arr)

    soilid_arr = np.array((soilid_0_5_arr, soilid_5_15_arr, soilid_15_30_arr,soilid_30_60_arr,soilid_60_100_arr))
    theta_r_arr = np.array((theta_r_0_5_arr, theta_r_5_15_arr, theta_r_15_30_arr, theta_r_30_60_arr, theta_r_60_100_arr))
    theta_e_arr = np.array((theta_e_0_5_arr, theta_e_5_15_arr, theta_e_15_30_arr, theta_e_30_60_arr, theta_e_60_100_arr))
    alpha_arr = np.array((alpha_0_5_arr, alpha_5_15_arr, alpha_15_30_arr, alpha_30_60_arr, alpha_60_100_arr))
    n_arr = np.array((n_0_5_arr, n_5_15_arr, n_15_30_arr, n_30_60_arr, n_60_100_arr))
    Ks_arr = np.array((Ks_0_5_arr, Ks_5_15_arr, Ks_15_30_arr, Ks_30_60_arr, Ks_60_100_arr))
    print(theta_r_arr.shape)
    print(f'{np.count_nonzero(np.isnan(soilid_arr))},{np.count_nonzero(np.isnan(theta_r_arr))}, {np.count_nonzero(np.isnan(theta_e_arr))}, '
          f'{np.count_nonzero(np.isnan(alpha_arr))}, {np.count_nonzero(np.isnan(n_arr))}, '
          f'{np.count_nonzero(np.isnan(Ks_arr))}')

    print(soilid_arr.shape)
    outPath = '../'
    np.save(f'{outPath}/soilid_arr.npy', soilid_arr)  # only need this
    # np.save(f'{outPath}/reorganize_EnKF/theta_r_arr.npy', theta_r_arr)
    # np.save(f'{outPath}/reorganize_EnKF/theta_e_arr.npy', theta_e_arr)
    # np.save(f'{outPath}/reorganize_EnKF/alpha_arr.npy', alpha_arr)
    # np.save(f'{outPath}/reorganize_EnKF/n_arr.npy', n_arr)
    # np.save(f'{outPath}/reorganize_EnKF/Ks_arr.npy', Ks_arr)










