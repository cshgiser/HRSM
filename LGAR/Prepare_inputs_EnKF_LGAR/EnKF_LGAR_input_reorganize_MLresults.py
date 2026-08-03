from scipy.stats import rankdata
import numpy as np
from datetime import datetime,timedelta
import pandas as pd
import matplotlib.pyplot as plt

def empirical_cdf(data):
    ranks = rankdata(data, method='average')  # Rank the data
    return ranks / len(data)  # Normalize by the number of points

def cdf_matching(source, target):
    sorted_target = np.sort(target)  # Target distribution sorted
    sorted_target_ranks = rankdata(sorted_target, method='average')
    sorted_target_quantiles = sorted_target_ranks / len(sorted_target)
    ranks = rankdata(source, method='average')  # Source data ranks
    percentiles = ranks / len(source)  # Percentile positions of source data
    matched_values = np.interp(percentiles, sorted_target_quantiles, sorted_target)  # Interpolation
    return matched_values


def correct_2_ML3(Sentinel_sl_arr, sentinel2_sl_arr, hls_sl_arr, Sentinel_rly_arr, sentinel2_rly_arr, hls_rly_arr):
    Sentinel_sl_arr_ = Sentinel_sl_arr.copy()
    sentinel2_sl_arr_ = sentinel2_sl_arr.copy()
    hls_sl_arr_ = hls_sl_arr.copy()
    Sentinel_rly_arr_ = Sentinel_rly_arr.copy()
    sentinel2_rly_arr_ = sentinel2_rly_arr.copy()
    hls_rly_arr_ = hls_rly_arr.copy()
    # Xid_all, Xyear_all, Xdoy_all, Xutc_all, y_hat, y_up_hat, y_median_hat, y_low_hat
    # surface layer
    ids = list(set(sentinel2_sl_arr_[:, 0]))
    for idd in ids:
        hls_mask = hls_sl_arr_[:, 0] == idd
        Sentinel_mask = Sentinel_sl_arr_[:, 0] == idd
        sentinel2_mask = sentinel2_sl_arr_[:, 0] == idd
        print(f'sly, {idd} numbers: {np.count_nonzero(Sentinel_mask)}, {np.count_nonzero(sentinel2_mask)}, {np.count_nonzero(hls_mask)}')

        Sentinel_sl_arr_[Sentinel_mask, 4] = cdf_matching(Sentinel_sl_arr_[Sentinel_mask, 4], sentinel2_sl_arr_[sentinel2_mask, 4])
        hls_sl_arr_[hls_mask, 4] = cdf_matching(hls_sl_arr_[hls_mask, 4], sentinel2_sl_arr_[sentinel2_mask, 4])


    # rootzone
    ids = list(set(sentinel2_rly_arr_[:, 0]))
    for idd in ids:
        hls_mask = hls_rly_arr_[:, 0] == idd
        Sentinel_mask = Sentinel_rly_arr_[:, 0] == idd
        sentinel2_mask = sentinel2_rly_arr_[:, 0] == idd
        print(f'rly, {idd} numbers: {np.count_nonzero(Sentinel_mask)}, {np.count_nonzero(sentinel2_mask)}, {np.count_nonzero(hls_mask)}')

        Sentinel_rly_arr_[Sentinel_mask, 4] = cdf_matching(Sentinel_rly_arr_[Sentinel_mask, 4],
                                                          sentinel2_rly_arr_[sentinel2_mask, 4])
        hls_rly_arr_[hls_mask, 4] = cdf_matching(hls_rly_arr_[hls_mask, 4], sentinel2_rly_arr_[sentinel2_mask, 4])

    return Sentinel_sl_arr_, sentinel2_sl_arr_, hls_sl_arr_, Sentinel_rly_arr_, sentinel2_rly_arr_, hls_rly_arr_

def calculate_sd_and_miu(ML1_sl_arr, ML2_sl_arr, ML3_sl_arr, ML1_rly_arr, ML2_rly_arr, ML3_rly_arr):
    # Xid_all, Xyear_all, Xdoy_all, Xutc_all, y_hat

    # soil moisture
    Sentinel_sl_arr = ML1_sl_arr.copy()
    temp_arr = np.zeros((Sentinel_sl_arr.shape[0], 6)) + 1
    temp_arr[:, 0:5] = Sentinel_sl_arr[:, 0:5]
    Sentinel_sl_arr = temp_arr.copy()

    sentinel2_sl_arr = ML2_sl_arr.copy()
    temp_arr = np.zeros((sentinel2_sl_arr.shape[0], 6)) + 2
    temp_arr[:, 0:5] = sentinel2_sl_arr[:, 0:5]
    sentinel2_sl_arr = temp_arr.copy()

    hls_sl_arr = ML3_sl_arr.copy()
    temp_arr = np.zeros((hls_sl_arr.shape[0], 6)) + 3
    temp_arr[:, 0:5] = hls_sl_arr[:, 0:5]
    hls_sl_arr = temp_arr.copy()

    ssm_sl_arr = np.vstack((hls_sl_arr, Sentinel_sl_arr, sentinel2_sl_arr))
    ssm_sl_arr_0 = ssm_sl_arr.copy()
    ssm_sl_arr_0[:, 3] += -1
    ssm_sl_arr_1 = ssm_sl_arr.copy()
    ssm_sl_arr_1[:, 3] += 1
    ssm_sl_arr = np.vstack((ssm_sl_arr_0, ssm_sl_arr, ssm_sl_arr_1)).copy()

    # soil moisture standard deviation
    # The ubRMSE is also referred to as the standard deviation of the error
    hls_sl_arr = ML3_sl_arr.copy()
    temp_arr = np.zeros((hls_sl_arr.shape[0], 6)) + 3
    temp_arr[:, 0:4] = hls_sl_arr[:, 0:4]
    temp_arr[:, 4] = (hls_sl_arr[:, 5] - hls_sl_arr[:, 7])/3.29
    temp_arr[temp_arr[:, 4] <= 0, 4] = 0.02  # a small value
    hls_sl_arr = temp_arr.copy()

    Sentinel_sl_arr = ML1_sl_arr.copy()
    temp_arr = np.zeros((Sentinel_sl_arr.shape[0], 6)) + 1
    temp_arr[:, 0:4] = Sentinel_sl_arr[:, 0:4]
    temp_arr[:, 4] = (Sentinel_sl_arr[:, 5] - Sentinel_sl_arr[:, 7]) / 3.29
    temp_arr[temp_arr[:, 4] <= 0, 4] = 0.02  # a small value
    Sentinel_sl_arr = temp_arr.copy()

    sentinel2_sl_arr = ML2_sl_arr.copy()
    temp_arr = np.zeros((sentinel2_sl_arr.shape[0], 6)) + 2
    temp_arr[:, 0:4] = sentinel2_sl_arr[:, 0:4]
    temp_arr[:, 4] = (sentinel2_sl_arr[:, 5] - sentinel2_sl_arr[:, 7]) / 3.29
    temp_arr[temp_arr[:, 4] <= 0, 4] = 0.02  # a small value
    sentinel2_sl_arr = temp_arr.copy()

    sd_sl_arr = np.vstack((hls_sl_arr, Sentinel_sl_arr, sentinel2_sl_arr))
    sd_sl_arr_0 = sd_sl_arr.copy()
    sd_sl_arr_0[:, 3] += -1
    sd_sl_arr_1 = sd_sl_arr.copy()
    sd_sl_arr_1[:, 3] += 1
    sd_sl_arr = np.vstack((sd_sl_arr_0, sd_sl_arr, sd_sl_arr_1)).copy()

    ids = list(set(ssm_sl_arr[:, 0]))
    temp_arr = np.zeros((len(ids), 2))
    for i in range(len(ids)):
        temp_arr[i, 0] = ids[i]
        mask = ssm_sl_arr[:, 0] == ids[i]
        temp_arr[i, 1] = np.mean(ssm_sl_arr[mask, 4])
    miu_sl_arr = temp_arr.copy()

    # rootzone
    hls_rly_arr = ML3_rly_arr.copy()
    temp_arr = np.zeros((hls_rly_arr.shape[0], 6)) + 3
    temp_arr[:, 0:5] = hls_rly_arr[:, 0:5]
    hls_rly_arr = temp_arr.copy()

    Sentinel_rly_arr = ML1_rly_arr.copy()
    temp_arr = np.zeros((Sentinel_rly_arr.shape[0], 6)) + 1
    temp_arr[:, 0:5] = Sentinel_rly_arr[:, 0:5]
    Sentinel_rly_arr = temp_arr.copy()

    sentinel2_rly_arr = ML2_rly_arr.copy()
    temp_arr = np.zeros((sentinel2_rly_arr.shape[0], 6)) + 2
    temp_arr[:, 0:5] = sentinel2_rly_arr[:, 0:5]
    sentinel2_rly_arr = temp_arr.copy()


    ssm_rly_arr = np.vstack((hls_rly_arr, Sentinel_rly_arr, sentinel2_rly_arr))
    ssm_rly_arr_0 = ssm_rly_arr.copy()
    ssm_rly_arr_0[:, 3] += -1
    ssm_rly_arr_1 = ssm_rly_arr.copy()
    ssm_rly_arr_1[:, 3] += 1
    ssm_rly_arr = np.vstack((ssm_rly_arr_0, ssm_rly_arr, ssm_rly_arr_1)).copy()

    # The ubRMSE is also referred to as the standard deviation of the error
    hls_rly_arr = ML3_rly_arr.copy()
    temp_arr = np.zeros((hls_rly_arr.shape[0], 6)) + 3
    temp_arr[:, 0:4] = hls_rly_arr[:, 0:4]
    temp_arr[:, 4] = (hls_rly_arr[:, 5] - hls_rly_arr[:, 7]) / 3.29
    temp_arr[temp_arr[:, 4] <= 0, 4] = 0.02  # a small value
    hls_rly_arr = temp_arr.copy()

    Sentinel_rly_arr = ML1_rly_arr.copy()
    temp_arr = np.zeros((Sentinel_rly_arr.shape[0], 6)) + 1
    temp_arr[:, 0:4] = Sentinel_rly_arr[:, 0:4]
    temp_arr[:, 4] = (Sentinel_rly_arr[:, 5] - Sentinel_rly_arr[:, 7]) / 3.29
    temp_arr[temp_arr[:, 4] <= 0, 4] = 0.02  # a small value
    Sentinel_rly_arr = temp_arr.copy()

    sentinel2_rly_arr = ML2_rly_arr.copy()
    temp_arr = np.zeros((sentinel2_rly_arr.shape[0], 6)) + 2
    temp_arr[:, 0:4] = sentinel2_rly_arr[:, 0:4]
    temp_arr[:, 4] = (sentinel2_rly_arr[:, 5] - sentinel2_rly_arr[:, 7]) / 3.29
    temp_arr[temp_arr[:, 4] <= 0, 4] = 0.02  # a small value
    sentinel2_rly_arr = temp_arr.copy()

    sd_rly_arr = np.vstack((hls_rly_arr, Sentinel_rly_arr, sentinel2_rly_arr))
    sd_rly_arr_0 = sd_rly_arr.copy()
    sd_rly_arr_0[:, 3] += -1
    sd_rly_arr_1 = sd_rly_arr.copy()
    sd_rly_arr_1[:, 3] += 1
    sd_rly_arr = np.vstack((sd_rly_arr_0, sd_rly_arr, sd_rly_arr_1)).copy()


    ids = list(set(ssm_rly_arr[:, 0]))
    temp_arr = np.zeros((len(ids), 2))
    for i in range(len(ids)):
        temp_arr[i, 0] = ids[i]
        mask = ssm_rly_arr[:, 0] == ids[i]
        temp_arr[i, 1] = np.mean(ssm_rly_arr[mask, 4])
    miu_rly_arr = temp_arr.copy()

    return ssm_sl_arr, ssm_rly_arr, sd_sl_arr, sd_rly_arr, miu_sl_arr, miu_rly_arr



def create_date_list(start_date_str, num, deltat, unit='d'):
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

    date_list = []
    for i in range(num):
        date_list.append(start_date)
        if unit == 'd':
            start_date += timedelta(days=deltat)
        elif unit == 'h':
            start_date += timedelta(hours=deltat)

    return date_list


def DoY2Date(year, doy, hh=10):
    base_date = datetime(year, 1, 1, hour=hh)
    target_date = base_date + timedelta(days=(doy - 1))
    return target_date


def reorgnize_2_2d(ssm_sl_arr, ssm_rly_arr, sd_sl_arr, sd_rly_arr, miu_sl_arr, miu_rly_arr, refer_id_arr):
    ids = refer_id_arr
    len_id = len(ids)

    # ----------------------------
    start_time = datetime(2016, 1, 1, 0)
    end_time = datetime(2024, 12, 31, 23)
    num_hours = int(((end_time - start_time).total_seconds() / 3600) + 1)
    len_seq = num_hours
    time_array = [start_time + timedelta(hours=i) for i in range(num_hours)]
    df_date = pd.DataFrame({'time': time_array})
    df_date.set_index('time', inplace=True)

    # surface layer - sm
    df_ssm_sly = pd.DataFrame(ssm_sl_arr, columns=['id', 'year', 'doy', 'hour', 'sm', 'sensor'])
    df_ssm_sly = df_ssm_sly[(df_ssm_sly['hour'] >= 0) & (df_ssm_sly['hour'] <= 23)]
    df_ssm_sly['time'] = df_ssm_sly.apply(lambda row: DoY2Date(int(row['year']), int(row['doy']), int(row['hour'])), axis=1)

    ssm_sl_arr3D = np.zeros((3, len_seq, len_id))
    ssm_sl_arr3D[:] = np.nan
    sors = [1, 2, 3]
    for i in range(len(sors)):
        for j in range(len(ids)):
            sor = sors[i]
            iid = ids[j]
            df_ssm_sly_sor_id = df_ssm_sly[(df_ssm_sly['sensor']==sor)&(df_ssm_sly['id']==iid)]
            df_ssm_sly_sor_id.set_index('time', inplace=True)
            df_new = df_date.join(df_ssm_sly_sor_id, how='left')
            df_new = df_new.loc[~df_new.index.duplicated(keep='first')]
            ssm_sl_arr3D[i, :, j] = df_new['sm'].to_numpy()[:]



    # rootzone - sm
    df_ssm_rly = pd.DataFrame(ssm_rly_arr, columns=['id', 'year', 'doy', 'hour', 'sm', 'sensor'])
    df_ssm_rly = df_ssm_rly[(df_ssm_rly['hour'] >= 0) & (df_ssm_rly['hour'] <= 23)]
    df_ssm_rly['time'] = df_ssm_rly.apply(lambda row: DoY2Date(int(row['year']), int(row['doy']), int(row['hour'])),
                                          axis=1)

    ssm_rly_arr3D = np.zeros((3, len_seq, len_id))
    ssm_rly_arr3D[:] = np.nan
    sors = [1, 2, 3]
    for i in range(len(sors)):
        for j in range(len(ids)):
            sor = sors[i]
            iid = ids[j]

            df_ssm_rly_sor_id = df_ssm_rly[(df_ssm_rly['sensor']==sor)&(df_ssm_rly['id']==iid)]
            df_ssm_rly_sor_id.set_index('time', inplace=True)
            df_new = df_date.join(df_ssm_rly_sor_id, how='left')
            df_new = df_new.loc[~df_new.index.duplicated(keep='first')]
            ssm_rly_arr3D[i, :, j] = df_new['sm'].to_numpy()[:]



    # surface layer - SD
    df_sd_sly = pd.DataFrame(sd_sl_arr, columns=['id', 'year', 'doy', 'hour', 'sm', 'sensor'])
    df_sd_sly = df_sd_sly[(df_sd_sly['hour'] >= 0) & (df_sd_sly['hour'] <= 23)]
    df_sd_sly['time'] = df_sd_sly.apply(lambda row: DoY2Date(int(row['year']), int(row['doy']), int(row['hour'])),
                                          axis=1)

    sd_sl_arr3D = np.zeros((3, len_seq, len_id))
    sd_sl_arr3D[:] = np.nan

    fsors = [1, 2, 3]
    for i in range(len(sors)):
        for j in range(len(ids)):
            sor = sors[i]
            iid = ids[j]
            df_sd_sly_sor_id = df_sd_sly[(df_sd_sly['sensor'] == sor) & (df_sd_sly['id'] == iid)]
            df_sd_sly_sor_id.set_index('time', inplace=True)
            df_new = df_date.join(df_sd_sly_sor_id, how='left')
            df_new = df_new.loc[~df_new.index.duplicated(keep='first')]
            sd_sl_arr3D[i, :, j] = df_new['sm'].to_numpy()[:]



    # rootzone - sd
    df_sd_rly = pd.DataFrame(sd_rly_arr, columns=['id', 'year', 'doy', 'hour', 'sm', 'sensor'])
    df_sd_rly = df_sd_rly[(df_sd_rly['hour'] >= 0) & (df_sd_rly['hour'] <= 23)]
    df_sd_rly['time'] = df_sd_rly.apply(lambda row: DoY2Date(int(row['year']), int(row['doy']), int(row['hour'])),
                                          axis=1)

    sd_rly_arr3D = np.zeros((3, len_seq, len_id))
    sd_rly_arr3D[:] = np.nan
    for i in range(len(sors)):
        for j in range(len(ids)):
            sor = sors[i]
            iid = ids[j]
            df_sd_rly_sor_id = df_sd_rly[(df_sd_rly['sensor'] == sor) & (df_sd_rly['id'] == iid)]
            df_sd_rly_sor_id.set_index('time', inplace=True)
            df_new = df_date.join(df_sd_rly_sor_id, how='left')
            df_new = df_new.loc[~df_new.index.duplicated(keep='first')]
            sd_rly_arr3D[i, :, j] = df_new['sm'].to_numpy()[:]

    # ======================================

    miu_sl_arr1D = np.zeros((len(ids)))
    miu_sl_arr1D[:] = np.nan
    for j in range(len(ids)):
        iid = ids[j]
        if len(miu_sl_arr[miu_sl_arr[:, 0]==iid, 1]) == 0:
            miu_sl_arr1D[j] = np.nan
            continue
        meanvalue = miu_sl_arr[miu_sl_arr[:, 0]==iid, 1][0]
        miu_sl_arr1D[j] = meanvalue

    miu_rly_arr1D = np.zeros((len(ids)))
    miu_rly_arr1D[:] = np.nan
    for j in range(len(ids)):
        iid = ids[j]
        if len(miu_rly_arr[miu_rly_arr[:, 0] == iid, 1]) == 0:
            miu_rly_arr1D[j] = np.nan
            continue
        meanvalue = miu_rly_arr[miu_rly_arr[:, 0] == iid, 1][0]
        miu_rly_arr1D[j] = meanvalue


    ssm_sl_arr2D = np.nanmean(ssm_sl_arr3D, axis=0)
    ssm_rly_arr2D = np.nanmean(ssm_rly_arr3D, axis=0)
    sd_sl_arr2D = np.nanmean(sd_sl_arr3D, axis=0)
    sd_rly_arr2D = np.nanmean(sd_rly_arr3D, axis=0)

    return ssm_sl_arr2D, ssm_rly_arr2D, sd_sl_arr2D, sd_rly_arr2D, miu_sl_arr1D, miu_rly_arr1D



if __name__=='__main__':
    inPath = '../'

    """
    In other script, I reorganized the ML results as follows (example):
    
    y_hat = ML1_sly_lgb_mean.predict(ML1_sly_X_pred_all) # mean
    
    y_pred_cqr, y_pis_cqr = ML1_sly_mapie_cqr.predict_interval(ML1_sly_X_pred_all) # quantile
    
    y_up_hat = y_pis_cqr[:, 1, 0]
    y_median_hat = y_pred_cqr
    y_low_hat = y_pis_cqr[:, 0, 0]

    ML1_sly_y_pred_all = np.column_stack((X_info_test_valid, y_hat, y_up_hat, y_median_hat, y_low_hat, y_test_valid))
    """

    ML1_sly_y_pred_all_arr = np.load(f'{inPath}/ML1_sly_y_pred_all.npy')
    ML1_rly_y_pred_all_arr = np.load(f'{inPath}/ML1_rly_y_pred_all.npy')
    ML2_sly_y_pred_all_arr = np.load(f'{inPath}/ML2_sly_y_pred_all.npy')
    ML2_rly_y_pred_all_arr = np.load(f'{inPath}/ML2_rly_y_pred_all.npy')
    ML3_sly_y_pred_all_arr = np.load(f'{inPath}/ML3_sly_y_pred_all.npy')
    ML3_rly_y_pred_all_arr = np.load(f'{inPath}/ML3_rly_y_pred_all.npy')

    referIDArr = np.load('../referIDArr.npy')


    # @title run functions
    ML1_sl_arr_corrected, ML2_sl_arr_corrected, ML3_sl_arr_corrected, \
        ML1_rly_arr_corrected, ML2_rly_arr_corrected, ML3_rly_arr_corrected = correct_2_ML3(ML1_sly_y_pred_all_arr,
                                                                                            ML2_sly_y_pred_all_arr,
                                                                                            ML3_sly_y_pred_all_arr,
                                                                                            ML1_rly_y_pred_all_arr,
                                                                                            ML2_rly_y_pred_all_arr,
                                                                                            ML3_rly_y_pred_all_arr)

    ssm_sl_arr, ssm_rly_arr, sd_sl_arr, \
        sd_rly_arr, miu_sl_arr, miu_rly_arr = calculate_sd_and_miu(ML1_sl_arr_corrected,
                                                                   ML2_sl_arr_corrected,
                                                                   ML3_sl_arr_corrected,
                                                                   ML1_rly_arr_corrected,
                                                                   ML2_rly_arr_corrected,
                                                                   ML3_rly_arr_corrected)

    ssm_sl_arr2D, ssm_rly_arr2D, sd_sl_arr2D, \
        sd_rly_arr2D, miu_sl_arr1D, miu_rly_arr1D = reorgnize_2_2d(ssm_sl_arr,
                                                                   ssm_rly_arr,
                                                                   sd_sl_arr,
                                                                   sd_rly_arr,
                                                                   miu_sl_arr,
                                                                   miu_rly_arr,
                                                                   referIDArr)

    outPath = '../'
    np.save(f'{outPath}/ssm_sl_arr2D.npy', ssm_sl_arr2D)
    np.save(f'{outPath}/ssm_rly_arr2D.npy', ssm_rly_arr2D)
    np.save(f'{outPath}/sd_sl_arr2D.npy', sd_sl_arr2D)
    np.save(f'{outPath}/sd_rly_arr2D.npy', sd_rly_arr2D)
    np.save(f'{outPath}/miu_sl_arr1D.npy', miu_sl_arr1D)
    np.save(f'{outPath}/miu_rly_arr1D.npy', miu_rly_arr1D)
    np.save(f'{outPath}/referIDArr.npy', referIDArr)

    del ssm_sl_arr, ssm_rly_arr, sd_sl_arr, sd_rly_arr, miu_sl_arr, miu_rly_arr

    print()
