from scipy.stats import rankdata
import numpy as np
import concurrent.futures
import os
import subprocess

import pandas as pd


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


max_workers = os.cpu_count()
def write_settings_to_txt(settings, filename="settings.txt"):
    with open(filename, "w") as f:
        for key, value in settings.items():
            # Convert lists to string format with brackets
            if isinstance(value, list):
                value_str = "[" + ", ".join(map(str, value)) + "]"
            else:
                value_str = str(value).lower() if isinstance(value, bool) else str(value)
            f.write(f"{key}: {value_str}\n")

settings = {
        "verbosity": "none",
        "site_ID": 11,
        "num_hours": 78912,
        "layer_thickness_cm": [5, 10, 15, 30, 40],
        "giuh_ordinates": [0.06, 0.51, 0.28, 0.12, 0.03],
        "layer_soil_type":[1,2,3,4,5],
        "timestep_h": 1,
        "endtime_s": 3600 * 78912,
        "field_capacity_psi": 340.9,
        "wilting_point_psi": 15495.0,
        "initial_psi": 2000.0,
        "initial_sm_perturb": 0,
        "TO_enabled": False,
        "adaptive_timestep": False,
        "free_drainage_enabled": False,
        "use_closed_form_G": False,
        "root_zone_depth_cm": 60,
        "mbal_tol": 80.0,
        "ponded_depth_max_cm": 0,
        "forcing_resolution_h": 1,
        "max_valid_soil_types": 12
    }


ssm_sl_arr2D = np.load('ssm_sl_arr2D.npy')
print(ssm_sl_arr2D.shape)
ssm_rly_arr2D = np.load('ssm_rly_arr2D.npy')
print(ssm_rly_arr2D.shape)
sd_sl_arr2D = np.load('sd_sl_arr2D.npy')
print(sd_sl_arr2D.shape)
sd_rly_arr2D = np.load('sd_rly_arr2D.npy')
print(sd_rly_arr2D.shape)
soilid_arr = np.load('soilid_arr.npy')
print(soilid_arr.shape)
referIDArr = np.load('referIDArr.npy')
print(referIDArr.shape)


PET_arr = np.load('PET_arr.npy')
print(PET_arr.shape)
Prec_arr = np.load('Prec_arr.npy')
print(Prec_arr.shape)

def parallel_processing(siteIndex):
    siteId = referIDArr[siteIndex]
    print(f'is running site {siteId}')

    site_sm_sly = ssm_sl_arr2D[:, siteIndex].astype(np.float64)
    site_sd_sly = sd_sl_arr2D[:, siteIndex].astype(np.float64)
    site_sm_rly = ssm_rly_arr2D[:, siteIndex].astype(np.float64)
    site_sd_rly = sd_rly_arr2D[:, siteIndex].astype(np.float64)
    site_prec_sly = Prec_arr[:, siteIndex].astype(np.float64)
    site_PET_sly = PET_arr[:, siteIndex].astype(np.float64)

    site_prec_sly.tofile(f'site{siteId}_prec.bin')
    site_PET_sly.tofile(f'site{siteId}_PET.bin')

    settings["layer_soil_type"] = list(soilid_arr[:, siteIndex])
    settings["site_ID"] = siteId

    # Example usage
    write_settings_to_txt(settings, filename=f'site{siteId}_settingInfo.txt')

    try:
        command = [
            "./LGAR",
            "--filename_prec", f'site{siteId}_prec.bin',
            "--filename_pet", f'site{siteId}_PET.bin',
            "--settings_path", f'site{siteId}_settingInfo.txt'
        ]
        subprocess.run(command, check=True)

        # cdf matching
        df = pd.read_csv(f'Original_LGAR_{siteId}.csv', header=None)
        lgar_sm_sly = df.iloc[:, 1].to_numpy()
        print(np.mean(lgar_sm_sly))
        lgar_sm_rly = df.iloc[:, 2].to_numpy()
        print(np.mean(lgar_sm_rly))

        masksly = ~np.isnan(site_sm_sly)
        site_sm_sly[masksly] = cdf_matching(site_sm_sly[masksly], lgar_sm_sly)
        maskrly = ~np.isnan(site_sm_rly)
        site_sm_rly[maskrly] = cdf_matching(site_sm_rly[maskrly], lgar_sm_rly)

        site_sm_sly.tofile(f'site{siteId}_sm_sly.bin')
        site_sd_sly.tofile(f'site{siteId}_sd_sly.bin')
        site_sm_rly.tofile(f'site{siteId}_sm_rly.bin')
        site_sd_rly.tofile(f'site{siteId}_sd_rly.bin')

        command = [
            "./LGAR_EnKF003002",
            "--filename_prec", f'site{siteId}_prec.bin',
            "--filename_pet", f'site{siteId}_PET.bin',
            "--filename_sm_sly", f'site{siteId}_sm_sly.bin',
            "--filename_sm_rly", f'site{siteId}_sm_rly.bin',
            "--filename_sd_sly", f'site{siteId}_sd_sly.bin',
            "--filename_sd_rly", f'site{siteId}_sd_rly.bin',
            "--settings_path", f'site{siteId}_settingInfo.txt'

        ]
        subprocess.run(command, check=True)
    except:
        print()


    # delete files
    files_to_delete = [
        f'site{siteId}_prec.bin',
        f'site{siteId}_PET.bin',
        f'site{siteId}_sm_sly.bin',
        f'site{siteId}_sm_rly.bin',
        f'site{siteId}_sd_sly.bin',
        f'site{siteId}_sd_rly.bin',
        f'site{siteId}_settingInfo.txt',
        f'Original_LGAR_{siteId}.csv'
    ]

    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except FileNotFoundError:
            print(f"File not found, skipped: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")


with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(parallel_processing, range(ssm_sl_arr2D.shape[1]))