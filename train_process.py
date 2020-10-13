import pandas as pd
import shutil
import os
import json
import math
import time
import numpy as np
import sys
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing import Manager
import os
import time


def MzDLoad(msfile, idx):
    msdict = dict()
    f = open(msfile)
    flag = 0
    mzs = []
    ms_scan = 0
    count = 0
    for ms in f:
        ms = ms.strip()
        if ms[0].isalpha() is True:
            if ms[0] == 'S':
                if flag > 0:
                    msdict[ms_scan] = mzs
                    count += 1
                    mzs = []
                ms_scan = str(idx) + '_' + str(int(ms.split()[1]))
                flag += 1
            else:
                continue
        else:
            mzs.append(ms.split(' '))
    msdict[ms_scan] = mzs
    f.close()
    return msdict


def compare(ms_mz, apl_mz, e):
    exist = False
    ms_mz = round(float(ms_mz), 3)
    apl_mz = round(float(apl_mz), 3)
    control = abs(ms_mz - apl_mz)
    if control < ms_mz * e:
        exist = True
    return exist


def BinarySearch(scan, l, r, x, e):
    while l <= r:
        mid = int(l + (r - l) / 2)
        if scan[mid][0] == x:
            return True, mid
        elif scan[mid][0] < x:
            l = mid + 1
        else:
            r = mid - 1
    if l > len(scan) - 1:
        exist = compare(x, scan[r][0], e)
        mid = r
    elif r < 0:
        exist = compare(x, scan[l][0], e)
        mid = l
    else:
        exist = compare(x, scan[l][0], e) | compare(x, scan[r][0], e)
        if compare(x, scan[l][0], e):
            mid = l
        else:
            mid = r
    return exist, mid


def chargeDetection(scan, key, return_dict):
    scan = np.asarray(scan, dtype=float)
    cz_list = np.asarray([])
    for peak_id in range(len(scan)):
        wsize = math.ceil(scan[peak_id][0] / 1000)  # mz window_size
        start_exist, start_id = BinarySearch(scan, 0, len(
            scan) - 1, scan[peak_id][0] - wsize, 0.0001)
        stop_exist, stop_id = BinarySearch(scan, 0, len(
            scan) - 1, scan[peak_id][0] + wsize, 0.0001)
        if start_exist == False:
            start_id = peak_id
        if stop_exist == False:
            stop_id = peak_id
        best_score = -1
        bestCZ = 0
        flag = 0
        for cz in range(1, 4):
            sum = 0
            for pair_id in range(start_id, stop_id + 1):
                if pair_id == 0:
                    #sum += fval(scan[:, 0][pair_id]) * fval(scan[:, 0][pair_id] + ((1 / cz) / 2))
                    #sum += fval(scan[:, 0][pair_id]) * fval(scan[:, 0][pair_id] + (1 / cz))
                    exist, mid = BinarySearch(scan, 0, len(
                        scan) - 1, scan[pair_id][0], 0.001 / cz)
                    if exist:
                        f1 = scan[mid][1]
                    else:
                        f1 = 0
                    exist, mid = BinarySearch(scan, 0, len(
                        scan) - 1, scan[pair_id][0] + (1 / cz), 0.001 / cz)
                    if exist:
                        f2 = scan[mid][1]
                    else:
                        f2 = 0
                    sum += f1 * f2

                elif pair_id == len(scan) - 1:
                    #sum += fval(scan[:, 0][pair_id] - ((1 / cz) / 2)) * fval(scan[:, 0][pair_id])
                    #sum += fval(scan[:, 0][pair_id] - (1 / cz)) * fval(scan[:, 0][pair_id])
                    exist, mid = BinarySearch(scan, 0, len(
                        scan) - 1, scan[pair_id][0] - (1 / cz), 0.00001 / cz)
                    if exist:
                        f1 = scan[mid][1]
                    else:
                        f1 = 0
                    exist, mid = BinarySearch(scan, 0, len(
                        scan) - 1, scan[pair_id][0], 0.00001 / cz)
                    if exist:
                        f2 = scan[mid][1]
                    else:
                        f2 = 0
                    sum += f1 * f2

                else:
                    # sum+=fval(scan[:,0][pair_id]-((1/cz)/2))*fval(scan[:,0][pair_id]+((1/cz)/2))
                    #sum += fval(scan[:, 0][pair_id] - (1 / cz)) * fval(scan[:, 0][pair_id] + (1 / cz))
                    exist, mid = BinarySearch(scan, 0, len(
                        scan) - 1, scan[pair_id][0] - (1 / cz), 0.00001 / cz)
                    if exist:
                        f1 = scan[mid][1]
                    else:
                        f1 = 1
                    exist, mid = BinarySearch(scan, 0, len(
                        scan) - 1, scan[pair_id][0] + (1 / cz), 0.00001 / cz)
                    if exist:
                        f2 = scan[mid][1]
                    else:
                        f2 = 1

                    sum += f1 * f2

            if sum >= best_score:
                if cz > 1:
                    if abs(sum - best_score) < 0.01:
                        flag += 1
                best_score = sum
                bestCZ = cz

            if flag == 2:
                bestCZ = 0

        cz_list = np.append(cz_list, bestCZ)

    return_list = np.append(scan, np.reshape(
        cz_list, (len(cz_list), 1)), axis=1)
    return_dict[key] = return_list
    # return np.append(scan,np.reshape(cz_list,(len(cz_list),1)),axis=1)
    return return_dict


def writeMzs(mzs, fout):
    for peak in mzs:
        s = str(peak[0]) + ' ' + str(peak[1]) + ' ' + str(peak[2]) + '\n'
        fout.write(s)
    fout.write('\n')


def exp(msfile, outfile):
    # 1. merge different apl files and remain unique scans
    # merge_apl()
    # 2. Load ms2 file and apl (after merge) file
    namearray = msfile.split('.')
    name = namearray[0]
    fileinfo = name.split('_')
    idx = int(fileinfo[-1])
    print('process: file ' + msfile)
    start=time.time()
    msdict = MzDLoad(msfile, idx)
    end=time.time()
    print('read data:'+str(end-start))
    # 3. detect and generate
    start=time.time()
    manager = Manager()
    return_dict = manager.dict()
    processors = os.cpu_count()
    pool = Pool(processes=processors)
    for key in msdict:
        pool.apply_async(chargeDetection, args=(msdict[key], key, return_dict))

    pool.close()
    pool.join()
    end=time.time()
    print('charge detection:'+str(end-start))

    start=time.time()
    with open(outfile, 'w') as f:
        for key in return_dict:
            f.write(key + '\n')
            writeMzs(return_dict[key], f)
    end=time.time()
    print('write data:'+str(end-start))

if __name__ == "__main__":
    msfile = sys.argv[1]
    outfile = sys.argv[2]
    start = time.time()
    exp(msfile, outfile)
    end = time.time()
    print(end - start)
