import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
import os
import pandas as pd

#gt_dir = 'test_processed'
gt_dir ='test_processed_uclan'
#res_dir = 'results/test_processed'
res_dir = 'results/test_processed_uclan'

def get_Landmarks(img, gt_res):
    if gt_res == 'gt':
        img = img.replace(img[-3:],'ptv')
        img = os.path.join(gt_dir,img)
        data = pd.read_csv(img,sep=',',header=None)
    elif gt_res == 'res':
        img = img.replace(img[-4:],'_pred.pts')
        img = os.path.join(res_dir,img)
        data = pd.read_csv(img,sep=' ',header=None)
    return data.values.astype(np.int64)


def LandmarkError(img_list, normalization='centers', showResults=False, verbose=False):
    errors = []

    for i, img in enumerate(img_list):
        gtLandmarks = get_Landmarks(img, 'gt')

        resLandmarks = get_Landmarks(img, 'res')
        #print(gtLandmarks.shape)
        if normalization == 'centers':
            normDist = np.linalg.norm(np.mean(gtLandmarks[36:42], axis=0) - np.mean(gtLandmarks[42:48], axis=0))
        elif normalization == 'corners':
            normDist = np.linalg.norm(gtLandmarks[36] - gtLandmarks[45])
        elif normalization == 'diagonal':
            height, width = np.max(gtLandmarks, axis=0) - np.min(gtLandmarks, axis=0)
            normDist = np.sqrt(width ** 2 + height ** 2)
        #print(normDist)
        error = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks) ** 2, axis=1))) / normDist
        errors.append(error)
        if verbose:
            print("{0}: {1}".format(i, error))

        if showResults:
            plt.imshow(img[0], cmap=plt.cm.gray)
            plt.plot(resLandmarks[:, 0], resLandmarks[:, 1], 'o')
            plt.show()

    if verbose:
        print("Image idxs sorted by error")
        print(np.argsort(errors))
    avgError = np.mean(errors)
    print("Average error: {0}".format(avgError))

    return errors


def AUCError(errors, failureThreshold, step=0.0001, showCurve=True):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))

    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    print("AUC @ {0}: {1}".format(failureThreshold, AUC))
    print("Failure rate: {0}".format(failureRate))
    plt.plot(xAxis, ced)
    if showCurve:
        plt.show()
    plt.savefig('AUC.png')

def main():
    img_list = []
    for file in os.listdir(gt_dir):
        if file.endswith('png') or file.endswith('jpg'):
            img_list.append(file)
    errors = LandmarkError(img_list,normalization='centers')
    AUCError(errors,0.08)    

if __name__ == '__main__':
    main()
