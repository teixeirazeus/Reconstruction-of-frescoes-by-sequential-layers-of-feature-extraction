#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import itertools as it
import numpy as np

def hist_comp(src_base, src_test):
    hsv_base = cv2.cvtColor(src_base, cv2.COLOR_BGR2HSV)
    hsv_test = cv2.cvtColor(src_test, cv2.COLOR_BGR2HSV)

    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges
    channels = [0, 1]
    hist_base = cv2.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_test = cv2.calcHist([hsv_test], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_test, hist_test, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return cv2.compareHist(hist_base, hist_test, 0)

def dis(a,b):
    return np.sqrt(abs(((b[0]-a[0])**2)+((b[1]-a[1])**2)))

def baricentro(p):
    x = (p[0][0]+p[1][0]+p[2][0])/3
    y = (p[0][1]+p[1][1]+p[2][1])/3
    return x,y

def corta(img, pts):
    pts = np.array([pts[0][0], pts[1][0], pts[2][0]])
    xcentro, ycentro = baricentro(pts)
    size = max(dis(pts[0],pts[1]),dis(pts[1],pts[2]),dis(pts[0],pts[2]))
    origin = [(xcentro-(size/2)),(ycentro-(size/2))]
    x = int(origin[0])
    y = int(origin[1])
    w = h = int(size)
    return img[y:y+h, x:x+w].copy()

def paste(base, cima, x_offset, y_offset):

    y1, y2 = int(y_offset), int(y_offset + cima.shape[0])
    x1, x2 = int(x_offset), int(x_offset + cima.shape[1])

    alpha_s = cima[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        base[y1:y2, x1:x2, c] = (alpha_s * cima[:, :, c] + alpha_l * base[y1:y2, x1:x2, c])
    return base

def percent(falta):
    count = 0
    for i in range(len(falta)):
        if falta[i] == False:
            count += 1
    return count/len(falta)

def distance_dot(a,b):
    return np.sqrt(abs(((b[1][0]-a[1][0])**2)+((b[1][1]-a[1][1])**2)))

def tri_min(pontos, max_size):
    if len(pontos) < 3 or max_size <= 0:
        return []

    # pega o menor triangulo nos pontos menor que max
    tri_list = list(it.combinations(pontos, 3))
    size = []
    for tri in tri_list:
        max = 0

        for i,j in [[0,1],[1,2],[2,0]]:
            d = distance_dot(tri[i],tri[j])
            if d > max and not np.isnan(d):
                max = d
        if max <= max_size:
            size.append([tri,max])
    if len(size) == 0:
        return []
    #print(size)
    matches = sorted(size, key=lambda x: x[1])
    tri, value = matches[int(len(matches)/2)]
    if value <= max_size:
        return [x for x,y in tri]
    else:
        tri, value = matches[0]
        if value <= max_size:
            return [x for x,y in tri]

    return []


def rotate(image, angle):
    if angle == 0: return image
    if len(image.shape) > 2:
        rows, cols, alpha = image.shape
    else:
        rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    dst = cv2.warpAffine(image, M, (cols,rows))
    return dst


def filtro(paint, frag):
    try:
        height, width, a = paint.shape

        original = corta(paint, frag[6])

        im = cv2.warpAffine(rotate(cv2.imread('frag_eroded/frag_eroded_'+str(frag[5])+'.png', -1), frag[1]), frag[0], (width, height))

        modificado = paste(paint, im, 0, 0)
        modificado = corta(modificado, frag[6])

        rate = hist_comp(original, modificado)
        print("H", frag[5], rate)
        if rate > 0.4:
            return True
        else:
            return False
    except:
        return True

def analise(file, mode):
    if mode == 1:
        si = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = si.detectAndCompute(file, None)
    elif mode == 2:
        su = cv2.xfeatures2d.SURF_create(extended=0)
        kp1, des1 = su.detectAndCompute(file, None)
    elif mode == 3:
        su = cv2.xfeatures2d.SURF_create(extended=1)
        kp1, des1 = su.detectAndCompute(file, None)
    elif mode == 4:
        su = cv2.xfeatures2d.SURF_create(extended=0, nOctaves=1, hessianThreshold=100)
        kp1, des1 = su.detectAndCompute(file, None)
    elif mode == 5:
        su = cv2.xfeatures2d.SURF_create(extended=1, nOctaves=1, hessianThreshold=100)
        kp1, des1 = su.detectAndCompute(file, None)
    elif mode == 6:
        si = cv2.xfeatures2d.SIFT_create(nOctaveLayers=2,contrastThreshold=0.02, edgeThreshold=15)
        kp1, des1 = si.detectAndCompute(file, None)
    elif mode == 7:
        si = cv2.xfeatures2d.SIFT_create(nOctaveLayers=2,contrastThreshold=0.01, edgeThreshold=25)
        kp1, des1 = si.detectAndCompute(file, None)
    elif mode == 8:
        si = cv2.xfeatures2d.SIFT_create(nOctaveLayers=2,contrastThreshold=0.01, edgeThreshold=25, sigma=1)
        kp1, des1 = si.detectAndCompute(file, None)
    elif mode == 9:
        si = cv2.xfeatures2d.SIFT_create(nOctaveLayers=2,contrastThreshold=0.01, edgeThreshold=30, sigma=0.8)
        kp1, des1 = si.detectAndCompute(file, None)
    elif mode == 10:
        si = cv2.xfeatures2d.SIFT_create(nOctaveLayers=1,contrastThreshold=0.01, edgeThreshold=40, sigma=0.7)
        kp1, des1 = si.detectAndCompute(file, None)

    return [kp1, des1]
