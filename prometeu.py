#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import sys
from threading import Thread
import threading
import os

import cv2
import time
import random

import multiprocessing
import itertools as it


from tools import *

#os.chdir(os.path.dirname(sys.argv[0]))


def assembler(frag_image, paint, heap, metodo, out, thread_id, threadLock):
    kp2, des2 = paint
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 100)

    fragments = []
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    end = False

    while True:

        threadLock.acquire()
        if len(heap) > 0:
            i = heap.pop()
            #print("Sou o", thread_id,"peguei a peça",i)
        else:
            #print(thread_id, 'out')
            end = True
        threadLock.release()
        if end: break

        #for angle in range(0,360,360//16):
        for angle in [0]:
            try:
                frag_rot = rotate(frag_image[i], angle)
                size = max(frag_rot.shape)
                frag_data = analise(frag_rot, metodo)
            except:
                print("Erro peça",i)
                break

            matches = flann.knnMatch(frag_data[1], des2, k=2)

            matches = [m for m,n in matches if m.distance < 0.7*n.distance]

            pontos = [[x.trainIdx, kp2[x.trainIdx].pt] for x in matches]
            pontos = tri_min(pontos, size)
            matches = [m for m in matches if m.trainIdx in pontos]

            if len(matches) >= 3:
                src_pts = np.float32([ frag_data[0][m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
                M = cv2.estimateRigidTransform(src_pts, dst_pts, fullAffine=False)
                if M is not None:

                    xfrag,yfrag = frag_image[i].shape
                    xfrag /= 2; yfrag /= 2

                    tx = M[0][0]*xfrag + M[0][1]*yfrag + M[0][2]
                    ty = M[1][0]*xfrag + M[1][1]*yfrag + M[1][2]

                    a = M[0][0]; c = M[0][1]
                    b = M[1][0]; d = M[1][1]

                    if np.isnan(a) or np.isnan(b):
                        r = np.arctan2(c,d)
                    else:
                        r = np.arctan2(-b,a)

                    r = np.degrees(r)

                    print(thread_id, 'get', i, ':', metodo)

                    rot = r+angle
                    if rot > 360: rot -= 360
                    if rot < -360: rot += 360

                    fragments.append([M, angle, tx, ty, rot, i, dst_pts])

                    break
    out[thread_id] = fragments

def main(args):

    #target = 'Domenichino_Virgin-and-unicorn.jpg'
    #target = 'Fra_Angelico_Cristoderiso_2536x3172.jpg'
    target = 'target.jpg'
    #target = input("Target:")

    total = 0
    for root, dirs, files in os.walk("./frag_eroded"):
        for file in files:
            n = int((file.split('_')[2]).split('.')[0])
            if n > total: total = n
    #total = 1191
    #total = int(input("npecas:"))

    #total = 39
    # modificado cuidado
    frag_image = [cv2.imread('frag_eroded/frag_eroded_'+str(i)+'.png', 0)for i in range(0, total+1)]
    #frag_image = [cv2.imread('frag_eroded/frag_eroded_'+str(i)+'.png', 0)for i in range(18, 20)]


    paint = cv2.imread(target, 0)
    paint_color = cv2.imread(target)
    height, width = paint.shape

    #n_cores = int(input("Cores:"))
    n_cores = -1

    if n_cores == -1: n_cores = multiprocessing.cpu_count()
    list_id = [i for i in range(len(frag_image)) if frag_image[i] is not None]
    all_id = list_id.copy()
    total = len(list_id) #total validos



    fragments_final = []
    threadLock = threading.Lock()
    #for metodo in [1]:
    for metodo in [1,2,3,4,5,6,7,8,9,10]:
        print("Layer",metodo)
        paint_info = analise(paint, metodo)

        threads = [None] * n_cores
        fragments = [None] * n_cores

        for i in range(len(threads)):
            threads[i] = Thread(target=assembler, args=(frag_image, paint_info, list_id, metodo, fragments, i, threadLock))
            threads[i].start()
        for i in range(len(threads)): threads[i].join()

        fragments = [y for x in fragments for y in x]

        # filtro
        '''
        tmp = []
        while(len(fragments) != 0): tmp.append(fragments.pop())
        for frag in tmp:
            if filtro(paint_color, frag):
                fragments.append(frag)
        del tmp
        '''

        for frag in fragments: all_id.remove(frag[5])
        list_id = all_id.copy()

        while(len(fragments) != 0): fragments_final.append(fragments.pop())

        p = (total-len(list_id))/total
        print('%',p)

    #monta
    #blank_image = np.zeros((len(paint),len(paint[0]),3), np.uint8)
    blank_image = cv2.cvtColor(paint,cv2.COLOR_GRAY2RGB)
    # aumento de brilho
    blank_image = cv2.add(blank_image,np.array([50.0]))


    log_file = open("log.txt","w")
    with open("percent.txt","w") as file:
        file.write(str(p)+"%")

    seq = 0
    get = []
    fragments_final.sort(key=lambda x: x[5])
    '''
    frag [M, angle, tx, ty, r, i, dst_pts]
    '''
    for frag in fragments_final:

        #FILTRO
        '''
        if not filtro(paint_color, frag):
            print("Rejeita:", frag[5])
            continue
        '''

        log_file.write(str(frag[5])+' '+str(frag[2])+' '+str(frag[3])+' '+str(frag[4])+'\n')
        im = cv2.warpAffine(rotate(cv2.imread('frag_eroded/frag_eroded_'+str(frag[5])+'.png', -1), frag[1]), frag[0], (width, height))
        blank_image = paste(blank_image, im, 0, 0)
        cv2.imwrite('passos/frag_'+str(seq)+'_'+str(frag[5])+'.png', blank_image)
        seq += 1
        get.append(frag[5])

    log_file.close()
    print(seq, 'peças montadas')
    print("Get:", get)

    #result
    cv2.imwrite('RECONSTRUCTED_fresco'+'.png', blank_image)


    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
