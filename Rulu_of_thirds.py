import cv2
import numpy as np
import random
import sys

def sanbunkatu(map, p0, p1):
    ih = map.shape[0] 
    iw = map.shape[1]
    # print(iw, ih)

    # 矩形の中心
    cx = (p0[0] + p1[0]) // 2
    cy = (p0[1] + p1[1]) // 2
    # print(cx, cy)
    cv2.rectangle(map, p0, p1, (0,0,255),3)

    # cv2.imwrite('rectangle.png', map)

    # 矩形のシェイプ
    w_side = p1[0] - p0[0] #横幅
    h_side = p1[1] - p0[1] #縦幅
    # 倍率
    r = random.uniform(3,4)
    # r = 5
    if w_side / h_side < 16 / 9:
        # 矩形の縦横幅に対するdst_mapの縦横までの距離
        h_dst = int(r * h_side)
        w_dst = int((h_dst / 9) * 16)
    else:
        w_dst = int(r * w_side)
        h_dst = int((w_dst / 16) * 9)
    print(r)
    # 横長の矩形でも試す

    # 仮の図形の座標
    # xd0とyd0も(0,0)の座標になる

    # 3分割の左上後転座標
    tcxs = [int((cx - w_dst / 3)),int((cx - w_dst * 2 / 3))]
    tcys = [int((cy - h_dst / 3)),int((cy - h_dst / 3))]  
    output_image = [] #三分割構図の左上と右上どっちを中心にしてるか
    for i in range(2): #左上と右上の場合を繰り返し処理
        xd0 = tcxs[i]
        yd0 = tcys[i] 

        xd1 = xd0 + w_dst 
        yd1 = yd0 + h_dst        #下の座標

        # print(xd0, yd0, xd1, yd1)
        dst_map = np.ones((h_dst, w_dst, 3), np.uint8) * 100 #255
        # print(dst_map)

        # 新画像の中にある入力画像の交点座標
        # xc0 = max(0, xd0)
        if 0 <= xd0:
            # mapと重なる交点座標と、一枚の新しい画像とみなしたときの座標が考えられるから2つ
            xsc0 = xd0 #dst_mapに基づく座標
            xdc0 = 0 #mapに基づく座標
        else:
            xsc0 = 0
            xdc0 = -xd0
        # yc0 = max(0, yd0)
        if 0 <= yd0:
            ysc0 = yd0
            ydc0 = 0
        else:
            ysc0 = 0
            ydc0 = -yd0
        # xc1 = min(iw, xd1)
        if iw <= xd1:
            xsc1 = iw
            xdc1 = w_dst - (xd1 - iw)  #xd1 - iw は右に飛び出た分の幅
        else:
            xsc1 = xd1
            xdc1 = w_dst #dst_mapに基づいた座標
        # yc1 = min(ih, yd1)
        if ih <= yd1:
            ysc1 = ih
            ydc1 = h_dst - (yd1 - ih)
        else:
            ysc1 = yd1
            ydc1 = h_dst

        # print(xc0, yc0, xc1, yc1)
        dst_map[ydc0:ydc1, xdc0:xdc1] = map[ysc0:ysc1, xsc0:xsc1]
        output_image.append(dst_map) #左上右上に配置される画像を2枚出力
    return output_image


if __name__ == "__main__":
    map = np.ones((1920, 1080, 3), np.uint8) * 255

    # cv2.imwrite('map.png', map)

    p0 = (int(sys.argv[1]), int(sys.argv[2])) #(200, 800)
    p1 = (int(sys.argv[3]), int(sys.argv[4])) #(400,1200)
    cv2.rectangle(map, p0, p1, (0,0,255),3) #矩形の描画
    dst_map = sanbunkatu(map, p0, p1)
    cv2.imwrite("out_sample1.png", dst_map)


    # cv2.imwrite('combine.png', img)