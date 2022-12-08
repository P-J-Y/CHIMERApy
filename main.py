import sunpy
import numpy as np
import scipy as sp
import os
# import pillow as pil
import sunpy.map
from aiapy.calibrate import normalize_exposure, register, update_pointing
import cv2
import matplotlib.pyplot as plt
import main_utils
import json
import tracking

#
garr = main_utils.psf_gaussian(4096, [2000, 2000])

# ==============Finds all fits files==============
###################
year_str = "2021"
month_str = "01"
day_str = "02"
hour_str = "00"
min_str = "00"

###################


# data_dir = "/Users/gyh/Desktop/research/CH_detect/py/data1"
data_dir = os.getcwd() + '\\data\\' + year_str + '\\' + month_str
# data_dir = os.getcwd() + '\\data\\thedata'
# os.chdir(data_dir)  # data_dir为数据所在的目录

##########################
#  #如何自动识别和匹配文件？
# # 提取文件名
# f = os.listdir(data_dir)
# f171 = []
# f193 = []
# f211 = []
# fhmi = []
#
# for i in range(0, np.size(f)):
#     if "171" in f[i]: f171 = data_dir + '//' + f[i]
#     if "193" in f[i]: f193 = data_dir + '//' + f[i]
#     if "211" in f[i]: f211 = data_dir + '//' + f[i]
#     if "magneto" in f[i]: fhmi = data_dir + '//' + f[i]
#
# if f171 == [] or f193 == [] or f211 == [] or fhmi == []:
#     print("Not all files are present.")
#
# # fil = []
# # fil.append(f171)
# # fil.append(f193)
# # fil.append(f211)

f171 = data_dir+'//'+'AIA'+year_str+month_str+day_str+'_'+hour_str+min_str+'_0171.fits'
f193 = data_dir+'//'+'AIA'+year_str+month_str+day_str+'_'+hour_str+min_str+'_0193.fits'
f211 = data_dir+'//'+'AIA'+year_str+month_str+day_str+'_'+hour_str+min_str+'_0211.fits'
fhmi = data_dir+'//'+'hmi.M_720s.'+year_str+month_str+day_str+'_'+hour_str+min_str+'00_TAI.fits'
#####################################

# =====Reads in data=====
# read_sdo,fhmi,hin,hd, /use_shared_lib
# read_sdo,fil,ind,data, /use_shared_lib
# 看sunpy的对应程序
map_il = sunpy.map.Map(f171, f193, f211)  # map_il为一个list
map_hmi = sunpy.map.Map(fhmi)  # map_hmi就是一个Map

# =====Attempts to verify data is level 1.5=====
if map_il[0].meta['lvl_num'] != 1.5:
    for i in range(3):
        m_updated_pointing = update_pointing(map_il[i])
        m_registered = register(m_updated_pointing)
        m_normalized = normalize_exposure(m_registered)
        map_il[i] = m_normalized

# if map_hmi.meta['lvl_num'] != 1.5:
#     m_updated_pointing.append = update_pointing(map_hmi)
#     m_registered = register(m_updated_pointing)
#     m_normalized = normalize_exposure(m_registered)
#     map_hmi = m_normalized


ind = []
data = []
for i in range(3):
    ind.append(map_il[i].meta)
    data.append(map_il[i].data)

hin = map_hmi.meta
hd = map_hmi.data

# =====Rotates magnetogrames if necessary======
if hin['crota2'] > 90: hd = np.rot90(hd, 2)  # 用rot90实现idl中rotate(hd,2)

# =====Resize and smooth image=====
# data = float(data)
for i in range(3): data[i] = cv2.resize(data[i], (1024, 1024))
# data=rebin(data,1024,1024,3)
# 或许可以使用cv2.resize，但是这里是三维的

# =====Alternative coordinate systems=====
# wcs=fitshead2wcs(ind[1]) 不知道怎么改？好像不需要
ind[0]['naxis1'] = 4096
ind[0]['naxis2'] = 4096
mxlon = 88  # Maximum longitude  allowed for CH centroid
for i in range(3): data[i] = cv2.resize(data[i], (4096, 4096))

s = data[0].shape  # (4096,4096)
# 暂时不管了，有点奇怪
outdata = data # 最后用于输出的data
outhd = cv2.resize(hd, (4096, 4096))

# =======setting up arrays to be used============
ident = 1
iarr = np.zeros(s, dtype='byte')
offarr = iarr
mas = np.zeros(s)
mak = np.zeros(s)
msk = np.zeros(s)
tem = np.zeros((4000, 4000))
tmp = np.zeros((4000, 4000))
tep = np.zeros((4000, 4000))
defi = np.zeros(s)
circ = np.zeros(s, dtype='int')
n = np.zeros(1, dtype='int')
x = np.zeros(1)
y = x
ch = np.zeros(1, dtype='int')

# =======creation of a 2d gaussian for magnetic cut offs===========
r_inner = (s[0] / 2.0) - 450  # solar radius [pixel], ~975"
xgrid = np.outer(np.ones(s[1]), np.arange(s[0]))
ygrid = np.outer(np.arange(s[1]), np.ones(s[0]))

center = [int(s[0] / 2.), int(s[1] / 2.)]

# w = ((xgrid-center[0])**2+(ygrid-center[1])**2 > r**2).nonzero()
# w = where((xgrid-center[0])^2+(ygrid-center[1])^2 gt r^2) ;	positions outside the solar disk
# circ[w] = 1.0 ;	setting 1.0 for pixels outside the solar disk
# garr=psf_gaussian(npixel=s[1],FWHM=[2000,2000])
# # 	Return a point spread function having Gaussian profiles as a 2D image
# # 	namely, the 2d gaussian for magnetic cut offs
# garr[w]=1.

# ======creation of array for CH properties==========
# %%%%%%%%%%%%%%%% 这个就是个表格，等会再改成python的格式
# formtab[0]='ID      XCEN       YCEN   CENTROID       X_EB       Y_EB       X_WB       Y_WB       X_NB       Y_NB       X_SB       Y_SB      WIDTH     WIDTH°       AREA      AREA%        <B>       <B+>       <B->       BMAX       BMIN     TOT_B+     TOT_B-      <PHI>     <PHI+>     <PHI->'
# formtab[1]='num        "          "         H°          "          "          "          "          "          "          "          "         H°          °       Mm^2          %          G          G          G          G          G          G          G         Mx         Mx         Mx'

# =====Sort data by wavelength=====
sortidx = tuple(sorted(range(len(ind)), key=lambda x: ind[x]['wavelnth']))
ind.sort(key=lambda x: x['wavelnth'])
data = np.array(data)[sortidx, :, :]

# =====Normalises data with respect to exposure time=====
for i in range(3):
    data[i, :, :] = data[i, :, :] / ind[i]['exptime']

# =====removes negative data values=====
data[data < 0] = 0

# =====Readies maps, specifies solar radius and calculates conversion value of pixel to arcsec=====
# 如果初始数据文件是1024×1024的低分辨率数据，用下面的步骤改一下头文件中的pixel size和crpix
rs = map_il[1].rsun_obs  # solar radius 975" <Quantity 976.002956 arcsec>

if ind[1]['cdelt1'] > 1:
    for i in range(3):
        ind[i]['cdelt1'] = ind[i]['cdelt1'] / 4
        ind[i]['cdelt2'] = ind[i]['cdelt2'] / 4
        ind[i]['crpix1'] = ind[i]['crpix1'] * 4
        ind[i]['crpix2'] = ind[i]['crpix2'] * 4

dattoarc = ind[1]['cdelt1']

# ======Seperate each image to an individual array=======
dat0 = data[0, :, :]
dat1 = data[1, :, :]
dat2 = data[2, :, :]

# ======Get pixels with useful intensities (>4000) and on disk======
# r_outer= ind[0]['r_sun']  # 是像素   1626.6714716666668
w = np.where(
	np.logical_and(
		(xgrid - center[0]) ** 2 + (ygrid - center[1]) ** 2 < r_inner ** 2,
		dat0 < 4000,
        (dat1 < 4000) & (dat2 < 4000)
	)
)

# =====create intensity ratio arrays=============
# 这一节是把日盘（4000应该是饱和或者耀斑得意思？？？）筛出来
for i in range(np.size(w[0])):
    tem[w[0][i], w[1][i]] += 1	# 171vs193
    tmp[w[0][i], w[1][i]] += 1	# 171vs211
    tep[w[0][i], w[1][i]] += 1	# 193vs211
# 只有满足上面条件（强度小于4000，在日面上）的点，值才为+1，其余为0

# ============make a multi-wavelength image for contours==================
truecolorimage=np.zeros((s[0],s[1],3))
truecolorimage[:,:,2]= main_utils.bytscl(np.log10(outdata[0]), Max=3.9, Min=1.2)
truecolorimage[:,:,1]= main_utils.bytscl(np.log10(outdata[1]), Max=3.0, Min=1.4)
truecolorimage[:,:,0]= main_utils.bytscl(np.log10(outdata[2]), Max=2.7, Min=0.8)
#	注意这里取了对数

t0=truecolorimage[:,:,0]
t1=truecolorimage[:,:,1]
t2=truecolorimage[:,:,2]

# ====create 3 segmented bitmasks bitmasks=====
msk[t2/t0 > (dat0.mean()*0.6357)/dat2.mean()] = 1
mak[t0 + t1 < 0.7*(dat1.mean() + dat2.mean())] = 1
mas[t2/t1 > ((dat0.mean()*1.5102)/(dat1.mean()))] = 1

# ====plot tricolour image with lon/lat conotours=======
ax = np.arange(s[0])
ay = np.arange(s[1])

# ======removes off detector mis-identifications and seperates on-disk and off-lib CHs==========

circ[:] = 1
rm = (s[0]/2.0)-100
r_outer = (rs/dattoarc).value #像素单位
xgrid = np.outer(np.arange(s[0]),np.ones(s[1],dtype='float'))
ygrid = np.outer(np.ones(s[0],dtype='float'),np.arange(s[1]))
center = [int(s[0]/2.), int(s[1]/2.)]
w = np.where((xgrid - center[0]) ** 2 + (ygrid - center[1]) ** 2 >= rm ** 2)
circ[w] = 0
w = np.where((xgrid - center[0]) ** 2 + (ygrid - center[1]) ** 2 >= (r_outer - 10) ** 2)
circ[w] = 0
Def = mas * msk * mak * circ


# =====contours the identified datapoints=======
# fig = plt.figure(figsize=(12, 12))
# CS = plt.contour(ax, ay, Def, [0.99999,1], alpha = 0.9)
# seg = CS.allsegs[0]
Def_grey=np.uint8(Def*255)
contours,hierarchy = cv2.findContours(Def_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
area_thres = 1000 / (dattoarc**2) # 选出比这个大的面积，像素^2
contours_large = [] # 比较大的contours
Cs = [] # 每个较大contour的中心坐标
# areas = []
# mags = [] # 平均视向磁场
# =====cycles through contours=========
id = 1
CHs = []
for contour in contours:
    # =====only takes values of minimum surface length and calculates area======
    # =====finds centroid=======
    area = cv2.contourArea(contour) #单位是像素2
    if  area >= area_thres:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        sintheta = np.sqrt((cX - center[0]) ** 2 + (cY - center[1]) ** 2) / r_inner  # 投影角度
        costheta = np.sqrt(1 - sintheta ** 2)
        area = (area/costheta**2)

        # =====classifies on disk coronal holes=======
        p = np.zeros(s)
        cv2.drawContours(p,[contour,],-1,255,-1)
        a = np.where(p==255)[0].reshape(-1,1)
        b = np.where(p==255)[1].reshape(-1,1)
        coordinate = np.concatenate([a,b],axis=1).tolist()
        inside = [tuple(x) for x in coordinate]

        # ====create an array for magnetic polarity
        binsize = 1.

        hd_contour = [outhd[x] for x in inside]
        binnum = int((np.nanmax(hd_contour)-np.nanmin(hd_contour))//binsize + 1)
        npix,bins = np.histogram(hd_contour,bins=binnum,range=(np.nanmin(hd_contour),np.nanmax(hd_contour)))
        npix[npix==0] = 1
        magpol = bins[:-1]+binsize/2
        wh1 = magpol > 0
        # print('wh1:',sum(wh1))
        if sum(wh1) < 1 : continue
        wh2 = magpol < 0
        # print('wh2:',sum(wh2))
        if sum(wh2) < 1 : continue
        # =====magnetic cut offs dependant on area=========
        if abs((sum(npix[wh1]) - sum(npix[wh2])) / np.sqrt(sum(npix))) <= 10 and area < (9000 / (dattoarc**2)) :
            # print(abs(np.nanmean(hd_contour)))
            # print(garr[cX, cY])
            # print(area*dattoarc**2)
            continue
        if abs(np.nanmean(hd_contour)) < garr[cX, cY] and area < (40000 / (dattoarc**2)) :
            # print(abs(np.nanmean(hd_contour)))
            # print(garr[cX, cY])
            # print(area*dattoarc**2)
            continue


        # ====create an accurate center point=======
        # ====save infomation of CHs with json format=========
        Cs.append((cX, cY))
        cX2arc = (cX - center[0]) * dattoarc
        cY2arc = (cY - center[1]) * dattoarc
        CH_center = '[' + str('%.1f' % cX2arc) + ' ' + str('%.1f' % cY2arc) + ']'
        contours_large.append(contour)
        area = area * (dattoarc ** 2)  # unit: arcsec^2
        # areas.append(area)
        # mags.append(np.nanmean(hd_contour))
        contour2arc = (contour - center[0]) * dattoarc
        np.set_printoptions(threshold=np.inf)
        contour2arc = contour2arc.swapaxes(1,2)
        contour2arc = np.squeeze(contour2arc,2)
        outline = contour2arc.tolist()
        # outline = np.array2string(contour2arc)
        # outline = outline.replace('[[', '[').replace(']]', ']').replace('\n', '')
        time_obs = map_il[0].meta['date-obs'][:-3]
        CH_info = json.dumps({'id': id,
                              'time': time_obs,
                              'centroid(arcsec)': CH_center,
                              'area(arcsec^2)': str('%.4g' % area),
                              'outlines(arcsec)': outline,
                              'average B_los(G)': np.nanmean(hd_contour),},
                            sort_keys=False, indent=4, separators=(',', ': '))
        id += 1
        CHs.append(CH_info)




# cv2.namedWindow('img',0)
# cv2.drawContours(Def_grey,contours_large,-1,(125,255,255),5)
# for C in Cs:
#     cv2.circle(Def_grey, C, 20, 125, -1)
# cv2.imshow('img',Def_grey)
# print('test')
# mask_rgb = cv2.merge([msk,mak,mas])



# print(CHs)
jsonfile = os.getcwd()+\
           '\\output\\'+\
           year_str+'\\'+\
           'CH_info_'+year_str+'_'+month_str+'_'+day_str+'.json'
with open(jsonfile,'w') as f:
    for CH_info in CHs:
        f.write(CH_info)
        f.write('\n')


#====plot EUV image with CHs marked=========
tci = np.uint8(truecolorimage)
cv2.namedWindow('mask_rgb',0)#b,g,r
cv2.drawContours(tci,contours_large,-1,(125,255,255),5)
cv2.imshow('mask_rgb',tci)
###################改一下编号######################
for i in range(len(Cs)):
    (x,y) = Cs[i]
    id = i+1
    cv2.putText(tci,'CH'+str(id),(x,y),cv2.FONT_HERSHEY_SIMPLEX,3,(100,100,255),8)
time_obs = map_il[0].meta['date-obs'][:-3]
cv2.putText(tci,str(time_obs)+' CHIMERA',(200,200),cv2.FONT_HERSHEY_SIMPLEX,2,(100,100,255),2)
cv2.imwrite(os.getcwd() +
            '\\output\\'+
            year_str+'\\' +
            'CH_masked_'+year_str+'_'+month_str+'_'+day_str+'.png',tci)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# if __name__ == '__main__':
#     plt.imshow(mas*msk*mak)
#     plt.show()

# matchi,ismatch = tracking.trackCH(contours_large,contours_large[2],s)   #match的编号是从0开始的，跟图上标记的不同
print("done")