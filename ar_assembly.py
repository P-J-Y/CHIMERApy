#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sunpy
from sunpy.net import Fido, attrs as a
import glob
import numpy as np
import sunpy.map
from aiapy.calibrate import normalize_exposure, register, update_pointing
import astropy.units as u
import astroscrappy
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import copy
from astropy.coordinates import SkyCoord
import skimage
import cv2
from skimage.segmentation import flood, flood_fill
from scipy import interpolate
from skimage import morphology,measure,feature
from skimage.morphology import dilation,erosion,closing,opening,square
import time
from sunpy.physics.differential_rotation import differential_rotate
from sunpy.coordinates import RotatedSunFrame
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# In[12]:


# connect the primarily chosen points based on their neighbor points and produce the mask before subsequent dilation and closing 
def ar_grow(data,seeds,mask,width,ratio):
    mask_out=np.zeros_like(mask)
    mask_seeds=np.zeros_like(mask)
    mask_seeds[seeds[0],seeds[1]]=1
    directx = np.arange(0,width,1)- width//2
    directy = np.arange(0,width,1)- width//2
    xx,yy=np.meshgrid(directx,directy)
    directs=np.transpose([np.ravel(xx),np.ravel(yy)])
    mask_shifted=np.zeros_like(mask_out)
    for direct in directs:
        mask_shifted=mask_shifted+np.roll(mask,(direct[0],direct[1]),axis=(0,1))
    mask_seeds_o=np.zeros_like(mask_seeds)
    while np.sum(mask_seeds):
        mask_out=np.where(mask_shifted[:,:]*mask_seeds[:,:]>ratio*width*width,1,mask_out)
        mask_seeds_o=np.where(mask_seeds == 1,0,1)
        mask_seeds=np.where(mask_shifted[:,:]*mask_seeds[:,:]>ratio*width*width,1,0)
        mask_seeds=np.where(mask_seeds_o == 0, 0, mask_seeds) 
    return mask_out


# In[13]:


# rebin an array to a lower dimension by averaging neighbor points
def rebin(a, shape):
    if len(np.shape(a)) == 1:
        return a.reshape((shape,a.shape[0]//shape)).mean(1)
    else:
        sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
        return a.reshape(sh).mean(-1).mean(1)


# In[47]:


def ar_detect(fn,mask_folder,png_folder,detect_fn):
    hmi_map_origin = register(sunpy.map.Map(fn))
    fudge=np.sin(80.0/180.0*np.pi)
    rsun=hmi_map_origin.rsun_obs
    threshold=70.0 # the maximum value of the quiscent field according to Higgins et al. 2011
    smooth_width=8.0*1e6*u.m/hmi_map_origin.rsun_meters*rsun # sigma of the Gaussian filter, G=A*exp(-x^2/(2sigma^2))

    # loading the map and create the coordinates
    hmi_map=copy.deepcopy(hmi_map_origin)
    ss=hmi_map.data.shape
    xcoor=(np.arange(0,ss[1])-hmi_map.reference_pixel[1]/u.pix)*hmi_map.scale[1]*u.pix
    ycoor=(np.arange(0,ss[0])-hmi_map.reference_pixel[0]/u.pix)*hmi_map.scale[0]*u.pix
    xcoors,ycoors=np.meshgrid(xcoor,ycoor)

    # plot the original map
    fig = plt.figure(figsize=(24, 12))
    ax1=plt.subplot(121,projection=hmi_map_origin)
    hmi_map_origin.plot(axes=ax1,title='Original Map '+str(hmi_map.date),vmin=-100,vmax=100)

    # do the cosmap correction
    rr=(xcoors**2.0+ycoors**2.0)**0.5
    thetas=np.arcsin(np.where(rr <= rsun, rr, rsun)/rsun)
    data=np.zeros_like(hmi_map.data)
    data=np.where(rr > rsun*fudge,hmi_map.data,hmi_map.data/np.cos(thetas))
    data=np.where(rr <= rsun,hmi_map.data,0.0)
    hmi_map.data[:,:]=data
    hmi_map_coscor=copy.deepcopy(hmi_map_origin)
    hmi_map_coscor.data[:,:]=np.where(rr <= rsun,hmi_map.data,np.nan)

    # Gaussian smooth
    pix_width=np.floor(smooth_width/(hmi_map.scale[1]*u.pix))
    data = ndi.gaussian_filter(hmi_map.data, sigma=(pix_width,pix_width), order=0)
    data=np.where(rr <= rsun,data,np.nan)
    hmi_map.data[:,:]=copy.deepcopy(data)

    # calculate the threshold for selecting the priliminary AR seeds
    data_ondisk=copy.deepcopy(data[np.where(rr <= rsun)])
    qs_seeds=np.where(abs(data_ondisk) <= threshold)
    qs_ave_level=np.mean(abs(data_ondisk[qs_seeds]))
    qs_level_std=np.std(abs(data_ondisk[qs_seeds]))

    #downsample to (1024,1024) for a higher efficiency
    rebin_index=4
    data=np.where(rr <= rsun,data,0.0)
    data_rebin=rebin(data,[int(4096/rebin_index),int(4096/rebin_index)])
    mask_rebin=np.zeros_like(data_rebin)
    seeds_rebin=np.where(abs(data_rebin) >= qs_ave_level)
    mask_rebin=np.where(abs(data_rebin) < (qs_ave_level+3.0*qs_level_std), mask_rebin,1.0)

    #region grow
    dila_width=np.int(np.ceil(pix_width/rebin_index)*2.0+1.0)
    start=time.process_time()
    mask_rebin=ar_grow(np.absolute(data_rebin),seeds_rebin,mask_rebin,dila_width,0.7)
    end=time.process_time()
    regions=measure.regionprops(morphology.label(mask_rebin,connectivity=2))
    
    # eliminate bad points
    for region in regions:
        if region.area <= dila_width*dila_width:
            mask_rebin[region.coords[:,0],region.coords[:,1]]=0.0
            
    if np.max(mask_rebin) != 0:
        #dilation and closing to connect the small regions
        mask_dilation_rebin=np.zeros_like(mask_rebin)
        mask_closed_rebin=np.zeros_like(mask_rebin)
        mask_dilation_rebin = morphology.dilation(mask_rebin,square(dila_width))
        mask_closed_rebin = closing(mask_dilation_rebin, morphology.disk(int(dila_width//2)))

        #upsample to the original size
        f_mask_upsample=interpolate.interp2d(rebin(xcoor,int(4096/rebin_index))/u.arcsec,rebin(ycoor,int(4096/rebin_index))/u.arcsec,mask_closed_rebin)
        mask_upsample=np.zeros_like(hmi_map.data)
        mask_upsample=f_mask_upsample(xcoor,ycoor)
        mask_upsample=np.where(mask_upsample < 0.7,0.0,1.0)
        
        #fill small holes inside each contour
        mask_edges=feature.canny(mask_upsample)
        mask_upsample=ndi.binary_fill_holes(mask_edges)

        #eliminate small regions and regions with only one kind of polarity
        mask_upsample=np.where(rr < rsun,mask_upsample,0.0)
        regions=measure.regionprops(morphology.label(mask_upsample,connectivity=2))
        for region in regions:
            num_p=np.size(np.where(hmi_map.data[region.coords[:,0],region.coords[:,1]] > 0.0))
            num_n=np.size(np.where(hmi_map.data[region.coords[:,0],region.coords[:,1]] < 0.0))
            if region.area <= (2*pix_width)*(2*pix_width):
                mask_upsample[region.coords[:,0],region.coords[:,1]]=0.0
            if num_p/(num_p+num_n)<0.05 or num_p/(num_p+num_n)>0.95:
                mask_upsample[region.coords[:,0],region.coords[:,1]]=0.0
    else:
        mask_upsample=np.zeros_like(hmi_map.data)
        
    # align contours and regions
    pmask = np.array(mask_upsample,int)
    pmask = morphology.label(pmask,connectivity=2)
    pmask=np.where(rr < rsun,pmask,0.0)
    contours=measure.find_contours(pmask,0)
    regions=measure.regionprops(np.array(pmask,int))
    regions_contours=[]
    for region in regions:
        for contour in contours:
            c = np.expand_dims(contour.astype(np.float32), 1)
            dist = cv2.pointPolygonTest(c,(float(region.coords[0,0]),float(region.coords[0,1])),False)
            if dist >=0:
                regions_contours.append(contour)
                break
    
    # display the results
    ax4=plt.subplot(122,projection=hmi_map_coscor)
    hmi_map_coscor.plot(axes=ax4,title='AR detection results',vmin=-100,vmax=100)          
    for contour,region in zip(regions_contours,regions):
        area=region.filled_area*u.pix*u.pix*hmi_map.scale[0]*hmi_map.scale[1]/(np.pi*hmi_map_origin.rsun_obs**2)
        ax4.plot(contour[:,1],contour[:,0],linewidth=2,label='AR %s'%region.label+' area: %4.3f'%area)
        ax4.text(region.centroid[1],region.centroid[0],'AR %s'%region.label,fontsize='xx-large',color='r')
#         ax4.text(100.0,100+region.label*100.0,('AR %s area:'%region.label)+
#                  (' %s'%(region.filled_area*u.pix*u.pix*hmi_map.scale[0]*hmi_map.scale[1]/(np.pi*hmi_map_origin.rsun_obs**2))))
    plt.legend()
    plt.savefig(png_folder+'/'+detect_fn+'.png',dpi=150)
    plt.show()
    
    with open(mask_folder+'/'+detect_fn+'.npy', 'wb') as f:
         np.save(f, pmask)


# In[29]:


def ar_trace_convex(fns,mask_fns,mask_updated_convex_folder,png_updated_convex_folder,updated_convex_fns):
    color_list=['light blue','teal','orange','light green','magenta','yellow',
            'light purple','tan','mauve','royal blue','olive','light pink',
            'forest green','gold','bright pink','pale orange','pastel blue',
            'scarlet','tangerine','raspberry','emerald','very light blue','lemon yellow']
    for loop in range(len(fns)-1):
        mask_updated_fns=sorted(glob.glob(mask_updated_folder+'/*.npy'))
        mask_fn1=mask_fns[loop]
        fn1=fns[loop]
        mask_fn2=mask_fns[loop+1]
        fn2=fns[loop+1]
        hmi_map_origin1 = register(sunpy.map.Map(fn1))
        if loop ==0:
            with open(mask_fn1, 'rb') as f:
                mask1=np.load(f)
            arnum=np.max(mask1)
        else:
            with open(mask_updated_fns[loop], 'rb') as f:
                mask1=np.load(f)
        hmi_map_origin2 = register(sunpy.map.Map(fn2))
        with open(mask_fn2, 'rb') as f:
            mask2=np.load(f)
        rsun=hmi_map_origin2.rsun_obs
        smooth_width=8.0*1e6*u.m/hmi_map_origin2.rsun_meters*rsun
        pix_width=np.floor(smooth_width/(hmi_map_origin2.scale[1]*u.pix))

        regions1=measure.regionprops(np.array(mask1,dtype=int))
        regions2=measure.regionprops(np.array(mask2,dtype=int))
        points=[]
        transformed_points=[]
        diffrot_points=[]
        transformed_diffrot_points=[]
        fig=plt.figure(figsize=(24,12))
        mask1_convex=np.zeros_like(mask1)
        for region in regions1:
            xc=(region.centroid[1]*u.pix-hmi_map_origin1.reference_pixel[1])*hmi_map_origin1.scale[1]
            yc=(region.centroid[0]*u.pix-hmi_map_origin1.reference_pixel[0])*hmi_map_origin1.scale[0]
            point = SkyCoord(xc, yc, frame=hmi_map_origin1.coordinate_frame)
            transformed_point = SkyCoord(xc, yc, frame=hmi_map_origin2.coordinate_frame)
            diffrot_point = RotatedSunFrame(base=point, rotated_time=hmi_map_origin2.date)
            transformed_diffrot_point = diffrot_point.transform_to(hmi_map_origin2.coordinate_frame)
            points.append(point)
            diffrot_points.append(diffrot_point)
            transformed_diffrot_points.append(transformed_diffrot_point)
            transformed_points.append(transformed_point)
            bbox=region.bbox
            convex_image=np.array(region.convex_image,dtype=int)
            mask1_convex[bbox[0]:bbox[2],bbox[1]:bbox[3]]=convex_image*(np.zeros_like(convex_image)+region.label)

        target_points=[]
        mask2_convex=np.zeros_like(mask2)
        for region in regions2:
            xc2=(region.centroid[1]*u.pix-hmi_map_origin2.reference_pixel[1])*hmi_map_origin2.scale[1]
            yc2=(region.centroid[0]*u.pix-hmi_map_origin2.reference_pixel[0])*hmi_map_origin2.scale[0]
            point2=SkyCoord(xc2, yc2, frame=hmi_map_origin2.coordinate_frame)
            target_points.append(point2)
            bbox=region.bbox
            convex_image=np.array(region.convex_image,dtype=int)
            mask2_convex[bbox[0]:bbox[2],bbox[1]:bbox[3]]=convex_image*(np.zeros_like(convex_image)+region.label)

        target_txs=np.array([])
        target_tys=np.array([])
        for target_point in target_points:
            target_txs=np.append(target_txs,target_point.Tx/u.arcsec)
            target_tys=np.append(target_tys,target_point.Ty/u.arcsec)

        visited=np.zeros(len(regions2))

        mask2_updated=np.zeros_like(mask2)
        contours2=measure.find_contours(mask2,0)
        contours2_convex=measure.find_contours(mask2_convex,0)
        regions2_contours2=[]
        regions2_contours2_convex=[]
        for region in regions2:
            for contour in contours2:
                c = np.expand_dims(contour.astype(np.float32), 1)
                dist = cv2.pointPolygonTest(c,(float(region.coords[0,0]),float(region.coords[0,1])),False)
                if dist >=0:
                    regions2_contours2.append(contour)
                    break
            for contour in contours2_convex:
                c = np.expand_dims(contour.astype(np.float32), 1)
                dist = cv2.pointPolygonTest(c,(float(region.coords[0,0]),float(region.coords[0,1])),False)
                if dist >=0:
                    regions2_contours2_convex.append(contour)
                    break
        for i in range(len(transformed_diffrot_points)):
            tx=np.floor((transformed_diffrot_points[i].Tx/hmi_map_origin1.scale[1]+hmi_map_origin1.reference_pixel[1])/u.pix)
            ty=np.floor((transformed_diffrot_points[i].Ty/hmi_map_origin1.scale[0]+hmi_map_origin1.reference_pixel[0])/u.pix)
            txo=np.floor((points[i].Tx/hmi_map_origin1.scale[1]+hmi_map_origin1.reference_pixel[1])/u.pix)
            tyo=np.floor((points[i].Ty/hmi_map_origin1.scale[0]+hmi_map_origin1.reference_pixel[0])/u.pix)
            for j in range(len(regions2_contours2_convex)):
                c = np.expand_dims(regions2_contours2_convex[j].astype(np.float32), 1)
                dist = cv2.pointPolygonTest(c,(int(ty),int(tx)),False)
                if dist >=0:
                    mask2_updated[regions2[j].coords[:,0],regions2[j].coords[:,1]]=mask1_convex[int(tyo),int(txo)]
                    visited[j]=1
                    break

        if np.sum(visited) < len(regions2):
            non_visited_inds=np.where(visited == 0)
            for i in range(np.size(non_visited_inds)):
                mask2_updated[regions2[non_visited_inds[0][i]].coords[:,0],regions2[non_visited_inds[0][i]].coords[:,1]]=arnum+1
                arnum=arnum+1

        regions1=measure.regionprops(np.array(mask1,dtype=int))
        regions2_updated=measure.regionprops(np.array(mask2_updated,dtype=int))
        regions2_contours2_updated=[]
        contours2_updated=measure.find_contours(mask2_updated,0)
        for region in regions2_updated:
            for contour in contours2_updated:
                c = np.expand_dims(contour.astype(np.float32), 1)
                dist = cv2.pointPolygonTest(c,(float(region.coords[0,0]),float(region.coords[0,1])),True)
                if dist >=0:
                    regions2_contours2_updated.append(contour)
                    break


        mask2_updated_convex=np.zeros_like(mask2_updated)
        for region in regions2_updated:
            bbox=region.bbox
            convex_image=np.array(region.convex_image,dtype=int)
            mask2_updated_convex[bbox[0]:bbox[2],bbox[1]:bbox[3]]=convex_image*(np.zeros_like(convex_image)+region.label)
        regions2_updated_convex=measure.regionprops(np.array(mask2_updated_convex,dtype=int))
        regions2_contours2_updated_convex=[]
        contours2_updated_convex=measure.find_contours(mask2_updated_convex,0)
        for region in regions2_updated_convex:
            for contour in contours2_updated_convex:
                c = np.expand_dims(contour.astype(np.float32), 1)
                dist = cv2.pointPolygonTest(c,(float(region.coords[0,0]),float(region.coords[0,1])),True)
                if dist >=0:
                    regions2_contours2_updated_convex.append(contour)
                    break
                
        if loop==0:
            mask_fn=mask_updated_convex_folder+'/'+updated_convex_fns[loop]+'_traced.npy'
            with open(mask_fn, 'wb') as f:
                 np.save(f, mask1_convex)
            fig=plt.figure(figsize=(12,12))
            ax4=plt.subplot(111,projection=hmi_map_origin1)
            hmi_map_origin1.plot(axes=ax4,vmin=-100,vmax=100)
            contours1=measure.find_contours(mask1,0)
            contours1_convex=measure.find_contours(mask1_convex,0)
            regions1_contours1=[]
            regions1_contours1_convex=[]
            for region in regions1:
                for contour in contours1:
                    c = np.expand_dims(contour.astype(np.float32), 1)
                    dist = cv2.pointPolygonTest(c,(float(region.coords[0,0]),float(region.coords[0,1])),False)
                    if dist >=0:
                        regions1_contours1.append(contour)
                        break
                for contour in contours1_convex:
                    c = np.expand_dims(contour.astype(np.float32), 1)
                    dist = cv2.pointPolygonTest(c,(float(region.coords[0,0]),float(region.coords[0,1])),False)
                    if dist >=0:
                        regions1_contours1_convex.append(contour)
                        break
            for region,contour in zip(regions1,regions1_contours1_convex):
                area=region.convex_area*u.pix*u.pix*hmi_map_origin2.scale[0]*hmi_map_origin2.scale[1]/(np.pi*hmi_map_origin2.rsun_obs**2)
                ax4.plot(contour[:,1],contour[:,0],linewidth=2,c=mcolors.XKCD_COLORS['xkcd:'+color_list[int(region.label % len(color_list))]],
                        label='AR %s'%region.label+' area: %4.3f'%area)
                ax4.text(region.centroid[1],region.centroid[0],'AR %s'%region.label,fontsize='xx-large',c='r')
            pngname=png_updated_convex_folder+'/'+updated_convex_fns[loop]+'_traced.png'
            plt.legend()
            plt.savefig(pngname,dpi=150)
            plt.show()
            
        mask_updated_fn=mask_updated_convex_folder+'/'+updated_convex_fns[loop+1]+'_traced.npy'
        with open(mask_updated_fn, 'wb') as f:
             np.save(f, mask2_updated_convex)

        fig=plt.figure(figsize=(12,12))
        ax4=plt.subplot(111,projection=hmi_map_origin2)
        hmi_map_origin2.plot(axes=ax4,vmin=-100,vmax=100)      
        for region,contour in zip(regions2_updated,regions2_contours2_updated_convex):
            area=region.convex_area*u.pix*u.pix*hmi_map_origin2.scale[0]*hmi_map_origin2.scale[1]/(np.pi*hmi_map_origin2.rsun_obs**2)
            ax4.plot(contour[:,1],contour[:,0],linewidth=2,c=mcolors.XKCD_COLORS['xkcd:'+color_list[int(region.label % len(color_list))]],
                    label='AR %s'%region.label+' area: %4.3f'%area)
            ax4.text(region.centroid[1],region.centroid[0],'AR %s'%region.label,fontsize='xx-large',c='r')

        pngname=png_updated_convex_folder+'/'+updated_convex_fns[loop+1]+'_traced.png'
        plt.legend()
        plt.savefig(pngname,dpi=150)
        plt.show()


# In[28]:


def ar_trace(fns,mask_fns,mask_updated_folder,png_updated_folder,updated_fns):
    color_list=['light blue','teal','orange','light green','magenta','yellow',
            'light purple','tan','mauve','royal blue','olive','light pink',
            'forest green','gold','bright pink','pale orange','pastel blue',
            'scarlet','tangerine','raspberry','emerald','very light blue','lemon yellow']
    for loop in range(len(fns)-1):
        mask_updated_fns=sorted(glob.glob(mask_updated_folder+'/*.npy'))
        mask_fn1=mask_fns[loop]
        fn1=fns[loop]
        mask_fn2=mask_fns[loop+1]
        fn2=fns[loop+1]
        hmi_map_origin1 = register(sunpy.map.Map(fn1))
        if loop ==0:
            with open(mask_fn1, 'rb') as f:
                mask1=np.load(f)
            arnum=np.max(mask1)
        else:
            with open(mask_updated_fns[loop], 'rb') as f:
                mask1=np.load(f)
        hmi_map_origin2 = register(sunpy.map.Map(fn2))
        with open(mask_fn2, 'rb') as f:
            mask2=np.load(f)
        rsun=hmi_map_origin2.rsun_obs
        smooth_width=8.0*1e6*u.m/hmi_map_origin2.rsun_meters*rsun
        pix_width=np.floor(smooth_width/(hmi_map_origin2.scale[1]*u.pix))

        regions1=measure.regionprops(np.array(mask1,dtype=int))
        regions2=measure.regionprops(np.array(mask2,dtype=int))
        points=[]
        transformed_points=[]
        diffrot_points=[]
        transformed_diffrot_points=[]
        fig=plt.figure(figsize=(24,12))
        mask1_convex=np.zeros_like(mask1)
        for region in regions1:
            xc=(region.centroid[1]*u.pix-hmi_map_origin1.reference_pixel[1])*hmi_map_origin1.scale[1]
            yc=(region.centroid[0]*u.pix-hmi_map_origin1.reference_pixel[0])*hmi_map_origin1.scale[0]
            point = SkyCoord(xc, yc, frame=hmi_map_origin1.coordinate_frame)
            transformed_point = SkyCoord(xc, yc, frame=hmi_map_origin2.coordinate_frame)
            diffrot_point = RotatedSunFrame(base=point, rotated_time=hmi_map_origin2.date)
            transformed_diffrot_point = diffrot_point.transform_to(hmi_map_origin2.coordinate_frame)
            points.append(point)
            diffrot_points.append(diffrot_point)
            transformed_diffrot_points.append(transformed_diffrot_point)
            transformed_points.append(transformed_point)
            bbox=region.bbox
            convex_image=np.array(region.convex_image,dtype=int)
            mask1_convex[bbox[0]:bbox[2],bbox[1]:bbox[3]]=convex_image*(np.zeros_like(convex_image)+region.label)

        target_points=[]
        mask2_convex=np.zeros_like(mask2)
        for region in regions2:
            xc2=(region.centroid[1]*u.pix-hmi_map_origin2.reference_pixel[1])*hmi_map_origin2.scale[1]
            yc2=(region.centroid[0]*u.pix-hmi_map_origin2.reference_pixel[0])*hmi_map_origin2.scale[0]
            point2=SkyCoord(xc2, yc2, frame=hmi_map_origin2.coordinate_frame)
            target_points.append(point2)
            bbox=region.bbox
            convex_image=np.array(region.convex_image,dtype=int)
            mask2_convex[bbox[0]:bbox[2],bbox[1]:bbox[3]]=convex_image*(np.zeros_like(convex_image)+region.label)

        target_txs=np.array([])
        target_tys=np.array([])
        for target_point in target_points:
            target_txs=np.append(target_txs,target_point.Tx/u.arcsec)
            target_tys=np.append(target_tys,target_point.Ty/u.arcsec)

        visited=np.zeros(len(regions2))

        mask2_updated=np.zeros_like(mask2)
        contours2=measure.find_contours(mask2,0)
        contours2_convex=measure.find_contours(mask2_convex,0)
        regions2_contours2=[]
        regions2_contours2_convex=[]
        for region in regions2:
            for contour in contours2:
                c = np.expand_dims(contour.astype(np.float32), 1)
                dist = cv2.pointPolygonTest(c,(float(region.coords[0,0]),float(region.coords[0,1])),False)
                if dist >=0:
                    regions2_contours2.append(contour)
                    break
            for contour in contours2_convex:
                c = np.expand_dims(contour.astype(np.float32), 1)
                dist = cv2.pointPolygonTest(c,(float(region.coords[0,0]),float(region.coords[0,1])),False)
                if dist >=0:
                    regions2_contours2_convex.append(contour)
                    break
        for i in range(len(transformed_diffrot_points)):
            tx=np.floor((transformed_diffrot_points[i].Tx/hmi_map_origin1.scale[1]+hmi_map_origin1.reference_pixel[1])/u.pix)
            ty=np.floor((transformed_diffrot_points[i].Ty/hmi_map_origin1.scale[0]+hmi_map_origin1.reference_pixel[0])/u.pix)
            txo=np.floor((points[i].Tx/hmi_map_origin1.scale[1]+hmi_map_origin1.reference_pixel[1])/u.pix)
            tyo=np.floor((points[i].Ty/hmi_map_origin1.scale[0]+hmi_map_origin1.reference_pixel[0])/u.pix)
            for j in range(len(regions2_contours2_convex)):
                c = np.expand_dims(regions2_contours2_convex[j].astype(np.float32), 1)
                dist = cv2.pointPolygonTest(c,(int(ty),int(tx)),False)
                if dist >=0:
                    mask2_updated[regions2[j].coords[:,0],regions2[j].coords[:,1]]=mask1_convex[int(tyo),int(txo)]
                    visited[j]=1
                    break

        if np.sum(visited) < len(regions2):
            non_visited_inds=np.where(visited == 0)
            for i in range(np.size(non_visited_inds)):
                mask2_updated[regions2[non_visited_inds[0][i]].coords[:,0],regions2[non_visited_inds[0][i]].coords[:,1]]=arnum+1
                arnum=arnum+1

        regions1=measure.regionprops(np.array(mask1,dtype=int))
        regions2_updated=measure.regionprops(np.array(mask2_updated,dtype=int))
        regions2_contours2_updated=[]
        contours2_updated=measure.find_contours(mask2_updated,0)
        for region in regions2_updated:
            for contour in contours2_updated:
                c = np.expand_dims(contour.astype(np.float32), 1)
                dist = cv2.pointPolygonTest(c,(float(region.coords[0,0]),float(region.coords[0,1])),True)
                if dist >=0:
                    regions2_contours2_updated.append(contour)
                    break


        mask2_updated_convex=np.zeros_like(mask2_updated)
        for region in regions2_updated:
            bbox=region.bbox
            convex_image=np.array(region.convex_image,dtype=int)
            mask2_updated_convex[bbox[0]:bbox[2],bbox[1]:bbox[3]]=convex_image*(np.zeros_like(convex_image)+region.label)
        regions2_updated_convex=measure.regionprops(np.array(mask2_updated_convex,dtype=int))
        regions2_contours2_updated_convex=[]
        contours2_updated_convex=measure.find_contours(mask2_updated_convex,0)
        for region in regions2_updated_convex:
            for contour in contours2_updated_convex:
                c = np.expand_dims(contour.astype(np.float32), 1)
                dist = cv2.pointPolygonTest(c,(float(region.coords[0,0]),float(region.coords[0,1])),True)
                if dist >=0:
                    regions2_contours2_updated_convex.append(contour)
                    break
                
        if loop==0:
            mask_fn=mask_updated_folder+'/'+updated_fns[loop]+'_traced.npy'
            with open(mask_fn, 'wb') as f:
                 np.save(f, mask1)
            fig=plt.figure(figsize=(12,12))
            ax4=plt.subplot(111,projection=hmi_map_origin1)
            hmi_map_origin1.plot(axes=ax4,vmin=-100,vmax=100)
            contours1=measure.find_contours(mask1,0)
            contours1_convex=measure.find_contours(mask1_convex,0)
            regions1_contours1=[]
            regions1_contours1_convex=[]
            for region in regions1:
                for contour in contours1:
                    c = np.expand_dims(contour.astype(np.float32), 1)
                    dist = cv2.pointPolygonTest(c,(float(region.coords[0,0]),float(region.coords[0,1])),False)
                    if dist >=0:
                        regions1_contours1.append(contour)
                        break
                for contour in contours1_convex:
                    c = np.expand_dims(contour.astype(np.float32), 1)
                    dist = cv2.pointPolygonTest(c,(float(region.coords[0,0]),float(region.coords[0,1])),False)
                    if dist >=0:
                        regions1_contours1_convex.append(contour)
                        break
            for region,contour in zip(regions1,regions1_contours1):
                area=region.filled_area*u.pix*u.pix*hmi_map_origin2.scale[0]*hmi_map_origin2.scale[1]/(np.pi*hmi_map_origin2.rsun_obs**2)
                ax4.plot(contour[:,1],contour[:,0],linewidth=2,c=mcolors.XKCD_COLORS['xkcd:'+color_list[int(region.label % len(color_list))]],
                        label='AR %s'%region.label+' area: %4.3f'%area)
                ax4.text(region.centroid[1],region.centroid[0],'AR %s'%region.label,fontsize='xx-large',c='r')
            pngname=png_updated_folder+'/'+updated_fns[loop]+'_traced.png'
            plt.legend()
            plt.savefig(pngname,dpi=150)
            plt.show()
            
        mask_updated_fn=mask_updated_folder+'/'+updated_fns[loop+1]+'_traced.npy'
        with open(mask_updated_fn, 'wb') as f:
             np.save(f, mask2_updated)

        fig=plt.figure(figsize=(12,12))
        ax4=plt.subplot(111,projection=hmi_map_origin2)
        hmi_map_origin2.plot(axes=ax4,vmin=-100,vmax=100)      
        for region,contour in zip(regions2_updated,regions2_contours2_updated):
            area=region.filled_area*u.pix*u.pix*hmi_map_origin2.scale[0]*hmi_map_origin2.scale[1]/(np.pi*hmi_map_origin2.rsun_obs**2)
            ax4.plot(contour[:,1],contour[:,0],linewidth=2,c=mcolors.XKCD_COLORS['xkcd:'+color_list[int(region.label % len(color_list))]],
                    label='AR %s'%region.label+' area: %4.3f'%area)
            ax4.text(region.centroid[1],region.centroid[0],'AR %s'%region.label,fontsize='xx-large',c='r')

        pngname=png_updated_folder+'/'+updated_fns[loop+1]+'_traced.png'
        plt.legend()
        plt.savefig(pngname,dpi=150)
        plt.show()


# In[61]:


folders=['./python_version/data1','./python_version/data2','./python_version/data3',
        './python_version/data4','./python_version/data5','./python_version/data6']
for folder in folders:
    start=time.process_time()
    fns=sorted(glob.glob(folder+'/*.fits'))
    for fn in fns:
        ar_detect(fn,folder+'/mask',folder+'/png',fn[-38:-23])

    mask_fns=sorted(glob.glob(folder+'/mask/*.npy'))
    updated_fns=[]
    for mask_fn in mask_fns:
        updated_fns.append(mask_fn[-19:-4])
    mask_updated_folder=folder+'/mask_trace'
    png_updated_folder=folder+'/png_trace'
    ar_trace(fns,mask_fns,mask_updated_folder,png_updated_folder,updated_fns)

    mask_updated_convex_folder=folder+'/mask_trace_convex'
    png_updated_convex_folder=folder+'/png_trace_convex'
    ar_trace_convex(fns,mask_fns,mask_updated_convex_folder,png_updated_convex_folder,updated_fns)
    end=time.process_time()
    print('loop time: %s seconds'%(end-start))


# In[ ]:




