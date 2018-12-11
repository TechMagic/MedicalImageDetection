import matplotlib.pyplot as plt
from scipy import ndimage
import pydicom
#import pydicom as dicom
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import shutil
import operator
import warnings
import random
import matplotlib.patches as patches
import scipy
import torch


patchsize=32
pathOut="Data/"
#ds = pydicom.dcmread(filename)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def importDicom(path1):
    lst=os.listdir(path1)
    img_lst=[(n,pydicom.dcmread(path1+n)) for n in lst]
    img_zpos = [(n,img.ImagePositionPatient[2],img) for (n,img) in img_lst]
    img_zpos.sort(key=lambda tup: tup[1])
    testImg=img_zpos[0][2]
    origin=testImg.ImagePositionPatient
    lenX=len(testImg.pixel_array)
    lenY=len(testImg.pixel_array[0])
    lenZ=len(img_zpos)
    spX=testImg.PixelSpacing[0]
    spY=testImg.PixelSpacing[1]
    spZ=img_zpos[1][1]-img_zpos[0][1]
    print("Image size: "+str(lenX)+ ", " +str(lenY)+ ", "+str(lenZ))
    print("Image spacing: "+str(spX)+ ", " +str(spY)+ ", "+str(spZ))

    xCoords=[origin[0]+i*spX for i in range(lenX)]
    yCoords=[origin[1]+i*spY for i in range(lenY)]
    zCoords=[origin[2]+i*spZ for i in range(lenZ)]
    print(origin)
    print(zCoords)
    spacing=[xCoords,yCoords,zCoords]
    image3D=np.zeros((lenX, lenY, lenZ))
    for i in range(lenZ):
        slice=np.asarray(img_zpos[i][2].pixel_array)
        #print(slice.shape)
        image3D[:,:,i]=slice
    return (image3D,spacing)


def printPatch(image3D,patch, spacing, coordinates, axis=0):
    ##if 0 or 1
    offset=int((patchsize-1)/2)
    if(axis==0):
        plotImg=ndimage.rotate(image3D[int(coordinates[0]),:,:], 90)
        plotAspect=(spacing[2][1]-spacing[2][0])/(spacing[1][1]-spacing[1][0])
        ax1=plt.subplot(121)

        ax1.imshow(plotImg, cmap=plt.cm.bone, aspect=plotAspect, extent=[0, 512, 0, len(spacing[2])])

        ax1.plot(coordinates[1],coordinates[2],'.', linewidth=1, color='yellow')
        # Create a Rectangle patch
        rect=patches.Rectangle((coordinates[1]-offset,coordinates[2]-offset),patchsize,patchsize,linewidth=1,edgecolor='r',facecolor='none')
        ax1.add_patch(rect)

        ax2=plt.subplot(122)
        plotPatch=ndimage.rotate(patch, 90)
        ax2.imshow(plotPatch,cmap=plt.cm.bone,aspect=plotAspect)

    else:
        if(axis==1):
            plotImg=ndimage.rotate(image3D[:,int(coordinates[1]),:], 90)
            plotAspect=(spacing[2][1]-spacing[2][0])/(spacing[1][1]-spacing[1][0])
            ax1=plt.subplot(121)

            ax1.imshow(plotImg, cmap=plt.cm.bone, aspect=plotAspect, extent=[0, 512, 0, len(spacing[2])])

            ax1.plot(coordinates[0],coordinates[2],'.', linewidth=1, color='yellow')
            # Create a Rectangle patch
            rect=patches.Rectangle((coordinates[0]-offset,coordinates[2]-offset),patchsize,patchsize,linewidth=1,edgecolor='r',facecolor='none')
            ax1.add_patch(rect)

            ax2=plt.subplot(122)
            plotPatch=ndimage.rotate(patch, 90)
            ax2.imshow(plotPatch,cmap=plt.cm.bone,aspect=plotAspect)

        else: ##axis=2
            plotImg=ndimage.rotate(image3D[:,:,int(coordinates[2])], 0)
            #plotAspect=(spacing[2][1]-spacing[2][0])/(spacing[1][1]-spacing[1][0])
            ax1=plt.subplot(121)

            ax1.imshow(plotImg, cmap=plt.cm.bone)

            ax1.plot(coordinates[1],coordinates[0],'.', linewidth=1, color='yellow')
            # Create a Rectangle patch
            rect=patches.Rectangle((coordinates[1]-offset,coordinates[0]-offset),patchsize,patchsize,linewidth=1,edgecolor='r',facecolor='none')
            ax1.add_patch(rect)

            ax2=plt.subplot(122)
            plotPatch=ndimage.rotate(patch, 0)
            ax2.imshow(plotPatch,cmap=plt.cm.bone)
    plt.show()


def importRstruct(path2, image3D, spacing, showPlot=False, axis=0):
    struct = pydicom.dcmread(path2 + '000000.dcm')
    roiNames=[c.ROIName for c in struct.StructureSetROISequence]
    print(roiNames)
    carinaNames=[x for x in roiNames if 'Carina' in x]
    carinaIndices=[roiNames.index(c) for c in carinaNames]
    print(carinaNames)
    print(carinaIndices)

#f = dicom.read_file(path + file)
# index 0 means that we are getting RTV information
    coord = []
    for i in range(len(carinaIndices)):
        c = struct.ROIContourSequence[carinaIndices[i]] #carinaIndices[i]
# get contour datasets in a list
        contours = [contour for contour in c.ContourSequence]
        print(len(contours))
        contour_coord=contours[0].ContourData
# x, y, z coordinates of the contour in mm

        for i in range(0, len(contour_coord), 3):
            coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))

    voxels=[(find_nearest(spacing[0], i), find_nearest(spacing[1], j),find_nearest(spacing[2], k)) for (i,j,k) in coord]
    print(voxels)
    y=[x for (x,y,z) in voxels]
    x=[y for (x,y,z) in voxels]
    z=[z for (x,y,z) in voxels]
    meanX=np.mean(x)
    meanY=np.mean(y)
    meanZ=np.mean(z)


    #print(np.min(y))
    #print(np.max(y))
    print("means: "+str(meanX)+", "+str(meanY)+", "+str(meanZ))

    offset=int((patchsize)/2)

    if(axis==0):
        patch=image3D[int(meanX),int(meanY-offset)-0:int(meanY+offset)-0,int(meanZ-offset):int(meanZ+offset)]
    else:
        if (axis==1):
            patch=image3D[int(meanX-offset):int(meanX+offset),int(meanY),int(meanZ-offset):int(meanZ+offset)]
        else: #axis==2
            patch=image3D[int(meanX-offset):int(meanX+offset),int(meanY-offset):int(meanY+offset),int(meanZ)]
    coordinates=[meanX,meanY,meanZ]
    if showPlot:
        printPatch(image3D,patch, spacing, coordinates,axis)
    return (coordinates,patch)


def getRandomTwoIntervals(a,b):
    r = random.choice([(-b,-a),(a,b)])
    print(r)
    rand = random.randint(*r)
    print(rand)
    return rand

def getRandomPatch(image3D, coordinates,spacing,printImage=False,axis=0):
    coordinates=[int(c) for c in coordinates]
    offset=int(patchsize/2)
    rand=[-1,-1,-1]
    s=image3D.shape
    print(s)
    while rand[2]-offset < 0 or rand[2]+offset > s[2]-1:
        rand=[getRandomTwoIntervals(offset,offset+20)+c for c in coordinates]
    print(image3D.size)
    print(rand)
    if axis==0:
        patch=image3D[rand[0],rand[1]-offset:rand[1]+offset,rand[2]-offset:rand[2]+offset]
    else:
        if axis==1:
            patch=image3D[rand[0]-offset:rand[0]+offset,rand[1],rand[2]-offset:rand[2]+offset]
        else:
            patch=image3D[rand[0]-offset:rand[0]+offset,rand[1]-offset:rand[1]+offset,rand[2]]
    if printImage:
        printPatch(image3D, patch, spacing, rand)
    print(coordinates)
    print(rand)
    return (coordinates,patch)

def getLungPatch(image3D, coordinates,spacing):
    offset=int(patchsize/2)
    offsetY=-80
    offsetZ=-10
    coordinates[1]=int(offsetY+coordinates[1])
    coordinates[2]=int(offsetZ+coordinates[2])
    patch=image3D[int(coordinates[0]),coordinates[1]-offset:coordinates[1]+offset,coordinates[2]-offset:coordinates[2]+offset]
    return coordinates,patch

def normalizeDicom(img):
    img=(img/4000)*255
    img=np.uint8(img)
    return img

def createDataset(writeToFile=False, axis=0):
    #traverse data and create image patches

    counter=0
    path="../../../Data/4D-Lung/"
    patients=os.listdir(path)
    patients=[p for p in patients if '.DS_Store' not in p]
    print(patients)
    for i in range(1): #" "+str(i*10)+".0"
        name=str(119+i)
        path1=path+name+"_HM10395/" #patients[i]+'/'
        dates=os.listdir(path1)
        dates=[l for l in dates if '.DS_Store' not in l]
        for j in range(len(dates)):
            path2=path1+dates[j]+'/'
            data=os.listdir(path2)
            #print(data)
            rstructs=[d for d in data if '.' in d[-3:-1]]
            images=[d for d in data if d not in rstructs and '.DS_Store' not in d]
            print(rstructs)
            print(images)
            print(path2 + ": Number of files: "+ str(len(images)) +" images and "+str(len(rstructs)) + " structs")
            for k in range (1): #len(images)
                rstruct=[s for s in rstructs if " "+str(k*10)+".0" in s]
                img=[s for s in images if " "+str(k*10)+".0" in s]


                print(path2+img[0]+"/")

                (image3D,spacing)=importDicom(path2+img[0]+"/")
                (coordinates,patch)=importRstruct(path2+rstruct[0]+"/",image3D,spacing,False, axis)
                print(coordinates)
                counter+=1

                (coordinatesL,patchL)=getLungPatch(image3D,coordinates,spacing)
                #printPatch(image3D,patchL, spacing, coordinatesL)
                (coordinatesR,patchR)=getRandomPatch(image3D, coordinates,spacing,False,axis)
                if writeToFile:
                    foldername = " "
                    if axis== 0:
                        foldername = "Sagittal"
                    else:
                        if axis == 1:
                            foldername = "Coronal"
                        else:
                            foldername = "Axial"
                    #np.save(pathOut+"Carina/"+name+"_"+str(k), patch)
                    #np.save(pathOut+"noCarina/"+name+"_"+str(k), patchR)
                    scipy.misc.imsave(pathOut+"noCarina" +foldername +"/"+name+"_"+str(k)+".png", normalizeDicom(patchR))
                    scipy.misc.imsave(pathOut+"carina"+foldername+"/"+name+"_"+str(k)+".png", normalizeDicom(patch))
                    #scipy.misc.imsave(pathOut+"Lung/"+name+"_"+str(k)+".png", normalizeDicom(patchL))
                #imgTest=scipy.misc.imread(pathOut+"noCarina/"+name+"_test.png")
                #print("patch size: "+ str(patch.shape))
                #print("patchR size: "+ str(patchR.shape))
                #print('max: '+str(image3D.max())) #4000
                #print('min: '+str(image3D.min())) #0

    print("Counter: "+str(counter))

createDataset(True, axis=1)


