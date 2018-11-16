import matplotlib.pyplot as plt
from scipy import ndimage
import pydicom
import pydicom as dicom
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import shutil
import operator
import warnings


patchsize=31
#ds = pydicom.dcmread(filename)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def getVoxelArrays(path1,path2):
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

    image3D=np.zeros((lenX, lenY, lenZ))
    for i in range(lenZ):
        slice=np.asarray(img_zpos[i][2].pixel_array)
        #print(slice.shape)
        image3D[:,:,i]=slice

    #plt.show()
    #for i in range(len(lst)):
#        img=pydicom.dcmread(path+lst[i])
#        zpos=img.ImagePositionPatient[2]
#        print(zpos)

    struct = pydicom.dcmread(path2 + '000000.dcm')
#print(img.PatientID)
#keys=[key for key, value in img.items()]
#keys=[key for key in img]
#print(keys)

#print(ds.dir("contour"))
#print(ds.ROIContourSequence[0].Contours[0].ContourData)
#ctrs = ds.ROIContourSequence

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
        c = struct.ROIContourSequence[carinaIndices[i]]
# get contour datasets in a list
        contours = [contour for contour in c.ContourSequence]
        print(len(contours))
        contour_coord=contours[0].ContourData
# x, y, z coordinates of the contour in mm

        for i in range(0, len(contour_coord), 3):
            coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))

    voxels=[(find_nearest(xCoords, i), find_nearest(yCoords, j),find_nearest(zCoords, k)) for (i,j,k) in coord]
    print(voxels)
    x=[x for (x,y,z) in voxels]
    y=[y for (x,y,z) in voxels]
    z=[z for (x,y,z) in voxels]
    meanX=np.mean(x)
    meanY=np.mean(y)
    meanZ=np.mean(z)
    print(np.min(y))
    print(np.max(y))
    print("means: "+str(meanX)+", "+str(meanY)+", "+str(meanZ))
    plotImg=ndimage.rotate(image3D[275,:,:], 90)
    #plotImg=ndimage.rotate(image3D[:,:,91], 0)
    plotAspect=spZ/spY
    #fig, ax = plt.subplots()
    #plt.imshow(plotImg, cmap=plt.cm.bone, aspect=plotAspect)

    #ax.imshow(img)
    fig, ax = plt.subplots()

    ax.imshow(plotImg, cmap=plt.cm.bone, aspect=plotAspect, extent=[0, 512, 0, 142])
    ax.plot(meanY,meanZ,'.', linewidth=1, color='yellow')
    ax.plot([np.min(y),np.max(y)],[meanZ,meanZ],'.', linewidth=1, color='blue')
    plt.show()

    offset=int((patchsize-1)/2)
    print(offset)
    patch=image3D[int(meanX+5),int(meanY-offset)-5:int(meanY+offset)-5,int(meanZ-offset):int(meanZ+offset)]
    fig, ax = plt.subplots()
    plotPatch=ndimage.rotate(patch, 90)
    ax.imshow(plotPatch,cmap=plt.cm.bone)
    plt.show()
    return patch


#traverse data and create image patches
pathOut="Data/"
counter=0
path="../../../Data/4D-Lung/100_HM10395/"
dates=os.listdir(path)
dates=[l for l in dates if '.DS_Store' not in l]
for i in range(len(dates)):
    path2=path+dates[i]+'/'
    data=os.listdir(path2)
    print(data)
    rstructs=[d for d in data if '.' in d[-3:-1]]
    images=[d for d in data if d not in rstructs and '.DS_Store' not in d]
    print(path2 + ": Number of files: "+ str(len(images)) +" images and "+str(len(rstructs)) + " structs")
    for i in range (1):
        rstruct=[s for s in rstructs if " "+str(i*10)+".0" in s]
        img=[s for s in images if " "+str(i*10)+".0" in s]
        print(rstruct)
        print(img)
        patch=getVoxelArrays(path2+img[0]+"/",path2+rstruct[0]+"/")
        np.save(pathOut+"Carina/"+str(counter), patch)
        counter+=1


#load dataset

carinaImg = dates=os.listdir(pathOut+"Carina/")
carina=[np.load(pathOut+"Carina/"+c) for c in carinaImg]
print(len(carina))
#fig, ax = plt.subplots()
#plotPatch=ndimage.rotate(carina[0], 90)
#ax.imshow(plotPatch,cmap=plt.cm.bone)
#plt.show()
#np.load(outfile)



#getVoxelArrays("Data/TestImage/","Data/TestAnno/")

dcm_1='Data/Testimage/000080.dcm'
dcm_anno = 'Data/TestAnno/000000.dcm'







#img_contour_arrays = [coord2pixels(cdata, path) for cdata in contours]  #
#print(len(img_contour_arrays[0]))

#print(ds)
#plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
#plt.show()
#<matplotlib.image.AxesImage object at ...>
