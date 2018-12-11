import torch
import torchvision
from torchvision import transforms, datasets
import os
#import scipy
from skimage import io
import random
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
#import testDicom
import pydicom
from scipy import ndimage

axisNames=['Sagittal','Coronal','Axial']
pathOut='Data/'
classes = ('Carina','noCarina')
epochs=40
patchsize=32

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2) ###2 classes

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
    #print("Image size: "+str(lenX)+ ", " +str(lenY)+ ", "+str(lenZ))
    #print("Image spacing: "+str(spX)+ ", " +str(spY)+ ", "+str(spZ))

    xCoords=[origin[0]+i*spX for i in range(lenX)]
    yCoords=[origin[1]+i*spY for i in range(lenY)]
    zCoords=[origin[2]+i*spZ for i in range(lenZ)]
    #print(origin)
    #print(zCoords)
    spacing=[xCoords,yCoords,zCoords]
    image3D=np.zeros((lenX, lenY, lenZ))
    for i in range(lenZ):
        slice=np.asarray(img_zpos[i][2].pixel_array)
        #print(slice.shape)
        image3D[:,:,i]=slice
    return (image3D,spacing)

def visualizeDetection(img):
    fig, ax = plt.subplots()
    im = ax.imshow(img,cmap='gist_gray') #'inferno'
    plt.show()

def trainNetwork(net, train_samples):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(epochs):  # loop over the dataset multiple times
        print("epoch: "+str(epoch))
        running_loss = 0.0
        for  i in range(len(train_samples)):
            # get the inputs
            input, label = train_samples[i]
            #print(input.shape)
            input=input.unsqueeze(0)
            input=input.unsqueeze(0)
            input = input.type('torch.FloatTensor')
            #print(input)
            label=torch.tensor(label)
            label=label.unsqueeze(0)
            label = label.type('torch.LongTensor')
            #print(label)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(input)
            #print(outputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # print statistics

            running_loss += loss.item()
            #print('[%d, %5d] loss: %.3f' %
            #(epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print('Finished Training')
    return net

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

def importPatches(path,axis,test_subject):
    samples0=os.listdir(path+"Carina"+axisNames[axis]+"/")
    samples0=[(path+"Carina"+axisNames[axis]+"/" + s,0) for s in samples0]
    samples1=os.listdir(path+"noCarina"+axisNames[axis]+"/")
    samples1=[(path+"noCarina"+axisNames[axis]+"/" + s,1) for s in samples1]
    samples=samples0+samples1
    random.shuffle(samples)
    #print(len(samples))

    for i in range(len(samples)):
        img=torch.from_numpy(io.imread(samples[i][0]))
        s=img.size()
        #if s[0]*s[1]!= 1024:
            #print(samples[i][0])
            #print('size = '+ str(s))


    train_samples=[(torch.from_numpy(io.imread(s)),c) for (s,c) in samples if test_subject not in s and '.DS_Store' not in s]
    test_samples=[(torch.from_numpy(io.imread(s)),c) for (s,c) in samples if test_subject in s and '.DS_Store' not in s]

    #print(len(train_samples))
    #print(len(test_samples))
    return (train_samples,test_samples)

def printPatch(image3D,patch, spacing, coordinates):
    offset=int((patchsize-1)/2)
    plotImg=ndimage.rotate(image3D[int(coordinates[0]),:,:], 90)
    plotAspect=(spacing[2][1]-spacing[2][0])/(spacing[1][1]-spacing[1][0])
    ax1=plt.subplot(121)

    ax1.imshow(plotImg, cmap=plt.cm.bone, aspect=plotAspect, extent=[0, 512, 0, len(spacing[2])])
    #ax1.plot(coordinates[1],coordinates[2],'.', linewidth=1, color='yellow')
    # Create a Rectangle patch
    #rect=patches.Rectangle((coordinates[1]-offset,coordinates[2]-offset),patchsize,patchsize,linewidth=1,edgecolor='r',facecolor='none')
    #ax1.add_patch(rect)

    #ax2=plt.subplot(122)
    #plotPatch=ndimage.rotate(patch, 90)
    #ax2.imshow(plotPatch,cmap=plt.cm.bone,aspect=plotAspect)
    plt.show()

def classifyImage(input,net):
    input=input.unsqueeze(0)
    input=input.unsqueeze(0)
    input = input.type('torch.FloatTensor')
    output = net(input)
    return output

def evaluateNet(net, test_samples):
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    with torch.no_grad():
        for  i in range(len(test_samples)): #len(test_samples)
            # get the inputs
            input, label = test_samples[i]
            output = classifyImage(input,net)
            #mx = torch.max(output)
            nx = output.numpy()[0]
            mx=max(nx)
            idx=np.where(nx == mx)[0][0]
            c = (idx == label).squeeze()
            #print(c.item())
            class_correct[label] += c.item()
            class_total[label] += 1

    for i in range(2):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    return ((class_correct[0] / class_total[0]),(class_correct[1] / class_total[1]))

def detect(netN, img3d, patchsize, axis,stride):
    size3d=img3d.shape
    #means: 289.0, 249.0, 81.0
    '''
    if(axis==0):
        results=np.zeros((size3d[0],size3d[1]-patchsize[0]+1,size3d[2]-patchsize[0]+1))
    else:
        if axis==1:
            results=np.zeros((size3d[0]-patchsize[0]+1,size3d[1],size3d[2]-patchsize[0]+1))
        else:
            results=np.zeros((size3d[0]-patchsize[0]+1,size3d[1]-patchsize[0]+1,size3d[2]))
    '''
    results=np.zeros((size3d[0]-patchsize+1,size3d[1]-patchsize+1,size3d[2]-patchsize+1))
    print(results.shape)
    print(size3d)
    with torch.no_grad():
        for x in range(results.shape[0]):
            if x%stride==0:
                print("slice: "+str(x))
                for y in range(results.shape[1]):
                    if y%stride==0:
                        for z in range(results.shape[2]):
                            if z%stride==0:
                                if axis==0:
                                    img=torch.from_numpy(img3d[x+16,y:y+patchsize,z:z+patchsize])
                                else:
                                    if axis==1:
                                        img=torch.from_numpy(img3d[x:x+patchsize,y+16,z:z+patchsize])
                                    else:
                                        #print(x)
                                        #print(y)
                                        #print(z)
                                        img=torch.from_numpy(img3d[x:x+patchsize,y:y+patchsize,z+16])
                                output=classifyImage(img,netN)
                                results[x,y,z]=output[0][0].numpy()

    return results

def detect2d(netN,img2d,patchsize,stride):
    size2d=img2d.shape
    results=np.zeros((size2d[0]-patchsize+1,size2d[1]-patchsize+1))
    print("size 2d")
    print(size2d)
    print(results.shape)
    with torch.no_grad():
        for x in range(results.shape[0]):
            if x%stride==0:
                print("slice: "+str(x))
                for y in range(results.shape[1]):
                    if y%stride==0:
                        img=torch.from_numpy(img2d[x:x+patchsize,y:y+patchsize])
                        #print(x)
                        #print(y)
                        #print(img.shape)
                        output=classifyImage(img,netN)
                        results[x,y]=output[0][0].numpy()
    return results

def showHeatmap(result,img,axis):
    if axis<2:
        plotResults=ndimage.rotate(result, 90)
        plotAspect=(spacing[2][1]-spacing[2][0])/(spacing[1][1]-spacing[1][0])
        img3dRotated=ndimage.rotate(img, 90)
    else:
        plotResults=ndimage.rotate(result, 0)
        plotAspect=1
        img3dRotated=ndimage.rotate(img, 0)

    resultSize=result.shape
    print(resultSize)
    fig, ax = plt.subplots()
    ax=plt.subplot(121)

    a=img3dRotated[16:16+resultSize[1],16:16+resultSize[0]]
    print(img3dRotated.shape)
    print(a.shape)

    im = ax.imshow(a,cmap=plt.cm.bone, aspect=plotAspect)
    im = ax.imshow(plotResults,cmap=plt.cm.RdYlGn, alpha=0.4, aspect=plotAspect)

    ax=plt.subplot(122)
    im = ax.imshow(plotResults,cmap=plt.cm.RdYlGn, aspect=plotAspect, alpha=1.0)
    plt.show()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def dist3D(x1,x2,y1,y2,z1,z2):
    return np.sqrt(np.square(x2-x1)+np.square(y2-y1)+np.square(z2-z1))

#######main#############################

##classification
'''
results=np.zeros([20,2])
for i in range(20):
    test_subject = 100+i
    train_samples,test_samples=importPatches(2,str(test_subject))

    net = Net()
    #print(net)

    net=trainNetwork(net, train_samples)
    (c,noC)=evaluateNet(net, test_samples)
    #
    results[i,0]=c
    results[i,1]=noC
print(results)
'''

##detection

axis=0
distances=[]


counter=0
pathIn="../../../Data/4D-Lung/"
patients=os.listdir(pathIn)
patients=[p for p in patients if '.DS_Store' not in p]
print(patients)
for i in range(20): #" "+str(i*10)+".0"
    test_subject=str(100+i)
    path1=pathIn+test_subject+"_HM10395/" #patients[i]+'/'
    dates=os.listdir(path1)
    dates=[l for l in dates if '.DS_Store' not in l]
    for j in range(1):
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
            print(img)
            image3D,spacing=importDicom(path2+img[0]+"/")
            coordinates,patch=importRstruct(path2+rstruct[0]+"/",image3D,spacing,False, axis)
            c=[int(coor) for coor in coordinates]
            print(coordinates)
            print(image3D.shape)
            off=30
            img3d=image3D[c[0]-off:c[0]+off,c[1]-off:c[1]+off,c[2]-off:c[2]+off]
            s=img3d.shape
            print(s)

            sumResults=np.zeros((s[0]-31,s[1]-31,s[2]-31))
            #3 directions
            for l in range(3): ##3
                axis=l
                test_nr=100+i
                train_samples,test_samples=importPatches(pathOut, axis,str(test_nr))
                net = Net()
                net=trainNetwork(net, train_samples)
                #patchsize = train_samples[0][0].shape
                results=detect(net,img3d, patchsize,i,1)
                sumResults=np.add(sumResults,results)
                print("size results")
                print(results.shape)

            #print(sumResults[5:11,5:11,5:11])
            mx=sumResults.max()
            print("Max value: "+str(mx))
            pos=np.where(sumResults==mx)
            mean=off-(32/2)
            print(mean)
            print(results.shape)
            distances.append(dist3D(pos[0][0],mean,pos[1][0],mean,pos[2][0],mean))


print(distances)







#idx=[289,249,81]
#visualization
'''
axis=2
train_samples,test_samples=importPatches(axis,str(test_subject))
net = Net()
net=trainNetwork(net, train_samples)
patchsize = train_samples[0][0].shape

results=detect2d(net,img3d[:,:,81], patchsize,1)
img=img3d[:,:,81]
showHeatmap(results,img,axis)
'''




#ax1=plt.subplot(121)

#ax1.imshow(plotImg, cmap=plt.cm.bone, aspect=plotAspect, extent=[0, 512, 0, len(spacing[2])])
#im=ax.imshow(results)
