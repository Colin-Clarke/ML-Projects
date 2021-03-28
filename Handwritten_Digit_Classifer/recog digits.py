import tensorflow as tf
import numpy as np
import pickle
import keras
import mnist #Database
import matplotlib.pyplot as plt
from keras.models import Sequential #ANN Architecture
from keras.layers import Dense #The layers in the ANN
from keras.utils import to_categorical
import pygame
from PIL import ImageGrab
from keras.models import model_from_json
pygame.init()
import sys
import os
import time

train_images=mnist.train_images() #Training data images
train_labels=mnist.train_labels() #Training data labels
test_images=mnist.test_images() #Training data images
test_labels=mnist.test_labels() #Training data labels

#Normalising pixel values(0-255)-->(-0.5-0.5) (makes network easier to train)
train_images=(train_images/255)#-0.5
test_images=(test_images/255)#-0.5

#Flattening the images from 28x28 images to 784-D vector to pass
#into ANN
train_images=train_images.reshape((-1, 784))
test_images=test_images.reshape((-1, 784))

#Build model
# 3 layers: 2 layers with 64 neurons and the relu function and
#1 layer with 10 neurons and softmax function
model=Sequential()
model.add(Dense(64, activation="relu", input_dim=784))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

#Compile the model
#The loss function measures how well the model did on training and then tries to improve on it using the optimizer
model.compile(
  optimizer="adam",
    loss="categorical_crossentropy", #Classes that are > 2
    metrics=["accuracy"]
)
'''
#Train the model
model.fit(
    train_images,
    to_categorical(train_labels),#Ex. 2 it expects [0,0,1,0,0,0,0,0,0,0]
    epochs=3, #The number of iters over the entire dataset to train on
    batch_size=32 #the number of samples per gradient update for training
)

#Evaluate the model
model.evaluate(
    test_images,
      to_categorical(test_labels)
)
'''
#model.save("model.h5")
model=keras.models.load_model("model.h5")
#predict on the first 10 test images
predictions=model.predict(test_images[:10])
#print our models prediction
#print(np.argmax(predictions, axis=1))
#print(test_labels[:10])

win = pygame.display.set_mode((500,500))
pygame.display.set_caption("Draw Digit")

x=25
y=425
width=28
height=28
vel=10
clock=pygame.time.Clock()
run=True
win.fill((255,255,255))
while run:
    keys = pygame.key.get_pressed()
    pos = pygame.mouse.get_pos()
    #pygame.mouse.set_visible(False)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run=False
        if pygame.mouse.get_pressed()[0]:
            pygame.draw.circle(win, (0, 0, 0),pos, 20, 20)
        if keys[pygame.K_SPACE]:
            win.fill((255,255,255))
        if keys[pygame.K_RETURN]:
            a=str(np.random.random(1))
            save_file="screenshot"+a+".png"
            #print(save_file)
            pygame.image.save(win, save_file)
            run=False

    pygame.display.update()
    clock.tick(3000)

from PIL import Image, ImageFilter
from matplotlib import pyplot as plt

def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Height becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # calculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

        # newImage.save("sample.png)

        tv = list(newImage.getdata())  # get pixel values

        # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
        tva = [((255 - x) * 1.0 / 255.0) for x in tv]
        #print(tva)
        return tva

x=[imageprepare(save_file)]#file path here#

#Now we convert 784 sized 1d array to 24x24 sized 2d array so that we can visualize it
newArr=[[0 for d in range(28)] for y in range(28)]
k = 0
for i in range(28):
    for j in range(28):
        newArr[i][j]=x[0][k]
        k=k+1

for i in range(28):
    for j in range(28):
        x=2
        #print(newArr[i][j])
        #print(' , ')
    #print('\n')


plt.imshow(newArr, interpolation='nearest')
newdigit='MNIST_IMAGE.png'
plt.savefig(newdigit)#save MNIST image
#plt.show()#Show / plot that image

y=np.array(newArr)
ynew=y.reshape((-1, 784))
predictions=model.predict(ynew)
x=np.argmax(predictions, axis=1)
print("The number you drew was a",x[0])

#print(predictions)
os.remove(save_file)
os.remove(newdigit)
#pygame.quit()