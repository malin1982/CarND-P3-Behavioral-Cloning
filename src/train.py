
import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Input, Activation
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout
from keras.layers import Lambda, ELU
import cv2


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

#crop above road image, remove car hood:bottom 25 pixels
new_size_col,new_size_row = 160, 80
def preprocessImage(image):
    shape = image.shape
    image = image[math.floor(shape[0]/4)+30:shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row), interpolation=cv2.INTER_AREA)
    return image

def preprocess_image_file_train(line_data):
    select = np.random.randint(3)
    if (select == 0):
        img_file = line_data['left'][0].strip()
        steer_corr = .25
    if (select == 1):
        img_file = line_data['center'][0].strip()
        steer_corr = 0.
    if (select == 2):
        img_file = line_data['right'][0].strip()
        steer_corr = -.25
    y_steer = line_data['steer'][0] + steer_corr
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = augment_brightness_camera_images(image)
    image = add_random_shadow(image)
    image = trans_image(image)
    image = preprocessImage(image)
    image = np.array(image)
    flip = np.random.randint(2)
    if flip==0:
        image = cv2.flip(image,1)
        y_steer = -y_steer

    return image,y_steer

def trans_image(image):
    # Translation
    rows, cols = image.shape[:2]
    #tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_x = 0
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))

    return image_tr

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

# input image dimensions
img_rows, img_cols = 80, 160

# number of convolutional filters to use
nb_filters = 24

# convolution kernel size
nb_conv = 5

input_shape=(img_rows,img_cols, 3)

model = Sequential()
model.add(Lambda(lambda x: x/255. - 0.5, input_shape = input_shape ))
model.add(Conv2D(nb_filters, nb_conv, nb_conv, border_mode='valid', subsample=(2, 2), init='he_normal'))
model.add(ELU())
model.add(Conv2D(36,5,5, border_mode='valid', init='he_normal', subsample=(2, 2)))
model.add(ELU())
model.add(Conv2D(48,5,5, border_mode='valid', init='he_normal', subsample=(2, 2)))
model.add(ELU())
model.add(Conv2D(64,3,3, border_mode='valid', init='he_normal'))
model.add(ELU())
model.add(Conv2D(64,3,3, border_mode='valid', init='he_normal'))
model.add(ELU())
model.add(Flatten())
model.add(Dense(1164, init='he_normal'))
model.add(ELU())
#model.add(Dropout(0.2))
model.add(Dense(100, init='he_normal'))
model.add(ELU())
#model.add(Dropout(0.2))
model.add(Dense(50, init='he_normal'))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(10, init='he_normal'))
model.add(ELU())
#model.add(Dropout(0.5))
model.add(Dense(1, init='he_normal'))
model.compile(loss='mse',optimizer='adam')

def myGenerator(data,batch_size):
    batch_images = np.zeros((batch_size, new_size_row, new_size_col, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data))
            line_s = data.iloc[[i_line]].reset_index()
            x,y = preprocess_image_file_train(line_s)

            # keep_pr = 0
            # while keep_pr == 0:
            #     x,y = preprocess_image_file_train(line_s)
            #     if abs(y)<.1:
            #         pr_val = np.random.uniform()
            #         if pr_val>pr_threshold:
            #             keep_pr = 1
            #     else:
            #         keep_pr = 1
            x = x.astype('float32')
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering

pr_threshold = 1

batch_size = 256

data_files_s = pd.read_csv('data/driving_log.csv', delimiter=',', dtype=None, usecols = (0,1,2,3))
for i_pr in range(8):
    #print(i_pr)
    #print(pr_threshold)
    train_generator = myGenerator(data_files_s, batch_size)
    model.fit_generator(train_generator, samples_per_epoch=22000, nb_epoch=1, verbose=1)
    #pr_threshold = 1/(i_pr+0.5)*1

# save weights
model.save_weights('model.h5')

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
