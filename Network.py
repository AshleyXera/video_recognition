from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt

import os
import cv2

print(tf.VERSION)
print(tf.keras.__version__)
print('cv2: ' + cv2.__version__)

# pull in vid_to_20 from savedFramesIn20
# might not need to if videos are preprocessed

tf.enable_eager_execution()


WorkingDir = 'C:\\Users\\Fletcher\\Documents\\McDaniel\\Summer 2019 research\\Python Code\\'
TestFrames1DIR = WorkingDir + 'Images-Videos\\TestFrames1.1\\'

CATEGORIES = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching',
              'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair',
              'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'Breaststroke',
              'BrushingTeeth', 'CleanandJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen',
              'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl',
              'GolfSwing', 'Haircut', 'HammerThrow', 'Hammering', 'HandstandPushups', 'HandstandWalking',
              'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow',
              'JugglingBalls', 'JumpRope', 'JumpingJack', 'Kayaking', 'Knitting', 'LongJump', 'Lunges',
              'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing',
              'PlayingGuitar', 'PlayingPiano', 'PlayingTabla', 'PlayingViolin', 'PlayingCello', 'PlayingDaf',
              'PlayingDhol', 'PlayingFlute', 'PlayingSitar', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch',
              'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard',
              'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty',
              'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing',
              'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog',
              'WallPushups', 'WritingOnBoard', 'YoYo']



def load_data(directory):
    data = []
    catNum = 0
    count = 0

    # iterate through each category
    for category in CATEGORIES:
    # limit to a certain number of categories, later use all categories
        if(catNum > 14):
            break

        # update the path, print the name of the category
        pathCat = os.path.join(directory, category)
        pathVid = directory + str(category)
        print( str(catNum) + ': ' + category )

        # iterate through each video
        for video in os.listdir(pathVid):
            pathImg = pathVid + '/' + str(video)
            #print(video)

            # holds each frame
            frame_set = []

            for img in os.listdir(pathImg):
                img_array = cv2.imread(pathImg + '/' + img)

                # adds the frame to the array
                #frame_set.append([np.array(img_array), catNum])
                data.append([np.array(img_array), catNum])

                # adds the frame_set to the input array
                #testing_data1.append([np.array(frame_set), catNum])

        catNum += 1
    print("done")
    return data

def split_data(data):
    X = []
    y = []


    for features, label in data:
       X.append(features)
       y.append(label)

    X = np.array(X) # convert to numpy array

    return X, y

def sort_frames(X, y):
    i=0
    # 20 arrays
    # X2[0] holds frame 0 of each video, X2[1] holds frame 1, etc.
    X2 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

    # holds the category tag of the videos; y2[0] is for the 0th frames, etc.
    y2 = []

    # put data from 0-255 into scale of 0-1
    X = X/255.

    # split each viedo along the columns of X2
    for frame in X:
        X2[ i % 20 ].append(frame)
        
        if i % 20 == 0:
            # add the video's tag to y2
            y2.append(y[i]) 

        i+=1
        if i % 1000 == 0:
            print(i)

    end = datetime.now()
    duration = end - start
    print( 'runtime: ' + str(duration))

def preprocess(directory):
    data = load_data(directory)
    frames, labels = split_data(data)
    frames, labels = sort_frames(frames, labels)

    return frames, labels



frames, labels = preprocess(TestFrames1DIR)

# !!! load the model here !!! #
# time = (datetime.now()).strftime('%Y-%m-%d-%H:%M')
filename = 'model-' # + time
input_file = open(filename, 'rb')
model = pickle.load(input_file)

model.fit(frames, labels, batch_size = 10, epochs = 5, validation_split=0.0)