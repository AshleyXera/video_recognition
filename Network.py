from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime

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
       
    testing_data1 = None # free up storage

    X = np.array(X) # convert to numpy array

    plt.imshow(X[1])
    plt.show()

    return X, y


data = load_data(TestFrames1DIR)
X, y = split_data(data)

