from django.shortcuts import render
from django.views.generic import View
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
# from time import strptime

# 학습 전용 모듈
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import os, re, glob
import cv2
from sklearn.model_selection import train_test_split

def index(req):
    context = {
        
    }
    
    return render(req, 'index.html', context=context)

@csrf_exempt
def load_obj_list(request):
    response = {
        'list' : ['tomato', 'flower', 'dog', 'cat']
    }
    
    return JsonResponse(response)

@csrf_exempt
def detail(request):
    response = {
        'obj': 'Apple',
        'results': [0.5, 0.6, 0.7, 0.8, 0.755],
    }
    return JsonResponse(response, safe=False)

@csrf_exempt
def training_data(objects, img_size, count):
    print('받은 데이터값:', objects, img_size)
    """
    학습 시킬 데이터 셋 생성 return X
    objects의 이름 = cnn_sample 디렉터리 안의 폴더 명과 일치
    len(objects) = categories 숫자
    return = void - img_data.npy 파일 생성
    """

    print("MAKING TRAINING DATA")
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    groups_folder_path = './static/images/'

    categories = objects

    image_w = img_size
    image_h = img_size

    X = []
    Y = []
    
    for idex, categorie in enumerate(categories):
        label = idex
        image_dir = groups_folder_path + categorie + '/data/'
    
        check = 0

        for top, dir, f in os.walk(image_dir):
            for filename in f:
                print('이미지는?', image_dir, filename)
                check = check + 1
                if(check > count):
                    break

                img = cv2.imread(image_dir+filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
                X.append(img)
                Y.append(label)

    X_train = np.array(X)
    Y_train = np.array(Y)
    # X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
    #########################################################################################################
    X = []
    Y = []
    
    for idex, categorie in enumerate(categories):
        label = idex
        image_dir = groups_folder_path + categorie + '/data/'
    
        check = 0

        for top, dir, f in os.walk(image_dir):
            for filename in f:
                check = check + 1
                if(check > 150):
                    break

                img = cv2.imread(image_dir+filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
                X.append(img)
                Y.append(label)

    X_test = np.array(X)
    Y_test = np.array(Y)
    #########################################################################################################
    xy = (X_train, X_test, Y_train, Y_test)
    print("train Data :", len(X_train), ", test Data :", len(X_test))
    np.save("./img_data.npy", xy)
    print("DONE")


def make_model(objects, size_img, num_classes):
    """
    training_data 함수의 모델(img_data.npy)을 사용하여 학습 후 테스트 후 신뢰도 return
    return = 신뢰도 - float
    """
    # ap = time.time()

    # allow_pickle이 기본적으로 False라서 True로 변환해 주지 않으면 읽지 못함
    train_images, test_images, train_labels, test_labels = np.load('img_data.npy', allow_pickle=True)
    class_names = objects

    """
    image 전처리
    기본 0 ~ 255 까지 이지만 모델링 과정에서 필요한 것은 0 ~ 1 의 실수
    """
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 모델링 시작
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(size_img, size_img, 3)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    # 모델 컴파일
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # 모델 학습 시작
    model.fit(train_images, train_labels, epochs=25)

    # 가중치 테스트
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('\n테스트 정확도:', str(test_acc * 100) + "%")
    # print(str(int(time.time() - ap)) + " second")

    return test_acc

@csrf_exempt
def make_model_train(request):
    data = JSONParser().parse(request)

    objects = data['objects']
    count = data['count']
    """
    user request -> objects, count
    
    모델 생성 및 학습 결과 return
    return = 신뢰도% - str
    """
    # 차수 당 이미지 숫자
    NUM_IMAGES = 10
    # 이미지 크기
    SIZE_IMAGE = 256
    # 데이터 모델 생성
    training_data(objects, SIZE_IMAGE, NUM_IMAGES * count)

    response = {
        'result' : make_model(objects, SIZE_IMAGE, len(objects))
    }

    return JsonResponse(response)