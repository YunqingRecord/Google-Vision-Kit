# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:53:40 2019

"""
import itchat
import socket 
import os
import threading
import time
import traceback
from aiy.board import Board, Led
from picamera import PiCamera
from PIL import Image
from Inference_Eigine import Recognizer
import numpy as np


# 标志位：用于判定是否退出任务线程
end_thread = False
lock = threading.Lock()


#%% 通过发送一个UDP包来获取自身的IP地址
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8',80))
        ip = s.getsockname()[0]
    except:
        ip = 'error'
    finally:
        s.close()   
    return ip


#%% 以微信备注名称查找用户的toUserName参数，用于发送消息到指定的好友
def get_user_name(name, isGroup=True):
    if isGroup:
        users= itchat.search_chatrooms(name)
    else:
        users= itchat.search_friends(name)
        
    user_name = users[0]['UserName']
    return user_name


#%% 微信登录成功后，发送一条消息到指定群组
def device_start(user_name):
    try:
        hello = "Google Vision Kit started successfully."
        itchat.send(msg=hello, toUserName=user_name)
        return True
    except:
        return False


#%% 监听指定群聊消息，并作出响应
def listen_wechat():
    # 绑定消息事件，监听指定群聊消息
    @itchat.msg_register(itchat.content.TEXT, isGroupChat=True)
    def reply_msg(msg):
        global end_thread
        # 只处理来自指定群聊的内容
        if msg['FromUserName'] == user_name: # 只处理来自指定群聊的内容
            message = msg['Content']
            # 消息处理方案
            if message == 'config':
                menu = "1:  输入1查找端设备的当前IP\n" + \
                       "2:  以cmd ...格式执行shell\n" + \
                       "3:  按下按钮以进行拍照\n" +\
                       "4:  输入end结束当前运行任务"
                itchat.send_msg(msg=menu, toUserName=user_name)
            if message == '1':
                ip = get_host_ip()
                response = "End-device started with IP address: " + ip
                itchat.send(msg=response, toUserName=user_name) 
            if message[:4] == 'cmd ':
                try:
                    pipeline = os.popen(message[4:])
                    response = pipeline.read()
                except:
                    response = 'command error'
                finally:
                    itchat.send(msg=response, toUserName=user_name)
            if message == '3':
                itchat.send('Press the button to take photos', user_name)
                
                
    itchat.run()

3
#%% 扫描指定缓存目录，一旦其中有图片就将其发送到微信，并在本地删除
def scan_image_buffer(folder_name=None):
    
    model = Recognizer()
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    while True:
        for file_name in os.listdir(folder_name):
            file_path = os.path.join(folder_name, file_name)
            if itchat.send_image(file_path, user_name):
                lock.acquire()
                try:
                    img = np.array(Image.open(file_path).convert('L'))
                    output,label = model.predict(img)
                    itchat.send('Class: '+str(label), user_name)
                    itchat.send('Output: '+str(output), user_name)
                    os.remove(file_path)
                finally:
                    lock.release()
            else:
                itchat.send('failed to send image.', user_name)
        # 每0.5s扫描一次，降低cpu利用率
        time.sleep(0.5)
        
    return None


def take_image():
    def shoot():
        with PiCamera() as camera:
            camera.resolution = (28,28)
            camera.capture('image_buffer/image.jpg')  
        return None
        
    print('Press the button to take a photo.')
    with Board() as board:    
        while True:
            board.button.wait_for_press()
            board.led.state = Led.ON
            shoot()
            itchat.send('Image taken: ')
            board.button.wait_for_release()
            board.led.state = Led.OFF
        
    return None


#%% 
# 登录wechat，指定监听群聊名称
while True:
    # 登录wechat
    try:
        itchat.auto_login(enableCmdQR=2)
        user_name = get_user_name("GoogleVisionKit", isGroup=True)
        device_start(user_name)
        break
    except:
        traceback.print_exc()
    finally:
        time.sleep(3)
            
# 启动多个线程
t1 = threading.Thread(target=listen_wechat, name='1')
t2 = threading.Thread(target=scan_image_buffer, args=('image_buffer',) )
t3 = threading.Thread(target=take_image)
t1.start()
t2.start()
t3.start()
t1.join()
t2.join()
t3.join()




    

    
    
    