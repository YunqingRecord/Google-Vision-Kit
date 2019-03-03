# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:53:40 2019

"""
import itchat
from socket import *
import os
import threading

end_device_ip = None
msg_buffer = None
lock = threading.Lock()


#%% 微信配置
def get_user_name(name, isGroup=False):
    name = u'' + name
    if isGroup:
        users= itchat.search_chatrooms(name)
    else:
        users= itchat.search_friends(name)
    user_name = users[0]['UserName']
    
    return user_name


def receive_data():
    # 初始化全局变量
    global end_device_ip 
    global user_name
    global msg_buffer
    # 创建本地端口号并绑定套接字
    serverPort = 12000
    serverSocket = socket(AF_INET, SOCK_STREAM)
    serverSocket.bind(('',serverPort))
    serverSocket.listen(1)
    print('The server is ready to receive.')
    # 持续监听
    while True:
        # 为监听到的连接创建连接套接字
        connectionSocket, addr = serverSocket.accept()
        # 接收消息并处理
        message = connectionSocket.recv(1024)
        message = message.decode('utf-8')
        
        # 消息处理方案
        if message == 'Start':
            end_device_ip = addr[0]
            response = "Google Vision Kit started with IP: " + end_device_ip
            itchat.send(msg=response, toUserName=user_name)
              
        elif message == 'query':
            if msg_buffer:
                connectionSocket.send(msg_buffer.encode('utf-8'))
                lock.acquire()
                try:
                    msg_buffer = None
                finally:
                    lock.release()

        else:
            itchat.send(message, user_name)
        
        connectionSocket.close()
    return None


def Listen_Wechat():
    # 绑定消息事件，监听指定群聊消息
    @itchat.msg_register(itchat.content.TEXT, isGroupChat=True)
    def reply_msg(msg):
        # 初始化全局变量
        global user_name
        global end_device_ip
        global msg_buffer
        # 只处理来自指定群聊的内容
        if msg['FromUserName'] == user_name:
            content = msg['Content']
            
            # 消息处理方案
            if content == 'config':
                config_menu = "Reply c1: 获取Vision Kit当前IP\n" + \
                              "Reply c2: 以“os xxxx”格式运行cmd\n" + \
                              "Reply c3: 待定\n"
                itchat.send_msg(config_menu, user_name)

            if content == 'c1':
                if end_device_ip:
                    response = "Google Vision Kit started with IP: " + end_device_ip
                else:
                    response = "Google Vision Kit does not start"
                itchat.send_msg(response, user_name)
                
            if content[:3] == 'os ':
                if content[:3]=='sudo poweroff' or content[:3]=='poweroff':
                    end_device_ip = None
                # 
                lock.acquire()   
                try:
                    msg_buffer = content
                finally:
                    lock.release()
    # 运行itchat              
    itchat.run()


# 登录wechat，指定监听群聊
itchat.auto_login(hotReload=True,enableCmdQR=False) 
user_name = get_user_name("Google Vision Kit", isGroup=True)
# 启动多线程
t1 = threading.Thread(target=receive_data, name='1')
t2 = threading.Thread(target=Listen_Wechat, name='2')
t1.start()
t2.start()
t1.join()
t2.join()



    

    
