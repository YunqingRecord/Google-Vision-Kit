# coding: utf-8

from socket  import * 
import time
import threading 
import os


def send_data(message=None):
    # 定义目标服务器的地址/主机名和端口号
    serverName =  '147.8.182.57'
    #'192.168.10.51'
    serverPort = 12000
    try:
        clientSocket = socket(AF_INET, SOCK_STREAM)
        clientSocket.connect((serverName,serverPort))
        clientSocket.send(message.encode('utf-8'))
        # 接收消息并处理
        message = clientSocket.recv(1024)
        message = message.decode('utf-8')
        clientSocket.close()
        return True, message
    except:
        print("remote server dose not response")
        return False, None
        


def query(): 
    while True:
        state, message = send_data('query')
        if state:
            ##定义命令操作
            if message[:3] == 'os ':
                try:
                    pipeline = os.popen(message[3:])
                    response = pipeline.read()
                except:
                    response = 'command error'
                finally:
                    send_data(response)
        time.sleep(0.2)
        
    return None
    

while True:
    state,_ = send_data('Start')
    if state:
        print('The rasp is ready to receive.')
        break
    else:
        time.sleep(5)

# 启动多线程
t1 = threading.Thread(target=query, name='Server to Rasp')
t1.start()
t1.join()


