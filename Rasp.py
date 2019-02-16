

from socket  import * 
import time
import threading 
import os


def send_data(message=None):
    # 定义目标服务器的地址/主机名和端口号
    serverName = '192.168.1.101'
    serverPort = 12000
    try:
        clientSocket = socket(AF_INET, SOCK_STREAM)
        clientSocket.connect((serverName,serverPort))
        clientSocket.send(message.encode('utf-8'))
        clientSocket.close()
        return True
    except:
        print("remote server dose not response")
        return False


def receive_data():
    # 定义本地端口号，并绑定端口号
    serverPort = 13000
    serverSocket = socket(AF_INET, SOCK_STREAM)
    serverSocket.bind(('',serverPort))
    serverSocket.listen(1)
    print('The rasp is ready to receive.')
    # 持续监听
    while True:
        # 为监听到的连接创建连接套接字
        connectionSocket, addr = serverSocket.accept()
        # 接收消息并处理
        message = connectionSocket.recv(1024)
        message = message.decode('utf-8')
       
        ##定义命令操作
        if message[:3] == 'os ':
          try:
              pipeline = os.popen(message[3:])
              response = pipeline.read()
          except:
              response = 'command error'
          finally:
              send_data(response)
               
        connectionSocket.close()
        
    return None
    

while True:
    if send_data('Start'):
        break
    else:
        time.sleep(5)

# 启动多线程
t1 = threading.Thread(target=receive_data, name='Server to Rasp')
t1.start()
t1.join()


