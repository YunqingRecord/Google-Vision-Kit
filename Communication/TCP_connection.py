
# coding: utf-8

# In[1]:


from socket import *
import sys

try:
    b = u'\u2588'
    sys.stdout.write(b + '\r')
    sys.stdout.flush()
except UnicodeEncodeError:
    BLOCK = 'MM'
else:
    BLOCK = b
    
#%%
def print_cmd_qr(qrText, white=BLOCK, black='  ', enableCmdQR=2):
    blockCount = int(enableCmdQR)
    if abs(blockCount) == 0:
        blockCount = 1
    white *= abs(blockCount)
    if blockCount < 0:
        white, black = black, white
    sys.stdout.write(' '*50 + '\r')
    sys.stdout.flush()
    qr = qrText.replace('0', white).replace('1', black)
    sys.stdout.write(qr)
    sys.stdout.flush()
    
    return None

#%%
# 定义本地端口号
serverPort = 12000
# 创建欢迎套接字并将其与端口号绑定
serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind(('',serverPort))
# 设置为监听模式
serverSocket.listen(1)
print('The server is ready to receive.\n\n')

# 持续监听
while True:
    # 为监听到的连接创建连接套接字
    connectionSocket, addr = serverSocket.accept()  
    # 接收消息并处理
    message = connectionSocket.recv(4096)
    message = message.decode('utf-8')
    print('Please scan the QR code to login.\n')
    print_cmd_qr(message)
    # 返回响应消息，并关闭该连接套接字
    response = 'OK'
    connectionSocket.send(response.encode('utf-8'))
    connectionSocket.close()

