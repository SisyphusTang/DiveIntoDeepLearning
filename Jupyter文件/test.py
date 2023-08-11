
import socket
# 1.创建socket
tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("正在尝试连接服务器...")


# 2. 链接服务器
server_addr = ("61.139.65.141", 50461)
tcp_socket.connect(server_addr)
print("已成功连接服务器")
# 3. 发送数据
send_data = input("请输入要发送的数据：")
tcp_socket.send(send_data.encode("gbk"))

# 4. 关闭套接字
tcp_socket.close()