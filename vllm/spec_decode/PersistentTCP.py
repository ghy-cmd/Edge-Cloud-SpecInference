import socket
import struct
import pickle
import time
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

class PersistentTCPClient:
    """客户端长连接工具（端侧使用）"""
    def __init__(self, ip: str, port: int, max_retries: int = 10):
        self.ip = ip
        self.port = port
        self.max_retries = max_retries
        self.timeout = None
        self.socket: Optional[socket.socket] = None  # 持久连接

    def _connect(self) -> socket.socket:
        """建立新连接，带重试"""
        logger.info(f"正在尝试连接到 {self.ip}:{self.port}")
        for retry in range(self.max_retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.connect((self.ip, self.port))
                return sock
            except (socket.timeout, ConnectionRefusedError) as e:
                if retry == self.max_retries - 1:
                    raise ConnectionError(f"连接失败（{self.max_retries}次重试）：{e}")
                time.sleep(0.1)  # 重试间隔
        raise RuntimeError("未预期的重试退出")

    def ensure_connected(self):
        """确保连接有效，断开则重连"""
        if not self.socket:
            self.socket = self._connect()
        else:
            # 简单检测连接是否存活（发送空数据会报错）
            try:
                self.socket.send(b"")
            except (BrokenPipeError, ConnectionResetError):
                self.socket.close()
                self.socket = self._connect()

    def send(self, data: object):
        """发送数据（带长度前缀，解决粘包）"""
        # serialized = pickle.dumps(data)
        # self.socket.sendall(struct.pack(">I", len(data)))
        self.socket.sendall(data)

    def recv(self) -> object:
        """接收数据（按长度前缀解析）"""
        # length_bytes = self.socket.recv(4)
        # if not length_bytes:
        #     raise ConnectionError("连接已关闭（长度前缀为空）")
        # data_length = struct.unpack(">I", length_bytes)[0]
        # # 按长度读取完整数据
        # data = b""
        # while len(data) < data_length:
        #     chunk = self.socket.recv(min(data_length - len(data), 4096))
        #     if not chunk:
        #         raise ConnectionError("连接已关闭（数据不完整）")
        #     data += chunk
        # return pickle.loads(data)
        data = self.socket.recv(4096)
        return data

    def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()
            self.socket = None


class PersistentTCPServer:
    """服务端长连接工具（云端使用）"""
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.server_socket.bind((self.ip, self.port))
        self.server_socket.listen(1)  # 只接受一个客户端（端侧）
        self.client_socket: Optional[socket.socket] = None

    def accept(self):
        """接受客户端连接（阻塞，建议在单独线程调用）"""
        logger.info(f"正在等待客户端连接 {self.ip}:{self.port}")
        self.client_socket, _ = self.server_socket.accept()
        logger.info(f"已接受客户端连接 {self.ip}:{self.port}")
        self.client_socket.settimeout(None)  # 服务端不超时
        self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def send(self, data: object):
        """发送数据（同客户端，带长度前缀）"""
        if not self.client_socket:
            raise RuntimeError("未接受客户端连接")
        # self.client_socket.sendall(struct.pack(">I", len(data)))
        self.client_socket.sendall(data)

    def recv(self) -> object:
        """接收数据（同客户端）"""
        # if not self.client_socket:
        #     raise RuntimeError("未接受客户端连接")
        # length_bytes = self.client_socket.recv(4)
        # if not length_bytes:
        #     raise ConnectionError("客户端断开连接（长度前缀为空）")
        # data_length = struct.unpack(">I", length_bytes)[0]
        # data = b""
        # while len(data) < data_length:
        #     chunk = self.client_socket.recv(min(data_length - len(data), 4096))
        #     if not chunk:
        #         raise ConnectionError("客户端断开连接（数据不完整）")
        #     data += chunk
        # return pickle.loads(data)
        data = self.client_socket.recv(4096)
        return data

    def close(self):
        """关闭服务端和客户端连接"""
        if self.client_socket:
            self.client_socket.close()
        self.server_socket.close()
		