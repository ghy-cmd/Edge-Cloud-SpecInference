# Edge-Cloud-SpecInference

<p align="center">
  <img src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width="500">
</p>

<h3 align="center">
支持边缘-云协同推测解码的高效大语言模型推理框架
</h3>

<p align="center">
| <a href="#关于项目"><b>关于项目</b></a> | <a href="#核心特性"><b>核心特性</b></a> | <a href="#快速开始"><b>快速开始</b></a> | <a href="#测试"><b>测试</b></a> | <a href="#windows-wsl-配置"><b>Windows WSL 配置</b></a> |
</p>

---

## 关于项目

Edge-Cloud-SpecInference 是基于 vLLM 0.9.1rc1 的增量开发项目，支持边缘-云协同推测推理。该项目的核心是构建一个完整的边缘-云推理演示，实现了在边缘/小型模型和云/大型模型之间的推测解码。该框架允许研究人员和开发人员：

- 通过边缘-云协作评估潜在的推理性能优势
- 分析边缘和云之间的网络延迟如何影响整体推理效率
- 探索针对不同边缘-云场景的优化部署策略

## 核心特性

我们的核心实现提供了一个完整的边缘-云推测推理演示，具有以下关键特性：

- **边缘端草稿令牌生成**：边缘设备上的小模型生成草稿令牌
- **云端令牌验证**：云中的大模型验证草稿令牌
- **持久TCP连接**：边缘和云之间的长连接减少传输延迟
- **优化通信协议**：边缘和云组件之间高效的数据交换
- **模块化架构**：易于扩展和适应不同的边缘-云场景
- **性能监控**：内置指标评估推理性能和网络影响

## 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/ghy-cmd/Edge-Cloud-SpecInference.git
cd Edge-Cloud-SpecInference

# 安装依赖
# 使用预编译二进制文件（安装更快，推理可能较慢）
VLLM_USE_PRECOMPILED=1 pip install --editable .

# 或从源码构建（安装较慢，推理可能更快）
pip install --editable .
```

验证安装：
```bash
python -c 'import vllm; print("VLLM 安装成功!")'
```

### 边缘-云协同部署示例

#### 边缘端（小模型）

```bash
# 修改 REMOTE_IP 为云 IP
export REMOTE_IP=76.69.21.202  # 更改为云（对端）IP
export LOCAL_IP=0.0.0.0 
export TCP_PORT=8083 
port_id=8082
export SERVER_PORT=$port_id 

python -m vllm.entrypoints.openai.api_server --model "/mnt/d/ProjectFile/model/Qwen3-8B" \
--served-model-name Qwen3-8B \
--max_model_len 2048 \
--port=$port_id \
--trust-remote-code \
--gpu-memory-utilization 0.7 \
--speculative_config '{"model": "/mnt/d/ProjectFile/model/Qwen3-1.7B", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1 , "remote_target": true, "remote_draft": false}'
```

#### 云端（大模型）

```bash
# 修改 LOCAL_IP 为边缘 IP
export REMOTE_IP=0.0.0.0 
export LOCAL_IP=75.255.203.181  # 更改为边缘（对端）IP
export TCP_PORT=8083 
port_id=8082
export SERVER_PORT=$port_id 

python -m vllm.entrypoints.openai.api_server --model "/mnt/d/download/weight/Qwen3-8B" \
--served-model-name Qwen3-8B \
--max_model_len 2048 \
--port=$port_id \
--trust-remote-code \
--gpu-memory-utilization 0.7 \
--speculative_config '{"model": "/mnt/d/download/weight/Qwen3-1.7B", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1 , "remote_target": false, "remote_draft": true}'
```

## 测试

成功部署边缘-云协同推理系统后，您可以使用以下方法测试功能和性能：

### 基本功能测试

在边缘设备上打开新终端并运行以下命令，测试系统是否正确响应：

```bash
curl http://localhost:8082/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B",
    "prompt": "AI的未来是"
  }'
```

### 性能基准测试

要进行性能基准测试，可以使用以下命令。`random-input-len` 和 `random-output-len` 参数分别表示输入和输出令牌长度，可以根据需要进行修改：

```bash
port_id=8082
python ./benchmarks/benchmark_serving.py \
    --backend vllm  \
    --model Qwen3-8B   \
    --dataset-name random  \
    --tokenizer /mnt/d/download/weight/Qwen3-8B  \
    --random-input-len 4000 \
    --random-output-len 2000 \
    --num-prompts 1 \
    --port $port_id \
    --save-result
```

这些测试方法允许您验证边缘-云协同推理部署的基本功能和性能指标。

## Windows WSL 配置

由于 vLLM 只能安装在基于 Linux/Unix 的操作系统上，Windows 用户需要使用 WSL（Windows Subsystem for Linux）虚拟化。使用 WSL 时，需要配置端口转发和端口开放才能进行外部访问。

> **注意**：此配置专门适用于 NAT 模式下运行的 WSL2，这是大多数 WSL2 安装的默认配置。

### WSL 端口转发设置

外部网络访问本地 WSL 端口需要配置端口转发。首先，在 WSL 终端中执行 `ip addr` 或 `ifconfig` 命令获取 WSL IP 地址。通常可以在 eth0 接口下找到 WSL IP 地址，这将在下一步的端口转发配置中使用。例如，在我们的案例中是 172.27.0.173。

在 Windows 系统中配置端口转发。以管理员权限打开 PowerShell 或命令提示符，使用 `netsh` 命令设置端口转发。例如，要将 Windows 主机的端口 8082 转发到 WSL 的端口 8082，请执行以下命令。这里我们转发端口 8082 和 8083，一个用于发送请求，一个用于令牌传输。`connectaddress` 是上一步获取的 WSL IP 地址：

```powershell
netsh interface portproxy add v4tov4 listenport=8082 listenaddress=0.0.0.0 connectport=8082 connectaddress=172.27.0.173

netsh interface portproxy add v4tov4 listenport=8083 listenaddress=0.0.0.0 connectport=8083 connectaddress=172.27.0.173
```

### Windows 防火墙配置

设置端口转发后，您需要配置 Windows 防火墙以允许 TCP 转发。执行以下命令：

```powershell
New-NetFireWallRule -DisplayName "Allow WSL 8082" -Direction Inbound -LocalPort 8082 -Protocol TCP -Action Allow

New-NetFireWallRule -DisplayName "Allow WSL 8083" -Direction Inbound -LocalPort 8083 -Protocol TCP -Action Allow
```

通过这些配置，外部网络可以通过 Windows 主机访问 WSL 中运行的服务。