# Edge-Cloud-SpecInference

<p align="center">
  <img src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width="500">
</p>

<h3 align="center">
Efficient Large Language Model Inference Framework with Edge-Cloud Collaborative Speculative Decoding
</h3>

<p align="center">
| <a href="#about-the-project"><b>About</b></a> | <a href="#key-features"><b>Key Features</b></a> | <a href="#quick-start"><b>Quick Start</b></a> | <a href="#testing"><b>Testing</b></a> | <a href="#windows-wsl-configuration"><b>Windows WSL Configuration</b></a> | <a href="README_zh.md"><b>中文</b></a> |
</p>

---

## About the Project

Edge-Cloud-SpecInference is an incremental development based on vLLM 0.9.1rc1 that enables edge-cloud collaborative speculative inference. The core of this project is to build a comprehensive edge-cloud inference demo that implements speculative decoding between edge/samll and cloud/large models. This framework allows researchers and developers to:
- Evaluate potential inference performance benefits through edge-cloud collaboration
- Analyze how network latency between edge and cloud affects overall inference efficiency
- Explore optimized deployment strategies for different edge-cloud scenarios

## Key Features

Our core implementation provides a complete edge-cloud speculative inference demo with these key features:

- **Draft Token Generation at Edge**: Small models on edge devices generate draft tokens
- **Token Verification in Cloud**: Large models in the cloud verify the draft tokens
- **Persistent TCP Connection**: Long-lived TCP connections between edge and cloud reduce transmission latency
- **Optimized Communication Protocol**: Efficient data exchange between edge and cloud components
- **Modular Architecture**: Easy to extend and adapt to different edge-cloud scenarios
- **Performance Monitoring**: Built-in metrics to evaluate inference performance and network impact

## Quick Start

### Installation

```bash
# Clone the project
git clone https://github.com/ghy-cmd/Edge-Cloud-SpecInference.git
cd Edge-Cloud-SpecInference

# Install dependencies
# Using precompiled binaries (faster installation, potentially slower inference)
VLLM_USE_PRECOMPILED=1 pip install --editable .

# Or build from source (slower installation, potentially faster inference)
pip install --editable .
```

Verify installation:
```bash
python -c 'import vllm; print("VLLM Installed Successfully!")'
```

### Edge-Cloud Collaborative Deployment Example

#### Edge Side (Small Model)

```bash
# Modify REMOTE_IP to the cloud IP
export REMOTE_IP=76.69.21.202  # Change to cloud (peer) IP
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

#### Cloud Side (Large Model)

```bash
# Modify LOCAL_IP to the edge IP
export REMOTE_IP=0.0.0.0 
export LOCAL_IP=75.255.203.181  # Change to edge (peer) IP
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

## Testing

After successfully deploying the Edge-Cloud collaborative inference system, you can test the functionality and performance using the following methods:

### Basic Functionality Test

Open a new terminal on the edge device and run the following command to test if the system responds correctly:

```bash
curl http://localhost:8082/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B",
    "prompt": "AI的未来是"
  }'
```

### Performance Benchmark Test

To conduct performance benchmark testing, you can use the following command. The `random-input-len` and `random-output-len` parameters represent the input and output token lengths respectively, which can be modified as needed:

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

These testing methods allow you to verify both the basic functionality and performance metrics of your Edge-Cloud collaborative inference deployment.

## Windows WSL Configuration

Since vLLM can only be installed on Linux/Unix-based operating systems, Windows users need to use WSL (Windows Subsystem for Linux) virtualization. When using WSL, port forwarding and port opening configurations are required for external access.

> **Note**: This configuration applies specifically to WSL2 running in NAT mode, which is the default configuration for most WSL2 installations.

### WSL Port Forwarding Setup

External network access to local WSL ports requires port forwarding configuration. First, obtain the WSL IP address by executing the `ip addr` or `ifconfig` command in the WSL terminal. Usually, the WSL IP address can be found under the eth0 interface, which will be used in the next step of port forwarding configuration. For example, in our case it is 172.27.0.173.

Configure port forwarding in Windows system. Open PowerShell or Command Prompt with administrator privileges, and use the `netsh` command to set up port forwarding. For example, to forward port 8082 of the Windows host to port 8082 of WSL, execute the following command. Here we forward ports 8082 and 8083, one for sending requests and one for token transmission. The `connectaddress` is the WSL IP address obtained in the previous step:

```powershell
netsh interface portproxy add v4tov4 listenport=8082 listenaddress=0.0.0.0 connectport=8082 connectaddress=172.27.0.173

netsh interface portproxy add v4tov4 listenport=8083 listenaddress=0.0.0.0 connectport=8083 connectaddress=172.27.0.173
```

### Windows Firewall Configuration

After setting up port forwarding, you need to configure the Windows firewall to allow TCP forwarding. Execute the following commands:

```powershell
New-NetFireWallRule -DisplayName "Allow WSL 8082" -Direction Inbound -LocalPort 8082 -Protocol TCP -Action Allow

New-NetFireWallRule -DisplayName "Allow WSL 8083" -Direction Inbound -LocalPort 8083 -Protocol TCP -Action Allow
```

With these configurations, external networks can access the services running in WSL through the Windows host.