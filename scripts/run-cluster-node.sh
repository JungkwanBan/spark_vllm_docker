#!/bin/bash
set -e

# Utility: export now and persist to .bashrc
export_persist() {
    local var_name="$1"
    local var_value="$2"
    export "$var_name"="$var_value"
    if ! grep -q "export $var_name=" ~/.bashrc; then
        echo "export $var_name=\"$var_value\"" >> ~/.bashrc
    else
        sed -i "s|export $var_name=.*|export $var_name=\"$var_value\"|" ~/.bashrc
    fi
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  -r, --role <head|node>    Node role"
    echo "  -h, --host-ip <ip>        IP of this node"
    echo "  -e, --eth-if <name>       Ethernet interface (e.g., enp1s0f0np0)"
    echo "  -i, --ib-if <name>        InfiniBand/RDMA HCA name"
    echo ""
    echo "Required for worker nodes:"
    echo "  -m, --head-ip <ip>        IP of the head node"
    echo ""
    echo "Example:"
    echo "  $0 --role head --host-ip 10.10.10.1 --eth-if enp1s0f0np0 --ib-if rocep1s0f0"
    echo "  $0 --role node --host-ip 10.10.10.2 --eth-if enp1s0f0np0 --ib-if rocep1s0f0 --head-ip 10.10.10.1"
    exit 1
}

NODE_TYPE="" HOST_IP="" ETH_IF_NAME="" IB_IF_NAME="" HEAD_IP=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--role) NODE_TYPE="$2"; shift ;;
        -h|--host-ip) HOST_IP="$2"; shift ;;
        -e|--eth-if) ETH_IF_NAME="$2"; shift ;;
        -i|--ib-if) IB_IF_NAME="$2"; shift ;;
        -m|--head-ip) HEAD_IP="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

if [[ -z "$NODE_TYPE" || -z "$HOST_IP" || -z "$ETH_IF_NAME" || -z "$IB_IF_NAME" ]]; then
    echo "Error: Missing required arguments."; usage
fi
if [[ "$NODE_TYPE" != "head" && "$NODE_TYPE" != "node" ]]; then
    echo "Error: --role must be 'head' or 'node'."; exit 1
fi
if [[ "$NODE_TYPE" == "node" && -z "$HEAD_IP" ]]; then
    echo "Error: --head-ip is required for worker nodes."; exit 1
fi

echo "Configuring [$NODE_TYPE] at $HOST_IP..."

export_persist VLLM_HOST_IP "$HOST_IP"
export_persist RAY_NODE_IP_ADDRESS "$HOST_IP"
export_persist RAY_OVERRIDE_NODE_IP_ADDRESS "$HOST_IP"
export_persist MN_IF_NAME "$ETH_IF_NAME"
export_persist UCX_NET_DEVICES "$ETH_IF_NAME"
export_persist NCCL_SOCKET_IFNAME "$ETH_IF_NAME"
export_persist NCCL_IB_HCA "$IB_IF_NAME"
export_persist NCCL_IB_DISABLE "0"
export_persist OMPI_MCA_btl_tcp_if_include "$ETH_IF_NAME"
export_persist GLOO_SOCKET_IFNAME "$ETH_IF_NAME"
export_persist TP_SOCKET_IFNAME "$ETH_IF_NAME"
export_persist RAY_memory_monitor_refresh_ms "0"

if [ "${NODE_TYPE}" == "head" ]; then
    echo "Starting Ray HEAD node..."
    exec ray start --block --head --port 6379 \
        --node-ip-address "$VLLM_HOST_IP" \
        --include-dashboard=True \
        --dashboard-host "0.0.0.0" \
        --dashboard-port 8265 \
        --disable-usage-stats
else
    echo "Starting Ray WORKER node -> $HEAD_IP..."
    exec ray start --block \
        --address="$HEAD_IP:6379" \
        --node-ip-address "$VLLM_HOST_IP"
fi
