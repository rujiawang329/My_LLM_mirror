# Docker 常用配置

## 目录
- [Docker 常用命令](#docker-常用命令)
- [Docker 安装](#docker-安装)
- [Docker 更新](#docker-更新)
- [Docker 配置SSH服务](#Docker-配置SSH服务)
- [Docker 复杂配置运行](#Docker-复杂配置运行)
- [Issue1: SSH连接时过于严格的检测导致连接失败](#issue1-ssh连接时过于严格的检测导致连接失败)
- [Issue2: VSCode Server 下载过慢](#issue2-vscode-server-下载过慢)
- [Issue3: Docker共享内存太小导致奇怪问题](#issue3-docker共享内存太小导致奇怪问题)

## Docker 常用命令

```bash
# 查看所有镜像
docker images

# 拉取镜像
docker pull

# 删除镜像
docker rmi <镜像名>

# 查看容器
docker ps
docker ps -a

# 运行镜像
docker run -it <镜像名> /bin/bash

# 停止容器
docker stop <容器ID>

# 删除容器
docker rm <容器ID>

# 退出
exit

# 重启容器/启动容器
docker restart <容器ID>
docker start <容器ID>

# 进入容器
docker exec -it <容器ID> /bin/bash
```

## Docker 安装

**步骤 1：更新系统包索引**

```bash
sudo apt update
```

 **步骤 2：安装必要的依赖工具**

```bash
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
```

**步骤 3：添加 Docker 的官方 GPG 密钥**

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```

**步骤 4：设置 Docker 的稳定版存储库**

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

**步骤 5：安装 Docker**

```bash
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
```

**步骤 6：验证安装**

检查 Docker 是否安装成功：

```bash
docker --version
```

**启动Docker并设置为开机自启**

```bash
sudo systemctl start docker
sudo systemctl enable docker
```



## Docker　更新

如果你已经安装了 Docker，但需要更新到最新版本，可以按以下步骤操作：

**步骤 1：移除旧版本的 Docker**

```bash
sudo apt remove -y docker docker-engine docker.io containerd runc
```

**步骤 2：安装最新版本的 Docker**

1. 更新包索引：

   ```bash
   sudo apt update
   ```

2. 安装最新版本：

   ```bash
   sudo apt install -y docker-ce docker-ce-cli containerd.io
   ```

**步骤 3：验证更新后的版本**

运行以下命令检查 Docker 版本是否为最新：

```bash
docker --version
```


## Docker 配置SSH服务
```bash
docker run -it <ImageID> /bin/bash

# 更新包列表
apt-get update

# 安装SSH服务
apt-get install -y openssh-server

# 创建必要的目录
mkdir /var/run/sshd
mkdir -p /root/.ssh

# 设置root密码（将'your_password'替换为你想要的密码）
passwd root

# 修改SSH配置允许root登录
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# 启动SSH服务
service ssh start

exit

# 获取容器ID
docker ps -a

# 保存为新镜像
docker commit <容器ID> <新镜像名称>

# 如果要换一个电脑，则需要保存为tar文件
# 保存为tar文件
docker save -o <新镜像名称>.tar <新镜像名称>

# 加载tar文件
docker load -i <新镜像名称>.tar
```

## Docker 复杂配置运行
```bash
docker run --shm-size=16g --gpus all -it -v /media/re/2384a6b4-4dae-400d-ad72-9b7044491b55/LLM_LR/Megatron-LM:/workspace/shenxiao -p 3333:22 <ImageID>
```

## Issue1: SSH连接时过于严格的检测导致连接失败
```bash
Host Megatron-LM
    HostName 127.0.0.1
    Port 3333
    User root
    # 添加下面的设置，防止SSH连接时过于严格的检测导致连接失败
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

## Issue2: VSCode Server 下载过慢

1. 首先获取VSCode Commit版本号
```
Help -> About -> Commit　-> <Commit ID>
```
2. 下载对应的VSCode Server
```bash
curl -L "https://update.code.visualstudio.com/commit:${commit_id}/server-linux-x64/stable" -o vscode-server-linux-x64.tar.gz
```
3. 将下载的文件解压到镜像中
```bash
mkdir -p ~/.vscode-server/bin/${commit_id}
tar zxvf vscode-server-linux-x64.tar.gz -C ~/.vscode-server/bin/${commit_id} --strip-components 1
```
4. 创建完成标记文件
```bash
touch ~/.vscode-server/bin/${commit_id}/0
```
5. 然后重新使用VSCode进行远程SSH连接.

## Issue3: Docker共享内存太小导致奇怪问题

在 Docker 中，默认情况下容器的共享内存大小（`/dev/shm`）通常是 64MB。

**方法 1：在 `docker run` 命令中指定共享内存大小**

通过 `--shm-size` 参数指定共享内存大小，例如设置为 16GB：

```bash
docker run --shm-size=16g -it <image_name> /bin/bash
```

**验证共享内存大小**

进入容器后，使用以下命令验证 `/dev/shm` 的大小：

```bash
df -h /dev/shm
```