# jetson nano基础教程

## 一.智能芯片介绍

### 1.边缘计算

<img src=".\What-is-Edge-Computing-768x422.png" style="zoom: 67%;" />

边缘计算是一种分布式计算框架，它使计算和数据存储更接近需要的位置，从而缩短了响应时间并节省了带宽。

随着网络边缘设备数量的增加，传统的在数据中心存储大量数据以进行计算的方法，将网络带宽的需求推到了极限。尽管网络技术也在不断的进步，但数据中心仍无法保证可接受的传输速率和响应时间。

边缘计算的目的是将计算从数据中心移向网络边缘，利用智能设备、移动终端等代替云执行任务并提供服务。通过将服务移至“边缘”，缩短了应用的响应时间。

<img src=".\FederatedLearning_FinalFiles_Flow Chart1.png"  />

Google提出的联合学习由手机设备下载当前模型，再通过手机上收集的数据学习并改进模型，然后将更新汇总为一个小的更新。使用加密通信将模型的更新上传到云，在云上将所有的更新汇总，以改善基础模型。所有的训练数据都保留在个人设备上，不会上传到云。

<img src=".\2017-04-06.gif"  />

   * **速度和效率**

     对于金融、医疗等行业，响应速度非常重要，几百毫秒的延迟就可能决定一个人的命运。边缘计算在本地处理数据，无需回传中央服务器，减少了网络传输的消耗。

   * **隐私和安全性**

     如果所有数据传回服务器，则容易遭受网络攻击。使用边缘计算技术后，一台或少数的针对设备的攻击不会给整个系统带来较大的影响。另外，也保护了用户的隐私，用户的数据都保存在自己的设备上，不会上传到服务器。

   * **可拓展性**

     拓展边缘网络的成本远低于拓展基础AI架构的成本。

   * **可靠性**

     边缘设备可以处理绝大多数功能，即使网络中断也不会影响其主要功能的使用。

     

<img src=".\0ab8eae1eac44adbc629971cfb8ed458.gif" style="zoom: 67%;" />

<img src=".\image4.gif"  />

背景虚化、重对焦......

### 2.智能芯片

目前，关于智能芯片的定义并没有一个严格和公认的标准。一般来说，运用了人工智能技术的芯片都可以称为智能芯片，但是狭义上的智能芯片特指针对人工智能算法做了特殊加速设计的芯片。

智能芯片分类：

* 按架构：

  GPU，FPGA，ASIC，神经拟态芯片

* 按功能：

  训练，推理

* 按应用场景：

  服务器端，移动端

智能芯片的特点：

* 执行大量的并行计算而不是顺序计算

  DNN的计算非常适合并行化，每个神经元之间相互独立，不依赖其他的计算结果。

  数据并行：通过将数据集拆分为不同的batch，以便对每个batch并行执行计算。

  模型并行：将算法模型的结构分为多个部分，在不同的芯片单元上并行执行，通过AI芯片的并行架构来实现。

* 实现精度较低的计算以在相同情况下使用更少的晶体管

  低精度的计算牺牲一定的精度以换取效率和速度，同时降低能耗。

  精度较好的模型通常不会受到噪音的干扰，因此在推理时对数字的四舍五入不会带来太大影响。

  在预先知道神经网络的参数是在较小的数字的范围的情况下，可以使用低位数。

* 在芯片中存储AI算法，以加快内存访问

* 使用内置的编程语言来加速计算机代码的转换和执行

<img src=".\性能对比.jpg"  />



### 3.Jetson nano

NVIDIA Jetson Nano是英伟达推出的小型AI智能开发板，适用于嵌入式物联网应用或AI应用的开发。

<img src=".\Jetson_Nano_Family.png"  />

* Jetson Nano 模组仅有 70 x 45 毫米，是体积非常小巧的 Jetson 设备。 为多个行业（从智慧城市到机器人）的边缘设备部署 AI 时，此生产就绪型模组系统 (SOM) 可以提供强大支持。
* Jetson Nano 提供 472 GFLOP，用于快速运行现代 AI 算法。 它可以并行运行多个神经网络，同时处理多个高分辨率传感器，非常适合入门级网络硬盘录像机 (NVR)、家用机器人以及具备全面分析功能的智能网关等应用。
* Jetson Nano 为您节约时间和精力，助力您实现边缘创新。 体验功能强大且高效的 AI、计算机视觉和高性能计算，功耗仅为 5 至 10 瓦。

应用场景：https://developer.nvidia.com/embedded/community/jetson-projects

Qrio https://www.youtube.com/watch?v=WIIX3dM1_Y8

ShAIdes https://www.youtube.com/watch?v=7UYi-exvHr0

DeepClean https://www.youtube.com/watch?v=Qy8Ks7UTtrA

Neuralet https://www.youtube.com/watch?v=n90W5AcCk34

## 二.系统安装

官方教程：https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

* **下载系统镜像**

  ```
  https://developer.nvidia.com/jetson-nano-sd-card-image
  ```

* **下载SD卡格式化工具,格式化SD卡**

  ```
  https://www.sdcard.org/downloads/formatter/eula_windows/
  ```

  <img src=".\Jetson_Nano-Getting_Started-Windows-SD_Card_Formatter.png"  />

  如果是已写入镜像的或写入失败的sd卡，工具无法检测到。

  <img src=".\无法检测.PNG" style="zoom:80%;" />

  使用cmd命令重新创建分区

  <img src=".\cmd创建分区.PNG"  />

* **下载镜像写入工具，写入镜像**

  ```
  https://www.balena.io/etcher/
  ```

  <img src=".\写入镜像1.PNG"  />

  <img src=".\写入镜像2.PNG"  />

  <img src=".\写入镜像3.PNG"  />

  <img src=".\写入镜像4.PNG"  />

  <img src=".\需要文件.PNG"  />

* **插入SD卡，启动系统**

  <img src=".\Jetson_Nano-Getting_Started-Setup-Insert_microSD-B01.png" style="zoom:50%;" />

  <img src=".\微信图片_20200802123350.jpg" style="zoom: 67%;" />

  <img src=".\微信图片_20200802123406.jpg" style="zoom:67%;" />

  

## 三.机器学习环境安装

官方论坛：https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/jetson-nano/76



* **切换软件apt源**

  ```bash
  sudo mv /etc/apt/sources.list /etc/apt/sources.list.bak
  sudo vim /etc/apt/sources.list
  ```

  ```
  deb https://repo.huaweicloud.com/ubuntu-ports/ bionic main restricted universe multiverse
  deb-src https://repo.huaweicloud.com/ubuntu-ports/ bionic main restricted universe multiverse
  
  deb https://repo.huaweicloud.com/ubuntu-ports/ bionic-security main restricted universe multiverse
  deb-src https://repo.huaweicloud.com/ubuntu-ports/ bionic-security main restricted universe multiverse
  
  deb https://repo.huaweicloud.com/ubuntu-ports/ bionic-updates main restricted universe multiverse
  deb-src https://repo.huaweicloud.com/ubuntu-ports/ bionic-updates main restricted universe multiverse
  
  deb https://repo.huaweicloud.com/ubuntu-ports/ bionic-backports main restricted universe multiverse
  deb-src https://repo.huaweicloud.com/ubuntu-ports/ bionic-backports main restricted universe multiverse
  ```

* **设置pip源**

  ```bash
  mkdir ~/.pip
  vim ~/.pip/pip.conf
  ```

  ```
  [global]
  trusted-host = tuna.tsinghua.edu.cn
  index-url = https://pypi.tuna.tsinghua.edu.cn/simple
  ```

* **安装tensorflow**

  ```bash
  sudo apt-get update
  sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
  sudo apt-get install python3-pip
  sudo pip3 install -U pip
  sudo pip3 install -U testresources setuptools numpy==1.16.1 future==0.17.1 mock==3.0.5 h5py==2.9.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
  sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow==2.2.0+nv20.6
  ```

* **验证tensorflow**

  ```python
  import tensoflow as tf
  print(tf.test.is_gpu_available())
  ```

* **安装jetson-stats**

  ```bash
  sudo pip3 install jetson-stats
  ```

  ```bash
  # 查看设备cpu、gpu使用状况等
  sudo jtop
  ```

  


## 四.其他可能问题

### 1.安装jupyter时报错“command 'aarch64-linux-gnu-gcc' failed with exit status 1”

```bash
sudo apt-get install libxml2-dev libxslt1-dev zlib1g-dev libffi-dev libssl-dev
```



### 2.安装opencv

https://pysource.com/2019/08/26/install-opencv-4-1-on-nvidia-jetson-nano/

https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/

```bash
sudo apt-get install zram-config

sudo vim /usr/bin/init-zram-swapping
# 将mem=$(((totalmem / 2 / ${NRDEVICES}) * 1024)) 修改为 mem=$(((totalmem / ${NRDEVICES}) * 1024))，，然后重启机器
```

```bash
sudo vim /etc/apt/sources.list
# 替换apt源

deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic main multiverse restricted universe
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-security main multiverse restricted universe
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-updates main multiverse restricted universe
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-backports main multiverse restricted universe
deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic main multiverse restricted universe
deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-security main multiverse restricted universe
deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-updates main multiverse restricted universe
deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-backports main multiverse restricted universe
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ xenial main multiverse restricted universe
# deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ xenial main multiverse restricted universe
```

```bash
sudo apt-get update
sudo apt install -y curl
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev libjasper-dev 
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt install -y libv4l-dev v4l-utils qv4l2 v4l2ucp libdc1394-22-dev
sudo apt install -y libxvidcore-dev libx264-dev
sudo apt install -y libgtk-3-dev
sudo apt install -y libatlas-base-dev gfortran
```

```bash
# md5 6757245aec18afe6ee649f2d0d20d092  https://versaweb.dl.sourceforge.net/project/opencvlibrary/4.3.0/OpenCV%204.3.0.zip
curl -L https://github.com/opencv/opencv/archive/4.3.0.zip -o opencv-4.3.0.zip
# md5 d1057144d0cd6f40628088549f8dc1a1  https://d.serctl.com/?uuid=cf624cc9-cac7-49e7-8605-6cdda94dd7a8
curl -L https://github.com/opencv/opencv_contrib/archive/4.3.0.zip -o opencv_contrib-4.3.0.zip
```

```bash
unzip opencv-4.3.0.zip 
unzip opencv_contrib-4.3.0.zip 
cd opencv-4.3.0/
mkdir release
cd release/
```

```
可提前下载face_landmark_model.dat
https://github.com/opencv/opencv_3rdparty/tree/contrib_face_alignment_20170818
vim /{}/opencv_contrib/modules/face/CMakeLists.txt 
将原有http链接置换为file:///下载目录/
```

```bash
cmake   -D WITH_CUDA=ON \
		-D WITH_CUDNN=ON \
		-D OPENCV_DNN_CUDA=ON \
		-D ENABLE_FAST_MATH=1 \
		-D CUDA_FAST_MATH=1 \
		-D CUDA_ARCH_BIN=5.3 \
		-D WITH_CUBLAS=1 \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.3.0/modules \
        -D WITH_GSTREAMER=ON \
        -D WITH_LIBV4L=ON \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=ON \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local ..
```

```bash
make -j4
sudo make install
```



### 3.代理使用

* 安装**privoxy**（转发socks代理为http代理）

  ```bash
  sudo apt-get install privoxy
  ```

* 修改privoxy配置文件

  ```bash
  vim /etc/privoxy/config
  # 添加以下内容
  forward-socks5t   /   127.0.0.1:1080 .
  ```

* 启动privoxy

  ```bash
  sudo systemctl start privoxy
  ```

* 临时设置系统代理

  ```bash
  export http_proxy=http://127.0.0.1:8118
  export https_proxy=http://127.0.0.1:8118
  ```

* apt启用代理（curl，wget不需要额外设置）

  ```bash
  sudo apt-get -o Acquire::http::proxy="http://127.0.0.1:8118/" update
  ```

  

