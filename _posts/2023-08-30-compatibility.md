---
layout: post
title: "The Compatibility between CUDA, GPU, Base Image, and PyTorch"
tags: ["Engineering"]
---

### TL; DR

- **Host CUDA VS Base Image CUDA**: The CUDA verision within a runtime docker image has no relationship with the CUDA version on the host machie. The only thing we need to care about is whether the driver version on the host supports the base image's CUDA runtime
- **PyTorch VS CUDA**: PyTorch is compatible with one or a few specific CUDA versions, more precisely, CUDA runtime APIs. Check the compatible matrix [here](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix)
- **CUDA VS GPU**: Each GPU architectures is compatible with certain CUDA versions, more precisely, CUDA driver versions. Quick check [here](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

- **PyTorch and GPU**:  PyTorch only supports GPU specified in ``TORCH_CUDA_ARCH_LIST`` when compiled

The relationship between the CUDA version, GPU architecture, and PyTorch version can be a bit complex but is crucial for the proper functioning of PyTorch-based deep learning tasks on a GPU.  

Suppose you're planning to deploy your awesome service on an **NVIDIA A100-PCIE-40Gb** server with **CUDA 11.2** and **Driver Version 460.32.03**. You've built your service using **PyTorch 1.12.1**, and your Docker image is built based on an NVIDIA base image, specifically [**nvidia-cuda:10.2-base-ubuntu20.04**](https://hub.docker.com/layers/andrewseidl/nvidia-cuda/10.2-base-ubuntu20.04/images/sha256-3d4e2bbbf5a85247db30cd3cc91ac4695dc0d093a1eead0933e0dbf09845d1b9?context=explore). How can you judge whether your service can run smoothly on the machine without iterative attempts?

To clarify this complicated compatible problem,  let’s take a quick recap of the key terminologies we mentioned above. 


## Basic Concepts
### GPU Architecture

NVIDIA releases new generations of GPUs every year that are based on different architectures, such as Kepler, Maxwell, Pascal, Volta, Turing, Ampere, and up to Hopper as of 2023. These architectures have different capabilities and features, specified by their Compute Capability version (e.g., sm_35, sm_60, sm_80, etc.). "sm" stands for "streaming multiprocessor," which is a key GPU component responsible for carrying out computations. The number following "sm" represents the architecture's version. We denote it as GPU code in the following context.

For example, "sm_70” which corresponds to the Tesla V100 GPU. When you specify a particular architecture with nvcc,  the compiler will optimize your code for that architecture. As a result, your compiled code may not be fully compatible with GPUs based on different architectures.

You can find more detailed explanations in [this](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) post.

### CUDA Version

The terms "CUDA" and "CUDA Toolkits" often appear together. "CUDA XX.X" is shorten for the version of the CUDA Toolkits.It serves as an interface between the software (like PyTorch) and the hardware (like NVIDIA GPU).

CUDA Toolkits include:

1. **Libraries and Utilities**: The CUDA Toolkit provides a collection of libraries and utilities that allow developers to build and profile CUDA-enabled applications, such as CuDNN.
2. **CUDA Runtime API**: The Toolkit includes the CUDA runtime, which provides the application programming interface (API) used for tasks like allocating memory on the GPU, transferring data between the CPU and GPU, and launching kernels (compute functions) on the GPU. CUDA runtime APIs are generally designed to be forward-compatible with newer drivers.
3. **NVCC Compiler**: The Toolkit includes the `nvcc` compiler for compiling CUDA code into GPU-executable code.

### PyTorch Version

PyTorch releases are often tightly bound to specific CUDA versions for compatibility and performance reasons.

### Base Image

Copied from NVIDIA docker [homepage](https://hub.docker.com/r/nvidia/cuda):

>  base: Includes the CUDA runtime (cudart)
>
> runtime: Builds on the base and includes the [CUDA math libraries](https://developer.nvidia.com/gpu-accelerated-libraries), and [NCCL](https://developer.nvidia.com/nccl). A runtime image that also includes [cuDNN](https://developer.nvidia.com/cudnn) is available.
>
> devel: Builds on the runtime and includes headers, development tools for building CUDA images. These images are particularly useful for multi-stage builds.

## Interrelation

### CUDA and Base Image

The base image only contains the minimum required dependencies to deploy a pre-built CUDA application.  Importantly, there's no requirement for the CUDA version in the base image to match the CUDA version on the host machine. 

Back to our deployment case

- our service is built based on `nvidia-cuda:10.2-base-ubuntu20.04` image
- The host machine has a CUDA driver that supports up to CUDA 11.2

In this setup, the service built with `nvidia-cuda:10.2-base-ubuntu20.04` image doesn't mean there installs a driver which supports CUDA 10.2 inside the image; instead, it relies on the host's driver which can support up to CUDA 11.7. 

Therefore, the service container will use the CUDA 10.2 runtime API, and because the host driver (supporting up to CUDA 11.2) is forward-compatible with older CUDA runtime versions, the application should run without any issues.

Therefore, the only one critical point you need to consider is that 

**Whether the driver version on the host supports the base image's CUDA runtime**

The CUDA runtime version inside the container must be less than or equal to the CUDA driver version on the host system, or else you might encounter compatibility issues and the service will fail to start with an error message as:

>  CUDA driver version is insufficient for CUDA runtime version

A version-compatible matrix between the CUDA and driver can be found [here](https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility).

Besides, there is still one consideration you should never miss. According to the line 16 in the [dockerfile](https://hub.docker.com/layers/andrewseidl/nvidia-cuda/10.2-base-ubuntu20.04/images/sha256-3d4e2bbbf5a85247db30cd3cc91ac4695dc0d093a1eead0933e0dbf09845d1b9?context=explore) of ``nvidia-cuda:10.2-base-ubuntu20.04``

> ENV NVIDIA_REQUIRE_CUDA=cuda>=10.2

The base image requires a minimum CUDA version of the host.

Up till now,

- **host has CUDA11.2 >= 10.2. the base image is compatible with host** ✅

- **host driver 460.32.03 meets the minimum requirements of CUDA 10.2** ✅

### PyTorch and CUDA

PyTorch versions is compatible  with one or a few specific CUDA versions, or more precisely, with corresponding CUDA runtime API versions. Using an incompatible version might lead to errors or sub-optimal performance.

Following is the Release Compatibility Matrix for PyTorch, copied from [here](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix):

| PyTorch version | Python        | Stable CUDA               | Experimental CUDA         |
| --------------- | ------------- | ------------------------- | ------------------------- |
| 2.1             | >=3.8, <=3.11 | CUDA 11.8, CUDNN 8.7.0.84 | CUDA 12.1, CUDNN 8.9.2.26 |
| 2.0             | >=3.8, <=3.11 | CUDA 11.7, CUDNN 8.5.0.96 | CUDA 11.8, CUDNN 8.7.0.84 |
| 1.13            | >=3.7, <=3.10 | CUDA 11.6, CUDNN 8.3.2.44 | CUDA 11.7, CUDNN 8.5.0.96 |
| 1.12            | >=3.7, <=3.10 | CUDA 11.3, CUDNN 8.3.2.44 | CUDA 11.6, CUDNN 8.3.2.44 |

The official PyTorch [webpage](https://pytorch.org/get-started/previous-versions/#v1121) provides three examples of CUDA version that are compatible with PyTorch 1.12, ranging from CUDA 10.2 to CUDA 11.6. Therefore, PyTorch 1.12.1 in our scenario passes the compatible test.

So far so good, we have:

- **PyTorch1.12 is compatible with CUDA 11.2** ✅

### CUDA and GPU

Each GPU architectures is compatible with certain CUDA versions, or more precisely, CUDA driver versions. As for Ampere, the compatibility is shown as below, copied from [this](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) post:

> **Ampere (CUDA 11.1 and later)**
>
>- **SM80 or `SM_80, compute_80`** –
>  NVIDIA [A100](https://amzn.to/3GqeDrq) (the name “Tesla” has been dropped – GA100), NVIDIA DGX-A100
>- **SM86 or `SM_86, compute_86`** – (from [CUDA 11.1 onwards](https://docs.nvidia.com/cuda/ptx-compiler-api/index.html))
>  Tesla GA10x cards, RTX Ampere – RTX 3080, GA102 – RTX 3090, RTX A2000, A3000, [RTX A4000](https://www.amazon.com/PNY-NVIDIA-Quadro-A6000-Graphics/dp/B08NWGS4X1?msclkid=45987a9faa0411ec98c321cb30a0780e&linkCode=ll1&tag=arnonshimoni-20&linkId=ccac0fed7c3cac61b4373d7dac6e7136&language=en_US&ref_=as_li_ss_tl), A5000, [A6000](https://www.amazon.com/PNY-VCNRTXA6000-PB-NVIDIA-RTX-A6000/dp/B09BDH8VZV?crid=3QY8KCKXO3FB8&keywords=rtx+a6000&qid=1647969665&sprefix=rtx+a6000%2Caps%2C174&sr=8-1&linkCode=ll1&tag=arnonshimoni-20&linkId=d292ba4d995d2b034a27441321668ffb&language=en_US&ref_=as_li_ss_tl), NVIDIA A40, GA106 – [RTX 3060](https://www.amazon.com/gp/product/B08W8DGK3X/ref=as_li_qf_asin_il_tl?ie=UTF8&tag=arnonshimoni-20&creative=9325&linkCode=as2&creativeASIN=B08W8DGK3X&linkId=5cb5bc6a11eb10aab6a98ad3f6c00cb9), GA104 – RTX 3070, GA107 – RTX 3050, RTX A10, RTX A16, RTX A40, A2 Tensor Core GPU
>
>- **SM87 or `SM_87, compute_87`** – (from [CUDA 11.4 onwards](https://docs.nvidia.com/cuda/ptx-compiler-api/index.html), introduced with PTX ISA 7.4 / Driver r470 and newer) – for Jetson AGX Orin and Drive AGX Orin only

We therefore draw the conclusion:

- **NVIDIA A100-PCIE-40Gb is compatible with CUDA 11.2** ✅

### PyTorch and GPU 
A particular version of PyTorch will be compatible only with the set of GPUs whose compatible CUDA versions overlap with the CUDA versions that PyTorch supports. 

PyTorch libraries can be compiled from source codes into two forms, binary *cubin* objects and forward-compatible *PTX* assembly for each kernel. Both cubin and PTX are generated for a certain target compute capability. A cubin generated for a certain compute capability is supported to run on any GPU with the same major revision and same or higher minor revision of compute capability. For example, a cubin generated for compute capability 7.0 is supported to run on a GPU with compute capability 7.5, however a cubin generated for compute capability 7.5 is *not* supported to run on a GPU with compute capability 7.0, and a cubin generated with compute capability 7.x is *not* supported to run on a GPU with compute capability 8.x.

When the developers of PyTorch release a new version, they include a flag, ``TORCH_CUDA_ARCH_LIST``, in the [setup.py](https://github.com/pytorch/pytorch/blob/78810d78e82f8e18dbc1c049a2b92e559ab567b2/setup.py#L134). In this flag, they can specify which CUDA architecture to build for, such as ``TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 8.0"``. Remember numbers in ``TORCH_CUDA_ARCH_LIST`` are not CUDA versions, these numbers refers to the NVIDIA GPU architectures, such as 7.5 for the Turing architecture and 8.x for the Ampere architecture.

Here is a helpful table for reference, credit to [dagelf](https://stackoverflow.com/questions/68496906/pytorch-installation-for-different-cuda-architectures/74962874#74962874)

| nvcc tag                | TORCH_CUDA_ARCH_LIST | GPU Arch                                                     | Year | eg. GPU           |
| ----------------------- | -------------------- | ------------------------------------------------------------ | ---- | ----------------- |
| sm_50, sm_52 and sm_53  | 5.0 5.1 5.3          | [Maxwell](https://en.wikipedia.org/wiki/Maxwell_(microarchitecture)) support | 2014 | GTX 9xx           |
| sm_60, sm_61, and sm_62 | 6.0 6.1 6.2          | [Pascal](https://en.wikipedia.org/wiki/Pascal_(microarchitecture)) support | 2016 | 10xx, Pxxx        |
| sm_70 and sm_72         | 7.0 7.2              | [Volta](https://en.wikipedia.org/wiki/Volta_(microarchitecture)) support | 2017 | Titan V           |
| sm_75                   | 7.5                  | [Turing](https://en.wikipedia.org/wiki/Turing_(microarchitecture)) support | 2018 | most 20xx         |
| sm_80, sm_86 and sm_87  | 8.0 8.6 8.7          | [Ampere](https://en.wikipedia.org/wiki/Ampere_(microarchitecture)) support | 2020 | RTX 30xx, Axx[xx] |
| sm_89                   | 8.9                  | [Ada](https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture)) support | 2022 | RTX xxxx          |
| sm_90, sm_90a           | 9.0 9.0a             | [Hopper](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) support | 2022 | H100              |

Back to our scenarios, we need check whether PyTorch 1.12.1 can be compatible with NVIDIA Ampere GPU

The quickest step towards judging the capability is to check if the application binary already contains compatible GPU code. As long as PyTorch libraries are built to include GPU arch>=8.0or PTX form or both in  in  ``TORCH_CUDA_ARCH_LIST``, they should work smoothly with the NVIDIA Ampere GPU architecture.

If the PyTorch libraries you are using is either compiled with corresponding ``TORCH_CUDA_ARCH_LIST``, nor compiled in PTX, you can find an error like:

> A100-PCIE-40Gb with CUDA capability sm_80 is not compatible with current PyTorch installation
>
> The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70

Back to our scenarios, this time, the combability test fails.

- **Pytorch 1.12.1 fails to be compatible with  NVIDIA A100-PCIE-40Gb** ❌



## Conclusion 

Now we can certainly know if the service which is built with **PyTorch 1.12.1**, and based on **nvidia-cuda:10.2-base-ubuntu20.04**, is compatible with an **NVIDIA A100-PCIE-40Gb** machine with **CUDA 11.2** and **Driver Version 460.32.03**.

| Compatibility       | Status |
| ------------------- | ------ |
| CUDA and Base Image | ✅      |
| PyTorch and GPU     | ❌      |
| PyTorch and CUDA    | ✅      |
| CUDA and GPU        | ✅      |

The answer is <u><b>NO</b></u>. Then, how do we fix it?

Since current PyTorch fails to be compatible with A100, we might want to upgrade to PyTorch 1.13.1 or even later version. Besisdes, since PyTorch 1.13.1 needs CUDA runtime api >= 11.6, we also need to upgrade the base image with a runtime >= 11.6. To be compatible with the CUDA runtime, the host CUDA driver should also be upgraded to the latest, like Driver Version: 525.116.03 which supports up to CUDA 11.7.

One good recipe is as below:

**host:** NVIDIA A100-PCIE-40Gb, Driver Version: 525.116.03 which supports up to CUDA 11.7

**service:** PyTorch: 1.13.1, base-image: [nvidia/cuda:11.7.1-base-ubuntu20.04](https://hub.docker.com/layers/nvidia/cuda/11.7.1-base-ubuntu20.04/images/sha256-335148f1f4b11529269e668ff3ac57667e5f21458d7f461fd70d667699cf7819?context=explore)

The compatitibilty matrix now passes all checks.

| Compatibility       | Status |
| ------------------- | ------ |
| CUDA and Base Image | ✅      |
| PyTorch and GPU     | ✅      |
| PyTorch and CUDA    | ✅      |
| CUDA and GPU        | ✅      |



## One More Thing

**Q:**  I initiate a container with a image without any CUDA runtime installed inside. Then, after I execute ``docker run --gpu all  <image_name>``, I access the container and find all CUDA-related files on the host system, including CUDA runtime api. My assumption is that ``--gpu all`` will map all CUDA Toolkits to the CPU image, and thereby turn it to a CUDA runtime image. However, this assumption seems wrong for a container initilized from a CUDA 10.2 runtime base image in the same way, since all applications inside such a container still use CUDA 10.2 runtime API, suggesting that the host system's CUDA runtime isn't being mapped into the container. What the hell is going on?

**A:** When you run a Docker container with the `--gpus all` flag, you enable that container to access the host's GPUs. However, this does not mean that all CUDA-related files and libraries from the host are automatically mapped into the container. What happens under the hood may differ based on whether the Docker image itself contains CUDA runtime libraries or not.

### Image without CUDA runtime:

When you start a container based on an image that doesn't contain any CUDA runtime libraries, and you use `--gpus all`, you might observe that certain CUDA functionalities are available in the container. This is often because NVIDIA's Docker runtime (nvidia-docker) ensures that the minimum necessary libraries and binaries related to the GPU are mounted into the container, including the compatible CUDA driver libraries.

### Image with CUDA runtime:

If you start a container from an image that already has a specific CUDA runtime version (say, CUDA 10.2), the container will use that version for its operations. NVIDIA's Docker runtime (nvidia-docker) generally won't override the CUDA libraries in a container that already has them. The container is designed to be a standalone, consistent environment, and one of the benefits of using containers is that they package the application along with its dependencies, ensuring that it runs the same way regardless of where it's deployed.

## Reference

- [NVIDIA Ampere GPU Architecture Compatibility Guide for CUDA Applications](https://docs.nvidia.com/cuda/ampere-compatibility-guide/index.html#building-applications-with-ampere-support)
- [Matching CUDA arch and CUDA gencode for various NVIDIA architectures](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
- [Install previous versions of Pytorch under different CUDA machine](https://pytorch.org/get-started/previous-versions/)
- [CUDA Compatibility Matrix from NVIDIA official documentations](https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility)
