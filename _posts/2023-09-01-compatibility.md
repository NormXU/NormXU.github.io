---
layout: post
title: "The Compatibility between CUDA, GPU, Base Image, and PyTorch"
tags: ["Engineering"]
---

The relationship between the CUDA version, GPU architecture, and PyTorch version can be a bit complex but is crucial for the proper functioning of PyTorch-based deep learning tasks on a GPU.  Suppose you're planning to deploy your awesome service on an **NVIDIA A100-PCIE-40Gb** server with **CUDA 11.2** and **Driver Version 460.32.03**. You've built your service using **PyTorch 1.12.1**, and your Docker image is built based on an NVIDIA base image, specifically **nvidia-cuda:10.2-base-ubuntu20.04**. How can you judge whether your service can run smoothly on the machine without iterative attempts?

To clarify this complicated compatible problem,  let’s take a quick recap of the key terminologies we mentioned above. 


## Basic Concepts
### GPU Architecture

VIDIA releases new generations of GPUs every year that are based on different architectures, such as Kepler, Maxwell, Pascal, Volta, Turing, Ampere, and up to Hopper as of 2023. These architectures have different capabilities and features, specified by their Compute Capability version (e.g., sm_35, sm_60, sm_80, etc.). "sm" stands for "streaming multiprocessor," which is a key GPU component responsible for carrying out computations. The number following "sm" represents the architecture's version. We denote it as GPU code in the following context.

For example, "sm_70” which corresponds to the Tesla V100 GPU. When you specify a particular architecture with nvcc,  the compiler will optimize your code for that architecture. As a result, your compiled code may not be fully compatible with GPUs based on different architectures.

You can find more detailed explanations in [this](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) blog.



### CUDA Version

Different GPUs require different CUDA versions based on their architecture. CUDA serves as an interface between the software (like PyTorch) and the hardware (NVIDIA GPU).

### PyTorch Version

PyTorch releases are often tightly bound to specific CUDA versions for compatibility and performance reasons.

### Base Image

Copied from NVIDIA docker homepage:

>  base: Includes the CUDA runtime (cudart)
>
> runtime: Builds on the base and includes the [CUDA math libraries](https://developer.nvidia.com/gpu-accelerated-libraries), and [NCCL](https://developer.nvidia.com/nccl). A runtime image that also includes [cuDNN](https://developer.nvidia.com/cudnn) is available.
>
> devel: Builds on the runtime and includes headers, development tools for building CUDA images. These images are particularly useful for multi-stage builds.

## Interrelation

### CUDA and Base Image

The base image only contains the minimum required dependencies to deploy a pre-built CUDA application.  Importantly, there's no requirement for the CUDA version in the base image to match the CUDA version on the host machine. Back to our deployment scenario, our service is built based on `nvidia-cuda:10.2-base-ubuntu20.04` image. However, it will not utilize CUDA 10.2 from the image; instead, it will rely on the host's CUDA, which is CUDA 11.7. One critical point you still need to consider is that if the driver version on the host is too old for the base image's CUDA, our service will fail to start and you will see an error message as:

>  CUDA driver version is insufficient for CUDA runtime version

A version-compatible matrix between the CUDA and driver can be found [here](https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility).

Besides, there is still one consideration you should never miss,

According to the [dockerfile](https://hub.docker.com/layers/andrewseidl/nvidia-cuda/10.2-base-ubuntu20.04/images/sha256-3d4e2bbbf5a85247db30cd3cc91ac4695dc0d093a1eead0933e0dbf09845d1b9?context=explore) of ``nvidia-cuda:10.2-base-ubuntu20.04``

> ENV NVIDIA_REQUIRE_CUDA=cuda>=10.2

The base image requires a minimum CUDA version of the host.

Up till now,

- host CUDA11.2 >= 10.2. the base image is compatible with host ✅

- host driver 460.32.03 meets the minimum requirements of CUDA 10.2 ✅

### PyTorch and CUDA

Each version of PyTorch is usually compatible with one or a few specific CUDA versions. Using an incompatible version might lead to errors or sub-optimal performance.

Following is the Release Compatibility Matrix for PyTorch, copied from [here](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix):

| PyTorch version | Python        | Stable CUDA               | Experimental CUDA         |
| --------------- | ------------- | ------------------------- | ------------------------- |
| 2.1             | >=3.8, <=3.11 | CUDA 11.8, CUDNN 8.7.0.84 | CUDA 12.1, CUDNN 8.9.2.26 |
| 2.0             | >=3.8, <=3.11 | CUDA 11.7, CUDNN 8.5.0.96 | CUDA 11.8, CUDNN 8.7.0.84 |
| 1.13            | >=3.7, <=3.10 | CUDA 11.6, CUDNN 8.3.2.44 | CUDA 11.7, CUDNN 8.5.0.96 |
| 1.12            | >=3.7, <=3.10 | CUDA 11.3, CUDNN 8.3.2.44 | CUDA 11.6, CUDNN 8.3.2.44 |

The official PyTorch [webpage](https://pytorch.org/get-started/previous-versions/#v1121) provides three examples of CUDA version that are compatible with PyTorch 1.12, ranging from CUDA 10.2 to CUDA 11.6. Therefore, PyTorch 1.12.1 in our scenario passes the compatible test.

Up till now,

- PyTorch1.12 is compatible with CUDA 11.2 ✅



### CUDA and GPU

Each CUDA version is compatible with only certain GPU architectures. 



### PyTorch and GPU 
A particular version of PyTorch will be compatible only with the set of GPUs whose compatible CUDA versions overlap with the CUDA versions that PyTorch supports. 

PyTorch libraries can be compiled from source codes into two forms, binary *cubin* objects and forward-compatible *PTX* assembly for each kernel. Both cubin and PTX are generated for a certain target compute capability. A cubin generated for a certain compute capability is supported to run on any GPU with the same major revision and same or higher minor revision of compute capability. For example, a cubin generated for compute capability 7.0 is supported to run on a GPU with compute capability 7.5, however a cubin generated for compute capability 7.5 is *not* supported to run on a GPU with compute capability 7.0, and a cubin generated with compute capability 7.x is *not* supported to run on a GPU with compute capability 8.x.

When the developers of PyTorch release a new version, they include a flag, ``TORCH_CUDA_ARCH_LIST``, in the [setup.py](https://github.com/pytorch/pytorch/blob/78810d78e82f8e18dbc1c049a2b92e559ab567b2/setup.py#L134). In this flag, they can specify which CUDA architecture to build for, such as ``TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 8.0"``. Remember numbers in ``TORCH_CUDA_ARCH_LIST`` are not CUDA versions, these numbers refers to the NVIDIA GPU architectures, such as 7.5 for the Turing architecture and 8.x for the Ampere architecture.

Here is a good table for reference, credit to [dagelf](https://stackoverflow.com/questions/68496906/pytorch-installation-for-different-cuda-architectures/74962874#74962874)

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

- Pytorch 1.12.1 fails to be compatible with  NVIDIA A100-PCIE-40Gb ❌

### Reference

- [NVIDIA Ampere GPU Architecture Compatibility Guide for CUDA Applications](https://docs.nvidia.com/cuda/ampere-compatibility-guide/index.html#building-applications-with-ampere-support)

