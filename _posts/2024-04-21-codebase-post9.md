---
layout: post
title: "Build a Codebase from Scratch"
tags: ["Engineering"]
excerpt_separator: <!--more-->
toc: true
---

<h3 class="no_toc"> TL; DR</h3>

When starting a new project, I often ponder which codebase would be the best foundation. While the open-source community offers numerous impressive repositories, they often cater to specific demos and may not prioritize training or inference efficiency.

As a result, I've chosen to construct my own codebase by drawing inspiration from several awesome open-source projects.

<!--more-->

The codebase should be designed with the following key characteristics:

- **Scalable:** Supporting TP/DP/MP/PP

- **Reproducibility:** Ensuring that results can be replicated precisely using the same configuration file.

- **Extensibility:** Can easily integrate operators or modules from other codebases, such as Megatron-LM.

<hr>

# Part1. Dataset and Data Stream

## Parallel

MgLM has a very comprehensive [documentations](https://github.com/NVIDIA/Megatron-LM/blob/ccfeda47cb5ca10ee3c4efd9b78c6bb15c2cd3d2/megatron/core/parallel_state.py#L310) about TP/CP/DP/MP. 

The initialize_model_parallel function mentioned 3 use cases: 

Let's say we have a total of 16 GPUs denoted by g0 ... g15

#### **Data Parallel**

When DP=8, we arrange groups like:

```
[g0, g2]
[g1, g3]
[g4, g6]
[g5, g7]
[g8, g10]
[g9, g11]
[g12, g14]
[g13, g15]
```

The arrangement indicates an alternating pattern where consecutive groups skip one GPU before pairing with the next. This pattern can be explained by two primary factors:

1. In many multi-GPU setups, GPUs are interconnected in a way that adjacent GPUs (like g0 and g1) might share certain system resources (e.g., memory bandwidth, PCIe lanes). By pairing GPUs that are not directly adjacent (e.g., g0 and g2), it might be possible to optimize the usage of these shared resources, potentially reducing bottlenecks that occur when adjacent GPUs are used simultaneously for similar tasks.

2. Alternating GPUs ensures a more uniform distribution of computational load across different parts of the GPU cluster.

#### **Tensor Parallel**

When TP=8:

```
[g0, g1]
[g2, g3]
[g4, g5]
[g6, g7]
[g8, g9]
[g10, g11]
[g12, g13]
[g14, g15]
```

While tensor model-parallel groups hve a more straightforward and intuitive pattern.

Tensor model parallelism involves splitting the model itself across multiple GPUs. Each GPU handles a part of the model's computations. This is particularly useful for very large models that might not fit into the memory of a single GPU. 

Adjacent GPUs often have faster or more direct communication paths between them. This can be due to their physical proximity on the motherboard or their direct connection via high-speed links like NVLink (in NVIDIA GPUs). Therefore, for tensor parallel groups, we arrange them using adjacent order.

#### **Pipeline Parallel**

When PP=4:

```
[g0, g4, g8, g12]
[g1, g5, g9, g13]
[g2, g6, g10, g14]
[g3, g7, g11, g15]
```

We arrange GPUs into 4  groups, ensuring that within each group, GPUs are not placed adjacent to one another. The reasons behind this practice goes the same as the 8 data parallel groups.

#### **Context Parallel**

It is a very interesting concept, but lacks documentations. As for transformer-based models, the sequence length could be very long and a large sequences may not fit entirely within the memory of a single GPU, context parallelism is here used to split the input sequence length across multiple GPUs. However, unlike simpler data or model parallelism, context parallelism requires frequent communication among GPUs to share parts of the input sequence they are processing, because that is how attention mechanism works.

This is critical because each part of the GPU cluster only sees a portion of the input, but computations (like calculating attention scores) require knowledge of the full input array. Therefore, a good practice of Context Group is composed of corresponding GPUs from other tensor parallel groups that handle different segments of the same sequence, which means each context parallel group contains one GPU from each tensor parallel group, ensuring that all segments of the sequence can be combined and communicated across the GPUs as needed.

For instance,

**Total GPUs**: 8 (g0 to g7)
**Context Parallel Size**: 4

**Tensor Parallel Groups**: Since context parallel size is 4, let's assume we have 2 tensor parallel groups containing 4 GPUs each. Specifically, the tensor parallel groups are arranged as follows:

- Group A: `[g0, g1]`
- Group B: `[g2, g3]`
- Group C: `[g4, g5]`
- Group D: `[g6, g7]`

However, they are actually divided into 4 groups for the purpose of context parallelism, each handling different segments of the data. Each context parallel group needs to contain one GPU from each tensor parallel group that corresponds to handling a portion of the sequence:

**Context Parallel**

- **Group 1**: Comprised of the first GPU from each tensor parallel group
  `[g0, g2, g4, g6]`
- **Group 2**: Comprised of the second GPU from each tensor parallel group:
  `[g1, g3, g5, g7]`

This setup ensures that for any given part of the input sequence, there is one GPU from each of the four context parallel groups that can communicate with GPUs from the other context parallel groups to exchange information about different parts of the sequence.

Each context parallel group can communicate within itself (g0 with g2, g4, g6 and so on) to share and gather information from the different segments of the data that each GPU processes.

#### **Virtual Pipline Parallel**

If **tensor_model_parallel_size is 1**, **pipeline_model_parallel_size is 4**, **virtual_pipeline_model_parallel_size is 2**, and there are 16 transformer layers in the model, the model will be split into 8 stages with two layers each and each GPU would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

Why do we need VP?
In standard pipeline parallelism, each GPU executes a fixed set of model layers and then remains idle while waiting for the next batch of data to process. This idle time arises because of dependencies between stages and the sequential nature of execution. Virtual pipeline model parallelism reduces this idle time by interleaving different segments of the workload across GPUs. This way, when one segment is waiting on data dependencies, another segment can be processed, thus keeping the GPUs busier.
Another reason is to reduced Bubble Time: Pipeline parallelism often suffers from "bubbles" or idle times, particularly when data is being passed between stages or during synchronization points. Virtual pipeline model parallelism can minimize these bubbles by ensuring that different stages are ready to execute as soon as they receive their inputs, thereby reducing the wait times that typically occur between stages.

## Dataloader

The dataset class should only handle data retrieval and define the  `__getitem__` method for various data formats, without be aware of any specific data or transformations required by the downstreaming tasks.

For instance, when utilizing the ImageNet dataset for downstream tasks such as classification and object detection, the required data formats vary significantly. For classification tasks, the expected format is (image_path, label), whereas for contrastive learning, it's (image_path, box coordinates).

To prepare the data format that a task want, I strongly suggest using MapDataset, a PyTorch hook-like style to post-process the data stream. 

There are two types of dataset objects, a [Dataset](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.Dataset) and an [IterableDataset](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.IterableDataset). Whichever type of dataset you choose to use or create depends on the size of the dataset. In general, an `IterableDataset` is ideal for big datasets (think hundreds of GBs!) due to its lazy behavior and speed advantages.

As for `IterableDataset`, you can access it using a `for` loop to load the data progressively as you iterate over the dataset. This way, only a small fraction of examples is loaded in memory, and you don’t write anything on disk.

If your dataset grows very large, since regular Dataset objects are based on Arrow for random access to the rows, its indices mapping will become 10x slower. This is because there is an extra step to get the row index to read using the indices mapping, and most importantly, you aren’t reading contiguous chunks of data anymore. While an `IterableDataset` and leveraging its fast approximate shuffling method. It only shuffles the shards order and adds a shuffle buffer to your dataset, which keeps the speed of your dataset optimal.

Currently, iterable-style datasets are incompatible with customized samplers in `torch.utils.data.DataLoader`.  Pytorch Dataloader always expects a map-style dataset. That is why we usually pass sampler inside an iterable-style dataset for initialization. Specifically, please check the code gists in [detectron2](https://github.com/facebookresearch/detectron2/blob/a2e43eab54d28ffbd59f5e9b4e3193b82faeb70f/detectron2/data/common.py#L221).

### Serialization

[This](https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/) blog provides a very clear explanation of why dataset serialization is necessary and how to do dataset serialization effectively.

Please check the code gist from [detectron2](https://github.com/facebookresearch/detectron2/blob/a2e43eab54d28ffbd59f5e9b4e3193b82faeb70f/detectron2/data/common.py#L114) for more details.

`_TorchSerializedList` is defined to serialize each object in the list using Python's `pickle` module. It converts the object into a binary format (`pickle.dumps`) and then converts the binary data into a numpy array of unsigned 8-bit integers(`np.frombuffer`). All serialized byte arrays are concatenated into a single numpy array and then converted into a PyTorch tensor (`self._lst`).

To better access data segment by index, the class also calculates the byte length of each serialized object and stores these lengths in another numpy array.

### pytree

Pytree was initially introduced within Jax. You can find a comprehensive discussion about pytree on HackerNews [here](https://news.ycombinator.com/item?id=36029368). PyTorch developers may find this feature highly useful, and decide to integrate it in a recent release. Now, we can use it straightforward inside pytorch, without any third-party packages:

```python
from torch.utils import _pytree as pytree

return_dict = {
    "pixel_tensors": torch.rand((3, 224, 224)),
    "labels": torch.tensor(1),
    "txt": "a dummy example"
}

return_dict = pytree.tree_map_only(torch.Tensor, 
                     lambda x: x.cuda(non_blocking=True), return_dict)

# all tensors in return_dict are moved to cuda device
```

`pytree.tree_map_only` is used to selectively apply operations to only those objects within a nested structure that are PyTorch tensors. This is quite helpful where you might have complex data structures containing a mix of tensors, lists, dictionaries, etc., and you want to process only the tensors.  Start using pytree today, your training codes will receive the following benefits for free !

**Efficiency and Convenience:** Manually checking the type of each element in a nested structure and applying a function to it can be cumbersome and error-prone, especially for deeply nested or complex structures. `pytree.tree_map_only` abstracts this logic, making the code cleaner and more efficient.  

**Data Preparation for Distributed Computing:** The specific use-case involves preparing tensor data for efficient serialization and transfer in a distributed computing environment. Using `tree_map_only` allows for a straightforward, generalized way to ensure all tensor data is correctly processed for this environment, without altering the overall structure or non-tensor elements of the data being processed.  

### Sampler

Detectron2 has a good [implementations](https://github.com/facebookresearch/detectron2/blob/5c380fdfc62b0124204155d6be3b1016e3dadb2d/detectron2/data/samplers/distributed_sampler.py#L15) of TrainingSampler.

In training, we only care about the "infinite stream" of training data. Therefore, the training sampler is designed to generate an infinite stream of indices and all workers cooperate to correctly shuffle the indices and sample different indices. Ensure that each rank can access different data. This could always lead to a silent bug for training and really hard to be found. Please pay attention when you build your Sampler.



```python
def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
    ...
    self._rank = dist.get_rank()
    self._world_size = dist.get_world_size()

def __iter__(self):
    start = self._rank
    yield from itertools.islice(_infinite_indices, start, None, self._world_size)
```
