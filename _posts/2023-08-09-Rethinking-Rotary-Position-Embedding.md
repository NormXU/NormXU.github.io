---
layout: post
title: "ReRoPE & Rethinking RoPE as a β-based Encoding, Part 1"
categories: LLM
---

> Translated from blogs [1](https://kexue.fm/archives/9675), [2](https://kexue.fm/archives/9706), [3](https://kexue.fm/archives/9708) originally written in Chinese by Su, Jianlin
>
> Translated by Norm Inui

(WIP🚧)
## 1. RoPE is a β-based encoding 
For developers who are interested in how to extend the context length of LLMs (Large Language Models), the open-source community has continuously presented us with fascinating proposals in the past few weeks. First, the user [@kaiokendev](https://www.reddit.com/user/kaiokendev) experimented with a "positional linear interpolation" approach in his project [SuperHOT](https://kaiokendev.github.io/til#extending-context-to-8k). 
He demonstrated that with minimal fine-tuning on long texts, existing LLMs can be easily adapted to handle contexts over their pretraining context length. Almost simultaneously, Meta proposed the same idea, publishing their comprehensive experimental results in the paper titled "[Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)." 
Shortly after the paper was published, [@bloc97](https://www.reddit.com/user/bloc97) introduced the [NTK-aware Scaled RoPE](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/), enabling LLM to extend its context length without fine-tuning!


With all these methods, especially the NTK-aware Scaled RoPE, it persuades me to rethink the idea behind RoPE. 
And I realized that the RoPE can be regarded as a $$ \beta $$-based encoding. From this perspective, the methods mentioned above can be understood as various extending for the encoding base.

### 1.1 Encoding the Number
Suppose we have an integer $$n$$ within $$1,000$$ (not including $$1,000$$) that we want to input into a model. What would be the best way to do this?

The most intuitive idea is to input it directly as a one-dimensional vector. However, the value of this vector has a large range from $$0$$ to $$999$$, which is not easy to optimize for gradient-based optimizers. What if we scale it between 0 and 1? That's not good either, because the difference between adjacent numbers changes from $$1$$ to $$0.001$$, making it challenging for both the model and optimizer to distinguish between the numbers. In general, gradient-based optimizers are a bit "vulnerable" and can only handle inputs that aren't too large or too small.

To avoid this problem, a new way to represent the input is necessary. We might think about how we humans do. For an integer, like $$759$$, it's a three-digit number if encoded with 10 as base, with each digit ranging from 0 to 9. This inspires me to represent the input into the model directly with the 10-based encoding. That is, we transform the integer 

$$n$$ as a three-dimensional vector $$[a,b,c]$$, where $$a$$, $$b$$, and $$c$$ represent the hundreds, tens, and units of $$n$$ respectively. This way, we can both reduce the range of the numbers and get rid of reducing the difference between adjacent numbers, by increasing the input dimension. Luckily, neural networks are good at handling high-dimensional data.

If we want to further reduce the span of each digit in the numbers, we can decrease the base, like using 8, 6, or even 2 as the encoding base at a cost of an increase in input vector dimensions.

### 1.2 Direct Extrapolation

Let's assume we have trained a model using a three-dimensional vector, which is a representation of a number ranging from $$0$$ to $$999$$ with 10 as the encoding base, as the input, and the results are quite well. Then we want to increase the upper bound of $$n$$ to be $$2,000$$. How can we realize this?

If we follow the same idea to represent the number, the input will now be a four-dimensional vector. However, the original model was designed and trained for a three-dimensional vector. Therefore, the model won't be able to process the input with an extra dimension. Some might wonder, why can't we reserve extra dimensions in advance? Indeed, we can pre-allocate a few more dimensions. During the training phase with the upper bound as $$1,000$$, they can be set to $$0$$, but during the inference phase with an upper bound as $$2,000$$, the pre-allocate dimension has to be set to numbers besides $$0$$. This approach is what we call Extrapolation.

![Extrapolation](https://raw.githubusercontent.com/NormXU/NormXU.github.io/main/_data/resources/blog/0/extrapolation.png)

However, the dimensions reserved during the training phase have always been set to 0. If these dimensions were changed to other numbers during the inference phase, the results might be harmed. This is because the model is never trained to adapt to those pre-allocated digits in different values. In other words, due to insufficient training data for those dimensions, the extrapolation usually leads to a significant performance drop.

### 1.3 Linear Interpolation 
Considering the challenges above, some proposed interpolation instead of extrapolation. This means compressing the value upper bound of $$2,000$$ down to $$1,000$$. For example, the number $$1749$$ can simply compress to $$874.5$$ by dividing $$2$$. Thus, $$874.5$$ can be converted into a three-dimensional vector $$[8,7,4.5]$$ and used as input to the model. Following the double mapping strategy, the vector $$[7,4,9]$$ actually corresponds to $$1498$$. 
However, the difference between neighboring numbers was $$1$$, but now it is $$0.5$$, making the last dimension more "crowded". Therefore, interpolation usually needs to fine-tune the model to make it adapt to the mapping which causes some dimensions "crowded". 

![interpolation](https://raw.githubusercontent.com/NormXU/NormXU.github.io/main/_data/resources/blog/0/interpolation.png)


You might notice that the extrapolation can also be fine-tuned to adapt to the pre-allocated dimension. Well, that is correct, but the interpolation requires far fewer steps compared to extrapolation. It is because position embedding (both absolute and relative) has taught the model to understand the relative concept, that is: $$875$$ is greater than $$874$$, without precisely knowing what the number it is. 
Given the model's inherent ability to generalize, learning that $$874.5$$ is greater than $$874$$ is not particularly challenging. 
However, the interpolation is not without flaws. When dealing with an even broader range, particularly in the unit dimension resolution becomes smaller, while the hundreds and tens dimensions still remain at a resolution of 1. This means that the interpolation implicitly introduces resolution inconsistency across dimensions. Each dimension is not equally interpolated, making fine-tuning/continuing learning even more challenging.


### 1.4 Base Conversion
Is there a method that doesn't require adding new dimensions and can still maintain resolution across dimensions? The answer is **YES**. 

It is a solution that we are all familiar with, base conversion! A three-digit decimal number can represent $$0$$ to $$999$$. What if it's in hexadecimal? Its maximum value in 16-base can be $$16^3-1=4095 > 1999$$. So, by simply converting to hexadecimal, with a base of 16, such as turning $$1749$$ into $$[6,13,5]$$, a three-dimensional vector can represent a larger range. The cost? Each dimension's value changes from a range of $$0-9$$ to $$0-15$$.

![conversion](https://raw.githubusercontent.com/NormXU/NormXU.github.io/main/_data/resources/blog/0/conversion.png)

It's indeed a clever idea. As mentioned earlier, what we care about is relative information in the sequence since a trained model has already learned $$875 > 874$$. This holds true in hexadecimal as well. The model doesn't care what base you use to represent the input, but the relative information. Now, the only problem would be if the value of each dimension exceeds $$9$$ (values between $$10-15$$), will the model still know how to compare accurately? Luckily, most LLMs have the generalization capability. Therefore, this base conversion approach can extend the range without fine-tuning! Furthermore, to make interpolation more robust, we can opt for a smaller base like $$\lceil\sqrt[3]{2000}\rceil = 13$$  instead of  $$16$$ to limit the value range of each dimension.


This idea of base conversion finally leads us to the NTK-aware scaled RoPE that was mentioned at the beginning.


### 1.5 Positional Encoding
Based on the explanation above, we can claim:
> The Rotational Positional Encoding (RoPE) at position $$n$$ is the $$\beta$$-based encoding of the number $$n$$

This might surprise you at first glance, however, it does hold true.

_Proof:_

Let's first recall a decimal number n. To calculate the digit at position m (counting from right to left) in its β-based encoding, we have:

**(eq1)**        $$\lfloor\dfrac{n}{\beta^{m-1}}\rfloor \mod \beta $$

As for RoPE, which is adapted from Sinusoidal Position Embedding

**(eq2)**        $$[\text{cos}(\dfrac{n}{\beta^0}), \text{sin}(\dfrac{n}{\beta^0}), \text{cos}(\dfrac{n}{\beta^1}), \text{sin}(\dfrac{n}{\beta^1}), …, \text{cos}(\dfrac{n}{\beta^{d/2-1}}), \text{sin}(\dfrac{n}{\beta^{d/2-1}})]$$, where $$\beta = 10000^{2/d}$$


we can notice that:

1) **eq1** and **eq2** share the same component $$\frac n {\beta^{m-1}}$$;

2) $$\text{mod}$$ introduces periodicity, while $$\text{sin}$$ and $$\text{cos}$$ are also periodical functions. 

Therefore, if we ignore the ceiling operation, we can say RoPE (or Sinusoidal Position Embedding) is a kind of β-based encoding.

With this property, we can now apply extrapolation on $$n$$ by simply replacing $$n$$ as $$n/k$$, $$k$$ is the scale we want to enlarge. This is the Positional Interpolation proposed in Meta's paper, and the experimental results show that extrapolation indeed requires more fine-tuning steps than interpolation.

Regarding numeral base conversion, the objective is to expand the representation range by $$k$$. Therefore, the β-base should be converted to at least $$β(k^{2/d})$$ (according to **eq2**, cos and sin appear in pairs. This can be regarded as a β-base representation with $$d/2$$ bits, not $$d$$). Alternatively, the original base $$10000$$ can be replaced with $$10000k$$, which is the NTK-aware Scaled RoPE. As discussed earlier, since positional embedding has taught the model the sequence relative information, NTK-aware Scaled RoPE can achieve good performance in longer contexts without fine-tuning.

