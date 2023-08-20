---
layout: post
title: "Expand the Context Length with RoPE, Part 3 -- Unlocking the Unlimited Extrapolation Potential with ReRoPE"
categories: LLM
---

> Translated from the [post](https://kexue.fm/archives/9708) and [post](https://kexue.fm/archives/9728), originally written in Chinese by Su, Jianlin
>
> Translated by Norm Inui

### TL; DR

- Introduce ReRoPE (Rectified RoPE), a post-processing optimization approach for RoPE.

- Experimental results reveal that ReRoPE's extrapolation capabilities, without fine-tuning, significantly surpass the previous NTK-aware Scaled RoPE

- ReRoPE appears to consistently perform well across any length

- ReRoPE significantly reduces inference speed. However, training with ReRoPE and inferring with RoPE can benefit the extrapolation ability of LLMs without sacrificing throughput in inference

- Code is available [here](https://github.com/bojone/rerope)

  


In a previous blog, I introduced the mixture-of-base encoding and believed we might have maxed out the potential of RoPE regarding extrapolation. It appeared we might need to explore other avenues for any further enhancement on context length. However, I remembered a method I previously set aside due to its complexity. Since we have run out of ideas, why not revisit it and see what can we learn from it? Sometimes, 'The best solution is the only solution'.

Surprisingly, even though this method will increase time complexity, the experimental results are promising and even shows a potential to unlock the unlimited extrapolation ability of the language model. I can’t wait to write this article and share the method with you. Due to its similarity with the ReLU activation function, I've named this method **ReRoPE (Rectified Rotary Position Embeddings)**

### Background

We explain in the previous blog that although RoPE is regarded as an absolute position embedding, it can inject relative positional information into the Attention matrix with a Toeplitz matrix.

$$ \begin{equation} \begin{pmatrix}
0 &  &  &  &  &  &  & & \\ 
1 & 0 &  &  &  &  &  & & \\ 
2 &  1& 0 &  &  &  &  & & \\ 
3  &  2& 1 & 0 &  &  &  & & \\ 
\ddots  & 3 & 2 & 1 & 0 &  &  & & \\ 
\ddots  & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & & \\ 
L-2 & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \\ 
L-1 & L-2 & \ddots & \ddots & \ddots & 3 & 2 & 1 & 0
\end{pmatrix} \end{equation} $$

$$L$$ is the input sequence length. When $$L$$ is greatly larger than the pretrained max sequence length, the model typically exhibits poor extrapolation because it hasn't been adequately trained on longer sequences.

The Position Interpolation modifies the Toeplitz matrix as:

$$ \begin{equation} \begin{pmatrix}
0 &  &  &  &  &  &  & & \\ 
1/k & 0 &  &  &  &  &  & & \\ 
2/k &  1/k& 0 &  &  &  &  & & \\ 
3/k  &  2/k& 1/k & 0 &  &  &  & & \\ 
\ddots  & 3/k & 2/k & 1/k & 0 &  &  & & \\ 
\ddots  & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & & \\ 
(L-2)/k & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \\ 
(L-1)/k & (L-2)/k & \ddots & \ddots & \ddots & 3/k & 2/k & 1/k & 0
\end{pmatrix} \end{equation} $$

Position Interpolation (PI) ensures that the maximum relative position does not exceed the training length by tuning $$k$$, therefore, it is free from any extrapolation on dimension. However, it makes each dimension carry more position information. Consequently, a few fine-tuning steps are necessary to get the model to adapt to the “crowded” dimension. Neural networks are often better at interpolation rather than extrapolation, just consider extrapolation as adding an extra dimension, while interpolation inserts more data into the already trained dimension. Intuitively, neural networks struggle with extrapolation. Therefore, PI is an efficient method to extend the context length with minimal fine-tuning.

As for the NTK-aware Scaled RoPE, it cleverly distributes the “crowded” dimension across every dimension. As a result, it can get even better perplexity value without fine-tuning. However, as we mention above,  neural networks struggle with extrapolation, which explains why an extended long context model can't quite match a pretrained model with an identical max sequence length.

### Combine Interpolation and Extrapolation

Let’s revisit extending methods we have through the lens of the definition of the locality. By mentioning 'locality,' we try to describe a preference of a language model when it predicts the next token, it heavily relies on the nearing tokens. Extrapolation preserves this locality since position encoding near 0s of the Toeptile matrix is unchanged, but its performance suffers due to the introduction of position encodings beyond the trained length. Although position interpolation doesn't introduce extrapolated position encodings, it harms the locality since position encoding near 0 is compressed to $$1/k$$, leading to necessary fine-tuning. On the other hand, NTK-aware Scaled RoPE combines the advantages of both methods by "high-frequency extrapolation and low-frequency interpolation". This ensures the preservation of locality without introducing new position encoding, yielding good results even without fine-tuning.
Besides NTK Scaled RoPE, is there any other method that can realize both extrapolation and interpolation? The answer is **YES**.
Suppose we set a window with size $$w$$, the interval between positions inside the window is $$1$$, while the interval outside the window is $$1/k$$, the Toepiltz matrix is shown as:

$$ \begin{equation}\begin{pmatrix} 
\color{red}{0} & \\ 
\color{red}{1} & \color{red}{0} & \\ 
\color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{red}{\tiny{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{green}{w} & \color{red}{\tiny{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{green}{\tiny{w + \frac{1}{k}}} & \color{green}{w} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{green}{\tiny{w + \frac{2}{k}}} & \color{green}{\tiny{w + \frac{1}{k}}} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{green}{\ddots} & \color{green}{\tiny{w + \frac{2}{k}}} & \color{green}{\ddots} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \\ 
\color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\tiny{w + \frac{2}{k}}} & \color{green}{\tiny{w + \frac{1}{k}}} & \color{green}{w} & \color{red}{\tiny{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{green}{\tiny{w + \frac{L-1-w}{k}}} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\tiny{w + \frac{2}{k}}} & \color{green}{\tiny{w + \frac{1}{k}}} & \color{green}{w} & \color{red}{\tiny{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\end{pmatrix}\end{equation} $$

Numbers in $$\color{red} \text{red}$$ are within the sliding window, in $$ \color{green} \text{green}$$ are outside the sliding window.


By adjusting $$k$$, we can ensure $$w < \text{max pretraining length}$$, which allows us to maintain locality while keeping the position encoding within the pretraining length. This sliding window approach to the input sequence achieves interpolation outside the window and preserves locality within the window concurrently.

Moreover, when we extend the context length $$\to \infty$$, then $$k \to \infty$$, the matrix can be formulated as:

$$ \begin{equation}\begin{pmatrix} 
\color{red}{0} & \\ 
\color{red}{1} & \color{red}{0} & \\ 
\color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{red}{\tiny{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{green}{w} & \color{red}{\tiny{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{green}{w} & \color{green}{w} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{green}{w} & \color{green}{w} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{green}{\ddots} & \color{green}{w} & \color{green}{\ddots} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \\ 
\color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{w} & \color{green}{w} & \color{green}{w} & \color{red}{\tiny{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\color{green}{w} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{w} & \color{green}{w} & \color{green}{w} & \color{red}{\tiny{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\ 
\end{pmatrix}\end{equation} $$

We can notice the locality can still be preserved within the window.
In conclusion, we can find a relation between **eq(3)**, **eq(4)** and **eq(1)**

$$eq(3) = \text{LeakyReLU}(eq(1))$$ and $$eq(4) = \text{ReLU}(eq(1))$$

### Computation Cost

The concept of incorporating sliding windows into the input sequence is not new today and has been widely used in Attention Bias, like T5-bias, and relative position embedding. Yet, integrating a sliding window with RoPE can increase computation costs. Regarding **eq(3)** and **eq(4)**, since the values in each matrix row don't increase linearly, RoPE needs to encode twice: one for positions within the window and another for those outside. These encodings are then combined together as ReRoPE.

To be specific, we compute attention scores with RoPE position encoding within the window:

 $$a_{i,j}^{(1)} = (R^i q_i)^T(R^j k_j) = q_i^T R^{j-i} k_j$$

$$R$$ is the RoPE rotation matrix, we omit the attention scale factor and softmax for concise. 
Then we compute attention scores outside the window, whose interval between numbers is $$1/k$$. We denote this equation as Leaky ReRoPE:

$$a_{i,j}^{(2)} = (R^{(i-w)/k+w} q_i)^T(R^{j/k} k_j) = q_i^T R^{(j-i+w)/k - w} k_j$$

When $$k \to \infty$$, the equation is simpler:

$$a_{i,j}^{(2)} = (R^{w} q_i)^T k_j = q_i^T R^{w} k_j$$

Let’s combine them together:

$$ \begin{equation}
    a_{i,j}=
    \begin{cases}
      a^{(1)}_{i,j}, &  i -j < w\\
      a^{(2)}_{i,j}, & i -j \ge w
    \end{cases}
  \end{equation} $$

According to the equations, we can notice both ReRoPE and Leaky ReRoPE inevitably require calculating the Attention matrix twice. If you have a more efficient implementation, please feel free to contact me. Moreover, this Attention matrix cannot directly be optimized with the current flash attention implementation, leading to more computational cost.


On the other hand, the non-linear relative positioning means that during autoregressive decoding, only the RoPE keys within the window can be cached. As the sequence length increases, keys that were once inside the window shift outside, and they need to be recomputed and appended with the cached keys for decoding tokens beyond the maximum sequence length. This process amplifies computation cost during inference. In token-by-token decoding, the query sequence length after the input prompt is merely $$1$$. Unless the prompt exceeds the maximum sequence length, only the keys need to be recalculated.

$$
\begin{equation}a_{i,j} = \left\{\begin{aligned} 
&\boldsymbol{q}_i^{\top}\left(\boldsymbol{\mathcal{R}}^{\max(j-i,-w)}\boldsymbol{k}_j\right), \quad(\text{ReRoPE})\\[8pt] 
&\boldsymbol{q}_i^{\top}\left(\boldsymbol{\mathcal{R}}^{\max(j-i,(j-i+w)/k-w)}\boldsymbol{k}_j\right), \quad(\text{Leaky ReRoPE}) 
\end{aligned}\right.\end{equation}
$$



However, using ReRoPE/Leaky ReRoPE  in LLMs is computationally intensive. While it enables LLMs to process longer extended contexts, the input length during inference often exceeds the pretrained maximum sequence length. This results in significant latency, making it challenging for real-time applications.

What if we train with ReRoPE/Leaky ReRoPE but infer using standard RoPE? ReRoPE/Leaky ReRoPE serves as an extrapolation method for the ideal goal: "Train Short, Test Long". Training an LLM with ReRoPE/Leaky ReRoPE certainly demands more time; however, this slowdown during training is acceptable when compared to the potential drop in inference speed.

To be specific, when a model is trained with RoPE and its context length is extrapolated using LeakyReRoPE, the interval outside the window is  $$1$$ during training and $$\dfrac{1}{k} < 1$$ during inference. When swapping the embedding strategy, the model is trained with an interval greater than  $$1$$ but infers with an interval of $$1$$. This means that, during inference, LeakyReRoPE behaves like RoPE. We refer to this approach as InvLeaky ReRoPE (Inverse Leaky ReRoPE). **Table 5** demonstrates the effectiveness of this strategy. Since the embedding behaves like RoPE at inference, optimization techniques like FlashAttenion can be seamlessly integrated. After experimenting the different hyperparameters, we propose the empirical optimal parameter rule:

expanding scale: 

$$b = \dfrac{\text{expanded\_len}}{\text{max\_seq\_len}}$$

number interval outside window: 

$$k=\dfrac{1}{2 b}$$

window size: 

$$w = \dfrac{\text{max\_seq\_len}}{4}$$

In **Table 5**, the model has 100M parameters, with a training length of 512. The training time for every 1,000 steps grows from $$330$$ seconds to $$350$$ seconds, an increase less than $$10\%$$. Since the model is a hybrid of Transformer and GAU (Gated Attention Unit), with single-head attention in HAU. As for a multi-head attention LLM, the time increase could be more significant, possibly up to $$50\%$$, but it is still acceptable.



### Ablation Experiments
We follow the same experiment setup as in [part 1](https://normxu.github.io/Rethinking-Rotary-Position-Embedding/) on an 100M [GAU](https://arxiv.org/abs/2202.10447) model. The result is shown below.

| context length                | 512(trained) | 4096 (repeated text) | 4096 (non-repeated text) |
| ----------------------------- | ------------ | -------------------- | ------------------------ |
| Baseline                      | 49.41%       | 24.17%               | 23.16%                   |
| Baseline-$$\log n$$           | 49.40%       | 24.60%               | 24.02%                   |
| PI-RoPE                       | 49.41%       | 15.04%               | 13.54%                   |
| PI-RoPE-$$\log n$$            | 49.40%       | 14.99%               | 16.51%                   |
| NTK-RoPE                      | 49.41%       | 51.28%               | 39.27%                   |
| NTK-RoPE-$$\log n$$           | 49.40%       | 61.71%               | 43.75%                   |
| NTK-RoPE-fixed                | 49.41%       | 51.86%               | 39.61%                   |
| NTK-RoPE-$$\log n^{*}$$-fixed | 49.41%       | 55.94%               | 41.11%                   |
| NTK-RoPE-$$\log n$$-fixed     | 49.40%       | 62.85%               | 44.14%                   |
| NTK-RoPE-mixed                | 49.41%       | 53.09%               | 40.12%                   |
| NTK-RoPE-$$\log n^{*}$$-mixed | 49.41%       | 59.11%               | 42.38%                   |
| NTK-RoPE-$$\log n$$-mixed     | 49.40%       | 68.91%               | 45.41%                   |
| ReRoPE-w256                   | 49.41%       | 77.90%               | 48.48%                   |
| ReRoPE-w256-$$\log n^{*}$$    | 49.41%       | 82.40%               | 48.85%                   |
| ReRoPE-w256-$$\log n$$        | 49.40%       | **85.12%**           | **49.07%**               |

**Table 1**: the average accuracy of predicting next token to match the ground-truth next token given previous context. The experiment is based on a hybrid Transformer-GAU (Gated Attention Unit) model with a size of 100M parameters. $$\log n$$ indicates we add the scale factor $$\log n$$ at pretraining stage; $$\log n^{*}$$ denotes we apply the scale factor $$\log n$$ is applied to the attention matrix only for text exceeding the max sequence length, without any pretraining ; $$w256$$ denotes $$w=256$$



| context length             | 512(trained) | 4096 (repeated text) | 4096 (non-repeated text) |
|----------------------------|--------------| -------------------- | ------------------------ |
| ReRoPE-w64                 | 49.41%       |     69.39%                 |      45.19%                    |
| ReRoPE-w64-$$\log n^{*}$$  | 49.41%       |     78.58%                 |       47.42%                   |
| ReRoPE-w64-$$\log n$$      | 49.40%       |     84.38%                 |      48.14%                    |
| ReRoPE-w128                | 49.41%       |     76.11%                 |       47.82%                   |
| ReRoPE-w128-$$\log n^{*}$$ | 49.41%       |     82.28%                 |        48.78%                  |
| ReRoPE-w128-$$\log n$$     | 49.40%       |     **85.47%**             |        48.87%                  |
| ReRoPE-w256                | 49.41%       |     77.90%                 |      48.48%                    |
| ReRoPE-w256-$$\log n^{*}$$ | 49.41%       |     82.40%                 |      48.85%                    |
| ReRoPE-w256-$$\log n$$     | 49.40%       |    85.12%                  |       **49.07%**               |
| ReRoPE-w384                | 49.41%       |   70.72%                   |       48.15%                   |
| ReRoPE-w384-$$\log n^{*}$$ | 49.41%       |    76.42%                  |       48.31%                   |
| ReRoPE-w384-$$\log n$$     | 49.40%       |    83.24%                  |      48.62%                    |
| ReRoPE-w512                | 49.41%       |    7.09%                  |        8.25%                  |
| ReRoPE-w512-$$\log n^{*}$$ | 49.41%       |     7.08%                 |          8.25%                |
| ReRoPE-w512-$$\log n$$     | 49.40%       |      15.84%                |        10.83%                  |

**Table 2**: Ablation on window size of ReRoPE; experiment setting is the same as **Table 1**

From **Table 2**, we can learn $$w$$ is robust to the performance; the optimal **w** is $$1/4$$ to $$1/2$$ of the pretraining max sequence length.



| context length                   | 512(trained) | 4096 (repeated text) | 4096 (non-repeated text) |
|----------------------------------| ------------ | -------------------- | ------------------------ |
| ReRoPE-w128-$$\log n$$           |  49.40%            |       **85.47%**           |  48.87%                        |
| Leaky-ReRoPE-w128-k64-$$\log n$$ |   49.40%           |      85.29%                |  48.96%                        |
| Leaky-ReRoPE-w128-k32-$$\log n$$ |   49.40%           |      85.31%                |    49.03%                      |
| Leaky-ReRoPE-w128-k16-$$\log n$$ |   49.40%           |      85.15%                |    **49.10%**                  |
| Leaky-ReRoPE-w128-k8-$$\log n$$  |   49.40%           |       80.00%               |     48.11%                     |
| ReRoPE-w256-$$\log n$$           |   49.40%           |       85.12%               |    49.07%                      |
| Leaky-ReRoPE-w256-k64-$$\log n$$ |   49.40%           |       84.60%               |     49.03%                     |
| Leaky-ReRoPE-w256-k32-$$\log n$$ |   49.40%           |      84.30%                |       48.97%                   |
| Leaky-ReRoPE-w256-k16-$$\log n$$ |   49.40%           |      83.59%                |       48.87%                   |
| Leaky-ReRoPE-w256-k8-$$\log n$$  |   49.40%           |      69.80%                |       45.72%                   |

**Table 3**: Ablation on interval $$k$$ of Leaky ReRoPE and ReRoPE; experiment setting is the same as **Table 1**

From **Table 3**: Fine-tuned Leaky ReRoPE, as a generalization of ReRoPE, might slightly surpass ReRoPE's performance, though the gains are minimal. When setting $$k$$ to a finite value, there's an inherent limitation on the maximum length it can manage. Since predicting the length LLM will generate in advance is impossible, we usually set a large value for $$k$$. However, even with a sufficiently large $$k$$, a siginificant long input could severely degrade performance due to position encoding surpassing the trained length. While ReRoPE doesn't have such an issue. In practical applications, fine-tuned Leaky ReRoPE may not be as universally adaptable as ReRoPE.



| context length | 4096(trained) | 8192   | 16384  |
| -------------- | ------------- | ------ | ------ |
| RoPE           | 1.4967        | 8.8615 | —      |
| NTK-RoPE       | 1.6081        | 1.5417 | 1.5163 |
| ReRoPE         | 1.4996        | 1.4267 | 1.4001 |

**Table 4**: Experiments on LLaMa-2-13B, the value represent loss; smaller is better.

ReRoPE effectively achieves near-optimal results, aligning with our intuition that "longer context results in lower loss", given that an extended context should benefit LLM comprehension ability. Furthermore, I evaluated the chat capabilities of the LLAMA2-13b model, open-source by [OpenBuddy](https://huggingface.co/OpenBuddy), and found its performance satisfying with an input length up to 20k tokens.




| context length                | 512(trained) | 4096 (repeated text) | 4096 (non-repeated text) |
| ----------------------------- | ------------ | -------------------- |--------------------------|
| Baseline                      | 49.41%       | 24.17%               | 23.16%                   |
| Baseline-$$\log n$$           | 49.40%       | 24.60%               | 24.02%                   |
| NTK-RoPE-fixed                | 49.41%       | 51.86%               | 39.61%                   |
| NTK-RoPE-$$\log n^{*}$$-fixed | 49.41%       | 55.94%               | 41.11%                   |
| NTK-RoPE-$$\log n$$-fixed     | 49.40%       | 62.85%               | 44.14%                   |
| NTK-RoPE-mixed                | 49.41%       | 53.09%               | 40.12%                   |
| NTK-RoPE-$$\log n^{*}$$-mixed | 49.41%       | 59.11%               | 42.38%                   |
| NTK-RoPE-$$\log n$$-mixed     | 49.40%       | 68.91%               | 45.41%                   |
| ReRoPE-w256                   | 49.41%       | 77.90%               | 48.48%                   |
| ReRoPE-w256-$$\log n^{*}$$    | 49.41%       | 82.40%               | 48.85%                   |
| ReRoPE-w256-$$\log n$$        | 49.40%       | **85.12%**           | **49.07%**               |
| InvLeaky ReRoPE-w128-$$\log n$$        | 49.38%       | 82.25%           | 48.32%                   |
| InvLeaky ReRoPE-w128-b8-$$\log n$$        | 49.62%       | 81.15%           | 48.85%                   |

**Table 5**: Experiment setting is the same as **Table 1**； b8: replace the RoPE base from $$10000$$ to $$80000$$; InvLeaky ReRoPE is inferior to ReRoPE, but still promising compared to vanilla NTK-RoPE




The ReRoPE and Leaky ReRoPE codes can be found here. Feel free to play with it.

> **Github: [https://github.com/bojone/rerope](https://github.com/bojone/rerope)**