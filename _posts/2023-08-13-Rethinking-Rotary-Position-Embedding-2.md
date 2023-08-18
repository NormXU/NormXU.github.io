---
layout: post
title: "Expand the Context Length with RoPE, Part 2 -- Further Research about β-Based Encoding"
categories: LLM
---

> Translated from the [post](https://kexue.fm/archives/9706), originally written in Chinese by Su, Jianlin
>
> Translated by Norm Inui

### TL; DR

- NTK-Scale RoPE has flaw
- Introduce a mixture-of-based encoding method, which can significantly enhance LLM performance beyond its pretraining max length, without the need for fine-tuning
- Introduce a scale factor $$\log n$$ for attention calculation, which can be incorporated either during the pretraining phase or directly applied to an off-the-shell LLM

In [part 1](https://normxu.github.io/Rethinking-Rotary-Position-Embedding/), we interpret RoPE using a β-based encoding and demonstrated why NTK-aware Scaled RoPE can extend the context length without the need for fine-tuning. Viewing position encoding through the lens of β-based encoding indeed offers me some fresh insights and inspiration.

### Modification to NTK
Suppose we encode integer $$n$$ in the $$\beta$$-base, and $$m$$ is the digit of the representation counting from the right.

$$ \begin{equation} \lfloor\dfrac{n}{\beta^{m-1}}\rfloor \mod \beta \end{equation} $$

If we represent it as a RoPE vector:

$$ \begin{equation} p_n = [\text{cos}\theta_1, \text{sin}\theta_1, \text{cos}\theta_2, \text{sin}\theta_2, …, \text{cos}\theta_{d/2}, \text{sin}\theta_{d/2}] \end{equation}$$

where $$\theta_m = \dfrac{n}{\beta^{m-1}}$$, $$\beta= 10000^{2/d}$$

We have successfully demonstrated that the NTK Scale RoPE exhibits extrapolation in the high-frequency dimension (for a large value of m), whereas it shows interpolation in the low-frequency dimension (for a small value of m). Since a densely interpolated dimension can harm the Language Model's (LLM) to accurately compare relative positions, the NTK Scale RoPE successfully mitigates the comparison confusion posed by extrapolation from a base conversion perspective, and ensure each dimension is not too crowded. This approach significantly benefits LLMs that rely on relative positional cues to understand context, enabling them to effectively expand their contextual understanding over pretrained max sequence length without fine-tuning. 

>  from translator: If you feel confused about how NTK Scale RoPE combines both interpolation and extrapolation together, I strongly suggest you read the [part 1](https://normxu.github.io/Rethinking-Rotary-Position-Embedding/)

Now let’s review **eq2**, notice that cos and sin share the same rotation frequency, which means RoPE encodes n with a base of $$\beta$$ into $$d/2$$ digits. If we want to extend the context length by $$k$$, the intuitive idea is to scale the $$\beta$$ to $$\beta \lambda$$, then:

$$\lambda^{d/2}=k \Rightarrow \lambda=k^{2/d}$$

Then, the RoPE becomes:

$$ \begin{equation} p_n = [\text{cos}\theta_1, \text{sin}\theta_1, \text{cos}\theta_2, \text{sin}\theta_2, …, \text{cos}\theta_{d/2}, \text{sin}\theta_{d/2}] \end{equation} $$

where $$\theta_m = \dfrac{n}{(\beta\lambda)^{m-1}}$$, $$\beta= 10000^{2/d}$$, $$ \lambda=k^{2/d}$$

This is how we implement NTK-RoPE.

However, back to **eq1**, we can see that if we want to encode $$n$$ with a base of $$\beta \lambda$$, the **eq1** should be:

$$ \begin{equation} \lfloor\dfrac{n}{(\beta\lambda)^{m-1}}\rfloor \mod (\beta\lambda) \end{equation} $$

Therefore, our derivation from **eq2** to **eq3** has flaws, besides replacing the $$\dfrac{n}{\beta^{m-1}}$$ with $$\dfrac{n}{(\beta\lambda)^{m-1}}$$, the $$\text{mod}$$ needs to scale up its period by $$\lambda$$ as well, then the corrected Scaled RoPE should be:

$$ \begin{equation} p_n = [\text{cos}\theta_1, \text{sin}\theta_1, \text{cos}\theta_2, \text{sin}\theta_2, …, \text{cos}\theta_{d/2}, \text{sin}\theta_{d/2}] \end{equation} $$

where $$\theta_m = \dfrac{n}{\lambda(\beta\lambda)^{m-1}}$$, $$\beta= 10000^{2/d}$$, $$ \lambda=k^{2/d}$$

In the following context, we denote **eq3** as **NTK-RoPE-old**, and **eq5** as **NTK-RoPE-fixed**.


### Why a mixture of base is necessary

If we can encode an integer in $$\beta$$ base, how about generalizing to a mixed-based encoding where each digit is encoded in a different base? Just like the time system we daily use, 60 seconds make up 1 minute, 60 minutes equal 1 hour, 24 hours is 1 day, and 7 days amount to 1 week. Here, the numbers 60, 60, 24, and 7 can be regarded as different encoding bases. In essence, any timestamp can be encoded into seconds, minutes, hours, days, and weeks with the mixed-based system.
Counted from right to left, the first digit is encoded in $$\beta_1$$, the second digit is in $$\beta_2$$, and the third is in $$\beta_3$$, …. The $$m$$th digit of an integer $$n$$ can then be represented as:

$$ \begin{equation} \lfloor\dfrac{n}{\beta^{1}\beta^{1}...\beta^{m-1}}\rfloor \mod \beta_m \end{equation} $$

Since RoPE is a relative position encoding, it can be viewed as a specific instance of the Toeplitz matrix, which looks like this (given our discussion mainly focuses on language models, the top-right part of the matrix is trimmed to fit the page).

$$
\begin{pmatrix}
0 \\
1 & 0 &  \\
2 & 1 & 0 \\
3 & 2 & 1  & 0\\ 
4 & 3 & 2 & 1 & 0\\
5 & 4 & 3 & 2 & 1 & 0\\
6 & 5 & 4 & 3 & 2 & 1 & 0\\

\end{pmatrix}
$$

Upon the matrix, it is evident that the distribution of relative position encoding is not uniform! The 0 is the most frequent, followed by 1, 2, and so on. In other words, as $$n$$ grows larger, its appearance becomes less frequent. This suggests that, as a form of $$\beta$$-base encoding, the higher bits of RoPE might be under-trained. This implies that the generalization capability of the higher bits might be inferior to the lower bits. As mentioned, NTK-RoPE mitigated the confusion introduced by extrapolation across all bits uniformly. However, if our hypothesis holds, this strategy might not be optimal. Lower bits can be more robust than higher bits and can hold a larger data range than the higher bits. Inspired by the timestamp encoding system, we should redesign RoPE with a mix-based encoding system.
### Encoding with a mixture of bases
To be specific, we extend the context length by $$k$$ with a mixture of bases, $$\beta_1$$, $$\beta_2$$, $$...$$, $$\beta_{d/2}$$, where $$\beta_m = \beta\lambda_m$$

Thus, **eq4** shold be be written as:

$$ \begin{equation} \lfloor\dfrac{n}{\beta^{m-1}(\lambda_1\lambda_2…\lambda_{m-1})}\rfloor \mod (\beta\lambda_m) \end{equation} $$

where $$\theta_m = \dfrac{n}{\beta^{m-1}(\lambda_1\lambda_2…\lambda_m)}$$, $$\beta = 10000^{2/d}$$

According to the goal to ensure lower digits hold a larger range of data and to extend the context length by a scale factor $$k$$, **eq 7** is subject to the conditions

$$ \lambda_1\lambda_2…\lambda_m = k$$ and  $$\lambda_1 \ge \lambda_2 \ge … \ge \lambda_{d/2} \ge 1$$

Given these two conditions, one possible solution is:

$$ \lambda_1\lambda_2…\lambda_m = \text{exp}(am^b)$$,  where $$a \ge 0$$, $$b \le 1$$

>  from translator: The original post doesn't cover any proof of this statement, please check Appendix for the proof I derive



When $$b=1$$,  $$\lambda_1 = \lambda_2 = … = \lambda_{d/2} > 1$$, we denote as "NTK-RoPE-fixed"; 

when $$b=0$$, $$\lambda_1 = \lambda_2 = … = \lambda_{d/2} = 1$$, this exactly meets the definition of “Positional Interpolation (PI)”

Given one of the constrains we mention above: 

$$\lambda_1 \lambda_2 … \lambda_{d/2} =k$$

We can derive:

$$a(\dfrac{d}{2})^b = \log k$$

$$b=0.625$$ is an empirical value that can achieve optimal performance in an expanded long context; (Optimal values may vary across models, feel free to tune it), and we denoted this method as NTK-RoPE-mixed.

## Experiment
We follow the same experiment setup as part 1 and compare the NTK-RoPE-mixed and NTK-RoPE-fixed in an extended context.

**Table 1**

| context length            | 512(trained) | 4096 (repeated text) | 4096 (non-repeated text) |
| ------------------------- | ------------ | -------------------- | ------------------------ |
| Baseline                  | 49.41%       | 24.17%               | 23.16%                   |
| Baseline-$$\log n$$       | 49.40%       | 24.60%               | 24.02%                   |
| PI-RoPE                   | 49.41%       | 15.04%               | 13.54%                   |
| PI-RoPE-$$\log n$$        | 49.40%       | 14.99%               | 16.51%                   |
| NTK-RoPE                  | 49.41%       | 51.28%               | 39.27%                   |
| NTK-RoPE-$$\log n$$       | 49.40%       | 61.71%               | 43.75%                   |
| NTK-RoPE-fixed            | 49.41%       | 51.86%               | 39.61%                   |
| NTK-RoPE-$$\log n$$-fixed | 49.40%       | 62.85%               | 44.14%                   |
| NTK-RoPE-mixed            | 49.41%       | 53.09%               | 40.12%                   |
| NTK-RoPE-$$\log n$$-mixed | 49.40%       | **68.91%**           | **45.41%**               |

From the **Table 1**, we can clearly see when compared to the "NTK-RoPE-old" and "NTK-RoPE-fixed," the mixture-of-base "NTK-RoPE-mixed" shows a significant accuracy improvement without fine-tuning. This effectively provides a 'free lunch' approach to enhance LLM performance in a longer context. In addition, the table shows the scale factor $$\log n$$ can benefit as well. But this trick requires $$\log n$$ to be inserted into attention during the pre-training phase, unaffordable and expensive. 

Can models like LLaMA leverage this technique without the need for pre-training? Based on my experiments, a compromised way is to apply the $$\log n$$ factor only to the attention beyond the pretraining length:

$$\max(1, \log_{\text{maxlen}}n)$$ , where $$\text{maxlen}$$  is the max sequence length during pretraining phase​;



For LLaMA-1, it is $$2048$$, and for LLaMA-2, it is $$4096$$; we can scale the attention of an off-the-shelf model on text that exceeds its $$\text{maxlen}$$


> from translator: it is simple to implement this log trick in LLaMA self-attention, see Appendix for more details.


| context length              | 512(trained) | 4096 (repeated text) | 4096 (non-repeated text) |
|-----------------------------|--------------|----------------------|--------------------------|
| NTK-RoPE-fixed              | 49.41%       | 51.86%               | 39.61%                   |
| NTK-RoPE-$$\log n^*$$-fixed | 49.41%       | 55.94%               | 41.11%                   |
| NTK-RoPE-mixed              | 49.41%       | 53.09%               | 40.12%                   |
| NTK-RoPE-$$\log n^*$$-mixed | 49.41%       | **59.11%**           | **42.38%**               |

**Table 2:** $$*$$ denotes we only apply $$\log n$$ on text beyond pretraining max length

We can see from **Table 2**, $$\log n$$ can still enhance performance even without adding it at pretraining phase.  In conclusion, if you are ready to start a pretraining, I suggest you consider incorporated this trick in your network; If you don't want to train at all, this trick can also benefit performance on long context.



------

### Appendix

#### 1. Proof
>  Suppose $$ \lambda_1\lambda_2…\lambda_m = \text{exp}(am^b)$$
>
> We claim that : When $$a \ge 0$$, $$b \le 1$$, then $$\lambda_1 \ge \lambda_2 \ge … \ge \lambda_{d/2} \ge 1$$


__Proof__:
According to the statement, 

when $$m=1$$:  $$\lambda_1 = \text{exp}(a)$$,

when $$m>1$$:

$$\begin{split}
\lambda_m &= \dfrac{\text{exp}(am^b)}{\text{exp}(a(m-1)^b)} \\
&=\text{exp}(a[m^b-(m-1)^b])
\end{split}$$

Therefore, when $$a \ge 0$$, we have $$\lambda_m \ge \text{exp}(0) = 1$$



Similarly, if the assumption is true, we can derive:

$$\lambda_m =\text{exp}(a[m^b-(m-1)^b])$$

$$\lambda_{m+1} =\text{exp}(a[(m+1)^b-m^b])$$

Since $$\text{exp}()$$ is a monotonically increasing function, suppose $$\lambda_m \ge \lambda_{m+1}$$

According to [Binomial Theorem](http://hyperphysics.phy-astr.gsu.edu/hbase/alg3.html), we can derive:

$$\begin{split}
\text{exp}(a[m^b - (m-1)^b]) &\ge \text{exp}(a[(m+1)^b - m^b)])\\
\Rightarrow m^b - (m-1)^b &\ge (m+1)^b - m^b\\
2m^b &\ge(m+1)^b + (m-1)^b\\
2m^b &\ge (m^b+bm^{b-1} + \dfrac{b(b-1)}{2}m^{b-2 }+ \dfrac{b(b-1)(b-2)}{6}m^{b-3 }...) + (m^b - bm^{b-1}+\dfrac{b(b-1)}{2}m^{b-2 }- \dfrac{b(b-1)(b-2)}{6}m^{b-3 } ...)\\
2m^b &\ge 2m^b + 2 (\dfrac{b(b-1)}{2}m^{b-2} + \dfrac{b(b-1)(b-2)(b-3)}{24}m^{b-4} + ...)\\
2m^b &\ge 2m^b + 2\sum_{k=2,4,6...}\dfrac{b!}{(b-k)!k!}m^{b-k}\\
\end{split}$$

Thus, only when $$b \le 1$$,  $$ \sum_{k=2,4,6...} \dfrac{b!}{(b-k)!k!}m^{b-k} \le 0$$

In conclusion, we can conclude the assumption stays true.

#### 2. Minor changes in LlamaAttention
```python
class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        ...
        self.max_position_embeddings = config.max_position_embeddings
        ...

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        ...
        bsz, q_len, _ = hidden_states.size()
        ...
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # ---- + new code
        query_states = max(1, math.log(q_len, self.max_position_embeddings)) * query_states
        # -------
        ...
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        ...
```













