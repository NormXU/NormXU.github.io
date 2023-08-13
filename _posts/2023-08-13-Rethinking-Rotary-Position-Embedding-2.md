---
layout: post
title: "Expand the Context Length with RoPE, Part 2 -- Further Research about β-Based Encoding"
categories: LLM
---

> Translated from the [post](https://kexue.fm/archives/9676), originally written in Chinese by Su, Jianlin
>
> Translated by Norm Inui  WIP 🚧

In our previous [post](https://normxu.github.io/Rethinking-Rotary-Position-Embedding/), we interpret RoPE using a β-based encoding and demonstrated why NTK-aware Scaled RoPE can extend the context length without the need for fine-tuning. Viewing position encoding through the lens of β-based encoding indeed offers me some fresh insights and inspiration.

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

Therefore, our derivation in **eq2** and **eq3** has flaws, besides replacing the $$\dfrac{n}{\beta^{m-1}}$$ with $$\dfrac{n}{(\beta\lambda)^{m-1}}$$, the $$\mod$$ needs to scale up its period by $$\lambda$$ as well, then the corrected Scaled RoPE should be:

$$ \begin{equation} p_n = [\text{cos}\theta_1, \text{sin}\theta_1, \text{cos}\theta_2, \text{sin}\theta_2, …, \text{cos}\theta_{d/2}, \text{sin}\theta_{d/2}] \end{equation} $$

where $$\theta_m = \dfrac{n}{\lambda(\beta\lambda)^{m-1}}$$, $$\beta= 10000^{2/d}$$, $$ \lambda=k^{2/d}$$

In the following context, we denote **eq3** as **NTK-RoPE-old**, and **eq5** as **NTK-RoPE-fixed**.


### Encoding with a mixture of base

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

$$ \begin{equation} \lfloor\dfrac{n}{\beta^{m-1}(\lambda_1\lambda_2…\lambda_m)}\rfloor \mod (\beta\lambda_m) \end{equation} $$

$$\theta_m = \dfrac{n}{\beta^{m-1}(\lambda_1\lambda_2…\lambda_m)}$$, $$\beta = 10000^{2/d}$$

According to the goal to ensure lower digits hold a larger range of data and to extend the context length by a scale factor $$k$$, **eq 7** is subject to the conditions

$$ \lambda_1\lambda_2…\lambda_m = k, \lambda_1 \ge \lambda_2 \ge … \ge \lambda_{d/2} \ge 1$$

Given these two conditions, one possible solution is:

$$ \lambda_1\lambda_2…\lambda_m = \text{exp}(am^b)$$,  where $$a \ge 0$$, $$b \le 1$$

>  from translator: The original post doesn't cover any proof of this statement, please check Appendix for the proof I derive



When $$b=1$$, 




### Appendix

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












