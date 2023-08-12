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

**(eq1)**          $$\lfloor\dfrac{n}{\beta^{m-1}}\rfloor \mod \beta $$

If we represent it as a RoPE vector:

**(eq2)**          $$p_n = [\text{cos}\theta_1, \text{sin}\theta_1, \text{cos}\theta_2, \text{sin}\theta_2, …, \text{cos}\theta_{d/2}, \text{sin}\theta_{d/2}]$$

where $$\theta_m = \dfrac{n}{\beta^{m-1}}$$, $$\beta= 10000^{2/d}$$

We have successfully demonstrated that the NTK Scale RoPE exhibits extrapolation in the high-frequency dimension (for a large value of m), whereas it shows interpolation in the low-frequency dimension (for a small value of m). Since a densely interpolated dimension can harm the Language Model's (LLM) to accurately compare relative positions, the NTK Scale RoPE successfully mitigates the comparison confusion posed by extrapolation from a base conversion perspective, and ensure each dimension is not too crowded. This approach significantly benefits LLMs that rely on relative positional cues to understand context, enabling them to effectively expand their contextual understanding over pretrained max sequence length without fine-tuning. 

>  from translator: If you feel confused about how NTK Scale RoPE combines both interpolation and extrapolation together, I strongly suggest you read the [part 1](https://normxu.github.io/Rethinking-Rotary-Position-Embedding/)

Now let’s review **eq2**, notice that cos and sin share the same rotation frequency, which means RoPE encodes n with a base of $$\beta$$ into $$d/2$$ digits. If we want to extend the context length by $$k$$, the intuitive idea is to scale the $$\beta$$ to $$\beta \lambda$$, then:

$$\lambda^{d/2}=k \Rightarrow \lambda=k^{2/d}$$

Then, the RoPE becomes:

**(eq3)**        $$p_n = [\text{cos}\theta_1, \text{sin}\theta_1, \text{cos}\theta_2, \text{sin}\theta_2, …, \text{cos}\theta_{d/2}, \text{sin}\theta_{d/2}]$$

where $$\theta_m = \dfrac{n}{(\beta\lambda)^{m-1}}$$, $$\beta= 10000^{2/d}$$, $$ \lambda=k^{2/d}$$

This is how we implement NTK-RoPE.

However, back to **eq1**, we can see that if we want to encode $$n$$ with a base of $$\beta \lambda$$, the **eq1** should be:

**(eq4)**        $$\lfloor\dfrac{n}{(\beta\lambda)^{m-1}}\rfloor \mod (\beta\lambda) $$

Therefore, our derivation in **eq2** and **eq3** has flaws, besides replacing the $$\dfrac{n}{\beta^{m-1}}$$ with $$\dfrac{n}{(\beta\lambda)^{m-1}}$$, the $$\mod$$ needs to scale up its period by $$\lambda$$ as well, then the corrected Scaled RoPE should be:

**(eq5)**        $$p_n = [\text{cos}\theta_1, \text{sin}\theta_1, \text{cos}\theta_2, \text{sin}\theta_2, …, \text{cos}\theta_{d/2}, \text{sin}\theta_{d/2}]$$

where $$\theta_m = \dfrac{n}{\lambda(\beta\lambda)^{m-1}}$$, $$\beta= 10000^{2/d}$$, $$ \lambda=k^{2/d}$$

In the following context, we denote **eq3** as **NTK-RoPE-old**, and **eq5** as **NTK-RoPE-fixed**.


