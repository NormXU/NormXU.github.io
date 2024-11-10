---
layout: post
title: "Cheat Sheet for Mathematical Diffusion"
tags: ["Diffusion"]
---

This is a cheat sheet of all denoising-based diffusion method. No mathematics derivations are included for concise. Check the reference links if you're interested in the derivations. 

[EDM](https://arxiv.org/pdf/2206.00364) introduces a **general denoising equation** that nearly unifies all widely recognized denoising and noise-adding processes within its framework.

## A Unified Perspective

### General Forward Processs

Noise-adding Process is also known as forward process. The general form of this Forward Process is:

$$p(x_t | x_0) = \mathcal{N}(x_t; s(t)x_0, \sigma^2(t) s^2(t) \mathbf{I})$$

This equation describes how noise is gradually added to an image, represented as  $$x_0$$, over time steps $$t$$. The two functions, $$s(t)$$ and $$\sigma(t)$$, control the scale and variance of the noise, determining the trajectory of noise addition as $$x_t$$ moves further from the original image.

### General Reverse Processs

The equation above can be written into a **Stochastic Differential Equation (SDE)** format defined as:

$$ dx_t = f(x_t, t) \, dt + g(t) \, d\mathbf{w} $$

This is the Forward Process written in SDE format.

Here, $$s(t) = e^{\int_{0}^{t} f(r) \, dr}$$ and $$\sigma^2(t) = \int_{0}^{t} \frac{g^2(r)}{s^2(r)} \, dr$$

$$d\mathbf{w}$$ represents a **Wiener process**, where $$\mathbf{w_t} \sim \mathcal{N}(0, t)$$; $$d\mathbf{w} = \sqrt{dt} \; \epsilon$$ , where $$ \epsilon \sim \mathcal{N}(\mu, \sigma^2)$$

Different methods define their own specific functions for $$f(x_t, t)$$ and $$g(t)$$. For example, in diffusion models like **DDPM** and **SMLD** (Score Matching with Langevin Dynamics), $$f(x_t, t)$$ is typically a linear function, $$f(x_t, t) = f(t)x_t$$, where $$f(t)$$ is a time-dependent term that modulates the trajectory of the image's transformation over time.  EDM simply set $$s(t) = 1, ; \sigma(t) = t$$.

Now we have a forward SDE to corrupt a data into noise. To reverse such as process, we have two choices:

```python
flowchart LR
    x_data -->|Forward SDE| --> x_noise
    x_noise -->|Reverse SDE| --> x_data
    x_noise -->|PFODE| --> x_data
```

- Using the **Fokker-Planck equation**, we transform the forward SDE into a reverse SDE.

We skip this part for now.

- Using the **Fokker-Planck equation**, we transform the forward SDE into an **Ordinary Differential Equation (ODE)**, called the **Probability Flow ODE** **(PFPDE)**:

Remember, PFPDE is a deterministic process:

$$d\mathbf{x}_t = \left[ f(t) x_t - \frac{1}{2} g^2(t) \nabla_{x_t} \log p_t(x_t) \right] \, dt$$

Interestingly, this equation lets us denoise without needing to directly solve for $$f(t)$$ and $$g(t)$$. By knowing only $$s(t)$$ and $$\sigma(t)$$, we can effectively denoise and sample high-quality images. 

A common question is: Why isn't there a forward PFODE process?

The answer is simple. PFODE is designed for sampling, not for the forward process. Since PFODE is a fully deterministic process, it cannot model a data distribution without incorporating noise. Our objective is to model the distribution of data, and noise injection is essential for this. Because PFODE cannot introduce stochasticity into the forward process, the neural network is unable to learn any distribution from it.

In practice, instead of explicitly computing the score function $$ \nabla_{x_t} \log p_t(x_t) $$, we approximate it with a neural network $$D_\theta$$. 

$$ \mathrm{d} \mathbf{x}_t = \left[ \left( \frac{\dot{s}(t)}{s(t)} + \frac{\dot{\sigma}(t)}{\sigma(t)} \right) \mathbf{x}_t - \frac{s(t) \, \dot{\sigma}(t)}{\sigma(t)} D_\theta \left( \frac{\mathbf{x}_t}{s(t)} ; \sigma \right) \right] \, dt  $$

The score function can also be derived from a neural network prediction:

$$\nabla_{\mathbf{x}} \log p(\mathbf{x}; \sigma) = \frac{D_{\theta}(\mathbf{x}; \sigma) - \mathbf{x}}{\sigma^2}$$

Now, we have a general PFODE equation that models the forward process. After training a neural network to approximate $$D_\theta (\mathbf{x}_t; \sigma(t))$$, we can sample from noise by simply solving the ODE.

To accurately sample from the distribution, we use **2nd-order Heun's method** rather than the simpler Euler method (1st-order). Heun’s method reduces sampling error by averaging the initial and predicted values of the function, providing a more stable path for denoising:

$$ x_{t + \Delta t} = x_t + \frac{1}{2} (f(x_t, t) + f(x_{t + \Delta t}, t + \Delta t)) \Delta t $$

However, since Heun’s method requires knowing both $$f(x_t, t)$$ and $$f(x_{t + \Delta t}, t + \Delta t)$$, we still use an **Euler step** to make an initial estimate before applying Heun’s correction.

### General Model Design

The score function $$ D_\theta \left( \frac{\mathbf{x}_t}{s(t)} ; \sigma \right) $$ has both a **forward** and a **reverse process**.

- **Forward Process**: The forward process is deterministic and generally involves adding noise to the image data over time. This process is usually designed with a known noise schedule and is entirely predictable based on the given parameters $$ s(t) $$ and $$ \sigma(t) $$.

- **Reverse Process**: The reverse process, however, is probabilistic and requires the use of a **neural network** to predict how the noisy image should be transformed back towards the clean image. 

In the case of **DDPM**, the forward process shows as below:

$$ D_\theta (\mathbf{x}_t; \sigma(t)) \approx \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \, \varepsilon}{\sqrt{\bar{\alpha}_t}} \approx \frac{1}{\sqrt{\bar{\alpha}_t}} \mathbf{x}_t - \frac{\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}} \, \varepsilon_\theta (\mathbf{x}_t, t) $$

In **Score Matching**, the forward process shows as below:

$$ D_\theta (\mathbf{x}; \sigma) \approx \mathbf{x} + \sigma^2 s_\theta (\mathbf{x}; \sigma)$$

In Flow Matching, the forward process shows as below:

$$ D_\theta (\mathbf{x}_t; \sigma(t)) \approx \mathbf{x}_t + (1 - t) v_\theta (\mathbf{x}_t, t)$$

Although DDPM, Score Matching, and Flow Matching have different formulations, they can be unified into one general equation:

$$ D_\theta (\hat{\mathbf{x}}; \sigma) = C_{\text{skip}}(\sigma) \, \hat{\mathbf{x}} + C_{\text{out}}(\sigma) F_\theta \left( C_{\text{in}}(\sigma) \, \hat{\mathbf{x}}; C_{\text{noise}}(\sigma) \right)$$

This equation introduces several new terms, each playing a key role in aligning the three approaches. Let’s break these down:

- **What is $$ \hat{\mathbf{x}} $$?**  
  To unify the input pixel range across the models, we convert the image from the noisy range $$[-s(t), s(t)]$$ to a normalized range of \([-1, 1]\). The term $$ \hat{\mathbf{x}} $$ represents this normalized version of the input image, making it easier to handle across different processes. Given the noise schedule, any image $$ \mathbf{x} = s(t) \hat{\mathbf{x}} $$ will have its pixel range scaled to $$[-s(t), s(t)]$$, and thus $$ \hat{\mathbf{x}} $$ allows us to operate within a consistent range for training.

- **What is $$ C_{\text{in}}(\sigma) $$?**  
  The term $$ C_{\text{in}}(\sigma) $$ scales the input image before it passes through the neural network. In the case of **DDPM** and **Flow Matching**, the input images at different time steps $$ t $$ have ranges from $$[-s(t), s(t)]$$. These terms ensure that the input is correctly scaled before passing into the model’s score network, where $$ s(t) $$ may change depending on the chosen process (e.g., DDPM's $$ s(t) = \sqrt{\bar{\alpha}_t} $$ or FM's $$ s(t) = t $$).

- **$$ F_{\theta}() $$
  
  The neural network can be a U-Net or a DiT. The proxy target can be predicting noise, $$x_0$$ or velocity

Since EDM framework supposes that $$s(t) = 1, ; \sigma(t) = t$$ and defining $$\sigma$$ based on a fixed schedule, we can achieve precise control over noise levels across different steps. A typical noise schedule could be:

$$ \sigma_{i < N} = \left( \sigma_{\text{max}}^{\frac{1}{\rho}} + \frac{i}{N-1} \left( \sigma_{\text{min}}^{\frac{1}{\rho}} - \sigma_{\text{max}}^{\frac{1}{\rho}} \right) \right)^{\rho} \quad \text{and} \quad \sigma_N = 0 $$

With this, the model can handle generation tasks while being independent of the complexities of $$f(t)$$ and $$g(t)$$, focusing only on $$s(t)$$ and $$\sigma(t)$$.

---

## DDPM

### Forward Process

The forward process is given by:

$$x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \varepsilon$$

This can also be written as:

$$x_t = \sqrt{\bar{\alpha_t}} x_{data} + \sqrt{1 - \bar{\alpha_t}} \varepsilon$$

We can express this in the form of a SDE [$$^{\text{ref-Section D}}$$](https://arxiv.org/pdf/2210.02747) :

$$dx_t = f(x_t, t) dt + g(t) d\mathbf{w} = -\frac{1}{2} \beta(t) x_t \;dt + \sqrt{\beta(t)} d\mathbf{w}$$

We can clearly see from the equation that DDPM forward process is a curve motion, where the magnitude and direction of velocity is time-dependent.

### Reverse Process

The reverse process is expressed as:

$$dx_t = \left(f(x_t, t) - g^2(t) \nabla_x \log p(x_t)\right) dt + g(t) d\bar{\mathbf{w}}$$

$$\bar{\mathbf{w}}$$ is a reverse Wiener process.

This process can be solved using any SDE solver you like.

### Loss Type

- Predict Velocity

$$v := \sqrt{\bar{\alpha_t}} \epsilon - \sqrt{1-\bar{\alpha_t}} x_{data}$$

- Convert to Predicted Noise $$\epsilon_{pred}$$ [$$^{\text{ref-Appendix A, page 12}}$$](https://arxiv.org/pdf/2301.11093); [$$^{\text{diffusers impl}}$$](https://github.com/huggingface/diffusers/blob/6a89a6c93ae38927097f5181030e3ceb7de7f43d/src/diffusers/schedulers/scheduling_ddim.py#L416-L429)

$$\epsilon_{pred} = \sqrt{\bar{\alpha_t}} v_{pred} + \sqrt{1-\bar{\alpha_t}} x_t$$

$$\bar{\alpha_t} \text{MSE}(v_{pred}, v) = \text{MSE}(\varepsilon_{pred}, \varepsilon)$$

- Convert to Predict $$x_{data}$$

$$\hat{x_0} = \sqrt{\bar{\alpha_t}} x_t - \sqrt{1-\bar{\alpha_t}} v_{pred} $$

$$(1-\bar{\alpha_t}) \text{MSE}(v_{pred}, v) = \text{MSE}(x_{data}, x_{pred})$$

---

## Score Matching

### Forward Process

The forward process in score matching is:

$$x_t = x_{data} + \sigma_t \varepsilon$$

Here,  $$x_{data} \sim p_{data}(x)$$ , where  $$p_{data}(x) $$ represents the distribution of the training dataset. The noise variance decreases over time:

$$\sigma_1 > \sigma_2 > \sigma_3 > \dots$$

This means the noise variance added to the data gradually decreases. The corresponding ODE is:

$$dx_t = \sqrt{\frac{d\sigma^2_t}{dt}} d\bar{\mathbf{w}}$$

The forward process can be imagined a straight line going from data to noise where the velocity (variance of noise) gradually decreases from large to small.

### Reverse Process

$$d\mathbf{x_t} = -\left(\frac{d[\sigma(t)^2]}{dt} \nabla_{\mathbf{x}} \log p(\mathbf{x_t}) \right) dt + \sqrt{\frac{d[\sigma(t)^2]}{dt}} d\bar{\mathbf{w}}$$

The reverse sampling follows the Langevin equation:

$$x_{t+1} = x_t + \tau \nabla_x \log p(x_t) + \sqrt{2\tau} z$$

where  $$z \sim N(0, I)$$ 

We can see from the sampling equation that although the forward process is linear, the reverse process is stochastic, which makes score-matching sampling hard to hack.

### Loss Type

- Noise Conditional Score Matching [$$^{\text{Theorem 3.3}}$$](https://arxiv.org/pdf/2403.18103)

$$J_{\text{NCSM}}(\theta, \sigma_i) = \sum_i^{L} \lambda_i \;\mathbb{E}_{p(\mathbf{x})} \left[ \frac{1}{2} \left\| s_\theta(x_{data} + \sigma_i \varepsilon) + \frac{\varepsilon}{\sigma_i} \right\|^2 \right]$$

Note this is the loss function when we use score matching forward process to add noise progressively.

If we use DDPM noise scheduler, which means $$x_t = \sqrt{\bar{\alpha_t}} x_{data} + \sqrt{1 - \bar{\alpha_t}} \varepsilon$$ holds true,

Then, the score can be approximated by $$\varepsilon$$

$$\nabla_x \log p(x_t \|x_{data}) = -\frac{x_t-\sqrt{ \bar{ \alpha_t} } x_{data}}{1-\bar{\alpha_t}} = -\frac{\varepsilon}{\sqrt{1-\bar{\alpha_t}}}$$

We can claim that:

$$x_t = \sqrt{\bar{\alpha_t}} x_{data} - (1 - \bar{\alpha_t}) s_{\theta}$$  holds true. $$s_{\theta}$$ is the score predicted by neural network.

---

## Flow Matching

### Forward Process

In flow matching, the forward process is:

$$x_t =  (1 - t) x_{data} + t \varepsilon = a_t x_{data} + b_t \varepsilon$$, where $$t \in [0, 1]$$

Flow matching can be regarded as a uniform linear motion between data and noise.

### Reverse Process

The reverse ODE is:

$$\frac{dx_t}{dt} = \varepsilon - x_{data} = v_t(x)$$

You can solve this ODE using Euler's method. An interesting fact is that the direction of the velocity in Rectified Flow is from noise to data; whereas in DDPM-v-pred, it is from data to noise.

### Loss Type

The objective function for flow matching [$$^{\text{ref-Section 2}}$$](https://arxiv.org/pdf/2403.03206);  [$$^{\text{ref-Theorem 3}}$$](https://arxiv.org/pdf/2210.02747)  is:

$$\text{MSE}(v(x_{data}, t) - u(x_t| \varepsilon))$$

where $$u(x_t \| \varepsilon) = \frac{a'_t}{a_t} x_t - \frac{b_t}{2} \left(\log \frac{a^2_t}{b^2_t}\right)' \varepsilon$$
