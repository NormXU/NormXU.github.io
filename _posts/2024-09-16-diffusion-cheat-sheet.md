---
layout: post
title: "Cheat Sheet for Mathematical Diffusion"
tags: ["Diffusion"]
---

This is a cheat sheet of all denoising-based diffusion method. No mathmatica derivations are included for concise. Check the reference links if you're interested in the derivations. 



Current denoising-based generation models consist of three main components:

1. **Forward Process**
2. **Sampling**
3. **Model Prediction**

### Forward Process

The forward process can be understood as an ODE:

$$dx_{t} = f(x_t, t) \, dt$$

If we add stochastic perturbation, the ODE is transformed into a SDE,

$$dx_{t} = f(x_t, t) \, dt + g(t) \, d\mathbf{w}$$

Here, $$d\mathbf{w}$$ represents a **Wiener process**, where $$\mathbf{w_t} \sim \mathcal{N}(0, t)$$. Therefore, $$d\mathbf{w} \sim \mathcal{N}(0, dt)$$

This can be approximated as:

$$d\mathbf{w} = \sqrt{dt} \; \epsilon$$ ,  where $$ \epsilon \sim \mathcal{N}(\mu, \sigma^2)$$

Different denoising methods define their own specific functions for $$f(x_t, t)$$  and  $$g(t)$$

### Sampling

By adding noise during the forward process, we train the model to reconstruct the distribution of the training data. During inference, this allows us to sample an image or video from the learned distribution. The sampling can be either deterministic or stochastic. A common equation is as:

$$x_{t + \Delta t} \sim \mathcal{N}(x_t + f(x_t, t) \Delta t, g^2(t)  \Delta t)$$

The reverse process expects to compute the posterior distribution $$p(x_t \| x_{t + \Delta t})$$

### Model Prediction

Which objects to learn. It can be score function, velocity, initial data and noises.



---

## DDPM

### 1.1 Forward Process

The forward process is given by:

$$x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \varepsilon$$

This can also be written as:

$$x_t = \sqrt{\bar{\alpha_t}} x_{data} + \sqrt{1 - \bar{\alpha_t}} \varepsilon$$

We can express this in the form of a SDE [$$^{\text{ref-Section D}}$$](https://arxiv.org/pdf/2210.02747) :

$$dx_t = f(x_t, t) dt + g(t) d\mathbf{w} = -\frac{1}{2} \beta(t) x_t \;dt + \sqrt{\beta(t)} d\mathbf{w}$$

We can clearly see from the equation that DDPM forward process is a curve motion, where the magnitude and direction of velocity is time-dependent.

### 1.2 Sampling

The reverse process is expressed as:

$$dx_t = \left(f(x_t, t) - g^2(t) \nabla_x \log p(x_t)\right) dt + g(t) d\bar{\mathbf{w}}$$

$$\bar{\mathbf{w}}$$ is a reverse Wiener process.

This process can be solved using any SDE solver you like.

### 1.3 Model Prediction

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

### 2.1 Forward Process

The forward process in score matching is:

$$x_t = x_{data} + \sigma_t \varepsilon$$

Here,  $$x_{data} \sim p_{data}(x)$$ , where  $$p_{data}(x) $$ represents the distribution of the training dataset. The noise variance decreases over time:

$$\sigma_1 > \sigma_2 > \sigma_3 > \dots$$

This means the noise variance added to the data gradually decreases. The corresponding ODE is:

$$dx_t = \sqrt{\frac{d\sigma^2_t}{dt}} d\bar{\mathbf{w}}$$

The forward process can be imagined a straight line going from data to noise where the velocity (variance of noise) gradually decreases from large to small.

### 2.2 Sampling

$$d\mathbf{x_t} = -\left(\frac{d[\sigma(t)^2]}{dt} \nabla_{\mathbf{x}} \log p(\mathbf{x_t}) \right) dt + \sqrt{\frac{d[\sigma(t)^2]}{dt}} d\bar{\mathbf{w}}$$

The reverse sampling follows the Langevin equation:

$$x_{t+1} = x_t + \tau \nabla_x \log p(x_t) + \sqrt{2\tau} z$$

where  $$z \sim N(0, I)$$ 

We can see from the sampling equation that although the forward process is linear, the reverse process is stochastic, which makes score-matching sampling hard to hack.

### 2.3 Model Prediction

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

### 3.1 Forward Process

In flow matching, the forward process is:

$$x_t =  (1 - t) x_{data} + t \varepsilon = a_t x_{data} + b_t \varepsilon$$, where $$t \in [0, 1]$$

Flow matching can be regarded as a uniform linear motion between data and noise.

### 3.2 Sampling

The reverse ODE is:

$$\frac{dx_t}{dt} = \varepsilon - x_{data} = v_t(x)$$

You can solve this ODE using Euler's method. An interesting fact is that the direction of the velocity in Rectified Flow is from noise to data; whereas in DDPM-v-pred, it is from data to noise.

### 3.3 Model Prediction

The objective function for flow matching [$$^{\text{ref-Section 2}}$$](https://arxiv.org/pdf/2403.03206);  [$$^{\text{ref-Theorem 3}}$$](https://arxiv.org/pdf/2210.02747)  is:

$$\text{MSE}(v(x_{data}, t) - u(x_t| \varepsilon))$$

where $$u(x_t \| \varepsilon) = \frac{a'_t}{a_t} x_t - \frac{b_t}{2} \left(\log \frac{a^2_t}{b^2_t}\right)' \varepsilon$$
