> Translated from blogs [1](https://kexue.fm/archives/9675), [2](https://kexue.fm/archives/9706), [3](https://kexue.fm/archives/9708) originally written in Chinese by Su, Jianlin
>
> Translated by Norm Inui

(WIP🚧)
## RoPE is a $$ \beta $$-based encoding 
For developers who are interested in how to extend the context length of LLMs (Large Language Models), the open-source community has continuously presented us with fascinating proposals in the past few weeks. First, the user @kaiokendev experimented with a "positional linear interpolation" approach in his project SuperHOT. 
He demonstrated that with minimal fine-tuning on long texts, existing LLMs can be easily adapted to handle contexts over their pretraining context length. Almost simultaneously, Meta proposed the same idea, publishing their comprehensive experimental results in the paper titled "Extending Context Window of Large Language Models via Positional Interpolation." 
Shortly after the paper was published, @bloc97 introduced the NTK-aware Scaled RoPE, enabling LLM to extend its context length without fine-tuning!


With all these methods, especially the NTK-aware Scaled RoPE, it persuades me to rethink the idea behind RoPE. 
And I realized that the RoPE can be regarded as a $$ \beta $$-based encoding. From this perspective, the methods mentioned above can be understood as various extending for the encoding base.

### Base Representation
Suppose we have an integer $$n$$ within $$1,000$$ (not including $$1,000$$) that we want to input into a model. What would be the best way to do this?

The most intuitive idea is to input it directly as a one-dimensional vector. However, the value of this vector has a large range from $$0$$ to $$999$$, which is not easy to optimize for gradient-based optimizers. What if we scale it between 0 and 1? That's not good either, because the difference between adjacent numbers changes from $$1$$ to $$0.001$$, making it challenging for both the model and optimizer to distinguish between the numbers. In general, gradient-based optimizers are a bit "vulnerable" and can only handle inputs that aren't too large or too small.

To avoid this problem, a new way to represent the input is necessary. We might think about how we humans do. For an integer, like $$759$$, it's a three-digit number if encoded with 10 as base, with each digit ranging from 0 to 9. This inspires me to represent the input into the model directly with the 10-based encoding. That is, we transform the integer 

$$n$$ as a three-dimensional vector $$[a,b,c]$$, where $$a$$, $$b$$, and $$c$$ represent the hundreds, tens, and units of $$n$$ respectively. This way, we can both reduce the range of the numbers and get rid of reducing the difference between adjacent numbers, by increasing the input dimension. Luckily, neural networks are good at handling high-dimensional data.

If we want to further reduce the span of each digit in the numbers, we can decrease the base, like using 8, 6, or even 2 as the encoding base at a cost of an increase in input vector dimensions.

### Direct Extrapolation

Let's assume we have trained a model using a three-dimensional vector, which is a representation of a number ranging from $$0$$ to $$999$$ with 10 as the encoding base, as the input, and the results are quite well. Then we want to increase the upper bound of $$n$$ to be $$2,000$$. How can we realize this?

If we follow the same idea to represent the number, the input will now be a four-dimensional vector. However, the original model was designed and trained for a three-dimensional vector. Therefore, the model won't be able to process the input with an extra dimension. Some might wonder, why can't we reserve extra dimensions in advance? Indeed, we can pre-allocate a few more dimensions. During the training phase with the upper bound as $$1,000$$, they can be set to $$0$$, but during the inference phase with an upper bound as $$2,000$$, the pre-allocate dimension has to be set to numbers besides $$0$$. This approach is what we call Extrapolation.
![Extrapolation](https://raw.githubusercontent.com/NormXU/NormXU.github.io/main/_data/resources/blog/0/extrapolation.png)

However, the dimensions reserved during the training phase have always been set to 0. If these dimensions were changed to other numbers during the inference phase, the results might be harmed. This is because the model is never trained to adapt to those pre-allocated digits in different values. In other words, due to insufficient training data for those dimensions, the extrapolation usually leads to a significant performance drop.
