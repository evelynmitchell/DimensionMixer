# DimensionMixer
Dimension Mixer model 

From:
[Dimension Mixer: A Generalized Method for
Structured Sparsity in Deep Neural Networks](https://arxiv.org/pdf/2311.18735.pdf)

"The recent success of multiple neural architectures
like CNNs, Transformers, and MLP-Mixers motivated us to look
for similarities and differences between them. We found that
these architectures can be interpreted through the lens of a
general concept of dimension mixing. Research on coupling flows
and the butterfly transform shows that partial and hierarchical
signal mixing schemes are sufficient for efficient and expressive
function approximation. In this work, we study group-wise
sparse, non-linear, multi-layered and learnable mixing schemes of
inputs and find that they are complementary to many standard
neural architectures. Following our observations and drawing
inspiration from the Fast Fourier Transform, we generalize
Butterfly Structure to use non-linear mixer function allowing
for MLP as mixing function called Butterfly MLP. We were
also able to mix along sequence dimension for Transformer-
based architectures called Butterfly Attention. Experiments on
CIFAR and LRA datasets demonstrate that the proposed Non-
Linear Butterfly Mixers are efficient and scale well when the
host architectures are used as mixing function. Additionally, we
propose Patch-Only MLP-Mixer for processing spatial "

