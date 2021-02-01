# Rethinking the Inception Architecture for Computer Vision

> enabling : 对... 有积极意义
>
> reliant：依赖性的
>
> compelling：强迫的、不可抗拒的
>
> inherently：固有地
>
> mitigate：减轻、缓和
>
> deem：认为
>
> speculative：推测的、猜测的
>
> deterioration：恶化、退化
>
> disentangled：分清，清理出
>
> aggregation：聚合



# 0. Abstract

尽管增大的模型大小和计算代价往往能转变成多数任务的直接质量提高(只要有足够的标记数据用于训练)，**计算效率和低参数数量**依然是许多应用场景的积极因素，比如移动视觉和大数据场景。

> Although increased model size and computational cost tend to translate to immediate quality gains for most tasks (as long as enough labeled data is provided for training),  **computational efficiency and low parameter**  count are still enabling factors for various use cases such as mobile vision and big-data scenarios.

我们正在探索了用来放大网络的方法，该方法旨在尽可能**有效地利用增加的计算**，通过合适地**分解的卷积和激进的正则化**。

>  Here we are exploring ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by suitably factorized convolutions and aggressive regularization.



# 1. Introduction

从AlexNet到VGG net，GoogLeNet，这些用于图像分类的网络结构的改变往往能在其他领域获得重大的质量提高。这意味着**深度卷积网络结构的提高能够被用来提高其他越来越依赖于高质量、可学习视觉特征的计算视觉任务**。此外，网络质量的提升带来了原先AlexNet无法与人类抗衡的新的应用领域。

VGG虽然有吸引人的简洁架构特点，但是代价太高，评估网络需要许多计算量。而GoogLeNet架构被设计用于在即使有着严格限制的内存和计算开销的场景。

**Inception的计算开销远小于VGGNet或其表现优秀的后继者。**这使得Inception可以应用在**大数据场景**、**内存和计算资源受限的**移动计算场景等。当然使用一些特定的解决方案来达到目标内存使用量、或通过计算技巧来优化特定操作的执行。但是这些方法增加了额外复杂度，此外，这些方法也可以用于优化Inception架构。

**然而，Inception架构的复杂性使得很难对网络进行修改。**如果天真地放大架构，很大一部分计算收益可能会立即丢失。并且原先没有提供对GoogLeNet设计决定的关键因素的清晰描述。使得很难将其调整到新的应用场合，同时还保持其有效性。

In this paper,  we start with describing **a few general principles and optimization ideas** that that proved to **be useful for scaling up convolution networks in efficient ways**. 

Although our principles are not limited to Inception-type networks, they are easier to observe in that context as the generic structure of the Inception style building blocks is flexible enough to incorporate those constraints naturally.



# 2. General Design Principles

1. 在网络早期，避免代表性的瓶颈。

   > Avoid representational bottlenecks, especially early in the  network. 

   应当**避免急剧压缩的瓶颈**。总体来说，**表示尺寸**应当在到达最终的表示之前从输入到输出**缓慢地减小**。

   > One should **avoid bottlenecks with extreme compression**.  In general the representation size should gently decrease from the inputs to the outputs before reaching the final representation used for the task at hand.  

2. **更高维度的表示**更容易在网络内本地处理。

   > Higher dimensional representations are easier to process locally within a network. 
   >
   > Increasing the activations  per  tile  in  a  convolutional  network  allows  for more  disentangled  features.   The  resulting  networks will train faster.

3. **空间聚合**能够在**低维的嵌入**上进行，而不会损失表示能力。

   > Spatial  aggregation  can  be  done  over  lower  dimensional  embeddings  without  much  or  any  loss  in  representational power.

   我们假设原因是**相邻单元的强大联系带来了维度归约时更少的信息损失**，如果输出备用在空间聚合上下文。

   > We hypothesize that the reason for that is the strong correlation between adjacent unit results in much less loss of information during dimension reduction,  if the outputs are used in a spatial aggregation context.

4. **平衡网络的深度和宽度**。

   > Optimal performance of the network can be reached by **balancing  the  number  of  filters  per  stage  and  the  depth  of the network.**

   同时增加网络的深度和宽度可以带来高质量的网络，如果并行增加深度、宽度，则可以达到最佳改进，计算预算也会因此在网络的深度和宽度之间以平衡的方式分配。

这些设计原则可能是合理的，未来还需要更多的实验证据来评价他们的准确率。



# 3. Factorizing Convolutions with Large Filter Size

## (1) Factorization into smaller convolutions



## (2) Spatial Factorization into Asymmetric Convolutions



# 4. Utility of Auxiliary Classifiers



# 5. Efficient Grid Size Reduction



# 6. Inception-v2



# 7. Model Regularization via Label Smoothing



# 8. Training Methodology



# 9. Performance on Lower Resolution Input



# 10. Experimental Results and Comparisons



# 11. Conclusions



