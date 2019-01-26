# Bibliography
---


There is a large body of research concerned with Bayesian learning of neural
networks. Some good entry points on various sub-topics is listed
[here](https://github.com/ssydasheng/Bayesian_neural_network_papers). If you
are interested in the subject you should certainly take a look at that list!


## Longform references

1. Radford Neal's thesis, [Bayesian learning for neural
   networks](http://www.csri.utoronto.ca/~radford/ftp/thesis.pdf)

2. Yarin Gal's thesis, [Uncertainty in Deep
   Learning](http://mlg.eng.cam.ac.uk/yarin/blog_2248.html)

3. David MacKay's book, an early proponent of Bayesian Neural Networks:
[Information Theory, Inference, and Learning
Algorithms](http://www.inference.org.uk/itprnn/book.pdf)


## Methods used in RL, usually in conjunction with Thompson Sampling:

1. [Randomized Prior Functions for Deep Reinforcement Learning](https://arxiv.org/abs/1806.03335)

2. [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)

3. [Efficient Exploration through Bayesan Deep Q-Networks](https://arxiv.org/pdf/1802.04412.pdf)

4. [Bayesian Inference with Anchored Ensembles of Neural Networks, and
   Application to Exploration in Reinforcement Learning]
   (https://arxiv.org/abs/1805.11324)

5. [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep
   Learning](https://arxiv.org/abs/1506.02142)


## Some papers on Thompson sampling and PSRL:

1. [Why Posterior Sampling is better than Optimism for RL?](https://arxiv.org/pdf/1607.00215.pdf)
2. [Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling](https://arxiv.org/abs/1802.09127)
3. [A Tutorial on Thompson Sampling](https://arxiv.org/abs/1707.02038)


## Other methods

We are interested however in methods that show good promise from a practical
perspective. Therefore the list below is sorted by how easy to implement and
robust the methods seem to be.

1. [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep
   Learning](https://arxiv.org/abs/1506.02142). This work is accompanied by a
   nice [blog
   post](http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_3d801aa532c1ce.html).

2. [Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in
   Adam](https://arxiv.org/abs/1806.04854). In some ways this is similar to
   [Noisy Natural Gradient as Variational
   Inference](https://arxiv.org/abs/1712.02390) which however does a more
   complex approximation of the Hessian with a Kronecker factorization of the
   Fisher.

3. There are several related papers from around 2015 which pose the problem of
   learning a distribution over weights as variational inference. [Weight
   Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424) and
   [Variational Dropout and the Local Reparameterization
   Trick](https://arxiv.org/abs/1506.02557) are such examples. I feel a simple
   and straightforward exposition can be found in [Good Initialization of
   Variational Bayes for Deep Models](https://arxiv.org/pdf/1810.08083.pdf).

4. I liked this paper a lot, a good read on the topic. It proposes Bayesian
   Ensembles with MAP sampling. [Uncertainty in Neural Networks: Bayesian
   Ensembling](https://arxiv.org/pdf/1810.05546.pdf)

5. [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks]
   (https://arxiv.org/abs/1512.05287)

## Calibration

A special topic in this field is that of **calibration**, that is a model is
well calibrated if for those predictions it assigns a probability _p_, the
model will actually make those predictions p% of the time as stated in [The
Well-Calibrated
Bayesian](https://www.tandfonline.com/doi/abs/10.1080/01621459.1982.10477856).

Some recent references on this topic (although this subject is talked in some
of the papers above as well):

1. [Accurate Uncertainties for Deep Learning Using Calibrated
   Regression](https://arxiv.org/pdf/1807.00263.pdf)

2. ... feel free to add some more.
