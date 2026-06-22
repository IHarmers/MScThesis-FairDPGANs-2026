# MSc Thesis: FairDPGANs (2026)
## Introducing Differentially Private and Fairness-Aware Generative Adversarial Networks

This repository contains the code for my MSc thesis on a novel collection of fairness-aware and differentially private GANs, dubbed FairDPGANs. This thesis was carried out for the Open Universiteit (Heerlen) under the guidance of dr. Mina Alishahi. 

## Abstract

**Context.** Synthetic data generation has become a popular solution for publicly releasing sensitive information of individuals by generating synthetic versions of real datasets. These synthetic datasets offer privacy protection and remove some of the discrimination present in the real datasets. However, there is a noticeable gap in literature: most works focus on privacy or fairness, but not on simultaneously achieving both. To the best of our knowledge, we propose the first generative adversarial networks (GANs), dubbed *FairDPGANs*, that satisfy differential privacy and optimize for group fairness constraints during training. These models are a combination of the differentially private DP-GAN [Rosenblatt et al., 2020; Xie et al., 2018 and the fairness-aware TabFairGAN [Rajabi and Garibay, 2022]. 

**Aims.** Our aim is to improve the fairness of DP-GAN while maintaining its privacy protection, and to improve the privacy of TabFairGAN while retaining its fair properties. Additionally, we introduce two novel fair preprocessing techniques: *sensitive balancing preprocessing* (SBP) and *individual fairness preprocessing* (IFP). Their aim is to improve the fairness of a GAN's synthetic datasets and of the machine learning (ML) classifiers trained on them by improving the fairness of the real datasets with preprocessing. 

**Methods.** We carried out extensive fairness, utility and privacy analyses with our FairDPGANs and state-of-the-art (SOTA) fairness-aware and private GANs using multiple benchmark datasets, sensitive attributes, ML classifiers and a variation of privacy levels. 

**Results.** Our results showed that the FairDPGANs frequently outperform DP-GAN and TabFairGAN with regard to the group and individual fairness of the ML classifiers, respectively, at the cost of utility. Furthermore, our FairDPGANs generally have privacy results similar to DP-GAN and outperform TabFairGAN in terms of privacy. The same level of success was not achieved with the fair preprocessing techniques, since SBP and IFP generally had inconsistent impact on the fairness results of our FairDPGANs and the other SOTA GANs. 

**Conclusions.** For improving the fairness of DP-GAN and the privacy of TabFairGAN, we recommend our FairDPGAN *Dis* whose group fairness constraint is based on the concept of disparate impact \citep{feldman-2014}. We tuned the strength of the fairness constraint with the hyperparameter $\lambda_f$, and found the variants *Dis* ($\lambda_f = 0.5$) and *Dis* ($\lambda_f = 1.5$) the most successful in improving the fairness of DP-GAN.  

## Communication

For the full data release of my work (which includes ~ 97 GB of data), please contact me at i.harmers@student.ou.nl.
