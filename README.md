# micae-experiments
> Collection of experiments on multiple-input convolutional autoencoder neural networks

This repository contains the collection of studies and explorations I undertook for part of my Bachelor thesis. The notebooks were written for prototyping and generating results quickly, so many of them are lacking documentation and are probably not too informative for a reader unfamiliar to the topic. I include those studies anyways to showcase the breadth of models and training methods that have been explored and compared during the research phase. 

## Research Objectives
1) Probing latent space: Better understand the compressed representation of autoencoder models.
2) Latent space concatenation: Understand if and how latent spaces of several autoencoders can be combined.

## Boundary Conditions
1) The encoder part of the models must share same weights, as for the underlying use case the input data patches are translational invariant and it is not feasible to train each encoder unit separately in the real application.
2) The encoder architecture must be relatively shallow, so that they can fit into the limited memory of their intended hardware.
3) The compression factor (input/latent) must be 3 or larger. (ratio input/output links (48x8b/16x9b) in the first stage)
4) The encoder architecture is as good as fixed. (Input->Conv2D->Flatten->Dense)

## Findings
1) Elementary encoding units can be trained in solitarity and their latent spaces can be concatenated.
2) The concatenated latent space can be used as input to a decoder that can be trained to reproduce the concanation of the elementary encoding units' input images.
3) Training the decoder with the same set of preloaded and fixed weights for all elementary encoding units results in a model with similar performance to training all encoders and the decoder of the model together from scratch. 
4) This model can also be trained from scratch using tied encoder weights, yielding similar performance.
5) Larger latent dimension does not mean better performance. (maxing out at latent_dim~64)
6) Neither CAE nor MICAE seem to overfit.
