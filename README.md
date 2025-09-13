# Comparative Analysis of VAE and GAN Latent Spaces

This repository contains the code and experiments for the university project titled "Comparative analysis of VAE and GAN latent spaces via MLP-based projections". The project investigates the structural relationship between the latent spaces of a Variational Autoencoder (VAE) and a Generative Adversarial Network (GAN) trained on the MNIST dataset.

## Project Overview

The core goal is to determine if a meaningful mapping can be learned to translate latent vectors from a VAE's space to a GAN's space (and vice-versa) while preserving the semantic identity of the data. This is achieved by training simple Multi-Layer Perceptrons (MLPs) as "bridges" between the two spaces.

The key findings suggest that while a structural correspondence exists, the stability and optimal complexity of the mapping are highly context-dependent, showing significant asymmetries and performance differences between global (multi-class) and specialized (single-class) scenarios.

### Key Concepts
- **Generative Models:** A VAE and a DCGAN are trained on MNIST with 2D latent spaces.
- **MLP Bridge:** An MLP is trained as a regressor to map latent vectors ($z_{vae} \to z_{gan}$ and $z_{gan} \to z_{vae}$).
- **GAN Inversion:** An optimization-based method is used to find the GAN latent vector for a given image.
- **Two Experiments:**
    1.  **Multi-Class:** Training a global bridge on all 10 MNIST digits.
    2.  **Single-Class:** Training a specialized bridge on only the digit '2'.
- **Evaluation:** The success of a projection is measured by visual similarity (Image MSE) and semantic consistency (accuracy from an independent CNN classifier).
