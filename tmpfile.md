## Contents
- I Introduction
- II Inference in Hierarchical Models
  - II.1 Neal’s funnel
  - II.2 Reparameterization and the standardizing transform
  - II.3 Sampling a toy model
- III Signal and noise components of pulsar timing analysis
  - III.1 Linearized timing model
  - III.2 White noise
  - III.3 Red and chromatic noise
  - III.4 Deterministic signals and continuous waves
- IV Hierarchical inference in pulsar timing arrays
  - IV.1 Priors
  - IV.2 Standardizing the pulsar timing posterior
  - IV.3 Generalizing to include deterministic signals
  - IV.4 Generalizing to include inter-frequency correlations
- V Implementation techniques
  - V.1 Single precision
  - V.2 Batching
- VI Analyses of real and simulated datasets
  - VI.1 Results
  - VI.2 Discussion and future work
    - Acknowledgements.
- Appendix A Hamiltonian Monte Carlo and No U-Turn Sampling
- Appendix B The standardizing transform for tempered likelihoods
- Appendix C The standardizing transform for split coefficients.
- Appendix D Non-Gaussian features
- References

## Abstract

Abstract Pulsar timing array data analysis is computationally expensive, limiting the complexity of models which can be studied. As pulsar timing datasets and their respective models grow in size and sophistication, faster and scalable inference methods are essential. In this paper, we accelerate pulsar timing analyses by sampling in the space of Fourier coefficients instead of analytically marginalizing over them. Previous studies have shown the Fourier space induces a complex, high-dimensional posterior geometry, from which it is generally difficult to sample. We show that under an appropriate coordinate transformation the Fourier coefficients approximately follow a standard normal distribution, and may be efficiently sampled using a Hamiltonian Monte Carlo scheme. Under this coordinate transformation, for datasets of size and complexity comparable to the NANOGrav 15-year release, the new method produces converged posterior distributions for a range of models which include inter-pulsar correlations, stochastic, and deterministic signals in approximately 15 minutes on an NVIDIA GeForce RTX 3090 GPU. By comparison, the legacy pulsar timing analysis software ENTERPRISE would require months of computation on a CPU cluster to analyze comparable datasets under the same joint stochastic and deterministic models.

## I Introduction

Pulsar timing arrays (PTAs) measure the times of arrival (TOAs) of radio pulses from pulsars. If TOA observations span decades over a network of pulsars distributed across the sky, then PTAs are sensitive to gravitational waves (GWs) of nanohertz frequencies [76, 25]. Presently, the North American Nanohertz Observatory for Gravitational Waves (NANOGrav), the Parkes Pulsar Timing Array, the European Pulsar Timing Array in collaboration with the Indian Pulsar Timing Array, the Chinese Pulsar Timing Array, and the MeerKAT Pulsar Timing Array have found evidence for a stochastic gravitational wave background (GWB), [8, 73, 30, 102, 64].

There are many hypotheses for what physical process gives rise to the stochastic GWB, the most popular being the background is realized by an incoherent superposition of GWs emitted by a population of supermassive black hole binaries (SMBHBs), formed during galaxy mergers [72, 47, 101, 78, 62, 21]. The background may also include contributions from cosmological phase transitions, primordial GWs, and other exotic sources, see e.g. [4] and references therein.

Besides the GWB, other signal and noise processes, covariant with the background, populate PTA datasets and must be jointly modeled. This includes, but is not limited to, radiometer noise in the telescopes, achromatic red noise intrinsic to the pulsars, stochastic fluctuations of the dispersion of radio pulses in the interstellar medium, and GWs from individual SMBHBs which are particularly massive or nearby, and discernible from the stochastic population.

Standard analyses of PTA datasets use numerical Bayesian inference, where a (time-domain) posterior probability density is constructed. The posterior describes the probability of model parameters conditioned on the data observed. Inference is performed by sampling the posterior with Markov Chain Monte Carlo (MCMC) methods, requiring many evaluations of the posterior. Recently, other approaches such as stochastic gradient-descent Bayesian variational inference and simulation based inference with GPU implementations have accelerated the analysis [91, 80].

PTA data analysis is computationally expensive. Typical datasets consist of hundreds of thousands of radio pulse TOAs recorded over decades. The observations are unevenly spaced and the noise heteroscedastic, necessitating the analysis to be conducted in the time-domain. Moreover, radio pulse timing delays due to pulsar proper motion, spin period, and other deterministic effects are fit using a per-pulsar time-domain model. The substantial number of TOAs in realistic datasets requires large and expensive linear algebra computations on the order of the size of the dataset. The analysis problem only worsens as datasets grow with more observations and as pulsars are added to the array. Complicated models further bottleneck the analysis as they generally involve higher-dimensional parameters spaces and expensive posterior evaluations which require more calls to adequately sample the distribution. Currently, the complexity of models used in PTA analyses are restricted by their computational burden, rather than the resolution of the data.

The aim of this paper is to accelerate Bayesian analyses of GWs in PTA datasets, allowing more in-depth modeling of larger datasets. The latest PTA analysis packages have leveraged the JAX [19] software package, GPU-acceleration, and gradient-based samplers to far outpace the flagship software ENTERPRISE [29], but continue to analytically marginalize over a large number of Fourier coefficients. The methods presented in this paper can analyze the latest publicly released datasets over an order of magnitude faster than even these accelerated approaches by sampling in the space of Fourier coefficients (i.e. by performing the marginalization numerically). We reproduce some results from the NANOGrav 15-year stochastic analysis [8] with these methods and perform previously computationally inaccessible analyses on simulated datasets. These speedups are achieved by applying a coordinate transform on the hyper-efficient posterior presented in [59]. This posterior, while hyper-efficient, is high-dimensional and exhibits a complex funnel geometry from which it is difficult to sample. The coordinate transform reparameterizes the funnel so the posterior more closely resembles a standard normal distribution, from which samples may be drawn very efficiently. The inverse transformation maps samples back to those of the original, untransformed, posterior.

Previous works have sampled this hyper-efficient, but geometrically difficult posterior to varying degrees of success. [59] uses a Hamiltonian Monte Carlo (HMC) sampling algorithm to sample the posterior distribution in the high signal-to-noise regime where the funnel did not manifest, [96, 51, 52] uses Gibbs sampling on a subset of models with a weak-enough funnel to sample efficiently, and [43, 42] extends the posterior to include deterministic signals, and uses a generalized (even higher-dimensional) model parameterized to weaken the sharpness of the funnel. Other approaches have applied coordinate transformations to PTA analyses, e.g. [87] sampled in the space of whitened power per frequency bin, our approach on the other hand applies a coordinate transformation to the latent mode amplitudes per frequency bin which directly represent the data.

Primarily, previous studies have focused on mitigation and sampling techniques after the construction of the posterior. Our objective, based on the work of [92], is to reparameterize the posterior from the start. That way, increasingly intricate sampling techniques need not be introduced to retain efficiency as the complexity of models and the size of datasets grow in the future. While the original approach of [92] is computationally similar to our transformation of the posterior distribution, it was so extremely inefficiently implemented that it was computationally challenging even for single-pulsar analyses. Our current implementation is many orders of magnitude faster due to our Fourier-domain focus, parallelization, and the use of JAX [19]. Moreover, our approach generalizes the coordinate transformation to include contributions from deterministic signals, inter-frequency correlations, and non-Gaussian features.

Despite its near normal distribution, the transformed posterior remains difficult to sample due to its high-dimension, having $\mathcal{O}(10^{3})$ parameters (under commonly used models for the latest datasets). For this reason we sample with HMC and a No U-Turn Sampler (NUTS) scheme, which is inspired by the Hamiltonian structure of classical mechanics (see e.g. [17]). HMC integrates trajectories through a phase space, in which the target posterior is embedded, sampling along the way. The integrated trajectories are deterministic, allowing the sampler to traverse larger distances through parameter space than alternative random walk proposals. It is advantageous for high-dimensional densities where, under careful tuning, it can achieve high (often $\sim 90\%$) proposal acceptance rates and low auto-correlation lengths. For example, the computational cost to sample a $d$-dimensional multivariate normal distribution with naive random-walk Metropolis is $\mathcal{O}(d^{2})$, [36], while the cost with HMC is $\mathcal{O}(d^{5/4})$ [14]. Such sampling schemes have previously been used in PTA data analysis, [34, 35], but the posterior sampled was one in which nuisance parameters were analytically marginalized from the model rendering a lower-dimensional, but computationally expensive, target distribution bottlenecking the sampler. The HMC sampling scheme is summarized in Appendix [A](#A1).

Altogether, the methods presented here allow us to analyze large PTA datasets under joint (stochastic and deterministic) models over an order of magnitude more efficiently than previous approaches. Section [II](#S2) reviews Bayesian hierarchical modeling and demonstrates the effectiveness of standardizing coordinate transformations on a toy example. The signal and noise components of PTA analyses, and their respective models, are discussed in Sec. [III](#S3). Section [IV](#S4) constructs the PTA hierarchical Bayesian model and the coordinate transformation under which the posterior may be efficiently sampled. The coordinate transform is first presented for a purely stochastic model, then generalized to include deterministic contributions and inter-frequency correlations. Previous results from the NANOGrav 15-year analysis [8] are reproduced Sec. [VI](#S6) along with an analysis of simulated data.

## II Inference in Hierarchical Models

Hierarchical models are powerful tools in Bayesian inference. Generally a set of low-level (or latent) parameters, $\mathbf{x}$, are used to describe the data, $\mathbf{d}$, for which a likelihood function may be constructed, $p(\mathbf{d}|\mathbf{x})$. Additional (high-level) hyper-parameters, $\mathbf{y}$, are used to parameterize the prior distribution of the low-level parameters, $p(\mathbf{x}|\mathbf{y})$, and a hyper-prior is placed on the hyper-parameters, $p(\mathbf{y})$. Bayes’ theorem allows us to construct the full hierarchical model,

$$ $p(\mathbf{x},\mathbf{y}|\mathbf{d})\propto p(\mathbf{d}|\mathbf{x})\cdot p(\mathbf{x}|\mathbf{y})\cdot p(\mathbf{y})\,.$ (1) $$

Such hierarchies are common when modeling data of individuals, while simultaneously wishing to conduct inference on the population. For example, the LIGO and VIRGO detectors, [20, 2], have observed hundreds of short duration GWs, [1]. Hierarchical models are necessary when inference is desired simultaneously on the parameters of individual compact binaries as well as on the parameters of the population. The parameters for each individual binary are the low-level parameters, whose priors are conditioned on the population parameters [3, 61, 99, 60, 86]. Beyond being useful tools, hierarchical models are required for the analysis of PTA datasets [98, 41, 40]. For a more in-depth discussion of Bayesian hierarchical models, see e.g. Ch. 5 of [37].

### II.1 Neal’s funnel

While extremely powerful, the coupling between low- and high-level parameters induced by hierarchical models can yield complicated posterior geometries, from which it is difficult to sample with standard techniques. Neal’s funnel is one such geometry, referring to an exponentially tapering probability density [66]. The funnel is present in regions of parameter space where a hyper-parameter dictates the variance of a low-level parameter. The opening of the funnel is formed where the low-level parameter is allowed significant variance and the throat where the variance is constricted. Standard MCMC methods, such as Random Walk Metropolis [63], struggle to sample distributions with such geometry because increasingly precise jump proposals must be made as the chain traverses the throat of the funnel. In practice, naive samplers will get stuck in the throat of the funnel, and fail to explore the target distribution.

There are several ways to improve sampling for densities exhibiting Neal’s funnel. More robust samplers, such as Riemannian Manifold Hamiltonian Monte Carlo, have been shown to effectively sample funnels [39, 16]. Alternatively, the low-level parameters of the funnel may be analytically marginalized from the model in some cases. Analytic marginalization results in a lower-dimensional, funnel-less, and overall easier parameter space to explore. However, the marginalized target density is often significantly more computationally expensive to evaluate, bottlenecking the sampler, as is the case of standard PTA analyses. The funnel may also be avoided by generalizing the parameter space to a higher-dimensional space in which the sharpness of the funnel is lessened. Once sampled, the higher-dimensional distribution may be constrained to lie in the original space containing the funnel [42, 43]. Lastly, one may perform a coordinate transformation, effectively reparameterizing the model so its geometry is more easily explored [68]. In this paper, we will address the funnel geometry of the PTA posterior with such a coordinate transformation.

### II.2 Reparameterization and the standardizing transform

Consider a set of random variables $\mathbf{x}$ which are distributed according to some probability density function, $p=p(\mathbf{x})$. The density may be arbitrarily reparameterized under a bijective and differentiable coordinate transformation, $T$, via the mapping $\mathbf{z}=T(\mathbf{x})$ and its inverse $\mathbf{x}=T^{-1}(\mathbf{z})$. The transformed density, $\tilde{p}$, is defined over the new coordinates, $\tilde{p}=\tilde{p}(\mathbf{z})$, and must conserve probability mass, $p\,d\mathbf{x}=\tilde{p}\,d\mathbf{z}$ from which it follows

$$ $\tilde{p}(\mathbf{z})=p(\mathbf{x})\cdot\text{det}(\partial\mathbf{x}/\partial\mathbf{z})=p\big(T^{-1}(\mathbf{z})\big)\cdot\text{det}(\partial\mathbf{x}/\partial\mathbf{z})$ (2) $$

where $\text{det}(\partial\mathbf{x}/\partial\mathbf{z})$ denotes the determinant of the Jacobian of the coordinate transformation. If we wish to draw random samples $\mathbf{x}$ from the target distribution, we may either sample the target distribution, $p$, directly or sample in $\mathbf{z}$ from the transformed distribution, $\tilde{p}$, and map our samples back to the original coordinates under the inverse transformation, $\mathbf{x}=T^{-1}(\mathbf{z})$. Both methods produce identical distributions of the random variables, $\mathbf{x}$, provided the coordinate transformation is bijective, differentiable, and has a non-vanishing Jacobian determinant.

While any well-behaved coordinate transformations may be employed, the most effective is one which yields a transformed target distribution from which it is easiest to sample. Generally the more a target density resembles an uncorrelated standard normal distribution, the more efficiently independent samples may be drawn. An effective reparameterization is then one in which the target density is “standardized”. We will refer to such transformations as standardizing transforms (also known as non-centered parameterizations, decentering transforms, or whitening transforms); see [68] for a rigorous discussion of such transformations in Bayesian hierarchical models.

To perform a standardizing transformation, the mean and covariance of the target density, $p(\mathbf{x})$, are estimated, $\boldsymbol{\mu}=\mathbb{E}[\mathbf{x}]$ and $\boldsymbol{\Sigma}=\mathbb{E}[(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^{\text{T}}]\equiv\text{cov}(\mathbf{x},\mathbf{x})$, and the standardizing transformation and its inverse are

$$ $\mathbf{z}=T(\mathbf{x})=\mathbf{L}^{-1}(\mathbf{x}-\boldsymbol{\mu})\;\leftrightarrow\;\mathbf{x}=T^{-1}(\mathbf{z})=\boldsymbol{\mu}+\mathbf{L}\mathbf{z}\;$ (3) $$

where $\mathbf{z}$ are the new standardized (or rather whitened) coordinates and $\mathbf{L}$ is the Cholesky decomposition of the covariance matrix, $\boldsymbol{\Sigma}=\mathbf{L}\mathbf{L}^{\text{T}}$. It is straightforward to check if $p(\mathbf{x})$ is well-approximated by the Gaussian moments, $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$, then $\tilde{p}(\mathbf{z})$ is approximately an uncorrelated standard normal density. Substituting Eq. ([3](#S2.E3)) into Eq. ([2](#S2.E2)), the relationship between the original target density and its standardized form is

$$ $\tilde{p}(\mathbf{z})=p(\boldsymbol{\mu}+\mathbf{L}\mathbf{z})\cdot\text{det}(\mathbf{L})\,.$ (4) $$

After samples are obtained from the standardized density Eq. ([4](#S2.E4)), random samples from the original target distribution may be calculated using the coordinate transformation, Eq. ([3](#S2.E3)). It is generally very efficient to sample from near-normal distributions like Eq. ([4](#S2.E4)) with standard sampling schemes. Note that the statistical moments, $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$, need not be exact, and the target density may exhibit significant higher-order statistical moments, beyond covariance (see Appendix [D](#A4)). We may still sample from the true target density with Eq. ([4](#S2.E4)), but the standardizing transformation may lose its effectiveness, converging slower, as the Gaussian approximation weakens.

### II.3 Sampling a toy model

In this section we define a simple “toy” hierarchical Bayesian model with a funnel, similar to that first explored in [66]. We sample the toy model both directly and under a standardizing transform to demonstrate the effectiveness of reparameterization. The toy hierarchical model is,

$$ $p(\mathbf{x},y)=p(\mathbf{x}|y)\cdot p(y),$ (5) $$

with support for low-level parameters $\mathbf{x}\in\mathbb{R}^{d}$ for some integer $d\geq 1$ and a hyper-parameter $y\in\mathbb{R}$. The conditional density and hyper-prior are

$$ $\displaystyle\mathbf{x}|y\sim\mathcal{N}(y^{2}\,\mathbf{1},e^{y}\,\mathbf{I})$ $\displaystyle y\sim\mathcal{N}(0,9),$ (6) $$

where $\mathbf{1}$ is the $d$-dimensional vector of ones, $\mathbf{I}$ is the identity matrix, and $\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$ denotes a multivariate normal distribution with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$.

Neal’s funnel is observed in the low-level parameters, $\mathbf{x}$. When the hyper-parameter $y$ is negative (positive) the covariance of the low-level parameters is small (large), and the throat (opening) of the funnel is formed. Naive samplers will struggle to explore regions of the parameter space where $y$ is sufficiently negative. We therefore seek to apply a standardizing transform to the low-level parameters. The mean and covariance of the low-level parameters are $\mathbb{E}[\mathbf{x}]=y^{2}\,\mathbf{1}$ and $\text{cov}(\mathbf{x},\mathbf{x})=e^{y}\,\mathbf{I}$, respectively. Note the mean and covariance are themselves parameterized by the hyper-parameter, $y$, and the standardizing transform is similarly parameterized,

$$ $(\mathbf{x},\;y)=T^{-1}(\mathbf{z},y)=(y^{2}\,\mathbf{1}+e^{y/2}\,\mathbf{z},\;y)$ (7) $$

where $\mathbf{z}$ are the standardized coordinates. As presented here, the standardizing transformation only whitens the low-level parameters, $\mathbf{x}$, and applies the identity mapping to the hyper-parameter, $y$. The determinant of the Jacobian of the transformation is $\text{det}(\partial(\mathbf{x},y)/\partial(\mathbf{z},y))=e^{y\cdot d/2}$. The standardized probability density is then

$$ $\tilde{p}(\mathbf{z},y)=p(y^{2}\,\mathbf{1}+e^{y/2}\,\mathbf{z},\;y)\,\cdot\,e^{y\cdot d/2}\,.$ (8) $$

After sampling from the standardized density, Eq. ([8](#S2.E8)), in which Neal’s funnel should be absent, samples from the original target density, $p(\mathbf{x},y)$, are obtained using Eq. ([7](#S2.E7)).

We sample the toy hierarchical Bayesian model for $d=8$ both directly, and under a standardizing transformation using HMC with a NUTS scheme. Samples from both methods are shown in Fig. [1](#S2.F1). The chain attempting direct sampling of the target distribution gets stuck in the throat of the funnel and is unable to resolve the distribution. The chain sampling under the standardizing transform is able to efficiently explore the parameter space, including constricted regions in the throat of the funnel, and obtains samples consistent with the target distribution.

Figure: Figure 1: Samples from the hiearchical toy model, Eq. ([5](#S2.E5)) and Eq. ([II.3](#S2.Ex1)). The sampling is conducted for $d=8$, but only samples for the hyper-parameter $y$ and the first low-level parameter $x_{1}$ are shown in the figure. The blue distribution attempts to directly sample the target density, and is unable to effectively explore the throat of the funnel. The orange distribution samples under a standardizing transform and is able to move through constricted regions of parameter space efficiently. The green distribution is independent draws from the target density. The samples obtained through the standardizing transform are consistent with those from the true distribution.
Refer to caption: https://arxiv.org/html/2607.06834/2607.06834v1/figs/toy_model.png

## III Signal and noise components of pulsar timing analysis

The radio pulse TOAs for a pulsar consist of deterministic and stochastic contributions,

$$ $\mathbf{t}_{\text{TOA}}=\mathbf{t}_{\text{det}}+\mathbf{t}_{\text{stoch}}$ (9) $$

where $\mathbf{t}_{\text{TOA}}$ are the $n$ observed TOAs, $\mathbf{t}_{\text{det}}$ are the deterministic components, and $\mathbf{t}_{\text{stoch}}$ stochastic. The deterministic part of the TOAs is described primarily by the timing model, constructed per-pulsar, which models pulsar spin period, spin derivative, proper motion, and other deterministic effects [9]. In the main body of this paper, the stochastic component of the TOAs is modeled as Gaussian random processes. Non-Gaussian features are discussed in Appendix [D](#A4).

The timing model depends on $m$ parameters, $\boldsymbol{\beta}$. The per-pulsar timing analysis yields a set of reference timing model parameters, $\boldsymbol{\beta}_{0}$, used to construct the timing residuals,

$$ $\boldsymbol{\delta t}=\mathbf{t}_{\text{TOA}}-\mathbf{f}(\boldsymbol{\beta}_{0}),$ (10) $$

where $\boldsymbol{\delta t}$ are the timing residuals and $\mathbf{f}$ the timing model. The timing residuals are expanded in terms of other signal and noise components,

$$ $\displaystyle\boldsymbol{\delta t}=\boldsymbol{\delta t}_{\text{TM}}$ $\displaystyle+\boldsymbol{\delta t}_{\text{WN}}+\boldsymbol{\delta t}_{\text{RN}}+\boldsymbol{\delta t}_{\text{DM}}+$ $\displaystyle\boldsymbol{\delta t}_{\text{GWB}}+\boldsymbol{\delta t}_{\text{det}}+\dots$ (11) $$

where the various delays, $\boldsymbol{\delta t}_{i}$, correspond to small deviations in the timing model, white noise contributions, intrinsic instabilities in the pulsars, dispersion in the interstellar medium, a gravitational wave background, and deterministic delays not included in the timing model. The models for each of these components is summarized below.

### III.1 Linearized timing model

We assume the set of timing model parameters which precisely describe the proper motion and spin of the pulsar are near the reference parameters. The timing model is then expanded to linear order, centered at the reference parameters, $\boldsymbol{\beta}_{0}$,

$$ $\boldsymbol{\delta t}_{\text{TM}}=\mathbf{M}\boldsymbol{\epsilon}$ (12) $$

where $\mathbf{M}$ is the $(n\times m)$ timing design matrix with elements $M_{ij}=(\partial\mathbf{f}_{i}(\boldsymbol{\beta})/\partial\boldsymbol{\beta}_{j})|_{\boldsymbol{\beta}_{0}}$. The parameter vector $\boldsymbol{\epsilon}=\boldsymbol{\beta}-\boldsymbol{\beta_{0}}$ is the linear deviation from the reference parameters. Rather than modeling the parameters of the deterministic timing model themselves, the degrees of freedom associated to the timing model are represented by the linear deviations from the reference point.

### III.2 White noise

A TOA is constructed by de-dispersing and folding pulses observed within an observing epoch, fitting an average pulse template, and assigning an uncertainty to the TOA measurement, which is due primarily to radiometer noise in the telescope. Not all template fitting uncertainties may be propagated into the final quoted uncertainty for the TOA, so an extra factor (EFAC) is introduced, which multiplicatively corrects uncertainties in telescope receivers and backends. Additional instrumental effects cannot be modeled by EFAC, so an extra quadrature (EQUAD) term is added to the white noise model. The phenomenological white noise model is represented with the covariance matrix,

$$ $\mathbf{N}=\mathbb{E}[\boldsymbol{\delta t}_{\text{WN},i\mu}\,\boldsymbol{\delta t}_{\text{WN},j\nu}^{\text{T}}]=\mathcal{F}_{\mu}^{2}\sigma_{i}^{2}\delta_{ij}\delta_{\mu\nu}+Q_{\mu}^{2}\delta_{ij}\delta_{\mu\nu}$ (13) $$

where $\mathcal{F}$ is the EFAC, $\mathcal{Q}$ EQUAD, and $\sigma$ TOA uncertainty. Latin indices label TOA observations, $i,j\in\{1,2,\dots,n\}$ and Greek indices ($\mu,\nu,\dots$) denote specific receiver-backend systems. Lastly, folding a finite number of pulses within an observing epoch leads to pulse phase jitter. This induces an extra correlation (ECORR) which correlates different bands, but is uncorrelated between observing epochs [7].

EFAC, EQUAD, and ECORR values may be estimated from single-pulsar analyses before the multi-pulsar analysis is conducted. While it is possible to infer the white noise parameters simultaneously with other parameters of the multi-pulsar model, we will assume the initial white noise estimates are accurate, and fix the white noise model throughout our analysis. That is, the white noise covariance matrix (Eq. ([13](#S3.E13)) including ECORR contributions) is computed once at the beginning of our analysis from a set of independent single-pulsar analyses and held constant thereafter.

### III.3 Red and chromatic noise

While largely stable rotators, millisecond pulsars exhibit quasi-random walk behavior in their pulse phase, period, and spindown rate due to internal instabilities [79]. The resulting stochastic time-correlated delays are known as intrinsic pulsar achromatic red noise (RN) and denoted $\boldsymbol{\delta t}_{\text{RN}}$. The gravitational wave background also gives rise to a stochastic achromatic red signal in the TOAs, $\boldsymbol{\delta t}_{\text{GWB}}$, [72, 70, 101, 47]. These delays are modeled with a set of discrete Fourier modes,

$$ $\displaystyle(\boldsymbol{\delta t}_{\text{RN}}+\boldsymbol{\delta t}_{\text{GWB}})_{i}$ $\displaystyle=\sum_{k=1}^{N_{f}}\big[a_{k}\sin(2\pi k\,\mathbf{t}_{\text{TOA},i}/T)+$ $\displaystyle\hskip 34.1433ptb_{k}\cos(2\pi k\,\mathbf{t}_{\text{TOA},i}/T)\big]$ (14) $$

where $i\in\{1,2,\dots,n\}$ indexes the TOA observation, $T$ is the the total observation span over all pulsars in the array, $k\in\{1,2,\dots,N_{f}\}$ indexes the frequency bin, and $N_{f}$ is the number of frequency bins modeled. The zero-frequency term is a component of the per-pulsar timing model, so $k=0$ is not included in the red noise model. The $(n\times N_{f})$ Fourier design matrix consists of alternating columns of basis functions,

$$ $F_{ik}=\big\{\sin(2\pi k\,\mathbf{t}_{\text{TOA},i}/T)\,,\;\;\cos(2\pi k\,\mathbf{t}_{\text{TOA},i}/T)\big\}\,,$ (15) $$

and compactly represents the Fourier series when multiplying the vector of Fourier coefficients $\mathbf{a}=[a_{1},b_{1},a_{2},b_{2},\dots,a_{N_{f}},b_{N_{f}}]^{\text{T}}$. That is,

$$ $\boldsymbol{\delta t}_{\text{RN}}+\boldsymbol{\delta t}_{\text{GWB}}=\mathbf{F}\mathbf{a}\,.$ (16) $$

In some cases it is advantageous to use distinct sets of Fourier coefficients to represent the GWB and intrinsic pulsar RN, $\mathbf{a}_{\text{GWB}}$ and $\mathbf{a}_{\text{RN}}$, respectively. The methods presented in this paper are generalized to such cases in Appendix [C](#A3).

As the radio pulses travel from the pulsar to the Earth, they propagate through the turbulent interstellar medium, and undergo frequency-dependent dispersion [48, 56]. Stochastic fluctuations in the dispersion measure (DM) induces chromatic red noise in the timing residuals which is also modeled by a Fourier series,

$$ $\boldsymbol{\delta t}_{\text{DM}}=\mathbf{F}_{\text{DM}}\,\mathbf{a}_{\text{DM}}\,,$ (17) $$

where $\mathbf{F}_{\text{DM}}=\mathbf{F}/(K\nu^{2}_{\text{obs},i})$, $K=2.41\times 10^{-16}\;\text{Hz}^{-2}\text{cm}^{-3}\text{pc}\,\text{s}^{-1}$ is the dispersion constant, and $\nu_{\text{obs},i}$ is the radio frequency of the $i^{\text{th}}$ observation. We will neglect DM contributions throughout the rest of this paper for brevity. However, it is straightforward to add DM delays to the model. The Fourier series for DM should be appended to that of RN when desired: $\mathbf{F}\mathbf{a}\rightarrow(\mathbf{F}\mathbf{a},\mathbf{F}_{\text{DM}}\,\mathbf{a}_{\text{DM}})$.

### III.4 Deterministic signals and continuous waves

Besides those of the timing model, there are potentially other deterministic signals in PTA datasets. For example, if an individual SMBHB is particularly nearby or massive, it may be discernible from the background population. The binary radiates a near monochromatic continuous gravitational wave (CW) and is modeled deterministically [23, 57]. Other potential deterministic signals are GW bursts [32, 33], perturbations to the solar system ephemeris [22, 90], and the timing model itself. We’ll denote an arbitrary deterministic signal model as $\mathbf{h}$ with parameters $\boldsymbol{\theta}$. The deterministic signal’s contribution to the timing residuals is

$$ $\boldsymbol{\delta t}_{\text{det}}=\mathbf{h}(\boldsymbol{\theta})\,.$ (18) $$

In Sec. [VI](#S6), we analyze a simulated dataset containing timing delays induced by an individual SMBHB to demonstrate the deterministic signal modeling methods presented in this paper. The delays are simulated consistent with the CW model presented in Corbin and Cornish [23], Ellis et al. [28], Ellis [27], Taylor et al. [84] and depends on $8+2N_{p}$ parameters, where $N_{p}$ is the number of pulsars in the array. 8 parameters are those of the binary itself: $\{\mathcal{M},f_{\text{CW}},\iota,\psi,h,\theta,\phi,\Phi_{0}\}$ which correspond to chirp mass, frequency, inclination angle, polarization, characteristic strain, sky location, and a reference phase, respectively. The other $2N_{p}$ parameters are the pulsar distances, $L_{I}$, and the phase of the CW at every pulsar, $\Phi_{I}$, where $I\in\{1,2,\dots,N_{p}\}$ indexes the pulsars in the array. Note that the phase at each pulsar, $\Phi_{I}$, can be determined from the 8 binary and $N_{p}$ pulsar distance parameters. However, they are treated as independent parameters to smooth out the posterior geometry for ease of sampling [84, 23].

## IV Hierarchical inference in pulsar timing arrays

Many parameters of PTA analyses are not modeled hierarchically, i.e. they use static priors. However, the Fourier coefficients which describe the stochastic GWB, intrinsic pulsar noise, and dispersion in the interstellar medium do use a spectral hyper-model, with a set of hyper-parameters $\boldsymbol{\eta}$. We will generalize the multi-pulsar model to include deterministic signals in Sec. [IV.3](#S4.SS3), but will neglect such contributions for now. The hierarchical Bayesian model for PTAs is then

$$ $p(\boldsymbol{\epsilon},\mathbf{a},\boldsymbol{\eta}|\boldsymbol{\delta t})\propto p(\boldsymbol{\delta t}|\boldsymbol{\epsilon},\mathbf{a})\cdot p(\mathbf{a}|\boldsymbol{\eta})\cdot p(\boldsymbol{\epsilon})\,.$ (19) $$

Eq. ([III](#S3.Ex2)) may be rearranged to yield a realization of white noise, $\boldsymbol{\delta t}_{\text{WN}}=\boldsymbol{\delta t}-\boldsymbol{\delta t}_{\text{TM}}-\boldsymbol{\delta t}_{\text{RN}}-\boldsymbol{\delta t}_{\text{GWB}}$, and the white noise covariance matrix may be computed from initial estimates of EFAC, EQUAD, and ECORR parameter values. The Gaussian likelihood is

$$ $p(\boldsymbol{\delta t}|\boldsymbol{\epsilon},\mathbf{a})=\frac{1}{\sqrt{\text{det}(2\pi\mathbf{N}})}\,\text{exp}\bigg[-\frac{1}{2}\big(\boldsymbol{\delta t}-\mathbf{M}\boldsymbol{\epsilon}-\mathbf{F}\mathbf{a}\big)^{\text{T}}\,\mathbf{N}^{-1}\big(\boldsymbol{\delta t}-\mathbf{M}\boldsymbol{\epsilon}-\mathbf{F}\mathbf{a}\big)\bigg]\,,$ (20) $$

using Eq. ([12](#S3.E12)) and Eq. ([16](#S3.E16)). The likelihood is factorized per-pulsar, each pulsar being endowed with unique timing residuals and constituent models. The likelihood for the full PTA is the product of the individual pulsar likelihoods. Nonetheless, we will use Eq. ([20](#S4.E20)) to represent the likelihood for the entire PTA, understanding that the timing residuals and models are concatenated across pulsars.

### IV.1 Priors

The marginalized distribution of linear deviations to the timing model is dominated by the likelihood, so it is difficult to differentiate between a wide- and infinitely-wide prior. It is convention to model the linear deviations with a normal prior of zero mean and infinite variance, $\boldsymbol{\epsilon}_{i}\sim\mathcal{N}(0,\infty)$, where $i\in\{1,2,\dots,m\}$ indexes the parameters of the timing model. The prior on Fourier coefficients is a multivariate normal distribution of zero mean and covariance

$$ $\boldsymbol{\phi}_{IJ\,ij}=\alpha_{IJ}\rho_{i}\delta_{ij}+\delta_{IJ}\kappa_{Ii}\delta_{ij}$ (21) $$

where $I,J$ label pulsars in the array, $i,j$ label the frequency bin, and $\boldsymbol{\rho}$ and $\boldsymbol{\kappa}$ denote the power spectrum of the GWB and RN, respectively. There is no summation over repeated indices. The intrinsic pulsar red noise is uncorrelated between pulsars while the GWB obeys the Hellings-Downs (HD) correlation pattern [45]

$$ $\alpha_{IJ}=\frac{3}{2}\beta_{IJ}\ln\beta_{IJ}-\frac{1}{4}\beta_{IJ}+\frac{1}{2}+\frac{1}{2}\delta_{IJ}$ (22) $$

where $\beta_{IJ}=(1-\cos\Theta_{IJ})/2$ and $\Theta_{IJ}$ is the angle between pulsars $I$ and $J$ on the sky.

Arbitrary spectral models may be used to describe the intrinsic pulsar RN and stochastic GWB. One common choice is a free spectral model, where the power is allowed to vary freely in each frequency bin. That is, each $\rho_{i}$ (or $\kappa_{Ii}$) is itself a free (hyper-) parameter. Another choice is a power law spectral model, which parameterizes the spectrum with an amplitude and spectral index,

$$ $\rho_{i}(A,\gamma)=\frac{A^{2}}{12\pi^{2}}\frac{1}{T}\left(\frac{i/T}{1\,\text{yr}^{-1}}\right)^{-\gamma}\,\text{yr}^{2}\,,$ (23) $$

where $A$ and $\gamma$ are the amplitude and spectral index of the power spectrum, respectively. We choose a reference frequency $f_{\text{ref}}=\text{yr}^{-1}$. The power law may model the GWB common to all pulsars $(A_{\text{GWB}},\gamma_{\text{GWB}})$, or intrinsic RN $(A_{I},\gamma_{I})$ unique to each pulsar. Arbitrary spectral models are amenable to the methods presented in this paper and are in use throughout PTA analyses, see e.g. [74, 77]. The covariance matrix may even be parameterized by galaxy stellar mass functions, merger rates, and other astrophysical parameters from which the GWB spectrum is derived [4, 5]. Let $\boldsymbol{\eta}$ denote the hyper-parameters for an arbitrary hyper-model, describing the spectrum of both the stochastic GWB and pulsar RN. The prior on the Fourier coefficients is

$$ $p(\mathbf{a}|\boldsymbol{\eta})=\frac{1}{\sqrt{\text{det}(2\pi\boldsymbol{\phi})}}\,\text{exp}\bigg[-\frac{1}{2}\mathbf{a}^{\text{T}}\,\boldsymbol{\phi}^{-1}\,\mathbf{a}\bigg],$ (24) $$

where the covariance is determined from a set of hyper-parameters and choice of spectral model, $\boldsymbol{\phi}=\boldsymbol{\phi}(\boldsymbol{\eta})$.

Historically the power law, Eq. ([23](#S4.E23)), was favored for both intrinsic pulsar RN and GWB models in many PTA analyses. It was advantageous requiring only two hyper-parameters (per pulsar and GWB), keeping the dimensionality of the posterior relatively low. Moreover, general relativity predicts $\gamma_{\text{GWB}}=13/3$ if the background is realized via a population of circularly inspiraling binaries [72, 70, 101]. While the free spectral model is more flexible, it was previously restricted in applications due to the computational cost of its dimension (most analyses could only afford to model the GWB and a small subset of pulsars with a free spectrum). By assuming statistical independence across frequency bins, despite uneven sampling of the data in the time-domain inducing such correlations [59, 96, 85], factorized likelihood methods [88, 53] are able to efficiently perform free spectral analyses per pulsar, then reweight and refit the recovered posterior distribution to obtain the likelihood for the full array under arbitrary spectral models. Using the NUTS scheme and a GPU implementation, our method is more robust to the dimensionality of the posterior and we are able to efficiently model the GWB and intrinsic RN of every pulsar in the array with a free spectrum, without neglecting inter-frequency or -pulsar correlations, if desired.

If the stochastic background is realized via a population of a few loud SMBHBs, then it will exhibit higher-order statistical moments, beyond covariance [54, 55, 50, 75, 103, 71]. We will assume the Gaussian approximation, Eq. ([24](#S4.E24)), is sufficient in the main body. Non-Gaussian features are discussed in Appendix [D](#A4).

The hyper-parameters, $\boldsymbol{\eta}$, are subject to a hyper-prior. The power law amplitude will use a log-uniform hyper-prior, $\log_{10}A_{\text{GWB}}\sim\text{Uniform}(-20,-10)$, and the spectral index a uniform hyper-prior, $\gamma_{\text{GWB}}\sim\text{Uniform}(0,7)$. Under a free spectral model, we’ll use a log-uniform prior, $\log_{10}\rho_{i}\sim\text{Uniform}(-20,-5)$ for $i\in\{1,2,\dots,N_{f}\}$. Identical hyper-priors are used for the spectral models of intrinsic pulsar red noise [98].

### IV.2 Standardizing the pulsar timing posterior

The hierarchical PTA posterior, Eq. ([19](#S4.E19)), can be constructed by multiplying Eq. ([20](#S4.E20)), Eq. ([24](#S4.E24)), the prior on the timing model parameters, $\boldsymbol{\epsilon}_{i}\sim\mathcal{N}(0,\infty)$, and the hyper-prior on the parameters of the spectral model, $p(\boldsymbol{\eta})$. However, this is generally not the posterior sampled in standard analyses. Instead, the parameters of the linearized timing model are marginalized analytically from the hierarchical Bayesian model. This is useful not only because it reduces the dimension of the parameter space, but also because the timing model parameters are highly covariant with other signal and noise processes.

The analytic marginalization is equivalent to projecting the components of the analysis into a space orthogonal to the timing model, and is accomplished by replacing the inverse white noise covariance matrix $\mathbf{N}^{-1}\rightarrow\mathbf{\tilde{N}}^{-1}$, where

$$ $\mathbf{\tilde{N}}^{-1}=\mathbf{G}(\mathbf{G}^{\text{T}}\mathbf{N}\mathbf{G})^{-1}\mathbf{G}^{\text{T}}$ (25) $$

and $\mathbf{G}$ is built from the singular value decomposition (SVD) of the timing design matrix as in [94, 97, 95]. Alternatively, this projection can be accomplished using QR-decomposition,

$$ $\mathbf{M}=\mathbf{Q}\mathbf{R}=[\mathbf{Q}_{1},\,\mathbf{Q}_{2}]\begin{bmatrix}\mathbf{R}_{1}\\ \mathbf{0}\end{bmatrix}$ (26) $$

where $\mathbf{Q}$ is a $(n\times m)$ unitary matrix and $\mathbf{R}$ is a $(n\times m)$ upper triangular matrix with zeros populating the bottom $(n-m)$ rows, where $n$ and $m$ are the number of TOAs and timing model parameters in a particular pulsar, respectively. $\mathbf{Q}$ and $\mathbf{R}$ are partitioned such that $\mathbf{R}_{1}$ is a $(m\times m)$ upper triangular matrix, $\mathbf{0}$ is a $((n-m)\times m)$ zero matrix, and $\mathbf{Q}_{1}$ and $\mathbf{Q}_{2}$ with orthogonal columns are size $(n\times m)$ and $(n\times(n-m))$, respectively. Similar to the $\mathbf{G}$-matrix method, the columns of $\mathbf{Q}_{2}$ form an orthonormal basis for the subspace orthogonal to the timing model. Hence, $\mathbf{Q}_{2}^{\text{T}}\mathbf{M}=\mathbf{0}$ and we may replace $\mathbf{G}$ in Eq. ([25](#S4.E25)) with $\mathbf{Q}_{2}$. Using column pivoted QR-decomposition we may achieve this projection more efficiently than SVD and with greater numerical stability for the (nearly) rank-deficient timing model design matrix [82].

The posterior, analytically marginalized over linear deviations to the timing model, is then

$$ $p(\mathbf{a},\boldsymbol{\eta}|\boldsymbol{\delta t})\propto\frac{p(\boldsymbol{\eta})}{\sqrt{\text{det}(2\pi\tilde{\mathbf{N}})\text{det}(2\pi\boldsymbol{\phi})}}\,\text{exp}\bigg[-\frac{1}{2}\big(\boldsymbol{\delta t}-\mathbf{F}\mathbf{a}\big)^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\big(\boldsymbol{\delta t}-\mathbf{F}\mathbf{a}\big)-\frac{1}{2}\mathbf{a}^{\text{T}}\,\boldsymbol{\phi}^{-1}\,\mathbf{a}\bigg]\,.$ (27) $$

While written in the notation of per-pulsar analyses as introduced in Sec. [III](#S3), it is understood that Eq. ([27](#S4.E27)) models every pulsar in the array. That is, all constituent objects of the posterior above (and in what follows) are concatenated across pulsars in the array, except $\boldsymbol{\phi}$ whose definition, Eq. ([21](#S4.E21)), already spans every pulsar. e.g. $\boldsymbol{\delta t}\equiv[\boldsymbol{\delta t}_{(1)},\boldsymbol{\delta t}_{(2)},\dots,\boldsymbol{\delta t}_{(N_{p})}]^{\text{T}}$ where $\boldsymbol{\delta t}_{(I)}$ denotes the TOAs of the $I^{\text{th}}$ pulsar, and $I\in\{1,2,\dots,N_{p}\}$. With the white noise model fixed, evaluating Eq. ([27](#S4.E27)) generally scales as $\mathcal{O}(N_{p}n^{2})$ as it requires $N_{p}$ $(n\times n)$ matrix multiplications, assuming each of the $N_{p}$ pulsars has exactly $n$ TOAs.

Following the procedure of [59], Eq. ([27](#S4.E27)) can be written in a more efficient form

$$ $p(\mathbf{a},\boldsymbol{\eta}|\boldsymbol{\delta t})\propto\frac{p(\boldsymbol{\eta})}{\sqrt{\text{det}(2\pi\boldsymbol{\phi})}}\,\text{exp}\bigg[-\frac{1}{2}\big(\mathbf{a}-\mathbf{\hat{a}}\big)^{\text{T}}\,\boldsymbol{\Sigma}^{-1}\,\big(\mathbf{a}-\mathbf{\hat{a}}\big)+\frac{1}{2}\mathbf{\hat{a}}^{\text{T}}\,\boldsymbol{\Sigma}^{-1}\,\mathbf{\hat{a}}\bigg]$ (28) $$

where $\mathbf{\hat{a}}=\boldsymbol{\Sigma}\mathbf{F}^{\text{T}}\mathbf{\tilde{N}}^{-1}\boldsymbol{\delta t}$, $\boldsymbol{\Sigma}^{-1}=\mathbf{F}^{\text{T}}\mathbf{\tilde{N}}^{-1}\mathbf{F}+\boldsymbol{\phi}^{-1}$, and we drop the $\boldsymbol{\delta t}^{\text{T}}\tilde{\mathbf{N}}^{-1}\boldsymbol{\delta t}$ term in the exponent and neglect the normalization factor $\text{det}(2\pi\tilde{\mathbf{N}})$. The white noise model is fixed in our analysis and these terms amount to constant multiplicative factors which do not influence the recovery of the target distribution. While mathematically equivalent, Eq. ([28](#S4.E28)), is significantly more efficient than Eq. ([27](#S4.E27)). $\mathbf{F}^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\boldsymbol{\delta t}$ and $\mathbf{F}^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\mathbf{F}$ can be computed once and stored for future evaluations, so Eq. ([28](#S4.E28)) requires only matrix multiplications of size $(2N_{f}N_{p}\times 2N_{f}N_{p})$. As the signal components of PTA datasets are red, relatively few low frequency bins are required to store signal information, and the Fourier domain is a compressed representation of PTA datasets, hence $N_{f}\ll n$. Posteriors that respect this compression, such as Eq. ([28](#S4.E28)), will be more computationally efficient than posteriors using alternative representations.

At this stage, most analyses analytically integrate Eq. ([28](#S4.E28)) with respect to the Fourier coefficients [c.f. [59]] marginalizing them from the model. The resulting density, while significantly lower-dimensional, contains a large $(N\times N)$ covariance matrix, where $N$ is the total number of TOAs for all pulsars in the array. This covariance matrix is dense due to inter-pulsar correlations from the GWB, and must be inverted for every posterior evaluation. The efficiency of the inversion is improved using the Woodbury matrix identity [100] but bottlenecks the analysis nonetheless. Rather than analytically marginalizing over the Fourier coefficients, we will keep Eq. ([28](#S4.E28)) in its present form, and sample the Fourier coefficients numerically. Eq. ([28](#S4.E28)) is computationally cheap to evaluate, requiring no expensive matrix inversions, but is high-dimensional due to the large number of Fourier coefficients. The inference on the spectral hyper-parameters $\boldsymbol{\eta}$, however, is equivalent to the analytic approach. It’s simply a question of when the marginalization is performed: analytically at the posterior evaluation, or numerically via sampling and Monte Carlo integration.

The hierarchical posterior above, Eq. ([28](#S4.E28)), is plagued by Neal’s funnel. This can be seen in the conditional prior on the Fourier coefficients, Eq. ([24](#S4.E24)). Say a power law spectral model is used to parameterize the covariance matrix, $\boldsymbol{\phi}$, a common modeling choice for PTA analyses. When the spectral index is large (small), the power is relatively constrained (free) in high frequency bins, and the respective coefficients to have small (large) variance. This forms a funnel analogous to that of Sec. [II.3](#S2.SS3), but one which is significantly more difficult to sample directly, being high-dimensional across coefficients and hyper-parameters. The sharpness of the funnel may be reduced if a free spectral model is used to describe all red processes as in [42]. However, Eq. ([28](#S4.E28)) is generally very difficult to sample directly which is why previous approaches have opted to analytically marginalize the Fourier coefficients from the model: not only is the dimension of the parameter space reduced, but the funnel is absent from the marginalized posterior geometry.

Rather than sampling Eq. ([28](#S4.E28)) directly, we will sample the Fourier coefficients under a standardizing transform. This eases the difficulty of exploring Neal’s funnel, while maintaining the hyper-efficient posterior evaluation. The high-dimensional parameter space is efficiently explored with HMC using a NUTS scheme. To perform a standardizing transform on the coefficients, we first estimate their mean and covariance. Examining Eq. ([28](#S4.E28)), the mean and covariance of the Fourier coefficients are approximately $\mathbf{\hat{a}}$ and $\boldsymbol{\Sigma}$, respectively. This can be verified under the Laplace approximation by computing the maximum a posteriori (MAP) solution for the Fourier coefficients, as in [59], and identifying it with the mean of the distribution. The covariance can be estimated using the Hessian of the log-posterior: $-\partial_{\mathbf{a}}\partial_{\mathbf{a}}\ln p(\mathbf{a},\boldsymbol{\eta}|\boldsymbol{\delta t})|_{\hat{\mathbf{a}}}=\boldsymbol{\Sigma}^{-1}$. The standardizing coordinate transform, Eq. ([3](#S2.E3)), for the Fourier coefficients in the PTA case is then

$$ $(\mathbf{a},\;\boldsymbol{\eta})=T^{-1}(\mathbf{z},\boldsymbol{\eta})=(\mathbf{\hat{a}}+\mathbf{L}\mathbf{z},\;\boldsymbol{\eta})$ (29) $$

where $\mathbf{L}$ is the Cholesky decomposition of the covariance matrix, $\boldsymbol{\Sigma}=\mathbf{L}\mathbf{L}^{\text{T}}$. As in Sec. [II.3](#S2.SS3), the mean and covariance used in the standardizing transform depend on the hyper-parameters, $\hat{\mathbf{a}}=\hat{\mathbf{a}}(\boldsymbol{\eta})$ and $\boldsymbol{\Sigma}=\boldsymbol{\Sigma}(\boldsymbol{\eta})$.

In practice, it is expensive to compute the Cholesky decomposition of the covariance matrix $\boldsymbol{\Sigma}$ which includes inter-pulsar correlations (^1^11Ironically, after using the Woodbury identity, the Cholesky decomposition of $\boldsymbol{\Sigma}$ is the computational bottleneck for evaluating the posterior in which the Fourier coefficients have been analytically marginalized. Our approach would be no faster than standard methods if we used the exact (inter-pulsar-correlated) Cholesky decomposition in the standardizing transform.). However, the standardizing transformation need not be exact, and we may approximate the covariance in favor of a more computationally efficient standardizing transform. We choose to approximate the covariance matrix to that of a common uncorrelated red noise (CURN) process. That is, inter-pulsar correlations are neglected in the stochastic GWB model allowing us to factor the Cholesky decomposition per-pulsar. The Cholesky decomposition of the CURN covariance is $\boldsymbol{\Sigma}_{\text{CURN}}=(\mathbf{F}^{\text{T}}\mathbf{\tilde{N}}^{-1}\mathbf{F}+\boldsymbol{\phi}^{-1}_{\text{CURN}})^{-1}=\mathbf{L}_{\text{CURN}}\mathbf{L}_{\text{CURN}}^{\text{T}}$, where the prior covariance $\boldsymbol{\phi}_{\text{CURN}}$ is identical to that of Eq. ([21](#S4.E21)), but $\alpha_{IJ}=\delta_{IJ}$, neglecting inter-pulsar correlations. Similarly, we define $\hat{\mathbf{a}}_{\text{CURN}}=\boldsymbol{\Sigma}_{\text{CURN}}\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\boldsymbol{\delta t}$.

While only an approximation, the CURN model is the primary component of the stochastic GWB, and effectively de-correlates the parameter space. The HD inter-pulsar correlation, Eq. ([22](#S4.E22)), has a maximum correlation of 0.5 and is sub-dominant to the CURN model. In other words, Eq. ([21](#S4.E21)) is diagonal-dominant. This has been seen empirically in the latest published datasets such as the NANOGrav 15-year release, [8], which found a Bayes factor of $\sim 10^{12}$ in favor of a model containing intrinsic pulsar RN and a CURN GWB over a model containing only pulsar noise. Meanwhile, a Bayes factor of $\sim 10^{2}$ was found in favor of a HD correlated GWB over a CURN model, suggesting the GWB is dominated by diagonal CURN contributions, and only weakly influenced by off-diagonal inter-pulsar correlations. Hence the CURN model can be used in the standardizing transform to approximately de-correlate the parameter space.

The approximate and computationally efficient standardizing transform used in practice is then

$$ $(\mathbf{a},\;\boldsymbol{\eta})=T^{-1}(\mathbf{z},\boldsymbol{\eta})=(\mathbf{\hat{a}}_{\text{CURN}}+\mathbf{L}_{\text{CURN}}\,\mathbf{z},\;\boldsymbol{\eta})$ (30) $$

and the standardized density which is sampled instead of Eq. ([28](#S4.E28)) is

$$ $\tilde{p}(\mathbf{z},\boldsymbol{\eta}|\boldsymbol{\delta t})\propto\frac{p(\boldsymbol{\eta})\cdot\text{det}(\mathbf{L}_{\text{CURN}})}{\sqrt{\text{det}(2\pi\boldsymbol{\phi})}}\,\text{exp}\bigg[-\frac{1}{2}\big(\hat{\mathbf{a}}_{\text{CURN}}+\mathbf{L}_{\text{CURN}}\,\mathbf{z}-\mathbf{\hat{a}}\big)^{\text{T}}\,\boldsymbol{\Sigma}^{-1}\,\big(\hat{\mathbf{a}}_{\text{CURN}}+\mathbf{L}_{\text{CURN}}\,\mathbf{z}-\mathbf{\hat{a}}\big)+\frac{1}{2}\mathbf{\hat{a}}^{\text{T}}\,\boldsymbol{\Sigma}^{-1}\,\mathbf{\hat{a}}\bigg]\,,$ (31) $$

where we’ve been careful to include the determinant of the Jacobian of the transformation. While we perform the coordinate transformation with respect to a CURN model, Eq. ([31](#S4.E31)) enforces the inter-pulsar correlations through the presence of $\boldsymbol{\Sigma}$ so arbitrary inter-pulsar correlations (e.g. HD inter-pulsar correlations) can be modeled efficiently. Again the transformation, neglecting inter-pulsar correlations, does not perfectly de-correlate the parameter space. However, the estimated moments $\hat{\mathbf{a}}_{\text{CURN}}=\hat{\mathbf{a}}_{\text{CURN}}(\boldsymbol{\eta})$ and $\boldsymbol{\Sigma}_{\text{CURN}}=\boldsymbol{\Sigma}_{\text{CURN}}(\boldsymbol{\eta})$ are sufficient such that $\mathbf{z}$ approximately obeys a standard normal distribution. The true Fourier coefficients, $\mathbf{a}$, may be generated from our samples in $\mathbf{z}$ using Eq. ([30](#S4.E30)).

Eq. ([31](#S4.E31)) is the main result of this paper. It’s worth noting that after sampling and mapping back to the original Fourier coefficients, our inference is identical to standard techniques. However, the standardized posterior does not require the inversion of any large dense matrices, retaining the hyper-efficient posterior formulation first presented in Lentati et al. [59]. While standard posterior formulations analytically marginalize over the Fourier coefficients, we keep them as model parameters and sample them, performing the marginalization numerically. This means we sample over thousands of more parameters than standard approaches. As we’ve removed Neal’s funnel via the coordinate transformation Eq. ([30](#S4.E30)), the extra parameters $\mathbf{z}$ approximately obey a standard normal distribution. Such approximately normal high-dimensional distributions may be sampled extremely efficiently with HMC algorithms using fast automatic differentiation and XLA (Accelerated Linear Algebra) methods with a GPU-backend. The main results of this paper are obtained by implementing Eq. ([31](#S4.E31)) in the JAX [19] package and performing HMC sampling with the NumPyro [69, 18] package on a GPU, see Section [VI](#S6).

### IV.3 Generalizing to include deterministic signals

It is straightforward to generalize the posterior, Eq. ([27](#S4.E27)), to include deterministic signals. If the timing delays are modeled with a deterministic contribution $\mathbf{h}$, parameterized by $\boldsymbol{\theta}$, the posterior is modified

$$ $p(\mathbf{a},\boldsymbol{\eta},\boldsymbol{\theta}|\boldsymbol{\delta t})\propto\frac{p(\boldsymbol{\eta})\cdot p(\boldsymbol{\theta})}{\sqrt{\text{det}(2\pi\mathbf{\tilde{N}})\text{det}(2\pi\boldsymbol{\phi})}}\,\text{exp}\bigg[-\frac{1}{2}\bigg(\boldsymbol{\delta t}-\mathbf{F}\mathbf{a}-\mathbf{h}(\boldsymbol{\theta})\bigg)^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\bigg(\boldsymbol{\delta t}-\mathbf{F}\mathbf{a}-\mathbf{h}(\boldsymbol{\theta})\bigg)-\frac{1}{2}\mathbf{a}^{\text{T}}\,\boldsymbol{\phi}^{-1}\,\mathbf{a}\bigg]\,,$ (32) $$

where $p(\boldsymbol{\theta})$ is the prior on the parameters of the deterministic model. We will follow [43] and express the deterministic model in a Fourier basis, so the deterministic delays are expressed as

$$ $\mathbf{h}(\boldsymbol{\theta})=\mathbf{F}_{\text{D}}\,\mathbf{a}_{\text{D}}(\boldsymbol{\theta})\;\longleftrightarrow\;\mathbf{a}_{\text{D}}(\boldsymbol{\theta})=\mathfrak{F}[\mathbf{h}(\boldsymbol{\theta})]\,,$ (33) $$

where $\mathbf{F}_{\text{D}}$ is the Fourier design matrix containing the basis and $\mathbf{a}_{\text{D}}$ the representation of the deterministic signal in a Fourier space. $\mathfrak{F}[\mathbf{h}(\boldsymbol{\theta})]$ denotes the discrete Fourier transform of the time-domain deterministic model, which may be performed analytically or numerically with a fast Fourier transform (FFT). A non-evolving continuous wave signal has a simple analytic Fourier transform, while more sophisticated deterministic delays may require numerical techniques.

The Fourier basis used for deterministic signals need not be identical to that of the stochastic components, $\mathbf{F}_{\text{D}}\neq\mathbf{F}$. In fact, it is advantageous to use a distinct basis as standard analyses define the Fourier basis for stochastic components with respect to the observation span of the array: the lowest frequency resolved is $1/T$, where $T$ is the total duration of observations. As deterministic models are generally not periodic over this window, it is preferable to use an extended basis to avoid biases induced by Gibbs phenomena [38] (see Appendix A of Gundersen and Cornish [43] for a discussion).

Replacing the deterministic model with its Fourier representation, the generalized posterior, Eq. ([32](#S4.E32)), can be written as

$$ $\displaystyle p(\mathbf{a},\boldsymbol{\eta},\boldsymbol{\theta}|\boldsymbol{\delta t})\propto\frac{p(\boldsymbol{\eta})\cdot p(\boldsymbol{\theta})}{\sqrt{\text{det}(2\pi\boldsymbol{\phi})}}\,\text{exp}\bigg[-$ $\displaystyle\frac{1}{2}\big(\mathbf{a}-\mathbf{\hat{a}})^{\text{T}}\,\boldsymbol{\Sigma}^{-1}\,(\mathbf{a}-\mathbf{\hat{a}})+\frac{1}{2}\mathbf{\hat{a}}^{\text{T}}\,\boldsymbol{\Sigma}^{-1}\,\mathbf{\hat{a}}$ $\displaystyle+\boldsymbol{\delta t}^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\mathbf{F}_{\text{D}}\,\mathbf{a}_{\text{D}}-\mathbf{a}^{\text{T}}\,\mathbf{F}^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\mathbf{F}_{\text{D}}\,\mathbf{a}_{\text{D}}-\frac{1}{2}\mathbf{a}_{\text{D}}^{\text{T}}\,\mathbf{F}_{\text{D}}^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\mathbf{F}_{\text{D}}\,\mathbf{a}_{\text{D}}\bigg]\,.$ (34) $$

Differing from those of the stochastic model, the Fourier coefficients for the deterministic model are not sampled directly or under a bijective coordinate transformation. Instead the parameters of the deterministic model, $\boldsymbol{\theta}$, are sampled and the coefficients are the Fourier transform of the corresponding time-domain deterministic signal, $\mathbf{h}=\mathbf{h}(\boldsymbol{\theta})$, so there exists the mapping $\mathbf{a}_{\text{D}}=\mathbf{a}_{\text{D}}(\boldsymbol{\theta})$. In similar fashion to the pure stochastic model above, Eq. ([IV.3](#S4.Ex4)) is more computationally efficient to evaluate than the equivalent Eq. ([32](#S4.E32)). Again, this is because large matrix multiplications over all TOAs are replaced with multiplications over the compressed Fourier basis, and relevant inner products (e.g. $\boldsymbol{\delta t}^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\mathbf{F}_{\text{D}}$, $\mathbf{F}^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\mathbf{F}_{\text{D}}$, etc.) are computed once and stored for future evaluations.

Another computational overhead is the conversion of the deterministic model into a Fourier representation. The numerical FFT is generally computationally cheaper than Eq. ([32](#S4.E32)), where the deterministic model must be evaluated over every observed TOA. Say the same frequency resolution, $N_{f}$, is desired in the deterministic model as in the stochastic model. Then we need only query the deterministic model $\mathcal{O}(N_{f})$ times and a FFT yields the frequency representation in $\mathcal{O}(N_{f}\log N_{f})$ operations. Generally $N_{f}\ll n$ in PTA analysis, so obtaining the frequency-domain representation of the deterministic signal is not a significant computational cost, relative to the cost of matrix multiplications in the posterior.

In the case $\mathbf{F}_{\text{D}}=\mathbf{F}$, Eq. ([IV.3](#S4.Ex4)) reduces to

$$ $p(\mathbf{a},\boldsymbol{\eta},\boldsymbol{\theta}|\boldsymbol{\delta t})\propto\frac{p(\boldsymbol{\eta})\cdot p(\boldsymbol{\theta})}{\sqrt{\text{det}(2\pi\boldsymbol{\phi})}}\,\text{exp}\bigg[-\frac{1}{2}\big(\mathbf{a}+\mathbf{a}_{\text{D}}-\mathbf{\hat{a}})^{\text{T}}\,\boldsymbol{\Sigma}^{-1}\,(\mathbf{a}+\mathbf{a}_{\text{D}}-\mathbf{\hat{a}})+\frac{1}{2}\mathbf{\hat{a}}^{\text{T}}\,\boldsymbol{\Sigma}^{-1}\,\mathbf{\hat{a}}+\mathbf{a}^{\text{T}}\,\boldsymbol{\phi}^{-1}\,\mathbf{a}_{\text{D}}+\frac{1}{2}\mathbf{a}^{\text{T}}_{\text{D}}\,\boldsymbol{\phi}^{-1}\,\mathbf{a}_{\text{D}}\bigg]\,,$ (35) $$

which is similar to the posterior for a purely stochastic model, Eq. ([28](#S4.E28)). The extra additive terms in the exponent are corrections so the deterministic signal is not influenced hierarchically by the spectral model, which is solely intended for stochastic contributions.

We may generalize the standardizing transform presented for the stochastic model above, Eq. ([29](#S4.E29)), to include contributions from deterministic signals. The goal is to estimate the (conditional) mean and covariance of the stochastic Fourier coefficients from the posterior generalized to include deterministic components, Eq. ([IV.3](#S4.Ex4)). The presence of a deterministic signal shifts the MAP solution from $\mathbf{\hat{a}}$, of the purely stochastic case, to

$$ $\mathbf{\bar{a}}=\mathbf{\hat{a}}-\boldsymbol{\Sigma}\mathbf{F}^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\mathbf{F}_{\text{D}}\,\mathbf{a}_{\text{D}}=\boldsymbol{\Sigma}\mathbf{F}^{\text{T}}\,\mathbf{\tilde{N}}^{-1}(\boldsymbol{\delta t}-\mathbf{h})$ (36) $$

which can be derived from the MAP condition, $\partial_{\mathbf{a}}\ln p(\mathbf{a},\boldsymbol{\eta},\boldsymbol{\theta})\big|_{\mathbf{a}=\mathbf{\bar{a}}}=\mathbf{0}$. Intuitively, to include deterministic contributions, one needs only subtract the deterministic signal from the data, then compute the MAP solution as in Sec. [IV.2](#S4.SS2). The covariance of the stochastic Fourier coefficients is unchanged by the presence of deterministic signals as seen by the Hessian $-\partial_{\mathbf{a}}\partial_{\mathbf{a}}\ln p(\mathbf{a},\boldsymbol{\eta},\boldsymbol{\theta})=\boldsymbol{\Sigma}^{-1}$.

As in Sec. [IV.2](#S4.SS2), we will approximate the standardizing transform by neglecting inter-pulsar correlations with the CURN model. The generalization of Eq. ([30](#S4.E30)) to include deterministic contributions is

$$ $(\mathbf{a},\;\boldsymbol{\eta},\;\boldsymbol{\theta})=T^{-1}(\mathbf{z},\boldsymbol{\eta},\boldsymbol{\theta})=(\mathbf{\bar{a}}_{\text{CURN}}+\mathbf{L}_{\text{CURN}}\,\mathbf{z},\;\boldsymbol{\eta},\;\boldsymbol{\theta})$ (37) $$

where $\bar{\mathbf{a}}_{\text{CURN}}=\mathbf{\hat{a}}_{\text{CURN}}-\boldsymbol{\Sigma}_{\text{CURN}}\mathbf{F}^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\mathbf{F}_{\text{D}}\,\mathbf{a}_{\text{D}}$. While mapped trivially by the coordinate transformation, the hyper-parameters and parameters of the deterministic model determine the standardizing transformation performed on the coefficients, $\bar{\mathbf{a}}_{\text{CURN}}=\bar{\mathbf{a}}_{\text{CURN}}(\boldsymbol{\eta},\boldsymbol{\theta})$ and $\boldsymbol{\Sigma}_{\text{CURN}}=\boldsymbol{\Sigma}_{\text{CURN}}(\boldsymbol{\eta})$. The standardized posterior sampled is

$$ $\displaystyle\tilde{p}(\mathbf{z},\boldsymbol{\eta},\boldsymbol{\theta}|\boldsymbol{\delta t})\propto\frac{p(\boldsymbol{\eta})\cdot p(\boldsymbol{\theta})\cdot\text{det}(\mathbf{L}_{\text{CURN}})}{\sqrt{\text{det}(2\pi\boldsymbol{\phi})}}\,\text{exp}\bigg[$ $\displaystyle-\frac{1}{2}\big(\mathbf{\bar{a}}_{\text{CURN}}+\mathbf{L}_{\text{CURN}}\,\mathbf{z}-\mathbf{\hat{a}})^{\text{T}}\,\boldsymbol{\Sigma}^{-1}\,(\mathbf{\bar{a}}_{\text{CURN}}+\mathbf{L}_{\text{CURN}}\,\mathbf{z}-\mathbf{\hat{a}})$ $\displaystyle+\frac{1}{2}\mathbf{\hat{a}}^{\text{T}}\,\boldsymbol{\Sigma}^{-1}\,\mathbf{\hat{a}}+\boldsymbol{\delta t}^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\mathbf{F}_{\text{D}}\,\mathbf{a}_{\text{D}}-\frac{1}{2}\mathbf{a}_{\text{D}}^{\text{T}}\,\mathbf{F}_{\text{D}}^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\mathbf{F}_{\text{D}}\,\mathbf{a}_{\text{D}}$ $\displaystyle-(\mathbf{\bar{a}}_{\text{CURN}}+\mathbf{L}_{\text{CURN}}\,\mathbf{z})^{\text{T}}\,\mathbf{F}^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\mathbf{F}_{\text{D}}\,\mathbf{a}_{\text{D}}\bigg]\,.$ (38) $$

Again, we’re careful to include the determinant of the Jacobian of the transformation. As in Sec. [IV.2](#S4.SS2), the standardizing transformation does not perfectly de-correlate the parameter space, but is computationally efficient and a sufficient estimate so the coefficients $\mathbf{z}$ approximately obey a standard normal distribution. While the standardizing transformation neglects inter-pulsar correlations, Eq. ([IV.3](#S4.Ex5)) does include such correlations so our inference is identical to standard analyses. The original Fourier coefficients can be generated from the transformed samples using Eq. ([37](#S4.E37)).

### IV.4 Generalizing to include inter-frequency correlations

Most spectral models, including the power law Eq. ([23](#S4.E23)) and free spectral model, assume the Fourier modes are uncorrelated. However, because PTA data is unevenly sampled in the time-domain, the inferred Fourier coefficients will inevitably be correlated [59, 96]. Moreover, as the signal and noise processes we measure persist longer than our finite observation span, a rectangular window has effectively been applied to the data. The Fourier representation of the signal is then the convolution of the of the original Fourier expansion with the Fourier transform of the window function - a cardinal sine function, inducing additional correlations between the Fourier modes [24, 11, 12]. The Fourier modes were assumed orthogonal in previous sections, Eq. ([21](#S4.E21)) being diagonal in frequency-space. This section illustrates how our coefficient sampling method is compatible with non-diagonal prior covariance matrices.

In the evaluation of the posterior, Eq. ([28](#S4.E28)), we must compute the inverse of the $(2N_{f}N_{p}\times 2N_{f}N_{p})$ prior covariance matrix, $\boldsymbol{\phi}$. When inter-frequency correlations are neglected, this can be accomplished efficiently by batching our inverse over frequency bins, and inverting $2N_{f}$ $(N_{p}\times N_{p})$ matrices in parallel. This is not possible when including inter-frequency correlations because $\boldsymbol{\phi}$ is dense in inter-pulsar and -frequency correlations, and a straightforward inversion is computationally expensive. To avoid this bottleneck, we will separate the GWB and pulsar RN into separate Gaussian processes as in Appendix [C](#A3), where the prior covariance matrix for the coefficients which represent the background and intrinsic pulsar noise is $\boldsymbol{\phi}_{\text{GWB}}$ and $\boldsymbol{\phi}_{\text{RN}}$, respectively. The posterior for this parameterization is derived in Appendix [C](#A3) at Eq. ([63](#A3.E63)). The evaluation of this posterior requires us to invert both $\boldsymbol{\phi}_{\text{RN}}$ and $\boldsymbol{\phi}_{\text{GWB}}$, but we can compute these inverses more efficiently than that of the combined covariance matrix.

As the intrinsic pulsar noise, by definition, is independent across pulsars, $\boldsymbol{\phi}_{\text{RN}}$ is block-diagonal (or rather diagonal in pulsar-space, but dense in frequency-space due to windowing effects). We can therefore compute $\boldsymbol{\phi}_{\text{RN}}^{-1}$ by batching our matrix inversion across pulsars, and inverting $N_{p}$ $(2N_{f}\times 2N_{f})$ matrices in parallel. To compute $\boldsymbol{\phi}_{\text{GWB}}^{-1}$ efficiently, we note the GWB contribution to Eq. ([21](#S4.E21)) can be written as $\boldsymbol{\phi}_{\text{GWB}}=\boldsymbol{\Gamma}\otimes\boldsymbol{\varphi}$, where $\boldsymbol{\Gamma}$ is the overlap reduction function encoding inter-pulsar correlations and $\boldsymbol{\varphi}$ the common power spectrum across pulsars. That is, $\boldsymbol{\phi}_{\text{GWB}}$ remains dense in pulsar-space (due to $\Gamma$ being $(N_{p}\times N_{p})$ dense) and frequency-space (due to $\boldsymbol{\varphi}$ being $(2N_{f}\times 2N_{f})$ dense). Thanks to its Kronecker structure, the inverse may still be computed cheaply, $\boldsymbol{\phi}_{\text{GWB}}^{-1}=\boldsymbol{\Gamma}^{-1}\otimes\boldsymbol{\varphi}^{-1}$, requiring only the inversion of one $(N_{p}\times N_{p})$ and one $(2N_{f}\times 2N_{f})$ matrix.

We are now in a position to sample Eq. ([63](#A3.E63)), with $\boldsymbol{\phi}_{\text{GWB/RN}}$ including inter-frequency correlations, under the standardizing transformation Eq. ([66](#A3.E66)). However, recall in computing the inverse prior covariance matrix $\boldsymbol{\phi}_{\text{RN}}$ we must invert $N_{p}$ dense $(2N_{f}\times 2N_{f})$ matrices. This is precisely the same computational cost as analytically marginalizing over the Fourier coefficients which represent the intrinsic pulsar red noise. Therefore, we might as well perform this analytic marginalization so we only have to numerically sample the remaining coefficients representing the GWB. This choice also reduces the computational cost of the standardizing transform. Completing the square, Eq. ([63](#A3.E63)) is rewritten

$$ $\displaystyle p(\mathbf{a}_{\text{GWB}},\mathbf{a}_{\text{RN}},\boldsymbol{\eta}|\boldsymbol{\delta t})\propto$ $\displaystyle\frac{p(\boldsymbol{\eta})}{\sqrt{\text{det}(2\pi\tilde{\mathbf{N}})\cdot\text{det}(2\pi\boldsymbol{\phi}_{\text{GWB}})\cdot\text{det}(2\pi\boldsymbol{\phi}_{\text{RN}})}}$ (39) $\displaystyle\times\text{exp}\bigg[-\frac{1}{2}(\mathbf{a}_{\text{RN}}-\grave{\mathbf{a}}_{\text{RN}})^{\text{T}}\,\boldsymbol{\Sigma}_{\text{RN}}^{-1}\,(\mathbf{a}_{\text{RN}}-\grave{\mathbf{a}}_{\text{RN}})+\frac{1}{2}\grave{\mathbf{a}}_{\text{RN}}^{\text{T}}\,\boldsymbol{\Sigma}_{\text{RN}}^{-1}\,\grave{\mathbf{a}}_{\text{RN}}\bigg]$ $\displaystyle\times\text{exp}\bigg[-\frac{1}{2}\mathbf{a}_{\text{GWB}}^{\text{T}}\,\boldsymbol{\Sigma}_{\text{GWB}}^{-1}\,\mathbf{a}_{\text{GWB}}+\boldsymbol{\delta t}^{\text{T}}\,\tilde{\mathbf{N}}^{-1}\,\mathbf{F}\,\mathbf{a}_{\text{GWB}}\bigg]$ $$

where as in Appendix [C](#A3) $\boldsymbol{\Sigma}_{\text{GWB/RN}}=\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}+\boldsymbol{\phi}_{\text{GWB/RN}}$ and $\grave{\mathbf{a}}=\boldsymbol{\Sigma}_{\text{RN}}(\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\boldsymbol{\delta t}-\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}\mathbf{a}_{\text{GWB}})$ represents the MAP coefficients representing intrinsic pulsar noise, conditioned on the coefficients representing the background. In this form, the posterior is Gaussian in the red noise coefficients and we may analytically marginalize them from the model. The resulting posterior is

$$ $\displaystyle p(\mathbf{a}_{\text{GWB}},\boldsymbol{\eta}|\boldsymbol{\delta t})$ $\displaystyle\propto\int p(\mathbf{a}_{\text{GWB}},\mathbf{a}_{\text{RN}},\boldsymbol{\eta}|\boldsymbol{\delta t})\,d\mathbf{a}_{\text{RN}}$ (40) $\displaystyle\propto\sqrt{\frac{\text{det}(\boldsymbol{\Sigma}_{\text{RN}})}{\text{det}(\tilde{\mathbf{N}})\cdot\text{det}(\boldsymbol{\phi}_{\text{RN}})\cdot\text{det}(\boldsymbol{\phi}_{\text{GWB}})}}\times\text{exp}\bigg[-\frac{1}{2}\mathbf{a}_{\text{GWB}}^{\text{T}}\,\boldsymbol{\Sigma}^{-1}_{\text{GWB}}\,\mathbf{a}_{\text{GWB}}+\boldsymbol{\delta t}^{\text{T}}\,\tilde{\mathbf{N}}^{-1}\,\mathbf{F}\,\mathbf{a}_{\text{GWB}}$ $\displaystyle\hskip 227.62204pt+\frac{1}{2}\grave{\mathbf{a}}_{\text{RN}}^{\text{T}}\,\boldsymbol{\Sigma}_{\text{RN}}\,\grave{\mathbf{a}}_{\text{RN}}\bigg]\,.$ $$

To sample the remaining Fourier coefficients which represent the GWB efficiently, we derive the appropriate standardizing transformation. The mean and covariance (estimated under the Laplace approximation as above) are $\grave{\mathbf{a}}_{\text{GWB}}=\boldsymbol{\Phi}_{\text{GWB}}(\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\boldsymbol{\delta t}-\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}\boldsymbol{\Sigma}_{\text{RN}}\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\boldsymbol{\delta t})$ and $\boldsymbol{\Phi}_{\text{GWB}}=(\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}+\boldsymbol{\phi}^{-1}_{\text{GWB}}-\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}\boldsymbol{\Sigma}_{\text{RN}}\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F})^{-1}$, respectively. The standardizing transformation, Eq. ([29](#S4.E29)) is modified under the replacement $\hat{\mathbf{a}}\rightarrow\grave{\mathbf{a}}$ and $\boldsymbol{\Sigma}\rightarrow\boldsymbol{\Phi}_{\text{GWB}}$ to

$$ $(\mathbf{a}_{\text{GWB}},\boldsymbol{\eta})=T^{-1}(\mathbf{z},\boldsymbol{\eta})=(\grave{\mathbf{a}}_{\text{GWB}}+\mathbf{L}\mathbf{z},\boldsymbol{\eta)}$ (41) $$

where $\mathbf{L}$ is the Cholesky decomposition of the covariance matrix $\boldsymbol{\Phi}_{\text{GWB}}$ under the CURN approximation. We do, however, include inter-frequency correlations in the construction of the transformation.

In practice, the inter-frequency correlated prior covariance matrix can be numerically unstable or computationally costly to compute. However, the FFTInt approach of [24] approximates the time-domain covariance matrix efficiently and accurately by interpolating the matrix from a sparse grid of regular time samples to the observed TOAs. We adopt this approach and obtain the necessary frequency-domain covariance matrix, $\boldsymbol{\phi}$, using a two-dimensional FFT. The frequency-domain covariance with and without frequency correlations from window effects evaluated using a power law spectral model is shown in Fig. [2](#S4.F2). Estimating inter-frequency correlations requires the spectral model to have a continuous power spectral density function (e.g. a power law is continuous Eq. ([23](#S4.E23))). Other models like the free spectral model are not endowed with a continuous definition. Nonetheless, we may approximate the inter-frequency correlations under these models using a spline or some interpolation scheme to achieve a continuum limit.

Figure: Figure 2: Frequency-domain prior covariance matrices, $\boldsymbol{\phi}$, with and without frequency correlations from window effects. Both are evaluated at the same power law parameters $\log_{10}A=-14.5$ and $\gamma=3.0$. The tick labels on the x- and y-axes correspond to the frequency bin and corresponding Fourier mode, e.g. “3c” corresponds to the cosine mode amplitude in the third frequency bin and “3s” the sine amplitude of the same bin.
Refer to caption: https://arxiv.org/html/2607.06834/2607.06834v1/figs/cov.png

In summary, Eq. ([40](#S4.E40)) is a posterior, equivalent to Eq. ([28](#S4.E28)) after marginalizing over the Fourier coefficients representing intrinsic pulsar noise, but one which can include inter-frequency correlations efficiently. Eq. ([28](#S4.E28)) cannot efficiently include inter-frequency correlations because computing $\boldsymbol{\phi}^{-1}$ is costly as it is a large dense matrix, with inter-pulsar and -frequency correlations. By separating the Gaussian processes which represent the intrinsic pulsar RN and GWB, we may compute $\boldsymbol{\phi}_{\text{RN/GWB}}^{-1}$ efficiently in the evaluation of Eq. ([40](#S4.E40)). The frequency-domain prior covariance matrix is computed with the two-dimensional FFT of the corresponding time-domain matrix from the FFTInt method. Our derivation above did not include deterministic signals as in Sec. [IV.3](#S4.SS3). Nonetheless, our results Eq. ([40](#S4.E40)) and Eq. ([41](#S4.E41)) can be generalized the include deterministic signals under the simple replacement $\boldsymbol{\delta t}\rightarrow\boldsymbol{\delta t}-\mathbf{h}=\boldsymbol{\delta t}-\mathbf{F}_{\text{D}}\mathbf{a}_{D}(\theta)$.

## V Implementation techniques

### V.1 Single precision

The efficiency of our methods is significantly improved by operating entirely in single precision. That is, floating-point numbers are represented with 32 bits, as opposed to 64 bits (double precision) which is the default for many Python environments. Roughly speaking, this decision reduces the number of reliable digits from 15 to 7. We use units of nano-seconds (ns) rather than seconds (s) to faithfully represent the data. In nano-seconds, timing residuals are $\mathcal{O}(10^{2})$ and more numerically stable than the alternative in seconds which is $\mathcal{O}(10^{-7})$. This convention is used for all objects in the posterior. For example, covariance matrices use units of ($\text{ns}^{2}$). Moreover, we drop the weighting of the determinant of the projected white noise covariance matrix, $\text{det}(2\pi\tilde{\mathbf{N}})$, in the posterior evaluation from Eq. ([27](#S4.E27)) to Eq. ([28](#S4.E28)) so evaluations of the posterior may be resolved in single precision. As we fix the white noise model during the analysis, this amounts to neglecting a constant scaling in our posterior and does not affect our inference.

In gamma-ray PTAs [89], the matrix $\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}$ is singular, and the likelihood not normalizable. To remedy this, we may regularize the posterior, Eq. ([28](#S4.E28)), as in Valtolina and van Haasteren [93], so that up to an overall scaling

$$ $p(\mathbf{a},\boldsymbol{\eta}|\boldsymbol{\delta t})\propto p(\boldsymbol{\eta})\,\sqrt{\frac{\text{det}(2\pi\boldsymbol{\phi}_{0})}{\text{det}(2\pi\boldsymbol{\phi})}}\,\text{exp}\bigg[-\frac{1}{2}\big(\mathbf{a}-\mathbf{\hat{a}}_{0}\big)^{\text{T}}\,\boldsymbol{\Sigma}^{-1}_{0}\,\big(\mathbf{a}-\mathbf{\hat{a}}_{0}\big)-\frac{1}{2}\mathbf{a}^{\text{T}}\,\big(\boldsymbol{\phi}^{-1}-\boldsymbol{\phi}^{-1}_{0}\big)\,\mathbf{a}+\frac{1}{2}\hat{\mathbf{a}}_{0}\,\boldsymbol{\Sigma}^{-1}_{0}\,\hat{\mathbf{a}}_{0}\bigg]$ (42) $$

where $\hat{\mathbf{a}}_{0}=\boldsymbol{\Sigma}_{0}\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\boldsymbol{\delta t}$. $\boldsymbol{\Sigma}_{0}$ and $\boldsymbol{\phi}_{0}$ are reference covariance matrices, identical to $\boldsymbol{\Sigma}$ and $\boldsymbol{\phi}$, respectively, but evaluated at a set of reference hyper-parameters, $\boldsymbol{\eta}_{0}$, and held fixed throughout the analysis. Note that Eq. ([42](#S5.E42)) is identical to Eq. ([28](#S4.E28)), up to an overall scaling in the case of radio PTAs. However in gamma-ray PTAs, Eq. ([28](#S4.E28)) is ill-conditioned whereas Eq. ([42](#S5.E42)) is a normalizable probability density.

We must also take care to implement deterministic models in single precision. If the CW model from [23, 57, 28, 27] is implemented naively in single precision, the model is prone to catastrophic cancellation. This is seen in the phase-evolution of the CW signal,

$$ $\Phi(t)=\Phi_{0}+\frac{1}{32{\mathcal{M}}^{5/3}}\left(\omega_{0}^{-\frac{5}{3}}-\omega^{-\frac{5}{3}}\right)\,$ (43) $$

where $\Phi_{0}$ and $\omega_{0}$ are the reference phase and frequency of the CW, respectively. $\mathcal{M}$ is the chirp mass and $\omega$ the frequency of the CW. The frequency itself evolves as

$$ $\omega(t)=\omega_{0}\bigg(1-\frac{256}{5}\mathcal{M}^{5/3}\omega_{0}^{8/3}t\bigg)^{-3/8}\,,$ (44) $$

where $t$ the time of evolution. In some regions of parameter space, the difference between the frequency and its reference value in Eq. ([43](#S5.E43)) is not resolvable in single precision. In such cases, we substitute Eq. ([44](#S5.E44)) into Eq. ([43](#S5.E43)) and express the phase of evolution of the CW as

$$ $\displaystyle\Phi(t)$ $\displaystyle=\Phi_{0}+\frac{1}{32(\mathcal{M}\omega_{0})^{5/3}}\bigg[1-(1-x)^{5/8}\bigg]$ (45) $\displaystyle\approx\Phi_{0}+\frac{x}{(\mathcal{M}\omega_{0})^{5/3}}\,,$ (46) $$

where $x\equiv\mathcal{M}^{5/3}\omega_{0}^{8/3}t\ll 1$. Similar situations arise in the amplitude and pulsar term calculations of the CW model where we apply an identical treatment, only keeping resolvable terms in the Taylor series.

### V.2 Batching

While Eq. ([31](#S4.E31)) and the analogous regularized Eq. ([42](#S5.E42)) are convenient forms for the posterior in which the mean, covariance, and standardizing transform can be “read-off”, it is more computationally efficient to implement them in the numerically equivalent form

$$ $\tilde{p}(\mathbf{z},\boldsymbol{\eta}|\boldsymbol{\delta t})\propto\frac{p(\boldsymbol{\eta})\cdot\text{det}(\mathbf{L}_{\text{CURN}})}{\sqrt{\text{det}(2\pi\boldsymbol{\phi})}}\,\text{exp}\bigg[\boldsymbol{\delta t}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}\mathbf{a}-\frac{1}{2}\mathbf{a}^{\text{T}}\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}\mathbf{a}-\frac{1}{2}\mathbf{a}^{\text{T}}\boldsymbol{\phi}^{-1}\mathbf{a}\bigg]$ (47) $$

where via the standardizing transform $\mathbf{a}=\hat{\mathbf{a}}+\mathbf{L}_{\text{CURN}}\mathbf{z}$. In this form, all terms without $\boldsymbol{\phi}$ are per-pulsar (including the standardizing transform which uses $\boldsymbol{\phi}_{\text{CURN}}$), and we may batch (or parallelize) our computations over pulsars using a GPU for extremely efficient evaluation. $\boldsymbol{\phi}$ includes inter-pulsar correlations, but does not include frequency correlations, and we may calculate its inverse, determinant, and derived matrix products by batching over frequencies (i.e. by inverting $2N_{f}$ $(N_{p}\times N_{p})$ matrices in parallel to obtain its inverse) for efficient evaluation on a GPU. A posterior which includes inter-pulsar and -frequency correlations and the associated standardizing coordinate transformation was presented in Sec. [IV.4](#S4.SS4).

Thanks to GPU parallelization, evaluating Eq. ([47](#S5.E47)) and its analogous forms which include regularization and/or deterministic signal contributions scales sub-linearly with the number of pulsars. Alternatively, if the Fourier coefficients had been analytically marginalized from the model the posterior evaluation scales worse than quadratically with the number of pulsars. We implement and time both the standardized posterior Eq. ([47](#S5.E47)) and the posterior marginalized over the Fourier coefficients (as derived in [59]) on an NVIDIA GeForce RTX 3090 GPU as a function of the number of pulsars. Each pulsar used in the timing was simulated and observed approximately monthly for 15 years. The results are shown in Fig. [3](#S5.F3). We estimate the polynomial scaling law of the posterior evaluation time with the number of pulsars, for $N_{p}>60$. The standardized posterior scales as $\sim\mathcal{O}(N_{p}^{0.6})$ and the marginalized posterior goes as $\sim\mathcal{O}(N_{p}^{2.4})$. While the standardized posterior requires us to sample thousands of additional parameters, they approximately obey uncorrelated standard normal distributions and are sampled efficiently with HMC.

Figure: Figure 3: Evaluation times of the posterior on an NVIDIA GeForce RTX 3090 GPU, as function of the number of pulsars in the array. The results in blue denote the evaluation times for the standardized posterior, Eq. ([47](#S5.E47)), and the results in orange are the evaluation times for the posterior which has the Fourier coefficients analytically marginalized, as in [59]. Each pulsar used to construct the timing dataset was simulated and observed approximately monthly for 15 years.
Refer to caption: https://arxiv.org/html/2607.06834/2607.06834v1/figs/timing.png

## VI Analyses of real and simulated datasets

To assess the methods presented above, we reproduce the parameter estimation results for the NANOGrav 15-year stochastic analysis [8]. That is, we model intrinsic pulsar noise and a stochastic gravitational wave background across 67 pulsars observed over 15 years [10]. The white noise parameters are estimated and fixed before the parameter estimation and linear deviations to the timing model are analytically marginalized. The intrinsic pulsar red noise is modeled with a power law Eq. ([23](#S4.E23)) and uses $N_{f}=30$ frequency bins. The GWB is modeled using a power law with $N_{f}=14$ frequency bins, and the Hellings-Downs inter-pulsar correlation Eq. ([22](#S4.E22)) is imposed.

To test the capabilities of the standardized transformation generalized to include deterministic contributions, we simulate a dataset consistent with the models presented in Sec. [III](#S3), including a continuous wave source. 100 pulsars are randomly distributed isotropically across the sky, at fixed distance $L_{I}=1\;\text{kpc}$ with an uncertainty of $0.2\;\text{kpc}$. Each pulsar is observed roughly every month for 15 years. The TOA uncertainty is set to $0.5\;\mu\text{s}$ for every observation, and the EFAC is fixed at 1 across pulsars and observations. EQUAD and ECORR are neglected.

Intrinsic pulsar RN and a HD correlated stochastic GWB obeying power laws are injected in each of the 100 pulsars consistent with the models described above. We analyze the GWB and RN using $N_{f}=14$ and $N_{f}=30$ frequency bins, respectively. The injected RN hyper-parameters of the power law are drawn from the distributions, $\log_{10}A\sim\text{Uniform}(-18,-13)$ and $\gamma\sim\text{Uniform}(2,\,5)$. The injected hyper-parameters of the GWB are $\log_{10}A_{\text{GWB}}=-14.5$ and $\gamma_{\text{GWB}}=13/3$. A CW signal consistent with the model presented in Corbin and Cornish [23], Ellis et al. [28], Ellis [27] is injected with parameters $\log_{10}\mathcal{M}/[\text{M}_{\odot}]=8.4$, $\log_{10}f_{\text{CW}}/[\text{Hz}]=-8.4$, $\cos\iota=\sqrt{2}/2$, $\psi=\pi/3$, $\log_{10}h=-14.4$, $\cos\theta=\sqrt{2}/2$, $\phi=\pi/4$, and $\Phi_{0}=\pi/4$. These parameters correspond to chirp mass, frequency, inclination, polarization, amplitude, polar sky location, azimuthal sky location, and phase, respectively. In the posterior evaluation, the CW is represented in a Fourier domain with $N_{f}=60$ frequency bins.

### VI.1 Results

The recovery of the stochastic GWB hyper-parameters and a couple pulsars’ RN hyper-parameters for the NANOGrav 15-year dataset are shown in Fig. [4](#S6.F4). The posterior recovered using the methods presented in this paper is consistent with that of standard methods. A dataset this large typically requires at least several hours to obtain a sufficient number of independent samples. On an NVIDIA GeForce RTX 3090 GPU, our analysis takes approximately 15 minutes to resolve the posterior shown in Fig [4](#S6.F4). More precisely, the standard analytically marginalized posterior achieves $\sim 0.12$ effective samples per second; our numerically marginalized posterior under the standardizing transformation achieves $\sim 1.53$ effective samples per second resulting in over an order of magnitude speed-up. So it’s a fair comparison, we implemented the analytically marginalized posterior in JAX with single precision on the same GPU-configuration, and sampled with NUTS exactly as we do for the numerically marginalized posterior. We conclude the numerical marginalization and standardizing transformation approach is over an order of magnitude faster than the analytically marginalized analysis. The same dataset is also analyzed using a free spectral model to describe the stochastic GWB. The magnitude of the timing delays induced by the GWB per frequency bin is shown in Fig. [5](#S6.F5), and is consistent with standard methods.

Figure: Figure 4: Corner plot illustrating the recovery of the GWB, J1745+1017’s noise spectrum, and J1853+1303’s noise spectrum under a power law model in the NANOGrav 15-year dataset. The blue posterior is obtained using the standardizing transform presented in this paper. The orange posterior is obtained using the standard PTA analysis software ENTERPRISE [29], in which the Fourier coefficients are analytically marginalized from the model. The contours of the two-dimensional histograms correspond to the $(0.5,1,1.5,2)-\sigma$ credible intervals.
Refer to caption: https://arxiv.org/html/2607.06834/2607.06834v1/figs/gwb_rn_corner.png

Figure: Figure 5: Free spectral analysis of the stochastic GWB in the NANOGrav 15-year dataset. The violins show the timing delays induced by the background per frequency bin. The blue violins are obtained using the standardizing transformation presented in this paper. The orange violins are obtained using the standard PTA analysis software ENTERPRISE [29], in which the Fourier coefficients are analytically marginalized from the model.
Refer to caption: https://arxiv.org/html/2607.06834/2607.06834v1/figs/violin.png

The results of the analysis on simulated data are shown in Fig. [6](#S6.F6) where the posterior over a subset of CW parameters, one pulsar’s RN hyper-parameters, and Fourier coefficients are plotted. The injected parameter values lie within the posterior distribution. Neal’s funnel is observed in the distribution over the power law amplitude parameter and a high-frequency Fourier coefficient as expected. The funnel appears to be well-sampled thanks to the standardizing transform generalized to include deterministic contributions, Eq. ([37](#S4.E37)).

It’s worth emphasizing this analysis of a simulated dataset jointly models a HD-correlated GWB, intrinsic pulsar RN, and a CW simultaneously in a relatively large PTA. Such analyses were extremely computationally intensive, typically requiring many hours of computation time or alternative approximations before this work. For example, the QuickCW software [13] neglects inter-pulsar correlations and models a CURN GWB and a CW simultaneously. If a joint HD-correlated GWB and CW analysis was desired, the samples from QuickCW could be reweighted to include inter-pulsar correlations. However, very few independent samples survive the reweighting process requiring lengthy runs to build up the initial sample set [6]. Using the methods presented above, the sampling chain for the joint GWB + CW + RN model converged in less than 20 minutes on an NVIDIA GeForce RTX 3090.

Figure: Figure 6: Corner plot illustrating the recovery of a subset of CW parameters, the $3^{\text{rd}}$ pulsar’s RN hyper-parameter, and corresponding Fourier coefficients, using the notation of Eq. ([III.3](#S3.Ex3)). Blue lines indicate injected parameter values.
Refer to caption: https://arxiv.org/html/2607.06834/2607.06834v1/figs/cw_corner.png

### VI.2 Discussion and future work

Historically, PTA data analysis has been computationally expensive. The methods presented in this paper dramatically improve efficiency, yielding more than an order of magnitude speed-up for analysis of realistic datasets. Rather than analytically marginalizing over the set of Fourier coefficients which represent stochastic red signals and noise, they are sampled (i.e. numerically marginalized) with a hyper-efficient posterior formulation as presented in [59]. While hyper-efficient, the posterior is high-dimensional and exhibits a complicated funnel-like geometry from which it is generally difficult to sample. We reparameterize the posterior with a standardizing coordinate transform so the Fourier coefficients are approximately described by a standard normal distribution. The high-dimensional posterior can then be efficiently sampled using HMC with the NUTS when run on a GPU.

The standardizing transform for PTAs is generalized to include contributions from deterministic signals. The recovery of a particular simulated CW signal is shown above, but future work needs to improve the sampling of deterministic signal parameters which may induce complicated posterior geometries. The posteriors under deterministic models are often multi-modal, significantly non-Gaussian, and troublesome for standard HMC samplers. Deterministic signal analysis may be improved by mixing other jump proposals, such as parallel tempering, with classic HMC techniques, see Appendix [B](#A2).

###### Acknowledgements.

###### Acknowledgements.

## Appendix A Hamiltonian Monte Carlo and No U-Turn Sampling

Hamiltonian Monte Carlo (HMC), originally known as Hybrid Monte Carlo, was first developed for calculations in lattice quantum chromodynamics [26]. It was later popularized for applied statistics in [65, 67]. [15] illustrated its robustness by setting it in the language of differential geometry. The sampling above is performed with HMC and its No U-Turn Sampler (NUTS) extension and implemented in NUMPYRO [69, 18]. We summarize HMC and NUTS in this appendix, following the treatment of [17] to which the reader is directed for a more thorough discussion.

The aim of Markov Chain Monte Carlo (MCMC) is to sample from a target distribution, $p(\mathbf{x})$, under some parameterization $\mathbf{x}\in\mathcal{X}$, where $\mathcal{X}$ is the target sample space and $\text{dim}(\mathbf{x})=d$. MCMC is performed by constructing a Markov Chain whose equilibrium distribution is the target distribution itself. Once sufficiently many samples have been drawn, they may be binned into a histogram to reconstruct the target distribution, or approximate expectations using Monte Carlo integration,

$$ $\mathbb{E}[f]=\int_{\mathcal{X}}f(\mathbf{x})\,p(\mathbf{x})\,d\mathbf{x}\approx\frac{1}{N}\sum_{i}^{N}f(\mathbf{x}^{(i)})$ (48) $$

where $\mathbf{x}^{(i)}$ denotes the $i^{\text{th}}$ of $N$ total random samples from the target distribution.

The standard algorithm to produce random samples from a target distribution is the Metropolis-Hastings (MH) MCMC algorithm, [63, 44]. The MH algorithm initializes a random sample $\mathbf{x}^{(0)}\in\mathcal{X}$, and after $i$ iterations, the chain may transition from state $\mathbf{x}^{(i)}$ to $\mathbf{y}$. The transition is proposed by drawing from the proposal distribution $g(\mathbf{y}|\mathbf{x}^{(i)})$, and accepted with transition probability

$$ $\alpha=\text{min}\bigg\{1,\;\;\frac{p(\mathbf{y})\cdot g(\mathbf{x}^{(i)}|\mathbf{y})}{p(\mathbf{x}^{(i)})\cdot g(\mathbf{y}|\mathbf{x}^{(i)})}\bigg\}\,.$ (49) $$

If the proposal is accepted, the next sample is set to the proposal, $\mathbf{x}^{(i+1)}=\mathbf{y}$. If the proposal is rejected, the chain remains at the same sample, $\mathbf{x}^{(i+1)}=\mathbf{x}^{(i)}$. The MH algorithm is repeated for the desired number of samples. If the proposal density is sufficiently well-tailored, then the algorithm is guaranteed to produce samples which converge to the target distribution. Samples obtained via MCMC are correlated with one another, by construction, and the chain of samples usually needs to be thinned until only statistically independent samples remain.

HMC operates in a similar fashion to MH, but the proposals are not drawn from some proposal distribution and implemented in random walk fashion. Instead, the proposals are generated by integrating Hamilton’s equations of motion, which dictate the chain’s evolution deterministically through a phase space (inspired by Hamilton’s formulation of classical mechanics). The parameters of the target density are extended to include a set of canonical momenta, $\mathbf{x}\rightarrow(\mathbf{x},\mathbf{k})\in\mathcal{P}$, where $\mathbf{x}$ assume the role of generalized coordinates, $\mathbf{k}$ canonical momenta, $\text{dim}(\mathbf{k})=d$, and $\mathcal{P}$ denotes the phase space. The target density is similarly lifted to the canonical distribution with support on the entire $2d$-dimensional phase space,

$$ $p(\mathbf{x},\mathbf{k})=p(\mathbf{k}|\mathbf{x})\cdot p(\mathbf{x})\,.$ (50) $$

We recover the target distribution, $p(\mathbf{x})$, if the canonical momenta are marginalized over.

The Hamiltonian governing dynamics on the phase space is defined as

$$ $\displaystyle H(\mathbf{x},\mathbf{k})$ $\displaystyle\equiv-\ln p(\mathbf{x},\mathbf{k})$ $\displaystyle=-\ln p(\mathbf{k}|\mathbf{x})-\ln p(\mathbf{x})\,.$ (51) $$

The two terms in Eq. ([A](#A1.Ex11)) are the so called kinetic and potential energy, respectively. Trajectories through this phase space are described by Hamilton’s equations of motion, which are $2d$ coupled first-order ordinary differential equations,

$$ $\displaystyle\dot{\mathbf{x}}^{i}$ $\displaystyle=\frac{\partial H}{\partial\mathbf{k}_{i}}$ $\displaystyle\dot{\mathbf{k}}_{i}$ $\displaystyle=-\frac{\partial H}{\partial\mathbf{x}^{i}}\,,$ (52) $$

where $i\in\{1,2,\dots,d\}$ indexes the components of the generalized coordinates and canonical momenta. To complete the probabilistic structure on phase space, we must choose a form for the kinetic energy, $p(\mathbf{k}|\mathbf{x})$. Rather than searching over infinite possibilities for the optimal kinetic energy, we’ll restrict ourselves to Euclidean-Gaussian kinetic energies in which the distance between two configurations with coordinates $\mathbf{x}$ and $\mathbf{x}^{\prime}$ is

$$ $\Delta(\mathbf{x},\mathbf{x}^{\prime})=\mathbf{g}_{ij}\,(\mathbf{x}-\mathbf{x}^{\prime})^{i}\,(\mathbf{x}-\mathbf{x}^{\prime})^{j}\,,$ (53) $$

where $i,j\in\{1,2,\dots,d\}$, $\mathbf{g}_{ij}$ are the components of the Euclidean metric tensor, and the Einstein summation convention is used. In order to preserve volumes in phase space under coordinate transformations, the canonical momenta must use the inverse transformation to that of the generalized coordinates when performing reparameterizations. The distance between momenta $\mathbf{k}$ and $\mathbf{k}^{\prime}$ is

$$ $\Delta(\mathbf{k},\mathbf{k}^{\prime})=\mathbf{g}^{ij}\,(\mathbf{k}-\mathbf{k}^{\prime})_{i}\,(\mathbf{k}-\mathbf{k}^{\prime})_{j}\,,$ (54) $$

where $\mathbf{g}^{ij}$ are the components of the inverse Euclidean metric. The metric allows the construction of probability densities over the momenta, and the Euclidean-Gaussian (zero mean) kinetic energy is

$$ $p(\mathbf{k}|\mathbf{x})=\frac{1}{\sqrt{\text{det}(2\pi\mathbf{M})}}\,\text{exp}\bigg[-\frac{1}{2}\mathbf{k}^{\text{T}}\,\mathbf{M}^{-1}\,\mathbf{k}\bigg]\,,$ (55) $$

where we’ve adopted the physics notation of a mass matrix, $\mathbf{M}$, to replace the inverse metric tensor. We have not yet chosen a form for the Euclidean metric (or mass matrix), and an optimal choice for arbitrary systems has not yet been derived. However, acting as a covariance matrix, the mass matrix has the ability to de-correlate the parameters of phase space. We may de-correlate the parameters of the target density, $\mathbf{x}$, by estimating their covariance, and because they transform in opposite fashion, the inverse covariance should be used to define the mass matrix,

$$ $\mathbf{M}^{-1}=\text{cov}(\mathbf{x},\mathbf{x})\,.$ (56) $$

Geometrically, this choice distributes the level sets of the Hamiltonian in phase space so that the sampling is easier. In practice, the covariance is often estimated empirically with a set of warm-up samples. No global transformation using the covariance is going to perfectly de-correlate the target space for arbitrary problems, unless they are perfectly Gaussian. To improve sampling further for systems with significant non-Gaussian features, it is possible to extend beyond Euclidean geometries and use a Riemannian-Gaussian kinetic energy. For such systems the covariance is computed locally, $\boldsymbol{\Sigma}=\boldsymbol{\Sigma}(\mathbf{x})$, using say a Fisher information matrix approach, $\boldsymbol{\Sigma}^{-1}(\mathbf{x})\approx-\partial_{\mathbf{x}}\partial_{\mathbf{x}}\ln p(\mathbf{x})$. Then the kinetic energy assumes a multivariate normal distribution, whose covariance is a position-dependent field over the target space,

$$ $\mathbf{k}|\mathbf{x}\sim\mathcal{N}(\mathbf{0},\boldsymbol{\Sigma}(\mathbf{x}))\,.$ (57) $$

Sampling with HMC is then performed as follows. At the $i^{\text{th}}$ iteration we have sample $\mathbf{x}^{(i)}$. To generate the $(i+1)^{\text{th}}$ sample, we first draw a set of momenta, $\mathbf{k}^{(i)}$, from the kinetic energy distribution, Eq. ([55](#A1.E55)). $(\mathbf{x}^{(i)},\mathbf{k}^{(i)})$ serve as the initial conditions to Hamilton’s equations of motion, Eq. ([A](#A1.Ex12)), which are integrated so the chain evolves through phase space. After some time, $\Delta t$, the integration is terminated and the chain’s state is $(\mathbf{x}_{\Delta t},\mathbf{k}_{\Delta t})$. This state is the proposal for the next sample and is accepted with probability

$$ $\alpha=\text{min}\bigg\{1,\frac{\text{exp}[-H(\mathbf{x}_{\Delta t},\mathbf{k}_{\Delta t})]}{\text{exp}[-H(\mathbf{x}^{(i)},\mathbf{k}^{(i)})]}\bigg\}\,.$ (58) $$

If the integration is exact, then the chain evolves along a level set of the Hamiltonian and the proposal is always accepted. In practice, the integration is performed numerically with a leapfrog integrator, see Sec. 5.1 of [17]. Leapfrog integration is symplectic, conserving volume in phase space, and the trajectories never stray far from level set at which they were initialized. Symplectic integrators introduce errors that are tangent to a nearby energy surface, while non-symplectic (such as primitive Euler or Runge–Kutta) integrators introduce errors with a systematic component normal to energy surfaces. HMC can achieve relatively high (often $\sim 90\%$) acceptance rates thanks to symplectic integrators moving along level sets of the Hamiltonian. At subsequent iterations, the momenta are redrawn from Eq. ([55](#A1.E55)), and the chain is integrated along a new level set. If the mass matrix accurately describes the distribution of level sets, the typical set of the target distribution is quickly explored after relatively few draws from the kinetic energy distribution.

The final piece is to decide how long to integrate Hamilton’s equations at each iteration. If $\Delta t$ is too short, the chain is highly correlated and resembles random walk. If $\Delta t$ is too long, significant computation time is spent integrating Hamilton’s equations for every sample. The No U-turn termination condition, [46], provides an empirical method to decide integration time. It integrates trajectories through the phase space until a “U-turn” is detected. This encourages trajectories long enough to reduce auto-correlations, but not so long they double back on themselves, wasting integration time in a previously covered region of phase space.

## Appendix B The standardizing transform for tempered likelihoods

While HMC is a highly efficient sampling scheme in many systems, the chain will get stuck on local maxima if the posterior exhibits multi-modality. For multi-modal target densities a parallel tempered sampling scheme [83] may be appropriate in which the log-likelihood is scaled by a temperature $\mathcal{T}$ to aid chain exploration. Below we generalize the standardizing transform to allow temperature scaling of the chain.

In a tempering scheme, the likelihood is raised to the power of $\beta=1/\mathcal{T}$ so the tempered posterior may be written

$$ $p_{\beta}(\mathbf{a},\boldsymbol{\eta}|\boldsymbol{\delta t})\propto p(\boldsymbol{\delta t}|\mathbf{a})^{\beta}\cdot p(\mathbf{a}|\boldsymbol{\eta})\cdot p(\boldsymbol{\eta})$ (59) $$

such that for large temperatures the likelihood is suppressed and we recover a distribution which more closely resembles the prior. In the infinite temperature limit the tempered posterior converges to the prior. Tempering the posterior, Eq. ([27](#S4.E27)), we have

$$ $p_{\beta}(\mathbf{a},\boldsymbol{\eta}|\boldsymbol{\delta t})\propto\frac{p(\boldsymbol{\eta})}{\text{det}(2\pi\tilde{\mathbf{N})}^{\beta/2}}\,\text{exp}\bigg[-\frac{\beta}{2}\big(\boldsymbol{\delta t}-\mathbf{F}\mathbf{a}\big)^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\big(\boldsymbol{\delta t}-\mathbf{F}\mathbf{a}\big)\bigg]\times\frac{1}{\sqrt{\text{det}(2\pi\boldsymbol{\phi})}}\text{exp}\bigg[-\frac{1}{2}\mathbf{a}^{\text{T}}\,\boldsymbol{\phi}^{-1}\,\mathbf{a}\bigg]\,.$ (60) $$

Completing the square, the posterior can be expressed as

$$ $p_{\beta}(\mathbf{a},\boldsymbol{\eta}|\boldsymbol{\delta t})\propto\frac{p(\boldsymbol{\eta})}{\sqrt{\text{det}(2\pi\boldsymbol{\phi})}}\,\text{exp}\bigg[-\frac{1}{2}\big(\mathbf{a}-\mathbf{\hat{a}}_{\beta}\big)^{\text{T}}\,\boldsymbol{\Sigma}_{\beta}^{-1}\,\big(\mathbf{a}-\mathbf{\hat{a}}_{\beta}\big)+\frac{1}{2}\mathbf{\hat{a}}_{\beta}^{\text{T}}\,\boldsymbol{\Sigma}_{\beta}^{-1}\,\mathbf{\hat{a}}_{\beta}\bigg]$ (61) $$

where the estimated mean and covariance of the Fourier coefficients is $\hat{\mathbf{a}}_{\beta}=\beta\cdot\boldsymbol{\Sigma}_{\beta}\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\boldsymbol{\delta t}$ and $\boldsymbol{\Sigma}_{\beta}=\big(\beta\cdot\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}+\boldsymbol{\phi}^{-1}\big)^{-1}$, respectively. The mean and covariance are not scaled trivially by the temperature as they involve likelihood and prior contributions, and only terms in the likelihood are scaled by temperature. The tempered standardizing transform is

$$ $(\mathbf{a},\boldsymbol{\eta})=T_{\beta}^{-1}(\mathbf{z},\boldsymbol{\eta})=(\hat{\mathbf{a}}_{\beta}+\mathbf{L}_{\beta}\,\mathbf{z},\boldsymbol{\eta})$ (62) $$

where $\boldsymbol{\Sigma}_{\beta}=\mathbf{L}_{\beta}\mathbf{L}_{\beta}^{\text{T}}$ is the Cholesky decomposition of tempered covariance matrix. While the sampling of a tempered posterior under an un-tempered standardizing transform is valid, the transform will consistently map the Fourier coefficients into a narrower region of parameter space relative to the tempered posterior volume, and the chain will take longer to converge. Beyond multi-modality tempered posteriors are useful in the calculation of evidences and Bayes factors via thermodynamic integration [81].

## Appendix C The standardizing transform for split coefficients.

Rather than combining the intrinsic pulsar red noise and gravitational wave background into one Gaussian process as in Eq. ([21](#S4.E21)), we may wish to treat them as independent stochastic processes (^2^22This is helpful when we wish to model inter-frequency correlations (see Sec. [IV.4](#S4.SS4)) or non-Gaussian features (see Appendix [D](#A4)).). This choice necessitates using two distinct sets of Fourier coefficients in the posterior: one to represent pulsar noise and one to represent the background.

If we use separate sets of Fourier coefficients to describe the intrinsic pulsar red noise and the stochastic gravitational wave background, the posterior Eq. ([27](#S4.E27)) is modified

$$ $\displaystyle p(\mathbf{a}_{\text{GWB}},\mathbf{a}_{\text{RN}},\boldsymbol{\eta}|\boldsymbol{\delta t})\propto$ $\displaystyle\frac{p(\boldsymbol{\eta})}{\sqrt{\text{det}(2\pi\tilde{\mathbf{N}})}}\,\text{exp}\bigg[-\frac{1}{2}\big(\boldsymbol{\delta t}-\mathbf{F}\mathbf{a}_{\text{GWB}}-\mathbf{F}\mathbf{a}_{\text{RN}}\big)^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\big(\boldsymbol{\delta t}-\mathbf{F}\mathbf{a}_{\text{GWB}}-\mathbf{F}\mathbf{a}_{\text{RN}}\big)\bigg]$ (63) $\displaystyle\times\frac{1}{\sqrt{\text{det}(2\pi\boldsymbol{\phi}_{\text{GWB}})}}\text{exp}\bigg[-\frac{1}{2}\mathbf{a}_{\text{GWB}}^{\text{T}}\,\boldsymbol{\phi}_{\text{GWB}}^{-1}\,\mathbf{a}_{\text{GWB}}\bigg]$ $\displaystyle\times\frac{1}{\sqrt{\text{det}(2\pi\boldsymbol{\phi}_{\text{RN}})}}\text{exp}\bigg[-\frac{1}{2}\mathbf{a}_{\text{RN}}^{\text{T}}\,\boldsymbol{\phi}_{\text{RN}}^{-1}\,\mathbf{a}_{\text{RN}}\bigg]\,,$ $$

where the subscripts “GWB” and “RN” represent the background and intrinsic noise respectively. Both sets of coefficients are defined with respect to the same basis $\mathbf{F}$, and a simple replacement $\mathbf{F}\rightarrow\mathbf{F}_{\text{RN}},\mathbf{F}_{\text{GWB}}$ is used to represent the processes with respect to distinct Fourier bases. The MAP Fourier coefficients, are defined by the extremum condition,

$$ $\begin{pmatrix}\partial_{\mathbf{a}_{\text{GWB}}}\ln p(\mathbf{a}_{\text{GWB}},\mathbf{a}_{\text{RN}},\boldsymbol{\eta}|\boldsymbol{\delta t})\big|_{\hat{\mathbf{a}}_{\text{GWB}},\hat{\mathbf{a}}_{\text{RN}}}\\ \partial_{\mathbf{a}_{\text{GWB}}}\ln p(\mathbf{a}_{\text{GWB}},\mathbf{a}_{\text{RN}},\boldsymbol{\eta}|\boldsymbol{\delta t})\big|_{\hat{\mathbf{a}}_{\text{GWB}},\hat{\mathbf{a}}_{\text{RN}}}\par\end{pmatrix}=\begin{pmatrix}0\\ 0\end{pmatrix}$ (64) $$

which yields the linear system

$$ $\begin{pmatrix}\boldsymbol{\Sigma}_{\text{GWB}}^{-1}&\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}\\ \mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}&\boldsymbol{\Sigma}_{\text{RN}}^{-1}\end{pmatrix}\begin{pmatrix}\hat{\mathbf{a}}_{\text{GWB}}\\ \hat{\mathbf{a}}_{\text{RN}}\end{pmatrix}=\begin{pmatrix}\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\boldsymbol{\delta t}\\ \mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\boldsymbol{\delta t}\end{pmatrix}\,,$ (65) $$

where the covariances, estimated from the Hessian of the log-posterior, are $\boldsymbol{\Sigma}_{\text{GWB}}=\big(\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}+\boldsymbol{\phi}_{\text{GWB}}^{-1}\big)^{-1}$ and $\boldsymbol{\Sigma}_{\text{RN}}=\big(\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}+\boldsymbol{\phi}_{\text{RN}}^{-1}\big)^{-1}$. While Eq. ([65](#A3.E65)) may be solved analytically, it is more computationally efficient to solve it numerically using a Cholesky decomposition. The standardizing transformation is then modified

$$ $(\mathbf{a}_{\text{GWB}},\mathbf{a}_{\text{RN}},\boldsymbol{\eta})=T^{-1}(\mathbf{z}_{\text{GWB}},\mathbf{z}_{\text{RN}},\boldsymbol{\eta})=(\hat{\mathbf{a}}_{\text{GWB}}+\mathbf{L}_{\text{GWB}}\mathbf{z}_{\text{GWB}},\hat{\mathbf{a}}_{\text{RN}}+\mathbf{L}_{\text{RN}}\mathbf{z}_{\text{RN}},\boldsymbol{\eta})$ (66) $$

where $\mathbf{L}_{\text{GWB/RN}}$ is the Cholesky decomposition of the covariance matrix $\boldsymbol{\Sigma}_{\text{GWB/RN}}$. As above, we approximate the covariance of background by neglecting inter-pulsar correlations in the transformation. By definition, the intrinsic pulsar red noise is independent across pulsars and its covariance matrix is already diagonal in the pulsar basis. Notice the covariance of the sets of Fourier coefficients is determined independently from the hyper-parameters of the respective stochastic process while the MAP GWB and RN coefficients are coupled.

## Appendix D Non-Gaussian features

It is well known in the PTA community that stochastic astrophysical signals are non-Gaussian. Namely the stochastic gravitational wave background, if realized via a finite population of supermassive black hole binaries, will exhibit statistical moments beyond second-order, [54, 55, 50, 75, 71, 103]. However for modeling efficiency, such signals are often approximated with a Gaussian distribution (as we have done above Eq. ([24](#S4.E24))). As the sensitivity of PTAs improve non-Gaussian features may become resolvable, and we’ll need analysis pipelines capable of modeling higher-order statistical moments.

Standard analyses, in which the Fourier coefficients are analytically marginalized from the model, will struggle to model such non-Gaussian features. The analytic marginalization is only possible because a conjugate prior is chosen for the Fourier coefficients, Eq. ([24](#S4.E24)), so that both the posterior and prior are a multivariate normal distribution in the Fourier coefficients, Eq. ([28](#S4.E28)), for which a closed form analytic expression for the marginalizing integral is known. If a prior for the Fourier coefficients with non-Gaussian features is chosen, then analytic marginalization may not be possible. Methods have been developed to model non-Gaussian features in PTA datasets nonetheless, e.g. via particular parameterizations of non-Gaussian features [58] or Gaussian mixture models [31]. In this appendix, we show how non-Gaussian priors on the Fourier coefficients may be implemented in tandem with the methods presented above and demonstrate how such non-Gaussian features can be resolved efficiently by analyzing a simulated dataset.

As we sample the Fourier coefficients jointly, our methods don’t require an analytic marginalization and we are free to choose arbitrary priors for the coefficients. Let $q(\mathbf{a}|\boldsymbol{\eta})$ be an arbitrary non-Gaussian prior on the coefficients, conditioned on the spectral hyper-parameters $\boldsymbol{\eta}$. Then the PTA posterior is proportional to

$$ $\displaystyle p(\mathbf{a},\boldsymbol{\eta}|\boldsymbol{\delta t})$ $\displaystyle\propto p(\boldsymbol{\delta t}|\mathbf{a})\cdot q(\mathbf{a}|\boldsymbol{\eta})\cdot p(\boldsymbol{\eta})$ $\displaystyle\propto\frac{q(\mathbf{a}|\boldsymbol{\eta})\cdot p(\boldsymbol{\eta})}{\sqrt{\text{det}(2\pi\tilde{\mathbf{N}})}}\,\text{exp}\bigg[-\frac{1}{2}\big(\boldsymbol{\delta t}-\mathbf{F}\mathbf{a}\big)^{\text{T}}\,\mathbf{\tilde{N}}^{-1}\,\big(\boldsymbol{\delta t}-\mathbf{F}\mathbf{a}\big)\bigg]$ $\displaystyle\propto\frac{q(\mathbf{a}|\boldsymbol{\eta})\cdot p(\boldsymbol{\eta})}{\sqrt{\text{det}(2\pi\tilde{\mathbf{N}})}}\,\text{exp}\bigg[\boldsymbol{\delta t}^{\text{T}}\,\tilde{\mathbf{N}}^{-1}\,\mathbf{F}\mathbf{a}-\frac{1}{2}\mathbf{a}^{\text{T}}\,\mathbf{F}^{\text{T}}\,\tilde{\mathbf{N}}^{-1}\,\mathbf{F}\mathbf{a}\bigg]$ (67) $$

where we have opted to analytically marginalize over linear deviations to the timing model for convenience. Note we may compute $\boldsymbol{\delta t}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}$ and $\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}$ once and store for all future evaluations, so Eq. ([D](#A4.Ex15)) is as computationally efficient to evaluate as the Gaussian posterior, Eq. ([28](#S4.E28)). The only additional cost may come from the evaluation of the modified prior $q(\mathbf{a}|\boldsymbol{\eta})$ with non-Gaussian moments.

Lastly, we must determine the proper standardizing transform for this posterior. While $q(\mathbf{a}|\boldsymbol{\eta})$ is non-Gaussian, we’ll assume the Laplace approximation is valid in a neighborhood about the MAP solution, $\mathbf{a}=\check{\mathbf{a}}$,

$$ $q(\mathbf{a}|\boldsymbol{\eta})\bigg|_{\mathbf{a}\approx\check{\mathbf{a}}}\approx\frac{1}{\sqrt{\text{det}(2\pi\boldsymbol{\varphi})}}\,\text{exp}\bigg[-\frac{1}{2}\big(\mathbf{a}-\boldsymbol{\xi}\big)^{\text{T}}\,\boldsymbol{\varphi}^{-1}\,\big(\mathbf{a}-\boldsymbol{\xi}\big)\bigg]$ (68) $$

where $\boldsymbol{\xi}$ and $\boldsymbol{\varphi}$ are the mean and covariance of the non-Gaussian prior under the Laplace approximation. Then we may estimate the MAP coefficients and the covariance of the posterior using the extremum condition and the Hessian of the log-posterior to find $\check{\mathbf{a}}=\boldsymbol{\Psi}\big(\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\boldsymbol{\delta t}+\boldsymbol{\varphi}^{-1}\boldsymbol{\xi}\big)$ and $\boldsymbol{\Psi}=\big(\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}+\boldsymbol{\varphi}^{-1}\big)^{-1}$, respectively. Under the replacement $\hat{\mathbf{a}}\rightarrow\check{\mathbf{a}}$ and $\boldsymbol{\Sigma}\rightarrow\boldsymbol{\Psi}$ we may write the standardizing transform, Eq. ([29](#S4.E29)), as

$$ $(\mathbf{a},\;\boldsymbol{\eta})=T^{-1}(\mathbf{z},\boldsymbol{\eta})=(\check{\mathbf{a}}+\boldsymbol{\mathcal{L}}\,\mathbf{z},\;\boldsymbol{\eta})$ (69) $$

where $\boldsymbol{\mathcal{L}}$ is the Cholesky decomposition of the covariance matrix $\boldsymbol{\Psi}=\boldsymbol{\mathcal{L}}\boldsymbol{\mathcal{L}}^{\text{T}}$. As above, we’ll also approximate the covariance of the standardizing transformation with a CURN model, neglecting inter-pulsar correlations for computational efficiency. This amounts to replacing $\boldsymbol{\mathcal{L}}\rightarrow\boldsymbol{\mathcal{L}}_{\text{CURN}}$ where $\boldsymbol{\mathcal{L}}_{\text{CURN}}$ is the Cholesky decomposition of the covariance matrix which neglects inter-pulsar correlations, $\boldsymbol{\Psi}_{\text{CURN}}=\big(\mathbf{F}^{\text{T}}\tilde{\mathbf{N}}^{-1}\mathbf{F}+\boldsymbol{\varphi}_{\text{CURN}}^{-1}\big)^{-1}=\boldsymbol{\mathcal{L}}_{\text{CURN}}\boldsymbol{\mathcal{L}}_{\text{CURN}}^{\text{T}}$with $\boldsymbol{\varphi}_{\text{CURN}}$ being identical to $\boldsymbol{\varphi}$, except inter-pulsar correlations are fixed to zero so the implemented transformation is

$$ $(\mathbf{a},\;\boldsymbol{\eta})=T^{-1}(\mathbf{z},\boldsymbol{\eta})=(\check{\mathbf{a}}+\boldsymbol{\mathcal{L}}_{\text{CURN}}\,\mathbf{z},\;\boldsymbol{\eta})\,.$ (70) $$

For significantly non-Gaussian priors, such as those with heavy tails, the Laplace approximation is not valid in a large portion of parameter space. Similarly, we rely on inter-pulsar correlations to claim a detection of a stochastic GWB, but the standardizing transformation above does not capture any of these features. This does not bias our inference as the purpose of the standardizing transform is only to approximately de-correlate the parameter space. After the transformation, the posterior which includes inter-pulsar correlations and all non-Gaussian features is evaluated, Eq. ([D](#A4.Ex15)), to determine the transition probability of the chain. Thus we may infer non-Gaussian features and inter-pulsar correlations while neglecting these contributions in the coordinate transformation. Moreover, the latest analyses [8] suggest there is significant evidence for a CURN model under the Gaussian approximation so a coordinate transformation under identical model assumptions will yield an effective reparameterization.

We test our non-Gaussian framework by simulating a stochastic GWB in a collection of simulated pulsars according to Student’s t-distribution, which includes non-Gaussian features (for an overview of Student’s t-distribution, see e.g. [49]). Student’s t-distribution is a useful toy example because it generalizes the normal distribution under a simple parameterization, and can achieve heavy non-Gaussian tails. The probability density function for Student’s t-distribution is

$$ $p_{\nu}(t)=\frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\pi\nu}\Gamma(\nu/2)}\bigg(1+\frac{t^{2}}{\nu}\bigg)^{-(\nu+1)/2}$ (71) $$

where $\nu$ is the number of degrees of freedom and $\Gamma$ is the gamma function. Student’s t-distribution has mean 0 (for $\nu>1$) and variance $\nu/(\nu-2)$ (for $\nu>2$). For $\nu=1$ and $\nu\rightarrow\infty$, Student’s t-distribution converges to the standard Cauchy and normal distributions, respectively. For finite $\nu$, Student’s t-distribution exhibits heavier tails than a Gaussian.

We simulate the timing delays induced by a non-Gaussian stochastic GWB by drawing $2N_{f}N_{p}$ independent random variables, $\mathbf{t}$, from Student’s t-distribution, Eq. ([D](#A4.Ex17)), with some chosen (finite) value for $\nu$. Then the prior covariance matrix, $\boldsymbol{\phi}$, for the Fourier coefficients due to an HD-correlated GWB, Eq. ([21](#S4.E21)), is constructed under a power law spectral model, and we compute the Cholesky decomposition of $\sqrt{(\nu-2)/\nu}\,\boldsymbol{\phi}=\mathbf{L}_{\nu}\mathbf{L}_{\nu}^{\text{T}}$. Finally we color the draws, $\mathbf{t}$, using this Cholesky decomposition $\mathbf{a}_{\nu}=\mathbf{L}_{\nu}\mathbf{t}$ and the induced timing delays are $\boldsymbol{\delta t}_{\nu}=\mathbf{F}\mathbf{a}_{\nu}$. This simulation determines our prior which exhibits the usual first two statistical moments, $\mathbb{E}[\mathbf{a}_{\nu}]=\mathbf{0}$ and $\mathbb{E}[\mathbf{a}_{\nu}\,\mathbf{a}_{\nu}^{\text{T}}]=\boldsymbol{\phi}$ as expected for the GWB, but the distribution of Fourier coefficients include higher-order statistical moments due to the initial draws from Student’s t-distribution. Because the coloring procedure is a linear coordinate transformation, the prior probability density function consistent with the simulation is derived from the probability density function for the initial independent draws,

$$ $\displaystyle q(\mathbf{a}_{\nu}|\boldsymbol{\eta})$ $\displaystyle=\tilde{q}(\mathbf{t}|\boldsymbol{\eta})\cdot\text{det}(\partial\mathbf{t}/\partial\mathbf{a}_{\nu})$ $\displaystyle=\bigg[\prod_{i=1}^{2N_{f}N_{p}}p_{\nu}(t_{i})\bigg]\cdot\text{det}(\partial\mathbf{t}/\partial\mathbf{a}_{\nu})$ $\displaystyle=\frac{1}{\text{det}(\mathbf{L}_{\nu})}\prod_{i=1}^{2N_{f}N_{p}}p_{\nu}((\mathbf{L}_{\nu}^{-1}\mathbf{a}_{\nu})_{i})$ (72) $$

where we’ve substituted the inverse transformation $\mathbf{t}=\mathbf{L}_{\nu}^{-1}\mathbf{a}_{\nu}$ in the last line. Note that in the $\nu\rightarrow\infty$ limit the GWB simulation procedure and the probability density function Eq. ([D](#A4.Ex17)) converge to that of the multivariate normal distribution.

We sample Eq. ([D](#A4.Ex15)) using Student’s t-distribution as a prior, Eq. ([D](#A4.Ex17)), under the standardizing transformation Eq. ([70](#A4.E70)). Data with non-Gaussian features has been simulated such that the mean and covariance of the Fourier coefficients is $\mathbb{E}[\mathbf{a}]=\mathbf{0}$ and $\mathbb{E}[\mathbf{a}\mathbf{a}^{\text{T}}]=\boldsymbol{\phi}$ respectively, so the standardizing transformation, Eq. ([70](#A4.E70)) reduces to the usual transformation, Eq. ([30](#S4.E30)). In other words, simulating data according to Student’s t-distribution induces higher-order statistical moments in the target distribution, but the standardizing transformation, which uses only the estimated first and second moments is unchanged. The recovery of the background parameters is shown in Fig. ([7](#A4.F7)). With a sufficiently loud injection, we are able to resolve non-Gaussian features in the background by recovering a finite value for $\nu$.

Figure: Figure 7: Recovery of a simulated gravitational wave background with non-Gaussian features. The blue lines are injected parameter values. Resolving finite $\nu$ indicates statistical moments beyond Gaussianity.
Refer to caption: https://arxiv.org/html/2607.06834/2607.06834v1/figs/student_t.png

## References

- [1]
A. G. Abac et al. (2026)
GWTC-5.0: an introduction to version 5.0 of the gravitational-wave transient catalog.
External Links: 2605.27223,
[Link](https://arxiv.org/abs/2605.27223)
Cited by: [§II](#S2.p2.1).
- [2]
F. Acernese et al. (2019-12)
Increasing the astrophysical reach of the advanced virgo detector via the application of squeezed vacuum states of light.
Phys. Rev. Lett. 123, pp. 231108.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevLett.123.231108),
[Link](https://link.aps.org/doi/10.1103/PhysRevLett.123.231108)
Cited by: [§II](#S2.p2.1).
- [3]
M. R. Adams, N. J. Cornish, and T. B. Littenberg (2012)
Astrophysical Model Selection in Gravitational Wave Astronomy.
Phys. Rev. D 86, pp. 124032.
External Links: 1209.6286,
[Document](https://dx.doi.org/10.1103/PhysRevD.86.124032)
Cited by: [§II](#S2.p2.1).
- [4]
A. Afzal et al. (2023-06)
The nanograv 15 yr data set: search for signals from new physics.
The Astrophysical Journal Letters 951 (1), pp. L11.
External Links: [Document](https://dx.doi.org/10.3847/2041-8213/acdc91),
[Link](https://doi.org/10.3847/2041-8213/acdc91)
Cited by: [§I](#S1.p2.1),
[§IV.1](#S4.SS1.p2.8).
- [5]
G. Agazie, et al, and Nanograv Collaboration (2023-08)
The NANOGrav 15 yr Data Set: Constraints on Supermassive Black Hole Binaries from the Gravitational-wave Background.
\apjl 952 (2), pp. L37.
External Links: [Document](https://dx.doi.org/10.3847/2041-8213/ace18b),
2306.16220
Cited by: [§IV.1](#S4.SS1.p2.8).
- [6]
G. Agazie et al. (2023-07)
The nanograv 15 yr data set: bayesian limits on gravitational waves from individual supermassive black hole binaries.
The Astrophysical Journal Letters 951 (2), pp. L50.
External Links: [Document](https://dx.doi.org/10.3847/2041-8213/ace18a),
[Link](https://doi.org/10.3847/2041-8213/ace18a)
Cited by: [§VI.1](#S6.SS1.p3.1).
- [7]
G. Agazie et al. (2023-06)
The nanograv 15 yr data set: detector characterization and noise budget.
The Astrophysical Journal Letters 951 (1), pp. L10.
External Links: [Document](https://dx.doi.org/10.3847/2041-8213/acda88),
[Link](https://doi.org/10.3847/2041-8213/acda88)
Cited by: [§III.2](#S3.SS2.p1.5).
- [8]
G. Agazie et al. (2023-06)
The nanograv 15 yr data set: evidence for a gravitational-wave background.
The Astrophysical Journal Letters 951 (1), pp. L8.
External Links: [Document](https://dx.doi.org/10.3847/2041-8213/acdac6),
[Link](https://doi.org/10.3847/2041-8213/acdac6)
Cited by: [Appendix D](#A4.p5.1),
[§I](#S1.p1.1),
[§I](#S1.p10.1),
[§I](#S1.p6.1),
[§IV.2](#S4.SS2.p9.2),
[§VI](#S6.p1.2).
- [9]
G. Agazie et al. (2023-06)
The nanograv 15 yr data set: observations and timing of 68 millisecond pulsars.
The Astrophysical Journal Letters 951 (1), pp. L9.
External Links: [Document](https://dx.doi.org/10.3847/2041-8213/acda9a),
[Link](https://doi.org/10.3847/2041-8213/acda9a)
Cited by: [§III](#S3.p1.4).
- [10]
G. Agazie et al. (2023-06)
The nanograv 15 yr data set: observations and timing of 68 millisecond pulsars.
The Astrophysical Journal Letters 951 (1), pp. L9.
External Links: [Document](https://dx.doi.org/10.3847/2041-8213/acda9a),
[Link](https://doi.org/10.3847/2041-8213/acda9a)
Cited by: [§VI](#S6.p1.2).
- [11]
B. Allen and J. D. Romano (2025-01)
Optimal reconstruction of the hellings and downs correlation.
Phys. Rev. Lett. 134, pp. 031401.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevLett.134.031401),
[Link](https://link.aps.org/doi/10.1103/PhysRevLett.134.031401)
Cited by: [§IV.4](#S4.SS4.p1.1).
- [12]
B. Allen, A. L. von Blanckenburg, and K. D. Olum (2026-05)
Pulsar timing array analysis in a legendre polynomial basis.
Phys. Rev. D 113, pp. 102001.
External Links: [Document](https://dx.doi.org/10.1103/2dft-4rjj),
[Link](https://link.aps.org/doi/10.1103/2dft-4rjj)
Cited by: [§IV.4](#S4.SS4.p1.1).
- [13]
B. Bécsy, N. J. Cornish, and M. C. Digman (2022)
Fast bayesian analysis of individual binaries in pulsar timing array data.
Physical Review D 105 (12), pp. 122003.
Cited by: [§VI.1](#S6.SS1.p3.1).
- [14]
A. Beskos, N. Pillai, G. Roberts, J. Sanz-Serna, and A. Stuart (2013)
Optimal tuning of the hybrid Monte Carlo algorithm.
Bernoulli 19 (5A), pp. 1501 – 1534.
External Links: [Document](https://dx.doi.org/10.3150/12-BEJ414),
[Link](https://doi.org/10.3150/12-BEJ414)
Cited by: [§I](#S1.p9.5).
- [15]
M. Betancourt, S. Byrne, S. Livingstone, and M. Girolami (2017)
The geometric foundations of Hamiltonian Monte Carlo.
Bernoulli 23 (4A), pp. 2257 – 2298.
External Links: [Document](https://dx.doi.org/10.3150/16-BEJ810),
[Link](https://doi.org/10.3150/16-BEJ810)
Cited by: [Appendix A](#A1.p1.1).
- [16]
M. Betancourt (2013)
A general metric for riemannian manifold hamiltonian monte carlo.
In Geometric Science of Information, F. Nielsen and F. Barbaresco (Eds.),
Berlin, Heidelberg, pp. 327–334.
External Links: ISBN 978-3-642-40020-9
Cited by: [§II.1](#S2.SS1.p2.1).
- [17]
M. Betancourt (2017-01)
A Conceptual Introduction to Hamiltonian Monte Carlo.
arXiv e-prints.
External Links: 1701.02434
Cited by: [Appendix A](#A1.p1.1),
[Appendix A](#A1.p7.1),
[§I](#S1.p9.5).
- [18]
E. Bingham, J. P. Chen, M. Jankowiak, F. Obermeyer, N. Pradhan, T. Karaletsos, R. Singh, P. A. Szerlip, P. Horsfall, and N. D. Goodman (2019)
Pyro: deep universal probabilistic programming.
J. Mach. Learn. Res. 20, pp. 28:1–28:6.
External Links: [Link](http://jmlr.org/papers/v20/18-403.html)
Cited by: [Appendix A](#A1.p1.1),
[§IV.2](#S4.SS2.p11.1).
- [19]
J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang (2018)
JAX: composable transformations of Python+NumPy programs.
Note: [http://github.com/jax-ml/jax](http://github.com/jax-ml/jax)
Cited by: [§I](#S1.p6.1),
[§I](#S1.p8.1),
[§IV.2](#S4.SS2.p11.1).
- [20]
A. Buikema et al. (2020-09)
Sensitivity and performance of the advanced ligo detectors in the third observing run.
Phys. Rev. D 102, pp. 062003.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevD.102.062003),
[Link](https://link.aps.org/doi/10.1103/PhysRevD.102.062003)
Cited by: [§II](#S2.p2.1).
- [21]
S. Burke-Spolaor, S. R. Taylor, M. Charisi, T. Dolch, J. S. Hazboun, A. M. Holgado, L. Z. Kelley, T. J. W. Lazio, D. R. Madison, N. McMann, C. M. F. Mingarelli, A. Rasskazov, X. Siemens, J. J. Simon, and T. L. Smith (2019/06/18)
The astrophysics of nanohertz gravitational waves.
The Astronomy and Astrophysics Review 27 (1), pp. 5.
External Links: [Document](https://dx.doi.org/10.1007/s00159-019-0115-7),
ISBN 1432-0754,
[Link](https://doi.org/10.1007/s00159-019-0115-7)
Cited by: [§I](#S1.p2.1).
- [22]
D. J. Champion, G. B. Hobbs, R. N. Manchester, R. T. Edwards, D. C. Backer, M. Bailes, N. D. R. Bhat, S. Burke-Spolaor, W. Coles, P. B. Demorest, R. D. Ferdman, W. M. Folkner, A. W. Hotan, M. Kramer, A. N. Lommen, D. J. Nice, M. B. Purver, J. M. Sarkissian, I. H. Stairs, W. van Straten, J. P. W. Verbiest, and D. R. B. Yardley (2010-08)
MEASURING the mass of solar system planets using pulsar timing.
The Astrophysical Journal Letters 720 (2), pp. L201.
External Links: [Document](https://dx.doi.org/10.1088/2041-8205/720/2/L201),
[Link](https://doi.org/10.1088/2041-8205/720/2/L201)
Cited by: [§III.4](#S3.SS4.p1.2).
- [23]
V. Corbin and N. J. Cornish (2010-08)
Pulsar Timing Array Observations of Massive Black Hole Binaries.
arXiv e-prints.
External Links: 1008.1782
Cited by: [§III.4](#S3.SS4.p1.2),
[§III.4](#S3.SS4.p2.9),
[§V.1](#S5.SS1.p3.7),
[§VI](#S6.p3.15).
- [24]
M. Crisostomi, R. van Haasteren, P. M. Meyers, and M. Vallisneri (2025-06)
Beyond diagonal approximations: improved covariance modeling for pulsar timing array data analysis.
arXiv e-prints.
External Links: 2506.13866
Cited by: [§IV.4](#S4.SS4.p1.1),
[§IV.4](#S4.SS4.p6.1).
- [25]
S. Detweiler (1979-12)
Pulsar timing measurements and the search for gravitational waves.
Astrophys. J.  234, pp. 1100–1104.
External Links: [Document](https://dx.doi.org/10.1086/157593)
Cited by: [§I](#S1.p1.1).
- [26]
S. Duane, A.D. Kennedy, B. J. Pendleton, and D. Roweth (1987)
Hybrid monte carlo.
Physics Letters B 195 (2), pp. 216–222.
External Links: ISSN 0370-2693,
[Document](https://dx.doi.org/https%3A//doi.org/10.1016/0370-2693%2887%2991197-X),
[Link](https://www.sciencedirect.com/science/article/pii/037026938791197X)
Cited by: [Appendix A](#A1.p1.1).
- [27]
J. A. Ellis (2013-11)
A bayesian analysis pipeline for continuous gw sources in the pta band.
Classical and Quantum Gravity 30 (22), pp. 224004.
External Links: [Document](https://dx.doi.org/10.1088/0264-9381/30/22/224004),
[Link](https://doi.org/10.1088/0264-9381/30/22/224004)
Cited by: [§III.4](#S3.SS4.p2.9),
[§V.1](#S5.SS1.p3.7),
[§VI](#S6.p3.15).
- [28]
J. A. Ellis, X. Siemens, and J. D. E. Creighton (2012-08)
OPTIMAL strategies for continuous gravitational wave detection in pulsar timing arrays.
The Astrophysical Journal 756 (2), pp. 175.
External Links: [Document](https://dx.doi.org/10.1088/0004-637X/756/2/175),
[Link](https://doi.org/10.1088/0004-637X/756/2/175)
Cited by: [§III.4](#S3.SS4.p2.9),
[§V.1](#S5.SS1.p3.7),
[§VI](#S6.p3.15).
- [29]
J. A. Ellis, M. Vallisneri, S. R. Taylor, and P. T. Baker (2019-12)
ENTERPRISE: Enhanced Numerical Toolbox Enabling a Robust PulsaR Inference SuitE.
Note: Astrophysics Source Code Library, record ascl:1912.015
Cited by: [§I](#S1.p6.1),
[Figure 4](#S6.F4),
[Figure 5](#S6.F5).
- [30]
EPTA Collaboration, InPTA Collaboration, et al. (2023-10)
The second data release from the European Pulsar Timing Array. III. Search for gravitational wave signals.
\aap 678, pp. A50.
External Links: [Document](https://dx.doi.org/10.1051/0004-6361/202346844),
2306.16214
Cited by: [§I](#S1.p1.1).
- [31]
M. Falxa and A. Sesana (2026-02)
Modeling non-gaussianities in pulsar timing array data analysis using gaussian mixture models.
Phys. Rev. D 113, pp. 043047.
External Links: [Document](https://dx.doi.org/10.1103/9jdd-6ct8),
[Link](https://link.aps.org/doi/10.1103/9jdd-6ct8)
Cited by: [Appendix D](#A4.p2.1).
- [32]
M. Favata (2009)
Nonlinear gravitational-wave memory from binary black hole mergers.
Astrophys. J. Lett. 696, pp. L159–L162.
External Links: 0902.3660,
[Document](https://dx.doi.org/10.1088/0004-637X/696/2/L159)
Cited by: [§III.4](#S3.SS4.p1.2).
- [33]
M. Favata (2009-07)
Post-newtonian corrections to the gravitational-wave memory for quasicircular, inspiralling compact binaries.
Phys. Rev. D 80, pp. 024002.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevD.80.024002),
[Link](https://link.aps.org/doi/10.1103/PhysRevD.80.024002)
Cited by: [§III.4](#S3.SS4.p1.2).
- [34]
G. E. Freedman, A. D. Johnson, R. van Haasteren, and S. J. Vigeland (2023-02)
Efficient gravitational wave searches with pulsar timing arrays using hamiltonian monte carlo.
Phys. Rev. D 107, pp. 043013.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevD.107.043013),
[Link](https://link.aps.org/doi/10.1103/PhysRevD.107.043013)
Cited by: [§I](#S1.p9.5).
- [35]
G. E. Freedman and S. J. Vigeland (2024-09)
Efficient pipeline for joint gravitational wave searches from individual binaries and a gravitational wave background with hamiltonian sampling.
Phys. Rev. D 110, pp. 063038.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevD.110.063038),
[Link](https://link.aps.org/doi/10.1103/PhysRevD.110.063038)
Cited by: [§I](#S1.p9.5).
- [36]
A. Gelman, W. R. Gilks, and G. O. Roberts (1997)
Weak convergence and optimal scaling of random walk Metropolis algorithms.
The Annals of Applied Probability 7 (1), pp. 110 – 120.
External Links: [Document](https://dx.doi.org/10.1214/aoap/1034625254),
[Link](https://doi.org/10.1214/aoap/1034625254)
Cited by: [§I](#S1.p9.5).
- [37]
A. Gelman, J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari, and D. B. Rubin (2013-11)
Bayesian data analysis.
Chapman and Hall/CRC.
External Links: ISBN 9780429113079,
[Link](http://dx.doi.org/10.1201/b16018),
[Document](https://dx.doi.org/10.1201/b16018)
Cited by: [§II](#S2.p2.1).
- [38]
J. W. Gibbs (1899-04)
Fourier’s series.
Nature 59 (1539), pp. 606–606.
External Links: ISSN 1476-4687,
[Link](http://dx.doi.org/10.1038/059606a0),
[Document](https://dx.doi.org/10.1038/059606a0)
Cited by: [§IV.3](#S4.SS3.p2.3).
- [39]
M. Girolami and B. Calderhead (2011)
Riemann manifold langevin and hamiltonian monte carlo methods.
Journal of the Royal Statistical Society: Series B (Statistical Methodology) 73 (2), pp. 123–214.
External Links: [Document](https://dx.doi.org/https%3A//doi.org/10.1111/j.1467-9868.2010.00765.x),
[Link](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2010.00765.x),
https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/j.1467-9868.2010.00765.x
Cited by: [§II.1](#S2.SS1.p2.1).
- [40]
B. Goncharov and S. Sardana (2025-03)
Ensemble noise properties of the european pulsar timing array.
Monthly Notices of the Royal Astronomical Society 537 (4), pp. 3470–3479.
External Links: ISSN 0035-8711,
[Document](https://dx.doi.org/10.1093/mnras/staf190),
[Link](https://doi.org/10.1093/mnras/staf190),
https://academic.oup.com/mnras/article-pdf/537/4/3470/61743825/staf190.pdf
Cited by: [§II](#S2.p2.1).
- [41]
B. Goncharov, E. Thrane, R. M. Shannon, J. Harms, N. D. R. Bhat, G. Hobbs, M. Kerr, R. N. Manchester, D. J. Reardon, C. J. Russell, X. Zhu, and A. Zic (2022-06)
Consistency of the parkes pulsar timing array signal with a nanohertz gravitational-wave background.
The Astrophysical Journal Letters 932 (2), pp. L22.
External Links: [Document](https://dx.doi.org/10.3847/2041-8213/ac76bb),
[Link](https://doi.org/10.3847/2041-8213/ac76bb)
Cited by: [§II](#S2.p2.1).
- [42]
A. Gundersen and N. J. Cornish (2025-10)
Escaping neal’s funnel: a multi-stage sampling method for hierarchical models.
arXiv e-prints.
External Links: 2510.12917
Cited by: [§I](#S1.p7.1),
[§II.1](#S2.SS1.p2.1),
[§IV.2](#S4.SS2.p6.1).
- [43]
A. Gundersen and N. J. Cornish (2025-10)
Rapid inference for individual binaries and a stochastic background with pulsar timing array data.
Phys. Rev. D 112, pp. 083035.
External Links: [Document](https://dx.doi.org/10.1103/5xh1-kgtk),
[Link](https://link.aps.org/doi/10.1103/5xh1-kgtk)
Cited by: [§I](#S1.p7.1),
[§II.1](#S2.SS1.p2.1),
[§IV.3](#S4.SS3.p1.3),
[§IV.3](#S4.SS3.p2.3).
- [44]
W. K. Hastings (1970-04)
Monte Carlo Sampling Methods using Markov Chains and their Applications.
Biometrika 57 (1), pp. 97–109.
External Links: [Document](https://dx.doi.org/10.1093/biomet/57.1.97)
Cited by: [Appendix A](#A1.p3.5).
- [45]
R. W. Hellings and G. S. Downs (1983-02)
Upper limits on the isotropic gravitational radiation background from pulsar timing analysis..
\apjl 265, pp. L39–L42.
External Links: [Document](https://dx.doi.org/10.1086/183954)
Cited by: [§IV.1](#S4.SS1.p1.6).
- [46]
M. D. Hoffman and A. Gelman (2014)
The no-u-turn sampler: adaptively setting path lengths in hamiltonian monte carlo.
Journal of Machine Learning Research 15 (47), pp. 1593–1623.
External Links: [Link](http://jmlr.org/papers/v15/hoffman14a.html)
Cited by: [Appendix A](#A1.p8.2).
- [47]
A. H. Jaffe and D. C. Backer (2003-02)
Gravitational waves probe the coalescence rate of massive black hole binaries.
The Astrophysical Journal 583 (2), pp. 616.
External Links: [Document](https://dx.doi.org/10.1086/345443),
[Link](https://doi.org/10.1086/345443)
Cited by: [§I](#S1.p2.1),
[§III.3](#S3.SS3.p1.2).
- [48]
M. J. Keith, W. Coles, R. M. Shannon, G. B. Hobbs, R. N. Manchester, M. Bailes, N. D. R. Bhat, S. Burke-Spolaor, D. J. Champion, A. Chaudhary, A. W. Hotan, J. Khoo, J. Kocz, S. Osłowski, V. Ravi, J. E. Reynolds, J. Sarkissian, W. van Straten, and D. R. B. Yardley (2013-03)
Measurement and correction of variations in interstellar dispersion in high-precision pulsar timing.
\mnras 429 (3), pp. 2161–2174.
External Links: [Document](https://dx.doi.org/10.1093/mnras/sts486),
1211.5887
Cited by: [§III.3](#S3.SS3.p2.6).
- [49]
J. L. Kirkby, D. Nguyen, and D. Nguyen (2024)
Moments of student’s t-distribution: a unified approach.
External Links: 1912.01607,
[Link](https://arxiv.org/abs/1912.01607)
Cited by: [Appendix D](#A4.p6.9).
- [50]
A. Kuntz, C. Smarra, and M. Vaglio (2026)
Looking for non-gaussianity in pulsar timing arrays through the four point correlator.
External Links: 2603.12311,
[Link](https://arxiv.org/abs/2603.12311)
Cited by: [Appendix D](#A4.p1.1),
[§IV.1](#S4.SS1.p4.1).
- [51]
N. Laal, W. G. Lamb, J. D. Romano, X. Siemens, S. R. Taylor, and R. van Haasteren (2023-09)
Exploring the capabilities of gibbs sampling in pulsar timing arrays.
Phys. Rev. D 108, pp. 063008.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevD.108.063008),
[Link](https://link.aps.org/doi/10.1103/PhysRevD.108.063008)
Cited by: [§I](#S1.p7.1).
- [52]
N. Laal, S. R. Taylor, R. van Haasteren, W. G. Lamb, and X. Siemens (2025-03)
Solving the pta data analysis problem with a global gibbs scheme.
Phys. Rev. D 111, pp. 063067.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevD.111.063067),
[Link](https://link.aps.org/doi/10.1103/PhysRevD.111.063067)
Cited by: [§I](#S1.p7.1).
- [53]
W. G. Lamb, S. R. Taylor, and R. van Haasteren (2023-11)
Rapid refitting techniques for bayesian spectral characterization of the gravitational wave background using pulsar timing arrays.
Phys. Rev. D 108, pp. 103019.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevD.108.103019),
[Link](https://link.aps.org/doi/10.1103/PhysRevD.108.103019)
Cited by: [§IV.1](#S4.SS1.p3.1).
- [54]
W. G. Lamb and S. R. Taylor (2024-08)
Spectral variance in a stochastic gravitational-wave background from a binary population.
The Astrophysical Journal Letters 971 (1), pp. L10.
External Links: [Document](https://dx.doi.org/10.3847/2041-8213/ad654a),
[Link](https://doi.org/10.3847/2041-8213/ad654a)
Cited by: [Appendix D](#A4.p1.1),
[§IV.1](#S4.SS1.p4.1).
- [55]
W. G. Lamb, J. M. Wachter, A. Mitridate, S. C. Sardesai, B. Bécsy, E. L. Hagen, S. R. Taylor, and L. Z. Kelley (2026-06)
Finite populations and finite time: the non-gaussianity of a gravitational wave background.
Phys. Rev. D 113, pp. 123065.
External Links: [Document](https://dx.doi.org/10.1103/96zk-qtck),
[Link](https://link.aps.org/doi/10.1103/96zk-qtck)
Cited by: [Appendix D](#A4.p1.1),
[§IV.1](#S4.SS1.p4.1).
- [56]
K. J. Lee, C. G. Bassa, G. H. Janssen, R. Karuppusamy, M. Kramer, K. Liu, D. Perrodin, R. Smits, B. W. Stappers, R. van Haasteren, and L. Lentati (2014-05)
Model-based asymptotically optimal dispersion measure correction for pulsar timing.
Monthly Notices of the Royal Astronomical Society 441 (4), pp. 2831–2844.
External Links: ISSN 0035-8711,
[Link](http://dx.doi.org/10.1093/mnras/stu664),
[Document](https://dx.doi.org/10.1093/mnras/stu664)
Cited by: [§III.3](#S3.SS3.p2.6).
- [57]
K. J. Lee, N. Wex, M. Kramer, B. W. Stappers, C. G. Bassa, G. H. Janssen, R. Karuppusamy, and R. Smits (2011-05)
Gravitational wave astronomy of single sources with a pulsar timing array: gw astronomy of single sources.
Monthly Notices of the Royal Astronomical Society 414 (4), pp. 3251–3264.
External Links: ISSN 0035-8711,
[Link](http://dx.doi.org/10.1111/j.1365-2966.2011.18622.x),
[Document](https://dx.doi.org/10.1111/j.1365-2966.2011.18622.x)
Cited by: [§III.4](#S3.SS4.p1.2),
[§V.1](#S5.SS1.p3.7).
- [58]
L. Lentati, M. P. Hobson, and P. Alexander (2014-11)
Bayesian estimation of non-gaussianity in pulsar timing analysis.
Monthly Notices of the Royal Astronomical Society 444 (4), pp. 3863–3878.
External Links: ISSN 0035-8711,
[Document](https://dx.doi.org/10.1093/mnras/stu1721),
[Link](https://doi.org/10.1093/mnras/stu1721),
https://academic.oup.com/mnras/article-pdf/444/4/3863/6362238/stu1721.pdf
Cited by: [Appendix D](#A4.p2.1).
- [59]
L. Lentati, P. Alexander, M. P. Hobson, S. Taylor, J. Gair, S. T. Balan, and R. van Haasteren (2013-05)
Hyper-efficient model-independent bayesian method for the analysis of pulsar timing data.
Phys. Rev. D 87, pp. 104021.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevD.87.104021),
[Link](https://link.aps.org/doi/10.1103/PhysRevD.87.104021)
Cited by: [§I](#S1.p6.1),
[§I](#S1.p7.1),
[§IV.1](#S4.SS1.p3.1),
[§IV.2](#S4.SS2.p11.1),
[§IV.2](#S4.SS2.p4.9),
[§IV.2](#S4.SS2.p5.3),
[§IV.2](#S4.SS2.p7.3),
[§IV.4](#S4.SS4.p1.1),
[Figure 3](#S5.F3),
[§V.2](#S5.SS2.p2.3),
[§VI.2](#S6.SS2.p1.1).
- [60]
M. Mancarella and D. Gerosa (2025-05)
Sampling the full hierarchical population posterior distribution in gravitational-wave astronomy.
Phys. Rev. D 111, pp. 103012.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevD.111.103012),
[Link](https://link.aps.org/doi/10.1103/PhysRevD.111.103012)
Cited by: [§II](#S2.p2.1).
- [61]
I. Mandel, W. M. Farr, and J. R. Gair (2019-03)
Extracting distribution parameters from multiple uncertain observations with selection biases.
Monthly Notices of the Royal Astronomical Society 486 (1), pp. 1086–1093.
External Links: ISSN 1365-2966,
[Link](http://dx.doi.org/10.1093/mnras/stz896),
[Document](https://dx.doi.org/10.1093/mnras/stz896)
Cited by: [§II](#S2.p2.1).
- [62]
S. T. McWilliams, J. P. Ostriker, and F. Pretorius (2014-07)
Gravitational Waves and Stalled Satellites from Massive Galaxy Mergers at z ¡= 1.
Astrophys. J.  789 (2), pp. 156.
External Links: [Document](https://dx.doi.org/10.1088/0004-637X/789/2/156),
1211.5377
Cited by: [§I](#S1.p2.1).
- [63]
N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, and E. Teller (1953)
Equation of state calculations by fast computing machines.
J. Chem. Phys. 21, pp. 1087–1092.
External Links: [Document](https://dx.doi.org/10.1063/1.1699114)
Cited by: [Appendix A](#A1.p3.5),
[§II.1](#S2.SS1.p1.1).
- [64]
M. T. Miles et al. (2025-01)
The meerkat pulsar timing array: the first search for gravitational waves with the meerkat radio telescope.
Monthly Notices of the Royal Astronomical Society 536 (2), pp. 1489–1500.
External Links: ISSN 0035-8711,
[Document](https://dx.doi.org/10.1093/mnras/stae2571),
[Link](https://doi.org/10.1093/mnras/stae2571),
https://academic.oup.com/mnras/article-pdf/536/2/1489/61215196/stae2571.pdf
Cited by: [§I](#S1.p1.1).
- [65]
R. M. Neal (1996)
Monte carlo implementation.
In Bayesian Learning for Neural Networks,
pp. 55–98.
External Links: ISBN 978-1-4612-0745-0,
[Document](https://dx.doi.org/10.1007/978-1-4612-0745-0%5F3),
[Link](https://doi.org/10.1007/978-1-4612-0745-0_3)
Cited by: [Appendix A](#A1.p1.1).
- [66]
R. M. Neal (2003)
Slice sampling.
The Annals of Statistics 31 (3), pp. 705 – 767.
External Links: [Document](https://dx.doi.org/10.1214/aos/1056562461),
[Link](https://doi.org/10.1214/aos/1056562461)
Cited by: [§II.1](#S2.SS1.p1.1),
[§II.3](#S2.SS3.p1.10).
- [67]
R. M. Neal (2011-05)
Handbook of Markov Chain Monte Carlo.
Chapman and Hall/CRC.
External Links: 1206.1901,
[Document](https://dx.doi.org/10.1201/b10905)
Cited by: [Appendix A](#A1.p1.1).
- [68]
O. Papaspiliopoulos, G. O. Roberts, and M. Sköld (2007)
A General Framework for the Parametrization of Hierarchical Models.
Statistical Science 22 (1), pp. 59 – 73.
External Links: [Document](https://dx.doi.org/10.1214/088342307000000014),
[Link](https://doi.org/10.1214/088342307000000014)
Cited by: [§II.1](#S2.SS1.p2.1),
[§II.2](#S2.SS2.p2.1).
- [69]
D. Phan, N. Pradhan, and M. Jankowiak (2019)
Composable effects for flexible and accelerated probabilistic programming in numpyro.
arXiv preprint arXiv:1912.11554.
External Links: [Link](https://arxiv.org/abs/1912.11554)
Cited by: [Appendix A](#A1.p1.1),
[§IV.2](#S4.SS2.p11.1).
- [70]
E. S. Phinney (2001-07)
A Practical theorem on gravitational wave backgrounds.
arXiv e-prints.
External Links: astro-ph/0108028
Cited by: [§III.3](#S3.SS3.p1.2),
[§IV.1](#S4.SS1.p3.1).
- [71]
J. Raidal, J. Urrutia, V. Vaskonen, and H. Veermäe (2026)
The heavy tailed non-gaussianity of the supermassive black hole gravitational wave background.
External Links: 2604.08506,
[Link](https://arxiv.org/abs/2604.08506)
Cited by: [Appendix D](#A4.p1.1),
[§IV.1](#S4.SS1.p4.1).
- [72]
M. Rajagopal and R. W. Romani (1995-06)
Ultra–Low-Frequency Gravitational Radiation from Massive Black Hole Binaries.
Astrophys. J.  446, pp. 543.
External Links: [Document](https://dx.doi.org/10.1086/175813),
astro-ph/9412038
Cited by: [§I](#S1.p2.1),
[§III.3](#S3.SS3.p1.2),
[§IV.1](#S4.SS1.p3.1).
- [73]
D. J. Reardon et al. (2023-06)
Search for an isotropic gravitational-wave background with the parkes pulsar timing array.
The Astrophysical Journal Letters 951 (1), pp. L6.
External Links: [Document](https://dx.doi.org/10.3847/2041-8213/acdd02),
[Link](https://doi.org/10.3847/2041-8213/acdd02)
Cited by: [§I](#S1.p1.1).
- [74]
L. Sampson, N. J. Cornish, and S. T. McWilliams (2015-04)
Constraining the solution to the last parsec problem with pulsar timing.
Phys. Rev. D 91, pp. 084055.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevD.91.084055),
[Link](https://link.aps.org/doi/10.1103/PhysRevD.91.084055)
Cited by: [§IV.1](#S4.SS1.p2.8).
- [75]
G. Sato-Polito and M. Zaldarriaga (2024)
The distribution of the gravitational-wave background from supermassive black holes.
External Links: 2406.17010,
[Link](https://arxiv.org/abs/2406.17010)
Cited by: [Appendix D](#A4.p1.1),
[§IV.1](#S4.SS1.p4.1).
- [76]
M. V. Sazhin (1978-02)
Opportunities for detecting ultralong gravitational waves.
\sovast 22, pp. 36–38.
Cited by: [§I](#S1.p1.1).
- [77]
A. Sesana, A. Vecchio, and C. N. Colacino (2008-10)
The stochastic gravitational-wave background from massive black hole binary systems: implications for observations with pulsar timing arrays.
Monthly Notices of the Royal Astronomical Society 390 (1), pp. 192–209.
External Links: ISSN 1365-2966,
[Link](http://dx.doi.org/10.1111/j.1365-2966.2008.13682.x),
[Document](https://dx.doi.org/10.1111/j.1365-2966.2008.13682.x)
Cited by: [§IV.1](#S4.SS1.p2.8).
- [78]
A. Sesana, F. Haardt, P. Madau, and M. Volonteri (2004-08)
Low-Frequency Gravitational Radiation from Coalescing Massive Black Hole Binaries in Hierarchical Cosmologies.
Astrophys. J.  611 (2), pp. 623–632.
External Links: [Document](https://dx.doi.org/10.1086/422185),
astro-ph/0401543
Cited by: [§I](#S1.p2.1).
- [79]
R. M. Shannon and J. M. Cordes (2010-11)
ASSESSING the role of spin noise in the precision timing of millisecond pulsars.
The Astrophysical Journal 725 (2), pp. 1607.
External Links: [Document](https://dx.doi.org/10.1088/0004-637X/725/2/1607),
[Link](https://doi.org/10.1088/0004-637X/725/2/1607)
Cited by: [§III.3](#S3.SS3.p1.2).
- [80]
D. Shih, M. Freytsis, S. R. Taylor, J. A. Dror, and N. Smyth (2024-07)
Fast parameter inference on pulsar timing arrays with normalizing flows.
Phys. Rev. Lett. 133, pp. 011402.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevLett.133.011402),
[Link](https://link.aps.org/doi/10.1103/PhysRevLett.133.011402)
Cited by: [§I](#S1.p4.1).
- [81]
J. Skilling (2013)
Maximum entropy and bayesian methods: cambridge, england, 1988.
Springer Science & Business Media.
Cited by: [Appendix B](#A2.p2.4).
- [82]
G. Strang et al. (2019)
Linear algebra and learning from data.
Vol. 4, Wellesley-Cambridge Press Cambridge.
Cited by: [§IV.2](#S4.SS2.p2.24).
- [83]
R. H. Swendsen and J. Wang (1986-11)
Replica monte carlo simulation of spin-glasses.
Phys. Rev. Lett. 57, pp. 2607–2609.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevLett.57.2607),
[Link](https://link.aps.org/doi/10.1103/PhysRevLett.57.2607)
Cited by: [Appendix B](#A2.p1.1).
- [84]
S. Taylor, J. Ellis, and J. Gair (2014-11)
Accelerated bayesian model-selection and parameter-estimation in continuous gravitational-wave searches with pulsar-timing arrays.
Phys. Rev. D 90, pp. 104028.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevD.90.104028),
[Link](https://link.aps.org/doi/10.1103/PhysRevD.90.104028)
Cited by: [§III.4](#S3.SS4.p2.9).
- [85]
S. R. Taylor, J. R. Gair, and L. Lentati (2013-02)
Weighing the evidence for a gravitational-wave background in the first international pulsar timing array data challenge.
Phys. Rev. D 87, pp. 044035.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevD.87.044035),
[Link](https://link.aps.org/doi/10.1103/PhysRevD.87.044035)
Cited by: [§IV.1](#S4.SS1.p3.1).
- [86]
S. R. Taylor and D. Gerosa (2018-10)
Mining gravitational-wave catalogs to understand binary stellar evolution: a new hierarchical bayesian framework.
Phys. Rev. D 98, pp. 083017.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevD.98.083017),
[Link](https://link.aps.org/doi/10.1103/PhysRevD.98.083017)
Cited by: [§II](#S2.p2.1).
- [87]
S. R. Taylor, J. Simon, and L. Sampson (2017-05)
Constraints on the dynamical environments of supermassive black-hole binaries using pulsar-timing arrays.
Phys. Rev. Lett. 118, pp. 181102.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevLett.118.181102),
[Link](https://link.aps.org/doi/10.1103/PhysRevLett.118.181102)
Cited by: [§I](#S1.p7.1).
- [88]
S. R. Taylor, J. Simon, L. Schult, N. Pol, and W. G. Lamb (2022-04)
A parallelized Bayesian approach to accelerated gravitational-wave background characterization.
Phys. Rev. D 105 (8), pp. 084049.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevD.105.084049),
2202.08293
Cited by: [§IV.1](#S4.SS1.p3.1).
- [89]
The Fermi-LAT Collaboration (2022)
A gamma-ray pulsar timing array constrains the nanohertz gravitational wave background.
Science 376 (6592), pp. 521–523.
External Links: [Document](https://dx.doi.org/10.1126/science.abm3231),
[Link](https://www.science.org/doi/abs/10.1126/science.abm3231),
https://www.science.org/doi/pdf/10.1126/science.abm3231
Cited by: [§V.1](#S5.SS1.p2.1).
- [90]
M. Vallisneri, S. R. Taylor, J. Simon, W. M. Folkner, R. S. Park, C. Cutler, J. A. Ellis, T. J. W. Lazio, S. J. Vigeland, K. Aggarwal, Z. Arzoumanian, P. T. Baker, A. Brazier, P. R. Brook, S. Burke-Spolaor, S. Chatterjee, J. M. Cordes, N. J. Cornish, F. Crawford, H. T. Cromartie, K. Crowter, M. DeCesar, P. B. Demorest, T. Dolch, R. D. Ferdman, E. C. Ferrara, E. Fonseca, N. Garver-Daniels, P. Gentile, D. Good, J. S. Hazboun, A. M. Holgado, E. A. Huerta, K. Islo, R. Jennings, G. Jones, M. L. Jones, D. L. Kaplan, L. Z. Kelley, J. S. Key, M. T. Lam, L. Levin, D. R. Lorimer, J. Luo, R. S. Lynch, D. R. Madison, M. A. McLaughlin, S. T. McWilliams, C. M. F. Mingarelli, C. Ng, D. J. Nice, T. T. Pennucci, N. S. Pol, S. M. Ransom, P. S. Ray, X. Siemens, R. Spiewak, I. H. Stairs, D. R. Stinebring, K. Stovall, J. K. Swiggum, R. van Haasteren, C. A. Witt, and W. W. Zhu (2020-04)
Modeling the uncertainties of solar system ephemerides for robust gravitational-wave searches with pulsar-timing arrays.
The Astrophysical Journal 893 (2), pp. 112.
External Links: [Document](https://dx.doi.org/10.3847/1538-4357/ab7b67),
[Link](https://doi.org/10.3847/1538-4357/ab7b67)
Cited by: [§III.4](#S3.SS4.p1.2).
- [91]
M. Vallisneri, M. Crisostomi, A. D. Johnson, and P. M. Meyers (2025-08)
Rapid parameter estimation for pulsar-timing-array datasets with variational inference and normalizing flows.
Phys. Rev. Lett. 135, pp. 071401.
External Links: [Document](https://dx.doi.org/10.1103/p3f7-rbmv),
[Link](https://link.aps.org/doi/10.1103/p3f7-rbmv)
Cited by: [§I](#S1.p4.1).
- [92]
M. Vallisneri and R. van Haasteren (2017-05)
Taming outliers in pulsar-timing data sets with hierarchical likelihoods and Hamiltonian sampling.
Monthly Notices of the Royal Astronomical Society 466 (4), pp. 4954–4959.
External Links: ISSN 0035-8711,
[Document](https://dx.doi.org/10.1093/mnras/stx069),
[Link](https://doi.org/10.1093/mnras/stx069)
Cited by: [§I](#S1.p8.1).
- [93]
S. Valtolina and R. van Haasteren (2025-08)
Regularizing the pulsar timing array likelihood: a path toward fourier space.
Phys. Rev. D 112, pp. 043046.
External Links: [Document](https://dx.doi.org/10.1103/s3gy-km61),
[Link](https://link.aps.org/doi/10.1103/s3gy-km61)
Cited by: [§V.1](#S5.SS1.p2.1).
- [94]
R. van Haasteren, Y. Levin, P. McDonald, and T. Lu (2009-05)
On measuring the gravitational-wave background using Pulsar Timing Arrays.
\mnras 395 (2), pp. 1005–1014.
External Links: [Document](https://dx.doi.org/10.1111/j.1365-2966.2009.14590.x),
0809.0791
Cited by: [§IV.2](#S4.SS2.p2.2).
- [95]
R. van Haasteren and Y. Levin (2013)
Understanding and analysing time-correlated stochastic signals in pulsar timing.
Mon. Not. Roy. Astron. Soc. 428, pp. 1147.
External Links: 1202.5932,
[Document](https://dx.doi.org/10.1093/mnras/sts097)
Cited by: [§IV.2](#S4.SS2.p2.2).
- [96]
R. van Haasteren and M. Vallisneri (2014-11)
New advances in the gaussian-process approach to pulsar-timing data analysis.
Phys. Rev. D 90, pp. 104012.
External Links: [Document](https://dx.doi.org/10.1103/PhysRevD.90.104012),
[Link](https://link.aps.org/doi/10.1103/PhysRevD.90.104012)
Cited by: [§I](#S1.p7.1),
[§IV.1](#S4.SS1.p3.1),
[§IV.4](#S4.SS4.p1.1).
- [97]
R. van Haasteren (2012-11)
Accelerating pulsar timing data analysis.
Monthly Notices of the Royal Astronomical Society 429 (1), pp. 55–62.
External Links: ISSN 0035-8711,
[Link](http://dx.doi.org/10.1093/mnras/sts308),
[Document](https://dx.doi.org/10.1093/mnras/sts308)
Cited by: [§IV.2](#S4.SS2.p2.2).
- [98]
R. van Haasteren (2024-07)
Pulsar timing arrays require hierarchical models.
The Astrophysical Journal Supplement Series 273 (2), pp. 23.
External Links: [Document](https://dx.doi.org/10.3847/1538-4365/ad530f),
[Link](https://doi.org/10.3847/1538-4365/ad530f)
Cited by: [§II](#S2.p2.1),
[§IV.1](#S4.SS1.p5.5).
- [99]
S. Vitale, D. Gerosa, W. M. Farr, and S. R. Taylor (2020)
Inferring the properties of a population of compact binaries in presence of selection effects.
In Handbook of Gravitational Wave Astronomy, C. Bambi, S. Katsanevas, and K. D. Kokkotas (Eds.),
pp. 1–60.
External Links: ISBN 978-981-15-4702-7,
[Document](https://dx.doi.org/10.1007/978-981-15-4702-7%5F45-1),
[Link](https://doi.org/10.1007/978-981-15-4702-7_45-1)
Cited by: [§II](#S2.p2.1).
- [100]
M. A. Woodbury (1950)
Inverting modified matrices.
In Memorandum Rept. 42, Statistical Research Group,
pp. 4.
Cited by: [§IV.2](#S4.SS2.p5.3).
- [101]
J. S. B. Wyithe and A. Loeb (2003-06)
Low-frequency gravitational waves from massive black hole binaries: predictions for lisa and pulsar timing arrays.
The Astrophysical Journal 590 (2), pp. 691.
External Links: [Document](https://dx.doi.org/10.1086/375187),
[Link](https://doi.org/10.1086/375187)
Cited by: [§I](#S1.p2.1),
[§III.3](#S3.SS3.p1.2),
[§IV.1](#S4.SS1.p3.1).
- [102]
H. Xu et al. (2023-07)
Searching for the Nano-Hertz Stochastic Gravitational Wave Background with the Chinese Pulsar Timing Array Data Release I.
Research in Astronomy and Astrophysics 23 (7), pp. 075024.
External Links: [Document](https://dx.doi.org/10.1088/1674-4527/acdfa5),
2306.16216
Cited by: [§I](#S1.p1.1).
- [103]
X. Xue, Z. Pan, and L. Dai (2025)
Non-gaussian statistics of nanohertz stochastic gravitational waves.
External Links: 2409.19516,
[Link](https://arxiv.org/abs/2409.19516)
Cited by: [Appendix D](#A4.p1.1),
[§IV.1](#S4.SS1.p4.1).