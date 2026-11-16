# References for the simplified COT correction

This contribution implements a simplified conditional OT regulariser through
fixed predicted-mass groups and a finite sliced-Wasserstein loss. Its conditional
structure follows the idea of matching distributions under shared covariates;
its numerical loss uses projection, sorting and squared differences. The group
boundaries, correction network and hyperparameters are project choices.

## Conditional transport

**Esteban G. Tabak, Giulio Trigila and Wenjun Zhao (2021). _Data driven conditional
optimal transport_. Machine Learning 110, 3135–3155.**

[Journal article](https://doi.org/10.1007/s10994-021-06060-0) ·
[Open manuscript](https://arxiv.org/abs/1910.11422)

This paper develops transport between conditional distributions sharing a set
of covariates. It provides the conditional-transport context for our simplified
choice of a discrete mass-group condition. Its conditional-map construction is
more general than the bounded regression correction used here.

## Projected distribution loss

**Ishan Deshpande, Ziyu Zhang and Alexander G. Schwing (2018). _Generative Modeling
Using the Sliced Wasserstein Distance_. CVPR, 3483–3491.**

[Publisher PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Deshpande_Generative_Modeling_Using_CVPR_2018_paper.pdf) ·
[arXiv](https://arxiv.org/abs/1803.11188)

The most direct algorithmic reference for using random one-dimensional
projections and sorted empirical values as a distribution-learning objective.
Here that construction is applied inside each fixed mass group and combined
with the paired regression loss.

## Projection estimation

**Mark Rowland, Jiri Hron, Yunhao Tang, Krzysztof Choromanski, Tamas Sarlos and
Adrian Weller (2019). _Orthogonal Estimation of Wasserstein Distances_.
AISTATS, PMLR 89, 186–195.**

[Proceedings and BibTeX](https://proceedings.mlr.press/v89/rowland19a.html) ·
[PDF](https://proceedings.mlr.press/v89/rowland19a/rowland19a.pdf)

This paper discusses efficient estimation using one-dimensional sorting,
projection sampling and related transport distances. Our implementation uses
32 fixed normalised Gaussian directions; orthogonal sampling is an alternative
studied in the paper. More directions refine the sliced-distance estimate,
whose limit is distinct from the unsliced multivariate Wasserstein distance.

## General mathematical background

**Gabriel Peyre and Marco Cuturi (2019). _Computational Optimal Transport_.
Foundations and Trends in Machine Learning 11(5–6), 355–607.**

[Open monograph](https://arxiv.org/abs/1803.00567)

Background on transport couplings, maps, Wasserstein distances and solvers,
including the one-dimensional rank-matching construction underlying the loss.

## How these references fit together

For a short description, cite Tabak et al. for conditional transport and
Deshpande et al. for the sliced loss. Rowland et al. supports discussion of
projection estimation; Peyre and Cuturi supplies the general OT definitions.
The [method note](conditional_ot.md) compares this implementation with a fuller
conditional-transport formulation, including the conditioning, distance and
model-capacity differences. The simplification preserves the full training
dataset. [BibTeX entries](conditional_ot_references.bib) are supplied alongside
this note.
