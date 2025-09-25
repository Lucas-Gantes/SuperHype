# SuperHype: Hypergraph generation via Graph-Superposition Decomposition

Hypergraphs are graph generalizations with key applications in domains such as
healthcare, where strict data privacy requirements apply, or bioinformatics, where
testing new compounds is costly. However, research into hypergraph synthesis is
limited, and state-of-the-art approaches yield limited generation quality in terms
of overall structural patterns and graph-level validity. This is caused by the hy-
pergraph’s combinatorial structure, which is composed of a number of possible
hyperedges that is factorial in the number of nodes. In fact, current solutions
rely on diffusion models denoising graph projections, which are exact but ineffi-
cient, or lightweight but approximate. To address such shortcomings, we intro-
duce SuperHype, the first hypergraph diffusion model with tractable and exact
modeling. To address the complexity of hypergraph representation, we introduce
the graph-superposition: a novel representation that embeds a hypergraph into a
multilayer graph. This enables a tractable representation while retaining its exact-
ness. To generate new samples from such representations, we introduce a Graph-
Superposition Transformer that treats the superposition as an interconnected se-
quence of layers. We optimize the model architecture to learn low-level patterns
within individual graphs in the superposition and high-level patterns between the
different graphs of the same superposition. Moreover, we enhance the model’s
performance with hypergraph-specific auxiliary features and triplet aggregation
of 2-hop node interactions. Our evaluation on five datasets shows that SuperHype
generally reproduces local and global connectivity patterns with superior fidelity
to state-of-the-art baselines.
