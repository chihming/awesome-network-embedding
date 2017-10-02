# awesome-network-embedding
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Gitter chat for developers at https://gitter.im/dmlc/xgboost](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/awesome-network-embedding/Lobby)

Also called network representation learning, graph embedding, knowledge embedding, etc.

The task is to learn the representations of the vertices from a given network.

<img src="http://cherry.cs.nccu.edu.tw/~g10018/portfolio/images/NE.png" width="480">

# Paper References with the implementation(s)

**StarSpace**

[StarSpace: Embed All The Things!](https://arxiv.org/pdf/1709.03856), arxiv'17

[[code]](https://github.com/facebookresearch/Starspace)

**ComE**

Learning Community Embedding with Community Detection and Node Embedding on Graphs, CIKM'17

[[Python]](https://github.com/andompesta/ComE)

**GraphSAGE**

Inductive Representation Learning on Large Graphs, NIPS'17

[[arxiv]](https://arxiv.org/abs/1706.02216) [[Python]](https://github.com/williamleif/GraphSAGE)

**ICE**

[ICE: Item Concept Embedding via Textual Information](http://dl.acm.org/citation.cfm?id=3080807), SIGIR'17

[[demo]](https://cnclabs.github.io/ICE/) [[code]](https://github.com/cnclabs/ICE)

**struc2vec**

struc2vec: Learning Node Representations from Structural Identity, KDD'17

[[arxiv]](https://arxiv.org/abs/1704.03165) [[Python]](https://github.com/leoribeiro/struc2vec)

**metapath2vec**

metapath2vec: Scalable Representation Learning for Heterogeneous Networks, KDD'17

[[paper]](https://www3.nd.edu/~dial/publications/dong2017metapath2vec.pdf) [[project website]](https://ericdongyx.github.io/metapath2vec/m2v.html)

**GCN**

Semi-Supervised Classification with Graph Convolutional Networks, ICLR'17

[[arxiv]](https://arxiv.org/abs/1609.02907)  [[Python Tensorflow]](https://github.com/tkipf/gcn)

**GAE**

Variational Graph Auto-Encoders, arxiv

[[arxiv]](https://arxiv.org/abs/1611.07308) [[Python Tensorflow]](https://github.com/tkipf/gae)


**CANE**

CANE: Context-Aware Network Embedding for Relation Modeling, ACL'17

[[paper]](http://www.thunlp.org/~tcc/publications/acl2017_cane.pdf) [[Python]](https://github.com/thunlp/cane)

**TransNet**

TransNet: Translation-Based Network Representation Learning for Social Relation Extraction, IJCAI'17

[[Python Tensorflow]](https://github.com/thunlp/TransNet)

**ConvE**

[Convolutional 2D Knowledge Graph Embeddings](https://arxiv.org/pdf/1707.01476v2.pdf), arxiv

[[source]](https://github.com/TimDettmers/ConvE)

**node2vec**

[node2vec: Scalable Feature Learning for Networks](http://dl.acm.org/citation.cfm?id=2939672.2939754), KDD'16

[[arxiv]](https://arxiv.org/abs/1607.00653) [[Python]](https://github.com/aditya-grover/node2vec)
[[Python-2]](https://github.com/apple2373/node2vec)

**DNGR**

[Deep Neural Networks for Learning Graph Representations](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12423), AAAI'16

[[Matlab]](https://github.com/ShelsonCao/DNGR) [[Python Keras]](https://github.com/MdAsifKhan/DNGR-Keras)

**HolE**

[Holographic Embeddings of Knowledge Graphs](http://dl.acm.org/citation.cfm?id=3016172), AAAI'16

[[Python-sklearn]](https://github.com/mnick/holographic-embeddings) [[Python-sklearn2]](https://github.com/mnick/scikit-kge)

**ComplEx**

[Complex Embeddings for Simple Link Prediction](http://dl.acm.org/citation.cfm?id=3045609), ICML'16

[[arxiv]](https://arxiv.org/abs/1606.06357) [[Python]](https://github.com/ttrouill/complex)

**MMDW**

Max-Margin DeepWalk: Discriminative Learning of Network Representation, IJCAI'16

[[paper]](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/ijcai2016_mmdw.pdf)  [[Java]](https://github.com/thunlp/MMDW)

**planetoid**

Revisiting Semi-supervised Learning with Graph Embeddings, ICML'16

[[arxiv]](https://arxiv.org/abs/1603.08861) [[Python]](https://github.com/kimiyoung/planetoid)

**PowerWalk**

[PowerWalk: Scalable Personalized PageRank via Random Walks with Vertex-Centric Decomposition](http://dl.acm.org/citation.cfm?id=2983713), CIKM'16

[[code]](https://github.com/lqhl/PowerWalk)

**LINE**

[LINE: Large-scale information network embedding](http://dl.acm.org/citation.cfm?id=2741093), WWW'15

[[arxiv]](https://arxiv.org/abs/1503.03578) [[C++]](https://github.com/tangjianpku/LINE)

**PTE**

[PTE: Predictive Text Embedding through Large-scale Heterogeneous Text Networks](http://dl.acm.org/citation.cfm?id=2783307), KDD'15

[[C++]](https://github.com/mnqu/PTE)

**GraRep**

[Grarep: Learning graph representations with global structural information](http://dl.acm.org/citation.cfm?id=2806512), CIKM'15

[[Matlab]](https://github.com/ShelsonCao/GraRep)

**KB2E**

[Learning Entity and Relation Embeddings for Knowledge Graph Completion](http://dl.acm.org/citation.cfm?id=2886624), AAAI'15

[[paper]](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/aaai2015_transr.pdf) [[C++]](https://github.com/thunlp/KB2E)  [[faster version]](https://github.com/thunlp/Fast-TransX)

**TADW**

[Network Representation Learning with Rich Text Information](http://dl.acm.org/citation.cfm?id=2832542), IJCAI'15

[[paper]](https://www.ijcai.org/Proceedings/15/Papers/299.pdf) [[Matlab]](https://github.com/thunlp/tadw)

**DeepWalk**

[DeepWalk: Online Learning of Social Representations](http://dl.acm.org/citation.cfm?id=2623732), KDD'14

[[arxiv]](https://arxiv.org/abs/1403.6652) [[Python]](https://github.com/phanein/deepwalk)



**GEM**

Graph Embedding Techniques, Applications, and Performance: A Survey

[[arxiv]](https://arxiv.org/abs/1705.02801) [[MIX]](https://github.com/palash1992/GEM)

# Paper References

**CONE**, [CONE: Community Oriented Network Embedding](https://arxiv.org/abs/1709.01554), arxiv'17

**LANE**, 
[Label Informed Attributed Network Embedding](http://dl.acm.org/citation.cfm?id=3018667), WSDM'17

**Graph2Gauss**,
[Deep Gaussian Embedding of Attributed Graphs: Unsupervised Inductive Learning via Ranking](https://arxiv.org/abs/1707.03815), arxiv
[[Bonus Animation]](https://twitter.com/abojchevski/status/885502050133585925)

[Scalable Graph Embedding for Asymmetric Proximity](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14696), AAAI'17

[Structural Deep Network Embedding](http://dl.acm.org/citation.cfm?id=2939753), KDD'16

[Query-based Music Recommendations via Preference Embedding](http://dl.acm.org/citation.cfm?id=2959169), RecSys'16

[Tri-party deep network representation](http://dl.acm.org/citation.cfm?id=3060886), IJCAI'16

[Heterogeneous Network Embedding via Deep Architectures](http://dl.acm.org/citation.cfm?id=2783296), KDD'15

[Neural Word Embedding As Implicit Matrix Factorization](http://dl.acm.org/citation.cfm?id=2969070), NIPS'14

[Distributed large-scale natural graph factorization](http://dl.acm.org/citation.cfm?id=2488393), WWW'13

[From Node Embedding To Community Embedding](https://arxiv.org/abs/1610.09950), arxiv

[Walklets: Multiscale Graph Embeddings for Interpretable Network Classification](https://arxiv.org/abs/1605.02115), arxiv

[Comprehend DeepWalk as Matrix Factorization](https://arxiv.org/abs/1501.00358), arxiv

[Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039), arXiv 2017




# Related List

[Must-read papers on NRL/NE.](https://github.com/thunlp/NRLPapers)

[Network Embedding Resources](https://github.com/nate-russell/Network-Embedding-Resources)

[awesome-embedding-models](https://github.com/Hironsan/awesome-embedding-models)

[2vec-type embedding models](https://github.com/MaxwellRebo/awesome-2vec)

# Related Project

**Stanford Network Analysis Project** [website](http://snap.stanford.edu/)

**proNet-core** [github](https://github.com/chihming/proNet-core)

