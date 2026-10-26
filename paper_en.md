# From Post Text to User Network: Generating a Temporal Social Network by Transferring tNodeEmbed

**Angelantonio Fedele Murolo**

*Affiliation: none.*

---

## Abstract

Building the link structure of a population for which only the text it produces is available — with no observed links and no dates — requires learning the edge-formation mechanism from a different network and applying it by inference. This work studies the problem on two datasets: a real social network with 25,714 users, 824,045 directed edges and 173,475 posts spread over 9 snapshots (2022 – July 2025), and a synthetic dataset of 10,000 users with 100,456 posts, with no edges and no dates, which we partition into 9 time steps. The only information shared by the two datasets is the text of the posts. The temporal model is tNodeEmbed in its node2vec-free variant, initialised with content representations: per-snapshot representations, a chain of Procrustes alignments and an LSTM yield one embedding per user, and a pairwise head estimates the probability of a link. Because the restriction of the original evaluation protocol admits no test edge on the real network, evaluation uses the whole transition with matched and random negatives. From the learned coefficients we derive one generator fitted on the real data, by sampling with feedback and with no imposed counts: an **arrival-event** model, in which the unit is the entering user. The delivered network is generated on the **progressive-arrival** partition — the one faithful to the fact that in the real data 58% of users enter during the observed period — and a second network on the "all present" partition is kept as a control. The official network reproduces the real one within the right order of magnitude for reciprocity (0.3224 vs 0.4177), mean degree (27.45 vs 32.05), maximum degree (274 vs 1,290), transitivity (0.0212 vs 0.0337), clustering (0.042 vs 0.080) and diameter (6 vs 6), while remaining somewhat "flatter" than the real one. The residual is attributed by the measurement to the **shape** of the tail and to cohesion, not to the hubs: the control reaches **97%** of what the scaling law estimated on the real network (exponent 1.445 over its nine snapshots) predicts at 10,000 users, the official network **81%**, and at matched size the tail is steeper (−3.98 vs −1.69).

**Keywords:** temporal networks; node embedding; graph generation; homophily; transfer across domains; tNodeEmbed.

---

## 1. Introduction

Many questions about a social system require its interaction network, yet the available data often contain only the text that people produce: no observed links, no dates. When a second dataset holds the network but different text, a precise possibility opens up — **learn the edge-formation mechanism on the dataset that has the network and apply it, by inference, to the dataset that has none** — provided a channel of information is found that is present in both. This work addresses exactly that problem: given a corpus of posts from 10,000 users with no edges at all, build their social network and study its structure and evolution over time, using the text of the posts as the only information shared with the reference dataset.

The problem has three difficulties. The first is **the initialisation of never-seen nodes**: classical network-embedding methods such as node2vec cannot represent a user who has no links yet, whereas in the real network we study no new edge connects two already-present users. The second is **transfer across populations of different size**: the dataset that carries the structure has 25,714 users, the one to be populated 10,000, so degree-related quantities must be expressed in relative form. The third is **evaluation**: in the dataset without edges there are no true links to predict, and a generative network model is not judged by a ranking measure.

The contribution is fourfold. (i) We adapt **tNodeEmbed** to our setting by removing node2vec and using post-content representations as the starting point: this is a possibility foreseen by the method itself, which admits any static embedding as initialisation. (ii) From the coefficients learned on the real dataset we derive a **fitted generator** by sampling with feedback and with no imposed counts, in which the unit is not the pair but the **arrival event**, i.e. the entry of a user, who brings a group of edges with them. (iii) We set up a **controlled evaluation** based on network statistics and distributions, on the content control with authentic text, text permuted across users and zeroed content, and on the decomposition of homophily with degree controls. (iv) We report negative results as results: content does not move the network, and encoder fine-tuning and trainable node features make things worse.

The summary of the results is as follows. The arrival-event generator reproduces the real network within the right order of magnitude: reciprocity 0.3224 (real 0.4177), mean degree 27.45 (real 32.05), maximum degree 274 (real 1,290), transitivity 0.0212 (real 0.0337), clustering 0.042 (real 0.080), diameter 6 as in the real network. The residual gaps are contained and all of the same sign; their cause is not an underestimation of the hubs — the control's maximum degree is at 97% of what the scaling law predicts — but the **shape** of the tail and the **cohesion**. On the content side the results are sharp and partly counter-intuitive: text knows the user (from content alone, political leaning is predicted with AUC 0.827, profession with 0.867, age band with 0.690) but **not their links**; the content homophily of the real network, 2.13, is largely network position rather than a choice of similarity.

---

## 2. Related Work

**Node embeddings on temporal graphs.** tNodeEmbed (Singer, Guy and Radinsky, IJCAI 2019) is the reference of this work: it builds one representation per snapshot, aligns the representations of consecutive snapshots and learns their evolution with a recurrent network, using a pairwise loss for the link-prediction task. The general setting of node embedding is that of node2vec (Grover and Leskovec, KDD 2016), which tNodeEmbed uses as a static initialisation; because node2vec requires observed links, it cannot be applied to never-seen users. The literature offers other families of temporal models: temporal walks on continuous-time networks (Nguyen et al., WWW 2018), self-attention over neighbourhoods and time (Sankar et al., WSDM 2020), graph convolutional networks whose parameters evolve over time (Pareja et al., AAAI 2020), recurrent architectures over transitions (Goyal et al., *Knowledge-Based Systems* 2020), and the triadic-closure process as an explicit mechanism (Zhou et al., AAAI 2018). The picture is surveyed by Kazemi et al. (*JMLR* 2020). Unlike these works, our problem is not predicting future edges in a known network, but **populating with edges a network in which no link is observed**, by transferring parameters learned elsewhere.

**Graph generation.** Classical models — Barabási–Albert (1999) for preferential attachment, Watts–Strogatz (1998) for the small world, Kronecker graphs (Leskovec et al., *JMLR* 2010) for recursive structure — generate realistic topologies but are not fitted to the mechanism of a specific dataset and use no content. Recent neural generative models, GraphRNN (You et al., ICML 2018) and NetGAN (Bojchevski et al., ICML 2018), learn to reproduce graph distributions and are evaluated by comparing statistics and distributions, which is the same evaluation principle we adopt; they are not, however, designed to transfer a structure from one population to another that has no edges. The evolution laws of real graphs (densification and shrinking diameters; Leskovec, Kleinberg and Faloutsos, KDD 2005) provide the empirical references against which we read the growth of our networks.

**Content representation.** The text representations we use belong to the family of sentence encoders trained for similarity (Sentence-BERT; Reimers and Gurevych, EMNLP 2019). The choice among them is guided by a benchmark independent of our task (MTEB; Muennighoff et al., EACL 2023) and by the availability of compact models (BGE; Xiao et al., SIGIR 2024).

**Network properties and measures.** Homophily (McPherson, Smith-Lovin and Cook, 2001) and triadic closure (Granovetter, 1973) are the phenomena we measure; community structure is identified with the Louvain algorithm (Blondel et al., 2008). The recurrent cell that aggregates the temporal sequence is the LSTM (Hochreiter and Schmidhuber, 1997) and the alignment between coordinate systems is generalised Procrustes analysis (Gower, 1975).

**The gap we occupy.** None of these works addresses the combination that interests us: learning the edge-formation mechanism from a real network **and applying it by generation to a population that has only text**, while showing with explicit controls how much of the result comes from content and how much from structure.

---

## 3. Data

The two datasets were **provided for the project by the doctoral student who supervised it**. The first, which we call **real**, contains users, their posts and the links between them, spread over **9 snapshots** from January 2022 to July 2025: 25,714 users, 824,045 **directed** edges, 173,475 cumulative posts (154,580 counted per period). The split of the periods is declared: 6 training snapshots, 1 validation, 2 test. For each user, the real dataset contains **only post-level fields** (the text, the language, counts of reactions and replies, the timestamp): **there are no profile attributes**. The second dataset, which we call **synthetic**, contains 10,000 users with **100,456 usable posts** (out of 100,474 rows, 18 discarded on reading because their field count differed from the header), **no edges and no dates**, plus 11 user profile attributes (including political leaning, profession and age band) which do not exist in the real dataset.

Because the model is trained on the real dataset and applied to the synthetic one, only what exists on both sides can enter as a feature: the **only shared column**, beyond identifiers, is the **text of the posts** (`content`). The synthetic attributes are never used as input and serve **only in evaluation**. The real dataset is not modified in any way: the synthetic network is a new artefact, and its links are exclusively **synthetic-to-synthetic**.

**Table 1 — The two datasets.**

| | real | synthetic |
|---|---|---|
| users | 25,714 | 10,000 |
| posts | 173,475 cumulative / 154,580 per period | 100,456 usable |
| edges (directed) | 824,045 | none |
| time | 9 snapshots, 2022 → 2025-07; split 6/1/2 | none: 4 batches in random order |
| information per user | only the post text | 11 profile attributes + text |
| shared column | **`content`** | **`content`** |

*Source: `results/structure_real_snapshots.csv`, `results/native_batches.csv`, `results/encoder_comparison.csv`.*

**The time convention in the synthetic data.** The synthetic dataset has no dates; the only available order is the dense post identifier, which we assume to reflect the generation order. We partition it into **9 steps**, equal to the number of real snapshots, using two rules measured on the real data. First, the **arrival curve**, i.e. the share of users entering at each step when users are ordered by their first post: **0.418 · 0.142 · 0.110 · 0.072 · 0.078 · 0.051 · 0.010 · 0.077 · 0.042**. Second, the **activity shape**, i.e. when a user writes: each user's posts are distributed over the steps of that user's lifetime by sampling the real distribution of posts per active user per period (**mean 3.7, median 2, p90 9**), with bursts placed **at random**, so that both full and silent periods appear. We do **not** use the real aggregate posts-per-snapshot curve, because the real snapshots have heterogeneous lengths (from one year to a few days) and those counts measure "activity × window length". What remains a declared choice on our side is the arrival order (inferred from the identifiers) and the order within each step.

*Source: arrival curve and activity distribution measured on the real data by the partition builder (`src/build_synthetic_snapshots.py`).*

**The two partitions.** The **progressive-arrival** partition applies the arrival curve; the "all present" one makes every user exist from the first step and keeps only the activity shape. In the real data 41.8% of the final users are already linked at the first snapshot and the remaining 58% enter during the observed period: the first partition is therefore the more faithful, and the second is kept as a **control**.

---

## 4. Method

![Pipeline schematic: from the real and synthetic corpora to the generated network](results/figures/paper_pipeline.png)

**Figure 1** — Pipeline schematic. The temporal model is trained on the real dataset; the synthetic dataset is partitioned into 9 steps using the arrival curve and activity shape measured on the real data; the network is generated step by step by sampling with feedback. File: `results/figures/paper_pipeline.png`.

### 4.1 Content representation

The only textual input is the text of the posts. Each post is encoded **individually** by a sentence encoder, and a user's vectors are **averaged** into one representation per user and snapshot. We chose the encoder with a comparison protocol rather than a priori: the whole temporal pipeline is trained on the real data with each encoder, and the AUCs on the validation and test transitions are compared. The deciding column is the one with **matched negatives** (see §4.2).

**Table 2 — Encoder selection on the real dataset** (AUC on validation / test1 / test2; mean = val+test; period text).

| encoder | AUC, matched negatives (val / test1 / test2) | mean | AUC, random negatives | edges satisfying the original protocol restriction |
|---|---|---|---|---|
| tf-idf + SVD | 0.489 / 0.521 / 0.505 | 0.505 | 0.986 / 0.931 / 0.941 | 0 / 0 / 0 |
| **all-MiniLM-L6-v2** | 0.571 / 0.521 / 0.510 | **0.534** | 0.995 / 0.954 / 0.961 | 0 / 0 / 0 |
| BGE-small | 0.541 / 0.517 / 0.519 | 0.526 | 0.996 / 0.958 / 0.967 | 0 / 0 / 0 |
| GTE-small | 0.529 / 0.514 / 0.507 | 0.516 | 0.996 / 0.956 / 0.963 | 0 / 0 / 0 |
| BGE-base | 0.540 / 0.517 / 0.535 | 0.530 | 0.996 / 0.960 / 0.968 | 0 / 0 / 0 |
| GTE-base | 0.543 / 0.523 / 0.531 | 0.532 | 0.996 / 0.959 / 0.969 | 0 / 0 / 0 |

*Source: `results/encoder_comparison.csv`. The last column is the number of test edges admitted by the original protocol restriction in each window.*

Encoding **per post** improves every encoder over concatenating a user's posts into a single string, which the encoder truncated at its token limit: the mean val+test rises from 0.534 to **0.549** for MiniLM, from 0.530 to **0.540** for BGE-base, from 0.505 to **0.524** for tf-idf. The pipeline adopts **all-MiniLM-L6-v2** with per-post encoding and averaged vectors; the encoder is trained only on training content.

*Source: `results/encoder_comparison.csv`, rows `*|mean_of_posts|frozen`.*

**How narrow the comparison is.** The differences among the top three encoders (0.534, 0.532, 0.530) lie **within** run-to-run variability: re-running the same encoder yields values that move by about 0.005 between processes, because the iteration order of edge sets is not fixed in Python and changes negative sampling. The choice should therefore be read as a **tie among three encoders** (MiniLM, GTE-base, BGE-base) clearly above tf-idf (0.505), not as a ranking. This is a declared limitation, not a detail: the selection metric lacks the resolution to separate the first from the third.

### 4.2 Temporal model

The temporal model is **tNodeEmbed without node2vec**. For each snapshot `t` we have a users × dimension matrix built from content (§4.1); a **chain of orthogonal Procrustes alignments** between consecutive snapshots brings the representations into a common coordinate system, estimated on the nodes present in both snapshots and applied to all, including new ones; an **LSTM** reads the sequence of the 9 steps of each user and compresses it into a temporal embedding; a **pairwise head**, fed by `[h_u, h_v, h_u * h_v]` (the embedding of the first user, that of the second and their element-wise product), produces the link score, optimised with binary cross-entropy on the training transitions. The test set is not touched until evaluation.

Removing node2vec is not an omission but a data-driven choice: node2vec initialises itself by observing links, so it cannot represent a user appearing for the first time, and on this dataset **no new edge connects two already-present users**. Applying the **restriction of the original evaluation protocol** (test limited to pairs between already-observed users) yields **0 test edges in every transition** (Table 2, last column): the restriction, correct in general, is empty here. Evaluation therefore uses the **whole transition** (all new edges of the period), with two kinds of negatives: **random** and **matched**, where each true pair is coupled with a false pair sharing the same first user and a destination with a similar arrival history and degree. The matched mode is the deciding column, because it prevents the model from "cheating" by recognising the new user from the sequence alone.

### 4.3 Network generation

A network does not automatically emerge from the temporal model's scores: decisions are needed on which pairs become edges. We make them by **fitting the mechanism on the real dataset** and applying it by generation to the synthetic one.

**Unit: the arrival event.** The starting point is a measurement. In the real data **no new edge connects two already-present users** (share "pre-existing → pre-existing" 0.0000% in every transition): each edge therefore belongs to the event of its new endpoint. An event carries on average **35–111 edges** depending on the transition (median 6–32), with a maximum of **1,202**, i.e. the same order as the real maximum degree: the hub is born from a *single* arrival, not from the sum of independent decisions. **51–63%** of the event's edges are outgoing and about **40%** of the pairs are reciprocated within the event.

**The generator.** Three things are estimated on the real data and generation proceeds **arrival by arrival**: how many edges an arrival carries (`k`, sampled from a table fitted on the real data and rescaled on the population); the direction of each edge (the outgoing share of the step); the target, with a logistic regression on seven variables — the content score of the pairwise head, `log(1+degree)`, degree percentile, "the counterpart is new", "the counterpart wrote in the period", direction, co-occurrence with the partners already chosen in the event. Reciprocated pairs are drawn among the outgoing edges already chosen, with a real table by direction and degree class. The state of the network (degrees, percentiles, novelty, activity, neighbourhood) is recomputed **at every step**; no threshold and no count is imposed.

*Source: the arrival quantities are in `results/arrival_events.csv`; the coefficients in `results/coefficients_synthetic_network.csv`; the final counts in `results/network_report.csv`.*

**The delivered partition.** Every figure in this article refers to the network generated on the **progressive-arrival** partition. The same pipeline run on the "all present" partition is the **control** and is reported alongside, because the choice between the two is conceptual rather than metric (§5.2).

### 4.4 Evaluation protocol and hierarchy of metrics

A **generated network is not evaluated with an AUC**: in the synthetic dataset there are no true edges to predict, so a ranking measure has no object. We adopt an explicit hierarchy.

1. **Primary — per-snapshot network statistics and distributions**, compared with the real network on the final snapshot: degree distribution (mean, maximum, CCDF), reciprocity, transitivity and average clustering, modularity (Louvain), diameter, growth composition.
2. **Secondary — content**, with the nature of the link: content homophily with degree controls, and the A/B/C control (authentic text, text permuted across users, zeroed content).
3. **Diagnostic, on the real data only** — AUC and average precision with matched and random negatives, used to choose the encoder and to fit the model. The AUC ablation on the real data is **saturated** (0.9818 → 0.9816) and by itself does not measure the contribution of text: the evidence is the comparison between networks (point 2).
4. **Not valid** — AUC as a quality measure of a generated network.

---

## 5. Results

### 5.1 The central comparison

**Table 3 — Final snapshot: real network, official network and control.**

| network | edges | nodes | density | mean degree | max degree | reciprocity | modularity | transitivity | clustering | diameter | homophily |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **real** | 824,045 | 25,714 | 0.00125 | 32.05 | 1,290 | 0.4177 | 0.275 | 0.0337 | 0.080 | 6 | 2.13 |
| **official** (progressive) | 271,646 | 9,897 | 0.00277 | 27.45 | 274 | 0.3224 | 0.1074 | 0.0212 | 0.0420 | 6 | 1.0009 |
| control (all present) | 280,298 | 9,901 | 0.00286 | 28.31 | 328 | 0.3035 | 0.1030 | 0.0209 | 0.0435 | 6 | 0.9993 |

*Source: `results/structure_*.csv`, `results/structure_heavy_*.csv`, `results/content_*.csv`; distances in `results/scorecard.csv`.*

The official network brings **within the right order of magnitude** reciprocity (0.3224 vs 0.4177), the degree scale (27.45 vs 32.05), transitivity (0.0212 vs 0.0337), clustering (0.042 vs 0.080) and diameter (6 vs 6). The control does slightly better on mean degree, maximum degree and clustering, and slightly worse on density, reciprocity, modularity and transitivity: over the seven comparable metrics it wins 4 to 3, and the relative distance from the real network, defined as `|value − real| / |real|` summed over the seven, is **3.84** for the official network and **3.89** for the control. Changing the normalisation reverses the order: the two partitions are **equivalent** on this plane.

![Comparing real, official and control: structure](results/figures/compare_structure.png)

**Figure 2** — Structure per snapshot: real, official and control (density, reciprocity, modularity). File: `results/figures/compare_structure.png`.

![Comparing real, official and control: growth](results/figures/compare_growth.png)

**Figure 3** — Nodes and edges per snapshot. File: `results/figures/compare_growth.png`.

![Comparing real, official and control: degree tail](results/figures/compare_degree_ccdf.png)

**Figure 4** — Out-degree distribution (CCDF), final snapshot. File: `results/figures/compare_degree_ccdf.png`.

### 5.2 Why the progressive partition, and what it costs

In the real data **41.8%** of the final users are already linked at the first snapshot and the remaining **58%** enter during the observed period. A network that is born must have arrivals: the progressive partition reproduces the arrival curve by construction, the "all present" one does not. The cost is declared and measured.

**Table 4 — The two partitions, final snapshot.**

| metric | real | official | control | closest |
|---|---:|---:|---:|---|
| density | 0.001246 | 0.002774 | 0.002860 | official |
| mean degree | 32.05 | 27.45 | 28.31 | control |
| maximum degree | 1,290 | 274 | 328 | control |
| reciprocity | 0.4177 | 0.3224 | 0.3035 | official |
| modularity | 0.2751 | 0.1074 | 0.1030 | official |
| transitivity | 0.0337 | 0.0212 | 0.0209 | official |
| clustering | 0.0804 | 0.0420 | 0.0435 | control |
| diameter | 6 | 6 | 6 | tie |
| tail slope (top decile) | −2.349 | −3.980 | −3.332 | control |
| edges between users arriving in the same step | 8.1% | 17.0% | 12.4% | control |

*Source: `results/scorecard.csv`, `results/tail_scaling_summary.csv`, `results/accretion_*.csv`.*

The most visible cost is on two rows: edges between co-arriving users rise from 12.4% to **17.0%** (real 8.1%), because with progressive entry there are more entering users available at each step and the positive weight on "the counterpart is new" spends more edges among peers; and the maximum degree drops to **274**, **81%** of what the real scaling law predicts at 10,000 users, against the control's 97%. The first step is also emptier than the real one (15.7% of nodes linked against 41.8%), because `k` is rescaled on the population present and at the first step that is only the entering cohort.

That last comparison must be read with care: the real 41.8% **is not a birth step**, it is a pre-existing stock (the first snapshot contains 10,755 users and 116,441 edges created before observation). Comparing it with the first generation of a network being born compares different objects, and it is not a criterion.

**An extrapolation to declare.** The law that rescales `k` is fitted on populations between 10,755 and 21,082 users; at the first step of the progressive partition `P ~ 4,100`, so it is an extrapolation below the fitting range. We have no real data below 10,000 users to validate it.

![Comparing the partitions: homophily](results/figures/compare_homophily.png)

**Figure 5** — Content homophily per snapshot: real, official and control. File: `results/figures/compare_homophily.png`.

### 5.3 The content control

Content is the only information that distinguishes the 10,000 synthetic users from one another, so its role must be measured, not postulated. With the **same model fitted on the real data and the same scenario**, we generate three networks that differ only in the textual channel on the synthetic side: **A** authentic text, **B** text permuted across users (the same text profiles reassigned to other people, so the text statistics stay identical), **C** content zeroed in the choice model.

**Table 5 — A/B/C control on the official network (progressive partition).**

| mode | edges | density | mean degree | max degree | reciprocity | modularity | transitivity | clustering | homophily |
|---|---|---|---|---|---|---|---|---|---|
| A — authentic text | 271,646 | 0.002774 | 27.45 | 274 | 0.3224 | 0.1074 | 0.0212 | 0.0420 | 1.0009 |
| B — permuted text | 261,680 | 0.002668 | 26.42 | 248 | 0.3229 | 0.1102 | 0.0207 | 0.0392 | 0.9995 |
| C — zeroed content | 271,640 | 0.002777 | 27.47 | 275 | 0.3227 | 0.1066 | 0.0213 | 0.0421 | 1.0010 |

*Source: `results/structure_synthetic_network{,_permuted,_off}.csv` and the corresponding `structure_heavy_*` and `content_*`. The two control networks are not part of the delivery: their figures live in those CSVs and are regenerated with `--content permuted` and `--content off`.*

The three networks are **indistinguishable** in their relational properties: volume varies by 3.7% and **without order** (the network with permuted text is closer to the zeroed one than to the one with authentic text), and reciprocity, transitivity, clustering, modularity and diameter coincide within the noise between trainings. The only visible effect of text is minimal: homophily goes from 1.0010 (zeroed content) to 1.0009 (authentic text), against the real 2.13. **Content does not move the network.**

This is consistent with the learned weights, on standardised variables: the content score weighs **+0.05** against **+1.22** for the novelty of the counterpart and **+1.02 / +0.91** for degree and percentile; "wrote in the period" weighs **−0.03** and co-occurrence **+0.19**. The model discriminates well on the validation transition (AUC 0.918), but it does so **with structure**: *who enters* and *how connected they are*, not *what they write*.

*Source: `results/coefficients_synthetic_network*.csv`, `results/network_report.csv`.*

### 5.4 Decomposing homophily

The number that seemed to contradict §5.3 is the content homophily of the real network: **2.13** on the final snapshot, i.e. the two endpoints of an edge are more than twice as similar as two randomly drawn users. The standard measure, however, compares every edge with two **arbitrary** users, whereas the endpoints of edges are not arbitrary users: in the real data the pre-existing endpoint has mean degree **403.6** against **61.7**, and the lexical similarity between two users **grows with degree** (0.079 in the lowest degree decile, 0.291 in the highest). If the comparison is made against **non-edges in the same position**, homophily changes face.

**Table 6 — Content homophily of the real network against structural controls (final snapshot).**

| comparison | ratio | AUC edges vs control |
|---|---|---|
| **random** pairs (standard measure) | 2.133 | 0.702 |
| non-edges with the **same degree** (broad classes) | 1.511 | 0.610 |
| non-edges with degree within **±10%** | 1.290 | 0.575 |
| non-edges with the same **text volume** | 1.554 | 0.648 |
| non-edges with similar degree and **text volume** | 1.367 | 0.585 |
| — internal similarity by degree decile: 1st decile 0.079 → 10th decile 0.291 | | |

*Source: `results/homophily_control_real_snapshots.csv`. The tightest controls (degree within ±10%) destabilise the sample (737,101 pairs), so 1.29 should be read as a lower bound.*

More than half of the gap disappears as soon as comparable users are compared: **the 2.13 is largely network position**, not a choice of similarity. On the official network the same protocol yields **1.001 / 1.000 / 0.980 / 1.001 / 1.014**: no content homophily at all. A **small lexical residual** remains in the real data (ratio 1.29; AUC 0.58), which lives in the lexical profiles used for this measure and not in the generator's channel (per-post, where the signal is null): the real 2.13 and the 1.00 of the generated networks **do not measure the same thing**.

*Source: `results/homophily_control_synthetic_network.csv`.*

### 5.5 Profiles

In the synthetic dataset the user attributes are known labels, so we can ask whether the text knows *who* a user is. We train a logistic regression on 80% of the users (stratified split by user, fixed seed) and measure the macro one-vs-rest AUC on the remaining 20%, comparing four representations.

**Table 7 — Predicting profile attributes (macro AUC, final snapshot, official network).**

| label (classes) | content only | content + graph | temporal embedding | embedding + graph |
|---|---|---|---|---|
| political leaning (7) | **0.827** | 0.826 | 0.619 | 0.626 |
| profession (41) | **0.867** | 0.865 | 0.587 | 0.577 |
| age band (8) | **0.690** | 0.683 | 0.514 | 0.505 |

*Source: `results/profiles_synthetic_network.csv`.*

**Content knows the user**, and adding the network changes nothing; the temporal embeddings, trained to predict *links*, lose the profile. This is **the mirror image of §5.3**: *who* a user is lies in the text, *whom they connect to* lies in the structure. The link, in fact, depends on state variables — the user's entry and their degree — which text does not contain. The task exists only on the synthetic data: the real dataset has no attributes.

![Profile classification](results/figures/compare_profiles.png)

**Figure 6** — Profile classification performance, official network and control. File: `results/figures/compare_profiles.png`.

### 5.6 Negative results

We present as results also the paths that did not work, because they delimit the contribution. Configurations and full metrics are in `archive/storico.md`.

- **Content does not carry the link.** The A/B/C control (§5.3) is indistinguishable; the AUC with matched negatives on the real data is around chance; the learned weight of content is an order of magnitude below that of structure.
- **Encoder fine-tuning fails.** Trained with a contrastive loss on real edges and with matched and temporal negatives, it worsens **the validation too**, besides both test windows: the val+test mean drops from 0.534 to **0.519** for MiniLM and from 0.530 to **0.523** for BGE-base. The supervision is too weak and the encoder adapts to the period it sees.
- **Trainable node features make things worse.** Making the content-initialised projection trainable lowers the val+test mean from 0.534 to **0.512** on the concatenated encoding: added capacity leads to overfitting. The full learned matrix of the paper (~70M parameters) does not fit the available memory.
- **A "mixed" negative sampling makes the network worse.** Replacing uniform negatives with a 30/30/40 mixture: transitivity 0.0115 vs 0.0209, clustering 0.0305 vs 0.043, edges between co-arriving users 18.4% vs 12.4%. The learned weights move a lot — novelty of the counterpart +0.61 vs +1.22, logarithmic degree −0.21 vs +1.02 — but reciprocity **does not move**: it is produced by the table of reciprocated pairs, not by the classifier.
- **"The post as an edge" does not work.** The text of a single post does not separate linked pairs from matched ones (AUC 0.515–0.517 against 0.5166 for the mean) and the posts of a reciprocated pair are not more similar than chance. The structural part of the idea cannot be verified: the dataset does not record which post is attached to each edge.
- **The AUC ablation is saturated and useless.** Removing the content score from the model moves the AUC on the real data from 0.9818 to 0.9816: a measure this high does not separate the two cases, which is why the evidence of §5.3 is the comparison between networks.

*Source: `results/encoder_comparison.csv`, `results/finetune_report.csv`; for the discarded configurations, `archive/storico.md`.*

---

## 6. Discussion

**What structure explains, and what content does not.** The measurements converge on a single reading. The mechanism that generates links in this system is **relational and sequential**: edges arise when a user enters, they point towards users more connected than average, they concentrate in a few large arrivals and are partly reciprocated immediately. The learned weights (novelty and degree dominant, content an order of magnitude smaller) and the A/B/C controls say the same thing. Text, by contrast, contains clearly *who* a user is — content predicts profile attributes with AUC up to 0.867 — but **not** *whom they connect to*: the channel that explains identity does not explain the relation. The second, less intuitive result is that the content homophily of the real network does not contradict this reading: 2.13 is largely network position (degree), and the residual attributable to similarity is small (1.29) and does not live in the channel used by the generator.

**How much transfer across networks of different size weighs.** To transfer a degree preference from 25,714 users to 10,000 we express degree as **percentiles**, and the mean degree of the pre-existing endpoint comes out at 153 in the official network against 437: it looked like an underestimation, but it is close to the **ratio between the populations** (0.35 against 0.385). The same holds for the maximum: the maximum degree grows with the population according to a law estimated **on the real network itself** (its nine snapshots: exponent **1.445**, R² 0.986), and at 10,000 users that law predicts **~339**; the control reaches **328** (97%), the official network **274** (81%). What the transfer does **not** reproduce is the **shape** of the tail — at matched size the top-decile slope is −3.98 (official) and −3.33 (control) against −1.69 for the real network — and the **cohesion** (transitivity 0.021 vs 0.034, clustering 0.042 vs 0.080). The volume rescaling is also a choice of target rather than an error: the configuration adopted makes the synthetic network carry the real's mature density, while the alternative dictated by the real's own law matches the density but halves transitivity and lowers the maximum to 0.48 of the prediction.

**Threats to validity.** *Construct validity.* The network statistics we use are proxies: modularity depends on the chosen community algorithm, and the standard homophily measure is confounded by degree, as shown in §5.4. This is why we always pair the statistics with controls (degree, text volume). *Internal validity.* Two assumptions remain ours: the arrival order in the synthetic data (inferred from the identifiers) and the order within a step; we turned them into control variables with two partitions, but they cannot be verified. Comparing them shows that the choice matters little on relational metrics (summed distance 3.84 vs 3.89), but it matters on two declared aspects: the share of edges between co-arriving users (17.0% vs 12.4%, real 8.1%) and the maximum degree (81% vs 97% of the value expected from the scaling law). The progressive partition was preferred for **fidelity to the birth mechanism** — in the real data 58% of users enter during observation — and not for a metric advantage, which does not exist. Moreover, transfer across populations of different size is an extrapolation, which we mitigated with relative variables, and the law that rescales `k` is applied below the range on which it is fitted. Finally, the learned coefficients fluctuate slightly between trainings (content around +0.05), so they should be read as estimates. *External validity.* The result concerns **one** real dataset and **one** synthetic dataset, both provided for the project; we do not know whether the learned mechanism generalises to other platforms or domains, and we have no second dataset with edges with which to verify it. *Conclusion validity.* The main tables come from a **single final execution** of the models with the same analyses, so they are comparable with one another but carry no confidence intervals; the fine-grained content controls are unstable across steps. The encoder comparison, in particular, has a resolution coarser than the margin between the first and the third classifier (§4.1).

---

## 7. Conclusions

We built the link structure of a population for which only text is observed, learning the edge-formation mechanism from a real network and applying it by inference, never coupling the two datasets. The arrival-event model reproduces the structure of the real network within the right order of magnitude for reciprocity (0.3224 vs 0.4177), mean degree (27.45 vs 32.05), maximum degree (274 vs 1,290), transitivity (0.0212 vs 0.0337), clustering (0.042 vs 0.080) and diameter (6 vs 6). Against the threshold and imposed-count families the advantage is one of **simultaneity**, not of supremacy on every single metric: over the seven comparable metrics it gets degree scale, reciprocity, modularity and transitivity **without imposing statistics on the networks**; its tail remains **steeper** than the real one at matched size (−3.98 vs −1.69). It remains somewhat "flatter" than the real network, and the two points on which the work can continue are the **shape** of the tail and the **cohesion**.

On content the work produces two controlled results. First: **content does not decide who connects to whom** — with authentic text, text permuted across users and zeroed content, the networks are indistinguishable. Second: the number that seemed to say the opposite, the 2.13 homophily of the real network, is **largely network position**, and drops to 1.51 and 1.29 under degree controls. Text knows the user (profile attributes with AUC up to 0.867) but not their links, which depend on the user's entry into the observed window and on their degree.

The directions that remain open are precise: a form of transfer that preserves the **degree tail** and not only relative position; the **dependence among the targets chosen within the same arrival event**, which would close the triadic part; and a content representation that acts **at the level of the pair** rather than the individual user, the only way in which text could still enter edge formation.

---

## Reproducibility and data availability

All figures in this article come from files in `results/` and from a **single final execution** of the models, with fixed seeds and with the same analyses applied to the real network and to every generated network. The generators iterate edges in **sorted order**, because the iteration order of sets of strings is not fixed in Python: without this precaution the network volume could vary by about 2% between processes. Reproducibility, however, has a boundary that we measure and declare: the **generator** is deterministic — re-running the documented command yields the **same** 7,776 arrival events, the same number of edges per event (34.934), the same share of reciprocated pairs (0.4435) and the same reciprocity (0.3224) — while the **training** is not bit-reproducible, because the network weights vary in the third decimal place between processes and the volume moves from 271,646 to **271,655** edges, that is 9 edges in 271 thousand (0.003%). The same variability affects the encoder comparison, where the margin between the first and the third is 0.002 within a variability of about 0.005: for this reason the comparison should be read as a tie among three encoders. The encoder selection, the fitting of the generator and the structural, content, growth and profile analyses are described in full in the source code; the numeric artefacts are in `results/*.csv` and the figures in `results/figures/`. The real network was never modified; the generated networks are new artefacts.

The two datasets are not redistributed here. Because the datasets are not public, full reproducibility requires access to the same inputs; parameters, seeds and procedures are documented nonetheless, so that the method can be reapplied to equivalent data.

---

## References

1. Singer, U., Guy, I., Radinsky, K. (2019). *Node Embedding over Temporal Graphs.* In Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI), 4605–4612. arXiv:1903.08889.
2. Grover, A., Leskovec, J. (2016). *node2vec: Scalable Feature Learning for Networks.* In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 855–864. arXiv:1607.00653.
3. Nguyen, G. H., Lee, J. B., Rossi, R. A., Ahmed, N. K., Koh, E., Kim, S. (2018). *Continuous-Time Dynamic Network Embeddings.* In Companion Proceedings of the Web Conference (WWW), 969–976. doi:10.1145/3184558.3191526.
4. Sankar, A., Wu, Y., Gou, L., Zhang, W., Yang, H. (2020). *DySAT: Deep Neural Representation Learning on Dynamic Graphs via Self-Attention Networks.* In Proceedings of the 13th ACM International Conference on Web Search and Data Mining (WSDM), 519–527. arXiv:1812.09430.
5. Pareja, A., Domeniconi, G., Chen, J., Ma, T., Toyoda, T., Holzer, M., Karypis, G. (2020). *EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs.* In Proceedings of the AAAI Conference on Artificial Intelligence, 34(04), 5363–5370. arXiv:1902.10191.
6. Goyal, P., Chhetri, S. R., Canedo, A. (2020). *dyngraph2vec: Capturing network dynamics using dynamic graph representation learning.* Knowledge-Based Systems, 187, 104816. arXiv:1809.02657.
7. Kazemi, S. M., Goel, R., Jain, K., Kobyzev, I., Sethi, A., Forsyth, P., Poupart, P. (2020). *Representation Learning for Dynamic Graphs: A Survey.* Journal of Machine Learning Research, 21(70), 1–73.
8. Zhou, L., Yang, Y., Ren, X., Wu, F., Zhuang, Y. (2018). *Dynamic Network Embedding by Modeling Triadic Closure Process.* In Proceedings of the AAAI Conference on Artificial Intelligence, 32(1).
9. You, J., Ying, R., Ren, X., Hamilton, W. L., Leskovec, J. (2018). *GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models.* In Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80, 5708–5717. arXiv:1802.08773.
10. Bojchevski, A., Shchur, O., Zügner, D., Günnemann, S. (2018). *NetGAN: Generating Graphs via Random Walks.* In Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80, 610–619. arXiv:1803.00816.
11. Leskovec, J., Chakrabarti, D., Kleinberg, J., Faloutsos, C., Ghahramani, Z. (2010). *Kronecker Graphs: An Approach to Modeling Networks.* Journal of Machine Learning Research, 11(33), 985–1042.
12. Barabási, A.-L., Albert, R. (1999). *Emergence of Scaling in Random Networks.* Science, 286(5439), 509–512.
13. Watts, D. J., Strogatz, S. H. (1998). *Collective dynamics of 'small-world' networks.* Nature, 393, 440–442.
14. Leskovec, J., Kleinberg, J., Faloutsos, C. (2005). *Graphs over Time: Densification Laws, Shrinking Diameters and Possible Explanations.* In Proceedings of the 11th ACM SIGKDD International Conference on Knowledge Discovery in Data Mining (KDD), 177–187.
15. Reimers, N., Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* In Proceedings of EMNLP-IJCNLP, 3982–3992. arXiv:1908.10084.
16. Muennighoff, N., Tazi, N., Magne, L., Reimers, N. (2023). *MTEB: Massive Text Embedding Benchmark.* In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics (EACL), 2014–2037. arXiv:2210.07316.
17. Xiao, S., Liu, Z., Zhang, P., Muennighoff, N. (2024). *C-Pack: Packed Resources for General Chinese Embeddings.* In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), 641–649. arXiv:2309.07597.
18. Blondel, V. D., Guillaume, J.-L., Lambiotte, R., Lefebvre, E. (2008). *Fast unfolding of communities in large networks.* Journal of Statistical Mechanics: Theory and Experiment, 2008(10), P10008. arXiv:0803.0476.
19. Hochreiter, S., Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
20. Gower, J. C. (1975). *Generalized Procrustes Analysis.* Psychometrika, 40(1), 33–51.
21. McPherson, M., Smith-Lovin, L., Cook, J. M. (2001). *Birds of a Feather: Homophily in Social Networks.* Annual Review of Sociology, 27, 415–444.
22. Granovetter, M. S. (1973). *The Strength of Weak Ties.* American Journal of Sociology, 78(6), 1360–1380.
23. sentence-transformers. *all-MiniLM-L6-v2* (model card). Hugging Face. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
