# PHANTOM, explained in plain language

*A guide for supervisors, lab-mates, and anyone who has to understand this
project in ten minutes without reading the paper.*

---

## 1. The one-paragraph version

Space telescopes find planets by watching a star dim slightly when a planet
passes in front of it. Automated software flags millions of these "dips", but
the overwhelming majority are not planets — they are pairs of stars eclipsing
each other, blended light from a neighbouring star, or instrument noise.
Deciding which dips are real planets is called **vetting**, and it used to be
done by humans, one at a time. We built a neural network, **PHANTOM**, that
looks at each candidate through nine different views of the same data at once
and decides. On the standard Kepler benchmark it beats the published
state-of-the-art architecture (Google's AstroNet) by a clear margin while using
*fewer* parameters. We then tried very hard to break our own result, and the
things we found while failing to break it are — honestly — the most useful part
of the paper.

---

## 2. The problem, from scratch

### What a transit is

A star has some brightness. If a planet orbits it and its orbit happens to be
edge-on from our point of view, then once per orbit the planet crosses the
star's disc and blocks a tiny fraction of the light. For an Earth-sized planet
around a Sun-like star that fraction is about **0.008%** — eight parts in a
hundred thousand. You do not see it in one crossing; you see it by folding
years of measurements on the orbital period and stacking every crossing on top
of each other.

### Why it is hard

The Kepler mission's automatic search produced roughly **34,000** flagged
events. Only a minority are planets. The rest are:

| Impostor | What it actually is | How you catch it |
|---|---|---|
| **Eclipsing binary** | Two stars orbiting each other; one eclipses the other | The eclipse is far too deep, and V-shaped rather than flat-bottomed |
| **Half-period alias** | The software locked onto half or twice the true period | Fold at *P/2* and *2P* and see which looks sane |
| **Secondary eclipse** | The "planet" is hot and glows — so it disappears behind the star too | Look at orbital phase 0.5 for a second, shallower dip |
| **Odd/even mismatch** | A binary whose two eclipses have different depths | Fold odd-numbered and even-numbered transits separately and compare |
| **Background blend** | A faint eclipsing binary in the same pixel as your target star | The image centroid shifts during the "transit" |
| **Stellar variability** | The star itself is spotty or pulsating | It doesn't look like a transit at all |

Every one of those checks is a *different view of the same light curve.* That
observation is the whole design of our model.

### Why anyone cares

Confirming a planet costs telescope time — sometimes months of it on a large
ground-based facility. If your candidate list is 20% junk, one fifth of that
time is wasted. Cutting the junk rate is directly worth money and telescope
nights.

---

## 3. What already existed

- **Human vetting.** Accurate, does not scale.
- **Classical tests.** Hand-written rules: is there a secondary eclipse? is it
  V-shaped? Fast, interpretable, and — as we measure — *weak*. Our
  reimplementation of the classical cascade gets an average precision of
  **0.40**.
- **AstroNet** (Shallue & Vanderburg, 2018). The paper that started deep
  learning in this field. A convolutional network fed **two** views: a "global"
  view of the whole folded orbit, and a "local" zoom on the transit itself. It
  works well — 0.94 average precision in our reimplementation.
- **ExoMiner** (Valizadegan et al., 2022). Adds more views and stellar
  catalogue parameters. Currently the strongest published system.

So the field already knew that more views help. Our question was: *what is the
right way to combine many views, and how much of the improvement is real?*

---

## 4. What we built

### 4.1 Nine views instead of two

Each candidate is turned into nine one-dimensional signals, each one designed to
expose a specific impostor from the table above:

| View | Length | What it is for |
|---|---|---|
| `global` | 2001 bins | The whole folded orbit — overall shape, secondary eclipses |
| `local` | 201 bins | Zoomed on the transit — depth, duration, ingress shape |
| `odd` | 201 | Odd-numbered transits only |
| `even` | 201 | Even-numbered transits only — compare with `odd` to catch binaries |
| `secondary` | 201 | Folded on the deepest non-primary dip |
| `half` | 201 | Folded at half the reported period |
| `double` | 2001 | Folded at twice the reported period |
| `cent_global` | 2001 | Image centroid position, whole orbit — catches blends |
| `cent_local` | 201 | Image centroid, zoomed |

### 4.2 A transformer that decides which view to trust

AstroNet concatenates its views and pushes them through a shared network. We
instead encode each view separately into a token, then let a **transformer**
attend across the nine tokens. The point is that the transformer can learn
*conditional* logic — "if the centroid channel is noisy, ignore it and lean on
the odd/even comparison" — which a fixed concatenation cannot express.

### 4.3 A physics bottleneck (the differentiable transit renderer)

This is the part most people find interesting. Alongside the classifier we
attach a small **decoder** that takes a 5-number latent vector and *draws* a
transit shape from it, using an analytic formula:

- how deep the dip is,
- how long it lasts,
- where its centre is,
- how limb-darkened the star is (stars are dimmer at the edge),
- how sharp the ingress and egress are.

We train this decoder **only** to reproduce the observed light curve — it never
sees a "correct" duration or depth label. Then, after training, we compare the
five numbers it invented against published values for 264 confirmed planets.

It recovers **durations to 3.2%** and **depths to 6.5%** median error. That is
worse than the Kepler mission's own careful fits (2.0% and 3.3%), and we say so
plainly. But it is not a coincidence — it means the network's internal
representation genuinely encodes transit geometry rather than an arbitrary
statistical code. **That is what makes the model interpretable:** you can ask it
"what shape did you think you saw?" and get a physically meaningful answer.

### 4.4 Harmonic contrast

Since we already fold at *P/2*, *P*, and *2P*, we explicitly feed the network the
*differences* between those three representations, not just the three
representations themselves. If a candidate looks identical at *P* and *2P*, its
period is probably wrong. This was our most "novel-sounding" idea. Read
Section 6 before you get attached to it.

### 4.5 Statistically honest candidate lists (conformal selection)

A network outputs a score between 0 and 1. Nobody knows what score is "good
enough". We use **conformal prediction** plus the **Benjamini–Hochberg**
procedure to turn scores into a list with a promise attached: *"at most 10% of
this list is expected to be junk."* That promise is what an observing proposal
actually needs.

---

## 5. How we tested it — the protocol is the point

This is the part to emphasise to a reviewer, because it is where most papers in
this area are weak.

1. **Real data, all of it.** We downloaded **67.8 GB** — 157,982 raw Kepler light
   curve files — and processed them ourselves from FITS to model input. Nothing
   is synthetic. Nothing is borrowed from someone else's preprocessed dataset.
2. **The standard benchmark.** Kepler DR24 threshold-crossing events with
   Autovetter labels: **15,683 labelled events**, the same benchmark AstroNet
   used.
3. **Star-level splits.** Many stars host several candidates. If a star's
   candidates are scattered across train and test, the model can memorise the
   star rather than learn the physics. We split by *star*, not by event.
4. **Identical splits for every model.** AstroNet, the gradient-boosted trees,
   the classical cascade and PHANTOM all see exactly the same training, validation
   and test rows. Differences are attributable to the model, not to luck.
5. **Five random seeds each**, with proper significance testing (Welch's t-test)
   on the seed-level results.
6. **Strong baselines, not straw men.** We reimplemented AstroNet faithfully and
   trained it under our own protocol rather than quoting its published number.
   We also included a gradient-boosted decision tree on summary statistics — the
   baseline most deep learning papers quietly omit.

**Total compute:** 61 training runs, 217 GPU-minutes on 2× NVIDIA L40S — about
two hours of wall-clock. This is not an expensive result to reproduce.

---

## 6. The results

### 6.1 Headline

| Model | Parameters | AUC | Average precision | Precision at 95% recall |
|---|---|---|---|---|
| Classical cascade | — | 0.7668 ± 0.0162 | 0.4017 ± 0.0193 | 0.271 |
| AstroNet (reimplemented) | 10.55 M | 0.9825 ± 0.0028 | 0.9385 ± 0.0093 | 0.805 |
| GBDT on summary statistics | — | 0.9923 ± 0.0025 | **0.9754 ± 0.0056** | 0.902 |
| **PHANTOM** | **9.16 M** | **0.9929 ± 0.0027** | 0.9732 ± 0.0097 | **0.927** |

**The number to quote:** at 95% recall — that is, keeping 95% of the real
planets — AstroNet's candidate list is 19.5% junk. PHANTOM's is 7.3% junk.
**That is 62% less wasted follow-up telescope time**, from a model with 13%
fewer parameters. The average-precision gap over AstroNet is +0.035 with
*p* = 4 × 10⁻⁴ across five seeds.

### 6.2 Negative result #1 — no single component explains the gain

We removed each component in turn and retrained, five seeds each. Eight
ablations. Here is what happened:

| Component removed | Average precision | Cost of removing it | *p* |
|---|---|---|---|
| *(nothing — full model)* | 0.9732 ± 0.0097 | — | — |
| Harmonic contrast head | 0.9738 ± 0.0065 | **−0.0007** *(it helped to remove it)* | 0.90 |
| Harmonic views (*P/2*, *2P*) | 0.9702 ± 0.0119 | +0.0030 | 0.68 |
| Differentiable decoder | 0.9690 ± 0.0061 | +0.0042 | 0.44 |
| Catalogue diagnostics | 0.9670 ± 0.0095 | +0.0062 | 0.34 |
| Centroid views | 0.9660 ± 0.0109 | +0.0072 | 0.31 |
| All views except global + local | 0.9627 ± 0.0117 | +0.0104 | 0.16 |
| All scalars except light-curve-derived | 0.9621 ± 0.0113 | +0.0110 | 0.14 |
| All scalar inputs | 0.9610 ± 0.0168 | +0.0121 | 0.21 |
| *(for contrast: AstroNet, a different architecture)* | 0.9385 ± 0.0093 | +0.0347 | **0.0004** |

**Not one ablation is statistically significant.** Yet removing components
*cumulatively* degrades performance monotonically, and the full model beats
AstroNet decisively. The evidence is **redundant**: when the centroid channel is
removed, the odd/even and secondary channels partly cover for it.

This has a consequence that goes beyond our paper. **Leave-one-out ablation —
the standard justification tool in this entire literature — is close to
uninformative when a model's evidence is redundant.** It systematically
understates the value of components. If you have ever read a paper that
justified an architecture with single-component ablations, this result says you
should be suspicious of it. We think this is our most defensible contribution.

### 6.3 Negative result #2 — the leakage we expected isn't there

We predicted that splitting by event instead of by star would inflate scores,
because **65.1%** of labelled events share a host star with another labelled
event. We ran it. The difference is **0.0002 in average precision, *p* = 0.97**.

There is no leakage. The reason turns out to be that sibling events on the same
star are *not near-duplicates* — different periods, different depths, different
folded shapes. Star-level splitting is still the right hygiene, but on this
benchmark it is not what separates a careful paper from a careless one.

### 6.4 Negative result #3 — a decision tree ties the neural network

Gradient-boosted trees on hand-computed summary statistics match PHANTOM on
average precision (0.9754 vs 0.9732). The deep model only pulls decisively
ahead at **high recall**, which is the regime that matters operationally
(0.927 vs 0.902 precision at 95% recall), but the honest headline is: *if you
only care about average precision, you do not need a neural network for this
task.* Most papers in this field do not run this baseline. We think that is why.

### 6.5 Negative result #4 — the statistical guarantee has a floor

Conformal FDR control works as promised at 5% and above:

| Target | Selected | Realised FDR | Recall |
|---|---|---|---|
| 0.01 | 181 | **0.033** ✗ | 0.478 |
| 0.02 | 290 | **0.034** ✗ | 0.765 |
| 0.05 | 353 | 0.057 ✓ | 0.910 |
| 0.10 | 389 | 0.090 ✓ | 0.967 |
| 0.20 | 449 | 0.194 ✓ | 0.989 |

Below 5% it fails — at a 1% target the realised rate is 3.3%, more than triple.
We diagnose why (the calibration set has ~1,200 negatives, so the finest
attainable *p*-value is ~8 × 10⁻⁴, and conformal *p*-values sharing a
calibration set are mutually dependent) and say what to do instead. We could
have reported only the levels that worked. We didn't.

### 6.6 Cross-mission transfer fails, and that is a finding

We took models trained on Kepler and applied them, with no retraining, to
**2,351 TESS events**. Both models collapse to roughly the same place:

| Model | Average precision on TESS |
|---|---|
| PHANTOM | 0.789 |
| AstroNet | 0.797 |
| Random guessing | 0.552 |

They are indistinguishable, and both are far from their Kepler performance.
Kepler and TESS differ in cadence, mission duration, pixel scale and noise
properties, and the network learned Kepler's specifics. This corroborates an
independent 2026 result by Kopparapu et al. Anyone planning to deploy a
Kepler-trained vetter on TESS data should read this section first.

---

## 7. Is it novel? An honest answer

Your guide will ask this. Answer it directly rather than defensively.

**Genuinely new:**
- Harmonic-hypothesis attention — explicitly contrasting folds at *P/2*, *P*, *2P*
  as first-class model input. Nobody has done this. *(Though see below.)*
- The methodological finding about ablation under redundant evidence. This is
  new and it generalises beyond astronomy.

**Novel in application, not in kind:**
- The differentiable transit renderer as an unsupervised physics bottleneck.
  Analytic transit models are decades old; wiring one in as a trainable decoder
  and showing the latent space recovers real parameters is new to this task.
- Transformer-over-views. Transformers are not new; this arrangement for
  transit vetting is.

**Not novel, and we do not claim it:**
- Conformal prediction with BH control (Bates et al. 2023). A concurrent 2026
  paper, ExoVeil, already applied conformal methods to transit *detection*. Our
  claim is narrowed to FDR control over a *selected list* for *vetting*.
- Star-level splitting. Standard hygiene.
- Multi-view vetting itself. ExoMiner got there first.

**The uncomfortable part:** our headline novelty (harmonic attention) is worth
−0.0007 in average precision and is not statistically significant. We report
this. The paper's centre of gravity therefore shifted from "we invented a
component" to "we ran the controlled experiment that shows why nobody can prove
they invented a component." That is a weaker novelty claim and a stronger
paper.

---

## 8. So — is it publishable?

**Yes, with the right venue and the right framing.**

**In its favour:**
- Real data, full pipeline from raw FITS, nothing synthetic.
- A large, statistically significant improvement over the standard baseline on
  the standard benchmark, with the baseline retrained under identical conditions.
- Rigorous protocol: identical splits, five seeds, significance tests, strong
  non-neural baselines.
- Four honest negative results, one of which (ablation under redundancy) is a
  genuine methodological contribution.
- Fully reproducible: every script, seed, and configuration is in this repo.

**Against it:**
- The novel architectural components do not individually survive ablation.
- A gradient-boosted tree ties the model on average precision.
- ExoMiner is still the stronger system overall and we do not beat it.
- Cross-mission transfer does not work.

**Realistic assessment:** this is a careful empirical and methodological study,
not a breakthrough-architecture paper. Framed as the former it is a solid
submission. Framed as the latter it would be rejected, because the ablations
would be turned against it. The manuscript is already written the honest way.

### Free-to-publish (subscription) venues

The paper is formatted for the first of these.

| Journal | Publisher | Indexed | Cost to publish |
|---|---|---|---|
| **Astronomy and Computing** ← *current target* | Elsevier | Scopus, WoS | **Free** under the subscription route; the ~USD 2,220 charge applies only if you opt into open access |
| **New Astronomy** | Elsevier | Scopus, WoS | Free under the subscription route |
| **Journal of Astrophysics and Astronomy** | Springer / Indian Academy of Sciences | Scopus, WoS | Hybrid — free under the subscription route |
| **Advances in Space Research** | Elsevier / COSPAR | Scopus, WoS | Free under the subscription route |

**Avoid if you cannot pay:** MNRAS became fully open access and charges about
**USD 2,540** per accepted paper (waivers exist for low- and middle-income
countries — check whether yours qualifies before assuming). ApJ and AJ are also
fully open access with charges.

Astronomy and Computing is the natural home: it explicitly publishes methods,
software and reproducibility work, which is exactly what this is. **Verify the
fee policy on the journal's own "open access options" page at submission time —
publishers change these, and this table reflects August 2026.**

---

## 9. The two-minute verbal version

> Space telescopes flag millions of possible planet transits, but most are false
> alarms — usually two stars eclipsing each other. Sorting them is the bottleneck
> between raw data and a planet catalogue.
>
> We built a network that looks at each candidate through nine different
> representations at once — zoomed in, zoomed out, odd transits versus even
> transits, folded at half and double the period, plus the image centroid — and
> uses a transformer to decide which of those views to trust for that particular
> candidate. We also attached a small physics module that tries to *redraw* the
> transit from five physical numbers. It's never told the right answer, but
> afterwards those five numbers match published planet parameters to a few per
> cent — so the model is genuinely reasoning about transit geometry, not
> pattern-matching.
>
> On the standard Kepler benchmark, with 15,683 real events, it beats Google's
> AstroNet architecture using fewer parameters. Practically: at the recall
> astronomers actually operate at, it cuts wasted follow-up observations by 62%.
>
> The interesting part is what happened when we tried to prove *which piece*
> made it work. We removed each component and retrained — eight ablations, five
> seeds each — and not one was statistically significant, even though the full
> model clearly wins. The evidence is redundant; the channels cover for each
> other. That means the ablation studies everyone uses to justify architectures
> in this field are close to uninformative. That's now the paper's main claim.

---

## 10. Questions your guide will ask

**"Did you actually download and process the data yourself?"**
Yes — 67.8 GB, 157,982 Kepler light curve files, from the public archive, in 1.18
hours at 16 MB/s. Preprocessing to model input took 11 minutes on 36 workers
with zero target failures. `scripts/download_kepler.py` and
`scripts/preprocess.py`.

**"How do I know you didn't just get lucky with a seed?"**
Every configuration is five seeds; we report mean ± standard deviation and use
Welch's t-test. The headline gap has *p* = 4 × 10⁻⁴.

**"Is the AstroNet comparison fair?"**
Fairer than quoting its published number. We reimplemented it (10.55 M
parameters), gave it the input views its paper specifies, and trained it on
*identical* splits with the same budget and the same preprocessing.

**"What's the actual scientific contribution if the ablations are null?"**
Three things. (1) A model that measurably beats the standard baseline on the
standard benchmark. (2) The demonstration that leave-one-out ablation cannot
attribute that gain — which is a methodological warning to the whole subfield.
(3) Four negative results that stop other groups from wasting time: the leakage
scare is unfounded, trees are competitive, conformal control has a floor, and
Kepler→TESS transfer does not work.

**"Why should a reviewer believe the physics decoder means anything?"**
Because it was never given parameter labels, yet its outputs match published
durations to 3.2% and depths to 6.5% for 264 confirmed planets. That is a
falsifiable prediction we made and checked.

**"Can I reproduce it?"**
Yes. Every script, split seed and configuration is in this repo, and the
raw data are public. See the README.

**"What are the limitations?"**
Stated in the paper: we do not beat ExoMiner; the ablations are null; conformal
control fails below *q* = 0.05; cross-mission transfer fails; and the conformal
analysis assumes exchangeability, which the shared-calibration dependence
partially violates.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **Transit** | A planet crossing in front of its star, dimming it slightly |
| **Light curve** | Brightness of a star measured over time |
| **TCE** | Threshold-crossing event — a dip the automatic search flagged |
| **Vetting** | Deciding whether a TCE is a real planet or a false positive |
| **PC / AFP / NTP** | Planet candidate / astrophysical false positive / non-transiting phenomenon — the three Autovetter labels |
| **Phase folding** | Wrapping the time axis modulo the orbital period so every transit stacks up |
| **Detrending** | Removing slow brightness changes from the star and instrument, keeping the sharp transit |
| **Eclipsing binary (EB)** | Two stars orbiting and eclipsing each other — the main impostor |
| **Centroid** | The measured centre-of-light position on the detector; shifts if the dip comes from a neighbouring star |
| **AUC** | Area under the ROC curve; 0.5 = random, 1.0 = perfect |
| **Average precision (AP)** | Area under the precision–recall curve; the right metric when classes are imbalanced |
| **Precision at 95% recall** | Of the candidates you keep when catching 95% of real planets, what fraction are real |
| **Ablation** | Removing one component and retraining, to see how much it mattered |
| **Conformal prediction** | A distribution-free way to attach calibrated confidence to predictions |
| **FDR** | False discovery rate — the expected fraction of your selected list that is junk |
| **Zero-shot transfer** | Applying a trained model to new data without retraining |

---

## 12. Where everything lives

| Path | What it is |
|---|---|
| `paper/main.tex` | The manuscript |
| `paper/main.pdf` | Built PDF, 23 pages |
| `paper/figures/` | All seven figures, PNG and PDF |
| `exonet/model.py` | PHANTOM architecture, transformer and transit decoder |
| `exonet/views.py` | The nine view representations |
| `exonet/spline.py` | BIC-selected spline detrending |
| `exonet/conformal.py` | Conformal *p*-values and BH selection |
| `scripts/download_kepler.py` | Bulk download from the public archive |
| `scripts/preprocess.py` | FITS → model input |
| `scripts/train.py` | Single training run |
| `scripts/run_sweep.py` | The full 56-run experiment grid |
| `scripts/baselines.py` | Classical cascade and GBDT |
| `scripts/evaluate.py` | Aggregation, significance tests, tables |
| `scripts/transfer_tess.py` | Zero-shot cross-mission evaluation |
| `results/comparison.csv` | Every number in the results table |
