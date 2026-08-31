# Referee Report

**Manuscript:** *Multi-channel neural vetting of exoplanet transit candidates: controlled ablations and distribution-free candidate selection*

**Authors:** Gopalakrishna, M. Immadisetty

**Venue:** Astronomy and Computing (Elsevier)

---

## Summary

The manuscript presents PHANTOM, a deep-vetting architecture that encodes nine
phase-folded representations of a Kepler threshold-crossing event — including
folds at the competing period hypotheses *P*/2 and 2*P* — into per-view tokens
that a transformer combines, with a differentiable five-parameter transit
renderer attached as a reconstruction-only physics bottleneck. On the standard
15,683-event Kepler DR24 benchmark the model reaches an average precision of
0.9732 ± 0.0097 against 0.9385 ± 0.0093 for a faithfully reimplemented AstroNet
trained on identical splits (*p* = 4×10⁻⁴, five seeds), and raises precision at
95% recall from 0.805 to 0.927. The authors then subject the architecture to
eight ablation experiments over five seeds each, all null, and report four
further negative results: star-level versus event-level splitting makes no
difference (*p* = 0.97); gradient-boosted trees on summary statistics tie the
network on average precision; conformal FDR control holds only above *q* ≈ 0.05;
and zero-shot transfer to TESS degrades both PHANTOM and AstroNet to
indistinguishable performance. The central methodological claim is that
leave-one-out ablation is near-uninformative when model evidence is redundant,
and that single-deletion studies in this literature systematically understate
component value.

## Assessment of strengths

This is an unusually honest, well-constructed empirical study, and the reviewers
should state plainly what the authors did well:

1. **Real data, full pipeline.** 67.8 GB of raw Kepler light curves are
   acquired, detrended, folded and rendered from scratch; nothing is synthetic
   and no borrowed preprocessed set is used.
2. **Fair baselines.** AstroNet is reimplemented and retrained under identical
   splits rather than quoted from the literature, and a strong non-neural GBDT
   baseline that most deep-learning papers omit is included.
3. **Rigorous protocol.** Star-level splits, five seeds, significance testing
   over per-seed values, and honest reporting of the resolution limit of the
   test.
4. **The negative results are reported in full.** The null ablations, the
   conformal floor at *q* < 0.05, and the TESS transfer failure would each be
   quietly dropped by most teams. Their inclusion is the study's principal
   virtue.
5. **Reproducibility** is exemplary, and the plain-language companion document
   (explain.md) is a model of scientific communication.

## Recommendation

**Major revision**, with the expectation that the changes below are tractable.
The paper's honesty is not in question; its framing and its support for the
headline claims are. I would expect to recommend acceptance after the revision
described below. This includes a caveat, discussed in Comment 1: the paper
should state a defensible central claim, and the materials to do so are largely
already present in its own tables.

---

## Major comments

**1. The central claim needs to be decomposed and re-anchored.** The abstract
and conclusions lead with "PHANTOM improves AP over AstroNet by 0.035" and
attribute this to "the architecture and the input channels." But the authors'
own Table 2 (ablation) implies a finer decomposition that the text never draws:
PHANTOM fed only the global and local views (0.9627) — i.e., the same two input
channels AstroNet receives — still beats the reimplemented AstroNet (0.9385) by
≈0.024, whereas stripping the seven extra views and all scalars together only
costs ≈0.010. Read this way, the dominant share of the gain is the
*transformer-over-views architecture on the same inputs*, not the extra channels
or scalars that the abstract and introduction stress. Please either (i) report
this decomposition explicitly and state the claim as "architecture-level," or
(ii) demonstrate with the appropriate controls (e.g., AstroNet fed all nine
views) that the gain is attributable to the input channels. As written, the
introductory list of contributions and the abstract's emphasis sell a different
(and weaker) claim than the ablations support.

**2. The strongest published baseline is absent.** ExoMiner (Valizadegan et
al., 2022) is the strongest published vetter on essentially this task and
dataset family, and it uses a per-diagnostic-test architecture plus catalogue
parameters — exactly the design space extended here. It is cited in the
introduction and never mentioned again. If the authors claim to improve on the
published state of the art, that comparison is the salient one; if they decline
to make it (the companion explainer states they do not beat it), the manuscript
must say so in the main text and temper the framing accordingly. At minimum,
report ExoMiner's published accuracy on the same DR24 benchmark and quantify the
gap, or state explicitly and early that the claim is bounded to the AstroNet
class of two-view convolutional models.

**3. "Monotone degradation" is not supported by the ablation table as
presented.** Section 5.3 and the discussion infer an orderly, monotone pattern —
"effects grow monotonically as more evidence is withdrawn" — but Table 2 is not
a nested chain. Rows such as "light-curve-derived scalars only," "no scalar
inputs," and "global and local views only" are alternative configurations, not
successive deletions contributing totalled evidence. The apparent ordering is
largely by hand-assigned "amount of evidence removed," which is post hoc. To
support the monotonicity claim the authors should provide a single nested chain
(delete one channel at a time in a fixed order, five seeds, and plot the
cumulative cost), or retreat the claim to what the table does show: every
single-component deletion is null and cumulative deletions degrade more. This
distinction matters because the monotonicity argument carries a large share of
the paper's methodological weight.

**4. The paper is ambiguous about single-run versus ensemble numbers.** Section
5.4 reports that a five-member ensemble on a fixed split improves AP by 0.0028,
and both the conformal analysis and the decoder parameter-recovery use
ensembles. Table 1 labels its columns "averaged over five random seeds." Which
of Table 1's figures are seed-means of *single* runs and which are *ensemble*
predictions? What is the ensemble protocol across seeds — does each seed
contribute a member, or is each of five seeds itself an ensemble member?
Because the significance tests (Welch's t on per-seed values, *p* = 4×10⁻⁴) must
be computed on single-run values, this ambiguity also affects whether the p-value
is on the same object as the reported mean. Please state the convention once,
up front, and apply it consistently to the headline, the ablations, the decoder,
and the conformal sections.

**5. The operational headline has no uncertainty.** The 62% reduction in wasted
follow-up rests on a single pair of point estimates (precision at 95% recall,
0.805 → 0.927), and the P@0.90/P@0.95 columns in Table 1 carry no seed spread or
significance test, unlike every other reported quantity. Report the per-seed
spread of precision-at-recall and a test for the 0.927-versus-0.902 gap against
the GBDT, or refrain from the 62% framing. An operational claim is exactly where
this readership needs the error bars.

**6. The null ablations should be reported as bounds, and the power problem made
quantitative.** The manuscript states in Section 6 that an effect of ≈0.005 in
average precision would not be reliably detected at five seeds. With the seed
standard deviations quoted (≈0.006–0.017), a two-tailed Welch test at 4 degrees
of freedom resolves roughly ΔAP ≳ 0.015–0.030; the maximum observed ΔAP of 0.0121
is therefore consistent with components individually worth up to ~1.3 points of
AP. The paper already gestures at this in the Limitations, but the discussion
section currently reads as though the ablations exclude meaningful component
value. State the minimum detectable effect explicitly (a one-line power
calculation), treat the nulls as upper bounds, and discuss the multiple-
comparison issue — eight tests at α = 0.05, uncorrected, is actually favourable
to the authors' conclusion, and the correction treatment should be stated.

**7. The decoder parameter-recovery result is partly anchored by the
catalogue.** The local-view grid is defined in units of the *reported* transit
duration (±4 durations), and the duration estimator is bounded a priori to
(0.05, 2) × θ_cat. A median relative error of 3.2% is therefore obtained under a
generous prior window centred near the pipeline value, and the comparison against
"published values" partly re-tests the pipeline's own duration through a bounded
affine head. The deeper claim — that the *depth* (6.5%, 66% within 10%) is
recovered from the raw normalisation scale κ, which is measured, not taken from
the catalogue — is the cleaner evidence and should be foregrounded. Please also
report the distribution of the fitted bounded parameters (any saturation near the
bounds, especially for non-transits where s saturates to the V-shaped regime)
and state explicitly that κ is part of the derived scalar group so the argument
covers TESS as well.

**8. Conformal analysis: add the diagnostics the paper's own argument
predicts.** Two cheap analyses would substantially strengthen Section 5.4.
First, the validity of the conformal p-values is directly testable: their
distribution on the *test* negatives should be (approximately) uniform; please
plot it (or a quantile-quantile comparison) and comment on the discrete grid.
Second, the authors' own diagnosis of the *q* < 0.05 failure — shared-set
dependence — is testable by implementing the conditional-calibration variant of
Bates et al. they cite; if it removes the floor, the diagnosis is confirmed and
the practical prescription (enlarge the calibration set or use the variant) is
concrete. Also state over precisely which set the Benjamini–Hochberg procedure
is applied and how FDR and recall are computed with respect to the PC/(AFP∪NTP)
composition of the test set.

**9. The cross-mission section is currently a corroboration, not a study.** The
finding that both architectures collapse to ≈0.79 AP is reported as corroborating
Kopparapu et al. That is fine, but the section could do more with the data the
authors already hold: (i) test whether temperature-scaling/recalibration on a
small TESS slice restores some of the drop, distinguishing a calibration collapse
from a representation failure; (ii) stratify TESS performance by period regime and
cadence, since the authors themselves identify these as the divergent covariates;
(iii) report precision at fixed recall on TESS, the operational metric used
throughout. As written, the section's conclusion (architectural advantages do not
survive the mission transfer) interacts with the *abstract's* claim in a way the
reader is left to reconcile.

---

## Minor comments

1. Consider reordering the abstract: the controlled, experimental framing and the
   two negative results are the paper's most distinctive content, and a journal
   of methods and reproducibility (Astronomy and Computing) will reward leading
   with them.

2. The claim about parameter parity deserves care: "9.16M vs 10.55M (13% fewer)"
   is presented as ruling out capacity as an explanation. Parameter count is a
   weak proxy for capacity/compute; if the authors want that argument, report
   FLOPs or training time under the matched budget instead.

3. Section 2: 20,367 DR24 TCEs → 15,737 labelled → 15,683 surviving
   preprocessing. Confirm the excluded 54 events are missing-data rather than
   processing failures, and confirm that all four models (cascade, AstroNet,
   GBDT, PHANTOM) see exactly the same 15,683 rows.

4. Table 2 (tab:views): the two harmonic rows share the label "Harmonic" with
   different bin counts and hypotheses; a label distinguishing the P/2 from the
   2P fold (e.g., "Harmonic (P/2)" / "Harmonic (2P)") would prevent confusion
   with the "harmonic contrast head."

5. State explicitly that all per-view normalisation statistics (median, minimum,
   and the depth divisor κ) are computed per-event and therefore cannot leak
   split information, and that standardisation of scalars uses training-split
   statistics only, as is done for imputation.

6. The red-noise proxy — ratio of six-hour-bin scatter to the white-noise
   expectation — deserves its defining equation or a clear verbal definition; as
   written it is the only scalar whose construction the reader cannot verify.

7. The random-mirroring augmentation is justified by transit symmetry. Note that
   limb-darkening and impact-parameter asymmetries make real profiles only
   approximately symmetric; a sentence acknowledging the approximation would
   head off a referee objection.

8. The classical-cascade baseline (AP 0.40): state whether its thresholds were
   tuned on the validation split (look-elsewhere correction is mentioned) and
   how it was applied to the same 15,683 rows.

9. The decoder's ingress-softness s is presented as separating U- and V-shaped
   profiles (Fig. 4b). The figure is qualitative; a quantitative discriminant
   (e.g., ROC-AUC of s alone against the class label, or against the I/U-shaped
   subsets) would let the reader judge the claim.

10. Add a per-class breakdown (AFP vs NTP) to the headline or supplementary
    results; the two negative classes are physically distinct and collapse a
    real part of the problem.

11. The related-work paragraph cites Islam (2026), which is architecturally the
    closest prior work (attention over the same two views plus temperature
    scaling). It is cited in one line and never discussed; one paragraph
    distinguishing the harmonic contrast, the formal FDR guarantee, and the
    treatment of calibration would earn its place.

12. Release the exact fold assignments (star → train/val/test) so the identical
    split claim is auditably reproducible, not merely described.

13. The manuscript has placeholder author metadata (EMAIL@INSTITUTION,
    DEPARTMENT, CITY) to be completed before submission; verify the confirmed-planet
    count (264) and the TESS test-set composition (1,298 vs 1,053) reconcile with
    the tables and with results/summary.json.

14. The Conclusions refer to "the star-level split we adopted to prevent leakage
    makes no measurable difference" — consistent with the body, but the body
    also establishes (Section 5.2) that its cost is nil and it should remain
    standard practice; keep that sentence prominent so the negative result is
    not misread as endorsing the weaker protocol.

15. Typography/labels: "standard deviation" is used for seed means; consider
    "±" over seeds throughout the tables being defined in a single caption note.

---

## Comments to the editor

This is a serious, well-executed, and unusual manuscript: a deep-vetting study
whose authors actively try to invalidate their own architecture and publish the
failures. Its distinctive value — the redundant-evidence argument about
leave-one-out ablation, the null split-protocol result, the bounded conformal
guarantee, and the competitive tabular baseline — is closer to the stated scope
of Astronomy and Computing than to an astronomy letter, and the submission is
well matched to the journal. The revision requested above is mostly a matter of
decomposition, conventions, and diagnostics rather than new experiments, though
Comments 1, 2 and 3 do require additional analysis and, in the case of Comment
3, a nested ablation row. The paper should not be accepted as is, principally
because its abstract's emphasis is at odds with what its own ablation tables
demonstrate; I would expect this to be resolved cleanly in major revision.

*Prepared for review purposes; role: Reviewer 1 (single-anonymised).*