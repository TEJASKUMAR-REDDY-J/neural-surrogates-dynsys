"""Typeset the paper to PDF.

This machine has no LaTeX and no pandoc (see ENVIRONMENT.md), so the paper is set with
reportlab rather than the CAISc LaTeX template. The section structure, the two mandatory
checklists and the page discipline of that template are followed; only the typesetting
engine differs.

    python paper/make_paper.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "RESULTS" / "figures"
OUT = HERE / "paper.pdf"

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Times-Roman",
                      fontSize=9.6, leading=13.2, alignment=TA_JUSTIFY,
                      spaceAfter=5)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Times-Bold", fontSize=11.5,
                    leading=14, spaceBefore=11, spaceAfter=5, textColor=colors.black)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=10,
                    leading=12.5, spaceBefore=8, spaceAfter=3, textColor=colors.black)
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="Times-Bold", fontSize=16,
                       leading=19, spaceAfter=4)
AUTH = ParagraphStyle("auth", parent=ss["Normal"], fontName="Times-Roman", fontSize=10,
                      alignment=TA_CENTER, spaceAfter=2)
ABS = ParagraphStyle("abs", parent=BODY, fontSize=9.2, leading=12.4,
                     leftIndent=0.8 * cm, rightIndent=0.8 * cm, spaceAfter=4)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.3, leading=10.6, alignment=TA_CENTER,
                     textColor=colors.HexColor("#333333"), spaceBefore=3, spaceAfter=9)
MONO = ParagraphStyle("mono", parent=BODY, fontName="Courier", fontSize=8, leading=10)


def P(t, s=BODY):
    return Paragraph(t, s)


def fig(name, caption, width=15.2 * cm):
    """Embed a figure scaled to `width`, preserving aspect.

    The size is read with PIL and passed to the Image constructor. Setting drawWidth and
    drawHeight on an already-constructed Image did not take effect here - the figures
    rendered at natural size and overflowed the page on both edges.
    """
    from PIL import Image as PILImage

    p = FIG / name
    with PILImage.open(p) as im:
        w_px, h_px = im.size
    img = Image(str(p), width=width, height=width * h_px / w_px)
    img.hAlign = "CENTER"
    return [img, P(caption, CAP)]


def table(rows, widths, header=True, size=8.4):
    t = Table(rows, colWidths=widths, hAlign="CENTER")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), size),
        ("LEADING", (0, 0), (-1, -1), size + 2.6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.black),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.black),
    ]
    if header:
        style += [("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black)]
    t.setStyle(TableStyle(style))
    return t


def build():
    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            leftMargin=2.4 * cm, rightMargin=2.4 * cm,
                            topMargin=2.0 * cm, bottomMargin=2.0 * cm,
                            title="Which model fits this data?",
                            author="T. K. Jaikrishnan")
    s = []

    # ---------------------------------------------------------------- title -------------
    s.append(P("Which model fits this data?<br/>"
               "<font size=12>Training-free model-family selection across 47 datasets</font>",
               TITLE))
    s.append(P("Tejaskumar Jaikrishnan", AUTH))
    s.append(P("<i>Lossfunk autovoila research track</i>", AUTH))
    s.append(Spacer(1, 8))

    s.append(P("<b>Abstract.</b> Choosing a model for a new dataset normally means training "
               "several and comparing them, which costs exactly the effort the choice was "
               "meant to save. We ask whether properties measurable <i>before</i> training "
               "can answer it instead. Across 47 datasets spanning 13 domains, evaluated "
               "under one prediction task with eight method families, the architecture is a "
               "second-order effect: choosing the best method over the second-best buys a "
               "median of 0.038 R<super>2</super>, while the achievable ceiling varies by "
               "1.232 across datasets, a factor of 33. Two things follow. First, that "
               "ceiling is itself predictable without training: an estimated noise floor "
               "correlates with the best achievable R<super>2</super> at "
               "&#961; = &#8722;0.87. Second, a model-family selector built on seven such "
               "properties beats every fixed strategy and, critically, holds up when entire "
               "domains are held out (mean regret 0.038 against 0.063 for the best fixed "
               "choice, p = 0.013). The practical motivation is a finding we did not "
               "anticipate: neural networks do not degrade gracefully. On 18 of 47 datasets "
               "they land within 0.05 of the best available method; on 16 they fall below "
               "R<super>2</super> = &#8722;0.5, worse than predicting the training mean. "
               "Which side you land on is predictable from rows-per-dimension and coverage "
               "before a single gradient step.", ABS))
    s.append(Spacer(1, 5))

    # ---------------------------------------------------------------- 1 -----------------
    s.append(P("1. Introduction", H1))
    s.append(P("The standard way to pick a model is to fit several and keep the best. That "
               "is defensible when training is cheap and honest when the comparison is fair, "
               "but it inverts the usual order of enquiry: we learn about the data by "
               "spending compute on models, rather than deciding how much compute the data "
               "deserves. This paper asks whether the second order is available."))
    s.append(P("We frame the question as model-<i>family</i> selection. Given a dataset never "
               "seen before, and only statistics computable without fitting anything, which "
               "family of method should be reached for &#8212; a trivial baseline, a "
               "classical statistical method, or a neural network? We answer it with a "
               "selector validated out of sample, and scored by regret rather than accuracy, "
               "since recommending a method that was very nearly as good as the winner is "
               "not a real cost."))
    s.append(P("Our contributions are: (i) a quantification, across 47 datasets and 13 "
               "domains under one task, of how much the choice of method matters relative to "
               "the choice of dataset; (ii) evidence that the achievable ceiling is "
               "predictable from training-free properties; (iii) a model-family selector "
               "that transfers across held-out domains; and (iv) the observation that neural "
               "networks in this regime either match the best method or fail catastrophically, "
               "with little in between, which is what makes advance prediction worth having."))

    # ---------------------------------------------------------------- 2 -----------------
    s.append(P("2. Setup", H1))
    s.append(P("<b>Data.</b> 47 datasets across 13 kinds: weather, sales, macroeconomics, "
               "sensors, audio, biomedical panels, chemistry, images, six simulated chaotic "
               "systems, and three controls whose answers are known in advance (white noise, "
               "a periodic signal, a random walk). The controls calibrate the pipeline: the "
               "best of all methods reaches R<super>2</super> = &#8722;0.000 on white noise "
               "and +1.000 on the periodic signal."))
    s.append(P("<b>Task.</b> One task everywhere, so nothing is confounded by framing. "
               "Standardise, take a window of the last eight observations, predict the next. "
               "Train and test are contiguous blocks separated by a gap of 5% of the series, "
               "so nothing adjacent in time leaks across the split."))
    s.append(P("<b>Methods.</b> Eight families spanning what practitioners actually reach "
               "for: the training-set mean and persistence (the floors); ridge regression, "
               "k-nearest-neighbour analogue lookup, and gradient boosting (classical); and "
               "an MLP, a 1-D CNN and a GRU (neural). Three seeds each, 1,128 finite results."))
    s.append(P("<b>Properties.</b> Seven quantities computed without training: coverage "
               "(median distance from a test point to its nearest training neighbour, over "
               "the spread of the training set); an estimated noise floor (near-identical "
               "inputs whose next values disagree); distribution shift between train and "
               "test; decorrelation time; spectral entropy; dimension; and length."))

    s += fig("01_data_vs_architecture.png",
             "Figure 1: Every method on every dataset, ordered by predictability. Vertical "
             "spread is the architecture; horizontal spread is the data. Choosing the best "
             "method over the second best buys 0.038 R<super>2</super>; the achievable "
             "ceiling spans 1.232.")

    # ---------------------------------------------------------------- 3 -----------------
    s.append(P("3. Results", H1))

    s.append(P("3.1 The choice of method is a second-order effect", H2))
    s.append(P("Figure 1 shows all eight methods on all 47 datasets. The median gain from "
               "picking the best method rather than the second best is 0.038 R<super>2</super>; "
               "the best achievable R<super>2</super> ranges from &#8722;0.232 to +1.000, a "
               "span of 1.232. The data therefore accounts for roughly 33 times more "
               "variation than the method does."))
    s.append(P("Counting winners makes the same point differently. Ridge regression is the "
               "most frequent winner at 14 of 47; persistence wins 10 and the training mean "
               "8, so the trivial baselines together take 18. A neural network wins on 9."))
    s.append(P("<b>One split must be made before any of this is read as a claim about "
               "architectures.</b> Nineteen of the 47 datasets are tabular panels with no "
               "genuine time axis, included deliberately to see whether the pipeline would "
               "notice. It does: median best R<super>2</super> is 0.029 for those against "
               "0.880 for the 28 genuinely temporal datasets, and a neural network wins 1 of "
               "19 against 8 of 28. Asking a sequence question of data with no sequence "
               "yields nothing, correctly, and those datasets are excluded from any "
               "architectural conclusion."))

    s.append(P("3.2 Neural networks tie or fail, rarely in between", H2))
    s.append(P("Across all 47 datasets the best neural network lands within 0.05 R<super>2"
               "</super> of the best available method on 18, and below R<super>2</super> = "
               "&#8722;0.5 &#8212; worse than predicting the training mean &#8212; on 16. "
               "Only 13 sit between. The distribution is bimodal, not graded."))
    s.append(P("The near-ties are genuinely near. On weekly CO<sub>2</sub> the network reaches "
               "0.984 against ridge's 0.992; on monthly sunspots 0.867 against 0.886. It did "
               "not fail; it simply did not beat a linear model, at considerably greater "
               "cost. The failures, by contrast, are total: R<super>2</super> = &#8722;1.000 "
               "on <i>ecoli</i> (335 rows) and &#8722;0.931 on <i>wine</i> (178 rows), where "
               "a trivial baseline reaches 0.276 and 0.467."))
    s.append(P("Which side a dataset falls on tracks its size. Above 5,000 rows the median "
               "gap to the best method is 0.000 and one of nine datasets blows up; below "
               "1,500 rows it is close to a coin flip. Rows per dimension separates the two "
               "groups more sharply still: a median of 274 where the network survived against "
               "81 where it inverted. Coverage predicts the gap at &#961; = +0.62."))
    s.append(P("This is the practical motivation for the rest of the paper. A method that "
               "degrades gently is safe to try; one that goes from tying the best to worse "
               "than the mean, with no warning, is not."))

    s.append(PageBreak())

    s.append(P("3.3 The achievable ceiling is predictable without training", H2))
    s.append(P("For each dataset we take the best R<super>2</super> achieved by any of the "
               "eight methods and correlate it against each training-free property."))
    s.append(Spacer(1, 3))
    s.append(table([
        ["Property", "Spearman ρ", "p"],
        ["Estimated noise floor", "−0.871", "< 0.0001"],
        ["Spectral entropy", "−0.739", "< 0.0001"],
        ["Decorrelation time", "+0.662", "< 0.0001"],
        ["Coverage", "−0.461", "0.001"],
        ["Dimension", "−0.403", "0.005"],
        ["Length (rows)", "+0.356", "0.014"],
        ["Distribution shift", "+0.072", "0.63"],
    ], [7.2 * cm, 3.2 * cm, 2.6 * cm]))
    s.append(P("Table 1: Training-free properties against the best R<super>2</super> any "
               "method achieved, 47 datasets.", CAP))
    s.append(P("The noise-floor estimate alone accounts for most of it. Decorrelation time is "
               "the sharpest categorical signal: a decorrelation time of one step &#8212; "
               "consecutive rows independent &#8212; appears on every dataset where nothing "
               "works, regardless of method."))
    s.append(P("<b>Distribution shift is the exception, and it is informative.</b> It does "
               "not predict the ceiling at all (p = 0.63). In a separate controlled "
               "experiment we added irrelevant columns of four kinds to a fixed dynamical "
               "system. Non-stationary columns &#8212; those whose distribution moves between "
               "train and test by 0.83 of their own standard deviation, against 0.01&#8211;"
               "0.03 for the others &#8212; raised the network's error by a factor of 896 and "
               "produced a generalisation gap of 38,415, while leaving a lookup baseline "
               "untouched. Non-stationarity does not lower what is achievable; it destroys "
               "the model you fitted. These are different failures needing different "
               "responses, and only the first is a property of the data."))

    s.append(P("3.4 A selector that transfers across domains", H2))
    s.append(P("We fit a random forest to predict which of the three families achieves the "
               "best R<super>2</super>, using only the seven properties. It is scored by "
               "regret &#8212; the R<super>2</super> lost against the best method actually "
               "available &#8212; under two validation schemes: leave-one-dataset-out, and "
               "the far harsher leave-one-domain-out, in which every dataset of a kind is "
               "held out together."))
    s.append(Spacer(1, 3))
    s.append(table([
        ["Strategy", "Regret (dataset held out)", "Regret (domain held out)"],
        ["Oracle (unreachable)", "0.000", "0.000"],
        ["Selector, from data properties", "0.040", "0.038"],
        ["Always trivial", "0.063", "0.063"],
        ["Always classical (majority)", "0.119", "0.119"],
        ["Random choice", "0.201", "0.201"],
        ["Always neural", "0.419", "0.419"],
    ], [6.4 * cm, 4.3 * cm, 4.3 * cm]))
    s.append(P("Table 2: Mean regret, lower is better. The selector beats the best fixed "
               "strategy by 1.6&#215; and always-neural by 11&#215;.", CAP))
    s.append(P("The result that matters is that domain-holdout regret (0.038) is no worse "
               "than dataset-holdout regret (0.040). This was our pre-registered failure "
               "condition: had the selector only worked because it had seen four other "
               "chaotic systems, it would have learned the collection rather than a property "
               "of data. Paired Wilcoxon tests against the fixed strategies give p = 0.013 "
               "(always trivial), 0.006 (always classical) and < 0.0001 (always neural)."))
    s.append(P("Family accuracy is 60% against a 33% chance baseline, which is modest; regret "
               "is the honest metric and is where the result lives. The winning method's "
               "identity also flips across seeds on 21 of 47 datasets, so this is a regime "
               "recommendation rather than a per-dataset oracle."))

    s += fig("03_model_selector.png",
             "Figure 2: Regret against fixed strategies. Middle panel is the leave-one-"
             "domain-out test. Right panel shows the two properties that separate the "
             "regimes most cleanly.")

    s.append(P("The learned rule is legible. A depth-3 tree splits first on the noise floor, "
               "then on coverage, then on spectral entropy or distribution shift. Feature "
               "importances are spread &#8212; coverage 0.21, noise floor 0.18, dimension "
               "0.14, rows 0.14, spectral entropy 0.14, shift 0.11, decorrelation 0.09 "
               "&#8212; which is why single correlations look weak while the combination does "
               "not. The regimes have distinct profiles: where a neural network wins, the "
               "median decorrelation time is 16 steps and coverage 0.077; where a trivial "
               "baseline wins, 2 steps and 0.747."))

    # ---------------------------------------------------------------- 4 -----------------
    s.append(P("4. Limitations", H1))
    s.append(P("<b>Scale.</b> Every result was measured at 1&#8211;60 dimensions, "
               "150&#8211;42,000 rows and 10<super>4</super>&#8211;10<super>5</super> "
               "parameters, on one prediction task with a fixed window. The claim that "
               "architecture is second-order is measured only where every method is small, "
               "and is the claim most likely to invert at scale."))
    s.append(P("<b>The noise-floor estimator degrades with dimension.</b> Against synthetic "
               "data with a known ceiling of 0.80 it reads 0.80 at four dimensions, 0.68 at "
               "eight and 0.42 at 32, always pessimistically, because it requires "
               "near-identical inputs and in high dimensions nothing is near anything. Above "
               "roughly 16 dimensions it should be read as a lower bound. This is our most "
               "useful single result and its most exposed one."))
    s.append(P("<b>Selector size.</b> The selector is fitted on 47 points. Domain holdout is "
               "the strongest available test at that size, but it is still 47."))
    s.append(P("<b>Not tested.</b> Multi-task, action-conditioned and representation-learning "
               "settings. A model whose predictions influence the data it later sees is "
               "outside everything measured here."))

    # ---------------------------------------------------------------- 5 -----------------
    s.append(P("5. Related work", H1))
    s.append(P("That simple baselines often match or beat deep models on time series is well "
               "established, from the M-competitions through to linear models outperforming "
               "transformers on standard forecasting benchmarks. Our contribution is not that "
               "observation but the question downstream of it: <i>which measurable property "
               "of the data decides it</i>, and does that property transfer to a domain the "
               "predictor has never seen. Analogue forecasting, and the classical result that "
               "natural analogues become vanishingly rare as dimension grows, supplies the "
               "mechanism behind our coverage measure. Work on dataset difficulty and "
               "learning-curve extrapolation shares our target but generally requires partial "
               "training; the properties here require none."))

    # ---------------------------------------------------------------- 6 -----------------
    s.append(P("6. Conclusion", H1))
    s.append(P("On this evidence the dominant practical decision is not which network to "
               "build but whether to build one, and that question is answerable in seconds "
               "from the data. The ceiling is predictable at &#961; = &#8722;0.87 without "
               "training; family selection beats every fixed strategy and survives domain "
               "holdout; and the reason it is worth doing is that neural networks in this "
               "regime fail catastrophically rather than gracefully."))
    s.append(P("Two recommendations follow for practice. Report a training-free baseline "
               "alongside any surrogate result &#8212; across 47 datasets a trivial or "
               "classical method won 38 times. And report the data regime alongside the "
               "error, because two results with the same R<super>2</super> in different "
               "regimes have not demonstrated the same thing."))

    # ---------------------------------------------------------------- back matter -------
    s.append(P("Reproducibility", H1))
    s.append(P("All code, raw per-run results and figures are in the project repository. "
               "Every experiment writes one JSONL record per unit of work carrying its "
               "configuration hash and git commit, so any number here traces to a run. "
               "Datasets are fetched from public mirrors and cached; failures are reported "
               "rather than dropped, so a stated dataset count matches what ran. Total "
               "compute for the study reported here is 80 minutes on two CPU cores.", BODY))

    s.append(P("AI involvement", H1))
    s.append(P("This work was carried out with an AI assistant (Claude) acting as the "
               "research agent under human direction. The assistant wrote the experiment "
               "code, ran the experiments, produced the figures and drafted this paper. The "
               "human author supplied the problem statement, chose the research direction at "
               "each decision point, supplied the reference material, and directed several "
               "changes of course &#8212; including the reframing from architecture "
               "comparison to model selection that this paper reports. All numerical claims "
               "were produced by code in the repository rather than generated by the model. "
               "Seven measurement artefacts were found and corrected during the work, four of "
               "them in our own instruments; each correction is recorded in the repository "
               "history alongside the result it changed.", BODY))

    doc.build(s)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
