# Business case: Salary Predictor

**A first salary estimate for a job posting, from role, degree, industry and experience.**

## Who buys this

The buyer is a talent-acquisition or compensation lead at a recruiting agency, job board or employer that writes many postings. The users are recruiters who need a starting salary range before a compensation specialist reviews it.

## The problem and what it costs

Recruiters often guess at a range for a new posting, then wait for a compensation review. A model that gives a quick first estimate shortens that loop. I didn't find a reliable figure for the time recruiters spend on this, so the model below uses an assumption.

## How the repo solves it

| Need | What in the repo does it |
|---|---|
| Estimate a salary | A gradient-boosting model behind `POST /predict` in a FastAPI service |
| See the effect of experience | The React app plots how this exact posting's pay changes from 0 to 24 years of experience |
| Compare with the market | Average salaries in the training data by job type, industry, degree or major |
| Reproducible model | The model is trained during `docker build` from the committed CSVs (1M rows), so no binary model is committed |
| Original analysis | `Salary Prediction Project.ipynb` and `docs/ANALYSIS.md` |

## Use-case scenarios

All of these organisations are fictional.

1. **A staffing agency** gives each recruiter a draft range for a new posting, which the compensation team then confirms.
2. **A job board** shows employers an estimated range when they leave the salary field blank.
3. **An HR team** checks how much a degree requirement adds to a posting's expected pay before deciding to keep it.

## ROI: a worked model (illustrative)

These figures are **illustrative assumptions** unless a source is linked.

| Input | Value | Basis |
|---|---|---|
| Postings a year | 2,000 | Assumption |
| Minutes saved per posting on the first range | 15 | Assumption |
| Recruiter loaded cost an hour | US$49 | Assumption |

Saving: 2,000 × 15 min = 500 hours, about **US$24,500 a year**. Small, and it relies on the model being trained on real salary data first.

## Path to production

1. Retrain on your own or licensed salary data. The training data here is a synthetic teaching set.
2. Show a range with uncertainty, not a single number.
3. Audit for bias by group, and keep a human compensation review on every posting.

## Risks and limits

- Pay predictions trained on historical data can carry past pay gaps forward. That is a legal risk under equal-pay laws, not only a quality issue.
- The `companyId` values and salaries come from a synthetic dataset, so there is no real-world accuracy claim.

**Case strength:** **Weak to moderate.** The workflow is real, but the dataset is synthetic and salary models need careful fairness checks before use.

## Sources

- U.S. Bureau of Labor Statistics, OEWS May 2024 (context for wage data): https://www.bls.gov/news.release/archives/ocwage_04022025.pdf
