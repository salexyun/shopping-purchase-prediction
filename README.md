# Shopping Purchase Prediction

An e-commerce analytics team can use this model to rank shopping sessions by their estimated likelihood of ending in a purchase. A reviewer with limited time could start with the highest-scoring sessions and look for patterns in the shopping funnel. Each API request supplies one session summary and receives a score from 0 to 1.

This is an ML Zoomcamp midterm project. It covers problem framing and data exploration, a linear classifier, evaluation on an imbalanced target, a prediction API in Docker, and tuned tree ensembles. The [executed notebook](notebook.ipynb) shows the analysis and model selection. [ML Zoomcamp project rubric](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/projects)

## Data

The [original UCI CSV](data/online_shoppers_intention.csv) is included and has 12,330 sessions. The binary target is `Revenue`. There are 1,908 purchase sessions (15.5%). The included file is unchanged and has SHA-256 `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`.

C. Sakar and Y. Kastro (2018), *Online Shoppers Purchasing Intention Dataset*, UCI Machine Learning Repository. [Dataset and variable definitions](https://doi.org/10.24432/C5F88Q) · [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)

The 14 inputs describe page counts, browsing durations, calendar context, and coded session attributes. `PageValues` uses transaction value, so it is excluded. `BounceRates` and `ExitRates` are also excluded to keep the service independent of exit analytics. Device, browser, region, and traffic codes are treated as categories. The notebook checks missing values, numerical ranges, repeated profiles, class balance, and differences across shopping sessions.

## Results

We held out 2,466 sessions and used three-fold cross-validation on the remaining 9,864. Identical predictor profiles are kept together at both stages. Each cross-validation fold fits its own preprocessing. The primary metric is average precision (AP), which measures the precision–recall trade-off as we move through the ranked sessions.

| Model | Mean CV AP | CV AP standard deviation | Mean CV ROC AUC |
| --- | ---: | ---: | ---: |
| Prior-only baseline | 0.155 | 0.000 | 0.500 |
| Logistic regression | 0.300 | 0.017 | 0.723 |
| Histogram gradient boosting | 0.358 | 0.018 | 0.767 |
| Random forest | **0.371** | 0.022 | 0.766 |

The selected forest reached **0.321 AP** and **0.750 ROC AUC** on the holdout. The purchase rate was **0.155**, which is the AP of a constant score on the same holdout. At a threshold of 0.5, the model identified 8 of 382 purchases and flagged 14 sessions that did not convert. That threshold gives poor recall, so the API returns a score for ranking. The notebook includes precision–recall and calibration plots. In its permutation analysis, month, visitor type, and time spent on product pages had the largest effects on average precision.

The table and threshold metrics are from the macOS notebook run. The Linux Docker build retrains from the same data and lockfile, selected a different maximum depth for the random forest, and reached 0.321 AP and 0.748 ROC AUC on the same holdout.

The dataset summarizes sessions from one site. It has no event sequence that lets us assess the model at a consistent moment before purchase. Live use would require training on features computed at the intended prediction time, followed by validation on later traffic from the target store. The score has not been calibrated for that population.

## Reproduce with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then open a terminal in this project folder. The checked-in `uv.lock` pins the dependency versions. `uv sync` creates the local `.venv`, and `uv run` runs commands inside it.

```bash
uv python install 3.12
uv sync --locked --group notebook
uv run --locked --group notebook jupyter lab
```

Open `notebook.ipynb` and run all cells. To execute it directly from the terminal, use `uv run --locked --group notebook jupyter nbconvert --to notebook --execute --inplace notebook.ipynb`. To activate the local environment manually on macOS or Linux, use `source .venv/bin/activate`.

Train and serve the model from a separate terminal. Training writes `artifacts/model.joblib` and `artifacts/model.metrics.json`.

```bash
uv run --locked python train.py
uv run --locked python predict.py
```

## Run in Docker

The image build uses the same uv lockfile, trains the model from the included CSV, and copies the fitted pipeline into the serving image.

```bash
docker build -t shopping-purchase-prediction .
docker run --rm -p 127.0.0.1:9696:9696 shopping-purchase-prediction
```

Open [interactive API docs](http://127.0.0.1:9696/docs), or send requests from a separate terminal in the project folder.

```bash
curl --fail-with-body http://127.0.0.1:9696/health
curl --fail-with-body http://127.0.0.1:9696/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/session.json
```

The example is synthetic and is also prefilled in the interactive docs. The full request schema includes the seven page and calendar numbers, the four integer category identifiers, month, visitor type, and the weekend flag. Unknown positive category codes are accepted by the encoder and mapped to all-zero indicator columns for that field. The API returns HTTP 422 for malformed input.

[Verified Docker run](examples/docker-demo.png) showing a health check, a prediction, and HTTP 422 for an invalid request.

## Optional cloud deployment on Render

The [Render Blueprint](render.yaml) defines a Docker web service on the Free plan with `/health` as the health check. Publish the project to GitHub, then in Render choose **New → Blueprint**, connect that repo, and apply the Blueprint. Use `render.yaml` as the Blueprint path if the project is at the repo root. If it is a subfolder of a coursework repo, the fastest setup is **New → Web Service**, select **Docker** and **Free**, set **Root Directory** to the project subfolder and **Health Check Path** to `/health`.

Once Render shows the service as live, open its `/docs` page and run the example. To capture proof for the submission, save the working service URL or a screenshot showing the successful request and response. Free instances can take a minute to start after idling. [Render Docker guide](https://render.com/docs/docker) · [Render free instance details](https://render.com/docs/free)
