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

Start Docker Desktop or Docker Engine and open a terminal at the repository root, where the `Dockerfile` is located. The build installs the locked dependencies and trains the model from the included CSV. Allow a few minutes for the first build to download its dependencies and finish training.

```bash
docker build -t shopping-purchase-prediction .
docker run --rm -p 127.0.0.1:9696:9696 shopping-purchase-prediction
```

Keep the container running and wait for Uvicorn to report that it has started. Open the [interactive API docs](http://127.0.0.1:9696/docs), or run the commands below in a second terminal at the repository root.

```bash
curl --fail-with-body http://127.0.0.1:9696/health
curl --fail-with-body http://127.0.0.1:9696/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/session.json
```

The health check should return `status: ok`. The example prediction should return a `purchase_probability` between 0 and 1 and the model name. Press **Ctrl+C** in the first terminal when finished. If port 9696 is already in use, change the mapping to `-p 127.0.0.1:9697:9696` and use port 9697 in the browser and requests.

The example is synthetic and is also prefilled in the interactive docs. The full request schema includes the seven page and calendar numbers, the four integer category identifiers, month, visitor type, and the weekend flag. Unknown positive category codes are accepted by the encoder and mapped to all-zero indicator columns for that field. The API returns HTTP 422 for malformed input.

[Verified Docker run](examples/docker-demo.png) showing a health check, a prediction, and HTTP 422 for an invalid request.

## Cloud deployment on Render

The FastAPI service is live on Render's Free plan. Open the [interactive API docs](https://shopping-purchase-prediction.onrender.com/docs), expand `POST /predict`, choose **Try it out**, and execute the prefilled example. The first request after an idle period can take a minute to start.

The deployed service returned HTTP 200 for the health check and predicted a purchase probability of 0.2762 for the synthetic example. [Verified Render run](examples/render-demo.png)

### Recreate the service

The [Render Blueprint](render.yaml) defines a Free Docker web service that listens on port 10000 and uses `/health` for readiness checks.

1. Push the project to a GitHub repository you control, with `Dockerfile` and `render.yaml` at the repository root. For a private repository, connect GitHub in Render and select that repository in the Render GitHub app's **Only select repositories** setting. If it is missing from the Render repository list, check the app's repository access. [GitHub connection guide](https://render.com/docs/git-provider#fixing-github-app-permissions)
2. In the Render Dashboard, choose **New → Blueprint** and click **Connect** next to your repository. Enter a Blueprint name, select the branch you pushed (for example, `main`), and leave **Blueprint Path** at its default, `render.yaml`. [Render Blueprint setup](https://render.com/docs/infrastructure-as-code#setup)
3. Confirm the preview lists the `shopping-purchase-prediction` web service and click **Deploy Blueprint**. Open the service's **Deploys** page and wait for the status to become **Live**. The Dockerfile installs dependencies and trains the model during the build.
4. Copy the service URL Render assigned and open `<service-url>/health`. Confirm the response has `status: ok`, then open `<service-url>/docs` and execute the prefilled `POST /predict` example. For the submission, save the working URL or an image of the successful response.
