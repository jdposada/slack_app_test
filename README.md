# OMOP 5.4 Slack App

This repo contains a Slack slash-command app that answers questions about OMOP CDM v5.4 from the official OHDSI CommonDataModel documentation set.

The runtime is intentionally simple:
- A build step crawls the approved `ohdsi.github.io/CommonDataModel/` pages relevant to OMOP 5.4.
- The crawler parses headings, narrative sections, and schema tables into structured chunks.
- A packaged SQLite FTS5 index is built offline and shipped with the container image.
- Slack commands retrieve the best matching chunks and send only that evidence to Vertex AI for synthesis.

There is no vector database, no GCS-hosted index, and no backward-compatibility path to the old FAISS architecture.

## Supported source set

The index builder stays within `https://ohdsi.github.io/CommonDataModel/` and fetches the OMOP 5.4 doc set, including:
- `cdm54.html`
- `cdm54Changes.html`
- `cdm54erd.html`
- `cdm54ToolingSupport.html`
- `dataModelConventions.html`
- `cdmPrivacy.html`
- `customConcepts.html`
- `download.html`
- `cdmRPackage.html`
- `drug_dose.html`
- `cdmDecisionTree.html`
- `faq.html`
- `sqlScripts.html`
- `contribute.html`

External pages such as forums, GitHub, Athena, demos, and Themis are excluded as answer sources.

## Commands

- `/omop54 <question>`: normal ephemeral answer with detailed citations
- `/omop54-debug <question>`: ephemeral answer plus retrieval diagnostics when `ENABLE_DEBUG_COMMAND=true`

Each answer includes a `Sources` section with page title, table name, field name when relevant, and a canonical URL.

## Examples

Normal command:

```text
/omop54 What is the PERSON table used for?
```

Example answer:

```text
The PERSON table is the central identity table for people in the OMOP CDM and stores one record per person with demographic information.

Sources:
- OMOP CDM v5.4 | table PERSON | Table Description | https://ohdsi.github.io/CommonDataModel/cdm54.html#person
- OMOP CDM v5.4 | table PERSON | User Guide | https://ohdsi.github.io/CommonDataModel/cdm54.html#person
```

Field-level question:

```text
/omop54 Is gender_concept_id required in PERSON?
```

Example answer:

```text
Yes. In the PERSON table, `gender_concept_id` is required and is a foreign key to the CONCEPT table in the Gender domain.

Sources:
- OMOP CDM v5.4 | table PERSON | field gender_concept_id | Field Specification | https://ohdsi.github.io/CommonDataModel/cdm54.html#person
```

Change question:

```text
/omop54 What changed in VISIT_OCCURRENCE in OMOP 5.4?
```

Example answer:

```text
In OMOP 5.4, VISIT_OCCURRENCE renamed `admitting_source_concept_id` to `admitted_from_concept_id`, renamed `admitting_source_value` to `admitted_from_source_value`, and renamed the discharge fields to `discharged_to_*`.

Sources:
- Changes by Table | table VISIT_OCCURRENCE | Changes | https://ohdsi.github.io/CommonDataModel/cdm54Changes.html#visit_occurrence
```

Debug command:

```text
/omop54-debug Is gender_concept_id required?
```

Example debug response:

```text
Yes. The documentation marks `gender_concept_id` as required.

Sources:
- OMOP CDM v5.4 | table PERSON | field gender_concept_id | Field Specification | https://ohdsi.github.io/CommonDataModel/cdm54.html#person

Debug diagnostics:
- score=178.4; overlap=2; reason=exact field match, fts lexical match; source=https://ohdsi.github.io/CommonDataModel/cdm54.html#person
  excerpt=CDM Field: gender_concept_id User Guide: This field is meant to capture the biological sex at birth of the Person. ...
```

## Environment variables

Required:
- `SLACK_SIGNING_SECRET`
- `GOOGLE_CLOUD_PROJECT`

Optional:
- `VERTEX_AI_LOCATION` default `us-central1`
- `VERTEX_AI_MODEL` default `gemini-2.5-flash`
- `OMOP_INDEX_PATH` default `data/omop54.db`
- `ENABLE_DEBUG_COMMAND` default `false`
- `TOP_K_DOCUMENTS` default `4`
- `SOURCE_COUNT` default `3`
- `PORT` default `8080`

## Local development

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

Build the packaged OMOP index:

```bash
python ingest.py --output data/omop54.db
```

Set runtime configuration:

```bash
export SLACK_SIGNING_SECRET=...
export GOOGLE_CLOUD_PROJECT=your-project
export VERTEX_AI_MODEL=gemini-2.5-flash
export OMOP_INDEX_PATH=data/omop54.db
export ENABLE_DEBUG_COMMAND=true
```

Run locally:

```bash
python app.py
```

Expose the app to Slack during development:

```bash
ngrok http 8080
```

Configure Slack slash commands to point at:
- `/omop54` -> `https://<your-url>/slack/events`
- `/omop54-debug` -> `https://<your-url>/slack/events`

## Docker and Cloud Run

The Docker build installs dependencies and runs the offline index build so the image already contains `/app/data/omop54.db`.

Build:

```bash
docker build -t omop54-slack-app .
```

Deploy to Cloud Run:

```bash
gcloud run deploy omop54-slack-app \
  --image gcr.io/YOUR_PROJECT/omop54-slack-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=YOUR_PROJECT,OMOP_INDEX_PATH=/app/data/omop54.db" \
  --update-secrets "SLACK_SIGNING_SECRET=slack-signing-secret:latest"
```

For production, use Secret Manager for Slack credentials.

## GitHub Actions deployment

The repo includes [.github/workflows/deploy.yml](/Users/alvaro1/Documents/Coral/Code/ChatOmop/.github/workflows/deploy.yml).

Behavior:
- Pull requests to `main` run the test suite.
- Pushes to `main` run the test suite, build the container image, push it to Artifact Registry, and deploy the Cloud Run service.

The workflow is currently wired for this deployment target:
- GCP project: `form-inspector`
- Artifact Registry repo: `omop54-slack-app`
- Cloud Run service: `omop54-slack-app`
- Runtime service account: `omop54-run@form-inspector.iam.gserviceaccount.com`

Before it can deploy from GitHub, configure these GitHub repository secrets:
- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_DEPLOY_SERVICE_ACCOUNT`

Recommended values:
- `GCP_WORKLOAD_IDENTITY_PROVIDER`: full Workload Identity Provider resource name, for example `projects/123456789/locations/global/workloadIdentityPools/github/providers/github`
- `GCP_DEPLOY_SERVICE_ACCOUNT`: deployer service account email, for example `github-actions-deployer@form-inspector.iam.gserviceaccount.com`

Recommended deployer service account roles:
- `roles/run.admin`
- `roles/artifactregistry.writer`
- `roles/iam.serviceAccountUser` on `omop54-run@form-inspector.iam.gserviceaccount.com`
- `roles/secretmanager.secretAccessor`

The workflow deploys the same runtime configuration used in production, including:
- Vertex AI model and region
- packaged OMOP SQLite index path
- Secret Manager reference for `SLACK_SIGNING_SECRET`
- `--min-instances=1` to reduce Slack cold-start timeouts

## Tests

Run:

```bash
pytest
```
