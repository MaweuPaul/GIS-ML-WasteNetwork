# GIS-ML-WasteNetwork

A GIS-driven decision-support tool for siting municipal waste infrastructure and optimizing collection routes, built as a final-year research project around Nyeri, Kenya. It combines multi-criteria suitability analysis, machine learning, and network routing behind a React frontend: upload an area of interest as a shapefile (converted to GeoJSON, with real coordinates and a bounding box), run the pipeline, and get back a ranked suitability surface, clustered candidate collection points, and optimized routes to existing landfills.

## Table of Contents
- [What Is This?](#what-is-this)
- [Why It's Built This Way](#why-its-built-this-way)
- [Running It](#running-it)
- [Repository Layout](#repository-layout)
- [The Suitability & Routing Pipeline](#the-suitability--routing-pipeline)
- [Contributing](#contributing)
- [Roadmap / To Be Done](#roadmap--to-be-done)
- [Disclaimer](#disclaimer)

## What Is This?

Deciding where a new landfill or waste-collection point should go is normally done by a planner overlaying constraint layers, rivers, roads, settlements, protected areas, soil, slope, land use, on a map and judging by eye whether a candidate site is "far enough" from each. That judgment is hard to defend or repeat: two planners can disagree, and nobody can rerun last year's decision to check it against new data. This project turns the same process into code: every constraint is buffered at defined distances, weighted by AHP-derived importance, and fed into a trained model that scores the whole study area, so "this site is suitable" becomes a reproducible number instead of an opinion.

The output isn't just a single best site. The pipeline produces a continuous suitability surface, classifies it into suitability bands, clusters the highest-scoring cells into a practical number of candidate collection points, and then routes between those points and existing landfills over the real road network, so the result is something a municipality could actually act on rather than just a heatmap.

## Why It's Built This Way

A few structural choices here aren't obvious from the code alone, so it's worth writing down the reasoning:

**Three separate services instead of one monolith.** The API service (Node/Express + Prisma + PostGIS) owns spatial CRUD, the frontend (React) is purely presentational, and the ML service (Flask + Socket.IO) does the actual analysis. Splitting them this way means the heavy, long-running spatial computation (network downloads, clustering, model inference) lives in one place with its own process model, and can emit live progress over its own socket connection without the request/response API needing to know anything about job state.

**Every constraint layer is its own table, not one generic "layer" table.** Soil, rivers, roads, settlements, geology, protected areas, and land use each have their own Prisma model and controller, even though they're structurally similar. Keeping them separate means each one's specific attributes (soil type, road class, protection designation) are first-class columns instead of buried in a generic JSON blob, which matters when the suitability pipeline needs to query and buffer each layer differently.

**Raw SQL for the spatial writes, Prisma for everything else.** PostGIS operations like `ST_SetSRID(ST_GeomFromGeoJSON(...))` aren't expressible through Prisma's query builder, so every controller that writes geometry drops down to `$queryRaw`/`$executeRaw` with tagged-template parameters for that one insert or update, then goes back to the ORM for reads and simple CRUD. That keeps the geometry-handling code isolated and easy to find rather than mixed through every query.

**Socket.IO for the ML pipeline, plain REST for everything else.** A suitability run can take minutes (training-data generation, model fit, clustering, then an OSM network download and routing pass), so the frontend needs incremental progress rather than a single blocking response. Everything else, layer CRUD, incident reports, special pickups, is a normal request/response and stays plain REST.

**Two-stage clustering (K-means then DBSCAN) instead of one algorithm.** K-means alone forces a fixed number of clusters and doesn't respect the actual shape of high-suitability regions; DBSCAN alone is sensitive to density variation across a whole study area. Running K-means first to get a manageable number of regional groups, then DBSCAN within each group with adaptively chosen `eps`/`min_samples`, gets clusters that are both regionally sensible and locally well-shaped.

## Running It

```bash
# API service (Express + Prisma + PostGIS)
cd API-SERVICE
npm install
npx prisma generate
npm start

# Frontend (React + Vite)
cd FRONTEND
npm install
npm run dev

# ML service (Flask + Socket.IO)
cd MACHINE-LEARNING-SERVICE
pip install -r requirements.txt
python ap.py
```

The API service needs a `DATABASE_URL` environment variable pointing at a PostGIS-enabled Postgres database (see `API-SERVICE/prisma/schema.prisma` for the schema Prisma expects). Open the frontend's printed URL, upload an area-of-interest shapefile (and the other constraint layers, also as shapefiles/rasters), and start a run from there.

## Repository Layout

```
GIS-ML-WasteNetwork/
├── API-SERVICE/                Express + Prisma + PostGIS
│   ├── Controllers/            One controller per spatial layer (soil, rivers, roads, ...)
│   ├── Routes/                 One route file per controller
│   ├── prisma/schema.prisma    DB schema for every layer + AOI + incidents
│   └── server.js               App entrypoint, mounts every /api/* route
├── FRONTEND/                   React (Vite) UI: layer upload + results viewer
│   └── src/
│       ├── PAGES/
│       └── components/
├── MACHINE-LEARNING-SERVICE/   Flask + Flask-SocketIO
│   ├── ap.py                   App entrypoint, kicks off/streams a pipeline run
│   ├── utils/
│   │   ├── ahp.py                     AHP weighting for constraint layers
│   │   ├── create_training_dataset.py Random-point sampling across buffers + rasters
│   │   ├── feature_preparation.py     Feature engineering for the model
│   │   ├── ml_analysis.py             RandomForestRegressor training/scoring
│   │   ├── grid_analysis.py           K-means + DBSCAN clustering
│   │   ├── network_analysis.py        Route optimization over the OSM road graph
│   │   ├── suitability_mapping.py     Suitability surface + classified output
│   │   ├── predict_suitability.py     Inference with a saved model
│   │   └── database.py                PostGIS read/write helpers
│   └── models/train_model.py   Standalone training script
└── docs/ML_DOCUMENTATION.md    Full write-up of the ML pipeline (weights, buffers, model, clustering, routing)
```

## The Suitability & Routing Pipeline

The full technical detail, feature weights, buffer distances per constraint, the Random Forest architecture and hyperparameter search, the clustering math, and the network-routing step, is documented separately in [docs/ML_DOCUMENTATION.md](docs/ML_DOCUMENTATION.md) rather than duplicated here.

## Contributing

Contributions are welcome. If you're adding a new constraint layer or changing a weight/buffer distance, update both the ML service's `utils/` code and `docs/ML_DOCUMENTATION.md` in the same change, the weights and buffers documented there are meant to always match what the code actually uses. If you're changing the API, keep the pattern of one controller/route pair per spatial layer rather than introducing a generic layer abstraction, that's a deliberate choice, see [Why It's Built This Way](#why-its-built-this-way).

## Roadmap / To Be Done

- **Persisted job history.** Runs currently exist only as files in the ML service's `output/` folder for the lifetime of that process; there's no record in the database of past runs, their parameters, or their results to compare against later.
- **Configurable weights and buffers from the UI.** The AHP weights and buffer distances are currently constants in the ML service's Python code; exposing them as run-time parameters from the frontend would let a user test different assumptions without editing code.
- **Automated retraining.** The suitability model is trained via a standalone script (`models/train_model.py`) and saved to disk; there's no pipeline to retrain it as new constraint data is added to the database.

## Disclaimer

This is academic research code built and assessed as a final-year project around a specific study area (Nyeri, Kenya), not a continuously maintained product. Suitability results are only as accurate and current as the GIS layers (rivers, roads, settlements, soil, protected areas, land use) loaded into the database for whatever area you point it at. The AHP criteria weights themselves are grounded in published landfill-siting literature, not arbitrary guesses, see [docs/ML_DOCUMENTATION.md](docs/ML_DOCUMENTATION.md) for the full breakdown. Treat every output as a decision-support input for further human review, not a final siting decision.
