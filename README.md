# Machine Learning Implementation for Waste Management Optimization

This document provides a comprehensive explanation of the machine learning implementation used in the GIS-ML-WasteNetwork project for optimizing waste management operations.

## 1. Training Dataset Creation

The system begins by creating a robust training dataset through the following steps:

### 1.1 Point Generation and Sampling
```python
def create_training_dataset(nyeri_gdf, buffer_sets, raster_criteria, n_points=10000):
    # Generate random points within study area
    points = []
    for point in generated_points:
        point_info = {
            'geometry': point,
            'x': point.x,
            'y': point.y
        }
        # Sample values from each buffer zone and raster
```

Key features:
- Generates random points within study area
- Samples values from buffer zones and raster criteria
- Validates points against study area boundaries
- Handles batch processing for large datasets

### 1.2 Feature Engineering

The system processes several key geographical features with specific weights:
```python
weights = {
    'River': 0.25,        # 25% - Environmental impact
    'Road': 0.25,         # 25% - Accessibility
    'Settlement': 0.20,   # 20% - Population considerations
    'Soil': 0.10,         # 10% - Ground stability
    'Protected_Areas': 0.10,
    'Land_Use': 0.05,
    'Slope': 0.05
}
```

Buffer distances for each feature:
```python
buffer_distances = {
    'River': [300, 1000, 1500, 2000],
    'Road': [400, 800, 1000, 1200],
    'Settlement': [400, 900, 1500, 2100],
    'Protected_Areas': [300, 1000, 1500, 2000, 2500]
}
```

## 2. Machine Learning Model

### 2.1 Model Architecture

The system uses a Random Forest Regressor with optimized hyperparameters:
```python
class LandfillSuitabilityML:
    def __init__(self, session_id, socketio):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
```

### 2.2 Feature Processing

Data preprocessing includes:
```python
def prepare_training_data(self, features_dict):
    # Normalize features
    X_scaled = self.scaler.fit_transform(X)
    
    # Create target variable using weighted criteria
    y = self.create_target_variable(X)
```

### 2.3 Model Training and Refinement

The training process includes hyperparameter optimization:
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    self.model, param_grid, cv=5, scoring='r2', n_jobs=-1
)
```

## 3. Clustering Analysis

The system implements a sophisticated two-stage clustering approach for optimizing collection points:

### 3.1 Initial K-means Clustering
```python
def perform_clustering_analysis(grid_gdf):
    # Extract features from highly suitable areas
    features = grid_gdf[grid_gdf['Suitability_Class'] == 'Highly Suitable']
    X = np.array([[geom.centroid.x, geom.centroid.y] for geom in features.geometry])
    scores = features['combined_score'].values.reshape(-1, 1)
    
    # Normalize and combine features
    X_scaled = scaler.fit_transform(X)
    scores_scaled = scaler.fit_transform(scores)
    clustering_features = np.hstack([X_scaled, scores_scaled])
    
    # Perform K-means
    n_clusters = min(5, len(features))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(clustering_features)
```

### 3.2 DBSCAN Refinement
```python
for k in range(n_clusters):
    mask = kmeans_labels == k
    cluster_points = X[mask]
    
    # Calculate adaptive parameters
    distances = pdist(cluster_points)
    eps = np.percentile(distances, 10) if len(distances) > 0 else 50
    min_samples = max(3, int(np.sum(mask) * 0.1))
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    db_labels = dbscan.fit_predict(cluster_points)
```

## 4. Network Analysis

The system performs advanced network analysis for route optimization:

### 4.1 Network Construction
```python
def perform_network_analysis(nyeri_gdf, collection_points_gdf, landfill_sites_gdf):
    # Download or fetch road network
    G = ox.graph_from_bbox(
        north=bounds[3], south=bounds[1],
        east=bounds[2], west=bounds[0],
        network_type='drive',
        simplify=True
    )
```

### 4.2 Route Optimization
- Calculates optimal routes between collection points and landfills
- Uses Dijkstra's algorithm for shortest paths
- Considers road types and traffic patterns
- Handles real-time updates

### 4.3 Collection Point Analysis
```python
def create_markers_map(grid_gdf, aoi_gdf):
    # Filter highly suitable areas
    highly_suitable = grid_gdf[grid_gdf['Suitability_Class'] == 'Highly Suitable']
    
    # DBSCAN clustering for collection points
    coords = np.array([[geom.centroid.x, geom.centroid.y] 
                      for geom in highly_suitable.geometry])
    
    clustering = DBSCAN(
        eps=50,  # 50 meters clustering distance
        min_samples=1
    ).fit(coords)
```

## 5. Database Integration

The system uses PostgreSQL with PostGIS for spatial data storage:

### 5.1 Collection Points Schema
```python
class CollectionPoint(Base):
    __tablename__ = 'collection_points'
    id = Column(Integer, primary_key=True)
    point_id = Column(Integer, unique=True)
    description = Column(String)
    geom = Column(Geometry('POINT', srid=21037))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
```

### 5.2 Route Storage
- Stores optimized routes between points
- Maintains spatial relationships
- Handles real-time updates

## 6. Output Generation

The system generates several outputs:

### 6.1 Suitability Maps
- Continuous suitability scores
- Classified suitability zones
- Collection point locations

### 6.2 Analysis Reports
- Feature importance plots
- Model performance metrics
- Spatial distribution statistics

### 6.3 Route Visualizations
- Interactive web maps
- Route statistics
- Collection point clusters

## 7. Performance Optimization

The system includes several optimizations:
- Batch processing for large datasets
- Spatial indexing for efficient queries
- Adaptive parameter selection
- Real-time progress updates

## Dependencies

Required Python packages:
- scikit-learn
- numpy
- pandas
- geopandas
- networkx
- osmnx
- rasterio
- SQLAlchemy
- GeoAlchemy2
- eventlet
