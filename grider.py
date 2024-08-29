import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.io import MemoryFile
from rasterio.features import shapes
from rasterio.transform import from_origin
from pathlib import Path
from shapely.geometry import box, shape


# Define raster properties
pixel_size = 0.0083  # Size of each pixel in degrees (adjust this value as needed)
buffer_size = 0.139  # Buffer size in degrees to expand the raster bounds (adjust this value as needed)
grid_size = 0.3 # Size of each grid cell in degrees (adjust this value as needed)


def init_grid(exp):
    minx, miny, maxx, maxy = exp.gdf.total_bounds  # Get bounding box of the GeoDataFrame

    # Expand the bounds by the buffer size
    minx -= buffer_size
    miny -= buffer_size
    maxx += buffer_size
    maxy += buffer_size

    # Calculate the number of rows and columns for the raster
    nrows = int((maxy - miny) / pixel_size) + 1
    ncols = int((maxx - minx) / pixel_size) + 1

    # Define the transformation matrix
    transform = from_origin(minx, maxy, pixel_size, pixel_size)

    # Initialize the raster grid with NaNs (or zeros if appropriate)
    raster = np.full((nrows, ncols), np.nan)

    # Loop through each point in the GeoDataFrame to assign values to the raster
    for _, row in exp.gdf.iterrows():
        # Extract x (longitude) and y (latitude) from the geometry
        x, y = row.geometry.x, row.geometry.y

        # Calculate the column and row index for each point
        col = int((x - minx) / pixel_size)
        row_idx = int((maxy - y) / pixel_size)

        # Assign the value to the corresponding cell in the raster
        raster[row_idx, col] = row['value']

    # Write the raster to a GeoTIFF file using rasterio
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=nrows,
            width=ncols,
            count=1,  # Number of bands
            dtype=rasterio.float32,  # Data type for the raster values
            crs='EPSG:4326',  # Coordinate reference system
            transform=transform,
        ) as dataset:
            dataset.write(raster, 1)  # Write raster data to the first band
            
            # Read back the raster data from the in-memory file
            raster_data = dataset.read(1)
            transform = dataset.transform
            crs = dataset.crs
    
    # Generate grid cells over the bounding box
    x_coords = np.arange(minx, maxx, grid_size)
    y_coords = np.arange(miny, maxy, grid_size)

    grid_cells = []
    for x in x_coords:
        for y in y_coords:
            cell = box(x, y, x + grid_size, y + grid_size)
            grid_cells.append(cell)

    grid_gdf = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)
    
    # Extract the shapes of the non-zero areas (islands)
    mask = raster_data > 0  # Create a mask for non-zero values
    island_shapes = list(shapes(raster_data, mask=mask, transform=transform))

    # Create a GeoDataFrame of island polygons
    island_polygons = [shape(geom) for geom, value in island_shapes if value > 0]
    islands_gdf = gpd.GeoDataFrame({'geometry': island_polygons}, crs=crs)

    # Select grid cells that intersect with the islands
    intersecting_cells = gpd.sjoin(grid_gdf, islands_gdf, how='inner', predicate='intersects')
    intersecting_cells = intersecting_cells.drop_duplicates(subset='geometry') #remove duplicates
    intersecting_cells['index_right'] = [chr(65 + i) for i in range(len(intersecting_cells))] #assign unique letter to each grid cell
    intersecting_cells.rename(columns = {'index_right': 'grid_letter'}, inplace = True)


    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot the original raster
    islands_gdf.plot(ax=ax, color='blue', legend=True, legend_kwds={'label': 'Grid Cells'})
    # Plot the intersecting grid cells
    intersecting_cells.plot(ax=ax, edgecolor='red', facecolor = 'none')
    plt.title("Grid Cells Over Islands")
    plt.legend()
    plt.show()

    return intersecting_cells
