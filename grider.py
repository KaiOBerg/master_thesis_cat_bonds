import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.io import MemoryFile
from rasterio.features import shapes, rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box, shape, LineString, GeometryCollection, Polygon
from shapely.ops import unary_union, split
import shapely


resolution = 1000


def create_islands(exp, crs="EPSG:3857"):
    exp_crs = exp.gdf.to_crs(crs)
    minx, miny, maxx, maxy = exp_crs.total_bounds

    # Calculate the number of rows and columns for the raster
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)

    # Define the transformation matrix
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Create a generator for the geometries and their associated values
    shapes_gen = ((geom, value) for geom, value in zip(exp_crs.geometry, exp_crs['value']))

    raster = rasterize(
        shapes=shapes_gen,
        out_shape=(height, width),
        transform=transform,
        fill=0,  # Fill value for areas with no geometry
        dtype='float32' # Data type of raster
    )

    # Write the raster to a GeoTIFF file using rasterio
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=height,
            width=width,
            count=1,  # Number of bands
            dtype='float32',  # Data type for the raster values
            crs=crs,  # Coordinate reference system
            transform=transform,
        ) as dataset:
            dataset.write(raster, 1)  # Write raster data to the first band
            
            # Read back the raster data from the in-memory file
            raster_data = dataset.read(1)
            transform = dataset.transform

    mask = raster_data > 0  # Create a mask for non-zero values
    # Convert raster mask to polygons (islands)
    cap_style='round'
    shapes_gen = list(shapes(raster_data, mask=mask, transform=transform))
    polygons = [shape(geom) for geom, value in shapes_gen if value > 0]
    # Return as GeoDataFrame
    gdf_islands = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    # Step 3: Merge all adjacent polygons into one using unary_union
    merged_polygon = unary_union(gdf_islands.geometry)

    # Step 4: Create a GeoDataFrame with the merged polygon
    gdf_islands = gpd.GeoDataFrame(geometry=[merged_polygon], crs=crs)
    return gdf_islands

def buffer_islands(islands_gdf, buffer_distance_km, crs="EPSG:3857"):
    # Reproject to a projected CRS to work with distances in meters
    islands_projected = islands_gdf.to_crs(crs)
    
    # Create rectangular buffer
    buffers = islands_projected.geometry.buffer(distance=buffer_distance_km * 1000)  # Convert km to meters
    
    # Return the buffered islands
    return gpd.GeoDataFrame(geometry=buffers, crs=crs)

def divide_into_grid(buffered_gdf, grid_cell_size_km, min_overlap_percent, crs="EPSG:3857"):
    grid_cells = []
    for buffered_island in buffered_gdf.geometry:
        # Create bounding box for the buffered area
        minx, miny, maxx, maxy = buffered_island.bounds

        # Generate grid cells within the bounding box
        for x in np.arange(minx, maxx, grid_cell_size_km * 1000):  # Convert km to meters
            for y in np.arange(miny, maxy, grid_cell_size_km * 1000):
                cell = box(x, y, x + grid_cell_size_km * 1000, y + grid_cell_size_km * 1000)
                if cell.intersects(buffered_island):  # Only include grid cells intersecting the buffered area
                    # Calculate intersection area between the grid cell and the buffered island
                    intersection_area = cell.intersection(buffered_island).area
                    # Calculate total area of the grid cell
                    cell_area = cell.area
                    # Calculate overlap percentage
                    overlap_percent = (intersection_area / cell_area) * 100
                    # Keep the grid cell if the overlap percentage is greater than the threshold
                    if overlap_percent >= min_overlap_percent:
                        grid_cells.append(cell)

    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=crs)
    grid_gdf['grid_letter'] = [chr(65 + i) for i in range(len(grid_gdf))]

    
    # Return as GeoDataFrame
    return grid_gdf

def process_islands(exposure, buffer_distance_km, grid_cell_size_km, min_overlap_percent, plt_true=True):
    exposure_crs = exposure.crs
    islands_gdf = create_islands(exposure)
    buffered_islands = buffer_islands(islands_gdf, buffer_distance_km)
    grid_gdf = divide_into_grid(buffered_islands, grid_cell_size_km, min_overlap_percent)

    islands_gdf = islands_gdf.to_crs(exposure_crs)
    buffered_islands = buffered_islands.to_crs(exposure_crs)
    grid_gdf = grid_gdf.to_crs(exposure_crs)


    #if plt_true:
    #    fig, ax = plt.subplots(figsize=(10, 10))
    #    islands_gdf.plot(ax=ax, color="green", label="Islands")
    #    #buffered_islands.plot(ax=ax, color="blue", alpha=0.3, label="Buffer")
    #    grid_gdf.plot(ax=ax, facecolor="none", edgecolor="red", label="Grid Cells")
    #    handles = [
    #        plt.Line2D([0], [0], color="green", lw=4, label="Islands"),           
    #        #plt.Line2D([0], [0], color="blue", lw=4, alpha=0.3, label="Buffer"),  
    #        plt.Line2D([0], [0], color="red", lw=2, label="Grid Cells")           
    #    ]
    #    ax.legend(handles=handles, loc="upper right")
    #    plt.show()
    
    return islands_gdf, buffered_islands, grid_gdf


def divide_islands(islands, num_divisions):
    """Divide a single polygon into equal-sized areas along its longest axis."""
    minx, miny, maxx, maxy = islands.bounds
    split_lines = []
    # Split the polygon horizontally or vertically
    if (maxx - minx) > (maxy - miny):
        # Vertical split along x-axis
        x_split_points = np.linspace(minx, maxx, num_divisions+1)
        split_lines = [LineString([(x, miny), (x, maxy)]) for x in x_split_points[1:-1]]
    else:
        # Horizontal split along y-axis
        y_split_points = np.linspace(miny, maxy, num_divisions+1)
        split_lines = [LineString([(minx, y), (maxx, y)]) for y in y_split_points[1:-1]]

    polygons = [islands]
    # Split the polygon by the lines
    for line in split_lines:
        new_polygons = []
        for polygon in polygons:
            if isinstance(polygon, Polygon):
                split_result = split(polygon, line)
                if isinstance(split_result, GeometryCollection):
                    new_polygons.extend(split_result.geoms)  # Extend with split geometries
                else:
                    new_polygons.append(split_result)  # Append single polygon
        polygons = new_polygons
    return polygons

def init_equ_pol(exposure, grid_size=600, buffer_size=1, crs="EPSG:3857"):
    divided_islands = []
    exposure_crs = exposure.crs
    islands_gdf = create_islands(exposure)
    islands_sng = islands_gdf.explode(index_parts=False)
    buffered_geometries = islands_sng.geometry.buffer(buffer_size * 1000)
    united_polygons = unary_union(buffered_geometries)
    if united_polygons.geom_type == 'MultiPolygon':
        islands_gdf = gpd.GeoDataFrame(geometry=list(united_polygons.geoms), crs=crs)
    else:
        islands_gdf = gpd.GeoDataFrame(geometry=[united_polygons], crs=crs)
    for i, geometry in enumerate(islands_gdf.geometry):
        # Handle single polygon case
        pol_area = islands_gdf.geometry[i].area / 1000**2
        num_grid = int(pol_area // grid_size)
        if num_grid > 0:
            divided_islands.extend(divide_islands(geometry, num_grid))   
        else:
            divided_islands.append(geometry)
    
    islands_split_gdf = gpd.GeoDataFrame(geometry=divided_islands, crs=crs)

    islands_split_gdf = islands_split_gdf.to_crs(exposure_crs)
    return islands_split_gdf