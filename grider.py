'''Script is used to create TC boundary and subareas in the single-country CAT bond development'''

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.io import MemoryFile
from rasterio.features import shapes, rasterize
from rasterio.transform import from_bounds, from_origin, from_bounds
from shapely.geometry import box, shape, LineString, GeometryCollection, Polygon
from shapely.ops import unary_union, split

resolution = 1000 #specify resultion to change exposure layer into country polygons

'''takes the exposure layer and creates polygons from it for the country'''
def create_islands(exp, crs="EPSG:3857"):
    exp_crs = exp.gdf.to_crs(crs)
    minx, miny, maxx, maxy = exp_crs.total_bounds

    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)

    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    shapes_gen = ((geom, value) for geom, value in zip(exp_crs.geometry, exp_crs['value']))

    raster = rasterize(
        shapes=shapes_gen,
        out_shape=(height, width),
        transform=transform,
        fill=0,  
        dtype='float32' 
    )
    mask = raster > 0 
    shapes_gen = list(shapes(raster, mask=mask, transform=transform))
    polygons = [shape(geom) for geom, value in shapes_gen if value > 0]
    gdf_islands = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    merged_polygon = unary_union(gdf_islands.geometry)
    gdf_islands = gpd.GeoDataFrame(geometry=[merged_polygon], crs=crs)
    return gdf_islands

'''create a buffer around the island -> used to create TC boundary'''
def buffer_islands(islands_gdf, buffer_distance_km, crs="EPSG:3857"):
    islands_projected = islands_gdf.to_crs(crs)
    buffers = islands_projected.geometry.buffer(distance=buffer_distance_km * 1000)  # Convert km to meters
    
    return gpd.GeoDataFrame(geometry=buffers, crs=crs)

'''create grids for the buffered areas: filter grids for minimal overlap and thereby get TC boundary'''
def divide_into_grid(buffered_gdf, grid_cell_size_km, min_overlap_percent, crs="EPSG:3857"):
    grid_cells = []
    for buffered_island in buffered_gdf.geometry:
        minx, miny, maxx, maxy = buffered_island.bounds

        for x in np.arange(minx, maxx, grid_cell_size_km * 1000):  # Convert km to meters
            for y in np.arange(miny, maxy, grid_cell_size_km * 1000):
                cell = box(x, y, x + grid_cell_size_km * 1000, y + grid_cell_size_km * 1000)
                if cell.intersects(buffered_island):  # Only include grid cells intersecting the buffered area
                    intersection_area = cell.intersection(buffered_island).area
                    cell_area = cell.area
                    overlap_percent = (intersection_area / cell_area) * 100
                    if overlap_percent >= min_overlap_percent:
                        grid_cells.append(cell)

    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=crs)
    grid_gdf['grid_letter'] = [chr(65 + i) for i in range(len(grid_gdf))]

    return grid_gdf

'''wrapper function to create TC boundaries'''
def process_islands(exposure, buffer_distance_km, grid_cell_size_km, min_overlap_percent, crs="EPSG:3857"):
    islands_gdf = create_islands(exposure, crs)
    buffered_islands = buffer_islands(islands_gdf, buffer_distance_km, crs)
    grid_gdf = divide_into_grid(buffered_islands, grid_cell_size_km, min_overlap_percent, crs)

    if crs == "EPSG:3857":
        exposure_crs = exposure.crs
        islands_gdf = islands_gdf.to_crs(exposure_crs)
        buffered_islands = buffered_islands.to_crs(exposure_crs)
        grid_gdf = grid_gdf.to_crs(exposure_crs)
    
    return islands_gdf, buffered_islands, grid_gdf


"""Divide a single polygon into equal-sized areas along its longest axis -> subareas, not used for final results"""
def divide_islands(islands, num_divisions):

    minx, miny, maxx, maxy = islands.bounds
    split_lines = []
    if (maxx - minx) > (maxy - miny):
        x_split_points = np.linspace(minx, maxx, num_divisions+1)
        split_lines = [LineString([(x, miny), (x, maxy)]) for x in x_split_points[1:-1]]
    else:
        y_split_points = np.linspace(miny, maxy, num_divisions+1)
        split_lines = [LineString([(minx, y), (maxx, y)]) for y in y_split_points[1:-1]]

    polygons = [islands]
    for line in split_lines:
        new_polygons = []
        for polygon in polygons:
            if isinstance(polygon, Polygon):
                split_result = split(polygon, line)
                if isinstance(split_result, GeometryCollection):
                    new_polygons.extend(split_result.geoms)  
                else:
                    new_polygons.append(split_result)  
        polygons = new_polygons
    return polygons

'''not used for final results, alternative way to derive subareas'''
def init_equ_pol(exposure, grid_size=6000, buffer_size=1, crs="EPSG:3857"):
    divided_islands = []
    exposure_crs = exposure.crs
    islands_gdf = create_islands(exposure, crs)
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

    if crs == "EPSG:3857":
        islands_split_gdf = islands_split_gdf.to_crs(exposure_crs)
    return islands_split_gdf


'''not used for final results, alternative way to derive subareas'''
def init_grid(exp, pixel_size, grid_size, plot_rst=True):
    minx, miny, maxx, maxy = exp.gdf.total_bounds  

    nrows = int((maxy - miny) / pixel_size) + 1
    ncols = int((maxx - minx) / pixel_size) + 1

    transform = from_origin(minx, maxy, pixel_size, pixel_size)

    raster = np.full((nrows, ncols), np.nan)

    for _, row in exp.gdf.iterrows():
        x, y = row.geometry.x, row.geometry.y

        col = int((x - minx) / pixel_size)
        row_idx = int((maxy - y) / pixel_size)

        raster[row_idx, col] = row['value']

    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=nrows,
            width=ncols,
            count=1,  
            dtype=rasterio.float32,  
            crs='EPSG:4326',  
            transform=transform,
        ) as dataset:
            dataset.write(raster, 1) 
            
            raster_data = dataset.read(1)
            transform = dataset.transform
            crs = dataset.crs
    
    x_coords = np.arange(minx, maxx, grid_size)
    y_coords = np.arange(miny, maxy, grid_size)

    grid_cells = [box(x, y, x + grid_size, y + grid_size) for x in x_coords for y in y_coords]
    grid_gdf = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)
    
    mask = raster_data > 0 
    island_shapes = list(shapes(raster_data, mask=mask, transform=transform))

    island_polygons = [shape(geom) for geom, value in island_shapes if value > 0]
    islands_gdf = gpd.GeoDataFrame({'geometry': island_polygons}, crs=crs)

    intersecting_cells = gpd.sjoin(grid_gdf, islands_gdf, how='inner', predicate='intersects')
    intersecting_cells = intersecting_cells.drop_duplicates(subset='geometry')
    intersecting_cells['grid_letter'] = [chr(65 + i) for i in range(len(intersecting_cells))]
    intersecting_cells = intersecting_cells.drop(columns=['index_right'])

    if plot_rst:
        fig, ax = plt.subplots(figsize=(10, 10))
        islands_gdf.plot(ax=ax, color='blue', legend=True, label='Islands')
        intersecting_cells.plot(ax=ax, edgecolor='red', facecolor = 'none', label='Grid cells')
        plt.title("Grid Cells Over Islands")
        plt.legend()
        plt.show()

    return intersecting_cells