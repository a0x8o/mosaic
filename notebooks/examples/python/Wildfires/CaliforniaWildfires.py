# Databricks notebook source
# MAGIC %md # Wildfire
# MAGIC 
# MAGIC ## Raster Ingestion
# MAGIC 
# MAGIC This notebook converts Raster data to H3, allowing you to spatially aggregate, spatially join, and visualize data in an efficient manner.
# MAGIC 
# MAGIC <img src="https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/GUID-6754AF39-CDE9-4F9D-8C3A-D59D93059BDD-web.png" width=250px> 
# MAGIC →
# MAGIC <img src="https://www.databricks.com/wp-content/uploads/2019/11/Processing-Geospatial-Data-at-Scale-With-Databricks-02.jpg" width=250px>
# MAGIC 
# MAGIC ⚠️**Important:** This notebook requires to be run on a cluster with **Photon enabled**. Documentation on how to enable Photon acceleration: <a href="https://docs.databricks.com/runtime/photon.html#databricks-clusters">here</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup
# MAGIC 
# MAGIC For demo purposes we are installing **rasterio** and **databricks-mosaic** in this notebook, but for production workloads keep in mind that accoding to the documentation: 
# MAGIC 
# MAGIC *"Using notebook-scoped libraries might result in more traffic to the driver node as it works to keep the environment consistent across executor nodes."* 
# MAGIC <br>Source: <a href="https://docs.databricks.com/libraries/notebooks-python-libraries.html">Notebook-scoped Python libraries</a>
# MAGIC 
# MAGIC Guide on how to install libraries on a cluster <a href="https://docs.databricks.com/libraries/cluster-libraries.html">here</a>.

# COMMAND ----------

# MAGIC %pip install rasterio databricks-mosaic

# COMMAND ----------

import json
from pyspark.sql.functions import col, lit, explode
from pyspark.sql.types import ArrayType, StringType, DoubleType, StructType, StructField, LongType, IntegerType
import pyspark.sql.functions as F
from pyspark.databricks.sql.functions import *
import pyspark.pandas as ps
from pyspark.sql.functions import broadcast

import rasterio
import rasterio.features
import rasterio.warp
from rasterio.io import MemoryFile
import rasterio.mask

import mosaic as mos
from mosaic import enable_mosaic
from datetime import datetime

import ipywidgets as widgets
from ipywidgets import interact

h3_resolution = 9
filter_res = 6
enable_mosaic(spark, dbutils)
start_time = datetime.now()

# COMMAND ----------

spark.conf.set("spark.databricks.delta.formatCheck.enabled", False)
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Load raster data
# MAGIC 
# MAGIC In this session you must define the raster location as well as the database to save the data. Databases in some organizations are defined by administrators. 

# COMMAND ----------

# Path to directory of geotiff images 
DATA_DIR = "/FileStore/geospatial"
DATA_DIR_FUSE = "/dbfs" + DATA_DIR
FILE = "ca_wf__pml_factor__0_005_Clipped_Raster_File.tif"
#FILE = "svi_2018_tract_socioeconomic_wgs84.tif"

# Database configuration: used to persist data on Delta Lake for consumption using SQL or KEPLER.GL
DATABASE = "geospatial" # set here the name of the database you want to use

# Uncomment the following command in case you want to create a database
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {DATABASE}") # This command will only work in case you have access to create a database
spark.sql(f"USE {DATABASE}")

df_bin = (spark.read
          .format("binaryFile")
          .option("pathGlobFilter", FILE)
          .load(DATA_DIR)
         )

# COMMAND ----------

@udf(returnType=StringType()) 
def get_crs(content):
  # Read the in-memory tiff file
  with MemoryFile(bytes(content)) as memfile:
    with memfile.open() as data:
      # Use rasterio with the data object
      return str(data.crs)

df_bin.withColumn("crs", get_crs("content"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Extract polygons from raster

# COMMAND ----------

@udf(returnType=ArrayType(StringType())) 
def get_mask_shapes(content):
  geometries = []
  
  # Read the in-memory tiff file
  with MemoryFile(bytes(content)) as memfile:
    with memfile.open() as data:

      # Read the dataset's valid data mask as a ndarray.
      mask = data.dataset_mask()

      # Extract feature shapes and values from the array.
      for geom, val in rasterio.features.shapes(
              mask, transform=data.transform):

        if val > 0: # Only append shapes that have a positive maks value
          
          # Transform shapes from the dataset's own coordinate
          # reference system to CRS84 (EPSG:4326).
          geom = rasterio.warp.transform_geom(
              data.crs, 'EPSG:4326', geom, precision=6)

          geometries.append(json.dumps(geom))
          
  return geometries

df_masks = (df_bin
            .withColumn("mask_json_shapes", get_mask_shapes("content"))
            .withColumn("mask_json", explode("mask_json_shapes"))
            # Convert geoJSON to WKB
            .withColumn("mask_wkb", mos.st_aswkb(mos.st_geomfromgeojson("mask_json")))
            .drop("content", "mask_json_shapes", "mask_json")
           )

df_polygons = df_masks.select("mask_wkb")

# COMMAND ----------

df_polygons.display()

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC df_polygons "mask_wkb" "geometry" 20000000

# COMMAND ----------

with rasterio.open(DATA_DIR_FUSE + "/" + FILE) as dataset:
    height = dataset.height
    width = dataset.width
    bands = dataset.read(1)
    transform = dataset.transform
    
    # Approximate area of the TIFF in Km^2
    area = dataset.bounds.bottom * dataset.bounds.left * dataset.bounds.right * dataset.bounds.top / 1000

    # Amount of pixels in the in the TIFF
    pixels = height * width

    # pixels per km^2
    pixels_per_km2 = pixels / area

# COMMAND ----------

# MAGIC %md 
# MAGIC # Checking h3 resolution
# MAGIC 
# MAGIC The goal is to avoid collecting map with only one point per cell, what in case of raster files would not allow visualization using Kepler.gl

# COMMAND ----------

h3_resolution = 9
filter_res = 6
h3_data = {
    0: [4357449.42, 2562182.16],
    1: [609788.44, 328434.59],
    2: [86801.78, 44930.90],
    3: [12393.43, 6315.47],
    4: [1770.35, 896.58],
    5: [252.90, 127.79],
    6: [36.13, 18.24],
    7: [5.16, 2.60],
    8: [0.74, 0.37],
    9: [0.11, 0.05],
    10: [0.02, 0.01],
    11: [0.00, 0.00],
    12: [0.00, 0.000154944],
    13: [0.00, 0.000022135],
    14: [0.000006267, 0.000003162],
    15: [0.000000895, 0.000000452],
}
h3_res_df = ps.DataFrame.from_dict(
    h3_data,
    orient="index",
    columns=["Avg_Hexagon_Area_km2", "Pentagon_Area_km2"],
)
h3_res_df.index.name = "Resolution"
h3_res_df['h3_per_cell'] = h3_res_df['Avg_Hexagon_Area_km2'].apply(lambda x: round(x * pixels_per_km2,0))
max_h3_res = h3_res_df[h3_res_df['h3_per_cell']>1].tail(1).index.values[0]

if filter_res >= h3_resolution:
  filter_res = h3_resolution
  
if h3_resolution > max_h3_res:
  h3_resolution = int(max_h3_res)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Extract points from raster

# COMMAND ----------

with rasterio.open(DATA_DIR_FUSE + "/" + FILE) as dataset:
    height = dataset.height
    width = dataset.width
    bands = dataset.read(1)
    transform = dataset.transform

df_x_idx = spark.range(height).withColumnRenamed("id", "x_idx")
df_y_idx = spark.range(width).withColumnRenamed("id", "y_idx")
df_xy = df_x_idx.crossJoin(df_y_idx)
df = df_xy.pandas_api()
ps.set_option('compute.ops_on_diff_frames', True)
df["value"] = df.apply(lambda x: bands[x.x_idx, x.y_idx], axis=1) 
df["coord"] = df.apply(lambda x: rasterio.transform.xy(transform, x.x_idx, x.y_idx), axis=1)
df = df.drop(['x_idx', 'y_idx'], axis=1)
ps.reset_option('compute.ops_on_diff_frames')
df_points = df.to_spark()

# COMMAND ----------

df_points = (
  df_points
  .withColumn("lat", F.col("coord").getItem(0))
  .withColumn("lon", F.col("coord").getItem(1))
  .withColumn("point", mos.st_point("lat", "lon"))
  .withColumn("index", mos.grid_longlatascellid("lat", "lon", F.lit(h3_resolution)))
  .withColumn("index_filter", mos.grid_longlatascellid("lat", "lon", F.lit(filter_res)))
  .select("point", "index", "index_filter", "value")
)

df_points.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Point in polygon join
# MAGIC 
# MAGIC You can now find the raster points that correspont to a certain chip in the tessellation of the polygons

# COMMAND ----------

# tessallate polygons
df_tessellated = (df_polygons
                        .withColumn("chips", mos.grid_tessellateexplode("mask_wkb", F.lit(filter_res)))
                        .withColumn("index_id", F.col("chips.index_id"))
                        .withColumn("is_core", F.col("chips.is_core"))
                        .withColumn("wkb", F.col("chips.wkb"))
                        .drop('chips')
                        )
df_tessellated.count()

# COMMAND ----------

# JOIN
df_j = (
  df_points
  .join(broadcast(df_tessellated), on=[df_points.index_filter == df_tessellated.index_id], how="inner")
  .filter(df_tessellated.is_core | mos.st_contains(df_tessellated.wkb, df_points.point))
)
df_j.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Aggregate over index
# MAGIC get the avarage raster band values for each hex

# COMMAND ----------

df_agg = df_j.groupBy("index").agg(F.round(F.avg("value"),2).alias("value_band_0"))
df_agg.count()

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType

schema = StructType(
    [
        StructField("types", StringType(), True),
        StructField("coordinates", ArrayType(DoubleType(), True), True),
    ]
)

df_agg = (
    df_agg.withColumn(
        "latlong", F.from_json(h3_centerasgeojson(F.col("index")), schema)
    )
    .withColumn("lon", F.col("latlong.coordinates").getItem(0))
    .withColumn("lat", F.col("latlong.coordinates").getItem(1))
    .drop("latlong")
)

# COMMAND ----------

df_agg.write.mode("overwrite").option("mergeSchema", "true").format("delta").saveAsTable("wildfire")

# COMMAND ----------

# MAGIC %md
# MAGIC # Filtering an area of interest by latitude
# MAGIC 
# MAGIC Once the data is saved on Delta you just need to execute the following commands to explore the dataset. The steps above should be executed every time you have a new dataset.

# COMMAND ----------

df_chipped = spark.read.table('geospatial.wildfire')
df_chipped = df_chipped.filter(df_chipped.value_band_0>=0)

# COMMAND ----------

# Loading widgets for filtering the data
boundaries = [[90, -90, 180, -180]]
    
lat_slider = widgets.IntRangeSlider(
    description="Latitude: ", min=boundaries[0][1], max=boundaries[0][0], value=[boundaries[0][1], boundaries[0][0]])
lon_slider = widgets.IntRangeSlider(
    description="Longitude: ", min=boundaries[0][3], max=boundaries[0][2], value=[boundaries[0][3], boundaries[0][2]]
)
items = [lat_slider, lon_slider]
box = widgets.Box(children=items)
box

# COMMAND ----------

# The boundaries of the entire dataset are:
df_chipped.select(
        F.ceil(F.min("lat")).alias('min lat'), F.ceil(F.max("lat")).alias('max lat'), F.ceil(F.min("lon")).alias('min lon'), F.ceil(F.max("lon")).alias('max lon')
    ).collect()

# COMMAND ----------

df_filter = df_chipped.filter(
    ((df_chipped.lat >= lat_slider.value[0])
    & (df_chipped.lat <= lat_slider.value[1]))
    & ((df_chipped.lon >= lon_slider.value[0])
    & (df_chipped.lon <= lon_slider.value[1]))
)
df_filter.count()

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC df_filter "index" "h3" 200000

# COMMAND ----------

# MAGIC %md
# MAGIC # Filtering by value

# COMMAND ----------

# Loading widgets for filtering the data
boundaries = df_chipped.select(
        F.ceil(F.min("value_band_0")).alias('min value'), F.ceil(F.max("value_band_0")).alias('max value')
    ).collect()
    
value_slider = widgets.FloatRangeSlider(
    description="Value: ", min=boundaries[0][0], max=boundaries[0][1],step=.05)
value_slider

# COMMAND ----------

df_filter = (df_chipped
             .filter(
                ((df_chipped.value_band_0 >= value_slider.value[0])
                & (df_chipped.value_band_0 <= value_slider.value[1])))
             .select('index','value_band_0'))
df_filter.count()

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC df_filter "index" "h3" 500000

# COMMAND ----------

end_time = datetime.now()
print(f"THE ANALYSIS COMPLETED IN {(end_time - start_time).total_seconds()} SECONDS")
