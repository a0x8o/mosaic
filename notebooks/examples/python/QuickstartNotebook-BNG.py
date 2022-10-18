# Databricks notebook source
# MAGIC %md
# MAGIC ## Setup NYC taxi zones
# MAGIC In order to setup the data please run the notebook available at "../../data/DownloadNYCTaxiZones". </br>
# MAGIC DownloadNYCTaxiZones notebook will make sure we have New York City Taxi zone shapes available in our environment.

# COMMAND ----------

user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

raw_path = f"dbfs:/tmp/mosaic/{user_name}"
raw_taxi_zones_path = f"{raw_path}/taxi_zones"

print(f"The raw data is stored in {raw_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable Mosaic in the notebook
# MAGIC To get started, you'll need to attach the wheel to your cluster and import instances as in the cell below.

# COMMAND ----------

from pyspark.sql.functions import *
import mosaic as mos
spark.conf.set("spark.databricks.labs.mosaic.index.system", "BNG")
mos.enable_mosaic(spark, dbutils)

# COMMAND ----------

# MAGIC %md ## Read polygons from GeoJson

# COMMAND ----------

# MAGIC %md
# MAGIC With the functionality Mosaic brings we can easily load GeoJSON files using spark. </br>
# MAGIC In the past this required GeoPandas in python and conversion to spark dataframe. </br>

# COMMAND ----------

def reproject(geom, src_srid, dest_srid):
  return mos.st_asbinary(mos.st_transform(mos.st_setsrid(geom, lit(src_srid)), lit(dest_srid)))

def reposition(geom):
  (xG, yG) = (0.0098, 51.4934)
  (xNY, yNY) = (-74.0060, 40.7128)
  return mos.st_translate(geom, lit(xG - xNY), lit(yG - yNY))

# COMMAND ----------

neighbourhoods = (
  spark.read
    .option("multiline", "true")
    .format("json")
    .load(raw_taxi_zones_path)
    .select("type", explode(col("features")).alias("feature"))
    .select("type", col("feature.properties").alias("properties"), to_json(col("feature.geometry")).alias("json_geometry"))
    .withColumn("geometry", mos.st_geomfromgeojson("json_geometry"))
    .withColumn("geometry", reposition(col("geometry")))
    .withColumn("geometry", reproject(col("geometry"), 4326, 27700))
    .withColumn("geometry", mos.st_astext(col("geometry")))
    .where(mos.st_isvalid(col("geometry")))
    .where(expr("st_hasvalidcoordinates(geometry, 'EPSG:27700', 'reprojected_bounds')"))
)

display(
  neighbourhoods
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Compute some basic geometry attributes

# COMMAND ----------

# MAGIC %md
# MAGIC Mosaic provides a number of functions for extracting the properties of geometries. Here are some that are relevant to Polygon geometries:

# COMMAND ----------

display(
  neighbourhoods
    .withColumn("calculatedArea", mos.st_area(col("geometry")))
    .withColumn("calculatedLength", mos.st_length(col("geometry")))
    # Note: The unit of measure of the area and length depends on the CRS used.
    # For GPS locations it will be square radians and radians
    .select("geometry", "calculatedArea", "calculatedLength")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read points data

# COMMAND ----------

# MAGIC %md
# MAGIC We will load some Taxi trips data to represent point data. </br>
# MAGIC We already loaded some shapes representing polygons that correspond to NYC neighbourhoods. </br>

# COMMAND ----------

tripsTable = spark.table("delta.`/databricks-datasets/nyctaxi/tables/nyctaxi_yellow`")

# COMMAND ----------

tripsTable = tripsTable.limit(10000000).repartition(200).cache()
display(tripsTable)

# COMMAND ----------

trips = (
  tripsTable
    .drop("vendorId", "rateCodeId", "store_and_fwd_flag", "payment_type")
    .withColumn("pickup_geom", mos.st_point(col("pickup_longitude"), col("pickup_latitude")))
    .withColumn("pickup_geom", reposition(col("pickup_geom")))
    .where(expr("st_hasvalidcoordinates(pickup_geom, 'EPSG:27700', 'bounds')"))
    .withColumn("pickup_geom", reproject(col("pickup_geom"), 4326, 27700))
    .withColumn("pickup_geom", mos.st_aswkt("pickup_geom"))
    .withColumn("dropoff_geom", mos.st_point(col("dropoff_longitude"), col("dropoff_latitude")))
    .withColumn("dropoff_geom", reposition(col("dropoff_geom")))
    .where(expr("st_hasvalidcoordinates(dropoff_geom, 'EPSG:27700', 'bounds')"))
    .withColumn("dropoff_geom", reproject(col("dropoff_geom"), 4326, 27700))
    .withColumn("dropoff_geom", mos.st_aswkt("dropoff_geom"))
    .where(mos.st_isvalid(col("pickup_geom")) & mos.st_isvalid(col("dropoff_geom")))
)

# COMMAND ----------

trips.write.format("delta").save(f"{raw_path}/bng_trips")

# COMMAND ----------

trips = spark.read.format("delta").load(f"{raw_path}/bng_trips")

# COMMAND ----------

trips.count()

# COMMAND ----------

display(trips.select("pickup_geom", "dropoff_geom"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spatial Joins

# COMMAND ----------

# MAGIC %md
# MAGIC We can use Mosaic to perform spatial joins both with and without Mosaic indexing strategies. </br>
# MAGIC Indexing is very important when handling very different geometries both in size and in shape (ie. number of vertices). </br>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Getting the optimal resolution

# COMMAND ----------

# MAGIC %md
# MAGIC We can use Mosaic functionality to identify how to best index our data based on the data inside the specific dataframe. </br>
# MAGIC Selecting an apropriate indexing resolution can have a considerable impact on the performance. </br>

# COMMAND ----------

from mosaic import MosaicFrame

neighbourhoods_mosaic_frame = MosaicFrame(neighbourhoods, "geometry")
optimal_resolution = neighbourhoods_mosaic_frame.get_optimal_resolution(sample_fraction=0.75)

print(f"Optimal resolution is {optimal_resolution}")

# COMMAND ----------

# MAGIC %md
# MAGIC Not every resolution will yield performance improvements. </br>
# MAGIC By a rule of thumb it is always better to under-index than over-index - if not sure select a lower resolution. </br>
# MAGIC Higher resolutions are needed when we have very imballanced geometries with respect to their size or with respect to the number of vertices. </br>
# MAGIC In such case indexing with more indices will considerably increase the parallel nature of the operations. </br>
# MAGIC You can think of Mosaic as a way to partition an overly complex row into multiple rows that have a balanced amount of computation each.

# COMMAND ----------

display(
  neighbourhoods_mosaic_frame.get_resolution_metrics(sample_rows=150)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Indexing using the optimal resolution

# COMMAND ----------

# MAGIC %md
# MAGIC We will use mosaic sql functions to index our points data. </br>
# MAGIC Here we will use resolution -4, index resolution depends on the dataset in use.

# COMMAND ----------

tripsWithIndex = (trips
  .withColumn("pickup_bng", mos.point_index_geom(col("pickup_geom"), lit(optimal_resolution)))
  .withColumn("dropoff_bng", mos.point_index_geom(col("dropoff_geom"), lit(optimal_resolution)))
)

# COMMAND ----------

display(tripsWithIndex)

# COMMAND ----------

to_display = tripsWithIndex.limit(10000).select(
  mos.st_geomfromwkb(mos.index_geometry("pickup_bng")).alias("pickup_bng"),
  mos.st_geomfromwkb(mos.index_geometry("dropoff_bng")).alias("dropoff_bng"),
  mos.st_geomfromwkt(col("pickup_geom")).alias("pickup_geom"),
  mos.st_geomfromwkt(col("dropoff_geom")).alias("dropoff_geom"),
).select(
  mos.st_astext(reproject(col("pickup_bng"), 27700, 4326)).alias("pickup_bng"),
  mos.st_astext(reproject(col("pickup_geom"), 27700, 4326)).alias("pickup_geom"),
  mos.st_astext(reproject(col("dropoff_bng"), 27700, 4326)).alias("dropoff_bng"),
  mos.st_astext(reproject(col("dropoff_geom"), 27700, 4326)).alias("dropoff_geom")
)

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC to_display "dropoff_bng" "geometry" 200

# COMMAND ----------

# MAGIC %md
# MAGIC We will also index our neighbourhoods using a built in generator function.

# COMMAND ----------

neighbourhoodsWithIndex = (neighbourhoods

                           # We break down the original geometry in multiple smoller mosaic chips, each with its
                           # own index
                           .withColumn("mosaic_index", mos.mosaic_explode(col("geometry"), lit(optimal_resolution)))

                           # We don't need the original geometry any more, since we have broken it down into
                           # Smaller mosaic chips.
                           .drop("json_geometry", "geometry")
                          )

# COMMAND ----------

neighbourhoodsWithIndex.printSchema()

# COMMAND ----------

to_display = neighbourhoodsWithIndex.where(~col("mosaic_index.is_core")).where("properties.location_id == 137").select(
  mos.st_geomfromwkb(mos.index_geometry("mosaic_index.index_id")).alias("index_geom"),
  mos.st_geomfromwkb(col("mosaic_index.wkb")).alias("geom")
).select(
  reproject(col("index_geom"), 27700, 4326).alias("index_geom"),
  reproject(col("geom"), 27700, 4326).alias("geom")
)

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC to_display "geom" "geometry" 5000

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performing the spatial join

# COMMAND ----------

# MAGIC %md
# MAGIC We can now do spatial joins to both pickup and drop off zones based on geolocations in our datasets.

# COMMAND ----------

pickupNeighbourhoods = neighbourhoodsWithIndex.select(col("properties.borough").alias("pickup_zone"), col("mosaic_index"))

withPickupZone = (
  tripsWithIndex.join(
    pickupNeighbourhoods,
    tripsWithIndex["pickup_bng"] == pickupNeighbourhoods["mosaic_index.index_id"]
  ).where(
    # If the borough is a core chip (the chip is fully contained within the geometry), then we do not need
    # to perform any intersection, because any point matching the same index will certainly be contained in
    # the borough. Otherwise we need to perform an st_contains operation on the chip geometry.
    col("mosaic_index.is_core") | mos.st_contains(col("mosaic_index.wkb"), col("pickup_geom"))
  ).select(
    "trip_distance", "pickup_geom", "pickup_zone", "dropoff_geom", "pickup_bng", "dropoff_bng"
  )
)

display(withPickupZone)

# COMMAND ----------

# MAGIC %md
# MAGIC We can easily perform a similar join for the drop off location.

# COMMAND ----------

dropoffNeighbourhoods = neighbourhoodsWithIndex.select(col("properties.borough").alias("dropoff_zone"), col("mosaic_index"))

withDropoffZone = (
  withPickupZone.join(
    dropoffNeighbourhoods,
    withPickupZone["dropoff_bng"] == dropoffNeighbourhoods["mosaic_index.index_id"]
  ).where(
    col("mosaic_index.is_core") | mos.st_contains(col("mosaic_index.wkb"), col("dropoff_geom"))
  ).select(
    "trip_distance", "pickup_geom", "pickup_zone", "dropoff_geom", "dropoff_zone", "pickup_bng", "dropoff_bng"
  )
  .withColumn("trip_line", mos.st_astext(mos.st_makeline(array(mos.st_geomfromwkt(col("pickup_geom")), mos.st_geomfromwkt(col("dropoff_geom"))))))
)

display(withDropoffZone)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualise the results in Kepler

# COMMAND ----------

# MAGIC %md
# MAGIC For visualisation there simply arent good options in scala. </br>
# MAGIC Luckily in our notebooks you can easily switch to python just for UI. </br>
# MAGIC Mosaic abstracts interaction with Kepler in python.

# COMMAND ----------

test_data =  withDropoffZone.limit(10000).cache()

# COMMAND ----------

to_display = test_data.limit(10000).withColumn(
  "pickup_geom", reproject(mos.st_geomfromwkt(col("pickup_geom")), 27700, 4326)
).withColumn(
  "pickup_bng_geom", reproject(mos.st_geomfromwkt(mos.index_geometry(col("pickup_bng"))), 27700, 4326)
).withColumn(
  "dropoff_geom", reproject(mos.st_geomfromwkt(col("dropoff_geom")), 27700, 4326)
).withColumn(
  "dropoff_bng_geom", reproject(mos.st_geomfromwkt(mos.index_geometry(col("dropoff_bng"))), 27700, 4326)
)

# COMMAND ----------

# MAGIC %python
# MAGIC %%mosaic_kepler
# MAGIC to_display "pickup_geom" "geometry" 5000

# COMMAND ----------


