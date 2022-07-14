// Databricks notebook source
// MAGIC %md
// MAGIC ## Setup NYC taxi zones
// MAGIC In order to setup the data please run the notebook available at "../../data/DownloadNYCTaxiZones". </br>
// MAGIC DownloadNYCTaxiZones notebook will make sure we have New York City Taxi zone shapes available in our environment.

// COMMAND ----------

val user_name = dbutils.notebook.getContext.userName.get

val raw_path = s"dbfs:/tmp/mosaic/$user_name"
val raw_taxi_zones_path = s"$raw_path/taxi_zones"

print(s"The raw data is stored in $raw_path")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Enable Mosaic in the notebook
// MAGIC To get started, you'll need to attach the wheel to your cluster and import instances as in the cell below.

// COMMAND ----------

import com.databricks.labs.mosaic.functions.MosaicContext
import com.databricks.labs.mosaic.BNG
import com.databricks.labs.mosaic.ESRI

val mosaicContext = MosaicContext.build(BNG, ESRI)
import mosaicContext.functions._
import org.apache.spark.sql.functions._

// COMMAND ----------

// MAGIC %md ## Read polygons from GeoJson

// COMMAND ----------

// MAGIC %md
// MAGIC With the functionality Mosaic brings we can easily load GeoJSON files using spark. </br>
// MAGIC In the past this required GeoPandas in python and conversion to spark dataframe. </br>

// COMMAND ----------

def reproject(geom: String, srcSRID: Int, destSRID: Int) = {
  st_asbinary(st_transform(st_setsrid(col(geom), lit(srcSRID)), lit(destSRID)))
}

def reposition(geom: String) = {
  val (xG, yG) = (0.0098, 51.4934)
  val (xNY, yNY) = (-74.0060, 40.7128)
  st_translate(col(geom), lit(xG - xNY), lit(yG - yNY))
}

// COMMAND ----------

val neighbourhoods = (
  spark.read
    .option("multiline", "true")
    .format("json")
    .load(raw_taxi_zones_path)
    .select(col("type"), explode(col("features")).alias("feature"))
    .select(col("type"), col("feature.properties").alias("properties"), to_json(col("feature.geometry")).alias("json_geometry"))
    .withColumn("geometry", st_geomfromgeojson(col("json_geometry")))
    .withColumn("geometry", reposition("geometry"))
    .withColumn("geometry", reproject("geometry", 4326, 27700))
    .withColumn("geometry", st_astext(col("geometry")))
    .where(st_isvalid(col("geometry")))
)

display(
  neighbourhoods
)

// COMMAND ----------

// MAGIC %md
// MAGIC ##  Compute some basic geometry attributes

// COMMAND ----------

// MAGIC %md
// MAGIC Mosaic provides a number of functions for extracting the properties of geometries. Here are some that are relevant to Polygon geometries:

// COMMAND ----------

display(
  neighbourhoods
    .withColumn("calculatedArea", st_area(col("geometry")))
    .withColumn("calculatedLength", st_length(col("geometry")))
    // Note: The unit of measure of the area and length depends on the CRS used.
    // For GPS locations it will be square radians and radians
    .select("geometry", "calculatedArea", "calculatedLength")
)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Read points data

// COMMAND ----------

// MAGIC %md
// MAGIC We will load some Taxi trips data to represent point data. </br>
// MAGIC We already loaded some shapes representing polygons that correspond to NYC neighbourhoods. </br>

// COMMAND ----------

val tripsTable = spark.table("delta.`/databricks-datasets/nyctaxi/tables/nyctaxi_yellow`")

// COMMAND ----------

val y = tripsTable.limit(4000000).repartition(120).cache()

// COMMAND ----------

display(y)

// COMMAND ----------

val trips = (
  y
    .drop("vendorId", "rateCodeId", "store_and_fwd_flag", "payment_type")
    .withColumn("pickup_geom", st_point(col("pickup_longitude"), col("pickup_latitude")))
    .withColumn("pickup_geom", reposition("pickup_geom"))
    .withColumn("pickup_geom", reproject("pickup_geom", 4326, 27700))
    .withColumn("pickup_geom", st_aswkt(col("pickup_geom")))
    .withColumn("dropoff_geom", st_point(col("dropoff_longitude"), col("dropoff_latitude")))
    .withColumn("dropoff_geom", reposition("dropoff_geom"))
    .withColumn("dropoff_geom", reproject("dropoff_geom", 4326, 27700))
    .withColumn("dropoff_geom", st_aswkt(col("dropoff_geom")))
)

// COMMAND ----------

spark.conf.set("spark.databricks.optimizer.adaptive.enabled", "false")

// COMMAND ----------

trips.count()

// COMMAND ----------

val x = trips

// COMMAND ----------

display(x.select("x.inputs.*"))

// COMMAND ----------

x.select("x.inputs").printSchema()

// COMMAND ----------

x.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("mosaic_bng_erros")

// COMMAND ----------

val x = spark.read.table("mosaic_bng_erros")

// COMMAND ----------



// COMMAND ----------

display(x.where("dropoff_geom is null").withColumn("valid", st_isvalid(st_point(col("dropoff_longitude"), col("dropoff_latitude")))))

// COMMAND ----------

import org.locationtech.proj4j.{CoordinateTransformFactory, CRSFactory, ProjCoordinate}


// COMMAND ----------

import com.databricks.labs.mosaic.core.geometry.point.MosaicPointESRI

val point = MosaicPointESRI(Seq(-73.994562, 40405))

val crsFactory = new CRSFactory
val crsFrom = crsFactory.createFromName(f"epsg:27700")
val crsTo = crsFactory.createFromName(f"epsg:4326")

val ctFactory = new CoordinateTransformFactory
val trans = ctFactory.createTransform(crsFrom, crsTo)

val pIn = new ProjCoordinate
val pOut = new ProjCoordinate

def mapper(x: Double, y: Double): (Double, Double) = {
    pIn.setValue(x, y)
    trans.transform(pIn, pOut)
    (pOut.x, pOut.y)
}
val mosaicGeometry = point.mapXY(mapper)
mosaicGeometry.setSpatialReference(27700)
mosaicGeometry

// COMMAND ----------

trans.getSourceCRS.getProjection.getMaxLatitudeDegrees

// COMMAND ----------

trans.getSourceCRS.getProjection.getMinLatitudeDegrees

// COMMAND ----------

val crs = crsFactory.createFromName(f"epsg:4326")
crs.getProjection.getTrueScaleLatitude

// COMMAND ----------

val crs = crsFactory.createFromName(f"epsg:27700")
crs.getProjection.project()

// COMMAND ----------

val pIn = new ProjCoordinate
pIn.setValue(-77, 440)
pIn.hasValidXandYOrdinates
trans.transform(pIn, pOut)

// COMMAND ----------

import com.databricks.labs.mosaic.core.geometry.point.MosaicPointESRI

val point = MosaicPointESRI(Seq(-73.994562, 404.405))

val crsFactory = new CRSFactory
val crs = crsFactory.createFromName(f"epsg:4326")

val xMin = crs.getProjection.getMinLongitudeDegrees


def mapper(x: Double, y: Double): (Double, Double) = {
    pIn.setValue(x, y)
    trans.transform(pIn, pOut)
    (pOut.x, pOut.y)
}
val mosaicGeometry = point.mapXY(mapper)
mosaicGeometry.setSpatialReference(27700)
mosaicGeometry

// COMMAND ----------

// MAGIC %python
// MAGIC display(trips.select("pickup_geom", "dropoff_geom"))

// COMMAND ----------

// MAGIC %python
// MAGIC display(tripsTable.withColumn("pickup_geom", mos.st_point(col("pickup_longitude"), col("pickup_latitude")))
// MAGIC     .withColumn("pickup_geom", reposition(col("pickup_geom"))).select("pickup_geom"))

// COMMAND ----------

// MAGIC %python
// MAGIC display(
// MAGIC   trips.select("pickup_geom", "dropoff_geom").withColumn(
// MAGIC     "pickup_geom2", mos.st_astext(reproject(mos.st_geomfromwkt(col("pickup_geom")), 27700, 4326))
// MAGIC   ).withColumn(
// MAGIC     "dropoff_geom2", mos.st_astext(reproject(mos.st_geomfromwkt(col("dropoff_geom")), 27700, 4326))
// MAGIC   )
// MAGIC )

// COMMAND ----------

// MAGIC %md
// MAGIC ## Spatial Joins

// COMMAND ----------

// MAGIC %md
// MAGIC We can use Mosaic to perform spatial joins both with and without Mosaic indexing strategies. </br>
// MAGIC Indexing is very important when handling very different geometries both in size and in shape (ie. number of vertices). </br>

// COMMAND ----------

// MAGIC %md
// MAGIC ### Getting the optimal resolution

// COMMAND ----------

// MAGIC %md
// MAGIC We can use Mosaic functionality to identify how to best index our data based on the data inside the specific dataframe. </br>
// MAGIC Selecting an apropriate indexing resolution can have a considerable impact on the performance. </br>

// COMMAND ----------

// MAGIC %python
// MAGIC from mosaic import MosaicFrame
// MAGIC 
// MAGIC neighbourhoods_mosaic_frame = MosaicFrame(neighbourhoods, "geometry")
// MAGIC optimal_resolution = neighbourhoods_mosaic_frame.get_optimal_resolution(sample_fraction=0.75)
// MAGIC 
// MAGIC print(f"Optimal resolution is {optimal_resolution}")

// COMMAND ----------

// MAGIC %md
// MAGIC Not every resolution will yield performance improvements. </br>
// MAGIC By a rule of thumb it is always better to under-index than over-index - if not sure select a lower resolution. </br>
// MAGIC Higher resolutions are needed when we have very imballanced geometries with respect to their size or with respect to the number of vertices. </br>
// MAGIC In such case indexing with more indices will considerably increase the parallel nature of the operations. </br>
// MAGIC You can think of Mosaic as a way to partition an overly complex row into multiple rows that have a balanced amount of computation each.

// COMMAND ----------

// MAGIC %python
// MAGIC display(
// MAGIC   neighbourhoods_mosaic_frame.get_resolution_metrics(sample_rows=150)
// MAGIC )

// COMMAND ----------

// MAGIC %md
// MAGIC ### Indexing using the optimal resolution

// COMMAND ----------

// MAGIC %md
// MAGIC We will use mosaic sql functions to index our points data. </br>
// MAGIC Here we will use resolution 9, index resolution depends on the dataset in use.

// COMMAND ----------

// MAGIC %python
// MAGIC tripsWithIndex = (trips
// MAGIC   .withColumn("pickup_bng", mos.point_index_geom(col("pickup_geom"), lit(optimal_resolution+1)))
// MAGIC   .withColumn("dropoff_bng", mos.point_index_geom(col("dropoff_geom"), lit(optimal_resolution+1)))
// MAGIC )

// COMMAND ----------

// MAGIC %python
// MAGIC display(tripsWithIndex)

// COMMAND ----------

// MAGIC %python
// MAGIC to_display = tripsWithIndex.select(
// MAGIC   mos.st_geomfromwkb(mos.index_geometry("pickup_bng")).alias("pickup_bng"),
// MAGIC   mos.st_geomfromwkb(mos.index_geometry("dropoff_bng")).alias("dropoff_bng"),
// MAGIC   mos.st_geomfromwkt(col("pickup_geom")).alias("pickup_geom"),
// MAGIC   mos.st_geomfromwkt(col("dropoff_geom")).alias("dropoff_geom"),
// MAGIC ).select(
// MAGIC   reproject(col("pickup_bng"), 27700, 4326).alias("pickup_bng"),
// MAGIC   reproject(col("pickup_geom"), 27700, 4326).alias("pickup_geom"),
// MAGIC   reproject(col("dropoff_bng"), 27700, 4326).alias("dropoff_bng"),
// MAGIC   reproject(col("dropoff_geom"), 27700, 4326).alias("dropoff_geom")
// MAGIC )

// COMMAND ----------

// MAGIC %python
// MAGIC to_display = tripsWithIndex.select(
// MAGIC   mos.st_geomfromwkb(mos.index_geometry("pickup_bng")).alias("pickup_bng"),
// MAGIC   
// MAGIC ).select(
// MAGIC   reproject(col("pickup_bng"), 27700, 4326).alias("pickup_bng")
// MAGIC   
// MAGIC )

// COMMAND ----------

// MAGIC %python
// MAGIC display(
// MAGIC   tripsWithIndex.select(
// MAGIC     mos.st_aswkt(mos.index_geometry("pickup_bng")).alias("pickup_bng"),
// MAGIC   ).withColumn(
// MAGIC     "bng2", mos.st_aswkt(reproject(mos.st_geomfromwkt(col("pickup_bng")), 27700, 4326))
// MAGIC   )
// MAGIC )

// COMMAND ----------

// MAGIC %python
// MAGIC x = tripsWithIndex.select(
// MAGIC     (mos.index_geometry("pickup_bng")).alias("pickup_bng"),
// MAGIC   ).withColumn(
// MAGIC     "bng2", (reproject(mos.st_geomfromwkt(col("pickup_bng")), 27700, 4326))
// MAGIC   )

// COMMAND ----------

// MAGIC %python
// MAGIC y = x.toPandas()

// COMMAND ----------

// MAGIC %python
// MAGIC %%mosaic_kepler
// MAGIC x "bng2" "geometry" 20

// COMMAND ----------

// MAGIC %md
// MAGIC We will also index our neighbourhoods using a built in generator function.

// COMMAND ----------

// MAGIC %python
// MAGIC neighbourhoodsWithIndex = (neighbourhoods
// MAGIC 
// MAGIC                            # We break down the original geometry in multiple smoller mosaic chips, each with its
// MAGIC                            # own index
// MAGIC                            .withColumn("mosaic_index", mos.mosaic_explode(col("geometry"), lit(optimal_resolution+1)))
// MAGIC 
// MAGIC                            # We don't need the original geometry any more, since we have broken it down into
// MAGIC                            # Smaller mosaic chips.
// MAGIC                            .drop("json_geometry", "geometry")
// MAGIC                           )

// COMMAND ----------

// MAGIC %python
// MAGIC display(neighbourhoodsWithIndex)

// COMMAND ----------

// MAGIC %python
// MAGIC to_display = neighbourhoodsWithIndex.where(col("properties.location_id") == 2).where(~col("mosaic_index.is_core")).select(
// MAGIC   mos.st_geomfromwkb(mos.index_geometry("mosaic_index.index_id")).alias("index_geom"),
// MAGIC   mos.st_geomfromwkb(col("mosaic_index.wkb")).alias("geom")
// MAGIC ).select(
// MAGIC   reproject(col("index_geom"), 27700, 4326).alias("index_geom"),
// MAGIC   reproject(col("geom"), 27700, 4326).alias("geom")
// MAGIC )

// COMMAND ----------

// MAGIC %python
// MAGIC %%mosaic_kepler
// MAGIC to_display "geom" "geometry" 5000

// COMMAND ----------

// MAGIC %md
// MAGIC ### Performing the spatial join

// COMMAND ----------

// MAGIC %md
// MAGIC We can now do spatial joins to both pickup and drop off zones based on geolocations in our datasets.

// COMMAND ----------

// MAGIC %python
// MAGIC pickupNeighbourhoods = neighbourhoodsWithIndex.select(col("properties.borough").alias("pickup_zone"), col("mosaic_index"))
// MAGIC 
// MAGIC withPickupZone = (
// MAGIC   tripsWithIndex.join(
// MAGIC     pickupNeighbourhoods,
// MAGIC     tripsWithIndex["pickup_bng"] == pickupNeighbourhoods["mosaic_index.index_id"]
// MAGIC   ).where(
// MAGIC     # If the borough is a core chip (the chip is fully contained within the geometry), then we do not need
// MAGIC     # to perform any intersection, because any point matching the same index will certainly be contained in
// MAGIC     # the borough. Otherwise we need to perform an st_contains operation on the chip geometry.
// MAGIC     col("mosaic_index.is_core") | mos.st_contains(col("mosaic_index.wkb"), col("pickup_geom"))
// MAGIC   ).select(
// MAGIC     "trip_distance", "pickup_geom", "pickup_zone", "dropoff_geom", "pickup_bng", "dropoff_bng"
// MAGIC   )
// MAGIC )
// MAGIC 
// MAGIC display(withPickupZone)

// COMMAND ----------

// MAGIC %md
// MAGIC We can easily perform a similar join for the drop off location.

// COMMAND ----------

// MAGIC %python
// MAGIC dropoffNeighbourhoods = neighbourhoodsWithIndex.select(col("properties.borough").alias("dropoff_zone"), col("mosaic_index"))
// MAGIC 
// MAGIC withDropoffZone = (
// MAGIC   withPickupZone.join(
// MAGIC     dropoffNeighbourhoods,
// MAGIC     withPickupZone["dropoff_bng"] == dropoffNeighbourhoods["mosaic_index.index_id"]
// MAGIC   ).where(
// MAGIC     col("mosaic_index.is_core") | mos.st_contains(col("mosaic_index.wkb"), col("dropoff_geom"))
// MAGIC   ).select(
// MAGIC     "trip_distance", "pickup_geom", "pickup_zone", "dropoff_geom", "dropoff_zone", "pickup_bng", "dropoff_bng"
// MAGIC   )
// MAGIC   .withColumn("trip_line", mos.st_astext(mos.st_makeline(array(mos.st_geomfromwkt(col("pickup_geom")), mos.st_geomfromwkt(col("dropoff_geom"))))))
// MAGIC )
// MAGIC 
// MAGIC display(withDropoffZone)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Visualise the results in Kepler

// COMMAND ----------

// MAGIC %md
// MAGIC For visualisation there simply arent good options in scala. </br>
// MAGIC Luckily in our notebooks you can easily switch to python just for UI. </br>
// MAGIC Mosaic abstracts interaction with Kepler in python.

// COMMAND ----------

// MAGIC %python
// MAGIC test_data =  withDropoffZone.limit(10000).cache()

// COMMAND ----------

// MAGIC %python
// MAGIC to_display = withDropoffZone.limit(10000).withColumn(
// MAGIC   "pickup_geom", reproject(mos.st_geomfromwkt(col("pickup_geom")), 27700, 4326)
// MAGIC ).select("pickup_geom")

// COMMAND ----------

// MAGIC %python
// MAGIC display(to_display)

// COMMAND ----------

// MAGIC %python
// MAGIC to_display.printSchema()

// COMMAND ----------

// MAGIC %python
// MAGIC display(to_display)

// COMMAND ----------

// MAGIC %python
// MAGIC %%mosaic_kepler
// MAGIC to_display "pickup_geom" "geometry" 5000

// COMMAND ----------


