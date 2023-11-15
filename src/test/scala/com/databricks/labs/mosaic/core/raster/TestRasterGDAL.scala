package com.databricks.labs.mosaic.core.raster

import com.databricks.labs.mosaic.core.raster.gdal.MosaicRasterGDAL
import com.databricks.labs.mosaic.test.mocks.filePath
import org.apache.spark.sql.test.SharedSparkSessionGDAL
import org.scalatest.matchers.should.Matchers._

import scala.sys.process._
import scala.util.Try

class TestRasterGDAL extends SharedSparkSessionGDAL {

    test("Verify that GDAL is enabled.") {
        assume(System.getProperty("os.name") == "Linux")

        val checkCmd = "gdalinfo --version"
        val resultDriver = Try(checkCmd.!!).getOrElse("")
        resultDriver should not be ""
        resultDriver should include("GDAL")

        val sc = spark.sparkContext
        val numExecutors = sc.getExecutorMemoryStatus.size - 1
        val resultExecutors = Try(
          sc.parallelize(1 to numExecutors)
              .pipe(checkCmd)
              .collect
        ).getOrElse(Array[String]())
        resultExecutors.length should not be 0
        resultExecutors.foreach(s => s should include("GDAL"))
    }

    test("Read raster metadata from GeoTIFF file.") {
        assume(System.getProperty("os.name") == "Linux")

        val testRaster = MosaicRasterGDAL.readRaster(
          filePath("/modis/MCD43A4.A2018185.h10v07.006.2018194033728_B01.TIF"),
          filePath("/modis/MCD43A4.A2018185.h10v07.006.2018194033728_B01.TIF")
        )
        testRaster.xSize shouldBe 2400
        testRaster.ySize shouldBe 2400
        testRaster.numBands shouldBe 1
        testRaster.proj4String shouldBe "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
        testRaster.SRID shouldBe 0
        testRaster.extent shouldBe Seq(-8895604.157333, 1111950.519667, -7783653.637667, 2223901.039333)
        testRaster.getRaster.GetProjection()
        noException should be thrownBy testRaster.spatialRef
        an[Exception] should be thrownBy testRaster.getBand(-1)
        an[Exception] should be thrownBy testRaster.getBand(Int.MaxValue)
        testRaster.cleanUp()
    }

    test("Read raster metadata from a GRIdded Binary file.") {
        assume(System.getProperty("os.name") == "Linux")

        val testRaster = MosaicRasterGDAL.readRaster(
          filePath("/binary/grib-cams/adaptor.mars.internal-1650626995.380916-11651-14-ca8e7236-16ca-4e11-919d-bdbd5a51da35.grib"),
          filePath("/binary/grib-cams/adaptor.mars.internal-1650626995.380916-11651-14-ca8e7236-16ca-4e11-919d-bdbd5a51da35.grib")
        )
        testRaster.xSize shouldBe 14
        testRaster.ySize shouldBe 14
        testRaster.numBands shouldBe 14
        testRaster.proj4String shouldBe "+proj=longlat +R=6371229 +no_defs"
        testRaster.SRID shouldBe 0
        testRaster.extent shouldBe Seq(-0.375, -0.375, 10.125, 10.125)
        testRaster.cleanUp()
    }

    test("Read raster metadata from a NetCDF file.") {
        assume(System.getProperty("os.name") == "Linux")

        val superRaster = MosaicRasterGDAL.readRaster(
          filePath("/binary/netcdf-coral/ct5km_baa-max-7d_v3.1_20220101.nc"),
          filePath("/binary/netcdf-coral/ct5km_baa-max-7d_v3.1_20220101.nc")
        )
        val subdatasetPath = superRaster.subdatasets("bleaching_alert_area")

        val testRaster = MosaicRasterGDAL.readRaster(
          subdatasetPath,
          subdatasetPath
        )

        testRaster.xSize shouldBe 7200
        testRaster.ySize shouldBe 3600
        testRaster.numBands shouldBe 1
        testRaster.proj4String shouldBe "+proj=longlat +a=6378137 +rf=298.2572 +no_defs"
        testRaster.SRID shouldBe 0
        testRaster.extent shouldBe Seq(-180.00000610436345, -89.99999847369712, 180.00000610436345, 89.99999847369712)

        testRaster.cleanUp()
        superRaster.cleanUp()
    }

    test("Raster pixel and extent sizes are correct.") {
        assume(System.getProperty("os.name") == "Linux")

        val testRaster = MosaicRasterGDAL.readRaster(
          filePath("/modis/MCD43A4.A2018185.h10v07.006.2018194033728_B01.TIF"),
          filePath("/modis/MCD43A4.A2018185.h10v07.006.2018194033728_B01.TIF")
        )

        testRaster.pixelXSize - 463.312716527 < 0.0000001 shouldBe true
        testRaster.pixelYSize - -463.312716527 < 0.0000001 shouldBe true
        testRaster.pixelDiagSize - 655.22312733 < 0.0000001 shouldBe true

        testRaster.diagSize - 3394.1125496954 < 0.0000001 shouldBe true
        testRaster.originX - -8895604.157333 < 0.0000001 shouldBe true
        testRaster.originY - 2223901.039333 < 0.0000001 shouldBe true
        testRaster.xMax - -7783653.637667 < 0.0000001 shouldBe true
        testRaster.yMax - 1111950.519667 < 0.0000001 shouldBe true
        testRaster.xMin - -8895604.157333 < 0.0000001 shouldBe true
        testRaster.yMin - 2223901.039333 < 0.0000001 shouldBe true

        testRaster.cleanUp()
    }

}
