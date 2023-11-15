package com.databricks.labs.mosaic.core.raster.operator.gdal

import com.databricks.labs.mosaic.core.raster.gdal.MosaicRasterGDAL
import org.gdal.gdal.{BuildVRTOptions, gdal}

/** GDALBuildVRT is a wrapper for the GDAL BuildVRT command. */
object GDALBuildVRT {

    /**
      * Executes the GDAL BuildVRT command.
      *
      * @param outputPath
      *   The output path of the VRT file.
      * @param isTemp
      *   Whether the output is a temp file.
      * @param rasters
      *   The rasters to build the VRT from.
      * @param command
      *   The GDAL BuildVRT command.
      * @return
      *   A MosaicRaster object.
      */
    def executeVRT(outputPath: String, isTemp: Boolean, rasters: => Seq[MosaicRasterGDAL], command: String): MosaicRasterGDAL = {
        require(command.startsWith("gdalbuildvrt"), "Not a valid GDAL Build VRT command.")
        val vrtOptionsVec = OperatorOptions.parseOptions(command)
        val vrtOptions = new BuildVRTOptions(vrtOptionsVec)
        val result = gdal.BuildVRT(outputPath, rasters.map(_.getRaster).toArray, vrtOptions)
        // TODO: Figure out multiple parents, should this be an array?
        // VRT files are just meta files, mem size doesnt make much sense so we keep -1
        MosaicRasterGDAL(result, outputPath, isTemp, rasters.head.getParentPath, "VRT", -1).flushCache()
    }

}
