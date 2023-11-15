package com.databricks.labs.mosaic.core.raster.operator.gdal

import com.databricks.labs.mosaic.core.raster.gdal.MosaicRasterGDAL
import org.gdal.gdal.{TranslateOptions, gdal}

import java.nio.file.{Files, Paths}

/** GDALTranslate is a wrapper for the GDAL Translate command. */
object GDALTranslate {

    /**
      * Executes the GDAL Translate command.
      *
      * @param outputPath
      *   The output path of the translated file.
      * @param isTemp
      *   Whether the output is a temp file.
      * @param raster
      *   The raster to translate.
      * @param command
      *   The GDAL Translate command.
      * @return
      *   A MosaicRaster object.
      */
    def executeTranslate(outputPath: String, isTemp: Boolean, raster: => MosaicRasterGDAL, command: String): MosaicRasterGDAL = {
        require(command.startsWith("gdal_translate"), "Not a valid GDAL Translate command.")
        val translateOptionsVec = OperatorOptions.parseOptions(command)
        val translateOptions = new TranslateOptions(translateOptionsVec)
        val result = gdal.Translate(outputPath, raster.getRaster, translateOptions)
        val size = Files.size(Paths.get(outputPath))
        MosaicRasterGDAL(result, outputPath, isTemp, raster.getParentPath, raster.getDriversShortName, size).flushCache()
    }

}
