package com.databricks.mosaic.core.geometry.polygon

import com.databricks.mosaic.core.geometry.MosaicGeometryOGC.spatialReference
import com.databricks.mosaic.core.geometry.multipolygon.MosaicMultiPolygonOGC
import com.databricks.mosaic.core.geometry.point.{MosaicPoint, MosaicPointOGC}
import com.databricks.mosaic.core.geometry.{GeometryReader, MosaicGeometry, MosaicGeometryJTS, MosaicGeometryOGC}
import com.databricks.mosaic.core.types.model.GeometryTypeEnum.POLYGON
import com.databricks.mosaic.core.types.model.{GeometryTypeEnum, InternalCoord, InternalGeometry}
import com.esri.core.geometry.ogc.{OGCGeometry, OGCLineString, OGCPolygon}
import org.apache.spark.sql.catalyst.InternalRow

class MosaicPolygonOGC(polygon: OGCPolygon)
  extends MosaicGeometryOGC(polygon) with MosaicPolygon {

  override def toInternal: InternalGeometry = {
    def ringToInternalCoords(ring: OGCLineString): Array[InternalCoord] = {
      for (i <- 0 until ring.numPoints())
        yield InternalCoord(MosaicPointOGC(ring.pointN(i)).coord)
    }.toArray

    val boundary = polygon.boundary().geometryN(0).asInstanceOf[OGCLineString]
    val shell = ringToInternalCoords(boundary)
    val holes = for (i <- 0 until polygon.numInteriorRing())
      yield ringToInternalCoords(polygon.interiorRingN(i))

    new InternalGeometry(POLYGON.id, Array(shell), Array(holes.toArray))
  }

  override def getBoundary: Seq[MosaicPoint] = getBoundaryPoints

  override def getBoundaryPoints: Seq[MosaicPoint] = {
    MosaicPolygonOGC.getPoints(polygon.exteriorRing())
  }

  override def getHoles: Seq[Seq[MosaicPoint]] = getHolePoints

  override def getHolePoints: Seq[Seq[MosaicPoint]] = {
    for (i <- 0 until polygon.numInteriorRing())
      yield MosaicPolygonOGC.getPoints(polygon.interiorRingN(i))
  }

  override def getLength: Double = MosaicGeometryOGC(polygon.boundary()).getLength

  override def flatten: Seq[MosaicGeometry] = List(this)
}

object MosaicPolygonOGC extends GeometryReader {

  def apply(ogcGeometry: OGCGeometry): MosaicPolygonOGC = {
    new MosaicPolygonOGC(ogcGeometry.asInstanceOf[OGCPolygon])
  }

  def getPoints(lineString: OGCLineString): Seq[MosaicPoint] = {
    for (i <- 0 until lineString.numPoints())
      yield MosaicPointOGC(lineString.pointN(i))
  }

  override def fromInternal(row: InternalRow): MosaicGeometry = {
    val internalGeom = InternalGeometry(row)
    val polygon = MosaicMultiPolygonOGC.createPolygon(internalGeom.boundaries, internalGeom.holes)
    MosaicGeometryOGC(new OGCPolygon(polygon, spatialReference))
  }

  override def fromPoints(inPoints: Seq[MosaicPoint], geomType: GeometryTypeEnum.Value = POLYGON): MosaicGeometry = {
    require(geomType.id == POLYGON.id)
    val boundary = inPoints.map(_.coord).map(InternalCoord(_)).toArray
    val polygon = MosaicMultiPolygonOGC.createPolygon(Array(boundary), Array(Array(Array())))
    MosaicGeometryOGC(new OGCPolygon(polygon, spatialReference))
  }

  override def fromWKB(wkb: Array[Byte]): MosaicGeometry = MosaicGeometryJTS.fromWKB(wkb)

  override def fromWKT(wkt: String): MosaicGeometry = MosaicGeometryJTS.fromWKT(wkt)

  override def fromJSON(geoJson: String): MosaicGeometry = MosaicGeometryJTS.fromJSON(geoJson)

  override def fromHEX(hex: String): MosaicGeometry = MosaicGeometryJTS.fromHEX(hex)

}