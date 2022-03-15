package com.databricks.labs

import com.databricks.labs.mosaic.core.geometry.api.GeometryAPI
import com.databricks.labs.mosaic.core.index.IndexSystem

package object mosaic {

    val JTS: GeometryAPI = mosaic.core.geometry.api.GeometryAPI.JTS
    val OGC: GeometryAPI = mosaic.core.geometry.api.GeometryAPI.OGC
    val H3: IndexSystem = mosaic.core.index.H3IndexSystem

}