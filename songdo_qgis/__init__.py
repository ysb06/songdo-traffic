# -*- coding: utf-8 -*-
"""
/***************************************************************************
 SongdoQGIS
                                 A QGIS plugin
 Songdo QGIS Linker
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2024-07-16
        copyright            : (C) 2024 by sbyim
        email                : ysb06@hotmail.com
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load SongdoQGIS class from file SongdoQGIS.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .songdo_qgis import SongdoQGIS
    return SongdoQGIS(iface)