# Chessboard_corner_detector
Este algortimo forma parte del proyecto final del curso *Procesamiento de Imágenes Digitales* de la Universidad de Ingeniería y Tecnología - UTEC. El presente algortimo realiza la detección de las esquinas de un tablero de ajedrez sin el uso de la librería OpenCV y retorna los vectores Imgpoints y Objpoints necesarios para el algortimo de calibración de una cámara digital.

## Fases del algoritmo
El algortimo está conformado por los siguiente pasos:

**1. Algoritmo de detección de Harris**: Se usa el algoritmo de detección de esquinas de Harris, donde se diferencian espacios planos bordes y esquinas.

![1 Harris](https://github.com/EdwardCaleb/Chessboard_corner_detector/assets/40768170/5612a61e-12a4-4167-8f3e-4e6793b818cd)

**2. Clustering mediante DBSCAN**: Mediante DBSCAN se agrupan elementos cercanos. Se realiza para agrupar las esquinas deseadas.

![2 DBSCAN](https://github.com/EdwardCaleb/Chessboard_corner_detector/assets/40768170/ff7774d6-eb6c-4653-8f0f-e326d8977334)

**3. Extracción de centroides de los grupos obtenidos**: Se extraen los centroides de los elementos de cada grupo. Se realiza para obtener una única coordenada por esquina. 

![3 Centroid](https://github.com/EdwardCaleb/Chessboard_corner_detector/assets/40768170/64887d23-4a23-40b0-95ef-2e267c25c079)
![3 Centroid_2](https://github.com/EdwardCaleb/Chessboard_corner_detector/assets/40768170/631a14b5-a17d-4b6f-b377-1d2fed2ab9e9)

**4. Filtro de elementos lejanos**: Se filtran mediante conexiones nodales los elementos más lejanos comunmente no pertencientes a las esquinas del tablero.

![4 FarFilter](https://github.com/EdwardCaleb/Chessboard_corner_detector/assets/40768170/9df9ab8b-c454-4cdb-8512-d8288ecaab37)
![4 FarFilter_2](https://github.com/EdwardCaleb/Chessboard_corner_detector/assets/40768170/0cce8676-a444-4b9f-baaa-5b2b5da3c5fc)

**5. Transformación de perspectiva**: La transformación de perspectiva se realiza para poder usar algoritmos más simplificados sobre un conjunto de puntos que no poseen deformación ni rotación por perspectiva propios de la obtención de las imágenes mediante la cámara.

![5 Perspective](https://github.com/EdwardCaleb/Chessboard_corner_detector/assets/40768170/1faff897-c3c3-4c36-b7b1-24591b68411e)

**6. Exclusión de extremos**: Se excluyen los puntos de los extremos del tablero para mantener solo los centrales.

![6 Exclusion](https://github.com/EdwardCaleb/Chessboard_corner_detector/assets/40768170/ef4188d5-c08d-41d1-b0b8-e3f603e95618)
![6 Exclusion_2](https://github.com/EdwardCaleb/Chessboard_corner_detector/assets/40768170/dbbd4f70-0edc-477b-9730-4cdccc54b2b1)

**7. Reordenamiento de puntos**: Los puntos se reordenan geométricamente para la obtención de los vectores Imgpoints y Objpoints.

![7 Reordering](https://github.com/EdwardCaleb/Chessboard_corner_detector/assets/40768170/864396a4-783d-4007-b767-51ab96e2db4b)
