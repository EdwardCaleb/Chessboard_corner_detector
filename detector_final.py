import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import distance


# Convolucion 2D para detector de Harris
def convolve2d(image, kernel):
    m, n = kernel.shape
    y, x = image.shape
    y = y - m + 1
    x = x - n + 1
    result = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            result[i, j] = np.sum(image[i:i+m, j:j+n] * kernel)
    return result

# Detector de esquinas Harris
def corner_detection(src_image_path, threshold, show = False):
    image = io.imread(src_image_path)

    gray = np.mean(image, axis=2)

    dx = np.gradient(gray, axis=1)
    dy = np.gradient(gray, axis=0)

    dx2 = dx ** 2
    dy2 = dy ** 2
    dxy = dx * dy

    window = np.ones((3, 3))
    dx2 = convolve2d(dx2, window)
    dy2 = convolve2d(dy2, window)
    dxy = convolve2d(dxy, window)

    corner_response = dx2 * dy2 - dxy ** 2 - 0.04 * (dx2 + dy2) ** 2

    corners = np.argwhere(corner_response > threshold)

    if show == True:
        # Graficar la imagen y los puntos encontrados
        fig, ax = plt.subplots()
        # Mostrar la imagen en los ejes
        ax.imshow(image)
        # Extraer las coordenadas x e y por separado
        x_coords, y_coords = zip(*corners)
        # Graficar los puntos sobre la imagen
        ax.scatter(y_coords, x_coords, color='red', marker='o')
        # Mostrar el gráfico
        plt.show()

    return corners


def segmentacion_DBSCAN(coordenadas, eps=5, min_samples=2, show = False):
    # Creando un objeto DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Ajustando el modelo a las coordenadas
    dbscan.fit(coordenadas)

    # Obteniendo las etiquetas de los grupos
    etiquetas = dbscan.labels_

    # Convirtiendo las coordenadas a un array de numpy
    coordenadas = np.array(coordenadas)

    # Obteniendo las coordenadas de los puntos en cada grupo
    grupos = np.unique(etiquetas)
    coordenadas_grupos = [coordenadas[etiquetas == grupo] for grupo in grupos]

    if show == True:
        # Graficando los puntos por grupos
        for i, grupo in enumerate(coordenadas_grupos):
            plt.scatter(grupo[:, 1], -grupo[:, 0], label=f"Grupo {i}")

        # Graficando los puntos de ruido (etiqueta -1)
        ruido = coordenadas[etiquetas == -1]
        plt.scatter(ruido[:, 1], ruido[:, 0], color='black', label="Ruido")

        # plt.legend()
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.title("Grupos obtenidos mediante DBSCAN")
        plt.show()

    return coordenadas_grupos, grupos


def Extraccion_centroide(coordenadas_grupos, grupos, show = False):

    # Convirtiendo las coordenadas a un array de numpy para facilitar la manipulación
    matriz_cent = np.ones((len(grupos),2))


    # Extraer centroides
    for i, grupo in enumerate(coordenadas_grupos):
        centroide = np.mean(grupo, axis=0)
        matriz_cent[i,:] = centroide

    if show == True:
        # Graficando los puntos por grupos
        for i, grupo in enumerate(coordenadas_grupos):
            plt.scatter(grupo[:, 1], -grupo[:, 0], label=f"Grupo {i}")
            plt.plot(matriz_cent[i,1], -matriz_cent[i,0], marker='x', color='red')

        # plt.legend()
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.title("Grupos obtenidos mediante DBSCAN")
        plt.show()

    return matriz_cent


def filtro_nodos(coordenadas, num_nodos=97, dist_max=100, show=False):
    # Distancia máxima de conexión
    distancia_maxima = dist_max

    # Calcula la matriz de distancias
    dist_matrix = distance.cdist(coordenadas, coordenadas, 'euclidean')

    # Crea una matriz de conexiones
    num_puntos = len(coordenadas)
    conexiones = np.zeros((num_puntos, num_puntos))

    # Conecta los puntos dentro del umbral de distancia máxima
    for i in range(num_puntos):
        conexiones[i, np.where(dist_matrix[i] <= distancia_maxima)[0]] = 1

    # Calcula el número de conexiones de cada nodo
    num_conexiones = np.sum(conexiones, axis=1)

    # Obtén los índices de los nodos con más conexiones
    indices_nodos_mas_conexiones = np.argsort(num_conexiones)[-num_nodos:]

    # Obtén las coordenadas de los nodos con más conexiones
    coord_nodos_mas_conexiones = coordenadas[indices_nodos_mas_conexiones]

    if show == True:
        # Grafica los puntos y las conexiones
        plt.scatter(coordenadas[:, 0], coordenadas[:, 1], color='b')
        for i in range(num_puntos):
            for j in range(num_puntos):
                if conexiones[i, j] == 1:
                    plt.plot([coordenadas[i, 0], coordenadas[j, 0]], [coordenadas[i, 1], coordenadas[j, 1]], color='r')
        plt.scatter(coord_nodos_mas_conexiones[:, 0], coord_nodos_mas_conexiones[:, 1], color='g', marker='x')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Conexiones de puntos')
        plt.show()

    return coord_nodos_mas_conexiones



def transformada_perspectiva(coordenadas, show = False):
    def encontrar_puntos_lejanos(coordenadas):

        coordenadas = np.asarray(coordenadas)

        top_izquierda = coordenadas[0][:]
        top_derecha = coordenadas[0][:]
        abajo_izquierda = coordenadas[0][:]
        abajo_derecha = coordenadas[0][:]

        for punto in coordenadas:
            x, y = punto

            diff_ti = punto - top_izquierda
            diff_td = punto - top_derecha
            diff_ai = punto - abajo_izquierda
            diff_ad = punto - abajo_derecha


            if - diff_ti[0] - diff_ti[1] > 0:
                top_izquierda = punto

            if - diff_td[0] + diff_td[1] > 0:
                top_derecha = punto

            if + diff_ai[0] - diff_ai[1] > 0:
                abajo_izquierda = punto

            if + diff_ad[0] + diff_ad[1] > 0:
                abajo_derecha = punto

        return top_izquierda, top_derecha, abajo_izquierda, abajo_derecha


    [top_izquierda, top_derecha, abajo_izquierda, abajo_derecha] = encontrar_puntos_lejanos(coordenadas)

    # Puntos distorsionados
    tx1 = top_izquierda[0]
    ty1 = top_izquierda[1]
    tx2 = top_derecha[0]
    ty2 = top_derecha[1]
    tx3 = abajo_izquierda[0]
    ty3 = abajo_izquierda[1]
    tx4 = abajo_derecha[0]
    ty4 = abajo_derecha[1]

    # Puntos reales
    x1 = 0
    y1 = 0
    x2 = 7
    y2 = 0
    x3 = 1
    y3 = 10
    x4 = 8
    y4 = 10

    # Puntos transformados
    puntos_transformados = np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1], [x4, y4, 1]], dtype=np.float32)

    # Puntos originales
    puntos_originales = np.array([[tx1, ty1], [tx2, ty2], [tx3, ty3], [tx4, ty4]], dtype=np.float32)

    # Construir la matriz A
    A = np.zeros((8, 8))
    b = np.zeros((8, 1))

    for i in range(4):
        X, Y = puntos_originales[i]
        x, y, _ = puntos_transformados[i]
        A[2 * i] = [X, Y, 1, 0, 0, 0, -x * X, -x * Y]
        A[2 * i + 1] = [0, 0, 0, X, Y, 1, -y * X, -y * Y]
        b[2 * i] = x
        b[2 * i + 1] = y

    # Resolver el sistema de ecuaciones lineales
    h = np.linalg.solve(A, b)

    # Obtener la ecuación inversa de transformación
    h11, h12, h13, h21, h22, h23, h31, h32 = h.flatten()

    new_points = np.zeros((len(coordenadas),2))

    for i in range(len(coordenadas)):
        x = coordenadas[i][0]
        y = coordenadas[i][1]
        nx = (h11*x + h12*y + h13) / (h31*x + h32*y + 1)
        ny = (h21*x + h22*y + h23) / (h31*x + h32*y + 1)

        new_points[i][0] = nx
        new_points[i][1] = ny

    if show == True:
        x = [punto[0] for punto in new_points]
        y = [punto[1] for punto in new_points]

        x_coords, y_coords = zip(*new_points)
        # Graficar los puntos sobre la imagen
        plt.scatter(y_coords, x_coords, color='red', marker='.')
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.title('Puntos con transformación de perspectiva')
        plt.grid(False)
        plt.xlim(-2, 12)  # Rango en el eje x
        plt.ylim(-2, 11)  # Rango en el eje y
        plt.show()

    return new_points




def filtrar_bordes(coordenadas, coor_pers, show = False):
    coordenadas = np.asarray(coordenadas)
    coor_pers = np.asarray(coor_pers)

    # Matriz vacía
    filtered_points = np.empty((0, 2))  # puntos filtrados
    filtered_pers_points = np.empty((0, 2)) # puntos de la perspectiva filtrados

    for i in range(len(coordenadas)):
        if coor_pers[i][0]>0.5 and coor_pers[i][0]<7.5 and coor_pers[i][1]>0.5 and coor_pers[i][1]<9.5:
            filtered_points = np.vstack((filtered_points, coordenadas[i][:]))
            filtered_pers_points = np.vstack((filtered_pers_points, coor_pers[i][:]))

    if show == True:
        x = [punto[0] for punto in filtered_pers_points]
        y = [punto[1] for punto in filtered_pers_points]

        x_coords, y_coords = zip(*filtered_pers_points)
        # Graficar los puntos sobre la imagen
        plt.scatter(y_coords, x_coords, color='red', marker='.')
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.title('Puntos con transformación de perspectiva con bordes filtrados')
        plt.grid(False)
        plt.xlim(-2, 12)  # Rango en el eje x
        plt.ylim(-2, 11)  # Rango en el eje y
        plt.show()

    return filtered_points, filtered_pers_points



def ordenar_puntos(coordenadas, coor_pers):
    # Redondear coordenadas de perspectiva
    coor_pers = np.round(coor_pers)

    # generar vector ordenado
    vector_pers_ord = np.zeros((63, 2))
    contador = 0
    for i in range(1, 10):
        for j in range(1, 8):
            vector_pers_ord[contador] = (j, i)
            contador += 1

    # Redondear coordenadas de perspectiva ordenadas
    vector_pers_ord = np.round(vector_pers_ord)

    # algoritmo de ordenamiento
    puntos_ordenados = np.empty((0, 2))  # crear matriz vacia
    for i in range(63):
        buscar = vector_pers_ord[i][:]
        for j in range(63):
            match = coor_pers[j][:]
            if buscar[0] == match[0] and buscar[1] == match[1]:
                puntos_ordenados = np.vstack((puntos_ordenados, coordenadas[j][:]))

    # generar el object_point
    columna_ceros = np.zeros((vector_pers_ord.shape[0], 1))
    object_point = np.hstack((vector_pers_ord-1, columna_ceros))


    print(puntos_ordenados)
    print(object_point)

    return(puntos_ordenados, object_point)



def graficar_puntos(src_image_path, points):
    image = io.imread(src_image_path)
    # Graficar la imagen y los puntos encontrados
    fig1, ax = plt.subplots()
    # Mostrar la imagen en los ejes
    ax.imshow(image)
    # Extraer las coordenadas x e y por separado
    x_coords, y_coords = zip(*points)
    # Graficar los puntos sobre la imagen
    ax.scatter(y_coords, x_coords, color='red', marker='.')
    # Mostrar el gráfico
    plt.show()


def graficar_puntos_unidos(src_image_path, points):
    image = io.imread(src_image_path)
    # Graficar la imagen y los puntos encontrados
    fig1, ax = plt.subplots()
    # Mostrar la imagen en los ejes
    ax.imshow(image)
    # Extraer las coordenadas x e y por separado
    x_coords, y_coords = zip(*points)
    # Graficar los puntos sobre la imagen
    ax.scatter(y_coords, x_coords, color='red', marker='.')
    #extraer x e y en vectores
    x = [punto[0] for punto in points]
    y = [punto[1] for punto in points]
    # plotear linea
    plt.plot(y, x, '-o', color='green', markersize=0)
    # Mostrar el gráfico
    plt.show()


show_plots = True

# Ruta de la imagen del tablero de ajedrez
src_image_path = "imagenes/IMG_2979red.JPG"
# src_image_path = "img_red/IMG_3151_red.JPG"
# src_image_path = "oth_img/IMG_2909.JPG"

# Detección Harris
corners = corner_detection(src_image_path,100000000, show=show_plots)

# Segmentacion DBSCAN
[coordenadas_grupos, grupos] = segmentacion_DBSCAN(corners, eps=1, min_samples=2, show = show_plots)

# Ubicación de centroide de los grupos hallados
centroides = Extraccion_centroide(coordenadas_grupos, grupos,show=show_plots)
graficar_puntos(src_image_path,centroides)

# Se filtran los elementos lejanos mediante sus conexiones
points = filtro_nodos(centroides, 97, 100, show=show_plots)
graficar_puntos(src_image_path,points)

# Se realiza una transformacion de perspectiva
points_pers = transformada_perspectiva(points, show=show_plots)

# Se excluyen los puntos del los bordes del tablero
[puntos_filtrados, puntos_pers_filtrados] = filtrar_bordes(points, points_pers, show=show_plots)
graficar_puntos(src_image_path, puntos_filtrados)

[puntos_ordenados, object_point] = ordenar_puntos(puntos_filtrados, puntos_pers_filtrados)

graficar_puntos_unidos(src_image_path, puntos_ordenados)
