import numpy
import cv2


def get_triangles(landmarks: numpy.ndarray) -> numpy.ndarray:
    '''
        Get Delaunay Triangles from the list of landmark points.
        
        Args:
            landmarks (numpy.ndarray)
        
        Returns:
            A numpy.ndarray contains Delaunay Triangles, each triangle
            is presented by its coordinates x and y.
    '''

    triangle_list = []

    convexhull = cv2.convexHull(landmarks)
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    
    for point in landmarks:
        subdiv.insert(numpy.float32(point))
    
    triangles = subdiv.getTriangleList()

    for triangle in triangles:
        p1 = [triangle[0], triangle[1]]
        p2 = [triangle[2], triangle[3]]
        p3 = [triangle[4], triangle[5]]

        triangle_list.append([p1, p2, p3])
    
    return numpy.array(triangle_list, dtype=numpy.int32)


def get_corresponding_triangles(
    triangles_1: numpy.ndarray,
    landmarks_1: numpy.ndarray,
    landmarks_2: numpy.ndarray
) -> numpy.ndarray:
    '''
        Get the list of Delaunay Triangles corresponding in index 
        with the given Delaunay Triangle list.
        
        Args:
            triangles_1 (numpy.ndarray)
            landmarks_1 (numpy.ndarray)
            landmarks_2 (numpy.ndarray)
        
        Returns:
            A list containing numpy.ndarray, each numpy.ndarray
            contains coordinates of 3 points in landmarks_2 forming
            a triangle that corresponding in index with a triangle
            in the give triangle list.
    '''
    
    corresponding_triangles = []

    for triangle in triangles_1:
        p1 = triangle[0]
        p2 = triangle[1]
        p3 = triangle[2]
        
        index_p1 = numpy.where((landmarks_1 == p1).all(axis=1))[0][0]
        index_p2 = numpy.where((landmarks_1 == p2).all(axis=1))[0][0]
        index_p3 = numpy.where((landmarks_1 == p3).all(axis=1))[0][0]
                    
        corresponding_triangles.append([
                landmarks_2[index_p1],
                landmarks_2[index_p2],
                landmarks_2[index_p3]
            ]
        )
    
    return numpy.array(corresponding_triangles, dtype=numpy.int32)


def get_rectangles(triangles: numpy.ndarray) -> numpy.ndarray:
    '''
        Find bounding rectangles around given triangles.

        Args:
            triangles (numpy.ndarray)
        
        Returns:
            rectangles (numpy.ndarray)
    '''
    
    rectangles = []
    
    for triangle in triangles:
        rectangles.append(cv2.boundingRect(triangle))
    
    return numpy.array(rectangles, dtype=numpy.int32)


def crop_triangles(
    triangles: numpy.ndarray,
    rectangles: numpy.ndarray,
    image: numpy.ndarray
) -> list:
    '''
        Crop triangle parts of an image.

        Args:
            triangles (numpy.ndarray)
            rectangles (numpy.ndarray)
            image (numpy.ndarray)
        
        Returns:
            A list containing fragment triangles of the given image.
    '''
    
    cropped_triangles = []
    
    for i in range(len(triangles)):
        p1 = triangles[i][0]
        p2 = triangles[i][1]
        p3 = triangles[i][2]
        
        x, y, w, h = rectangles[i]
        
        points = numpy.array(
            [[p1[0] - x, p1[1] - y],
             [p2[0] - x, p2[1] - y],
             [p3[0] - x, p3[1] - y]]
        )
        
        image_frag = image[y:y + h, x:x + w]

        mask = numpy.zeros((h, w), dtype=numpy.uint8)

        cv2.fillConvexPoly(mask, points, (255, 255, 255))
        
        cropped_triangles.append(
            cv2.bitwise_and(image_frag, image_frag, mask=mask)
        )

    return cropped_triangles


def swap_faces(
    face_image_1: numpy.ndarray,
    face_image_2: numpy.ndarray,
    face_landmarks_1: numpy.ndarray,
    face_landmarks_2: numpy.ndarray
) -> numpy.ndarray:
    '''
        
    '''
    
    face_triangles_1 = get_triangles(face_landmarks_1)

    face_triangles_2 = get_corresponding_triangles(
        face_triangles_1, face_landmarks_1, face_landmarks_2
    )

    face_rectangles_1 = get_rectangles(face_triangles_1)

    face_rectangles_2 = get_rectangles(face_triangles_2)
        
    face_cropped_triangles_1 = crop_triangles(
        face_triangles_1, face_rectangles_1, face_image_1
    )

    face_affine_triangles_1 = []

    for i in range(len(face_triangles_1)):
        triangle_1 = face_triangles_1[i]
        triangle_2 = face_triangles_2[i]
        
        p1 = triangle_1[0]
        p2 = triangle_1[1]
        p3 = triangle_1[2]

        p4 = triangle_2[0]
        p5 = triangle_2[1]
        p6 = triangle_2[2]

        x1, y1, w1, h1 = face_rectangles_1[i]
        x2, y2, w2, h2 = face_rectangles_2[i]

        points_1 = numpy.array(
            [[p1[0] - x1, p1[1] - y1],
            [p2[0] - x1, p2[1] - y1],
            [p3[0] - x1, p3[1] - y1]],
        )

        points_2 = numpy.array(
            [[p4[0] - x2, p4[1] - y2],
            [p5[0] - x2, p5[1] - y2],
            [p6[0] - x2, p6[1] - y2]],
        )

        M = cv2.getAffineTransform(
            numpy.float32(points_1),
            numpy.float32(points_2)
        )
        

        warped_triangle = cv2.warpAffine(
            numpy.float32(face_cropped_triangles_1[i]),
            M,
            (w2, h2),
            None,
            cv2.INTER_NEAREST
        )

        face_affine_triangles_1.append(warped_triangle)

    cv2.fillConvexPoly(
        face_image_2,
        cv2.convexHull(face_landmarks_2),
        (0, 0, 0)
    )

    for i in range(len(face_triangles_2)):
        x, y, w, h = face_rectangles_2[i]
        dst_area = face_image_2[y:y + h, x:x + w]

        face_gray_area_2 = cv2.cvtColor(dst_area, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(face_gray_area_2, 1, 255, cv2.THRESH_BINARY_INV)

        face_affine_triangles_1[i] = cv2.bitwise_and(
            numpy.uint8(face_affine_triangles_1[i]),
            numpy.uint8(face_affine_triangles_1[i]),
            None,
            mask
        )

        dst_area = cv2.add(face_affine_triangles_1[i], dst_area)

        face_image_2[y:y + h, x:x + w] = dst_area

    return face_image_2
