import numpy
import cv2

    
def get_triangle_list(landmark: numpy.ndarray) -> numpy.ndarray:
    '''
        Get Delaunay Triangles from the list of landmark points.
        
        Args:
            landmarks (numpy.ndarray)
        
        Returns:
            A numpy.ndarray contains Delaunay Triangles, each triangle
            is presented by its coordinates x and y.
    '''

    triangle_list = []

    convexhull = cv2.convexHull(landmark)
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    
    for point in landmark:
        subdiv.insert(numpy.float32(point))
    
    triangles = subdiv.getTriangleList()

    for triangle in triangles:
        p1 = [triangle[0], triangle[1]]
        p2 = [triangle[2], triangle[3]]
        p3 = [triangle[4], triangle[5]]

        triangle_list.append([p1, p2, p3])
    
    return numpy.array(triangle_list, dtype=numpy.int32)


def get_corresponding_triangles(
    triangle_list: numpy.ndarray,
    landmarks_1: numpy.ndarray,
    landmarks_2: numpy.ndarray
) -> numpy.ndarray:
    '''
        Get the list of Delaunay Triangles corresponding in index 
        with the given Delaunay Triangle list.
        
        Args:
            triangle_list (numpy.ndarray)
            landmarks_1 (numpy.ndarray)
            landmarks_2 (numpy.ndarray)
        
        Returns:
            A list containing numpy.ndarray, each numpy.ndarray
            contains coordinates of 3 points in landmarks_2 forming
            a triangle that corresponding in index with a triangle
            in the give triangle list.
    '''
    
    corresponding_triangles = []

    for triangle in triangle_list:
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


def get_rectangle_list(triangle_list: numpy.ndarray) -> numpy.ndarray:
    '''
        Find bounding rectangles around given triangles.

        Args:
            triangle_list (numpy.ndarray)
        
        Returns:
            rectangle_list (numpy.ndarray)
    '''
    
    rectangle_list = []
    
    for triangle in triangle_list:
        rectangle_list.append(cv2.boundingRect(triangle))
    
    return numpy.array(rectangle_list, dtype=numpy.int32)


def crop_triangle(
    triangle_list: numpy.ndarray,
    rectangle_list: numpy.ndarray,
    image: numpy.ndarray
) -> list:
    '''
        Crop triangle parts of an image.

        Args:
            triangle_list (numpy.ndarray)
            rectangle_list (numpy.ndarray)
            image (numpy.ndarray)
        
        Returns:
            A list containing fragment triangles of the given image.
    '''
    
    cropped_triangles = []
    
    for i in range(len(triangle_list)):
        p1 = triangle_list[i][0]
        p2 = triangle_list[i][1]
        p3 = triangle_list[i][2]
        
        rect_x, rect_y, rect_w, rect_h = rectangle_list[i]
        
        points = numpy.array(
            [[p1[0] - rect_x, p1[1] - rect_y],
             [p2[0] - rect_x, p2[1] - rect_y],
             [p3[0] - rect_x, p3[1] - rect_y]]
        )
        
        cropped_frag = image[
            rect_y:rect_y + rect_h,
            rect_x:rect_x + rect_w
        ]

        mask = numpy.zeros(cropped_frag.shape, dtype=cropped_frag.dtype)

        cv2.fillConvexPoly(mask, points, (255, 255, 255))
        
        cropped = cv2.bitwise_and(cropped_frag, mask)

        cropped_triangles.append(cropped)

    return cropped_triangles
