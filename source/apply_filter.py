import numpy
import cv2
import pandas


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


def crop_triangle(
    triangles: numpy.ndarray,
    rectangles: numpy.ndarray,
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


def load_filter(filter_path: str) -> tuple:
    '''
        Load the filter image and its csv file.

        Args:
            filter_path (str): name of the filter (not containing the
            file type)

        Returns:
            filter image (numpy.ndarray): the filter image.
            filter_landmark (numpy.ndarray): landmark points of the filter.
    '''
    
    filter_image = cv2.imread(filter_path + '.png')
    
    df = pandas.read_csv(filter_path + '_annotations.csv')
    
    x = df.iloc[:, 1].values
    y = df.iloc[:, 2].values
    
    filter_landmarks = numpy.column_stack((x, y))

    return filter_image, filter_landmarks


def apply_filter(
    face_image: numpy.ndarray,
    face_landmark: numpy.ndarray,
    filter_image: numpy.ndarray,
    filter_landmark: numpy.ndarray,
    flag: str
) -> numpy.ndarray:
    '''
        Apply filter image onto the given image.

        Args:
            face_image (numpy.ndarray): the image containing human faces to
            be applied a filter.
            face_landmark (numpy.ndarray): landmark points of one human face
            on the given image.
            filter_image (numpy.ndarray): the image of a filter.
            filter_landmark (numpy.ndarray): landmark points of a filter.
        
        Returns:
            result (numpy.ndarray): an image that is applied a filter.
    '''
    try:

        face_clone = numpy.array(face_image)

        face_triangles = delaunay_triangle.get_triangle_list(
            landmark=face_landmark
        )

        filter_triangles = delaunay_triangle.get_corresponding_triangles(
            triangle_list=face_triangles,
            landmarks_1=face_landmark,
            landmarks_2=filter_landmark
        )

        face_rectangles = delaunay_triangle.get_rectangle_list(
            triangle_list=face_triangles
        )

        filter_rectangles = delaunay_triangle.get_rectangle_list(
            triangle_list=filter_triangles
        )
            
        filter_cropped_triangles = delaunay_triangle.crop_triangle(
            triangle_list=filter_triangles,
            rectangle_list=filter_rectangles,
            image=filter_image
        ) 

        cv2.fillConvexPoly(
            img=face_image,
            points=cv2.convexHull(points=face_landmark),
            color=(0, 0, 0)
        )

        img2 = face_image
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        filter_affine_triangles = []

        for i in range(len(face_triangles)):
            triangle_1 = face_triangles[i]
            triangle_2 = filter_triangles[i]
            
            p1 = triangle_1[0]
            p2 = triangle_1[1]
            p3 = triangle_1[2]

            p4 = triangle_2[0]
            p5 = triangle_2[1]
            p6 = triangle_2[2]

            x1, y1, w1, h1 = face_rectangles[i]
            x2, y2, w2, h2 = filter_rectangles[i]

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
                numpy.float32(points_2),
                numpy.float32(points_1)
            )
            

            warped_triangle = cv2.warpAffine(
                src=numpy.float32(filter_cropped_triangles[i]),
                M=M,
                dsize=(w1, h1),
                flags=cv2.INTER_NEAREST
            )

            filter_affine_triangles.append(warped_triangle)

        for i in range(len(face_triangles)):
            x, y, w, h = face_rectangles[i]
            dst_area = face_image[y:y + h, x:x + w]

            img2_new_area_gray = cv2.cvtColor(dst_area, cv2.COLOR_BGR2GRAY)
            _, mask_designed = cv2.threshold(img2_new_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            filter_affine_triangles[i] = cv2.bitwise_and(
                numpy.uint8(filter_affine_triangles[i]),
                numpy.uint8(filter_affine_triangles[i]),
                mask=mask_designed
            )

            dst_area = cv2.add(filter_affine_triangles[i], dst_area)
            
            if numpy.int32(filter_affine_triangles[i]).size\
                != numpy.uint8(dst_area).size:
                return None

            face_image[y:y + h, x:x + w] = dst_area

        if flag == 'filter':
            return face_image

        convexhull2 = cv2.convexHull(face_landmark)
        img2_face_mask = numpy.zeros_like(img2_gray)
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
        img2_face_mask = cv2.bitwise_not(img2_head_mask)


        img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
        result = cv2.add(img2_head_noface, face_image)

        (x, y, w, h) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

        seamlessclone = cv2.seamlessClone(result, face_clone, img2_head_mask, center_face2, cv2.MIXED_CLONE)
        return seamlessclone
    except:
        return face_image
