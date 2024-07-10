import numpy
import cv2
import pandas

import delaunay_triangle


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
    filter_image = cv2.cvtColor(filter_image, cv2.COLOR_BGR2RGB)
    
    df = pandas.read_csv(filter_path + '_annotations.csv')
    
    x = df.iloc[:, 1].values
    y = df.iloc[:, 2].values
    
    filter_landmark = numpy.column_stack((x, y))

    return filter_image, filter_landmark


def show_triangles(triangles: numpy.ndarray, img: numpy.ndarray) -> None:
    '''
        Show Delaunay Triangles on the given image.

        Args:
            triangles (numpy.ndarray): contains delaunay triangles.
            img (numpy.ndarray): the given image.
        
        Returns:
            None
    '''

    for triangle in triangles:
        p1 = triangle[0]
        p2 = triangle[1]
        p3 = triangle[2]

        cv2.line(img=img, pt1=p1, pt2=p2, color=(0, 255, 0), thickness=1)
        cv2.line(img=img, pt1=p2, pt2=p3, color=(0, 255, 0), thickness=1)
        cv2.line(img=img, pt3=p1, pt2=p1, color=(0, 255, 0), thickness=1)
    
    cv2.imshow(winname='Delaunay Trianglation', mat=img)


def show_landmark(landmark: numpy.ndarray, img: numpy.ndarray) -> None:
    '''
        Show landmark points on the given image.

        Args:
            landmark (numpy.ndarray): the given landmark points to be shown.
            img (numpy.ndarray): the given image.
        
        Returns:
            None
    '''

    for point in landmark:
        cv2.circle(
            img=img, center=point, radius=2, color=(255, 0, 0), thickness=-1
        )
    
    cv2.imshow('Landmark Points', img)


def apply_filter(
    face_image: numpy.ndarray,
    face_landmark: numpy.ndarray,
    filter_image: numpy.ndarray,
    filter_landmark: numpy.ndarray,
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
        )

        filter_affine_triangles.append(warped_triangle)

    for i in range(len(face_triangles)):
        x, y, w, h = face_rectangles[i]
        dst_area = face_image[y:y + h, x:x + w]
        dst_area = cv2.add(
            src1=numpy.int32(filter_affine_triangles[i]),
            src2=numpy.int32(dst_area),
        )

        face_image[y:y + h, x:x + w] = dst_area
    
    return face_image
