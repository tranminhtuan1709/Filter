import xml
from PIL import Image
import matplotlib
import matplotlib.pyplot


def print_element(root: xml.etree.ElementTree) -> None:
    '''
        Print information about tab name, attributes, and text
        of all elements in an XML file.

        Args:
            root (xml.etree.ElementTree): the root of the xml file.
        Returns:
            None
    '''
    
    print(f'Tag: {root.tag}')
    print(f'Attributes: {root.attrib}')
    print(f'Text: {root.text}')
    print()

    for child in root:
        print_element(child)


def visualize_original_data(image_list: list, index: int) -> None:
    '''
        Visualize original images, bounding box and landmark points.

        Args:
            image_list (list)
            index (int)
        
        Returns:
            None
    '''

    print(f'File name: {image_list[index]['filename']}')
    print(f'Image width: {image_list[index]['image_width']}')
    print(f'Image height: {image_list[index]['image_height']}')
    print(f'Box top: {image_list[index]['box_top']}')
    print(f'Box left: {image_list[index]['box_left']}')
    print(f'Box width: {image_list[index]['box_width']}')
    print(f'Box height: {image_list[index]['box_height']}')

    image = Image.open('resources\\' + image_list[index]['filename']).convert('RGB')
    
    matplotlib.pyplot.imshow(image)

    for x, y in image_list[index]['landmark_list']:
        matplotlib.pyplot.scatter(x, y, s=1, c='red', marker='o', linewidths=1)

    rect = matplotlib.pyplot.Rectangle((image_list[index]['box_left'],
                                        image_list[index]['box_top']),
                                        image_list[index]['box_width'],
                                        image_list[index]['box_height'],
                                        fill=False,
                                        edgecolor='green',
                                        linewidth=2)
    matplotlib.pyplot.gca().add_patch(rect)

    matplotlib.pyplot.show()

def visualize_adjusted_data(cropped_images: list,
                            adjusted_landmarks: list,
                            index: int) -> None:
    '''
        Visualize cropped images and adjusted landmark points.

        Args:
            cropped_images (list)
            adjusted_landmarks (list)

        Returns:
            None
    '''

    matplotlib.pyplot.imshow(cropped_images[index])

    for x, y in adjusted_landmarks[index]:
        matplotlib.pyplot.scatter(x, y, s=1, c='red', marker='o', linewidths=1)
    
    matplotlib.pyplot.show()
