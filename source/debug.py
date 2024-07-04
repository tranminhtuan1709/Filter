import process_data
import visualize_data
import xml.etree.ElementTree
import augment_data

xml_train = xml.etree.ElementTree.parse('resources/labels_ibug_300W_train.xml')
root = xml_train.getroot()

image_list = process_data.create_image_list(root)
visualize_data.visualize_original_data(image_list, 0)
cropped_images, adjusted_landmarks = process_data.crop_images(image_list)
visualize_data.visualize_adjusted_data(cropped_images, adjusted_landmarks, 0)

augment_data.augment_data(cropped_images,
                                adjusted_landmarks,
                                augment_data.transform_train)
