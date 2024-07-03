import xml


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