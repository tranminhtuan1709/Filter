import matplotlib.pyplot
import torch
from torch.utils.data import DataLoader
import matplotlib


class LandmarkDataset(DataLoader):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        '''
            Returns the lenght of labels.

            Args:

            Returns:
                An integer value that is the lenght of labels.
        '''

        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        '''
            Returns specified items at index idx.

            Args:
                idx (int): the index of the required item.
            
            Returns:
                A tuple containing required items.
        '''

        image = self.images[idx]
        label = self.labels[idx]

        return image, label

def draw(image, landmark):
    matplotlib.pyplot.imshow(image)
    landmark = (landmark + 0.5) * 224
    matplotlib.pyplot.scatter(landmark[:, 0], landmark[:, 1], s=3, c='cyan')