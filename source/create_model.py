import numpy
import torch
import matplotlib
import torchvision.models
import torch.optim


class LandmarkDataset(torch.utils.data.Dataset):
    def __init__(self, images: list, landmarks: list) -> None:
        '''
            Initialization.
            
            Args:
                images (list): a list containing all augmented images in
                the form of tensor.
                landmarks (list): a list containing all augmented landmark
                points in the form of tensor.
            
            Returns:
                None
        '''
        
        self.images = images
        self.landmarks = landmarks
    
    def __len__(self) -> int:
        '''
            Returns the length of data.

            Args:

            Returns:
                An integer value that is the length of data.
        '''

        return len(self.images)

    def __getitem__(self, index: int) -> tuple:
        '''
            Returns specified items at the given index.

            Args:
                index (int): the index of the required item.
            
            Returns:
                A tuple containing required items.
        '''

        image = self.images[index]
        landmark = self.landmarks[index]

        return image, landmark


class LandmarkDetectionModel(torch.nn.Module):
    def __init__(self, num_of_point: int) -> None:
        '''
            Initializes a landmark detection mode using efficientnet_b0 and
            specifies classifier.
            
            Args:
                num_of_point (int): the number of landmark points that the
                model will output.
            
            Returns:
                None
        '''
        
        super().__init__()
        self.num_of_point = num_of_point
        self.model = torchvision.models.efficientnet_b0(
            weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT,
            progress=True
        )

        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(self.model.classifier[1].in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, num_of_point * 2)
        )
    
    def forward(self, inp: torch.tensor) -> torch.Tensor:
        '''
            Pass an image into the model and return the result.
            
            Args:
                inp (torch.tensor): an image in the type of tensor.
            
            Returns:
                Landmarks point of input image in the type of tensor.
        '''
        
        inp = self.model(inp)
        inp = inp.view(-1, self.num_of_point, 2)
        return inp
    
def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
    criterion: torch.nn.MSELoss,
    train_data: torch.utils.data.DataLoader,
    device: torch.device,
    num_of_epoch: int
) -> torch.nn.Module:
    '''
        Train the given model using optimizer, MSELoss criteria and the
        number of epoch.
        
        Args:
            model (torch.nn.Module): the model need to train.
            optimizer (torch.optim.Adam): Adam optimizer.
            criterion (torch.nn.MSELoss): criterion used to calculate
            the loss when training the model.
            train_data (torch.utils.data.DataLoader): data used to train.
            device (torch.device): determine using GPU or CPU to train.
            num_of_epoch (int): the number of times to train the model.
        
        Returns:
            A trained model.
    '''
    
    best_test_loss = 1e9
    train_loss_history = []
    test_loss_history = []
    
    for epoch in range(num_of_epoch):
        model.train()
        running_loss = 0.0

        for image, landmark in train_data:
            image = image.to(device)
            landmark = landmark.to(device)
            
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, landmark)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * image.size(0)
        
        train_loss = running_loss / len(train_data.dataset)
        train_loss_history.append(train_loss)
    
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for image, landmark in train_data:
                image = image.to(device)
                landmark = landmark.to(device)

                output = model(image)
                loss = criterion(output, landmark)
                test_loss += loss.item() * image.size(0)

        test_loss = test_loss / len(train_data.dataset)
        test_loss_history.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(
                model.state_dict(),
                'resources/trained_model.pth'
            )
    
    matplotlib.pyplot.plot(
        numpy.arange(len(train_loss_history)),
        numpy.array(train_loss_history),
        color='red',
        label='Train'
    )
    
    matplotlib.pyplot.plot(
        numpy.arange(len(test_loss_history)),
        numpy.array(test_loss_history),
        color='green',
        label='Test'
    )
    
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
    
    return model
