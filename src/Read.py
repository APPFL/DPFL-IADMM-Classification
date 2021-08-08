import math
import torch
import torchvision
import torchvision.transforms as transforms

from mlxtend.data import loadlocal_mnist
import json


def Read_CIFAR10(par, Agent):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    npixels = trainset.data.shape[1]
    classes = trainset.classes

    par.num_features = npixels * npixels * 3  # RGB!
    par.num_classes = len(classes)
    par.split_number = int(Agent)
    par.total_data = trainset.data.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Flatten the arrays
    x_train = torch.tensor(
        trainset.data.reshape([-1, par.num_features]),
        dtype=torch.float32,
        device=device,
    )
    x_test = torch.tensor(
        testset.data.reshape([-1, par.num_features]), dtype=torch.float32, device=device
    )
    y_train = torch.tensor(trainset.targets, dtype=torch.int8, device=device)
    y_test = torch.tensor(testset.targets, dtype=torch.int8, device=device)

    # Split data per agent
    x_train_agent = torch.split(x_train, math.ceil(par.total_data / par.split_number))
    y_train_agent = torch.split(y_train, math.ceil(par.total_data / par.split_number))

    return x_test, y_test, x_train_agent, y_train_agent


def Read_MNIST(par, Agent):
    ################################################################################################################################################
    # MNIST
    # type=np.ndarray (x_train: 60000 x 784 (float:0.~1.), y_train: 60000 x 1 (int:0~9), ... )
    ################################################################################################################################################
    # Load MNIST
    x_train, y_train = loadlocal_mnist(
        images_path="./Inputs/train-images-idx3-ubyte",
        labels_path="./Inputs/train-labels-idx1-ubyte",
    )
    x_test, y_test = loadlocal_mnist(
        images_path="./Inputs/t10k-images-idx3-ubyte",
        labels_path="./Inputs/t10k-labels-idx1-ubyte",
    )

    par.num_features = 784  # 28*28
    par.num_classes = 10  # 0 to 9 digits
    par.split_number = int(Agent)
    par.total_data = len(y_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to float32.
    x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.int8, device=device)
    y_test = torch.tensor(y_test, dtype=torch.int8, device=device)

    # Flatten images to 1-D vector of 784 features (28*28).
    # Normalize images value from [0, 255] to [0, 1].
    x_train = x_train.reshape([-1, par.num_features]) / 255.0
    x_test = x_test.reshape([-1, par.num_features]) / 255.0

    # Split data per agent
    x_train_agent = torch.split(x_train, math.ceil(par.total_data / par.split_number))
    y_train_agent = torch.split(y_train, math.ceil(par.total_data / par.split_number))

    return x_test, y_test, x_train_agent, y_train_agent


def Read_FEMNIST(par, Agent):
    ################################################################################################################################################
    ##### FEMNIST (datatype=dict)
    # Example: "users": ["f3795_00", "f3608_13"], "num_samples": [149, 162], "user_data": {"f3795_00": {"x": [], ..., []}, "y": [4, ..., 31]},
    ################################################################################################################################################
    par.num_features = 784  # 28*28
    par.num_classes = 62  # 0 to 9 digits + alphabet (26 + 26)
    TS = 36  # TS <= 36 for FEMNIST dataset

    train_data = {}
    x_train_agent = {}
    y_train_agent = {}
    tmp_x_train = []
    tmp_y_train = []
    tmpcnt = 0
    for testset in range(TS):
        with open(
            "./Inputs/Femnist_Train_%s/all_data_%s_niid_05_keep_0_train_9.json"
            % (Agent, testset)
        ) as f:
            train_data[testset] = json.load(f)
        for user in train_data[testset]["users"]:
            Temp_x_train = []
            Temp_y_train = []
            # x_train
            for x_elem in train_data[testset]["user_data"][user]["x"]:
                tmp_x_train.append(1.0 - np.array(x_elem))
                Temp_x_train.append(1.0 - np.array(x_elem))
            Temp_x_train = np.array(Temp_x_train, np.float32)
            x_train_agent[tmpcnt] = Temp_x_train
            # y_train
            for y_elem in train_data[testset]["user_data"][user]["y"]:
                tmp_y_train.append(np.array(y_elem))
                Temp_y_train.append(np.array(y_elem))
            Temp_y_train = np.array(Temp_y_train, np.uint8)
            y_train_agent[tmpcnt] = Temp_y_train
            tmpcnt += 1

    par.total_data = len(tmp_y_train)
    par.split_number = tmpcnt

    # Testing
    test_data = {}
    temp_x_test = []
    temp_y_test = []
    total_num_test_data = 0
    for testset in range(TS):
        with open(
            "./Inputs/Femnist_Test_%s/all_data_%s_niid_05_keep_0_test_9.json"
            % (Agent, testset)
        ) as f:
            test_data[testset] = json.load(f)
        for user in test_data[testset]["users"]:
            total_num_test_data += len(test_data[testset]["user_data"][user]["y"])

            # x_test
            for x_elem in test_data[testset]["user_data"][user]["x"]:
                temp_x_test.append(1.0 - np.array(x_elem))

            # y_test
            for y_elem in test_data[testset]["user_data"][user]["y"]:
                temp_y_test.append(y_elem)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_test = torch.tensor(temp_x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(temp_y_test, dtype=torch.uint8, device=device)

    return (
        x_test,
        y_test,
        torch.tensor(x_train_agent, dtype=torch.float32, device=device),
        torch.tensor(y_train_agent, dtype=torch.float32, device=device),
    )
