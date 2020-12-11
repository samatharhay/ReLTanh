import argparse

from os import path, makedirs

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tb
import torchvision.transforms as transforms

from torchvision import datasets
from tqdm import tqdm

from models import FCNN, CNN


## Training loop
def train(model, transform, device, args):
    # construct data
    train_data = datasets.MNIST(root='data', train=True,
                                       download=True, transform=transform)
    train_data, valid_data = torch.utils.data.random_split(train_data, [50000, 10000],
        generator=torch.Generator().manual_seed(args.split_seed))

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
        num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size,
        num_workers=args.num_workers)

    # configurations
    criterion = nn.CrossEntropyLoss()
    n_epochs = args.epochs
    optimizer = None
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=args.momentum)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate)
    else:
        raise NotImplementedError(f"Not implemented optimizer type: {args.optim}")

    # move model to correct device
    model = model.to(device)

    print("----------- training -----------")
    print("==== CONFIG ====")
    print(args)
    print(f"There are {len(train_data)} examples in train set.")
    print(f"There are {len(valid_data)} examples in valid set.")
    print("==== OPTIMIZER ====")
    print(optimizer)
    print("==== MODEL ====")
    print(model)

    # TODO: implement continue training if necessary

    # initialize tensorboard writers
    global_step = 0
    train_logger = tb.SummaryWriter(path.join(args.log_dir, args.model_name, 'train'), flush_secs=1)
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, args.model_name, 'valid'), flush_secs=1)
    train_losses, train_accs, valid_losses, valid_accs = [], [], [], []

    # evaluate 0th step
    loss, acc, _ = evaluate(model, valid_loader, device)
    valid_losses.append(loss)
    valid_accs.append(acc)
    valid_logger.add_scalar('loss', loss, 0)     # log eval loss to tensorboard
    valid_logger.add_scalar('accuracy', acc, 0)  # log eval accuracy to tensorboard

    for epoch in range(n_epochs):
        # monitor training loss
        model.train() # prep model for training
        train_loss = []

        ###################
        # train the model #
        ###################
        for data, target in tqdm(train_loader):
            # move data to device
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss.append(loss.detach().cpu().numpy())
            # update tensorboard
            train_logger.add_scalar('loss_per_step', loss, global_step)
            global_step += 1

        # print training statistics
        # calculate average loss over an epoch
        train_loss = sum(train_loss) / len(train_loss)

        print('Epoch: {}\tStep: {}\tTraining Loss: {:.6f}'.format(
            epoch+1,
            global_step,
            train_loss
            ))

        # training evaluation
        _, acc, _ = evaluate(model, train_loader, device, "training")
        train_logger.add_scalar('loss', train_loss, epoch + 1)
        train_logger.add_scalar('accuracy', acc, epoch + 1)
        train_losses.append(train_loss)
        train_accs.append(acc)

        # validation evaluation
        loss, acc, _ = evaluate(model, valid_loader, device)
        valid_logger.add_scalar('loss', loss, epoch + 1)
        valid_logger.add_scalar('accuracy', acc, epoch + 1)
        valid_losses.append(loss)
        valid_accs.append(acc)

    # save model
    model_path = path.join(args.model_dir, args.model_name)
    makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), path.join(model_path, "model.pt"))
    torch.save(optimizer.state_dict(), path.join(model_path, "optimizer.pt"))
    print(f"==== Training Completed ====\nSaved model and optimizer to {model_path}")

    return model, train_losses, train_accs, valid_losses, valid_accs


## Evaluation loop
def evaluate(model, data_loader, device, eval_type="validation"):
    # configurations
    criterion = nn.CrossEntropyLoss()

    accumulated_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model = model.to(device)
    model.eval()

    for data, target in tqdm(data_loader):
        # move data to device
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update loss
        accumulated_loss += loss.detach().cpu().numpy()
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(target.shape[0]):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    accuracy = 100. * np.sum(class_correct) / np.sum(class_total)
    print(f'{eval_type} accuracy (overall): %2d%% (%2d/%2d)' % (
        accuracy,
        np.sum(class_correct), np.sum(class_total)))

    class_accuracy = np.array(class_correct) / np.array(class_total)
    print(f'{eval_type} accuracy (per class): {[round(acc, 2) for acc in class_accuracy]}')

    # TODO: If necessary for evaluation,
    # add precision, recall, F1, and confusion matrix

    return loss, accuracy, class_accuracy


## Test evaluation call
def evaluate_test(model, transform, device, args):
    print("----------- testing -----------")

    # prepare data
    test_data = datasets.MNIST(root='data', train=False,
                                      download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
        num_workers=args.num_workers)
    print(f"There are {len(test_data)} examples in test set.")

    # run evaluations
    return evaluate(model, test_loader, device, "test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--hidden_dims', type=int, default=128,
                        help="hidden dimensions per layer")
    parser.add_argument('-l', '--hidden_layers', type=int, default=5,
                        help="number of hidden layers")
    parser.add_argument('-a', '--activation_function_type', type=str, default="reltanh",
                        choices=["reltanh", "relu", "tanh"],
                        help="type of activation function after each hidden layer")
    parser.add_argument('--reltanh_alpha', type=float, default=0.0,
                        help="positive boundary for reltanh")
    parser.add_argument('--reltanh_beta', type=float, default=-1.5,
                        help="negative boundary for reltanh")
    parser.add_argument('--split_seed', type=int, default=42,
                        help="rng seed for train and valid split")
    parser.add_argument('--init_seed', type=int, default=42,
                        help="rng seed for model weight initializations")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="threads for reading and processing data")
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help="dataloader batch size")
    parser.add_argument('-o', '--optimizer', type=str, default="sgd", choices=["sgd", "adam"],
                        help="optimizer for training")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help="optimizer learning rate")
    parser.add_argument('--momentum', type=float, default=0.0,
                        help="optimizer momentum if applicable")
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help="training epochs")
    parser.add_argument('--log_dir', type=str, default='logs',
                        help="logging save directory")
    parser.add_argument('--model', type=str, default='fcnn', choices=['fcnn', 'cnn'],
                        help='type of model to use')
    parser.add_argument('--model_dir', type=str, default='models',
                        help="models save directory")
    parser.add_argument('--model_name', type=str, default=None,
                        help="model name, initialized if not given")
    parser.add_argument('--cuda', action='store_true',
                        help="run training on a GPU")
    parser.add_argument('--run_test', action='store_true',
                        help="run test on testing set after training")
    args = parser.parse_args()

    # set up model name
    if args.model_name == None:
        args.model_name = (args.model + '_' +
                           args.activation_function_type + '_' +
                           str(args.hidden_layers) + 'l_' +
                           str(args.epochs) + 'e')
        if args.activation_function_type == "reltanh":
            args.model_name += ('_a_' + str(args.reltanh_alpha) +
                                '_b_' + str(args.reltanh_beta))

    # seed for consistency in model weights
    torch.manual_seed(args.init_seed)

    # check if CUDA is available
    device = torch.device("cpu")
    if args.cuda:
        if torch.cuda.is_available():
            print("Using GPU with CUDA")
            torch.cuda.manual_seed_all(args.init_seed)
            device = torch.device("cuda")
        else:
            print("Training on CPU because CUDA is not available. Please install it "
                  "if you wish to train this model on a GPU.")
            args.cuda = False

    # TODO: set up better transforms if necessary
    transform = transforms.ToTensor()

    # create and train model
    model = None
    if args.model == 'fcnn':
        model = FCNN(args,
                     hidden_dims=args.hidden_dims,
                     hidden_layers=args.hidden_layers)
    elif args.model == 'cnn':
        model = CNN(args)
    else:
        raise NotImplementedError(f"model {args.model} is not implemented")
    model, train_losses, train_accs, valid_losses, valid_accs = train(model, transform, device, args)

    # print training logs to file
    loss_dir = path.join(args.log_dir, args.model_name, "loss")
    accuracy_dir = path.join(args.log_dir, args.model_name, "accuracy")
    makedirs(loss_dir, exist_ok=True)
    makedirs(accuracy_dir, exist_ok=True)
    pd.DataFrame(train_losses).to_csv(path.join(loss_dir, "train.csv"),
                                      header=False, index=False)
    pd.DataFrame(valid_losses).to_csv(path.join(loss_dir, "valid.csv"),
                                      header=False, index=False)
    pd.DataFrame(train_accs).to_csv(path.join(accuracy_dir, "train.csv"),
                                    header=False, index=False)
    pd.DataFrame(valid_accs).to_csv(path.join(accuracy_dir, "valid.csv"),
                                    header=False, index=False)

    # evaluate test set
    if args.run_test:
        _ = evaluate_test(model, transform, device, args)
