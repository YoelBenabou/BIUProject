import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import csv
from stressNet import StressNet
from wesad_dataset import WESADDataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

feats = ['BVP_mean', 'BVP_std', 'BVP_min', 'BVP_max',
         'EDA_phasic_mean', 'EDA_phasic_std', 'EDA_phasic_min', 'EDA_phasic_max', 'EDA_smna_mean',
         'EDA_smna_std', 'EDA_smna_min', 'EDA_smna_max', 'EDA_tonic_mean',
         'EDA_tonic_std', 'EDA_tonic_min', 'EDA_tonic_max', 'TEMP_mean', 'TEMP_std', 'TEMP_min',
         'TEMP_max', 'TEMP_slope', 'BVP_peak_freq', 'subject', 'label']

layer_1_dim = len(feats) - 2


def read_and_prepare_data():
    df = pd.read_csv('data/WESAD/all_combined_feats.csv', index_col=0)
    subject_id_list = df['subject'].unique()

    df['label'] = df['label'].apply(change_label)
    df = df[feats]

    return subject_id_list, df


def change_label(label):
    if label == 2:
        return 1
    else:
        return 0


def get_data_loaders(model, df, subject_id, ):
    train_df = df[df['subject'] != subject_id].reset_index(drop=True)
    test_df = df[df['subject'] == subject_id].reset_index(drop=True)

    train_dset = WESADDataset(train_df)
    test_dset = WESADDataset(test_df)

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=model.train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=model.test_batch_size)

    return train_loader, test_loader


def train(model, optimizer, train_loader, validation_loader, num_epochs, criterion, device):
    history = {'train_loss': {}, 'train_acc': {}, 'valid_loss': {}, 'valid_acc': {}}

    for epoch in range(num_epochs):

        # Train:
        total = 0
        correct = 0
        trainlosses = []

        for batch_index, (images, labels) in enumerate(train_loader):
            # Send to GPU (device)
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images.float())

            # Loss
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainlosses.append(loss.item())

            # Compute accuracy
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()  # .mean()
            total += len(labels)

        history['train_loss'][epoch] = np.mean(trainlosses)
        history['train_acc'][epoch] = correct / total

        if epoch % 10 == 0:
            with torch.no_grad():

                losses = []
                total = 0
                correct = 0

                for images, labels in validation_loader:
                    images, labels = images.to(device), labels.to(device)

                    # Forward pass
                    outputs = model(images.float())
                    loss = criterion(outputs, labels)

                    # Compute accuracy
                    _, argmax = torch.max(outputs, 1)
                    correct += (labels == argmax).sum().item()  # .mean()
                    total += len(labels)

                    losses.append(loss.item())

                history['valid_acc'][epoch] = np.round(correct / total, 3)
                history['valid_loss'][epoch] = np.mean(losses)

                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(losses):.4}, Acc: {correct / total:.2}')

    return history


def test(model, validation_loader, criterion, device):
    model.eval()

    total = 0
    correct = 0
    testlosses = []
    correct_labels = []
    predictions = []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(validation_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images.float())

            # Loss
            loss = criterion(outputs, labels)

            testlosses.append(loss.item())

            # Compute accuracy
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            total += len(labels)

            correct_labels.extend(labels)
            predictions.extend(argmax)

    test_loss = np.mean(testlosses)
    accuracy = np.round(correct / total, 2)
    print(f'Loss: {test_loss:.4}, Acc: {accuracy:.2}')

    y_true = [label.item() for label in correct_labels]
    y_pred = [label.item() for label in predictions]

    return y_true, y_pred, test_loss, accuracy


def do_training(subject_id_list, df):
    y_preds = []
    y_truths = []
    histories = []
    test_losses = []
    test_accs = []

    for _ in subject_id_list:
        print('\nSubject: ', _)
        model = StressNet(layer_1_dim)
        model = model.to(model.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)

        train_loader, test_loader = get_data_loaders(model, df, _)

        history = train(model, optimizer, train_loader, test_loader, model.num_epochs, model.criterion, model.device)
        histories.append(history)

        y_true, y_pred, test_loss, test_acc = test(model, test_loader, model.criterion, model.device)

        current_f1 = f1_score(y_true, y_pred)
        print(f"Subject {_} with F1-score: {current_f1:.4f}")
        torch.save(model.state_dict(), f'model/subject_{_}_model.pth')

        test_losses.append(test_loss)
        test_accs.append(test_acc)
        y_preds.append(y_pred)
        y_truths.append(y_true)

    return y_truths, y_preds, test_accs, test_losses


def load_and_predict(data, model_path='model/subject_S5_model.pth'):
    """
    Load the best trained model and predict on new data.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the new data.
    - model_path (str): Path to the saved model.

    Returns:
    - predictions (list): List containing model predictions.
    """

    data['subject'] = 'Spredict'
    data['label'] = -1
    dataset = WESADDataset(data)

    # Load the model and set it to evaluation mode
    model = StressNet(layer_1_dim)
    model.load_state_dict(torch.load(model_path))

    loader = torch.utils.data.DataLoader(dataset, batch_size=model.test_batch_size)

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(model.device)
            outputs = model(images.float())

            # Compute accuracy
            _, argmax = torch.max(outputs, 1)

            predictions.extend(argmax)

    y_pred = [label.item() for label in predictions]
    save_predictions_to_list(y_pred)

    return y_pred


def save_predictions_to_list(predictions):
    with open('predictions/predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for item in predictions:
            writer.writerow([item])


def load_predictions_from_csv():
    with open('predictions/predictions.csv', 'r') as file:
        reader = csv.reader(file)
        predictions = [row[0] for row in reader]

    return predictions


def plot_accuracies_losses(test_accs, test_losses, subject_id_list):
    print('Mean Accuracy:', np.mean(test_accs))
    print('Accuracy std:', np.std(test_accs))
    print('Mean losses:', np.mean(test_losses))

    plt.figure(figsize=(14, 6))
    plt.title('Testing Accuracies in Leave One Out Cross Validation by Subject Left Out as Testing Data')
    sns.barplot(x=subject_id_list, y=test_accs)

    plt.savefig('graphs/test_accuracies.png')
    plt.show()

    plt.figure(figsize=(14, 3))
    plt.title('Testing Losses in Leave One Out Cross Validation by Subject Left Out as Testing Data')
    sns.barplot(x=subject_id_list, y=test_losses)

    plt.savefig('graphs/test_losses.png')
    plt.show()


def do_confusion_matrix(y_truths, y_preds, subject_id_list):
    confusion_matrices = [confusion_matrix(y_true, y_pred) for y_true, y_pred in zip(y_truths, y_preds)]

    plt.figure(figsize=(15, 10))

    for i in range(len(confusion_matrices)):
        plt.subplot(4, 5, i + 1)
        cm = confusion_matrices[i]

        sns.heatmap(cm, annot=True, fmt='d', cbar=False)
        plt.title(f'S{subject_id_list[i]}')
        plt.xlabel('Prediction')
        plt.ylabel('Ground Truth')
    plt.tight_layout()

    plt.savefig('graphs/confusion_matrices.png')
    plt.show()


def calculate_f1_score(y_truths, y_preds, subject_id_list):
    classif_reports = []
    for i, (y_true, y_pred) in enumerate(zip(y_truths, y_preds)):
        print('Subject', subject_id_list[i], ':')
        target_names = ['Not-Stress', 'Stress']
        cr = classification_report(y_true, y_pred, target_names=target_names)
        classif_reports.append(cr)
        print(cr)
        print()

    f1_scores = [float(cr.split('\n')[3].strip().split('      ')[3]) for cr in classif_reports]

    print('f1 score mean: ' + str(np.mean(f1_scores)))
    print('f1 score std: ' + str(np.std(f1_scores)))
