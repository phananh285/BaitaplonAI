import pandas as pd
import torch
import argparse
from torch import nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from bodulieu import SoundDS
from Seq2Seq import RNNSeq2seq
import sys
import os
from pathlib import Path
def prepara_data():
    data_path = os.path.join("E:", os.sep,"ProjectPython", "VoiceDATASET", "cv-corpus-15.0-delta-2023-09-08", "en", "clips")

    # Read metadata file
    metadata_file = os.path.join("E:", os.sep, "ProjectPython", "VoiceDATASET", "cv-corpus-15.0-delta-2023-09-08", "updated", "valid2.tsv")
    df = pd.read_csv(metadata_file,sep='\t')
    df.head()
    # Construct file path by concatenating fold and file name
    df['relative_path'] = "\\"+df['path'].astype(str)
   
    myds = SoundDS(df, data_path)
    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

    return (train_dl, val_dl)

def training(train_dl,num_epochs):
    # Tensorboard 
    writer = SummaryWriter()
    # Create the model and put it on the GPU if available
    model = RNNSeq2seq(input_size=4,hidden_size=6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    print("helloworld")
    for name, param in model.named_parameters():
        print(name)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the model: {num_params}")
    print("Parameters for Encoder:")
    for name, param in model.encoder.named_parameters():
        print(name)
    print("\nParameters for Decoder:")
    for name, param in model.decoder.named_parameters():
        print(name)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')
    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            mfcc_slice, labels_tuple = data[0], data[1]
        
            labels=[]
            pre_inputs = []
            for s in mfcc_slice:
                pre_inputs.append(s.to(device))
             
            # Normalize the inputs
            inputs=[]
            for i in pre_inputs:
               Ni = (i - i.mean()) / i.std()
               inputs.append(Ni)
                
            for label in labels_tuple:
                labels.append(label.to(device))
             
            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            #if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        avg_acc = correct_prediction/total_prediction
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Acc/train", avg_acc, epoch)
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {avg_acc:.2f}')
    
    torch.save(model.state_dict(), 'model.pt')    
    print('Finished Training')

# ----------------------------
# Inference
# ----------------------------
def inference (model, test_dl):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in test_dl:
            # Get the input features and target labels, and put them on the GPU
            mfcc_slice, labels_tuple = data[0], data[1]
        
            
            pre_inputs = []
            for s in mfcc_slice:
                pre_inputs.append(s.to(device))
             
            # Normalize the inputs
            inputs=[]
            for i in pre_inputs:
               Ni = (i - i.mean()) / i.std()
               inputs.append(Ni)
               
            labels=[]   
            for label in labels_tuple:
                labels.append(label.to(device))

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
        
    acc = correct_prediction/total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=True)
    args = ap.parse_args()
    print("Command-line arguments:", sys.argv)
    train_dl, test_dl = prepara_data()
    if args.mode  == 'train':
        # Run training model
        training(train_dl, num_epochs=100)
    else:
        # Run inference on trained model with the validation set load best model weights
        # Load trained/saved model
        model_inf = RNNSeq2seq(input_size=13+7,hidden_size=6)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_inf = model_inf.to(device)
        model_inf.load_state_dict(torch.load('model.pt'))
        model_inf.eval()

        # Perform inference
        inference(model_inf, test_dl)