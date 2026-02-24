import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import sys

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from torch.utils.data import TensorDataset, DataLoader
                
def display_imgs(w, figsize=(6,6)):
    plt.figure(figsize=figsize)
    w = (w-np.min(w))/(np.max(w)-np.min(w))
    for i in range(w.shape[-1]):
        plt.subplot(10,10,i+1)
        plt.imshow(w[:,:,:,i], interpolation="none")
        plt.axis("off")
        

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def show_labeled_image_mosaic(imgs, labels, figsize=(12, 12), idxs=None):

    plt.figure(figsize=figsize)
    for labi,lab in [i for i in enumerate(np.unique(labels))]:
        k = imgs[labels == lab]
        _idxs = idxs[:10] if idxs is not None else np.random.permutation(len(k))[:10]
        for i, idx in enumerate(_idxs):
            if i == 0:
                plt.subplot(10, 11, labi*11+1)
                plt.title("LABEL %d" % lab)
                plt.plot(0, 0)
                plt.axis("off")

            img = k[idx]
            plt.subplot(10, 11, labi*11+i+2)
            plt.imshow(img, cmap=plt.cm.Greys_r)
            plt.axis("off")

def show_preds(x, y, preds):
    for i in range(len(x)):
        plt.figure(figsize=(5,2.5))
        plt.subplot(122)
        plt.imshow(x[i])
        plt.axis("off")
        plt.subplot(121)
        plt.bar(np.arange(len(preds[i])), preds[i], color="blue", alpha=.5, label="prediction")
        plt.bar(np.arange(len(preds[i])), np.eye(len(preds[i]))[int(y[i])], color="red", alpha=.5, label="label")
        plt.xticks(list(range(len(preds[i]))), list(range(len(preds[i]))), rotation="vertical");
        plt.xlim(-.5,len(preds[i])-.5);
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, +1.35),ncol=5)



class SimpleTrainer:

    def __init__(self, model, xtrain, ytrain, xval, yval, n_epochs, batch_size, optimizer, loss_fn, num_workers=1):
        self.model = model
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xval   = xval
        self.yval   = yval

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer  = optimizer

        self.loss_fn = loss_fn
        self.num_workers = num_workers

    def train(self):

        train_dataset = TensorDataset(self.xtrain, self.ytrain)
        val_dataset = TensorDataset(self.xval, self.yval)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=True,      
            num_workers=self.num_workers      
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=32, 
            shuffle=False,
            num_workers=self.num_workers
        )

        self.lossh = []
        for epoch in range(self.n_epochs):
                self.model.train() 
                self.running_loss = 0.0
        
                # discard y
                for batch_idx, (x,y) in enumerate(train_loader):
                    # Move data to GPU if available
                    #data, target = data.to(device), target.to(device)
                    
                    # --- The Core Training Steps ---
                    # 1. Clear gradients from the previous step
                    self.optimizer.zero_grad()
                    
                    # 2. Forward pass: compute predicted outputs
                    output = self.model(x)
                    
                    # 3. Calculate loss
                    loss =self.loss_fn(output, y)
                    
                    # 4. Backward pass: compute gradient of the loss
                    loss.backward()
                    
                    # 5. Optimization: update weights
                    self.optimizer.step()
                    
                    self.running_loss += loss.item()
        
                    self.lossh.append(loss.detach().numpy())
                print(f"Epoch {epoch+1}/{self.n_epochs} | Loss: {self.running_loss/len(train_loader):.4f}")
    
        