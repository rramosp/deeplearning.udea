import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import sys
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.neural_network import MLPClassifier

from torch.utils.data import TensorDataset, DataLoader
                
def display_imgs(w, figsize=(6,6)):
    plt.figure(figsize=figsize)
    w = (w-np.min(w))/(np.max(w)-np.min(w))
    for i in range(w.shape[-1]):
        plt.subplot(10,10,i+1)
        plt.imshow(w[:,:,:,i], interpolation="none")
        plt.axis("off")
        
def plot_decision_boundary(perceptron, X, y, title, ax):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA','#AAAAFF']))
    ax.contour(xx, yy, Z, colors='k', linewidths=1.5)
    colors = ['red' if label == 0 else 'blue' for label in y]
    ax.scatter(X[:,0], X[:,1], c=colors, s=120, zorder=5, edgecolors='k')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    
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
    
class MLP:
    """A simple 2-layer MLP with sigmoid activations, trained via backpropagation."""

    def __init__(self, hidden_size=4, lr=0.5, n_epochs=5000):
        self.hidden_size = hidden_size
        self.lr = lr
        self.n_epochs = n_epochs

    @staticmethod
    def sigmoid(z):      return 1 / (1 + np.exp(-z))
    @staticmethod
    def sigmoid_d(a):    return a * (1 - a)    # derivative given output a

    def fit(self, X, y):
        n_in, n_h, n_out = X.shape[1], self.hidden_size, 1
        self.W1 = np.random.randn(n_in, n_h) * 0.5
        self.b1 = np.zeros((1, n_h))
        self.W2 = np.random.randn(n_h, n_out) * 0.5
        self.b2 = np.zeros((1, n_out))
        self.losses = []

        y = y.reshape(-1, 1)
        for epoch in range(self.n_epochs):
            # Forward
            z1 = X @ self.W1 + self.b1;  a1 = self.sigmoid(z1)
            z2 = a1 @ self.W2 + self.b2; a2 = self.sigmoid(z2)

            # Loss (Binary Cross-Entropy)
            loss = -np.mean(y * np.log(a2 + 1e-9) + (1-y) * np.log(1-a2 + 1e-9))
            if epoch % 500 == 0: self.losses.append(loss)

            # Backward
            d2 = (a2 - y) * self.sigmoid_d(a2)
            d1 = (d2 @ self.W2.T) * self.sigmoid_d(a1)

            self.W2 -= self.lr * a1.T @ d2
            self.b2 -= self.lr * d2.sum(axis=0, keepdims=True)
            self.W1 -= self.lr * X.T @ d1
            self.b1 -= self.lr * d1.sum(axis=0, keepdims=True)
        return self

    def predict_proba(self, X):
        a1 = self.sigmoid(X @ self.W1 + self.b1)
        a2 = self.sigmoid(a1 @ self.W2 + self.b2)
        return a2

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int).ravel()        

def plot_losses():
    _, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ── MSE surface ──
    y_true_reg = 2.0
    y_pred_range = np.linspace(-2, 6, 300)
    mse_vals = (y_true_reg - y_pred_range)**2
    axes[0].plot(y_pred_range, mse_vals, 'steelblue', linewidth=2.5)
    axes[0].axvline(y_true_reg, color='red', linestyle='--', label=f'True y = {y_true_reg}', linewidth=2)
    axes[0].set_title('MSE Loss (Regression)', fontweight='bold')
    axes[0].set_xlabel('Predicted ŷ'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].annotate('Quadratic –\npenalizes large errors more', xy=(4.5, 5), xytext=(3.5, 12),
                    arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9, color='gray')

    # ── Binary Cross-Entropy ──
    p = np.linspace(1e-6, 1-1e-6, 300)
    bce_y1 = -np.log(p)         # when true label = 1
    bce_y0 = -np.log(1 - p)    # when true label = 0
    axes[1].plot(p, bce_y1, 'green', linewidth=2.5, label='y = 1: -log(p̂)')
    axes[1].plot(p, bce_y0, 'red', linewidth=2.5, label='y = 0: -log(1-p̂)')
    axes[1].set_ylim(0, 6)
    axes[1].set_title('Binary Cross-Entropy', fontweight='bold')
    axes[1].set_xlabel('Predicted probability p̂'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].annotate('High loss when\nconfident & wrong', xy=(0.05, 4.0), xytext=(0.25, 4.8),
                    arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9, color='gray')

    # ── Comparison: loss vs prediction error ──
    errors = np.linspace(-3, 3, 300)
    mse = errors**2
    mae = np.abs(errors)
    huber_delta = 1.0
    huber = np.where(np.abs(errors) <= huber_delta,
                    0.5 * errors**2,
                    huber_delta * (np.abs(errors) - 0.5 * huber_delta))

    axes[2].plot(errors, mse, 'steelblue', linewidth=2.5, label='MSE')
    axes[2].plot(errors, mae, 'darkorange', linewidth=2.5, label='MAE (L1)')
    axes[2].plot(errors, huber, 'green', linewidth=2.5, linestyle='--', label='Huber (δ=1)')
    axes[2].set_title('Regression Loss Comparison', fontweight='bold')
    axes[2].set_xlabel('Prediction Error (y - ŷ)'); axes[2].set_ylabel('Loss')
    axes[2].set_ylim(0, 6)
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Loss Functions for Neural Networks', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_cce():
    # ── Softmax + Categorical Cross-Entropy illustration ──
    classes = ['Cat', 'Dog', 'Bird']

    scenarios = {
        'Correct & Confident': {'logits': [3.0, 0.5, -1.0], 'true': 0},
        'Correct & Uncertain':  {'logits': [1.2, 0.9, 0.8], 'true': 0},
        'Wrong & Confident':    {'logits': [-2.0, 0.5, 3.5], 'true': 0},
    }

    def softmax(z):
        e = np.exp(z - np.max(z))
        return e / e.sum()

    def cce(probs, true_class):
        return -np.log(probs[true_class] + 1e-9)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors_bar = ['#2196F3', '#FF9800', '#4CAF50']

    for ax, (scenario, info) in zip(axes, scenarios.items()):
        probs = softmax(info['logits'])
        loss  = cce(probs, info['true'])
        bars  = ax.bar(classes, probs, color=colors_bar, edgecolor='black', linewidth=1.2)
        bars[info['true']].set_edgecolor('red')
        bars[info['true']].set_linewidth(3)
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{prob:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.set_title(f'{scenario}\nCCE Loss = {loss:.3f}', fontweight='bold', fontsize=10)
        ax.set_ylabel('Softmax Probability'); ax.set_ylim(0, 1.1)
        ax.text(1.5, 0.95, f'True: {classes[info["true"]]}', ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Categorical Cross-Entropy: Effect of Confidence on Loss', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
def loss_surface(w1, w2):
    """A simplified non-convex 2D loss landscape for illustration."""
    return (np.sin(w1*2)*0.5 + w1**2*0.2 + w2**2*0.3 +
            np.sin(w2*3)*0.3 + 0.1*w1*w2)

def gradient(w1, w2):
    dw1 = np.cos(w1*2)*1.0 + w1*0.4 + 0.1*w2
    dw2 = w2*0.6 + np.cos(w2*3)*0.9 + 0.1*w1
    return dw1, dw2

def run_gd(lr, noise_std, n_steps=40, start=(-2.0, 2.0)):
    path = [start]
    w1, w2 = start
    for _ in range(n_steps):
        dw1, dw2 = gradient(w1, w2)
        dw1 += np.random.randn() * noise_std
        dw2 += np.random.randn() * noise_std
        w1 -= lr * dw1
        w2 -= lr * dw2
        path.append((w1, w2))
    return np.array(path)

def plot_GD_batch():
    w1s = np.linspace(-3, 3, 300)
    w2s = np.linspace(-3, 3, 300)
    W1, W2 = np.meshgrid(w1s, w2s)
    L = loss_surface(W1, W2)

    configs = [
        ('Batch GD\n(Full dataset, stable)', 0.4, 0.0),
        ('SGD\n(1 sample, noisy)', 0.15, 1.0),
        ('Mini-Batch GD\n(batch=64, balanced)', 0.3, 0.3),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (title, lr, noise) in zip(axes, configs):
        ax.contourf(W1, W2, L, levels=30, cmap='viridis', alpha=0.8)
        ax.contour(W1, W2, L, levels=30, colors='white', linewidths=0.3, alpha=0.4)
        path = run_gd(lr, noise)
        ax.plot(path[:,0], path[:,1], 'w-o', markersize=3, linewidth=1.5, alpha=0.9)
        ax.plot(path[0,0], path[0,1], 'g^', markersize=12, label='Start')
        ax.plot(path[-1,0], path[-1,1], 'r*', markersize=14, label='End')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('w₁'); ax.set_ylabel('w₂')
        ax.legend(fontsize=9)

    plt.suptitle('Gradient Descent Variants on a Loss Landscape', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
def plot_activation_functions():
    z = np.linspace(-5, 5, 300)

    activations = {
        'Sigmoid':      (1 / (1 + np.exp(-z)),      'steelblue'),
        'Tanh':         (np.tanh(z),                 'darkorange'),
        'ReLU':         (np.maximum(0, z),            'green'),
        'Leaky ReLU':   (np.where(z > 0, z, 0.1*z),  'red'),
        'ELU':          (np.where(z > 0, z, np.exp(z)-1), 'purple'),
    }

    derivatives = {
        'Sigmoid':      lambda z: (1/(1+np.exp(-z))) * (1 - 1/(1+np.exp(-z))),
        'Tanh':         lambda z: 1 - np.tanh(z)**2,
        'ReLU':         lambda z: (z > 0).astype(float),
        'Leaky ReLU':   lambda z: np.where(z > 0, 1.0, 0.1),
        'ELU':          lambda z: np.where(z > 0, 1.0, np.exp(z)),
    }

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for i, (name, (vals, color)) in enumerate(activations.items()):
        ax_f = axes[0, i]
        ax_d = axes[1, i]

        ax_f.plot(z, vals, color=color, linewidth=2.5)
        ax_f.axhline(0, color='k', linewidth=0.5, alpha=0.5)
        ax_f.axvline(0, color='k', linewidth=0.5, alpha=0.5)
        ax_f.set_title(f'{name}', fontweight='bold')
        ax_f.set_ylabel('f(z)' if i == 0 else '')
        ax_f.grid(True, alpha=0.3)
        ax_f.set_ylim(-1.5, 2.5)

        deriv = derivatives[name](z)
        ax_d.plot(z, deriv, color=color, linewidth=2.5, linestyle='--')
        ax_d.axhline(0, color='k', linewidth=0.5, alpha=0.5)
        ax_d.axvline(0, color='k', linewidth=0.5, alpha=0.5)
        ax_d.set_ylabel("f'(z)" if i == 0 else '')
        ax_d.set_xlabel('z')
        ax_d.grid(True, alpha=0.3)
        ax_d.set_ylim(-0.3, 1.5)

    axes[0,0].text(-4.5, 2.2, 'f(z)', fontsize=9, style='italic')
    axes[1,0].text(-4.5, 1.3, "f'(z)", fontsize=9, style='italic')

    fig.suptitle('Activation Functions and Their Derivatives', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def simulate_gradient_flow(activation, n_layers=20):
    """Simulate gradient magnitude across layers for a given activation function."""
    np.random.seed(42)
    gradient_norms = [1.0]
    x = np.random.randn(100)
    g = np.ones(100)  # start with unit gradient

    for _ in range(n_layers):
        W = np.random.randn(100, 100) * 0.1
        z = W @ x
        if activation == 'sigmoid':
            a = 1 / (1 + np.exp(-z))
            da = a * (1 - a)  # max 0.25
        elif activation == 'tanh':
            a = np.tanh(z)
            da = 1 - a**2
        elif activation == 'relu':
            a = np.maximum(0, z)
            da = (z > 0).astype(float)
        else:  # leaky relu
            a = np.where(z > 0, z, 0.01*z)
            da = np.where(z > 0, 1.0, 0.01)

        g = g * da
        g = W.T @ g
        gradient_norms.append(np.linalg.norm(g))
        x = a

    return gradient_norms

def mlp_2_class(X,y):
    clf = MLPClassifier(activation='logistic',solver='sgd', learning_rate_init = 0.1, learning_rate = 'constant', hidden_layer_sizes=(5,), random_state=1,n_iter_no_change = 500, max_iter=2000)
    clf.fit(X, y)
    # Test
    cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA',])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
    Z = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            #print([xx[1,i],yy[j,1]])
            Z[i,j]=clf.predict([[xx[1,i],yy[j,1]]])
    Z = np.round(Z)
    plt.figure(figsize=(5, 4))
    plt.title('2-class problem', fontsize=14)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.pcolormesh(xx, yy, Z.T, cmap="Accent",alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y, cmap="Accent")
    plt.show()
    
def simulate_activation_variance(init_method, activation, n_layers=10, n_neurons=512):
    """Track activation variance through a deep network under different init schemes."""
    np.random.seed(42)
    x = np.random.randn(1000, n_neurons)
    variances = [np.var(x)]

    for _ in range(n_layers):
        if init_method == 'random_small':
            W = np.random.randn(n_neurons, n_neurons) * 0.01
        elif init_method == 'random_large':
            W = np.random.randn(n_neurons, n_neurons) * 1.0
        elif init_method == 'xavier':
            std = np.sqrt(2.0 / (n_neurons + n_neurons))
            W = np.random.randn(n_neurons, n_neurons) * std
        else:  # he
            std = np.sqrt(2.0 / n_neurons)
            W = np.random.randn(n_neurons, n_neurons) * std

        z = x @ W
        if activation == 'relu':
            x = np.maximum(0, z)
        else:
            x = np.tanh(z)
        variances.append(np.var(x))
    return variances

def plot_weight_init_effects():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    configs_init = [
        ('random_small',  'Too Small (std=0.01)', 'gray',       '--'),
        ('random_large',  'Too Large (std=1.0)',  'red',        '--'),
        ('xavier',        'Xavier',               'darkorange', '-'),
        ('he',            'He',                   'green',      '-'),
    ]

    for ax, act in zip(axes, ['tanh', 'relu']):
        for method, label, color, ls in configs_init:
            variances = simulate_activation_variance(method, act)
            ax.semilogy(range(len(variances)), variances, color=color, linestyle=ls, linewidth=2.5, label=label)
        act_str = 'Tanh' if act == 'tanh' else 'ReLU'
        ax.set_title(f'Activation Variance – {act_str}', fontweight='bold')
        ax.set_xlabel('Layer'); ax.set_ylabel('Activation Variance (log scale)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.axhline(1.0, color='black', linewidth=1, linestyle=':', alpha=0.6, label='Ideal = 1')

    plt.suptitle('Effect of Weight Initialization on Activation Variance', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("Xavier is best for Tanh; He initialization is best for ReLU.")
