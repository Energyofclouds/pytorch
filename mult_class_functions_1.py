import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def Train(model, train_DL, **kwargs):
    # optimizer = optim.SGD(model.parameters(),lr=kwargs["LR"])
    optimizer = optim.Adam(model.parameters(),lr=kwargs["LR"])

    loss_history=[]
    NoT = len(train_DL.dataset) # The Number of training data

    model.train() # train mode로 전환
    for ep in range(kwargs["EPOCH"]):
        rloss = 0 # running loss
        for x_batch, y_batch in train_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            # inference
            y_hat = model(x_batch)
            # loss
            loss = F.cross_entropy(y_hat,y_batch)
            # loss = F.mse_loss(y_hat,y)
            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss accumulation
            loss_b = loss.item() * x_batch.shape[0] # batch loss # BATCH_SIZE를 곱하면 18개도 32를 곱하니까..
            rloss += loss_b
        # print 
        loss_e = rloss / NoT # epoch loss
        loss_history += [loss_e]
        print(f"Epoch: {ep+1}, train loss: {round(loss_e,3)}")
        print("-"*20)
    return loss_history

def Test(model,test_DL):
    NoTes = len(test_DL.dataset)
    model.eval()
    with torch.no_grad():
        rcorrect = 0
        for x_batch, y_batch in test_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            # inference
            y_hat = model(x_batch)
            # accuracy
            pred = y_hat.argmax(dim=1)
            corrects_b = torch.sum(pred == y_batch).item()
            rcorrect += corrects_b
        test_acc = rcorrect/NoTes * 100
    print(f"Test accuracy: {rcorrect}/{NoTes} ({round(test_acc,1)} %)")

def count_params(model):
    num=0
    for param in model.parameters():
        if param.requires_grad:
            num+=param.numel()
    print(num)

def Test_plot(model, test_DL):
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(test_DL))
        x_batch = x_batch.to(DEVICE)
        y_hat = model(x_batch)
        pred = y_hat.argmax(dim=1)

    x_batch = x_batch.to("cpu")
    pred = pred.to("cpu")

    plt.figure(figsize=(8,4))
    for idx in range(6):
        plt.subplot(2,3, idx+1, xticks=[], yticks=[])
        plt.imshow(x_batch[idx,...].permute(1,2,0).squeeze())
        pred_class = test_DL.dataset.classes[pred[idx].item()]
        true_class = test_DL.dataset.classes[y_batch[idx].item()]
        plt.title(f"{pred_class} ({true_class})", color = "g" if pred_class==true_class else "r")