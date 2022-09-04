import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def Train(model, train_DL, val_DL, **kwargs):
    # optimizer = optim.SGD(model.parameters(),lr=kwargs["LR"])
    optimizer = optim.Adam(model.parameters(),lr=kwargs["LR"])
    lr_scheduler = StepLR(optimizer, step_size = kwargs["LR_STEP"], gamma = kwargs["LR_GAMMA"])

    loss_history = {"train": [], "val":[]}
    acc_history = {"train": [], "val":[]}
    best_loss = torch.inf
    for ep in range(kwargs["EPOCH"]):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {ep}, current_LR = {round(current_lr,3)}")

        model.train()
        train_loss, train_acc, _ = loss_epoch(model, train_DL, optimizer = optimizer)
        loss_history["train"] += [train_loss]
        acc_history["train"] += [train_acc]

        model.eval()
        with torch.no_grad():
            val_loss, val_acc, _ = loss_epoch(model, val_DL)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model, kwargs["save_model_path"])
                # hyperparam도 같이 save, load 하는 방법
                torch.save({"EPOCH": kwargs["EPOCH"],
                            "LR": kwargs["LR"],
                            "LR_STEP": kwargs["LR_STEP"],
                            "LR_GAMMA": kwargs["LR_GAMMA"],
                            "BATCH_SIZE": kwargs["BATCH_SIZE"],
                            "TRAIN_RATIO": kwargs["TRAIN_RATIO"],
                            "loss_history": loss_history,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict()}, kwargs["save_model_path"])
        loss_history["val"] += [val_loss]
        acc_history["val"] += [val_acc]

        lr_scheduler.step()

        # print loss
        print(f"train loss: {round(train_loss,2)}"
              f"val loss: {round(val_loss,2)}"
              f"train acc: {round(train_acc,1)} %,"
              f"val acc: {round(val_acc,1)} %, time: {round(time.time() - epoch_start)} s")
        print("-"*20)

    return loss_history, acc_history

def loss_epoch(model, DL, optimizer = None):
    N = len(DL.dataset) # the number of data
    rloss = 0; rcorrect=0
    for x_batch, y_batch in tqdm(DL, leave=True):
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        # inference
        y_hat = model(x_batch)
        # loss
        loss = F.cross_entropy(y_hat,y_batch)
        # loss = F.mse_loss(y_hat,y)
        # update
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # loss accumulation
        loss_b = loss.item() * x_batch.shape[0] # batch loss # BATCH_SIZE를 곱하면 18개도 32를 곱하니까..
        rloss += loss_b
        # accuracy accumulation
        pred = y_hat.argmax(dim=1)
        corrects_b = torch.sum(pred == y_batch).item()
        rcorrect += corrects_b
    loss_e = rloss / N # epoch loss
    accuracy_e = rcorrect/N * 100

    return loss_e, accuracy_e, rcorrect

def Test(model,test_DL):
    model.eval()
    with torch.no_grad():
        test_loss, test_acc, rcorrect = loss_epoch(model,test_DL)
    print()
    print(f"Test loss: {round(test_loss,2)}")
    print(f"Test accuracy: {rcorrect}/{len(test_DL.dataset)} ({round(test_acc,1)} %)")

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

def im_plot(DL):
    x_batch, y_batch = next(iter(DL))
    plt.figure(figsize=(8,4))
    for idx in range(6):
        plt.subplot(2,3, idx+1, xticks=[], yticks=[])
        plt.imshow(x_batch[idx,...].permute(1,2,0).squeeze())
        true_class = DL.dataset.classes[y_batch[idx].item()]
        plt.title(true_class, color="g")
