from cProfile import label
import time
import copy
import torch


def convert_to_onnx(model, path, input_names, output_names):
    model.to("cpu")
    model.train(False)
    dummy_input = torch.randn(
        1, 3, 224, 224, requires_grad=True, device=torch.device("cpu")
    )
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        verbose=True,
        opset_version=11,
    )


def test_model(model, test_data_loader, device="cpu"):
    with torch.no_grad():
        model.to(device)
        model.eval()

        n_correct = 0

        for inputs, labels in test_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            n_correct += torch.sum(preds == labels.data)

        test_acc = n_correct / len(test_data_loader.dataset)
        print(f"Test Acc: {test_acc:.4f}")


def train_model(
    model,
    train_data_loader,
    val_data_loader,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
    device="cpu",
):
    since = time.time()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:

            if phase == "train":
                model.train()

            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            dataloader = train_data_loader if phase == "train" else val_data_loader
            # Iterate over data.
            for inputs, labels in dataloader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train" and scheduler is not None:
                scheduler.step()

            dataset_size = (
                len(train_data_loader.dataset)
                if phase == "train"
                else len(val_data_loader.dataset)
            )
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
