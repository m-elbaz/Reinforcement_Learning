# Training



def fit(X_train, net, optimizer, criterion, device, epoch):
    net.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in enumerate(X_train):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()