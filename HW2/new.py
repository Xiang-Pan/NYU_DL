import time
start_time = time.time()

MAX_DELAY = 8
SEQ_LENGTH = 20

model = VariableDelayGRUMemory(hidden_size=256, max_delay=MAX_DELAY).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[100, 300], gamma=0.88
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
criterion = torch.nn.CrossEntropyLoss(reduction=none)

epoch = 0
test_acc = 0
while test_acc <= 0.993:
    dataset = VariableDelayEchoDataset(max_delay=8, seq_length=20, size=2000)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=512)
    for x, delay, y in train_dataloader:
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        model.train()
        optimizer.zero_grad()
        x = x.cuda()
        y = y.cuda()

        logits = model(x, delay)
        
        mask = [torch.zeros(d) for d in delay]
        mask.append(torch.zeros(max_len))
        mask = torch.nn.utils.rnn.pad_sequence(mask, padding_value=1, batch_first=True)[:-1].reshape(-1, N + 1)
        logits = logits.reshape(-1, N + 1)
        logits = mask * logits
            
        loss = criterion(logits, y.reshape(-1))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0005)
        loss.backward()
        optimizer.step()

    test_acc = test_variable_delay_model(model)
    scheduler.step()
    print(f'epoch {epoch}: {test_acc}', f'loss: {loss.item()}')
    epoch += 1

end_time = time.time()
duration = end_time - start_time
accuracy = test_variable_delay_model(model)
assert duration < 600, 'execution took f{duration:.2f} seconds, which longer than 10 mins'
assert accuracy > 0.99, f'accuracy is too low, got {accuracy}, need 0.99'
print('tests passed')