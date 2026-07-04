def train_one_epoch(model, loader, optimizer,
                    criterion, scaler, device):
    model.train()
    total_loss = 0
    valid_batches = 0
    nan_batches = 0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Training")):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # FIX: forward pass + loss now run under autocast, matching validate().
        # Previously this ran in full FP32 while GradScaler assumed mixed precision —
        # that mismatch is a plausible cause of the early instability seen in the
        # first ConvNeXt run (best checkpoint at epoch 2).
        with autocast('cuda'):
            predictions = model(images)
            loss = criterion(predictions, labels)

        # FIX: skip the batch entirely on NaN loss, BEFORE backward/step/update.
        # Previously NaN batches were excluded only from the running average,
        # but backward() / clip_grad_norm_() / scaler.step() still ran on them —
        # meaning NaN gradients could still update the weights every time this fired.
        if torch.isnan(loss).any():
            nan_batches += 1
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        total_loss += loss_val
        valid_batches += 1

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss_val:.4f}")

    if nan_batches > 0:
        print(f"  WARNING: skipped {nan_batches} NaN batch(es) this epoch")

    # FIX: average over valid batches only, not len(loader), so a handful of
    # skipped NaN batches don't silently bias the reported train_loss.
    return total_loss / valid_batches if valid_batches > 0 else float('nan')
