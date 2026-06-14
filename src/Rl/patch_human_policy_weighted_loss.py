from pathlib import Path

p = Path("supervised/train_human_policy.py")
text = p.read_text()

backup = Path("supervised/train_human_policy.py.backup_before_weighted_loss")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add helper after button_accuracy.
marker = '''def button_accuracy(logits, targets):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds == targets).float().mean().item()


'''

helper = '''def compute_pos_weight(dataset, action_dim=8, max_weight=12.0):
    """
    Compute positive-class weights for each Doom button.

    Rare buttons like use/strafe/turn should matter more than easy negative zeros.
    BCEWithLogitsLoss(pos_weight=...) increases the loss when the model misses
    a positive button press.
    """

    positives = torch.zeros(action_dim)
    total = 0

    for _, y in dataset:
        positives += y.float()
        total += 1

    negatives = max(1, total) - positives
    pos_weight = negatives / torch.clamp(positives, min=1.0)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=max_weight)

    return pos_weight


'''

if helper.strip() not in text:
    if marker not in text:
        raise SystemExit("Could not find button_accuracy marker.")
    text = text.replace(marker, marker + helper, 1)

# Add CLI args.
text = text.replace(
    '    parser.add_argument("--max-no-op-ratio", type=float, default=0.20)\n',
    '    parser.add_argument("--max-no-op-ratio", type=float, default=0.10)\n'
    '    parser.add_argument("--max-pos-weight", type=float, default=12.0)\n',
    1,
)

# Replace criterion.
old = '''    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
'''

new = '''    pos_weight = compute_pos_weight(dataset, action_dim=8, max_weight=args.max_pos_weight).to(device)
    print(f"[train_human_policy] pos_weight={pos_weight.detach().cpu().tolist()}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
'''

if old not in text:
    raise SystemExit("Could not find criterion block.")

text = text.replace(old, new, 1)

p.write_text(text)
print("Added weighted BCE loss for rare buttons.")
