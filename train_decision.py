"""Train decision KAN via supervised learning on shooting-generated data.

Input:  (s_norm, s_target_norm) = 6 dims
Output: (a_norm, H_class) — continuous action + discrete horizon class
Loss:   MSE(action) + alpha * CrossEntropy(horizon)
"""
import torch, argparse
from decision_network import DecisionKAN


def main(n_epochs=2000, alpha=0.5, lr=1e-2, device='cpu'):
    torch.manual_seed(42)

    data = torch.load('decision_data.pt', weights_only=True)
    s = data['s_norm'].to(device)
    s_tgt = data['s_target_norm'].to(device)
    a_label = data['a_norm'].to(device)
    h_label = data['H_class'].to(device)

    n_train = int(len(s) * 0.8)
    idx = torch.randperm(len(s))
    s_tr, s_tgt_tr, a_tr, h_tr = s[idx[:n_train]], s_tgt[idx[:n_train]], a_label[idx[:n_train]], h_label[idx[:n_train]]
    s_va, s_tgt_va, a_va, h_va = s[idx[n_train:]], s_tgt[idx[n_train:]], a_label[idx[n_train:]], h_label[idx[n_train:]]

    n_classes = h_label.max().item() + 1
    model = DecisionKAN(hidden_dim=10, n_horizon_classes=n_classes).to(device)
    print(f"Decision KAN: [6, 10, {1+n_classes}]  params={sum(p.numel() for p in model.parameters())}")
    print(f"Train: {n_train}, Val: {len(s)-n_train}, H_classes: {n_classes}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=600, gamma=0.5)
    mse_fn = torch.nn.MSELoss()
    ce_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        a_pred, h_logits = model(s_tr, s_tgt_tr)
        loss = mse_fn(a_pred, a_tr) + alpha * ce_fn(h_logits, h_tr)
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 400 == 0:
            model.eval()
            with torch.no_grad():
                a_pred_v, h_logits_v = model(s_va, s_tgt_va)
                loss_v = mse_fn(a_pred_v, a_va) + alpha * ce_fn(h_logits_v, h_va)
                h_acc = (h_logits_v.argmax(-1) == h_va).float().mean().item()
            print(f"Epoch {epoch:4d}  lr={scheduler.get_last_lr()[0]:.4f}  "
                  f"train={loss.item():.4f}  val={loss_v.item():.4f}  h_acc={h_acc:.3f}")

    torch.save(model.state_dict(), 'kan_decision.pt')
    print("Saved: kan_decision.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--alpha', type=float, default=0.5, help='CrossEntropy weight')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(n_epochs=args.epochs, alpha=args.alpha, lr=args.lr, device=args.device)
