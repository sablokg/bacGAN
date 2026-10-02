import click
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Gaurav Sablok
# gsablok@proton.me

# --------------------------------------------------
# 1. DNA vocabulary
# --------------------------------------------------

VOCAB = ["A", "T", "G", "C"]
stoi = {base: i for i, base in enumerate(VOCAB)}
itos = {i: base for i, base in enumerate(VOCAB)}

VOCAB_SIZE = 4

DEFAULT_REAL_SEQUENCES = [
    "ATGCGTACGTAGCTAGCTAGCGTACGTAGCTAGCTAGCTAGCTAG",
    "CGTACGTAGCTAGCTACGATCGATCGTACGTAGCTAGCTAGCTAA",
    "GCTAGCTAGCTACGATCGTACGTAGCTAGCTAGCGTACGTAGCTA",
    "ATCGATCGTACGTAGCTAGCTACGATCGATCGTACGTAGCTAGCT",
]


# --------------------------------------------------
# 2. Generator
# --------------------------------------------------

class Generator(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, x):
        """
        x shape: [batch, sequence_length]
        """

        emb = self.embedding(x)

        output, _ = self.lstm(emb)

        logits = self.fc(output)

        return logits

    def generate(self, batch_size, seq_len):
        """
        Generate DNA sequences.
        """

        device = next(self.parameters()).device

        x = torch.zeros(
            batch_size,
            1,
            dtype=torch.long,
            device=device
        )

        generated = []

        for _ in range(seq_len):

            logits = self.forward(x)

            next_logits = logits[:, -1, :]

            probs = F.softmax(next_logits, dim=-1)

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            generated.append(next_token)

            x = torch.cat([x, next_token], dim=1)

        return torch.cat(generated, dim=1)


# --------------------------------------------------
# 3. Discriminator
# --------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            embed_dim
        )

        self.conv = nn.Conv1d(
            embed_dim,
            128,
            kernel_size=5
        )

        self.fc = nn.Linear(
            128,
            1
        )

    def forward(self, x):

        # [batch, seq, embedding]
        emb = self.embedding(x)

        # Conv1D expects:
        # [batch, channels, seq]
        emb = emb.transpose(1, 2)

        h = F.relu(self.conv(emb))

        # Global max pooling
        h = torch.max(h, dim=2).values

        output = self.fc(h)

        return output


# --------------------------------------------------
# 4. Dataset helpers
# --------------------------------------------------

def encode_sequence(seq):
    return torch.tensor(
        [stoi[b] for b in seq],
        dtype=torch.long
    )


def load_sequences(seq_file, seq_len):
    if seq_file:
        with open(seq_file) as f:
            seqs = [line.strip().upper() for line in f if line.strip()]
    else:
        seqs = DEFAULT_REAL_SEQUENCES

    bad = [s for s in seqs if len(s) < seq_len or any(b not in stoi for b in s[:seq_len])]
    if bad:
        raise click.ClickException(
            f"Found {len(bad)} sequence(s) that are shorter than --seq-len "
            f"or contain characters outside {VOCAB}. First offender: {bad[0]!r}"
        )

    return torch.stack([encode_sequence(s[:seq_len]) for s in seqs])


# --------------------------------------------------
# 5. Training routines
# --------------------------------------------------

def pretrain_discriminator(G, D, real_data, optimizer_D, epochs, seq_len, log_every, device):
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        fake_data = G.generate(
            batch_size=real_data.size(0),
            seq_len=seq_len
        ).detach()

        real_labels = torch.ones(real_data.size(0), 1, device=device)
        fake_labels = torch.zeros(fake_data.size(0), 1, device=device)

        real_output = D(real_data)
        fake_output = D(fake_data)

        loss_real = criterion(real_output, real_labels)
        loss_fake = criterion(fake_output, fake_labels)
        loss_D = loss_real + loss_fake

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        if log_every and epoch % log_every == 0:
            click.echo(f"Discriminator epoch {epoch}, loss = {loss_D.item():.4f}")


def pretrain_generator(G, real_data, optimizer_G, epochs, log_every):
    for epoch in range(epochs):
        x = real_data[:, :-1]
        target = real_data[:, 1:]

        logits = G(x)

        loss_G = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            target.reshape(-1)
        )

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if log_every and epoch % log_every == 0:
            click.echo(f"Generator pretrain {epoch}, loss = {loss_G.item():.4f}")


# --------------------------------------------------
# 6. CLI
# --------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--seq-len", default=45, show_default=True, help="DNA sequence length used for training/generation.")
@click.option("--embed-dim", default=32, show_default=True, help="Embedding dimension.")
@click.option("--hidden-dim", default=64, show_default=True, help="LSTM hidden dimension.")
@click.option("--disc-epochs", default=100, show_default=True, help="Discriminator pretraining epochs.")
@click.option("--gen-epochs", default=200, show_default=True, help="Generator pretraining epochs.")
@click.option("--lr", default=0.001, show_default=True, help="Learning rate for both optimizers.")
@click.option("--num-generate", default=10, show_default=True, help="Number of DNA sequences to generate at the end.")
@click.option("--seq-file", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Optional text file with one real DNA sequence per line. Defaults to the built-in toy examples.")
@click.option("--output", type=click.Path(dir_okay=False), default=None,
              help="Optional path to write generated sequences to (one per line).")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility.")
@click.option("--log-every", default=20, show_default=True, help="Print training loss every N epochs (0 to disable).")
@click.option("--cpu", is_flag=True, default=False, help="Force CPU even if a GPU is available.")
def main(seq_len, embed_dim, hidden_dim, disc_epochs, gen_epochs, lr,
         num_generate, seq_file, output, seed, log_every, cpu):
    """Train a tiny SeqGAN-style generator/discriminator on DNA sequences and sample new ones."""

    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device("cuda" if (torch.cuda.is_available() and not cpu) else "cpu")
    click.echo(f"Using device: {device}")

    real_data = load_sequences(seq_file, seq_len).to(device)
    click.echo(f"Loaded {real_data.size(0)} real sequence(s), length {seq_len}.")

    G = Generator(embed_dim, hidden_dim).to(device)
    D = Discriminator(embed_dim).to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=lr)
    optimizer_D = optim.Adam(D.parameters(), lr=lr)

    click.echo("\n--- Pretraining discriminator ---")
    pretrain_discriminator(G, D, real_data, optimizer_D, disc_epochs, seq_len, log_every, device)

    click.echo("\n--- Pretraining generator ---")
    pretrain_generator(G, real_data, optimizer_G, gen_epochs, log_every)

    click.echo(f"\n--- Generating {num_generate} sequence(s) ---")
    G.eval()
    with torch.no_grad():
        generated = G.generate(batch_size=num_generate, seq_len=seq_len)

    dna_sequences = [
        "".join(itos[int(token)] for token in sequence)
        for sequence in generated
    ]

    for dna in dna_sequences:
        click.echo(dna)

    if output:
        with open(output, "w") as f:
            f.write("\n".join(dna_sequences) + "\n")
        click.echo(f"\nWrote {len(dna_sequences)} sequences to {output}")


if __name__ == "__main__":
    main()