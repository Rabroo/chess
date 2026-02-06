import os

d = './raw/chess_quality'
o = './raw/chess_quality.tsv'

print("Scanning directory...")
files = set(os.listdir(d))
print(f"Found {len(files)} files")

w = 0
with open(o, 'w') as out:
    for i in range(1000000):
        if i % 50000 == 0:
            print(f'{i:,} checked, {w:,} written', flush=True)

        pf = f'position_{i:08d}.txt'
        sf = f'position_{i:08d}_score.txt'

        if pf not in files:
            continue

        try:
            fen = open(f'{d}/{pf}').read().strip()
            score = open(f'{d}/{sf}').readline().strip()
            out.write(f'{fen}\t{score}\n')
            w += 1
        except Exception as e:
            print(f'Error at {i}: {e}')

print(f'Done: {w:,} positions')
