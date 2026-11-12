import csv
import glob
import os
import sys

csv.field_size_limit(sys.maxsize)

root = os.path.join('Esame Angelantonio Fedele Murolo', 'synthetic_progressive_posts')
users = {}
for path in sorted(glob.glob(os.path.join(root, 'snapshot_*', 'posts_current_snapshot.tsv'))):
    with open(path, encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            users.setdefault(row['account_id'], row)
print('users with a profile', len(users))
columns = ['username', 'user_bio', 'nationality', 'state_of_origin', 'gender', 'ethnicity',
           'religion', 'political_leaning', 'interests', 'age_interval', 'profession',
           'format_style', 'exact_level_found']
for column in columns:
    values = [(row.get(column) or '').strip() for row in users.values()]
    filled = [value for value in values if value]
    print(f'{column:20s} distinct {len(set(filled)):>7,}  empty {len(values) - len(filled):>6,}'
          f'  e.g. {filled[0][:40] if filled else "-"}')
terms = set()
counts = []
for row in users.values():
    items = [item.strip() for item in (row.get('interests') or '').split(',') if item.strip()]
    counts.append(len(items))
    terms.update(items)
print('interest terms', len(terms), '| mean per user', round(sum(counts) / len(counts), 2),
      '| max', max(counts))
