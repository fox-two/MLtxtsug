import csv


emails = []

with open('dados_email.csv', 'r', encoding='utf-8') as f:
    leitor = csv.reader(f)
    for i, row in enumerate(leitor):
        if i == 0:
            continue
        if row[5] == '':
            continue
        emails.append((row[4], row[5]))
    
with open('Casos Fabricante.csv', 'r', encoding='utf-8') as f:
    leitor = csv.reader(f)
    for i, row in enumerate(leitor):
        if i == 0:
            continue
        if row[13] == '' or row[13] == '__suppressed__':
            continue
        emails.append((row[11], row[13]))
    

import pickle
with open('emails.bin', 'wb') as f:
    pickle.dump(emails, f)
    
        
        