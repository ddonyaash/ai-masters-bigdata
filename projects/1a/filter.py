#!/opt/conda/envs/dsenv/bin/python

#!/opt/conda/envs/dsenv/bin/python

import sys
import os
from glob import glob
import logging

# Добавляем текущую директорию в пути, чтобы найти model.py
sys.path.append('.')
from model import fields_val

logging.basicConfig(level=logging.DEBUG)

# Ищем файл с условием фильтрации
filter_cond_files = glob('filter_cond*.py')
if len(filter_cond_files) != 1:
    logging.critical("Must supply exactly one filter")
    sys.exit(1)

# Загружаем filter_cond
exec(open(filter_cond_files[0]).read())

# Определяем поля для вывода
if len(sys.argv) == 1:
    outfields = fields_val
else:
    op, field = sys.argv[1][0], sys.argv[1][1:]
    if not op in "+-" or not field in fields_val:
        sys.exit(1)
    elif op == '+':
        outfields = [fields_val[0], field]
    else:
        outfields = list(fields_val)
        outfields.remove(field)

# Читаем данные из stdin
for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith(fields_val[0]):
        continue

    values = line.split('\t')
    record = dict(zip(fields_val, values))

    # Проверка условия
    try:
        if filter_cond(record):
            # Безопасно собираем строку через .get()
            output = "\t".join([str(record.get(x, "")) for x in outfields])
            print(output)
    except Exception:
        continue
