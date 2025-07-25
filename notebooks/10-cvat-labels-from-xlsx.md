---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python

```

```python
ROOT = "/Users/gimli/cvr/data/zavity/EDU 2025_04_24/Repliky ZH EDU03 kolektor/"
edu_data = os.listdir(ROOT)
```

```python

```

```python

```

```python
for folder in tqdm(edu_data):
    if os.path.isdir(os.path.join(ROOT, folder)):
        # Load the Excel file
        df = pd.read_excel(os.path.join(ROOT, folder, f"vady č.{folder}.xlsx"), header=None)
        # parse defect from the dataframe
        defects = parse_defects(df)
        # write boxes into annotation format
        write_tracklets(defects, os.path.join("/Users/gimli/cvr/data/zavity/EDU 2025_04_24/cvat-import/annotations", folder))
```

```python

```
