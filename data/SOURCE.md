# Dataset source and provenance

## ⚠️ These are not real people

`WA_Fn-UseC_-HR-Employee-Attrition.csv` is **fictional data created by IBM data
scientists**. It is not a record of real employees, real resignations, or real
salaries. IBM published it as sample data for a Watson Analytics demo.

This matters more here than it would for most toy datasets. Everything downstream
of this file reads like an HR system of record: names of job roles, monthly income,
overtime flags, satisfaction scores, who left and who stayed. A reader who assumes
these are real employment records is being misled, and the honest fix is to say so
in the same folder as the file rather than burying it in a notebook cell.

So: no conclusion drawn in this repository is a finding about any real workforce.
The notebook is a modelling exercise on invented data. The relationships in it are
whatever IBM's generator put there.

## 📄 The file

    file       data/WA_Fn-UseC_-HR-Employee-Attrition.csv
    rows       1470 (plus one header line)
    columns    35
    size       227,977 bytes
    sha256     a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659
    encoding   UTF-8 with a BOM (pandas strips it, so the first column reads as `Age`)

Row and column counts above were recomputed from the committed file, not copied from
the dataset description.

## 🔗 Where it came from

    original    IBM Watson Analytics sample data, "IBM HR Analytics Employee
                Attrition & Performance"
    mirrored    Kaggle, pavansubhasht/ibm-hr-analytics-attrition-dataset
    committed   18 May 2019, unmodified since

The copy in this repository is byte-identical to what was downloaded in 2019. No
rows were dropped, no values were edited, no columns were renamed. All cleaning
happens in the notebook.

## 📊 What is actually in it

    target            Attrition, "Yes" / "No"
    class balance     237 Yes, 1233 No  (16.1% positive)
    missing values    none
    duplicate rows    none
    EmployeeNumber    1470 unique ids, running 1 to 2068 with gaps
    Age               18 to 60

Three columns are constant across all 1470 rows and carry no signal:

    EmployeeCount     always 1
    Over18            always "Y"
    StandardHours     always 80

The 16.1% positive rate is the number to keep in mind when reading any accuracy
figure from the notebook: a model that predicts "No" for everybody scores 83.9%.

## ✅ Verifying this yourself

```bash
python - <<'PY'
import hashlib, pandas as pd
p = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(p)
print(df.shape)                                  # (1470, 35)
print(df["Attrition"].value_counts().to_dict())  # {'No': 1233, 'Yes': 237}
print(hashlib.sha256(open(p, "rb").read()).hexdigest())
PY
```

## ⚖️ Licensing

The repository's [`LICENSE`](../LICENSE) (MIT) covers the code. It does not cover this
CSV. The dataset is IBM's sample data, redistributed here for reproducibility, and if
you plan to use it for anything beyond running this notebook you should check IBM's
and Kaggle's terms yourself rather than relying on its presence here as permission.
