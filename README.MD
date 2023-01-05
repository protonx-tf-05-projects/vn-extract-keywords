# Vietnam Keywords Extraction

<p align="center">
    <img src='protonx-k.png' width=200 class="center">
</p>


Authors:
- Github: [@bangoc123](https://github.com/bangoc123) [@ngavu](https://github.com/ngavu2004) [@vubao108](https://github.com/vubao108)


## I.  Set up environment
### Create Enviroment

```python
conda create -n vn-extract-keywords python==3.8.0
```

### Active Enviroment

```python
conda activate vn-extract-keywords
```

### Install Requirements

```python
pip install -r requirements.txt
```

## II.  Extract Keywords

Copy your text to `input_line.txt` file and run commands

```python
python3 generate_keywords.py
```

Example:

Input:

`Túi xách là một trong những vật dụng cần thiết của hầu hết chị em phụ nữ mỗi khi ra đường.`

Output:

```python
['vật_dụng', 'hầu_hết', 'chị_em', 'đường', 'túi_xách', 'cần_thiết', 'phụ_nữ']
```







