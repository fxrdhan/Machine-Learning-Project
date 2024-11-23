<a href="https://colab.research.google.com/github/fxrdhan/Machine-Learning-Project/blob/main/%5BClustering%5D_Submission_Akhir_BMLP_Firdaus_Arif_Ramadhani.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **1. Dataset Introduction**

**Air Quality Measurements Dataset**

1.  Description

    This dataset contains detailed air quality measurements collected over a specified period. It focuses on various pollutants, providing a comprehensive overview of air quality metrics.

2. Feature	Description

  - Date: The date of the measurement.
  - Time: The time of the measurement.
  - CO(GT): Concentration of carbon monoxide (CO) in the air (µg/m³).
  - PT08.S1(CO):	Sensor measurement for CO concentration.
  - NMHC(GT):	Concentration of non-methane hydrocarbons (NMHC) (µg/m³).
  - C6H6(GT):	Concentration of benzene (C6H6) in the air (µg/m³).
  - PT08.S2(NMHC):	Sensor measurement for NMHC concentration.
  - NOx(GT):	Concentration of nitrogen oxides (NOx) in the air (µg/m³).
  - PT08.S3(NOx):	Sensor measurement for NOx concentration.
  - NO2(GT):	Concentration of nitrogen dioxide (NO2) in the air (µg/m³).

  Missing Attribute Values
  - Some measurements may be recorded as -200, indicating missing or invalid data points.
3. Total Rows: 9357
4. Source: [Kaggle](https://www.kaggle.com/datasets/dakshbhalala/uci-air-quality-dataset/data)

# **2. Import Library**


```python
!python --version
```

    Python 3.10.12



```python
!pip install pandas numpy matplotlib seaborn plotly scikit-learn yellowbrick scipy
```

    Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (2.2.2)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (1.26.4)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (3.8.0)
    Requirement already satisfied: seaborn in /usr/local/lib/python3.10/dist-packages (0.13.2)
    Requirement already satisfied: plotly in /usr/local/lib/python3.10/dist-packages (5.24.1)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (1.5.2)
    Requirement already satisfied: yellowbrick in /usr/local/lib/python3.10/dist-packages (1.5)
    Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (1.13.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2024.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas) (2024.2)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (4.55.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (1.4.7)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (24.2)
    Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (11.0.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (3.2.0)
    Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from plotly) (9.0.0)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.4.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (3.5.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)



```python
# System and Settings
import time
import warnings
warnings.filterwarnings('ignore')

# Core Data Processing Libraries
import numpy as np
import pandas as pd
# Configure pandas display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Machine Learning
## Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

## Clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

## Dimensionality Reduction
from sklearn.decomposition import PCA

## Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif

# Statistical Analysis
from scipy import stats
from scipy.stats.mstats import winsorize
from scipy.spatial.distance import cdist
```

# **3. Import Dataset**


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/AirQualityUCI.csv")
```


```python
df.head()
```





  <div id="df-500ea715-d801-4194-9687-3b5544be23cb" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>NMHC(GT)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
      <th>Unnamed: 15</th>
      <th>Unnamed: 16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3/10/2004</td>
      <td>18:00:00</td>
      <td>2.600</td>
      <td>1360.000</td>
      <td>150.000</td>
      <td>11.900</td>
      <td>1046.000</td>
      <td>166.000</td>
      <td>1056.000</td>
      <td>113.000</td>
      <td>1692.000</td>
      <td>1268.000</td>
      <td>13.600</td>
      <td>48.900</td>
      <td>0.758</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3/10/2004</td>
      <td>19:00:00</td>
      <td>2.000</td>
      <td>1292.000</td>
      <td>112.000</td>
      <td>9.400</td>
      <td>955.000</td>
      <td>103.000</td>
      <td>1174.000</td>
      <td>92.000</td>
      <td>1559.000</td>
      <td>972.000</td>
      <td>13.300</td>
      <td>47.700</td>
      <td>0.726</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3/10/2004</td>
      <td>20:00:00</td>
      <td>2.200</td>
      <td>1402.000</td>
      <td>88.000</td>
      <td>9.000</td>
      <td>939.000</td>
      <td>131.000</td>
      <td>1140.000</td>
      <td>114.000</td>
      <td>1555.000</td>
      <td>1074.000</td>
      <td>11.900</td>
      <td>54.000</td>
      <td>0.750</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3/10/2004</td>
      <td>21:00:00</td>
      <td>2.200</td>
      <td>1376.000</td>
      <td>80.000</td>
      <td>9.200</td>
      <td>948.000</td>
      <td>172.000</td>
      <td>1092.000</td>
      <td>122.000</td>
      <td>1584.000</td>
      <td>1203.000</td>
      <td>11.000</td>
      <td>60.000</td>
      <td>0.787</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3/10/2004</td>
      <td>22:00:00</td>
      <td>1.600</td>
      <td>1272.000</td>
      <td>51.000</td>
      <td>6.500</td>
      <td>836.000</td>
      <td>131.000</td>
      <td>1205.000</td>
      <td>116.000</td>
      <td>1490.000</td>
      <td>1110.000</td>
      <td>11.200</td>
      <td>59.600</td>
      <td>0.789</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-500ea715-d801-4194-9687-3b5544be23cb')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-500ea715-d801-4194-9687-3b5544be23cb button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-500ea715-d801-4194-9687-3b5544be23cb');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-5209947b-b131-4434-b5e5-fe6fed588ac9">
  <button class="colab-df-quickchart" onclick="quickchart('df-5209947b-b131-4434-b5e5-fe6fed588ac9')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-5209947b-b131-4434-b5e5-fe6fed588ac9 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




# **4. Exploratory Data Analysis (EDA)**


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9471 entries, 0 to 9470
    Data columns (total 17 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Date           9357 non-null   object 
     1   Time           9357 non-null   object 
     2   CO(GT)         9357 non-null   float64
     3   PT08.S1(CO)    9357 non-null   float64
     4   NMHC(GT)       9357 non-null   float64
     5   C6H6(GT)       9357 non-null   float64
     6   PT08.S2(NMHC)  9357 non-null   float64
     7   NOx(GT)        9357 non-null   float64
     8   PT08.S3(NOx)   9357 non-null   float64
     9   NO2(GT)        9357 non-null   float64
     10  PT08.S4(NO2)   9357 non-null   float64
     11  PT08.S5(O3)    9357 non-null   float64
     12  T              9357 non-null   float64
     13  RH             9357 non-null   float64
     14  AH             9357 non-null   float64
     15  Unnamed: 15    0 non-null      float64
     16  Unnamed: 16    0 non-null      float64
    dtypes: float64(15), object(2)
    memory usage: 1.2+ MB



```python
df.sample(5)
```





  <div id="df-f040116c-20e5-454d-8a52-95b58ae79d9f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>NMHC(GT)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
      <th>Unnamed: 15</th>
      <th>Unnamed: 16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3860</th>
      <td>8/18/2004</td>
      <td>14:00:00</td>
      <td>-200.000</td>
      <td>987.000</td>
      <td>-200.000</td>
      <td>5.400</td>
      <td>782.000</td>
      <td>-200.000</td>
      <td>875.000</td>
      <td>-200.000</td>
      <td>1514.000</td>
      <td>593.000</td>
      <td>34.900</td>
      <td>29.500</td>
      <td>1.627</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2443</th>
      <td>6/20/2004</td>
      <td>13:00:00</td>
      <td>0.600</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>23.000</td>
      <td>-200.000</td>
      <td>36.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6809</th>
      <td>12/19/2004</td>
      <td>11:00:00</td>
      <td>2.500</td>
      <td>1077.000</td>
      <td>-200.000</td>
      <td>9.600</td>
      <td>963.000</td>
      <td>386.000</td>
      <td>761.000</td>
      <td>129.000</td>
      <td>1081.000</td>
      <td>1151.000</td>
      <td>5.800</td>
      <td>49.800</td>
      <td>0.461</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>89</th>
      <td>3/14/2004</td>
      <td>11:00:00</td>
      <td>2.800</td>
      <td>1445.000</td>
      <td>148.000</td>
      <td>10.900</td>
      <td>1009.000</td>
      <td>176.000</td>
      <td>878.000</td>
      <td>114.000</td>
      <td>1696.000</td>
      <td>1355.000</td>
      <td>16.900</td>
      <td>46.100</td>
      <td>0.879</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3519</th>
      <td>8/4/2004</td>
      <td>9:00:00</td>
      <td>-200.000</td>
      <td>1113.000</td>
      <td>-200.000</td>
      <td>10.600</td>
      <td>1001.000</td>
      <td>-200.000</td>
      <td>663.000</td>
      <td>-200.000</td>
      <td>1783.000</td>
      <td>1146.000</td>
      <td>29.900</td>
      <td>42.100</td>
      <td>1.747</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f040116c-20e5-454d-8a52-95b58ae79d9f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f040116c-20e5-454d-8a52-95b58ae79d9f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f040116c-20e5-454d-8a52-95b58ae79d9f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-3f7761c7-b9c2-4cfd-8217-b9818d622c25">
  <button class="colab-df-quickchart" onclick="quickchart('df-3f7761c7-b9c2-4cfd-8217-b9818d622c25')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-3f7761c7-b9c2-4cfd-8217-b9818d622c25 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df.describe()
```





  <div id="df-9fd018c8-a2ea-4525-9520-942992395daf" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>NMHC(GT)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
      <th>Unnamed: 15</th>
      <th>Unnamed: 16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-34.208</td>
      <td>1048.990</td>
      <td>-159.090</td>
      <td>1.866</td>
      <td>894.595</td>
      <td>168.617</td>
      <td>794.990</td>
      <td>58.149</td>
      <td>1391.480</td>
      <td>975.072</td>
      <td>9.778</td>
      <td>39.485</td>
      <td>-6.838</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>77.657</td>
      <td>329.833</td>
      <td>139.789</td>
      <td>41.380</td>
      <td>342.333</td>
      <td>257.434</td>
      <td>321.994</td>
      <td>126.940</td>
      <td>467.210</td>
      <td>456.938</td>
      <td>43.204</td>
      <td>51.216</td>
      <td>38.977</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>-200.000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.600</td>
      <td>921.000</td>
      <td>-200.000</td>
      <td>4.000</td>
      <td>711.000</td>
      <td>50.000</td>
      <td>637.000</td>
      <td>53.000</td>
      <td>1185.000</td>
      <td>700.000</td>
      <td>10.900</td>
      <td>34.100</td>
      <td>0.692</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.500</td>
      <td>1053.000</td>
      <td>-200.000</td>
      <td>7.900</td>
      <td>895.000</td>
      <td>141.000</td>
      <td>794.000</td>
      <td>96.000</td>
      <td>1446.000</td>
      <td>942.000</td>
      <td>17.200</td>
      <td>48.600</td>
      <td>0.977</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.600</td>
      <td>1221.000</td>
      <td>-200.000</td>
      <td>13.600</td>
      <td>1105.000</td>
      <td>284.000</td>
      <td>960.000</td>
      <td>133.000</td>
      <td>1662.000</td>
      <td>1255.000</td>
      <td>24.100</td>
      <td>61.900</td>
      <td>1.296</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11.900</td>
      <td>2040.000</td>
      <td>1189.000</td>
      <td>63.700</td>
      <td>2214.000</td>
      <td>1479.000</td>
      <td>2683.000</td>
      <td>340.000</td>
      <td>2775.000</td>
      <td>2523.000</td>
      <td>44.600</td>
      <td>88.700</td>
      <td>2.231</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9fd018c8-a2ea-4525-9520-942992395daf')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-9fd018c8-a2ea-4525-9520-942992395daf button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9fd018c8-a2ea-4525-9520-942992395daf');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-a6c9f20e-d48e-4099-94e4-1d31a9a795fd">
  <button class="colab-df-quickchart" onclick="quickchart('df-a6c9f20e-d48e-4099-94e4-1d31a9a795fd')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-a6c9f20e-d48e-4099-94e4-1d31a9a795fd button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df.isnull().sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Date</th>
      <td>114</td>
    </tr>
    <tr>
      <th>Time</th>
      <td>114</td>
    </tr>
    <tr>
      <th>CO(GT)</th>
      <td>114</td>
    </tr>
    <tr>
      <th>PT08.S1(CO)</th>
      <td>114</td>
    </tr>
    <tr>
      <th>NMHC(GT)</th>
      <td>114</td>
    </tr>
    <tr>
      <th>C6H6(GT)</th>
      <td>114</td>
    </tr>
    <tr>
      <th>PT08.S2(NMHC)</th>
      <td>114</td>
    </tr>
    <tr>
      <th>NOx(GT)</th>
      <td>114</td>
    </tr>
    <tr>
      <th>PT08.S3(NOx)</th>
      <td>114</td>
    </tr>
    <tr>
      <th>NO2(GT)</th>
      <td>114</td>
    </tr>
    <tr>
      <th>PT08.S4(NO2)</th>
      <td>114</td>
    </tr>
    <tr>
      <th>PT08.S5(O3)</th>
      <td>114</td>
    </tr>
    <tr>
      <th>T</th>
      <td>114</td>
    </tr>
    <tr>
      <th>RH</th>
      <td>114</td>
    </tr>
    <tr>
      <th>AH</th>
      <td>114</td>
    </tr>
    <tr>
      <th>Unnamed: 15</th>
      <td>9471</td>
    </tr>
    <tr>
      <th>Unnamed: 16</th>
      <td>9471</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
df.duplicated().sum()
```




    113




```python
df = df.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1)
```


```python
numeric_df = df.select_dtypes(include=['float64', 'int64'])
numeric_df.columns.tolist()
```




    ['CO(GT)',
     'PT08.S1(CO)',
     'NMHC(GT)',
     'C6H6(GT)',
     'PT08.S2(NMHC)',
     'NOx(GT)',
     'PT08.S3(NOx)',
     'NO2(GT)',
     'PT08.S4(NO2)',
     'PT08.S5(O3)',
     'T',
     'RH',
     'AH']




```python
# winsorization
for col in df.select_dtypes(include='number').columns:
    df[col] = winsorize(df[col], limits=[0.01, 0.01])
```


```python
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_df, 1):
    plt.subplot(4, 4, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


    
![png](nb_files/nb_20_0.png)
    


**Insight:**
- Most features show a right-skewed distribution.
- Features like T (Temperature), RH (Relative Humidity), and AH (Absolute Humidity) show a more symmetric distribution.


```python
plt.figure(figsize=(15, 10))
df.boxplot()
plt.xticks(rotation=90)
plt.title('Box Plot of Features')
plt.tight_layout()
plt.show()
```


    
![png](nb_files/nb_22_0.png)
    


**Insight:**
- Several features, such as NMHC(GT) and NOx(GT), have significant outliers that are far from the interquartile range (IQR).
- Features like C6H6(GT) and CO(GT) have a wide spread, indicating substantial variability in the data.


```python
plt.figure(figsize=(16, 10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
```


    
![png](nb_files/nb_24_0.png)
    



```python
# Filter for strong correlations
correlation_matrix = numeric_df.corr()
strong_corr = correlation_matrix[(correlation_matrix > 0.7) | (correlation_matrix < -0.7)]

plt.figure(figsize=(12, 8))
sns.heatmap(strong_corr, annot=True, cmap='coolwarm', cbar=True)
plt.title('Strong Feature Correlations (> 0.7 or < -0.7)')
plt.show()
```


    
![png](nb_files/nb_25_0.png)
    



```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# C6H6(GT) vs PT08.S2(NMHC)
sns.scatterplot(ax=axes[0], x='C6H6(GT)', y='PT08.S2(NMHC)', data=df, alpha=0.7)
axes[0].set_title('C6H6(GT) vs PT08.S2(NMHC)')
axes[0].set_xlabel('C6H6(GT)')
axes[0].set_ylabel('PT08.S2(NMHC)')

# CO(GT) vs C6H6(GT)
sns.scatterplot(ax=axes[1], x='CO(GT)', y='C6H6(GT)', data=df, alpha=0.7)
axes[1].set_title('CO(GT) vs C6H6(GT)')
axes[1].set_xlabel('CO(GT)')
axes[1].set_ylabel('C6H6(GT)')

plt.tight_layout()
plt.show()
```


    
![png](nb_files/nb_26_0.png)
    



```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# C6H6(GT) vs PT08.S5(O3)
sns.scatterplot(ax=axes[0], x='C6H6(GT)', y='PT08.S5(O3)', data=df, alpha=0.7)
axes[0].set_title('C6H6(GT) vs PPT08.S5(O3)')
axes[0].set_xlabel('C6H6(GT)')
axes[0].set_ylabel('PT08.S5(O3)')

# CO(GT) vs PT08.S5(O3)
sns.scatterplot(ax=axes[1], x='CO(GT)', y='PT08.S5(O3)', data=df, alpha=0.7)
axes[1].set_title('CO(GT) vs PT08.S5(O3))')
axes[1].set_xlabel('CO(GT)')
axes[1].set_ylabel('PT08.S5(O3)')

plt.tight_layout()
plt.show()
```


    
![png](nb_files/nb_27_0.png)
    



```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# NOx(GT) vs PT08.S3(NOx)
sns.scatterplot(ax=axes[0], x='NOx(GT)', y='PT08.S3(NOx)', data=df, alpha=0.7)
axes[0].set_title('NOx(GT) vs PT08.S3(NOx)')
axes[0].set_xlabel('NOx(GT)')
axes[0].set_ylabel('PT08.S3(NOx)')

# CO(GT) vs PT08.S1(CO)
sns.scatterplot(ax=axes[1], x='CO(GT)', y='PT08.S1(CO)', data=df, alpha=0.7)
axes[1].set_title('CO(GT) vs PT08.S1(CO)')
axes[1].set_xlabel('CO(GT)')
axes[1].set_ylabel('PT08.S1(CO)')

plt.tight_layout()
plt.show()
```


    
![png](nb_files/nb_28_0.png)
    



```python
daily_avg = df.groupby('Date')['CO(GT)'].mean()

plt.figure(figsize=(18, 5))
daily_avg.plot(kind='line')
plt.title('Daily CO(GT) Avg Levels Over Time')
plt.xlabel('Date')
plt.ylabel('CO(GT)')
plt.grid(True)
plt.show()
```


    
![png](nb_files/nb_29_0.png)
    


# **5. Data Preprocessing**

## Remove Duplicated Rows & Missing Values


```python
df.drop_duplicates(inplace=True)
```


```python
df.dropna(inplace=True)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 9357 entries, 0 to 9356
    Data columns (total 15 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Date           9357 non-null   object 
     1   Time           9357 non-null   object 
     2   CO(GT)         9357 non-null   float64
     3   PT08.S1(CO)    9357 non-null   float64
     4   NMHC(GT)       9357 non-null   float64
     5   C6H6(GT)       9357 non-null   float64
     6   PT08.S2(NMHC)  9357 non-null   float64
     7   NOx(GT)        9357 non-null   float64
     8   PT08.S3(NOx)   9357 non-null   float64
     9   NO2(GT)        9357 non-null   float64
     10  PT08.S4(NO2)   9357 non-null   float64
     11  PT08.S5(O3)    9357 non-null   float64
     12  T              9357 non-null   float64
     13  RH             9357 non-null   float64
     14  AH             9357 non-null   float64
    dtypes: float64(13), object(2)
    memory usage: 1.1+ MB


## Feature Engineering


```python
# Create a new feature by combining 'CO(GT)' and 'PT08.S1(CO)'
df['CO_Product'] = df['CO(GT)'] * df['PT08.S1(CO)']
df['CO_Avg'] = (df['CO(GT)'] + df['PT08.S1(CO)']) / 2
df['CO_Ratio'] = df['CO(GT)'] / (df['PT08.S1(CO)'] + 1e-6)

df[['CO_Product', 'CO_Avg', 'CO_Ratio']].describe()
```





  <div id="df-1113464c-2639-4be5-b656-45dcae6817e0" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO_Product</th>
      <th>CO_Avg</th>
      <th>CO_Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-34822.759</td>
      <td>507.391</td>
      <td>-0.030</td>
    </tr>
    <tr>
      <th>std</th>
      <td>82975.074</td>
      <td>170.984</td>
      <td>0.100</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-383000.000</td>
      <td>-200.000</td>
      <td>-0.294</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>395.000</td>
      <td>439.150</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1414.500</td>
      <td>512.850</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2957.800</td>
      <td>601.500</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>max</th>
      <td>40000.000</td>
      <td>1024.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1113464c-2639-4be5-b656-45dcae6817e0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-1113464c-2639-4be5-b656-45dcae6817e0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1113464c-2639-4be5-b656-45dcae6817e0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-5d10b7fa-8614-4e90-9293-4dcf37bc63b1">
  <button class="colab-df-quickchart" onclick="quickchart('df-5d10b7fa-8614-4e90-9293-4dcf37bc63b1')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-5d10b7fa-8614-4e90-9293-4dcf37bc63b1 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Create a new feature by combining 'NMHC(GT)' and 'PT08.S2(NMHC)'
df['NMHC_Product'] = df['NMHC(GT)'] * df['PT08.S2(NMHC)']
df['NMHC_Avg'] = (df['NMHC(GT)'] + df['PT08.S2(NMHC)']) / 2
df['NMHC_Ratio'] = df['NMHC(GT)'] / (df['PT08.S2(NMHC)'] + 1e-6)

df[['NMHC_Product', 'NMHC_Avg', 'NMHC_Ratio']].describe()
```





  <div id="df-48b587d0-d969-4281-b0ea-16456a16f7d4" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NMHC_Product</th>
      <th>NMHC_Avg</th>
      <th>NMHC_Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-137052.864</td>
      <td>367.753</td>
      <td>-0.148</td>
    </tr>
    <tr>
      <th>std</th>
      <td>171180.822</td>
      <td>191.879</td>
      <td>0.273</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-442800.000</td>
      <td>-200.000</td>
      <td>-2.560</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-215000.000</td>
      <td>264.500</td>
      <td>-0.262</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-170800.000</td>
      <td>358.000</td>
      <td>-0.208</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-129400.000</td>
      <td>470.500</td>
      <td>-0.161</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1946393.000</td>
      <td>1413.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-48b587d0-d969-4281-b0ea-16456a16f7d4')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-48b587d0-d969-4281-b0ea-16456a16f7d4 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-48b587d0-d969-4281-b0ea-16456a16f7d4');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-40a1bec7-3fe3-4c1b-b0c5-fc6839da1b49">
  <button class="colab-df-quickchart" onclick="quickchart('df-40a1bec7-3fe3-4c1b-b0c5-fc6839da1b49')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-40a1bec7-3fe3-4c1b-b0c5-fc6839da1b49 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Total Polutan
df['Total_Polutan'] = df[['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']].sum(axis=1)

df['Total_Polutan'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total_Polutan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9357.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>35.334</td>
    </tr>
    <tr>
      <th>std</th>
      <td>451.200</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1000.000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-116.600</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>74.200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>292.700</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1739.000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>




```python
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# Extract temporal components
df['Year'] = df['Datetime'].dt.year
df['Month'] = df['Datetime'].dt.month
df['Day'] = df['Datetime'].dt.day
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek  # Monday=0, Sunday=6
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)  # Binary feature: 1 for weekend, 0 for weekday

df[['Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'IsWeekend']].sample(5)
```





  <div id="df-97fb391e-c927-4f4d-8c56-ce3910b5ff7d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Hour</th>
      <th>DayOfWeek</th>
      <th>IsWeekend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8616</th>
      <td>2005</td>
      <td>3</td>
      <td>4</td>
      <td>18</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>2004</td>
      <td>5</td>
      <td>5</td>
      <td>11</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8513</th>
      <td>2005</td>
      <td>2</td>
      <td>28</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3921</th>
      <td>2004</td>
      <td>8</td>
      <td>21</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1393</th>
      <td>2004</td>
      <td>5</td>
      <td>7</td>
      <td>19</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-97fb391e-c927-4f4d-8c56-ce3910b5ff7d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-97fb391e-c927-4f4d-8c56-ce3910b5ff7d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-97fb391e-c927-4f4d-8c56-ce3910b5ff7d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-b17c8c82-e04f-45cc-b048-547fd4e1de84">
  <button class="colab-df-quickchart" onclick="quickchart('df-b17c8c82-e04f-45cc-b048-547fd4e1de84')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-b17c8c82-e04f-45cc-b048-547fd4e1de84 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
monthly_avg = df.groupby('Month')['Total_Polutan'].mean()
hourly_avg = df.groupby('Hour')['Total_Polutan'].mean()

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
monthly_avg.plot(kind='line', marker='o')
plt.title('Average Total_Polutan by Month')
plt.xlabel('Month')
plt.ylabel('Average Total_Polutan')
plt.grid(True)

plt.subplot(1, 2, 2)
hourly_avg.plot(kind='line', marker='o')
plt.title('Average Total_Polutan by Hour')
plt.xlabel('Hour')
plt.ylabel('Average Total_Polutan')
plt.grid(True)

plt.tight_layout()
plt.show()
```


    
![png](nb_files/nb_40_0.png)
    



```python
df['Date'] = pd.to_datetime(df['Date'])
daily_avg = df.groupby('Date')['Total_Polutan'].mean()

plt.figure(figsize=(18, 5))
daily_avg.plot(kind='line')
plt.title('Daily Total_Polutan Avg Levels Over Time')
plt.xlabel('Date')
plt.ylabel('Total_Polutan')
plt.grid(True)
plt.show()
```


    
![png](nb_files/nb_41_0.png)
    


## Handle Outliers


```python
# 1. Replace -200 with NaN
df = df.replace(-200, np.nan)

# 2. Linear interpolation for short gaps (3 periods)
df = df.interpolate(method='linear', limit=3)

# 3. Forward fill for medium gaps (6 periods)
df = df.fillna(method='ffill', limit=6)

# 4. Use seasonal mean for long gaps
df['hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
# Group by hour and calculate mean for each column separately
seasonal_mean = df.groupby('hour')[df.select_dtypes(include=[np.number]).columns].transform('mean')
df = df.fillna(value=seasonal_mean)
df = df.drop('hour', axis=1)

# 5. If any NaN remains, use backward fill
df = df.fillna(method='bfill')
```


```python
float_features = df.select_dtypes(include='float')

num_features = len(float_features.columns)
num_cols = 4
num_rows = (num_features // num_cols) + (num_features % num_cols > 0)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15))
axes = axes.flatten()

for i, col in enumerate(float_features.columns):
    float_features.boxplot(column=col, ax=axes[i])
    axes[i].set_title(f'{col}')
    axes[i].set_xticks([])

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
```


    
![png](nb_files/nb_44_0.png)
    


## Data Encoding


```python
# Binary Encoding for DayOfWeek
df['DayOfWeek_binary'] = df['DayOfWeek'].apply(lambda x: format(x, '03b'))

binary_columns = ['DayOfWeek_bit0', 'DayOfWeek_bit1', 'DayOfWeek_bit2']
df[binary_columns] = df['DayOfWeek_binary'].apply(lambda x: pd.Series(list(x))).astype(int)

df.drop(['DayOfWeek_binary'], axis=1, inplace=True)
df.drop(['DayOfWeek'], axis=1, inplace=True)

df.head()
```





  <div id="df-1c18c61d-2443-4a6c-abf2-617b8116b2e7" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>NMHC(GT)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
      <th>CO_Product</th>
      <th>CO_Avg</th>
      <th>CO_Ratio</th>
      <th>NMHC_Product</th>
      <th>NMHC_Avg</th>
      <th>NMHC_Ratio</th>
      <th>Total_Polutan</th>
      <th>Datetime</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Hour</th>
      <th>IsWeekend</th>
      <th>DayOfWeek_bit0</th>
      <th>DayOfWeek_bit1</th>
      <th>DayOfWeek_bit2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2004-03-10</td>
      <td>18:00:00</td>
      <td>2.600</td>
      <td>1360.000</td>
      <td>150.000</td>
      <td>11.900</td>
      <td>1046.000</td>
      <td>166.000</td>
      <td>1056.000</td>
      <td>113.000</td>
      <td>1692.000</td>
      <td>1268.000</td>
      <td>13.600</td>
      <td>48.900</td>
      <td>0.758</td>
      <td>3536.000</td>
      <td>681.300</td>
      <td>0.002</td>
      <td>156900.000</td>
      <td>598.000</td>
      <td>0.143</td>
      <td>443.500</td>
      <td>2004-03-10 18:00:00</td>
      <td>2004</td>
      <td>3</td>
      <td>10</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2004-03-10</td>
      <td>19:00:00</td>
      <td>2.000</td>
      <td>1292.000</td>
      <td>112.000</td>
      <td>9.400</td>
      <td>955.000</td>
      <td>103.000</td>
      <td>1174.000</td>
      <td>92.000</td>
      <td>1559.000</td>
      <td>972.000</td>
      <td>13.300</td>
      <td>47.700</td>
      <td>0.726</td>
      <td>2584.000</td>
      <td>647.000</td>
      <td>0.002</td>
      <td>106960.000</td>
      <td>533.500</td>
      <td>0.117</td>
      <td>318.400</td>
      <td>2004-03-10 19:00:00</td>
      <td>2004</td>
      <td>3</td>
      <td>10</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2004-03-10</td>
      <td>20:00:00</td>
      <td>2.200</td>
      <td>1402.000</td>
      <td>88.000</td>
      <td>9.000</td>
      <td>939.000</td>
      <td>131.000</td>
      <td>1140.000</td>
      <td>114.000</td>
      <td>1555.000</td>
      <td>1074.000</td>
      <td>11.900</td>
      <td>54.000</td>
      <td>0.750</td>
      <td>3084.400</td>
      <td>702.100</td>
      <td>0.002</td>
      <td>82632.000</td>
      <td>513.500</td>
      <td>0.094</td>
      <td>344.200</td>
      <td>2004-03-10 20:00:00</td>
      <td>2004</td>
      <td>3</td>
      <td>10</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004-03-10</td>
      <td>21:00:00</td>
      <td>2.200</td>
      <td>1376.000</td>
      <td>80.000</td>
      <td>9.200</td>
      <td>948.000</td>
      <td>172.000</td>
      <td>1092.000</td>
      <td>122.000</td>
      <td>1584.000</td>
      <td>1203.000</td>
      <td>11.000</td>
      <td>60.000</td>
      <td>0.787</td>
      <td>3027.200</td>
      <td>689.100</td>
      <td>0.002</td>
      <td>75840.000</td>
      <td>514.000</td>
      <td>0.084</td>
      <td>385.400</td>
      <td>2004-03-10 21:00:00</td>
      <td>2004</td>
      <td>3</td>
      <td>10</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2004-03-10</td>
      <td>22:00:00</td>
      <td>1.600</td>
      <td>1272.000</td>
      <td>51.000</td>
      <td>6.500</td>
      <td>836.000</td>
      <td>131.000</td>
      <td>1205.000</td>
      <td>116.000</td>
      <td>1490.000</td>
      <td>1110.000</td>
      <td>11.200</td>
      <td>59.600</td>
      <td>0.789</td>
      <td>2035.200</td>
      <td>636.800</td>
      <td>0.001</td>
      <td>42636.000</td>
      <td>443.500</td>
      <td>0.061</td>
      <td>306.100</td>
      <td>2004-03-10 22:00:00</td>
      <td>2004</td>
      <td>3</td>
      <td>10</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1c18c61d-2443-4a6c-abf2-617b8116b2e7')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-1c18c61d-2443-4a6c-abf2-617b8116b2e7 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1c18c61d-2443-4a6c-abf2-617b8116b2e7');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-79e57703-3856-4587-abb0-a4faa08e1f0a">
  <button class="colab-df-quickchart" onclick="quickchart('df-79e57703-3856-4587-abb0-a4faa08e1f0a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-79e57703-3856-4587-abb0-a4faa08e1f0a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Label Encoding for Time
le = LabelEncoder()
df['Time'] = le.fit_transform(df['Time'])
```


```python
# Cyclic Encoding for Hour & Month
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
```

## Data Binning


```python
# Binning for CO(GT)
bins_co = [0, 2, 5, 12]
labels_co = ['Low', 'Medium', 'High']
df['CO_Category'] = pd.cut(df['CO(GT)'], bins=bins_co, labels=labels_co)

# Binning for T (Temperature)
df['Temperature_Category'] = pd.qcut(df['T'], q=3, labels=['Cold', 'Moderate', 'Hot'])

# Binning for NO2(GT)
bins_no2 = [0, 100, 200, 340]
labels_no2 = ['Low', 'Medium', 'High']
df['NO2_Category'] = pd.cut(df['NO2(GT)'], bins=bins_no2, labels=labels_no2)

# Binning for Hour (Time of Day)
def categorize_hour(hour):
    if hour < 6:
        return 'Night'
    elif hour < 12:
        return 'Morning'
    elif hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

df['Time_of_Day'] = df['Hour'].apply(lambda x: categorize_hour(x))

# Binning for C6H6(GT)
bins_c6h6 = [0, 10, 30, 64]
labels_c6h6 = ['Low', 'Medium', 'High']
df['C6H6_Category'] = pd.cut(df['C6H6(GT)'], bins=bins_c6h6, labels=labels_c6h6)

df[['CO_Category', 'Temperature_Category', 'NO2_Category', 'Time_of_Day', 'C6H6_Category']].sample(5)
```





  <div id="df-8ac01f4e-1f62-49e4-9b14-41231c8f6248" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO_Category</th>
      <th>Temperature_Category</th>
      <th>NO2_Category</th>
      <th>Time_of_Day</th>
      <th>C6H6_Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1683</th>
      <td>Low</td>
      <td>Moderate</td>
      <td>Medium</td>
      <td>Evening</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>7155</th>
      <td>Medium</td>
      <td>Cold</td>
      <td>Medium</td>
      <td>Evening</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>7282</th>
      <td>Low</td>
      <td>Cold</td>
      <td>Low</td>
      <td>Night</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>2654</th>
      <td>High</td>
      <td>Hot</td>
      <td>Medium</td>
      <td>Morning</td>
      <td>High</td>
    </tr>
    <tr>
      <th>7573</th>
      <td>Low</td>
      <td>Cold</td>
      <td>Medium</td>
      <td>Morning</td>
      <td>Low</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8ac01f4e-1f62-49e4-9b14-41231c8f6248')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-8ac01f4e-1f62-49e4-9b14-41231c8f6248 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8ac01f4e-1f62-49e4-9b14-41231c8f6248');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-0ecd8985-00e3-4f18-81d3-306093d1cc47">
  <button class="colab-df-quickchart" onclick="quickchart('df-0ecd8985-00e3-4f18-81d3-306093d1cc47')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-0ecd8985-00e3-4f18-81d3-306093d1cc47 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df['Time_of_Day'].value_counts()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>Time_of_Day</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Evening</th>
      <td>2340</td>
    </tr>
    <tr>
      <th>Night</th>
      <td>2340</td>
    </tr>
    <tr>
      <th>Morning</th>
      <td>2340</td>
    </tr>
    <tr>
      <th>Afternoon</th>
      <td>2337</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
# Label encoding for each binned column
label_cols = ['CO_Category', 'Temperature_Category', 'NO2_Category', 'Time_of_Day', 'C6H6_Category']
le = LabelEncoder()

for col in label_cols:
    df[col] = le.fit_transform(df[col])

df[['CO_Category', 'Temperature_Category', 'NO2_Category', 'Time_of_Day', 'C6H6_Category']].sample(5)
```





  <div id="df-e859c1e2-c0e1-4d04-869f-a54c6e382d4a" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO_Category</th>
      <th>Temperature_Category</th>
      <th>NO2_Category</th>
      <th>Time_of_Day</th>
      <th>C6H6_Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3325</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4147</th>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5728</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8090</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7264</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e859c1e2-c0e1-4d04-869f-a54c6e382d4a')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e859c1e2-c0e1-4d04-869f-a54c6e382d4a button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e859c1e2-c0e1-4d04-869f-a54c6e382d4a');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-73db21ae-a592-4bd0-9cfd-dc15f92b561f">
  <button class="colab-df-quickchart" onclick="quickchart('df-73db21ae-a592-4bd0-9cfd-dc15f92b561f')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-73db21ae-a592-4bd0-9cfd-dc15f92b561f button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
columns_to_drop = ['Datetime', 'Date', 'Time', 'Month', 'Hour']
df.drop(columns=columns_to_drop, inplace=True)

df.describe()
```





  <div id="df-e6493fd5-9850-4712-a8a9-aa4e4a12effe" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>NMHC(GT)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
      <th>CO_Product</th>
      <th>CO_Avg</th>
      <th>CO_Ratio</th>
      <th>NMHC_Product</th>
      <th>NMHC_Avg</th>
      <th>NMHC_Ratio</th>
      <th>Total_Polutan</th>
      <th>Year</th>
      <th>Day</th>
      <th>IsWeekend</th>
      <th>DayOfWeek_bit0</th>
      <th>DayOfWeek_bit1</th>
      <th>DayOfWeek_bit2</th>
      <th>Hour_sin</th>
      <th>Hour_cos</th>
      <th>Month_sin</th>
      <th>Month_cos</th>
      <th>CO_Category</th>
      <th>Temperature_Category</th>
      <th>NO2_Category</th>
      <th>Time_of_Day</th>
      <th>C6H6_Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.113</td>
      <td>1100.677</td>
      <td>221.968</td>
      <td>10.133</td>
      <td>940.735</td>
      <td>240.273</td>
      <td>834.944</td>
      <td>110.882</td>
      <td>1455.857</td>
      <td>1025.052</td>
      <td>18.334</td>
      <td>49.107</td>
      <td>1.024</td>
      <td>-34816.319</td>
      <td>509.721</td>
      <td>-0.030</td>
      <td>-137052.864</td>
      <td>389.640</td>
      <td>-0.148</td>
      <td>35.324</td>
      <td>2004.240</td>
      <td>15.877</td>
      <td>0.287</td>
      <td>0.431</td>
      <td>0.429</td>
      <td>0.428</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.058</td>
      <td>-0.007</td>
      <td>1.365</td>
      <td>0.997</td>
      <td>1.485</td>
      <td>1.500</td>
      <td>1.371</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.389</td>
      <td>215.017</td>
      <td>135.482</td>
      <td>7.414</td>
      <td>265.275</td>
      <td>199.909</td>
      <td>253.466</td>
      <td>46.325</td>
      <td>341.806</td>
      <td>395.963</td>
      <td>8.732</td>
      <td>17.140</td>
      <td>0.398</td>
      <td>82978.597</td>
      <td>165.826</td>
      <td>0.100</td>
      <td>171180.822</td>
      <td>158.691</td>
      <td>0.273</td>
      <td>451.206</td>
      <td>0.427</td>
      <td>8.809</td>
      <td>0.453</td>
      <td>0.495</td>
      <td>0.495</td>
      <td>0.495</td>
      <td>0.707</td>
      <td>0.707</td>
      <td>0.724</td>
      <td>0.687</td>
      <td>0.557</td>
      <td>0.817</td>
      <td>0.577</td>
      <td>1.118</td>
      <td>0.522</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.100</td>
      <td>647.000</td>
      <td>7.000</td>
      <td>0.100</td>
      <td>383.000</td>
      <td>2.000</td>
      <td>322.000</td>
      <td>2.000</td>
      <td>551.000</td>
      <td>221.000</td>
      <td>-1.900</td>
      <td>9.200</td>
      <td>0.185</td>
      <td>-383000.000</td>
      <td>-99.850</td>
      <td>-0.294</td>
      <td>-442800.000</td>
      <td>-67.000</td>
      <td>-2.560</td>
      <td>-1000.000</td>
      <td>2004.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-1.000</td>
      <td>-1.000</td>
      <td>-1.000</td>
      <td>-1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.100</td>
      <td>939.000</td>
      <td>82.051</td>
      <td>4.500</td>
      <td>736.000</td>
      <td>101.000</td>
      <td>664.000</td>
      <td>76.078</td>
      <td>1235.000</td>
      <td>737.000</td>
      <td>12.000</td>
      <td>36.000</td>
      <td>0.742</td>
      <td>397.000</td>
      <td>439.850</td>
      <td>0.001</td>
      <td>-215000.000</td>
      <td>277.000</td>
      <td>-0.262</td>
      <td>-116.600</td>
      <td>2004.000</td>
      <td>8.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.707</td>
      <td>-0.707</td>
      <td>-0.500</td>
      <td>-0.500</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.892</td>
      <td>1067.000</td>
      <td>237.396</td>
      <td>8.400</td>
      <td>914.000</td>
      <td>186.000</td>
      <td>805.000</td>
      <td>108.396</td>
      <td>1463.000</td>
      <td>969.000</td>
      <td>17.800</td>
      <td>49.400</td>
      <td>1.001</td>
      <td>1415.400</td>
      <td>513.200</td>
      <td>0.001</td>
      <td>-170800.000</td>
      <td>367.339</td>
      <td>-0.208</td>
      <td>74.200</td>
      <td>2004.000</td>
      <td>16.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.800</td>
      <td>1231.000</td>
      <td>299.000</td>
      <td>14.000</td>
      <td>1116.000</td>
      <td>317.000</td>
      <td>968.000</td>
      <td>137.000</td>
      <td>1669.000</td>
      <td>1270.000</td>
      <td>24.200</td>
      <td>61.900</td>
      <td>1.300</td>
      <td>2958.800</td>
      <td>601.500</td>
      <td>0.002</td>
      <td>-129400.000</td>
      <td>476.000</td>
      <td>-0.161</td>
      <td>292.700</td>
      <td>2004.000</td>
      <td>23.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.707</td>
      <td>0.707</td>
      <td>0.866</td>
      <td>0.500</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>3.000</td>
      <td>2.000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11.900</td>
      <td>2040.000</td>
      <td>1189.000</td>
      <td>63.700</td>
      <td>2214.000</td>
      <td>1479.000</td>
      <td>2683.000</td>
      <td>340.000</td>
      <td>2775.000</td>
      <td>2523.000</td>
      <td>44.600</td>
      <td>88.700</td>
      <td>2.231</td>
      <td>40000.000</td>
      <td>1024.000</td>
      <td>1.000</td>
      <td>1946393.000</td>
      <td>1413.000</td>
      <td>1.000</td>
      <td>1739.000</td>
      <td>2005.000</td>
      <td>31.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>3.000</td>
      <td>2.000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e6493fd5-9850-4712-a8a9-aa4e4a12effe')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e6493fd5-9850-4712-a8a9-aa4e4a12effe button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e6493fd5-9850-4712-a8a9-aa4e4a12effe');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-e84fb415-5247-4c87-94af-7ad853ee808b">
  <button class="colab-df-quickchart" onclick="quickchart('df-e84fb415-5247-4c87-94af-7ad853ee808b')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e84fb415-5247-4c87-94af-7ad853ee808b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 9357 entries, 0 to 9356
    Data columns (total 35 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   CO(GT)                9357 non-null   float64
     1   PT08.S1(CO)           9357 non-null   float64
     2   NMHC(GT)              9357 non-null   float64
     3   C6H6(GT)              9357 non-null   float64
     4   PT08.S2(NMHC)         9357 non-null   float64
     5   NOx(GT)               9357 non-null   float64
     6   PT08.S3(NOx)          9357 non-null   float64
     7   NO2(GT)               9357 non-null   float64
     8   PT08.S4(NO2)          9357 non-null   float64
     9   PT08.S5(O3)           9357 non-null   float64
     10  T                     9357 non-null   float64
     11  RH                    9357 non-null   float64
     12  AH                    9357 non-null   float64
     13  CO_Product            9357 non-null   float64
     14  CO_Avg                9357 non-null   float64
     15  CO_Ratio              9357 non-null   float64
     16  NMHC_Product          9357 non-null   float64
     17  NMHC_Avg              9357 non-null   float64
     18  NMHC_Ratio            9357 non-null   float64
     19  Total_Polutan         9357 non-null   float64
     20  Year                  9357 non-null   int32  
     21  Day                   9357 non-null   int32  
     22  IsWeekend             9357 non-null   int64  
     23  DayOfWeek_bit0        9357 non-null   int64  
     24  DayOfWeek_bit1        9357 non-null   int64  
     25  DayOfWeek_bit2        9357 non-null   int64  
     26  Hour_sin              9357 non-null   float64
     27  Hour_cos              9357 non-null   float64
     28  Month_sin             9357 non-null   float64
     29  Month_cos             9357 non-null   float64
     30  CO_Category           9357 non-null   int64  
     31  Temperature_Category  9357 non-null   int64  
     32  NO2_Category          9357 non-null   int64  
     33  Time_of_Day           9357 non-null   int64  
     34  C6H6_Category         9357 non-null   int64  
    dtypes: float64(24), int32(2), int64(9)
    memory usage: 2.5 MB


## Standardize Features


```python
df_scaled = df.copy()

X = df_scaled.select_dtypes(include=['float64', 'int64']).columns
exclude_columns = ['IsWeekend', 'DayOfWeek_bit0', 'DayOfWeek_bit1', 'DayOfWeek_bit2', 'CO_Category', 'Temperature_Category', 'C6H6_Category']
X = [col for col in X if col not in exclude_columns]

scaler = StandardScaler()

df_scaled[X] = scaler.fit_transform(df_scaled[X])

df_scaled[X].head()
```





  <div id="df-c3b6a82f-9616-49cf-9dbe-1edb82799ca2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>NMHC(GT)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
      <th>CO_Product</th>
      <th>CO_Avg</th>
      <th>CO_Ratio</th>
      <th>NMHC_Product</th>
      <th>NMHC_Avg</th>
      <th>NMHC_Ratio</th>
      <th>Total_Polutan</th>
      <th>Hour_sin</th>
      <th>Hour_cos</th>
      <th>Month_sin</th>
      <th>Month_cos</th>
      <th>NO2_Category</th>
      <th>Time_of_Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.351</td>
      <td>1.206</td>
      <td>-0.531</td>
      <td>0.238</td>
      <td>0.397</td>
      <td>-0.372</td>
      <td>0.872</td>
      <td>0.046</td>
      <td>0.691</td>
      <td>0.614</td>
      <td>-0.542</td>
      <td>-0.012</td>
      <td>-0.668</td>
      <td>0.462</td>
      <td>1.035</td>
      <td>0.318</td>
      <td>1.717</td>
      <td>1.313</td>
      <td>1.069</td>
      <td>0.905</td>
      <td>-1.415</td>
      <td>-0.000</td>
      <td>1.302</td>
      <td>0.010</td>
      <td>0.893</td>
      <td>-0.448</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.081</td>
      <td>0.890</td>
      <td>-0.812</td>
      <td>-0.099</td>
      <td>0.054</td>
      <td>-0.687</td>
      <td>1.338</td>
      <td>-0.408</td>
      <td>0.302</td>
      <td>-0.134</td>
      <td>-0.577</td>
      <td>-0.082</td>
      <td>-0.749</td>
      <td>0.451</td>
      <td>0.828</td>
      <td>0.314</td>
      <td>1.426</td>
      <td>0.907</td>
      <td>0.973</td>
      <td>0.627</td>
      <td>-1.367</td>
      <td>0.366</td>
      <td>1.302</td>
      <td>0.010</td>
      <td>-0.841</td>
      <td>-0.448</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.063</td>
      <td>1.401</td>
      <td>-0.989</td>
      <td>-0.153</td>
      <td>-0.007</td>
      <td>-0.547</td>
      <td>1.204</td>
      <td>0.067</td>
      <td>0.290</td>
      <td>0.124</td>
      <td>-0.737</td>
      <td>0.285</td>
      <td>-0.687</td>
      <td>0.457</td>
      <td>1.160</td>
      <td>0.314</td>
      <td>1.283</td>
      <td>0.781</td>
      <td>0.887</td>
      <td>0.685</td>
      <td>-1.225</td>
      <td>0.707</td>
      <td>1.302</td>
      <td>0.010</td>
      <td>0.893</td>
      <td>-0.448</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.063</td>
      <td>1.281</td>
      <td>-1.048</td>
      <td>-0.126</td>
      <td>0.027</td>
      <td>-0.342</td>
      <td>1.014</td>
      <td>0.240</td>
      <td>0.375</td>
      <td>0.449</td>
      <td>-0.840</td>
      <td>0.636</td>
      <td>-0.595</td>
      <td>0.456</td>
      <td>1.082</td>
      <td>0.315</td>
      <td>1.244</td>
      <td>0.784</td>
      <td>0.853</td>
      <td>0.776</td>
      <td>-1.000</td>
      <td>1.000</td>
      <td>1.302</td>
      <td>0.010</td>
      <td>0.893</td>
      <td>-0.448</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.369</td>
      <td>0.797</td>
      <td>-1.262</td>
      <td>-0.490</td>
      <td>-0.395</td>
      <td>-0.547</td>
      <td>1.460</td>
      <td>0.110</td>
      <td>0.100</td>
      <td>0.215</td>
      <td>-0.817</td>
      <td>0.612</td>
      <td>-0.590</td>
      <td>0.444</td>
      <td>0.766</td>
      <td>0.311</td>
      <td>1.050</td>
      <td>0.339</td>
      <td>0.767</td>
      <td>0.600</td>
      <td>-0.708</td>
      <td>1.224</td>
      <td>1.302</td>
      <td>0.010</td>
      <td>0.893</td>
      <td>-0.448</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c3b6a82f-9616-49cf-9dbe-1edb82799ca2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-c3b6a82f-9616-49cf-9dbe-1edb82799ca2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c3b6a82f-9616-49cf-9dbe-1edb82799ca2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-4a968fc7-f245-4d07-aacf-3e9dde0cb4af">
  <button class="colab-df-quickchart" onclick="quickchart('df-4a968fc7-f245-4d07-aacf-3e9dde0cb4af')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-4a968fc7-f245-4d07-aacf-3e9dde0cb4af button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df_scaled.describe()
```





  <div id="df-95724d33-42bb-4133-a249-22a347290fba" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>NMHC(GT)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
      <th>CO_Product</th>
      <th>CO_Avg</th>
      <th>CO_Ratio</th>
      <th>NMHC_Product</th>
      <th>NMHC_Avg</th>
      <th>NMHC_Ratio</th>
      <th>Total_Polutan</th>
      <th>Year</th>
      <th>Day</th>
      <th>IsWeekend</th>
      <th>DayOfWeek_bit0</th>
      <th>DayOfWeek_bit1</th>
      <th>DayOfWeek_bit2</th>
      <th>Hour_sin</th>
      <th>Hour_cos</th>
      <th>Month_sin</th>
      <th>Month_cos</th>
      <th>CO_Category</th>
      <th>Temperature_Category</th>
      <th>NO2_Category</th>
      <th>Time_of_Day</th>
      <th>C6H6_Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.000</td>
      <td>0.000</td>
      <td>-0.000</td>
      <td>-0.000</td>
      <td>0.000</td>
      <td>-0.000</td>
      <td>-0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.000</td>
      <td>0.000</td>
      <td>-0.000</td>
      <td>0.000</td>
      <td>-0.000</td>
      <td>2004.240</td>
      <td>15.877</td>
      <td>0.287</td>
      <td>0.431</td>
      <td>0.429</td>
      <td>0.428</td>
      <td>-0.000</td>
      <td>-0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.365</td>
      <td>0.997</td>
      <td>-0.000</td>
      <td>-0.000</td>
      <td>1.371</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.427</td>
      <td>8.809</td>
      <td>0.453</td>
      <td>0.495</td>
      <td>0.495</td>
      <td>0.495</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.557</td>
      <td>0.817</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.522</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.449</td>
      <td>-2.110</td>
      <td>-1.587</td>
      <td>-1.353</td>
      <td>-2.103</td>
      <td>-1.192</td>
      <td>-2.024</td>
      <td>-2.351</td>
      <td>-2.647</td>
      <td>-2.031</td>
      <td>-2.317</td>
      <td>-2.328</td>
      <td>-2.107</td>
      <td>-4.196</td>
      <td>-3.676</td>
      <td>-2.639</td>
      <td>-1.786</td>
      <td>-2.878</td>
      <td>-8.837</td>
      <td>-2.295</td>
      <td>2004.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-1.415</td>
      <td>-1.414</td>
      <td>-1.460</td>
      <td>-1.445</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-2.576</td>
      <td>-1.342</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.729</td>
      <td>-0.752</td>
      <td>-1.033</td>
      <td>-0.760</td>
      <td>-0.772</td>
      <td>-0.697</td>
      <td>-0.674</td>
      <td>-0.751</td>
      <td>-0.646</td>
      <td>-0.728</td>
      <td>-0.725</td>
      <td>-0.765</td>
      <td>-0.708</td>
      <td>0.424</td>
      <td>-0.421</td>
      <td>0.304</td>
      <td>-0.455</td>
      <td>-0.710</td>
      <td>-0.417</td>
      <td>-0.337</td>
      <td>2004.000</td>
      <td>8.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-1.000</td>
      <td>-1.000</td>
      <td>-0.770</td>
      <td>-0.718</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>-0.841</td>
      <td>-0.448</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.159</td>
      <td>-0.157</td>
      <td>0.114</td>
      <td>-0.234</td>
      <td>-0.101</td>
      <td>-0.272</td>
      <td>-0.118</td>
      <td>-0.054</td>
      <td>0.021</td>
      <td>-0.142</td>
      <td>-0.061</td>
      <td>0.017</td>
      <td>-0.057</td>
      <td>0.437</td>
      <td>0.021</td>
      <td>0.312</td>
      <td>-0.197</td>
      <td>-0.141</td>
      <td>-0.220</td>
      <td>0.086</td>
      <td>2004.000</td>
      <td>16.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.000</td>
      <td>-0.000</td>
      <td>-0.079</td>
      <td>0.010</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.893</td>
      <td>0.447</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.495</td>
      <td>0.606</td>
      <td>0.569</td>
      <td>0.522</td>
      <td>0.661</td>
      <td>0.384</td>
      <td>0.525</td>
      <td>0.564</td>
      <td>0.624</td>
      <td>0.619</td>
      <td>0.672</td>
      <td>0.746</td>
      <td>0.695</td>
      <td>0.455</td>
      <td>0.553</td>
      <td>0.320</td>
      <td>0.045</td>
      <td>0.544</td>
      <td>-0.047</td>
      <td>0.570</td>
      <td>2004.000</td>
      <td>23.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.117</td>
      <td>0.738</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>0.893</td>
      <td>1.341</td>
      <td>2.000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.046</td>
      <td>4.369</td>
      <td>7.138</td>
      <td>7.225</td>
      <td>4.800</td>
      <td>6.197</td>
      <td>7.292</td>
      <td>4.946</td>
      <td>3.860</td>
      <td>3.783</td>
      <td>3.008</td>
      <td>2.310</td>
      <td>3.033</td>
      <td>0.902</td>
      <td>3.101</td>
      <td>10.301</td>
      <td>12.172</td>
      <td>6.449</td>
      <td>4.208</td>
      <td>3.776</td>
      <td>2005.000</td>
      <td>31.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.414</td>
      <td>1.414</td>
      <td>1.302</td>
      <td>1.465</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>0.893</td>
      <td>1.341</td>
      <td>2.000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-95724d33-42bb-4133-a249-22a347290fba')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-95724d33-42bb-4133-a249-22a347290fba button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-95724d33-42bb-4133-a249-22a347290fba');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-7dc40069-5cb9-4396-9983-8996ffedfaf9">
  <button class="colab-df-quickchart" onclick="quickchart('df-7dc40069-5cb9-4396-9983-8996ffedfaf9')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-7dc40069-5cb9-4396-9983-8996ffedfaf9 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




## PCA


```python
# Initialize PCA for 2 components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)

# Create a DataFrame for the principal components
x_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Display the explained variance ratio
print('Explained Variance Ratio by Principal Components:')
print(pca.explained_variance_ratio_)

# visualize pca
plt.figure(figsize=(10, 8))
sns.scatterplot(data=x_pca, x='PC1', y='PC2')
plt.title('PCA Result')
```

    Explained Variance Ratio by Principal Components:
    [0.73243509 0.08877611]





    Text(0.5, 1.0, 'PCA Result')




    
![png](nb_files/nb_59_2.png)
    


# **6. Clustering Model Building**

## **Clustering Model Building & Evaluation**


```python
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=11)

visualizer.fit(x_pca)
visualizer.show()
```


    
![png](nb_files/nb_62_0.png)
    





    <Axes: title={'center': 'Distortion Score Elbow for KMeans Clustering'}, xlabel='k', ylabel='distortion score'>




```python
model = KMeans(random_state=0)
visualizer = KElbowVisualizer(model, k=11, metric='silhouette', timings=False)

visualizer.fit(x_pca)
visualizer.show()

optimal_k = visualizer.elbow_value_
best_score_k = visualizer.elbow_score_
```


    
![png](nb_files/nb_63_0.png)
    



```python
def dbscan_silhouette_analysis(x_pca, eps_values, min_samples=5):
    silhouette_scores = []
    num_clusters_list = []

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(x_pca)

        # Compute the number of clusters
        num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        num_clusters_list.append(num_clusters)

        # Compute silhouette score if there are more than one cluster
        if num_clusters > 1:
            sil_score = silhouette_score(x_pca, dbscan_labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(None)  # Add None if silhouette score cannot be calculated

    # Filter valid silhouette scores
    valid_scores = [(eps, score) for eps, score in zip(eps_values, silhouette_scores) if score is not None]
    best_eps, best_score = None, None
    if valid_scores:
        best_eps, best_score = max(valid_scores, key=lambda x: x[1])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(
        eps_values,
        [s if s is not None else 0 for s in silhouette_scores],
        marker='o', label="Silhouette Score"
    )
    plt.xticks(eps_values)
    plt.xlabel("eps (DBSCAN parameter)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for DBSCAN Clustering")

    # Only draw the vertical line if best_eps is not None
    if best_eps is not None:
        plt.axvline(x=best_eps, linestyle='--', color='black',
                    label=f"Best eps = {best_eps}, score = {best_score:.3f}")

    plt.legend()
    plt.grid(True)
    plt.show()

    return best_eps, best_score, silhouette_scores
```


```python
eps_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]

best_eps, best_score, silhouette_scores = dbscan_silhouette_analysis(x_pca, eps_values)
```


    
![png](nb_files/nb_65_0.png)
    


## **Feature Selection**


```python
selected_features = ['NMHC_Ratio', 'NMHC_Product', 'CO_Ratio', 'CO_Product', 'Total_Polutan']
x = df_scaled[selected_features]
```


```python
principal_components = pca.fit_transform(x)
x_pca1 = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

print('Explained Variance Ratio by Principal Components:')
print(pca.explained_variance_ratio_)

# visualize pca
plt.figure(figsize=(10, 8))
sns.scatterplot(data=x_pca1, x='PC1', y='PC2')
plt.title('PCA Result')
```

    Explained Variance Ratio by Principal Components:
    [0.48926818 0.26600465]





    Text(0.5, 1.0, 'PCA Result')




    
![png](nb_files/nb_68_2.png)
    



```python
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=11)

visualizer.fit(x_pca)
visualizer.show()
```


    
![png](nb_files/nb_69_0.png)
    





    <Axes: title={'center': 'Distortion Score Elbow for KMeans Clustering'}, xlabel='k', ylabel='distortion score'>




```python
model = KMeans(random_state=0)
visualizer = KElbowVisualizer(model, k=11, metric='silhouette', timings=False)

visualizer.fit(x_pca1)
visualizer.show()

optimal_k1 = visualizer.elbow_value_
best_score_k1 = visualizer.elbow_score_
```


    
![png](nb_files/nb_70_0.png)
    



```python
eps_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]

best_eps, best_score, silhouette_scores = dbscan_silhouette_analysis(x_pca1, eps_values)
```


    
![png](nb_files/nb_71_0.png)
    


## **Cluster Visualization & Analysis**

### Before Feature Selection


```python
dbscan = DBSCAN(eps=0.8, min_samples=5)
clusters = dbscan.fit_predict(x_pca)

X_dbscan = x_pca.copy()
X_dbscan['Cluster'] = clusters
X_dbscan['Cluster'].value_counts()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>Cluster</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>312</td>
    </tr>
    <tr>
      <th>4</th>
      <td>312</td>
    </tr>
    <tr>
      <th>10</th>
      <td>311</td>
    </tr>
    <tr>
      <th>7</th>
      <td>311</td>
    </tr>
    <tr>
      <th>2</th>
      <td>311</td>
    </tr>
    <tr>
      <th>24</th>
      <td>311</td>
    </tr>
    <tr>
      <th>25</th>
      <td>311</td>
    </tr>
    <tr>
      <th>1</th>
      <td>310</td>
    </tr>
    <tr>
      <th>16</th>
      <td>310</td>
    </tr>
    <tr>
      <th>19</th>
      <td>310</td>
    </tr>
    <tr>
      <th>12</th>
      <td>309</td>
    </tr>
    <tr>
      <th>15</th>
      <td>309</td>
    </tr>
    <tr>
      <th>17</th>
      <td>309</td>
    </tr>
    <tr>
      <th>3</th>
      <td>308</td>
    </tr>
    <tr>
      <th>18</th>
      <td>307</td>
    </tr>
    <tr>
      <th>26</th>
      <td>306</td>
    </tr>
    <tr>
      <th>11</th>
      <td>306</td>
    </tr>
    <tr>
      <th>14</th>
      <td>306</td>
    </tr>
    <tr>
      <th>20</th>
      <td>304</td>
    </tr>
    <tr>
      <th>8</th>
      <td>303</td>
    </tr>
    <tr>
      <th>5</th>
      <td>299</td>
    </tr>
    <tr>
      <th>27</th>
      <td>297</td>
    </tr>
    <tr>
      <th>0</th>
      <td>293</td>
    </tr>
    <tr>
      <th>32</th>
      <td>288</td>
    </tr>
    <tr>
      <th>31</th>
      <td>287</td>
    </tr>
    <tr>
      <th>21</th>
      <td>286</td>
    </tr>
    <tr>
      <th>22</th>
      <td>286</td>
    </tr>
    <tr>
      <th>28</th>
      <td>286</td>
    </tr>
    <tr>
      <th>29</th>
      <td>286</td>
    </tr>
    <tr>
      <th>30</th>
      <td>286</td>
    </tr>
    <tr>
      <th>23</th>
      <td>188</td>
    </tr>
    <tr>
      <th>-1</th>
      <td>69</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10</td>
    </tr>
    <tr>
      <th>33</th>
      <td>8</td>
    </tr>
    <tr>
      <th>34</th>
      <td>6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



**DBSCAN RESULT:**

The clusters formed are unrepresentative (too many cluster), thus providing no meaningful insight into the patterns/clustering in the data.


```python
kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(x_pca)

X_kmeans = x_pca.copy()
X_kmeans['Cluster'] = clusters
X_kmeans['Cluster'].value_counts()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>Cluster</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4824</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4533</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



**K-MEANS RESULT**:

From the clustering results using K-Means with k=2 (n_clusters=2), we see better results than the previous DBSCAN

The cluster distribution is more balanced


```python
plt.figure(figsize=(10, 8))
sns.scatterplot(data=X_kmeans, x='PC1', y='PC2', hue='Cluster', palette='deep')
plt.title('K-Means Clustering Result (k=2) | Before Feature Selection')

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](nb_files/nb_78_0.png)
    



```python
print(f"Optimal k: {optimal_k}")
print(f"Best silhouette score: {best_score_k}")
```

    Optimal k: 2
    Best silhouette score: 0.5373234622126103


**NOTE:**

The silhouette score from the KMeans clustering is 0.54, which is below 0.55. This indicates poor cluster separation.

### After Feature Selection


```python
kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(x_pca1)

X_kmeans1 = x_pca1.copy()
X_kmeans1['Cluster'] = clusters
X_kmeans1['Cluster'].value_counts()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>Cluster</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7710</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1647</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
plt.figure(figsize=(10, 8))
sns.scatterplot(data=X_kmeans1, x='PC1', y='PC2', hue='Cluster', palette='deep')
plt.title('K-Means Clustering Result (k=2) | After Feature Selection')

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](nb_files/nb_83_0.png)
    



```python
print(f"Optimal k: {optimal_k1}")
print(f"Best silhouette score: {best_score_k1}")
```

    Optimal k: 3
    Best silhouette score: 0.8109608946859179


**INSIGHT:**

The silhoutte score of the clustering after feature selection is 0.81. Which is where the score results are above 0.55 which indicates a fairly good cluster separation.

## **Interpretation of Cluster Results**

1. Cluster 1:
2. Cluster 2:
3. Cluster 3:

# **7. Exporting Data**


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 9357 entries, 0 to 9356
    Data columns (total 35 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   CO(GT)                9357 non-null   float64
     1   PT08.S1(CO)           9357 non-null   float64
     2   NMHC(GT)              9357 non-null   float64
     3   C6H6(GT)              9357 non-null   float64
     4   PT08.S2(NMHC)         9357 non-null   float64
     5   NOx(GT)               9357 non-null   float64
     6   PT08.S3(NOx)          9357 non-null   float64
     7   NO2(GT)               9357 non-null   float64
     8   PT08.S4(NO2)          9357 non-null   float64
     9   PT08.S5(O3)           9357 non-null   float64
     10  T                     9357 non-null   float64
     11  RH                    9357 non-null   float64
     12  AH                    9357 non-null   float64
     13  CO_Product            9357 non-null   float64
     14  CO_Avg                9357 non-null   float64
     15  CO_Ratio              9357 non-null   float64
     16  NMHC_Product          9357 non-null   float64
     17  NMHC_Avg              9357 non-null   float64
     18  NMHC_Ratio            9357 non-null   float64
     19  Total_Polutan         9357 non-null   float64
     20  Year                  9357 non-null   int32  
     21  Day                   9357 non-null   int32  
     22  IsWeekend             9357 non-null   int64  
     23  DayOfWeek_bit0        9357 non-null   int64  
     24  DayOfWeek_bit1        9357 non-null   int64  
     25  DayOfWeek_bit2        9357 non-null   int64  
     26  Hour_sin              9357 non-null   float64
     27  Hour_cos              9357 non-null   float64
     28  Month_sin             9357 non-null   float64
     29  Month_cos             9357 non-null   float64
     30  CO_Category           9357 non-null   int64  
     31  Temperature_Category  9357 non-null   int64  
     32  NO2_Category          9357 non-null   int64  
     33  Time_of_Day           9357 non-null   int64  
     34  C6H6_Category         9357 non-null   int64  
    dtypes: float64(24), int32(2), int64(9)
    memory usage: 2.5 MB



```python
df['Cluster'] = X_kmeans1['Cluster'].values
```


```python
df['Cluster'].value_counts()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>Cluster</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7710</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1647</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
df.describe()
```





  <div id="df-2cda080e-9ec4-42ed-85e7-9a87dbc1b626" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>NMHC(GT)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
      <th>CO_Product</th>
      <th>CO_Avg</th>
      <th>CO_Ratio</th>
      <th>NMHC_Product</th>
      <th>NMHC_Avg</th>
      <th>NMHC_Ratio</th>
      <th>Total_Polutan</th>
      <th>Year</th>
      <th>Day</th>
      <th>IsWeekend</th>
      <th>DayOfWeek_bit0</th>
      <th>DayOfWeek_bit1</th>
      <th>DayOfWeek_bit2</th>
      <th>Hour_sin</th>
      <th>Hour_cos</th>
      <th>Month_sin</th>
      <th>Month_cos</th>
      <th>CO_Category</th>
      <th>Temperature_Category</th>
      <th>NO2_Category</th>
      <th>Time_of_Day</th>
      <th>C6H6_Category</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
      <td>9357.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.113</td>
      <td>1100.677</td>
      <td>221.968</td>
      <td>10.133</td>
      <td>940.735</td>
      <td>240.273</td>
      <td>834.944</td>
      <td>110.882</td>
      <td>1455.857</td>
      <td>1025.052</td>
      <td>18.334</td>
      <td>49.107</td>
      <td>1.024</td>
      <td>-34816.319</td>
      <td>509.721</td>
      <td>-0.030</td>
      <td>-137052.864</td>
      <td>389.640</td>
      <td>-0.148</td>
      <td>35.324</td>
      <td>2004.240</td>
      <td>15.877</td>
      <td>0.287</td>
      <td>0.431</td>
      <td>0.429</td>
      <td>0.428</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.058</td>
      <td>-0.007</td>
      <td>1.365</td>
      <td>0.997</td>
      <td>1.485</td>
      <td>1.500</td>
      <td>1.371</td>
      <td>0.176</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.389</td>
      <td>215.017</td>
      <td>135.482</td>
      <td>7.414</td>
      <td>265.275</td>
      <td>199.909</td>
      <td>253.466</td>
      <td>46.325</td>
      <td>341.806</td>
      <td>395.963</td>
      <td>8.732</td>
      <td>17.140</td>
      <td>0.398</td>
      <td>82978.597</td>
      <td>165.826</td>
      <td>0.100</td>
      <td>171180.822</td>
      <td>158.691</td>
      <td>0.273</td>
      <td>451.206</td>
      <td>0.427</td>
      <td>8.809</td>
      <td>0.453</td>
      <td>0.495</td>
      <td>0.495</td>
      <td>0.495</td>
      <td>0.707</td>
      <td>0.707</td>
      <td>0.724</td>
      <td>0.687</td>
      <td>0.557</td>
      <td>0.817</td>
      <td>0.577</td>
      <td>1.118</td>
      <td>0.522</td>
      <td>0.381</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.100</td>
      <td>647.000</td>
      <td>7.000</td>
      <td>0.100</td>
      <td>383.000</td>
      <td>2.000</td>
      <td>322.000</td>
      <td>2.000</td>
      <td>551.000</td>
      <td>221.000</td>
      <td>-1.900</td>
      <td>9.200</td>
      <td>0.185</td>
      <td>-383000.000</td>
      <td>-99.850</td>
      <td>-0.294</td>
      <td>-442800.000</td>
      <td>-67.000</td>
      <td>-2.560</td>
      <td>-1000.000</td>
      <td>2004.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-1.000</td>
      <td>-1.000</td>
      <td>-1.000</td>
      <td>-1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.100</td>
      <td>939.000</td>
      <td>82.051</td>
      <td>4.500</td>
      <td>736.000</td>
      <td>101.000</td>
      <td>664.000</td>
      <td>76.078</td>
      <td>1235.000</td>
      <td>737.000</td>
      <td>12.000</td>
      <td>36.000</td>
      <td>0.742</td>
      <td>397.000</td>
      <td>439.850</td>
      <td>0.001</td>
      <td>-215000.000</td>
      <td>277.000</td>
      <td>-0.262</td>
      <td>-116.600</td>
      <td>2004.000</td>
      <td>8.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.707</td>
      <td>-0.707</td>
      <td>-0.500</td>
      <td>-0.500</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.892</td>
      <td>1067.000</td>
      <td>237.396</td>
      <td>8.400</td>
      <td>914.000</td>
      <td>186.000</td>
      <td>805.000</td>
      <td>108.396</td>
      <td>1463.000</td>
      <td>969.000</td>
      <td>17.800</td>
      <td>49.400</td>
      <td>1.001</td>
      <td>1415.400</td>
      <td>513.200</td>
      <td>0.001</td>
      <td>-170800.000</td>
      <td>367.339</td>
      <td>-0.208</td>
      <td>74.200</td>
      <td>2004.000</td>
      <td>16.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.800</td>
      <td>1231.000</td>
      <td>299.000</td>
      <td>14.000</td>
      <td>1116.000</td>
      <td>317.000</td>
      <td>968.000</td>
      <td>137.000</td>
      <td>1669.000</td>
      <td>1270.000</td>
      <td>24.200</td>
      <td>61.900</td>
      <td>1.300</td>
      <td>2958.800</td>
      <td>601.500</td>
      <td>0.002</td>
      <td>-129400.000</td>
      <td>476.000</td>
      <td>-0.161</td>
      <td>292.700</td>
      <td>2004.000</td>
      <td>23.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.707</td>
      <td>0.707</td>
      <td>0.866</td>
      <td>0.500</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>3.000</td>
      <td>2.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11.900</td>
      <td>2040.000</td>
      <td>1189.000</td>
      <td>63.700</td>
      <td>2214.000</td>
      <td>1479.000</td>
      <td>2683.000</td>
      <td>340.000</td>
      <td>2775.000</td>
      <td>2523.000</td>
      <td>44.600</td>
      <td>88.700</td>
      <td>2.231</td>
      <td>40000.000</td>
      <td>1024.000</td>
      <td>1.000</td>
      <td>1946393.000</td>
      <td>1413.000</td>
      <td>1.000</td>
      <td>1739.000</td>
      <td>2005.000</td>
      <td>31.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>2.000</td>
      <td>3.000</td>
      <td>2.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2cda080e-9ec4-42ed-85e7-9a87dbc1b626')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2cda080e-9ec4-42ed-85e7-9a87dbc1b626 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2cda080e-9ec4-42ed-85e7-9a87dbc1b626');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-07522fbb-6f73-44a2-8ced-813de0aec470">
  <button class="colab-df-quickchart" onclick="quickchart('df-07522fbb-6f73-44a2-8ced-813de0aec470')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-07522fbb-6f73-44a2-8ced-813de0aec470 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Export
df.to_csv('/content/drive/MyDrive/Colab Notebooks/df_cluster.csv', index=False)
```
