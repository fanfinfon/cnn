#!/usr/bin/env python
# coding: utf-8

# In[1]:


# BACKGROUND-RUN SUPPORT
import os, time, threading, pickle, json
    
BASE = os.path.join(os.path.expanduser("~"), "project")
OUTPUT_DIR = f"{BASE}/outputs"
CHECKPOINT_DIR = f"{BASE}/checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DATA_CSV = f"{BASE}/data.csv"
IMEI_CSV = f"{BASE}/imei_summary.csv"
IMEI_CSV_FALLBACK = f"{BASE}/imei.csv"
GRADE_CSV = f"{BASE}/driver_risk_grades_axa.csv"
TRIP_CSV = f"{BASE}/trip.csv"

print("Drive folder:", BASE)
print("Files in folder:", os.listdir(BASE))

# Keep periodic output so the runtime is less likely to be treated as idle.
def keep_alive(interval_seconds=300):
    while True:
        print(f"[keep-alive] {time.strftime('%H:%M:%S')} still running...", flush=True)
        time.sleep(interval_seconds)

if not globals().get("KEEP_ALIVE_STARTED", False):
    threading.Thread(target=keep_alive, daemon=True).start()
    KEEP_ALIVE_STARTED = True


# In[2]:


# ============================================================
# CELL 1 — LOAD & CLEAN RAW GPS DATA
# ============================================================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

FILE_PATH = DATA_CSV  # Drive path: /content/drive/MyDrive/gpu_paid/data.csv

df = pd.read_csv(FILE_PATH, encoding='utf-8', engine='python')
print('Original columns:', list(df.columns))

df['imei']      = df['imei'].astype(str).str.strip()
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['speed']     = pd.to_numeric(df['speed'], errors='coerce')

for col in ['direction', 'mileage', 'satellite', 'acceleration', 'deceleration']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['imei', 'timestamp', 'speed']).copy()
print('After cleaning shape:', df.shape)

# Key-duplicate aggregation
agg_dict = {'speed': 'mean'}
for col, agg in [('direction','mean'), ('mileage','max'), ('satellite','mean'),
                  ('acceleration','mean'), ('deceleration','mean'),
                  ('speed_violation','last'), ('road_type','last')]:
    if col in df.columns:
        agg_dict[col] = agg

df = df.groupby(['imei', 'timestamp'], as_index=False).agg(agg_dict)
print('After dedup shape:', df.shape)
print('Unique drivers   :', df['imei'].nunique())


# In[3]:


# ============================================================
# CELL 2 — PARAMETERS & ACCEL RECOMPUTE FROM SPEED DIFF
# ============================================================
TRIP_GAP_SECONDS = 300    # 5-min gap -> new trip
MIN_TRIP_RECORDS = 20     # meaningful trip threshold
MIN_MOVING_SHARE = 0.10   # at least 10% moving points
HARSH_A          = 3.0    # harsh accel threshold (m/s^2)
HARSH_B          = 3.0    # harsh brake threshold (m/s^2)
SEVERE_B         = 5.0    # severe brake threshold (m/s^2)
MAX_GAP_S        = 300    # cap for time-weighted intervals
NIGHT_HOURS      = set(range(22, 24)) | set(range(0, 6))
SPEED_THRESH     = 120    # km/h for pct_above_120

df = df.sort_values(['imei', 'timestamp']).copy()
df['dt_s']      = df.groupby('imei')['timestamp'].diff().dt.total_seconds()
df['dv_kmh']    = df.groupby('imei')['speed'].diff()
df['dv_ms']     = df['dv_kmh'] / 3.6
df['accel_ms2'] = np.where(
    (df['dt_s'] > 0) & (df['dt_s'] <= TRIP_GAP_SECONDS),
    df['dv_ms'] / df['dt_s'], np.nan
)
df['accel_abs'] = df['accel_ms2'].abs()

print('Accel recomputed from speed diff.')
print('Max accel (m/s2):', round(df['accel_ms2'].max(), 3))
print('Min accel (m/s2):', round(df['accel_ms2'].min(), 3))


# In[4]:


# ============================================================
# CELL 3 — TRIP SEGMENTATION
# ============================================================
df = df.sort_values(['imei', 'timestamp']).copy()

df['gap_to_prev'] = df.groupby('imei')['timestamp'].diff().dt.total_seconds().fillna(0)
df['new_trip']    = (df['gap_to_prev'] > TRIP_GAP_SECONDS).astype(int)
df['trip_id']     = df.groupby('imei')['new_trip'].cumsum() + 1
df['hour']        = df['timestamp'].dt.hour
df['is_night']    = df['hour'].isin(NIGHT_HOURS)

def trip_features(g):
    n       = len(g)
    moving  = (g['speed'] > 0).mean()
    dt      = g['timestamp'].diff().dt.total_seconds().clip(upper=MAX_GAP_S).fillna(0)
    dist_km = (g['speed'] / 3.6 * dt).sum() / 1000
    dur_min = dt.sum() / 60
    spd_kmh = g['speed']
    accel   = g['accel_ms2'].dropna()
    pos_acc = accel[accel > 0]
    neg_acc = accel[accel < 0].abs()
    night_r = g.loc[g['speed'] > 0, 'is_night'].mean() if (g['speed'] > 0).any() else 0.0
    return pd.Series({
        'trip_start'      : g['timestamp'].min(),
        'trip_end'        : g['timestamp'].max(),
        'n_records'       : n,
        'moving_share'    : moving,
        'night_ratio'     : night_r,
        'speed_mean_kmh'  : spd_kmh.mean(),
        'speed_p95_kmh'   : spd_kmh.quantile(0.95),
        'accel_p95'       : pos_acc.quantile(0.95) if len(pos_acc) else 0.0,
        'brake_p95'       : neg_acc.quantile(0.95) if len(neg_acc) else 0.0,
        'decel_kmhps_p95' : (neg_acc * 3.6).quantile(0.95) if len(neg_acc) else 0.0,
        'harsh_accel_cnt' : (pos_acc >= HARSH_A).sum(),
        'harsh_brake_cnt' : (neg_acc >= HARSH_B).sum(),
        'severe_brake_cnt': (neg_acc >= SEVERE_B).sum(),
        'dist_km'         : dist_km,
        'duration_min'    : dur_min,
    })

trip_summary = (
    df.groupby(['imei', 'trip_id'])
    .apply(trip_features)
    .reset_index()
)

trip_summary['is_meaningful_trip'] = (
    (trip_summary['n_records']    >= MIN_TRIP_RECORDS) &
    (trip_summary['moving_share'] >= MIN_MOVING_SHARE)
)

print('Trip summary shape:', trip_summary.shape)
print(trip_summary['is_meaningful_trip'].value_counts())


# In[5]:


# ============================================================
# CELL 4 — DRIVER FEATURE MATRIX (imei_summary)
#
# 4 dimensions:
#   A. Exposure    — how much the driver is on the road
#   B. Aggression  — how aggressively the driver behaves
#   C. Variability — consistency across trips
#   D. Context     — when and how the driver operates
# ============================================================
tsm = trip_summary[trip_summary['is_meaningful_trip']].copy()
eps = 1e-6

imei_summary = (
    tsm.groupby('imei', as_index=False)
    .agg(
        # A. Exposure
        n_trips                 = ('trip_id',         'nunique'),
        total_km                = ('dist_km',          'sum'),
        total_duration_min      = ('duration_min',     'sum'),
        avg_trip_distance       = ('dist_km',          'mean'),
        avg_trip_duration       = ('duration_min',     'mean'),
        # B. Aggression
        mean_speed              = ('speed_mean_kmh',   'mean'),
        p95_speed               = ('speed_p95_kmh',    'mean'),
        mean_accel_p95          = ('accel_p95',        'mean'),
        mean_brake_p95          = ('brake_p95',        'mean'),
        mean_decel_kmhps        = ('decel_kmhps_p95',  'mean'),
        harsh_accel_total       = ('harsh_accel_cnt',  'sum'),
        harsh_brake_total       = ('harsh_brake_cnt',  'sum'),
        severe_brake_total      = ('severe_brake_cnt', 'sum'),
        # C. Variability
        speed_std_between_trips = ('speed_mean_kmh',   'std'),
        brake_std_between_trips = ('brake_p95',        'std'),
        accel_std_between_trips = ('accel_p95',        'std'),
        duration_std            = ('duration_min',     'std'),
        # D. Context
        mean_moving_share       = ('moving_share',     'mean'),
        night_ratio             = ('night_ratio',      'mean'),
    )
)

# Normalised event rates per 100 km
imei_summary['harsh_brake_per_100km']  = imei_summary['harsh_brake_total']  / (imei_summary['total_km'] + eps) * 100
imei_summary['harsh_accel_per_100km'] = imei_summary['harsh_accel_total']  / (imei_summary['total_km'] + eps) * 100
imei_summary['severe_brake_per_100km']= imei_summary['severe_brake_total'] / (imei_summary['total_km'] + eps) * 100

print('imei_summary shape:', imei_summary.shape)
print('Drivers           :', len(imei_summary))
print('Columns           :', list(imei_summary.columns))


# In[6]:


# ============================================================
# CELL 5 — pct_above_120 & night_driving_h (time-weighted)
# ============================================================
df_s = df.sort_values(['imei', 'timestamp']).copy()
df_s['gap_s'] = (
    df_s.groupby('imei')['timestamp']
    .diff().dt.total_seconds()
    .clip(upper=MAX_GAP_S)
    .fillna(0)
)
df_s['is_fast']   = df_s['speed'] > SPEED_THRESH
df_s['is_night2'] = df_s['timestamp'].dt.hour.isin(NIGHT_HOURS)

feat_speed = (
    df_s.groupby('imei')
    .apply(lambda g: pd.Series({
        'total_driving_s'  : g['gap_s'].sum(),
        'time_above_120_s' : g.loc[g['is_fast'],   'gap_s'].sum(),
        'time_night_s'     : g.loc[g['is_night2'], 'gap_s'].sum(),
    }))
    .reset_index()
)
feat_speed['pct_above_120']   = np.where(
    feat_speed['total_driving_s'] > 0,
    feat_speed['time_above_120_s'] / feat_speed['total_driving_s'] * 100, 0.0
)
feat_speed['night_driving_h'] = feat_speed['time_night_s'] / 3600

imei_summary = imei_summary.merge(
    feat_speed[['imei', 'pct_above_120', 'night_driving_h']],
    on='imei', how='left'
)
imei_summary[['pct_above_120', 'night_driving_h']] = (
    imei_summary[['pct_above_120', 'night_driving_h']].fillna(0)
)
print('pct_above_120 max   :', round(imei_summary['pct_above_120'].max(), 4))
print('night_driving_h max :', round(imei_summary['night_driving_h'].max(), 4))
print('imei_summary shape  :', imei_summary.shape)


# In[7]:


# ============================================================
# CELL 6 — AXA RISK SCORE -> A-E LABELS
# ============================================================
from sklearn.preprocessing import MinMaxScaler

LABEL_FEATURES = [
    'p95_speed',
    'harsh_brake_per_100km',
    'mean_accel_p95',
    'speed_std_between_trips',
]

AXA_WEIGHTS = {
    'p95_speed'               : 0.30,
    'harsh_brake_per_100km'   : 0.30,
    'mean_accel_p95'          : 0.20,
    'speed_std_between_trips' : 0.20,
}

df_label = imei_summary[['imei'] + LABEL_FEATURES].copy()
df_label = df_label[df_label['p95_speed'] > 0].reset_index(drop=True)
df_label[LABEL_FEATURES] = df_label[LABEL_FEATURES].fillna(0)

sc_label = MinMaxScaler()
normed   = pd.DataFrame(
    sc_label.fit_transform(df_label[LABEL_FEATURES]),
    columns=LABEL_FEATURES
)
df_label['risk_score'] = sum(
    normed[col] * AXA_WEIGHTS[col] for col in LABEL_FEATURES
)

df_label['grade'], bin_edges = pd.qcut(
    df_label['risk_score'], q=5,
    labels=['A', 'B', 'C', 'D', 'E'],
    retbins=True, duplicates='drop'
)

print('AXA grade distribution:')
print(df_label['grade'].value_counts().sort_index())
print('Risk score quantile edges:', np.round(bin_edges, 4))

df_label.to_csv(GRADE_CSV, index=False)
print(f'Saved -> {GRADE_CSV}')
_, bins = pd.qcut(df_label['risk_score'], q=3,
                  labels=['Low', 'Medium', 'High'],
                  retbins=True, duplicates='drop')
print("\n3-class risk score thresholds (quantile-based):")
print(f"  Low    : risk_score < {bins[1]:.4f}")
print(f"  Medium : {bins[1]:.4f} <= risk_score < {bins[2]:.4f}")
print(f"  High   : risk_score >= {bins[2]:.4f}")
print("\nRisk score distribution:")
print(df_label['risk_score'].describe().round(4))


# In[8]:


# ============================================================
# CELL 7 — EXTENDED FEATURE MATRIX (17 features, 4 dimensions)
# ============================================================
FEATURES = {
    'A_Exposure'   : ['total_km', 'n_trips', 'total_duration_min', 'avg_trip_distance'],
    'B_Aggression' : ['p95_speed', 'mean_accel_p95', 'mean_brake_p95',
                      'harsh_brake_per_100km', 'harsh_accel_per_100km',
                      'severe_brake_per_100km'],
    'C_Variability': ['speed_std_between_trips', 'brake_std_between_trips',
                      'accel_std_between_trips', 'duration_std'],
    'D_Context'    : ['night_ratio', 'mean_moving_share', 'pct_above_120'],
}
ALL_FEATURES = [f for grp in FEATURES.values() for f in grp]

print(f'Total features: {len(ALL_FEATURES)}')
for dim, feats in FEATURES.items():
    print(f'  {dim}: {feats}')

df_model = (
    imei_summary[['imei'] + ALL_FEATURES]
    .merge(df_label[['imei', 'grade']], on='imei', how='inner')
    .dropna(subset=['grade'])
    .reset_index(drop=True)
)
df_model[ALL_FEATURES] = df_model[ALL_FEATURES].fillna(0)

print(f'\nFinal driver count : {len(df_model)}')
print('Grade distribution :')
print(df_model['grade'].value_counts().sort_index())
print('\nFeature value ranges (raw):')
for col in ALL_FEATURES:
    print(f'  {col:<30}  min={df_model[col].min():.3f}  max={df_model[col].max():.3f}')

print('\nNotebook 1 complete.')
print('df_model and imei_summary are ready in memory.')


# In[9]:


# ============================================================
# CELL 9b — 3-CLASS REMAPPING
# ============================================================
GRADE_MAP = {'A':'Low', 'B':'Low', 'C':'Medium', 'D':'High', 'E':'High'}

df_model['grade'] = df_model['grade'].astype(str).map(GRADE_MAP)

print('3-class remapping applied:')
print('  A + B  -> Low    (discount)')
print('  C      -> Medium (standard)')
print('  D + E  -> High   (loading)')
print()
print('New distribution:')
print(df_model['grade'].value_counts().sort_index())
print()
print('Random baseline: 1/3 = 0.333  (was 1/5 = 0.200)')
print('-> Continue to Cell 10 (X/y prep).')


# In[11]:


# ============================================================
# CELL — ML IMPORTS + VERIFY
# ============================================================
import numpy as np
import pandas as pd
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K, regularizers
import ray
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import time
import logging
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

logging.basicConfig(
    filename=f'{OUTPUT_DIR}/pso_run.log',
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S'
)
def log(msg):
    print(msg)
    logging.info(msg)
TIMING = {}

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print('TensorFlow:', tf.__version__)
assert 'df_model'     in dir(), 'Run Fast Load or pipeline cells first'
assert 'ALL_FEATURES' in dir(), 'Run Fast Load or pipeline cells first'
print(f'Drivers : {len(df_model)}  Features: {len(ALL_FEATURES)}')
print('Grade dist:', df_model['grade'].value_counts().sort_index().to_dict())


# In[12]:


# ============================================================
# CELL — PREPARE X AND y
# ============================================================
X_raw = df_model[ALL_FEATURES].values.astype(np.float32)
le    = LabelEncoder()
y     = le.fit_transform(df_model['grade'])

N_FEATURES  = X_raw.shape[1]
N_CLASSES   = len(le.classes_)
CLASS_NAMES = list(le.classes_)

print(f'X shape : {X_raw.shape}')
print(f'Classes : {le.classes_}  Per-class: {np.bincount(y)}')

import ray
ray.init(num_gpus=3, ignore_reinit_error=True)
X_raw_id = ray.put(X_raw)
y_id = ray.put(y)


# In[14]:


# ============================================================
# CELL — MODEL BUILDER + CV HELPERS
# ============================================================
N_SPLITS   = 10
MAX_EPOCHS = 1000
BATCH_SIZE = 32

ES = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=80,
    restore_best_weights=True, verbose=0)
LR_REDUCE = keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.5,
    patience=30, min_lr=1e-5, verbose=0)


def rbf_activation(x):
    return K.exp(-K.square(x))


def build_cnn(cnn_act='relu', fc_act='linear',
              filters_1=32, filters_2=64, kernel_size=3,
              dropout=0.3, dense_units=64, lr=1e-3):
    inp = keras.Input(shape=(N_FEATURES, 1))
    x = layers.Conv1D(filters_1, kernel_size=kernel_size, padding='same', activation=None, name='conv1')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(cnn_act)(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool1')(x)
    x = layers.Conv1D(filters_2, kernel_size=kernel_size, padding='same', activation=None, name='conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(cnn_act)(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(dropout, name='dropout')(x)
    if fc_act == 'rbf':
        x = layers.Dense(dense_units,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
            kernel_regularizer=regularizers.l2(1e-4), name='fc')(x)
        x = layers.Lambda(rbf_activation, name='rbf_act')(x)
    else:
        x = layers.Dense(dense_units, activation=fc_act,
            kernel_regularizer=regularizers.l2(1e-4), name='fc')(x)
    out = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def augment(X, y_arr, noise_std=0.02, copies=2):
    parts_X, parts_y = [X], [y_arr]
    for _ in range(copies):
        noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
        parts_X.append(np.clip(X + noise, 0, 1))
        parts_y.append(y_arr)
    return np.vstack(parts_X), np.concatenate(parts_y)


@ray.remote(num_gpus=1)
def _cv_cnn_fold_remote(fold, tr, te, X_raw_data, y_data, cnn_act, fc_act, collect_history, collect_preds, seed, max_epochs, batch_size):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import backend as K
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score
    import os
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    np.random.seed(seed + fold)
    tf.random.set_seed(seed + fold)
    sc    = MinMaxScaler()
    Xtr_s = sc.fit_transform(X_raw_data[tr]).astype(np.float32)
    Xte_s = sc.transform(X_raw_data[te]).astype(np.float32)
    Xtr_aug, ytr_aug = augment(Xtr_s, y_data[tr])
    m = build_cnn(cnn_act=cnn_act, fc_act=fc_act)
    ES_local = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=80,
        restore_best_weights=True, verbose=0)
    LR_REDUCE_local = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5,
        patience=30, min_lr=1e-5, verbose=0)
        
    hist = m.fit(Xtr_aug.reshape(-1, N_FEATURES, 1), ytr_aug,
                 validation_split=0.10, epochs=max_epochs,
                 batch_size=batch_size, callbacks=[ES_local, LR_REDUCE_local], verbose=0)
                 
    history_val = hist.history['val_accuracy'] if collect_history else None
    preds = np.argmax(m.predict(Xte_s.reshape(-1, N_FEATURES, 1), verbose=0), axis=1)
    acc = round(accuracy_score(y_data[te], preds), 3)
    K.clear_session()
    
    return acc, (y_data[te] if collect_preds else None), (preds if collect_preds else None), history_val

def cv_cnn_softmax(cnn_act='relu', fc_act='linear', label='',
                   collect_preds=False, collect_history=False):
    if not label: label = f'CNN-Softmax {cnn_act}+{fc_act}'
    print(f'\n[{label}]')
    _t0 = time.time()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    accs, all_true, all_pred, fold_histories = [], [], [], []
    
    futures = []
    for fold, (tr, te) in enumerate(skf.split(X_raw, y)):
        futures.append(_cv_cnn_fold_remote.remote(fold, tr, te, X_raw_id, y_id, cnn_act, fc_act, collect_history, collect_preds, SEED, MAX_EPOCHS, BATCH_SIZE))
        
    results = ray.get(futures)
    for fold, res in enumerate(results):
        acc, t_labels, p_labels, history_val = res
        accs.append(acc)
        if collect_preds:
            all_true.extend(t_labels)
            all_pred.extend(p_labels)
        if collect_history:
            fold_histories.append(history_val)
        print(f'  Fold {fold+1:2d}: {acc:.3f}')
        
    _elapsed = time.time() - _t0
    TIMING[label] = _elapsed
    print(f'  -- Average: {np.mean(accs):.3f}  |  Time: {_elapsed:.1f}s ({_elapsed/60:.1f} min)')
    log(f'{label}: avg={np.mean(accs):.3f}  time={_elapsed:.1f}s')
    if collect_preds and collect_history:
        return accs, np.array(all_true), np.array(all_pred), fold_histories
    if collect_preds:
        return accs, np.array(all_true), np.array(all_pred)
    if collect_history:
        return accs, fold_histories
    return accs


@ray.remote(num_cpus=1)
def _cv_svm_fold_remote(fold, tr, te, X_raw_data, y_data, seed):
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC
    
    np.random.seed(seed + fold)
    sc  = MinMaxScaler()
    clf = SVC(kernel='rbf', random_state=seed)
    clf.fit(sc.fit_transform(X_raw_data[tr]), y_data[tr])
    preds = clf.predict(sc.transform(X_raw_data[te]))
    acc   = round(accuracy_score(y_data[te], preds), 3)
    return acc, (y_data[te] if True else None), (preds if True else None)

def cv_svm(collect_preds=False):
    print('\n[SVM - RBF kernel]')
    _t0 = time.time()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    accs, all_true, all_pred, fold_histories = [], [], [], []
    
    futures = []
    for fold, (tr, te) in enumerate(skf.split(X_raw, y)):
        futures.append(_cv_svm_fold_remote.remote(fold, tr, te, X_raw_id, y_id, SEED))
        
    results = ray.get(futures)
    for fold, res in enumerate(results):
        acc, t_labels, p_labels = res
        accs.append(acc)
        if collect_preds:
            all_true.extend(t_labels)
            all_pred.extend(p_labels)
        print(f'  Fold {fold+1:2d}: {acc:.3f}')
        
    print(f'  -- Average: {np.mean(accs):.3f}')
    if collect_preds:
        return accs, np.array(all_true), np.array(all_pred)
    return accs


@ray.remote(num_gpus=1)
def _cv_shared_fold_remote(fold, tr, te, X_raw_data, y_data, cnn_act, k_knn, n_rf, filters_1, filters_2, kernel_size, dropout, dense_units, lr, seed, batch_size):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import backend as K
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    import os
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    np.random.seed(seed + fold)
    tf.random.set_seed(seed + fold)
    sc        = MinMaxScaler()
    Xtr_s     = sc.fit_transform(X_raw_data[tr]).astype(np.float32)
    Xte_s     = sc.transform(X_raw_data[te]).astype(np.float32)
    Xtr_aug, ytr_aug = augment(Xtr_s, y_data[tr])
    Xtr_aug_r = Xtr_aug.reshape(-1, N_FEATURES, 1)
    Xtr_r     = Xtr_s.reshape(-1, N_FEATURES, 1)
    Xte_r     = Xte_s.reshape(-1, N_FEATURES, 1)
    cnn = build_cnn(cnn_act=cnn_act, fc_act='relu',
                    filters_1=filters_1, filters_2=filters_2,
                    kernel_size=kernel_size, dropout=dropout,
                    dense_units=dense_units, lr=lr)
    ES_local = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=80,
        restore_best_weights=True, verbose=0)
    LR_REDUCE_local = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5,
        patience=30, min_lr=1e-5, verbose=0)
        
    cnn.fit(Xtr_aug_r, ytr_aug, validation_split=0.10,
            epochs=500, batch_size=batch_size,
            callbacks=[ES_local, LR_REDUCE_local], verbose=0)
    ext  = keras.Model(inputs=cnn.input, outputs=cnn.get_layer('fc').output)
    Ftr  = ext(Xtr_r, training=False).numpy()
    Fte  = ext(Xte_r, training=False).numpy()
    K.clear_session()
    
    clfs = {
        'cnn_svm': SVC(kernel='rbf', random_state=seed),
        'cnn_rf' : RandomForestClassifier(n_estimators=n_rf, random_state=seed, n_jobs=-1),
        'cnn_knn': KNeighborsClassifier(n_neighbors=k_knn),
        'cnn_xgb': XGBClassifier(n_estimators=200, learning_rate=0.1,
                                  max_depth=4, use_label_encoder=False,
                                  eval_metric='mlogloss', random_state=seed,
                                  verbosity=0, tree_method='hist', device='cuda'),
    }
    
    fold_accs = {}
    fold_preds = {}
    for name, clf in clfs.items():
        clf.fit(Ftr, y_data[tr])
        preds = clf.predict(Fte)
        acc   = round(accuracy_score(y_data[te], preds), 3)
        fold_accs[name] = acc
        fold_preds[name] = preds
        
    return fold_accs, fold_preds, y_data[te]

def cv_shared_hybrid(cnn_act='relu', k_knn=5, n_rf=200,
                     filters_1=32, filters_2=64, kernel_size=3,
                     dropout=0.3, dense_units=64, lr=1e-3):

    print(f'\n[Shared CNN Extractor  cnn_act={cnn_act}]')
    print(f'  SVM(RBF) + RF(n={n_rf}) + kNN(k={k_knn}) from same features')
    _t0 = time.time()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    res = {k: {'accs':[],'true':[],'pred':[]} for k in ['cnn_svm','cnn_rf','cnn_knn','cnn_xgb']}
    
    futures = []
    for fold, (tr, te) in enumerate(skf.split(X_raw, y)):
        futures.append(_cv_shared_fold_remote.remote(
            fold, tr, te, X_raw_id, y_id, cnn_act, k_knn, n_rf, 
            filters_1, filters_2, kernel_size, dropout, dense_units, lr, SEED, BATCH_SIZE
        ))
        
    results = ray.get(futures)
    for fold, fold_res in enumerate(results):
        fold_accs, fold_preds, y_te = fold_res
        print(f'  Fold {fold+1:2d}: '
              f'SVM={fold_accs["cnn_svm"]:.3f}  '
              f'RF={fold_accs["cnn_rf"]:.3f}  '
              f'kNN={fold_accs["cnn_knn"]:.3f}  '
              f'XGB={fold_accs["cnn_xgb"]:.3f}')
        for name in res:
            res[name]['accs'].append(fold_accs[name])
            res[name]['true'].extend(y_te)
            res[name]['pred'].extend(fold_preds[name])
            
    _elapsed = time.time() - _t0
    TIMING['Shared_CNN_hybrid'] = _elapsed
    print(f'  -- Total hybrid time: {_elapsed:.1f}s ({_elapsed/60:.1f} min)')
    log(f'Shared_CNN_hybrid: time={_elapsed:.1f}s')
    for name in res:
        res[name]['true'] = np.array(res[name]['true'])
        res[name]['pred'] = np.array(res[name]['pred'])
        avg = np.mean(res[name]['accs'])
        print(f'  -- {name} Average: {avg:.3f}')
        log(f'  {name}: avg={avg:.3f}')
    return res

@ray.remote(num_gpus=1)
def _cv_xgboost_fold_remote(fold, tr, te, X_raw_data, y_data, seed):
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier
    
    np.random.seed(seed + fold)
    sc  = MinMaxScaler()
    Xtr = sc.fit_transform(X_raw_data[tr])
    Xte = sc.transform(X_raw_data[te])
    clf = XGBClassifier(
        n_estimators=200, learning_rate=0.1,
        max_depth=4, eval_metric='mlogloss',
        random_state=seed, verbosity=0,
        tree_method='hist', device='cuda'
    )
    clf.fit(Xtr, y_data[tr])
    preds = clf.predict(Xte)
    acc   = round(accuracy_score(y_data[te], preds), 3)
    return acc, (y_data[te] if True else None), (preds if True else None)

def cv_xgboost(collect_preds=False):
    print('\n[XGBoost - standalone]')
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    accs, all_true, all_pred = [], [], []
    
    futures = []
    for fold, (tr, te) in enumerate(skf.split(X_raw, y)):
        futures.append(_cv_xgboost_fold_remote.remote(fold, tr, te, X_raw_id, y_id, SEED))
        
    results = ray.get(futures)
    for fold, res in enumerate(results):
        acc, t_labels, p_labels = res
        accs.append(acc)
        if collect_preds:
            all_true.extend(t_labels)
            all_pred.extend(p_labels)
        print(f'  Fold {fold+1:2d}: {acc:.3f}')
        
    print(f'  -- Average: {np.mean(accs):.3f}')
    if collect_preds:
        return accs, np.array(all_true), np.array(all_pred)
    return accs

def make_table(fold_results, title):
    df = pd.DataFrame({'Data set': list(range(1, N_SPLITS + 1))})
    for col, vals in fold_results.items(): df[col] = vals
    avg = {'Data set': 'Average accuracy'}
    for col, vals in fold_results.items(): avg[col] = round(np.mean(vals), 3)
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    print(f'\n{"="*65}\n{title}\n{"="*65}')
    print(df.to_string(index=False))
    return df


print(f'Ready. N_FEATURES={N_FEATURES}  N_CLASSES={N_CLASSES}')
build_cnn().summary()


# In[15]:


# ============================================================
# CELL — CNN: ALL ACTIVATIONS + SOFTMAX VARIANTS
# ============================================================
R = {}
H = {}

print('=== Tab.3  CNN ACTIVATION COMPARISON ===')
R['cnn_relu'],    H['relu']    = cv_cnn_softmax('relu',    'relu',    label='CNN reLU',    collect_history=True)
R['cnn_sigmoid'], H['sigmoid'] = cv_cnn_softmax('sigmoid', 'sigmoid', label='CNN Sigmoid', collect_history=True)
R['cnn_tanh'],    H['tanh']    = cv_cnn_softmax('tanh',    'tanh',    label='CNN Tanh',    collect_history=True)

print('\n=== Tab.5  CNN-SOFTMAX FC LAYER ===')
R['cnn_relu_linear'] = cv_cnn_softmax('relu', 'linear',
                                       label='CNN-Softmax reLU+Linear',
                                       collect_preds=True)
R['cnn_relu_rbf']    = cv_cnn_softmax('relu', 'rbf',
                                       label='CNN-Softmax reLU+RBF')

def _accs(k): return R[k][0] if isinstance(R[k], tuple) else R[k]
print('\n=== CNN BLOCK COMPLETE ===')
for k in R:
    print(f'  {k:<22} avg={np.mean(_accs(k)):.3f}')


# In[ ]:


# ============================================================
# CELL — OTHER CLASSIFIERS
# ============================================================
from xgboost import XGBClassifier
print('=== BASELINES ===')
R['svm']     = cv_svm(collect_preds=True)
R['xgboost'] = cv_xgboost(collect_preds=True)

print('\n=== SHARED CNN EXTRACTOR ===')
hybrid = cv_shared_hybrid(cnn_act='relu', k_knn=5, n_rf=200)
R['cnn_svm'] = (hybrid['cnn_svm']['accs'], hybrid['cnn_svm']['true'], hybrid['cnn_svm']['pred'])
R['cnn_rf']  = (hybrid['cnn_rf']['accs'],  hybrid['cnn_rf']['true'],  hybrid['cnn_rf']['pred'])
R['cnn_knn'] = (hybrid['cnn_knn']['accs'], hybrid['cnn_knn']['true'], hybrid['cnn_knn']['pred'])
R['cnn_xgb'] = (hybrid['cnn_xgb']['accs'], hybrid['cnn_xgb']['true'], hybrid['cnn_xgb']['pred'])

def _accs(k): return R[k][0] if isinstance(R[k], tuple) else R[k]
print('\n=== CLASSIFIER BLOCK COMPLETE ===')
for k in ['svm','xgboost','cnn_svm','cnn_rf','cnn_knn','cnn_xgb']:
    print(f'  {k:<22} avg={np.mean(_accs(k)):.3f}')


# In[ ]:


# ============================================================
# CELL 14 — TABLE 3: CNN Activation Comparison
# ============================================================
def _accs(k): return R[k][0] if isinstance(R[k], tuple) else R[k]

tab3 = make_table(
    {'Accuracy (reLU)'   : _accs('cnn_relu'),
     'Accuracy (Sigmoid)': _accs('cnn_sigmoid'),
     'Accuracy (Tanh)'   : _accs('cnn_tanh')},
    title='Tab. 3  CNN accuracy: reLU vs Sigmoid vs Tanh'
)


# In[ ]:


# ============================================================
# CELL 15 — TABLE 5: CNN-Softmax FC Layer
# ============================================================
def _accs(k): return R[k][0] if isinstance(R[k], tuple) else R[k]

tab5 = make_table(
    {'Accuracy (Linear)': _accs('cnn_relu_linear'),
     'Accuracy (RBF)'   : _accs('cnn_relu_rbf')},
    title='Tab. 5  CNN-Softmax accuracy when CNN activation = reLU'
)


# In[ ]:


# ============================================================
# CELL 16 — TABLE 8: Full Model Comparison
# ============================================================
def _accs(k): return R[k][0] if isinstance(R[k], tuple) else R[k]

tab8 = make_table(
    {'CNN'        : _accs('cnn_relu'),
     'CNN-Softmax': _accs('cnn_relu_linear'),
     'SVM'        : _accs('svm'),
     'XGBoost'    : _accs('xgboost'),
     'CNN-SVM'    : _accs('cnn_svm'),
     'CNN-RF'     : _accs('cnn_rf'),
     'CNN-kNN'    : _accs('cnn_knn'),
     'CNN-XGBoost': _accs('cnn_xgb')},
    title='Tab. 8  Accuracy of user driving behaviour judgement'
)


# In[ ]:


# ============================================================
# CELL 17 — ANALYSIS
# ============================================================
GRADE_LABELS = list(le.classes_)
def _accs(k): return R[k][0] if isinstance(R[k], tuple) else R[k]
def _preds(k): return R[k][1], R[k][2]

cm_models = [
    ('SVM',                     'svm'),
    ('XGBoost',                 'xgboost'),
    ('CNN-SVM',                 'cnn_svm'),
    ('CNN-RF',                  'cnn_rf'),
    ('CNN-kNN',                 'cnn_knn'),
    ('CNN-XGBoost',             'cnn_xgb'),
    ('CNN-Softmax\n(reLU+Lin)', 'cnn_relu_linear'),
]
fig, axes = plt.subplots(2, 4, figsize=(28, 12))
fig.suptitle('Confusion Matrices — 10-Fold Aggregated Predictions',
             fontsize=13, fontweight='bold')
for ax, (name, key) in zip(axes.flatten(), cm_models):
    y_t, y_p = _preds(key)
    cm_n = confusion_matrix(y_t, y_p).astype(float)
    cm_n = cm_n / cm_n.sum(axis=1, keepdims=True)
    sns.heatmap(cm_n, annot=True, fmt='.2f',
                xticklabels=GRADE_LABELS, yticklabels=GRADE_LABELS,
                cmap='Blues', ax=ax, vmin=0, vmax=1, linewidths=0.5)
    ax.set_title(f'{name}\nAcc={accuracy_score(y_t,y_p):.3f}',
                 fontweight='bold', fontsize=10)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

f1_records = []
for name, key in cm_models:
    y_t, y_p = _preds(key)
    f1_per = f1_score(y_t, y_p, average=None, labels=list(range(len(GRADE_LABELS))))
    row = {'Model': name.replace('\n',' ')}
    for i, g in enumerate(GRADE_LABELS): row[f'F1-{g}'] = round(f1_per[i], 3)
    row['Macro F1']    = round(f1_score(y_t, y_p, average='macro'), 3)
    row['Weighted F1'] = round(f1_score(y_t, y_p, average='weighted'), 3)
    f1_records.append(row)
    print(f'\n{name.replace(chr(10)," ")}')
    print(classification_report(y_t, y_p, target_names=GRADE_LABELS, digits=3))

f1_df = pd.DataFrame(f1_records)
f1_df.to_csv(f'{OUTPUT_DIR}/f1_scores_per_grade.csv', index=False)
fig, ax = plt.subplots(figsize=(10, 4))
f1_cols = [f'F1-{g}' for g in GRADE_LABELS]
sns.heatmap(f1_df.set_index('Model')[f1_cols],
            annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
            linewidths=0.5, ax=ax)
ax.set_title('Per-Grade F1 Score by Model', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('f1_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

act_data = {
    'CNN reLU'       : (np.mean(_accs('cnn_relu')),        '#2ecc71'),
    'CNN Sigmoid'    : (np.mean(_accs('cnn_sigmoid')),     '#e74c3c'),
    'CNN Tanh'       : (np.mean(_accs('cnn_tanh')),        '#3498db'),
    'SM reLU+Linear' : (np.mean(_accs('cnn_relu_linear')), '#27ae60'),
    'SM reLU+RBF'    : (np.mean(_accs('cnn_relu_rbf')),    '#e67e22'),
}
labels = list(act_data.keys())
vals   = [act_data[k][0] for k in labels]
cols   = [act_data[k][1] for k in labels]
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(labels, vals, color=cols, edgecolor='white')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
            f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
baseline = round(1/N_CLASSES, 3)
ax.axhline(baseline, color='gray', linestyle='--', linewidth=1, alpha=0.6, label=f'Random baseline (1/{N_CLASSES}={baseline})')
ax.set_ylim(0, 0.8)
ax.set_title('Activation Function Comparison', fontweight='bold', fontsize=12)
ax.set_ylabel('Avg 10-Fold Accuracy')
ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('activation_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: confusion_matrices.png, f1_heatmap.png, activation_comparison.png')

def avg_history(fold_curves):
    max_len = max(len(c) for c in fold_curves)
    padded  = [c + [c[-1]] * (max_len - len(c)) for c in fold_curves]
    return np.mean(padded, axis=0)

if H:
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, color in [('relu','CNN reLU','#2ecc71'),
                               ('sigmoid','CNN Sigmoid','#e74c3c'),
                               ('tanh','CNN Tanh','#3498db')]:
        if key in H:
            avg = avg_history(H[key])
            ax.plot(range(1, len(avg)+1), avg,
                    label=label, color=color, linewidth=2)
    bl = round(1/N_CLASSES, 3)
    ax.axhline(bl, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label=f'Random baseline (1/{N_CLASSES}={bl})')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Avg Val Accuracy (10 folds)', fontsize=11)
    ax.set_title('CNN Activation Functions — Val Accuracy vs Epoch',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves_activation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: training_curves_activation.png')
else:
    print('H dict empty — run CNN block (Cell 13) first')


# In[ ]:


# ============================================================
# CELL 18 — SUMMARY DASHBOARD
# ============================================================
def _accs(k): return R[k][0] if isinstance(R[k], tuple) else R[k]

summary = pd.DataFrame({
    'Model': [
        'CNN (reLU)','CNN (Sigmoid)','CNN (Tanh)',
        'CNN-Softmax reLU+Linear','CNN-Softmax reLU+RBF',
        'SVM','XGBoost','CNN-SVM','CNN-RF','CNN-kNN','CNN-XGBoost'
    ],
    'Avg Acc': [
        round(np.mean(_accs('cnn_relu')),        3),
        round(np.mean(_accs('cnn_sigmoid')),     3),
        round(np.mean(_accs('cnn_tanh')),        3),
        round(np.mean(_accs('cnn_relu_linear')), 3),
        round(np.mean(_accs('cnn_relu_rbf')),    3),
        round(np.mean(_accs('svm')),             3),
        round(np.mean(_accs('xgboost')),         3),
        round(np.mean(_accs('cnn_svm')),         3),
        round(np.mean(_accs('cnn_rf')),          3),
        round(np.mean(_accs('cnn_knn')),         3),
        round(np.mean(_accs('cnn_xgb')),         3),
    ]
}).sort_values('Avg Acc', ascending=False).reset_index(drop=True)

print(f'====  FINAL SUMMARY  ====')
print(f'Features: {N_FEATURES}  |  Drivers: {len(df_model)}  |  Folds: {N_SPLITS}')
print(summary.to_string(index=False))
best = summary.iloc[0]
print(f"\nBest: {best['Model']}  ->  {best['Avg Acc']:.3f}")

colors = ['#e74c3c' if r['Model']==best['Model'] else '#3498db'
          for _,r in summary.iterrows()]
fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(summary['Model'], summary['Avg Acc'],
              color=colors, edgecolor='white', linewidth=0.8)
for bar, acc in zip(bars, summary['Avg Acc']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{acc:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
baseline = round(1/N_CLASSES, 3)
ax.axhline(baseline, color='gray', linestyle=':', linewidth=1, alpha=0.7, label=f'Random baseline (1/{N_CLASSES}={baseline})')
ax.axhline(summary['Avg Acc'].max(), color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Average 10-Fold Accuracy')
ax.set_title(f'UBI Driver Classification | {N_FEATURES} features, {len(df_model)} drivers',
             fontsize=12, fontweight='bold')
ax.set_xticklabels(summary['Model'], rotation=30, ha='right', fontsize=9)
ax.legend(handles=[
    mpatches.Patch(color='#e74c3c', label='Best model'),
    mpatches.Patch(color='#3498db', label='Other models'),
    mpatches.Patch(color='gray',    label=f'Random baseline (1/{N_CLASSES}={baseline})'),
], fontsize=9)
plt.tight_layout()
plt.savefig('model_comparison_final.png', dpi=150, bbox_inches='tight')
plt.show()

tab3.to_csv(f'{OUTPUT_DIR}/table3_activation_comparison.csv', index=False)
tab5.to_csv(f'{OUTPUT_DIR}/table5_cnnsoftmax_relu.csv',        index=False)
tab8.to_csv(f'{OUTPUT_DIR}/table8_full_comparison.csv',        index=False)
summary.to_csv(f'{OUTPUT_DIR}/summary_final.csv',              index=False)
print('All outputs saved.')
print('\n====  TIMING SUMMARY  ====')
timing_df = pd.DataFrame([
    {'Model': k, 'Time (s)': round(v, 1),
     'Time (min)': round(v/60, 2)}
    for k, v in TIMING.items()
]).sort_values('Time (s)', ascending=False)
print(timing_df.to_string(index=False))
timing_df.to_csv(f'{OUTPUT_DIR}/timing_summary.csv', index=False)
log('Timing summary saved to timing_summary.csv')


# In[ ]:


# ============================================================
# PSO — HYPERPARAMETER SEARCH (3-fold CV for speed)
#
# PSO_PATIENCE = 100  →  early stop after 100 iters without improvement
# Full 300 iterations will always run.
# Convergence point is tracked in log for thesis reporting.
# ============================================================
import random

PSO_CKPT_PATH = f"{CHECKPOINT_DIR}/pso_checkpoint.pkl"
PSO_LOG_CSV   = f"{OUTPUT_DIR}/pso_log.csv"
FORCE_FRESH_PSO = False

def save_pso_checkpoint(state):
    with open(PSO_CKPT_PATH, "wb") as f:
        pickle.dump(state, f)

def load_pso_checkpoint():
    if (not FORCE_FRESH_PSO) and os.path.exists(PSO_CKPT_PATH):
        with open(PSO_CKPT_PATH, "rb") as f:
            return pickle.load(f)
    return None

PSO_SEARCH_FOLDS = 3
PSO_PARTICLES    = 30
PSO_ITERATIONS   = 300
PSO_W  = 0.7
PSO_C1 = 1.5
PSO_C2 = 1.5
PSO_PATIENCE = 100  # early stop after 100 iterations without improvement

PSO_SPACE = [
    [16, 32, 64, 128],          # filters_1
    [64, 128, 256],             # filters_2
    [3, 5],                     # kernel_size
    [0.1, 0.2, 0.3, 0.5],      # dropout
    [64, 128, 256],             # dense_units
    [1e-4, 5e-4, 1e-3, 5e-3],  # lr
]
PARAM_NAMES = ['filters_1','filters_2','kernel_size','dropout','dense_units','lr']
N_DIMS = len(PSO_SPACE)


@ray.remote(num_gpus=1)
def pso_objective_remote(params, X_raw_data, y_data):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold
    import os
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    ES_local = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=80,
        restore_best_weights=True, verbose=0)
    LR_REDUCE_local = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5,
        patience=30, min_lr=1e-5, verbose=0)
        
    f1, f2, ks, dr, du, lr = params
    skf = StratifiedKFold(n_splits=PSO_SEARCH_FOLDS, shuffle=True, random_state=SEED)
    accs = []
    for fold, (tr, te) in enumerate(skf.split(X_raw_data, y_data)):
        np.random.seed(SEED + fold)
        tf.random.set_seed(SEED + fold)
        sc = MinMaxScaler()
        Xtr_s = sc.fit_transform(X_raw_data[tr]).astype(np.float32)
        Xte_s = sc.transform(X_raw_data[te]).astype(np.float32)
        Xtr_aug, ytr_aug = augment(Xtr_s, y_data[tr])
        cnn = build_cnn(cnn_act='relu', fc_act='relu',
                        filters_1=int(f1), filters_2=int(f2),
                        kernel_size=int(ks), dropout=dr,
                        dense_units=int(du), lr=lr)
        cnn.fit(Xtr_aug.reshape(-1, N_FEATURES, 1), ytr_aug,
                validation_split=0.10, epochs=300,
                batch_size=BATCH_SIZE, callbacks=[ES_local, LR_REDUCE_local], verbose=0)
        preds = np.argmax(cnn.predict(
            Xte_s.reshape(-1, N_FEATURES, 1), verbose=0), axis=1)
        accs.append(accuracy_score(y_data[te], preds))
        keras.backend.clear_session()
    return np.mean(accs)


def snap(position):
    return [PSO_SPACE[d][int(np.clip(round(position[d]), 0, len(PSO_SPACE[d])-1))]
            for d in range(N_DIMS)]


random.seed(SEED); np.random.seed(SEED)
ckpt = load_pso_checkpoint()

if ckpt is not None:
    positions    = ckpt['positions']
    velocities   = ckpt['velocities']
    pbest_pos    = ckpt['pbest_pos']
    pbest_val    = ckpt['pbest_val']
    gbest_pos    = ckpt['gbest_pos']
    gbest_val    = ckpt['gbest_val']
    pso_log      = ckpt.get('pso_log', [])
    start_iter   = ckpt['next_iter']
    _no_improve  = ckpt.get('no_improve', 0)
    _prev_best   = ckpt.get('prev_best', gbest_val)
    _pso_start   = time.time() - ckpt.get('elapsed_seconds', 0)
    random.setstate(ckpt['random_state'])
    np.random.set_state(ckpt['numpy_random_state'])
    print(f'Resuming PSO from checkpoint: iteration {start_iter+1}/{PSO_ITERATIONS}, best={gbest_val:.3f}')
else:
    positions  = [[random.randint(0, len(PSO_SPACE[d])-1) for d in range(N_DIMS)]
                   for _ in range(PSO_PARTICLES)]
    velocities = [[random.uniform(-1, 1) for _ in range(N_DIMS)]
                   for _ in range(PSO_PARTICLES)]
    pbest_pos  = [p[:] for p in positions]
    pbest_val  = [-np.inf] * PSO_PARTICLES
    gbest_pos  = None
    gbest_val  = -np.inf
    pso_log    = []
    start_iter = 0
    _pso_start = time.time()
    _no_improve = 0
    _prev_best  = -np.inf
    print('Starting fresh PSO search')

print(f'PSO search: {PSO_PARTICLES} particles × {PSO_ITERATIONS} iterations')
print(f'Early stopping: patience={PSO_PATIENCE} iterations')
print(f'Each evaluation: {PSO_SEARCH_FOLDS}-fold CV')
print(f'Total planned evaluations: {PSO_PARTICLES * PSO_ITERATIONS}')
print(f'Checkpoint path: {PSO_CKPT_PATH}')
print()

# ray.init and object references moved to imports section

pbar = tqdm(range(start_iter, PSO_ITERATIONS), desc="PSO Search Iterations")
for it in pbar:
    futures = []
    for p in range(PSO_PARTICLES):
        params = snap(positions[p])
        futures.append(pso_objective_remote.remote(params, X_raw_id, y_id))
        
    _it0 = time.time()
    results = ray.get(futures)
    _it_time = time.time() - _it0
    
    for p, val in enumerate(results):
        params = snap(positions[p])
        if val > pbest_val[p]:
            pbest_val[p] = val
            pbest_pos[p] = positions[p][:]
        if val > gbest_val:
            gbest_val = val
            gbest_pos = positions[p][:]
            
    pso_log.append({'iter': it+1, 'gbest': gbest_val,
                    'params': snap(gbest_pos)})
    msg2 = f'  >> Iter {it+1} best: {gbest_val:.3f} — {dict(zip(PARAM_NAMES, snap(gbest_pos)))} (Time: {_it_time:.1f}s)'
    logging.info(msg2)
    pbar.set_postfix({'Best Acc': f"{gbest_val:.3f}"})

    pd.DataFrame(pso_log).to_csv(PSO_LOG_CSV, index=False)

    # Early stopping check
    if gbest_val > _prev_best + 0.001:
        _prev_best  = gbest_val
        _no_improve = 0
    else:
        _no_improve += 1
        print(f'  (no improvement: {_no_improve}/{PSO_PATIENCE})')

    save_pso_checkpoint({
        'next_iter': it + 1,
        'positions': positions,
        'velocities': velocities,
        'pbest_pos': pbest_pos,
        'pbest_val': pbest_val,
        'gbest_pos': gbest_pos,
        'gbest_val': gbest_val,
        'pso_log': pso_log,
        'no_improve': _no_improve,
        'prev_best': _prev_best,
        'elapsed_seconds': time.time() - _pso_start,
        'random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
    })
    print(f'  checkpoint saved -> {PSO_CKPT_PATH}')

    if _no_improve >= PSO_PATIENCE:
        print(f'\nPSO early stop at iteration {it+1} — no improvement for {PSO_PATIENCE} iterations')
        logging.info(f'PSO early stop at iteration {it+1}')
        break

    # Update velocities and positions
    for p in range(PSO_PARTICLES):
        for d in range(N_DIMS):
            r1, r2 = random.random(), random.random()
            velocities[p][d] = (PSO_W * velocities[p][d]
                + PSO_C1 * r1 * (pbest_pos[p][d] - positions[p][d])
                + PSO_C2 * r2 * (gbest_pos[d]    - positions[p][d]))
            positions[p][d] = np.clip(
                positions[p][d] + velocities[p][d],
                0, len(PSO_SPACE[d]) - 1
            )

PSO_BEST = dict(zip(PARAM_NAMES, snap(gbest_pos)))
_pso_total = time.time() - _pso_start
TIMING['PSO_search'] = _pso_total
print(f'\n=== PSO COMPLETE ===')
print(f'Best 3-fold accuracy : {gbest_val:.3f}')
print(f'Best hyperparameters : {PSO_BEST}')
print(f'Total PSO time       : {_pso_total:.1f}s ({_pso_total/3600:.2f} hours)')
log(f'PSO complete: best={gbest_val:.3f} time={_pso_total:.1f}s params={PSO_BEST}')
with open(f'{OUTPUT_DIR}/pso_best.json', 'w') as f:
    json.dump({'best_accuracy': float(gbest_val), 'best_params': PSO_BEST,
               'time_seconds': float(_pso_total)}, f, indent=2)
print(f'Saved PSO best -> {OUTPUT_DIR}/pso_best.json')


# In[ ]:


# ============================================================
# PSO-TUNED FINAL EVALUATION (10-fold)
# ============================================================
print('PSO best config:', PSO_BEST)
print('Running 10-fold final evaluation...')

R_OPT = {}

R_OPT['svm']     = R.get('svm',     cv_svm(collect_preds=True))
R_OPT['xgboost'] = R.get('xgboost', cv_xgboost(collect_preds=True))

print('\n=== PSO-TUNED SHARED CNN EXTRACTOR (10-fold) ===')
hybrid_opt = cv_shared_hybrid(
    cnn_act     = 'relu',
    k_knn       = 5,
    n_rf        = 200,
    filters_1   = PSO_BEST['filters_1'],
    filters_2   = PSO_BEST['filters_2'],
    kernel_size = PSO_BEST['kernel_size'],
    dropout     = PSO_BEST['dropout'],
    dense_units = PSO_BEST['dense_units'],
    lr          = PSO_BEST['lr'],
)
for k in ['cnn_svm','cnn_rf','cnn_knn','cnn_xgb']:
    R_OPT[k] = (hybrid_opt[k]['accs'],
                hybrid_opt[k]['true'],
                hybrid_opt[k]['pred'])

def _accs(k): return R_OPT[k][0] if isinstance(R_OPT[k], tuple) else R_OPT[k]

print('\n=== PSO-TUNED RESULTS ===')
for k in ['svm','xgboost','cnn_svm','cnn_rf','cnn_knn','cnn_xgb']:
    base = np.mean((R[k][0] if isinstance(R[k], tuple) else R[k]))
    opt  = np.mean(_accs(k))
    print(f'  {k:<14} baseline={base:.3f}  PSO-tuned={opt:.3f}  delta={opt-base:+.3f}')

opt_table = make_table(
    {
        'SVM'        : _accs('svm'),
        'XGBoost'    : _accs('xgboost'),
        'CNN-SVM'    : _accs('cnn_svm'),
        'CNN-RF'     : _accs('cnn_rf'),
        'CNN-kNN'    : _accs('cnn_knn'),
        'CNN-XGBoost': _accs('cnn_xgb'),
    },
    title='Tab. X  PSO-tuned results (10-fold CV, 3-class)'
)
