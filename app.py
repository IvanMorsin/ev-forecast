import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
#import torch
#import torch.nn as nn

# Константы
house_meta = {
    'house_1': {'n_flats': 383, 'n_floors': 12, 'r_ud': 1.2777},
    'house_2': {'n_flats': 191, 'n_floors': 12, 'r_ud': 1.3726},
    'house_3': {'n_flats': 124, 'n_floors': 12, 'r_ud': 1.4664},
    'house_4': {'n_flats': 263, 'n_floors': 12, 'r_ud': 1.3317},
    'house_5': {'n_flats': 127, 'n_floors': 7, 'r_ud': 1.4622},
    'house_6': {'n_flats': 497, 'n_floors': 25, 'r_ud': 1.2506},
    'house_7': {'n_flats': 471, 'n_floors': 17, 'r_ud': 1.2558},
    'house_8': {'n_flats': 171, 'n_floors': 23, 'r_ud': 1.4006},
}

r_ud_default = np.mean([v['r_ud'] for v in house_meta.values()])

horizons = {
    '4h': 8,
    '8h': 16,
    '24h': 48,
    '7d': 336,
    '14d': 672,
    '1m': 1488,
}

feature_cols = [
    'hour', 'minute', 'weekday', 'month', 'day_of_year',
    'is_weekend', 'is_holiday',
    'lag_1', 'lag_2', 'lag_48', 'lag_96', 'lag_336',
    'rolling_mean_48', 'rolling_mean_336',
    'temp_c', 'humidity', 'cloudiness',
    'n_flats', 'n_floors',
]

cities = {
    'Москва': {'lat': 55.7522, 'lon': 37.6156},
    'Балашиха': {'lat': 55.7964, 'lon': 37.9381},
    'Химки': {'lat': 55.8890, 'lon': 37.4289},
    'Подольск': {'lat': 55.4316, 'lon': 37.5447},
    'Королёв': {'lat': 55.9162, 'lon': 37.8522},
    'Мытищи': {'lat': 55.9116, 'lon': 37.7328},
    'Люберцы': {'lat': 55.6783, 'lon': 37.8932},
    'Электросталь': {'lat': 55.7897, 'lon': 38.4447},
    'Коломна': {'lat': 55.0817, 'lon': 38.7792},
    'Одинцово': {'lat': 55.6800, 'lon': 37.2783},
    'Серпухов': {'lat': 54.9167, 'lon': 37.4167},
    'Щёлково': {'lat': 55.9283, 'lon': 38.0083},
    'Домодедово': {'lat': 55.4333, 'lon': 37.7667},
    'Жуковский': {'lat': 55.6000, 'lon': 38.1167},
    'Реутов': {'lat': 55.7606, 'lon': 37.8631},
    'Санкт-Петербург': {'lat': 59.9311, 'lon': 30.3609},
    'Нижний Новгород': {'lat': 56.3269, 'lon': 44.0072},
    'Казань': {'lat': 55.7887, 'lon': 49.1221},
    'Екатеринбург': {'lat': 56.8519, 'lon': 60.6122},
    'Новосибирск': {'lat': 54.9833, 'lon': 82.8964},
    'Челябинск': {'lat': 55.1547, 'lon': 61.4286},
    'Самара': {'lat': 53.2028, 'lon': 50.1408},
    'Уфа': {'lat': 54.7388, 'lon': 55.9721},
    'Пермь': {'lat': 58.0105, 'lon': 56.2502},
    'Воронеж': {'lat': 51.6717, 'lon': 39.2106},
    'Владимир': {'lat': 56.1289, 'lon': 40.4067},
    'Тула': {'lat': 54.1961, 'lon': 37.6182},
    'Ярославль': {'lat': 57.6261, 'lon': 39.8845},
    'Иваново': {'lat': 57.0000, 'lon': 40.9833},
    'Рязань': {'lat': 54.6269, 'lon': 39.6916},
    'Тверь': {'lat': 56.8587, 'lon': 35.9176},
    'Калуга': {'lat': 54.5293, 'lon': 36.2754},
    'Брянск': {'lat': 53.2434, 'lon': 34.3636},
    'Орёл': {'lat': 52.9651, 'lon': 36.0785},
    'Курск': {'lat': 51.7304, 'lon': 36.1927},
    'Липецк': {'lat': 52.6031, 'lon': 39.5708},
    'Белгород': {'lat': 50.5958, 'lon': 36.5873},
    'Смоленск': {'lat': 54.7818, 'lon': 32.0401},
    'Тамбов': {'lat': 52.7212, 'lon': 41.4523},
    'Кострома': {'lat': 57.7667, 'lon': 40.9333},
}

# Модели



@st.cache_resource
def load_point_models():
    models = {}
    for hz in horizons:
        with open(f'models/lgbm_{hz}.pkl', 'rb') as f:
            models[hz] = joblib.load(f)
    with open('models/model_meta.pkl', 'rb') as f:
        meta = joblib.load(f)
    return models, meta


@st.cache_resource
def load_quantile_models():
    quantiles = [10, 50, 90, 95]
    models = {}
    for hz in horizons:
        models[hz] = {}
        for q in quantiles:
            with open(f'models_ev/lgbm_{hz}_q{q}.pkl', 'rb') as f:
                models[hz][q] = joblib.load(f)
    with open('models_ev/conformal_thresholds.pkl', 'rb') as f:
        thresholds = joblib.load(f)
    with open('models_ev/model_meta.pkl', 'rb') as f:
        meta = joblib.load(f)
    return models, thresholds, meta

@st.cache_resource
def load_autoencoder():
    import torch
    import torch.nn as nn

    class Autoencoder(nn.Module):
        def __init__(self, input_size, hidden_size=32, latent_size=8):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, latent_size),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_size),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = Autoencoder(input_size=11)
    model.load_state_dict(
        torch.load('models/autoencoder_all_houses.pth', map_location='cpu')
    )
    model.eval()
    with open('models/autoencoder_scaler_all_houses.pkl', 'rb') as f:
        scaler = joblib.load(f)
    return model, scaler

# вспомогательные функции
def get_holidays(timestamps):
    try:
        from workalendar.europe import Russia
        cal = Russia()
        years = timestamps.dt.year.unique()
        holiday_dates = set()
        for y in years:
            for date, _ in cal.holidays(y):
                holiday_dates.add(date)
        return timestamps.dt.date.apply(lambda d: int(d in holiday_dates))
    except ImportError:
        return pd.Series(0, index=timestamps.index)


def get_weather_forecast(lat, lon):
    url = 'https://api.open-meteo.com/v1/forecast'
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m,relativehumidity_2m,cloudcover',
        'forecast_days': 16,
        'timezone': 'Europe/Moscow',
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        df_weather = pd.DataFrame({
            'timestamp': pd.to_datetime(data['hourly']['time']),
            'temp_c': data['hourly']['temperature_2m'],
            'humidity': data['hourly']['relativehumidity_2m'],
            'cloudiness': data['hourly']['cloudcover'],
        })
        df_weather = (
            df_weather
            .set_index('timestamp')
            .resample('30min')
            .interpolate('linear')
            .reset_index()
        )
        return df_weather
    except Exception:
        n = 16 * 48
        now = pd.Timestamp.now().floor('30min')
        return pd.DataFrame({
            'timestamp': pd.date_range(now, periods=n, freq='30min'),
            'temp_c': [10.0] * n,
            'humidity': [70.0] * n,
            'cloudiness': [50.0] * n,
        })


def validate_csv(df):
    required = {'timestamp', 'power'}
    missing = required - set(df.columns)
    if missing:
        return False, f'Отсутствуют колонки: {missing}'
    try:
        pd.to_datetime(df['timestamp'])
    except Exception:
        return False, 'Не удалось распознать timestamp как дату.'
    if len(df) < 336:
        return False, f'Мало данных: {len(df)} строк, нужно минимум 336.'
    if df['power'].isnull().mean() > 0.1:
        return False, 'Более 10% пропусков в колонке power.'
    return True, 'ok'


def make_features(df_house, n_flats, n_floors, df_weather=None):
    data = df_house[['timestamp', 'power']].copy()
    data = data.sort_values('timestamp').reset_index(drop=True)

    data['hour'] = data['timestamp'].dt.hour
    data['minute'] = data['timestamp'].dt.minute
    data['weekday'] = data['timestamp'].dt.weekday
    data['month'] = data['timestamp'].dt.month
    data['day_of_year'] = data['timestamp'].dt.dayofyear
    data['is_weekend'] = (data['weekday'] >= 5).astype(int)
    data['is_holiday'] = get_holidays(data['timestamp'])

    for lag in [1, 2, 48, 96, 336]:
        data[f'lag_{lag}'] = data['power'].shift(lag)
    data['rolling_mean_48'] = data['power'].shift(1).rolling(48).mean()
    data['rolling_mean_336'] = data['power'].shift(1).rolling(336).mean()

    if df_weather is not None:
        df_weather = df_weather.copy()
        df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = pd.merge_asof(
            data.sort_values('timestamp'),
            df_weather[['timestamp', 'temp_c', 'humidity', 'cloudiness']].sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta('1h'),
        )
        for col in ['temp_c', 'humidity', 'cloudiness']:
            data[col] = data[col].ffill().fillna(0)
    else:
        data['temp_c'] = 10.0
        data['humidity'] = 70.0
        data['cloudiness'] = 50.0

    data['n_flats'] = n_flats
    data['n_floors'] = n_floors
    st.write(f'DEBUG before dropna: {len(data)}')
    st.write(f'DEBUG NaN counts: {data[feature_cols].isnull().sum().to_dict()}')
    data = data.dropna(subset=[c for c in feature_cols
                               if c not in ['temp_c', 'humidity', 'cloudiness']]).reset_index(drop=True)
    for col in ['temp_c', 'humidity', 'cloudiness']:
        data[col] = data[col].fillna(0)
    st.write(f'DEBUG after dropna: {len(data)}')
    return data

# конфиг страницы
st.set_page_config(
    page_title='EV Forecast',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded',
)

# sidebar
with st.sidebar:
    st.title('Forecasting electrical load multy-appartment buildings')
    st.caption('Прогнозирование нагрузки МКД и доступной мощности для ЭЗС')
    st.divider()

    st.subheader('Данные дома')
    uploaded_file = st.file_uploader(
        'Загрузите данные по потребляемой мощности МКД в формате "csv"',
        type=['csv'],
        help='Колонки: timestamp, power. Минимум 336 строк.',
    )

    st.subheader('Параметры МКД (на одно ВРУ)')
    n_flats = st.number_input('Число квартир', min_value=10, max_value=2000, value=200, step=1)
    n_floors = st.number_input('Число этажей', min_value=1, max_value=50, value=12, step=1)

    st.subheader('Местоположение МКД')
    city = st.selectbox('Город', options=list(cities.keys()), index=0)

    st.markdown('**Расчётная нагрузка, кВт**')
    p_calc_input = st.number_input(
        'Расчётная нагрузка, кВт',
        min_value=10.0,
        max_value=5000.0,
        value=283.0,
        step=1.0,
        label_visibility='collapsed',
    )

    st.subheader('Горизонт прогноза электрической нагрузки МКД')
    horizon = st.selectbox(
        'Горизонт',
        options=list(horizons.keys()),
        index=2,
        format_func=lambda x: {
            '4h': '4 часа',
            '8h': '8 часов',
            '24h': '24 часа',
            '7d': '7 дней',
            '14d': '14 дней',
            '1m': '1 месяц',
        }[x],
    )

    st.divider()
    run_btn = st.button('Запустить анализ', type='primary', use_container_width=True)

# вкладки
tab1, tab2, tab3 = st.tabs([
    'Прогноз электрической нагрузки МКД',
    'Доступная мощность для зарядных станций',
    'Детекция аномального электропотребления (майнинг)',
])

# инициализация session state
if 'results' not in st.session_state:
    st.session_state['results'] = None

# обработка кнопки
if run_btn:
    if uploaded_file is None:
        st.error('Загрузите CSV-файл.')
        st.stop()

    df_raw = pd.read_csv(uploaded_file)
    ok, msg = validate_csv(df_raw)
    if not ok:
        st.error(msg)
        st.stop()

    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)

    coords = cities[city]
    with st.spinner('Загрузка погоды'):
        df_weather = get_weather_forecast(coords['lat'], coords['lon'])

    with st.spinner('Формирование признаков'):
        df_feat = make_features(df_raw, n_flats=n_flats, n_floors=n_floors,
                                df_weather=df_weather)

    st.session_state['results'] = {
        'df_raw': df_raw,
        'df_feat': df_feat,
        'df_weather': df_weather,
        'n_flats': n_flats,
        'n_floors': n_floors,
        'p_calc': p_calc_input,
        'horizon': horizon,
        'city': city,
    }
st.write(f'DEBUG df_raw shape: {df_raw.shape}')
st.write(f'DEBUG df_feat shape: {df_feat.shape}')
st.write(f'DEBUG df_feat NaN: {df_feat.isnull().sum().sum()}')
# вкладка 1 - прогноз электрической нагрузки
with tab1:
    if st.session_state['results'] is None:
        st.info('Загрузите данные и нажмите Запустить анализ.')
    else:
        r = st.session_state['results']
        df_feat = r['df_feat']
        df_raw = r['df_raw']
        hz = r['horizon']
        hz_steps = horizons[hz]

        point_models, _ = load_point_models()
        base_model = point_models[hz]

        # fine-tuning на загруженных данных
        from lightgbm import LGBMRegressor

        status = st.status('Выполняется анализ', expanded=True)
        with status:
            st.write('Подготовка данных')
            df_feat = df_feat.copy()
            df_feat['power_target'] = df_feat['power'].shift(-hz_steps)
            df_train = df_feat.dropna(subset=['power_target'])
            x_new = df_train[feature_cols]
            y_new = df_train['power_target']

            st.write('Fine-tuning модели')
            st.write(f'DEBUG df_train shape: {df_train.shape}')
            st.write(f'DEBUG x_new shape: {x_new.shape}')
            ft_model = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            ft_model.fit(x_new, y_new, init_model=base_model)

            st.write('Формирование прогноза')
            last_ts = df_raw['timestamp'].max()
            future_timestamps = pd.date_range(
                start=last_ts + pd.Timedelta('30min'),
                periods=hz_steps,
                freq='30min',
            )

            history = df_raw['power'].values.tolist()
            weather = r['df_weather']
            preds = []

            for i, ts in enumerate(future_timestamps):
                row = {}
                row['hour'] = ts.hour
                row['minute'] = ts.minute
                row['weekday'] = ts.weekday()
                row['month'] = ts.month
                row['day_of_year'] = ts.dayofyear
                row['is_weekend'] = int(ts.weekday() >= 5)
                row['is_holiday'] = 0

                row['lag_1'] = history[-1]
                row['lag_2'] = history[-2] if len(history) >= 2 else history[-1]
                row['lag_48'] = history[-48] if len(history) >= 48 else history[0]
                row['lag_96'] = history[-96] if len(history) >= 96 else history[0]
                row['lag_336'] = history[-336] if len(history) >= 336 else history[0]

                row['rolling_mean_48'] = np.mean(history[-48:])
                row['rolling_mean_336'] = np.mean(history[-336:])

                w_row = weather[weather['timestamp'] == ts]
                if len(w_row) > 0:
                    row['temp_c'] = float(w_row['temp_c'].values[0])
                    row['humidity'] = float(w_row['humidity'].values[0])
                    row['cloudiness'] = float(w_row['cloudiness'].values[0])
                else:
                    row['temp_c'] = 10.0
                    row['humidity'] = 70.0
                    row['cloudiness'] = 50.0

                row['n_flats'] = r['n_flats']
                row['n_floors'] = r['n_floors']

                x_pred = pd.DataFrame([row])[feature_cols]
                pred = float(ft_model.predict(x_pred)[0])
                preds.append(pred)
                history.append(pred)

            st.write('Построение графика электрической нагрузки')

        # датафрейм с прогнозом
        df_forecast = pd.DataFrame({
            'timestamp': future_timestamps,
            'power_forecast': np.round(preds, 2),
        })

        # график
        import plotly.graph_objects as go

        fig = go.Figure()

        # факт - последние 336 точек (7 дней)
        df_hist = df_raw.tail(336)
        fig.add_trace(go.Scatter(
            x=df_hist['timestamp'],
            y=df_hist['power'],
            mode='lines',
            name='Факт',
            line=dict(color='#1f77b4', width=1.5),
        ))

        # прогноз
        fig.add_trace(go.Scatter(
            x=df_forecast['timestamp'],
            y=df_forecast['power_forecast'],
            mode='lines',
            name='Прогноз',
            line=dict(color='#d62728', width=2, dash='dash'),
        ))

        fig.update_layout(
            title=f'Прогноз нагрузки на горизонт {hz}',
            xaxis_title='Время',
            yaxis_title='Мощность, кВт',
            hovermode='x unified',
            template='plotly_white',
            height=450,
        )
        st.plotly_chart(fig, width='stretch')

        # скачать csv
        csv = df_forecast.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Скачать прогноз CSV',
            data=csv,
            file_name=f'forecast_{hz}.csv',
            mime='text/csv',
        )

# вкладка 2 - доступная мощность для зарядных станций электромобилей
with tab2:
    if st.session_state['results'] is None:
        st.info('Загрузите данные и нажмите Запустить анализ.')
    else:
        r = st.session_state['results']
        df_feat = r['df_feat']
        hz = r['horizon']
        hz_steps = horizons[hz]
        p_calc = r['p_calc']

        q_models, thresholds, _ = load_quantile_models()

        status2 = st.status('Выполняется анализ', expanded=True)
        with status2:
            st.write('Подготовка данных')
            df_feat = df_feat.copy()
            df_feat['power_target'] = df_feat['power'].shift(-hz_steps)
            df_train = df_feat.dropna(subset=['power_target'])

            st.write('Квантильный прогноз')
            last_ts = r['df_raw']['timestamp'].max()
            future_timestamps = pd.date_range(
                start=last_ts + pd.Timedelta('30min'),
                periods=hz_steps,
                freq='30min',
            )

            history = r['df_raw']['power'].values.tolist()
            weather = r['df_weather']
            preds_q = {q: [] for q in [10, 50, 90]}

            for ts in future_timestamps:
                row = {}
                row['hour'] = ts.hour
                row['minute'] = ts.minute
                row['weekday'] = ts.weekday()
                row['month'] = ts.month
                row['day_of_year'] = ts.dayofyear
                row['is_weekend'] = int(ts.weekday() >= 5)
                row['is_holiday'] = 0
                row['lag_1'] = history[-1]
                row['lag_2'] = history[-2] if len(history) >= 2 else history[-1]
                row['lag_48'] = history[-48] if len(history) >= 48 else history[0]
                row['lag_96'] = history[-96] if len(history) >= 96 else history[0]
                row['lag_336'] = history[-336] if len(history) >= 336 else history[0]
                row['rolling_mean_48'] = np.mean(history[-48:])
                row['rolling_mean_336'] = np.mean(history[-336:])

                w_row = weather[weather['timestamp'] == ts]
                if len(w_row) > 0:
                    row['temp_c'] = float(w_row['temp_c'].values[0])
                    row['humidity'] = float(w_row['humidity'].values[0])
                    row['cloudiness'] = float(w_row['cloudiness'].values[0])
                else:
                    row['temp_c'] = 10.0
                    row['humidity'] = 70.0
                    row['cloudiness'] = 50.0

                row['n_flats'] = r['n_flats']
                row['n_floors'] = r['n_floors']

                x_pred = pd.DataFrame([row])[feature_cols]
                for q in [10, 50, 90]:
                    preds_q[q].append(float(q_models[hz][q].predict(x_pred)[0]))
                history.append(preds_q[50][-1])

            st.write('Конформная калибровка')
            q_corr = thresholds[hz]['q_corr_80']
            q90_conf = np.array(preds_q[90]) + q_corr
            q10_conf = np.array(preds_q[10]) - q_corr

            st.write('Расчёт доступной мощности')
            p_avail = p_calc - q90_conf

            st.write('Построение графика')

        import plotly.graph_objects as go
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=list(future_timestamps) + list(future_timestamps[::-1]),
            y=list(np.full(len(future_timestamps), p_calc)) + list(q90_conf[::-1]),
            fill='toself',
            fillcolor='rgba(34,139,34,0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Доступная мощность',
        ))

        fig2.add_trace(go.Scatter(
            x=list(future_timestamps) + list(future_timestamps[::-1]),
            y=list(q90_conf) + list(q10_conf[::-1]),
            fill='toself',
            fillcolor='rgba(200,200,200,0.3)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Интервал Q0.1–Q0.9',
        ))

        fig2.add_trace(go.Scatter(
            x=future_timestamps,
            y=preds_q[50],
            mode='lines',
            name='Прогноз Q0.5',
            line=dict(color='#d62728', width=2),
        ))

        fig2.add_trace(go.Scatter(
            x=future_timestamps,
            y=q90_conf,
            mode='lines',
            name='Q0.9 (конформный)',
            line=dict(color='#ff7f0e', width=1.5, dash='dash'),
        ))

        fig2.add_trace(go.Scatter(
            x=[future_timestamps[0], future_timestamps[-1]],
            y=[p_calc, p_calc],
            mode='lines',
            name=f'P_расч = {p_calc:.0f} кВт',
            line=dict(color='darkgreen', width=1.5, dash='dot'),
        ))

        fig2.update_layout(
            title=f'Доступная мощность — горизонт {hz}',
            xaxis_title='Время',
            yaxis_title='Мощность, кВт',
            hovermode='x unified',
            template='plotly_white',
            height=450,
        )
        st.plotly_chart(fig2, width='stretch')

        col1, col2 = st.columns(2)
        col1.metric('P_доступная мин, кВт', f'{float(np.min(p_avail)):.1f}')
        col2.metric('P_доступная макс, кВт', f'{float(np.max(p_avail)):.1f}')

        df_avail = pd.DataFrame({
            'timestamp': future_timestamps,
            'q10_conf': np.round(q10_conf, 2),
            'q50': np.round(preds_q[50], 2),
            'q90_conf': np.round(q90_conf, 2),
            'p_calc': np.round(p_calc, 2),
            'p_avail': np.round(p_avail, 2),
        })

        csv2 = df_avail.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Скачать CSV',
            data=csv2,
            file_name=f'power_avail_{hz}.csv',
            mime='text/csv',
        )

# вкладка 3 - детекция майнинга
with tab3:
    if st.session_state['results'] is None:
        st.info('Загрузите данные и нажмите Запустить анализ.')
    else:
        r = st.session_state['results']
        df_raw = r['df_raw']

        status3 = st.status('Выполняется детекция', expanded=True)
        with status3:
            st.write('Загрузка автоэнкодера')
            ae_model, ae_scaler = load_autoencoder()

            st.write('Подготовка признаков')
            data = df_raw[['timestamp', 'power']].copy()
            data = data.sort_values('timestamp').reset_index(drop=True)

            data['hour'] = data['timestamp'].dt.hour
            data['weekday'] = data['timestamp'].dt.weekday
            data['month'] = data['timestamp'].dt.month
            data['is_weekend'] = (data['weekday'] >= 5).astype(int)
            data['lag_1'] = data['power'].shift(1)
            data['lag_48'] = data['power'].shift(48)
            data['lag_336'] = data['power'].shift(336)
            data['rolling_mean_48'] = data['power'].shift(1).rolling(48).mean()
            data['rolling_mean_336'] = data['power'].shift(1).rolling(336).mean()
            data['rolling_std_48'] = data['power'].shift(1).rolling(48).std()
            data = data.dropna().reset_index(drop=True)

            ae_features = [
                'power', 'hour', 'weekday', 'month', 'is_weekend',
                'lag_1', 'lag_48', 'lag_336',
                'rolling_mean_48', 'rolling_mean_336', 'rolling_std_48',
            ]

            st.write('Вычисление ошибки')
            import torch
            x_all = data[ae_features].values
            x_all_scaled = ae_scaler.transform(x_all)
            x_all_tensor = torch.FloatTensor(x_all_scaled)

            ae_model.eval()
            with torch.no_grad():
                x_reconstructed = ae_model(x_all_tensor).numpy()

            recon_error = np.mean((x_all_scaled - x_reconstructed) ** 2, axis=1)
            threshold = np.percentile(recon_error, 95)
            raw_anomaly = (recon_error > threshold).astype(int)

            st.write('Фильтрация одиночных выбросов (мин. 24ч)')
            min_duration = 48
            anomaly_filtered = np.zeros_like(raw_anomaly)
            i = 0
            while i < len(raw_anomaly):
                if raw_anomaly[i] == 1:
                    j = i
                    while j < len(raw_anomaly) and raw_anomaly[j] == 1:
                        j += 1
                    if j - i >= min_duration:
                        anomaly_filtered[i:j] = 1
                    i = j
                else:
                    i += 1

            data['recon_error'] = recon_error
            data['anomaly'] = anomaly_filtered
            n_anomalies = int(data['anomaly'].sum())

        import plotly.graph_objects as go
        fig3 = go.Figure()

        fig3.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['power'],
            mode='lines',
            name='Нагрузка',
            line=dict(color='#1f77b4', width=1),
        ))

        anomalies = data[data['anomaly'] == 1]
        fig3.add_trace(go.Scatter(
            x=anomalies['timestamp'],
            y=anomalies['power'],
            mode='markers',
            name='Аномалия',
            marker=dict(color='red', size=5),
        ))

        fig3.update_layout(
            title='Детекция аномального потребления',
            xaxis_title='Время',
            yaxis_title='Мощность, кВт',
            hovermode='x unified',
            template='plotly_white',
            height=450,
        )
        st.plotly_chart(fig3, width='stretch')

        anomaly_rate = n_anomalies / len(data)
        if anomaly_rate > 0.02:
            st.warning(
                f'Обнаружено {n_anomalies} аномальных точек ({anomaly_rate * 100:.1f}%) - '
                f'возможен майнинг или резкое изменение режима потребления из-за погодных факторов.'
            )
        else:
            st.success(f'Аномалий не обнаружено ({n_anomalies} точек, {anomaly_rate * 100:.1f}%).')