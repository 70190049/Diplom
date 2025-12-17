import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import threading
import os
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
import re

pd.set_option('future.no_silent_downcasting', True)
plt.rcParams.update({
    'font.size': 9,
    'axes.formatter.useoffset': False,
    'axes.formatter.limits': (-4, 4)
})


def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


class SimpleNN:
    def __init__(self, input_dim, lr=0.02, epochs=100, batch_size=32):
        self.W = np.random.randn(input_dim, 1) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1,))
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_dim = input_dim

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(np.float64)

    def forward(self, X):
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Ожидалось {self.input_dim} признаков, получено {X.shape[1]}")
        z = X @ self.W + self.b
        a = self.relu(z)
        return a, z

    def fit(self, X, y, residuals=None):
        targets = residuals if residuals is not None else y.reshape(-1, 1)
        n = X.shape[0]

        if self.W.shape[1] != 1:
            self.W = self.W[:, :1].copy()
        if self.b.ndim == 0:
            self.b = np.array([self.b])

        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X[idx], targets[idx]
            for i in range(0, n, self.batch_size):
                X_batch = X_shuf[i:i + self.batch_size]
                y_batch = y_shuf[i:i + self.batch_size]

                y_pred, z_batch = self.forward(X_batch)
                error = y_batch - y_pred
                dz = error * self.relu_derivative(z_batch)
                dW = -X_batch.T @ dz / X_batch.shape[0]
                db = -np.mean(dz, axis=0, keepdims=True)

                if dW.shape[1] != 1:
                    dW = dW[:, :1]
                if db.shape != (1, 1):
                    db = db[:1, :1]

                self.W -= self.lr * dW
                self.b -= self.lr * db[0, 0]

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return y_pred.flatten()


class NNBoost:
    def __init__(self, n_estimators=50, learning_rate=0.03):
        self.models = []
        self.n_estimators = n_estimators
        self.lr = learning_rate

    def fit(self, X, y):
        residuals = y.astype(np.float64).copy()
        for t in range(self.n_estimators):
            model = SimpleNN(X.shape[1], lr=0.02, epochs=100)
            model.fit(X, y, residuals=residuals)
            pred = model.predict(X)
            residuals -= self.lr * pred
            self.models.append(model)

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for model in self.models:
            pred += self.lr * model.predict(X)
        return pred


class SalesAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор продаж")
        self.root.geometry("1550x680")
        self.root.configure(bg="#f8f9fa")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        top_frame = tk.Frame(root, bg="#f8f9fa", pady=5)
        top_frame.pack(fill=tk.X)
        self.status_label = tk.Label(
            top_frame,
            text="Прогноз на 2026 год (12 мес.)",
            font=("Arial", 14, "bold"),
            fg="#1a5276",
            bg="#f8f9fa"
        )
        self.status_label.pack(side=tk.LEFT, padx=20)

        control_frame = tk.Frame(root, bg="#f8f9fa", pady=5)
        control_frame.pack(fill=tk.X)

        center_container = tk.Frame(control_frame, bg="#f8f9fa")
        center_container.pack(expand=True)

        tk.Label(center_container, text="Файл:", font=("Arial", 11), bg="#f8f9fa").pack(side=tk.LEFT, padx=5)
        self.path_entry = tk.Entry(center_container, width=65, font=("Consolas", 10))
        self.path_entry.pack(side=tk.LEFT, padx=5)
        self.path_entry.insert(0, "")

        self.browse_btn = tk.Button(
            center_container, text="Выбрать", command=self.browse_file,
            bg="#3498db", fg="white", padx=10
        )
        self.browse_btn.pack(side=tk.LEFT, padx=5)

        self.load_btn = tk.Button(
            center_container, text="Запустить анализ",
            command=self.start_analysis,
            bg="#27ae60", fg="white", font=("Arial", 10, "bold"), padx=15
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.export_btn = tk.Button(
            center_container, text="Сохранить в Excel",
            command=self.export_to_excel,
            bg="#218359", fg="white", font=("Arial", 10, "bold"), padx=10
        )
        self.export_btn.pack(side=tk.LEFT, padx=5)
        self.export_btn.config(state="disabled")

        cat_frame = tk.Frame(control_frame, bg="#f8f9fa", pady=5)
        cat_frame.pack(expand=True)

        tk.Label(cat_frame, text="Категории:", font=("Arial", 11), bg="#f8f9fa").pack(side=tk.LEFT, padx=5)

        list_frame = tk.Frame(cat_frame)
        list_frame.pack(side=tk.LEFT, padx=5)

        self.cat_listbox = tk.Listbox(
            list_frame,
            selectmode=tk.EXTENDED,
            height=5,
            width=28,
            font=("Arial", 10),
            exportselection=False
        )
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.cat_listbox.yview)
        self.cat_listbox.config(yscrollcommand=scrollbar.set)

        self.cat_listbox.pack(side=tk.LEFT, fill=tk.BOTH)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.apply_cat_btn = tk.Button(
            cat_frame, text="Применить", command=self.apply_category_selection,
            bg="#3498db", fg="white", padx=10, font=("Arial", 9)
        )
        self.apply_cat_btn.pack(side=tk.LEFT, padx=(10, 0))

        self.cat_listbox.insert(tk.END, "Все категории")
        self.cat_listbox.select_set(0)
        self.selected_categories = ["Все категории"]

        self.df_full_raw = None
        self.df_full = None
        self.monthly_df = None
        self.future_monthly = None
        self.monthly_by_cat = {}
        self.future_by_cat = {}
        self.is_loading = False
        self.anim1 = self.anim2 = None

        self.charts_frame = tk.Frame(root, bg="#f8f9fa")
        self.charts_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig1, self.ax1 = plt.subplots(figsize=(6.5, 4.8))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.charts_frame)
        self.canvas1.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.fig2, self.ax2 = plt.subplots(figsize=(6.5, 4.8))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.charts_frame)
        self.canvas2.get_tk_widget().grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.charts_frame.grid_columnconfigure(0, weight=1)
        self.charts_frame.grid_columnconfigure(1, weight=1)

    def _on_closing(self):
        for anim in [self.anim1, self.anim2]:
            if anim:
                try:
                    anim.event_source.stop()
                except:
                    pass
        try:
            self.root.destroy()
        except:
            pass
        os._exit(0)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls")])
        if path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, path)
            try:
                df = pd.read_excel(path)
                df.columns = df.columns.str.strip()
                required = ['Дата', 'Товар', 'Категория', 'Цена за шт', 'Количество продаж', 'Канал продаж', 'Скидки']
                missing = [c for c in required if c not in df.columns]
                if missing:
                    raise ValueError(f"Отсутствуют столбцы: {missing}")
                self.df_full_raw = df.copy()
                categories = sorted(df['Категория'].dropna().unique().tolist())
                self.cat_listbox.delete(1, tk.END)
                for cat in categories:
                    self.cat_listbox.insert(tk.END, cat)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{e}")

    def apply_category_selection(self):
        indices = self.cat_listbox.curselection()
        if not indices:
            messagebox.showwarning("Внимание", "Выберите хотя бы одну категорию.")
            self.cat_listbox.select_set(0)
            return

        selected = [self.cat_listbox.get(i) for i in indices]

        if len(selected) > 5:
            messagebox.showwarning("Внимание", "Можно выбрать не более 5 категорий.")
            self.cat_listbox.selection_clear(0, tk.END)
            for i in indices[:5]:
                self.cat_listbox.selection_set(i)
            return

        if "Все категории" in selected:
            self.selected_categories = ["Все категории"]
            self.cat_listbox.selection_clear(0, tk.END)
            self.cat_listbox.select_set(0)
        else:
            self.selected_categories = selected

        if not self.is_loading and self.df_full_raw is not None:
            self.load_btn.config(bg="#27ae60", text="Запустить анализ")

    def start_analysis(self):
        if self.is_loading:
            return
        path = self.path_entry.get().strip()
        if not path:
            messagebox.showerror("Ошибка", "Укажите путь к файлу.")
            return
        self._set_ui_loading(True)
        threading.Thread(target=self._background_analysis, args=(path,), daemon=True).start()

    def _background_analysis(self, path):
        try:
            self.df_full = None
            self.monthly_df = None
            self.future_monthly = None
            self.monthly_by_cat = {}
            self.future_by_cat = {}

            df = self.df_full_raw.copy()

            if "Все категории" in self.selected_categories:
                pass
            else:
                df = df[df['Категория'].isin(self.selected_categories)].copy()
                if df.empty:
                    raise ValueError(f"Нет данных по выбранным категориям: {self.selected_categories}")

            df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Дата'])
            df['Цена за шт'] = pd.to_numeric(df['Цена за шт'], errors='coerce')
            df['Количество продаж'] = pd.to_numeric(df['Количество продаж'], errors='coerce')
            df['Скидки'] = df['Скидки'].fillna('нет').astype(str).str.strip().str.lower()
            df['Скидки'] = df['Скидки'].replace({'нет': 0, '': 0, '-': 0})
            df['Скидки'] = pd.to_numeric(df['Скидки'], errors='coerce').fillna(0)
            df = df.dropna(subset=['Цена за шт', 'Количество продаж'])

            if len(df) < 5:
                raise ValueError(f"Слишком мало данных (требуется ≥5 записей, имеется: {len(df)})")

            self.df_full = df.copy()

            le_cat = LabelEncoder()
            le_channel = LabelEncoder()
            df['код_категории'] = le_cat.fit_transform(df['Категория'])
            df['код_канала'] = le_channel.fit_transform(df['Канал продаж'])

            df['месяц'] = df['Дата'].dt.month
            df['квартал'] = df['Дата'].dt.quarter
            df['день_недели'] = df['Дата'].dt.dayofweek
            df['выходной'] = (df['день_недели'] >= 5).astype(int)

            numeric_cols = ['Цена за шт', 'Скидки', 'месяц', 'квартал', 'день_недели', 'выходной']
            categorical_cols = ['код_категории', 'код_канала']

            scaler = StandardScaler()
            X_numeric = scaler.fit_transform(df[numeric_cols])
            X_categorical = df[categorical_cols].values
            X = np.hstack([X_numeric, X_categorical])
            y = df['Количество продаж'].values

            try:
                model = NNBoost(n_estimators=50, learning_rate=0.03)
                model.fit(X, y)
                pred = model.predict(X)
                df['прогноз_спроса'] = pred
                df['оценка_выручки'] = df['Цена за шт'] * df['прогноз_спроса']
            except Exception as e_inner:
                print(f"Обучение не удалось: {e_inner}. Fallback.")
                df['прогноз_спроса'] = df['Количество продаж']
                df['оценка_выручки'] = df['Количество продаж'] * df['Цена за шт']

            self.df_full = df.copy()

            df['месяц_period'] = df['Дата'].dt.to_period('M')

            monthly = df.groupby('месяц_period', as_index=False).agg({
                'Количество продаж': 'sum',
                'прогноз_спроса': 'sum',
                'оценка_выручки': 'sum',
                'Цена за шт': 'mean'
            }).reset_index(drop=True)
            for col in ['прогноз_спроса', 'оценка_выручка']:
                if col not in monthly.columns:
                    monthly[col] = monthly['Количество продаж'] * (monthly['Цена за шт'] if col == 'оценка_выручка' else 1)
            monthly['месяц_str'] = monthly['месяц_period'].astype(str)
            monthly = monthly.sort_values('месяц_period').reset_index(drop=True)
            self.monthly_df = monthly

            if "Все категории" not in self.selected_categories and len(self.selected_categories) > 1:
                for cat in self.selected_categories:
                    cat_df = df[df['Категория'] == cat].copy()
                    if 'прогноз_спроса' not in cat_df.columns:
                        cat_df['прогноз_спроса'] = cat_df['Количество продаж']
                    if 'оценка_выручка' not in cat_df.columns:
                        cat_df['оценка_выручка'] = cat_df['Количество продаж'] * cat_df['Цена за шт']
                    if cat_df.empty:
                        self.monthly_by_cat[cat] = pd.DataFrame({
                            'месяц_period': monthly['месяц_period'],
                            'Количество продаж': np.zeros(len(monthly)),
                            'оценка_выручка': np.zeros(len(monthly))
                        }).set_index('месяц_period')
                        self.future_by_cat[cat] = np.zeros(12)
                        continue
                    cat_monthly = cat_df.groupby('месяц_period').agg({
                        'Количество продаж': 'sum',
                        'оценка_выручка': 'sum'
                    })
                    cat_monthly = cat_monthly.reindex(monthly['месяц_period']).fillna(0)
                    self.monthly_by_cat[cat] = cat_monthly

                    try:
                        future_rows = []
                        last_date = cat_df['Дата'].max()
                        for i in range(1, 13):
                            next_date = last_date + pd.DateOffset(months=i)
                            hist = cat_df[cat_df['Дата'].dt.month == next_date.month]
                            if len(hist) == 0:
                                hist = cat_df.sample(min(50, len(cat_df)), replace=True)
                            for _, row in hist.iterrows():
                                new_price = row['Цена за шт'] * (1.005 ** i)
                                future_rows.append({
                                    'Цена за шт': new_price,
                                    'Скидки': row['Скидки'],
                                    'месяц': next_date.month,
                                    'квартал': next_date.quarter,
                                    'день_недели': 0,
                                    'выходной': 0,
                                    'код_категории': row['код_категории'],
                                    'код_канала': row['код_канала']
                                })
                        future_df = pd.DataFrame(future_rows)
                        X_fut = np.hstack([
                            scaler.transform(future_df[numeric_cols]),
                            future_df[categorical_cols].values
                        ])
                        future_df['прогноз_спроса'] = model.predict(X_fut)
                        future_df['прогноз_выручка'] = future_df['Цена за шт'] * future_df['прогноз_спроса']
                        rev_series = future_df.groupby(future_df.index // max(1, len(future_df)//12))['прогноз_выручка'].sum()
                        vals = rev_series.values[:12]
                        if len(vals) < 12:
                            vals = np.pad(vals, (0, 12 - len(vals)), constant_values=(vals[-1] if len(vals) > 0 else 50000))
                        self.future_by_cat[cat] = vals
                    except Exception as e_inner:
                        print(f"Прогноз для {cat} не удался: {e_inner}")
                        avg_monthly = cat_df.groupby(cat_df['Дата'].dt.to_period('M'))['Количество продаж'].sum().mean()
                        avg_price = cat_df['Цена за шт'].mean()
                        self.future_by_cat[cat] = np.full(12, avg_monthly * avg_price)
            else:
                future_rows = []
                last_date = df['Дата'].max()
                for i in range(1, 13):
                    next_date = last_date + pd.DateOffset(months=i)
                    hist = df[df['Дата'].dt.month == next_date.month]
                    if len(hist) == 0:
                        hist = df.sample(min(50, len(df)), replace=True)
                    for _, row in hist.iterrows():
                        new_price = row['Цена за шт'] * (1.005 ** i)
                        future_rows.append({
                            'Цена за шт': new_price,
                            'Скидки': row['Скидки'],
                            'месяц': next_date.month,
                            'квартал': next_date.quarter,
                            'день_недели': 0,
                            'выходной': 0,
                            'код_категории': row['код_категории'],
                            'код_канала': row['код_канала']
                        })
                future_df = pd.DataFrame(future_rows)
                X_fut = np.hstack([
                    scaler.transform(future_df[numeric_cols]),
                    future_df[categorical_cols].values
                ])
                try:
                    future_df['прогноз_спроса'] = model.predict(X_fut)
                except:
                    future_df['прогноз_спроса'] = future_df['Цена за шт']
                future_df['прогноз_выручка'] = future_df['Цена за шт'] * future_df['прогноз_спроса']
                grouped = future_df.groupby(future_df.index // max(1, len(future_df)//12))
                future_monthly = pd.DataFrame({
                    'month_id': range(12),
                    'прогноз_выручка': grouped['прогноз_выручка'].sum().values[:12],
                    'прогноз_спроса': grouped['прогноз_спроса'].sum().values[:12]
                })
                while len(future_monthly) < 12:
                    future_monthly = pd.concat([future_monthly, pd.DataFrame([{
                        'month_id': len(future_monthly),
                        'прогноз_выручка': future_monthly['прогноз_выручка'].mean(),
                        'прогноз_спроса': future_monthly['прогноз_спроса'].mean()
                    }])], ignore_index=True)
                future_monthly = future_monthly.head(12)
                future_monthly['месяц_str'] = [(last_date + pd.DateOffset(months=i+1)).strftime('%Y-%m') for i in range(12)]
                self.future_monthly = future_monthly

            self.root.after(0, self._update_ui_with_results, self.selected_categories)

        except Exception as e:
            import traceback
            err_msg = str(e)
            print("Ошибка в фоновом потоке:", err_msg)
            print(traceback.format_exc())
            self.root.after(0, lambda msg=err_msg: messagebox.showerror("Ошибка", msg))
        finally:
            self.root.after(0, self._set_ui_loading, False)

    def _update_ui_with_results(self, categories):
        try:
            self._draw_animation(categories)
            self.export_btn.config(state="normal")
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda msg=err_msg: messagebox.showerror("Ошибка визуализации", msg))

    def _set_ui_loading(self, state):
        self.is_loading = state
        if state:
            self.load_btn.config(state="disabled", text="Анализ...")
            self.browse_btn.config(state="disabled")
            self.apply_cat_btn.config(state="disabled")
            self.cat_listbox.config(state="disabled")
            self.export_btn.config(state="disabled")
        else:
            self.load_btn.config(state="normal", text="Запустить анализ")
            self.browse_btn.config(state="normal")
            self.apply_cat_btn.config(state="normal")
            self.cat_listbox.config(state="normal")

    def _draw_animation(self, categories):
        for anim in [self.anim1, self.anim2]:
            if anim:
                try:
                    anim.event_source.stop()
                except:
                    pass
        self.anim1 = self.anim2 = None

        self.fig1.clear()
        self.fig2.clear()
        self.ax1 = self.fig1.add_subplot(111)
        self.ax2 = self.fig2.add_subplot(111)

        df = self.monthly_df
        if df is None or df.empty:
            self.ax1.text(0.5, 0.5, "Нет данных", ha='center', va='center', fontsize=14, color='gray')
            self.ax2.text(0.5, 0.5, "Нет данных", ha='center', va='center', fontsize=14, color='gray')
            self.canvas1.draw()
            self.canvas2.draw()
            return

        n = len(df)
        x = np.arange(n)
        x_labels = df['месяц_str'].tolist() if 'месяц_str' in df.columns else [f"M{i+1}" for i in range(n)]

        colors = plt.cm.tab10.colors
        if "Все категории" in categories or len(categories) == 1:
            bars = self.ax1.bar(x, df['Количество продаж'], width=0.6, color='#76c68f', alpha=0.8, label='Факт: спрос')
            ax1b = self.ax1.twinx()
            line, = ax1b.plot(x, df['оценка_выручки'], 'o-', color='#22a7f0', linewidth=2.2, markersize=6, label='Оценка: выручка')
        else:
            ax1b = self.ax1.twinx()
            bottom = np.zeros(n)
            for i, cat in enumerate(categories):
                color = colors[i % len(colors)]
                if cat in self.monthly_by_cat:
                    series = self.monthly_by_cat[cat]
                    qty = series['Количество продаж'].values
                    rev = series['оценка_выручка'].values
                    self.ax1.bar(x, qty, bottom=bottom, width=0.6, color=color, alpha=0.7, label=f'{cat}: факт')
                    bottom += qty
                    ax1b.plot(x, rev, '^-', color=color, linewidth=1.8, markersize=5, label=f'{cat}: выручка')
                else:
                    self.ax1.bar(x, np.zeros(n), width=0.6, color=color, alpha=0.3, label=f'{cat} (нет данных)')
                    ax1b.plot(x, np.zeros(n), 'x-', color=color, linewidth=1, markersize=3, label=f'{cat} (прогноз недоступен)')

        self.ax1.set_ylabel('Спрос, шт', fontsize=10)
        ax1b.set_ylabel('Выручка, ₽', fontsize=10)
        self.ax1.set_xlabel('Месяц', fontsize=10)
        title = "все категории" if "Все категории" in categories else ", ".join(categories)
        self.ax1.set_title(f'Факт и прогноз спроса\n({title})', fontweight='bold', fontsize=11)
        self.ax1.set_xticks(x)
        self.ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        self.ax1.grid(True, linestyle='--', alpha=0.5, linewidth=0.7)
        self.ax1.spines['top'].set_visible(False)
        ax1b.spines['top'].set_visible(False)

        handles1, labels1 = self.ax1.get_legend_handles_labels()
        handles2, labels2 = ax1b.get_legend_handles_labels()
        self.ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=8,
                        ncol=1 if len(labels1 + labels2) < 8 else 2)

        def animate1(frame):
            frame = min(frame, n)
            if isinstance(bars, list):
                for i, bar_container in enumerate(bars):
                    qty_vals = actual_qty_by_cat[i]
                    for j, rect in enumerate(bar_container):
                        h = qty_vals[j] if j < frame else 0.0
                        rect.set_height(h)
            else:
                qty_vals = actual_qty
                for j, rect in enumerate(bars):
                    h = qty_vals[j] if j < frame else 0.0
                    rect.set_height(h)

            for i, line in enumerate(lines):
                if "Все категории" in categories or len(categories) == 1:
                    x_data = x[:frame]
                    y_data = pred_rev[:frame]
                else:
                    x_data = x[:frame]
                    y_data = pred_rev_by_cat[i][:frame]
                line.set_data(x_data, y_data)

            return tuple([rect for bar in ([bars] if not isinstance(bars, list) else bars) for rect in bar]) + tuple(lines)

        if "Все категории" in categories or len(categories) == 1:
            actual_qty = df['Количество продаж'].values
            pred_rev = df['оценка_выручки'].values
            lines = [line]
        else:
            actual_qty_by_cat = []
            pred_rev_by_cat = []
            for i, cat in enumerate(categories):
                if cat in self.monthly_by_cat:
                    series = self.monthly_by_cat[cat]
                    qty = series['Количество продаж'].values
                    rev = series['оценка_выручка'].values
                    actual_qty_by_cat.append(qty)
                    pred_rev_by_cat.append(rev)
                else:
                    actual_qty_by_cat.append(np.zeros(n))
                    pred_rev_by_cat.append(np.zeros(n))
            lines = []

        self.anim1 = FuncAnimation(self.fig1, animate1, frames=n + 1, interval=100, blit=False, repeat=False)
        self.canvas1.draw()

        x_vals = np.arange(12)
        month_labels = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]

        if "Все категории" in categories or len(categories) == 1:
            y = self.future_monthly['прогноз_выручка'].values[:12].astype(float)
            if len(y) < 12:
                y = np.pad(y, (0, 12 - len(y)), constant_values=(y[-1] if len(y) > 0 else 50000))
            self.ax2.plot(x_vals, y, 's-', color='#a6d75b', linewidth=2.5, markersize=7, label='2026 (суммарно)')
            self.ax2.legend(loc='upper left', fontsize=9)
        else:
            plotted = False
            for i, cat in enumerate(categories):
                color = colors[i % len(colors)]
                if cat in self.future_by_cat and len(self.future_by_cat[cat]) >= 12:
                    y = self.future_by_cat[cat][:12]
                    self.ax2.plot(x_vals, y, 'o--', color=color, linewidth=2, markersize=5, label=cat)
                    plotted = True
            if not plotted:
                self.ax2.plot([], [], label="Нет данных для прогноза")
            self.ax2.legend(loc='upper left', fontsize=8, ncol=1 if len(categories) < 5 else 2)

        self.ax2.set_title(f'Прогноз выручки на 2026\n({title})', fontweight='bold', fontsize=11)
        self.ax2.set_xlabel('Месяц', fontsize=10)
        self.ax2.set_ylabel('Выручка (₽)', fontsize=10)
        self.ax2.set_xticks(x_vals)
        self.ax2.set_xticklabels(month_labels, fontsize=9)
        self.ax2.grid(True, linestyle='--', alpha=0.5, linewidth=0.7)
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.set_ylim(bottom=0)

        def animate2(frame):
            frame = min(frame, 12)
            for i, line in enumerate(lines2):
                x_data = x_vals[:frame]
                y_data = future_by_cat[cat][:frame] if cat in self.future_by_cat else np.zeros(frame)
                line.set_data(x_data, y_data)
            return lines2

        lines2 = []
        if "Все категории" in categories or len(categories) == 1:
            y = self.future_monthly['прогноз_выручка'].values[:12].astype(float)
            if len(y) < 12:
                y = np.pad(y, (0, 12 - len(y)), constant_values=(y[-1] if len(y) > 0 else 50000))
            line2, = self.ax2.plot([], [], 's-', color='#a6d75b', linewidth=2.5, markersize=7, label='2026 (суммарно)')
            lines2 = [line2]
        else:
            for i, cat in enumerate(categories):
                color = colors[i % len(colors)]
                if cat in self.future_by_cat and len(self.future_by_cat[cat]) >= 12:
                    y = self.future_by_cat[cat][:12]
                    line2, = self.ax2.plot([], [], 'o--', color=color, linewidth=2, markersize=5, label=cat)
                    lines2.append(line2)

        self.anim2 = FuncAnimation(self.fig2, animate2, frames=13, interval=100, blit=False, repeat=False)
        self.canvas2.draw()

    def export_to_excel(self):
        if self.df_full is None:
            messagebox.showwarning("Внимание", "Сначала запустите анализ.")
            return

        try:
            cat_name = "все" if "Все категории" in self.selected_categories else "_".join(self.selected_categories[:3])
            default_name = f"Прогноз_спроса_{sanitize_filename(cat_name)}" + ".xlsx"
            path = filedialog.asksaveasfilename(
                initialfile=default_name,
                defaultextension=".xlsx",
                filetypes=[("Excel", "*.xlsx")],
                title="Сохранить прогноз"
            )
            if not path:
                return
            if not path.lower().endswith('.xlsx'):
                path += '.xlsx'

            wb = openpyxl.Workbook()
            ws1 = wb.active
            ws1.title = "Прогноз на 2026"
            ws1.append(["Месяц", "Прогноз выручки (₽)", "Прогноз спроса (шт)"])
            for cell in ws1[1]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="DEEBF7")
                cell.alignment = Alignment(horizontal="center")

            if self.future_monthly is not None and not self.future_monthly.empty:
                for _, row in self.future_monthly.iterrows():
                    month = str(row['месяц_str']) if 'месяц_str' in row else f"2026-{int(row['month_id'])+1:02d}"
                    rev = int(round(row['прогноз_выручка'])) if pd.notna(row['прогноз_выручка']) else 0
                    qty = int(round(row['прогноз_спроса'])) if pd.notna(row['прогноз_спроса']) else 0
                    ws1.append([month, rev, qty])
            else:
                rev_total = np.zeros(12)
                qty_total = np.zeros(12)
                for cat in self.selected_categories:
                    if cat in self.future_by_cat:
                        rev_total += self.future_by_cat[cat]
                for i in range(12):
                    ws1.append([f"2026-{i+1:02d}", int(rev_total[i]), int(rev_total[i] // 100)])

            ws2 = wb.create_sheet("Оценка по месяцам")
            ws2.append(["Месяц", "Факт: спрос (шт)", "Оценка: спрос (шт)", "Факт: выручка (₽)", "Оценка: выручка (₽)"])
            for cell in ws2[1]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="E2F0D9")
                cell.alignment = Alignment(horizontal="center")

            for _, row in self.monthly_df.iterrows():
                month = str(row['месяц_str'])
                fact_qty = int(row['Количество продаж'])
                pred_qty = int(round(row['прогноз_спроса'])) if 'прогноз_спроса' in row and pd.notna(row['прогноз_спроса']) else fact_qty
                fact_rev = int(fact_qty * row['Цена за шт'])
                pred_rev = int(round(row['оценка_выручки'])) if 'оценка_выручки' in row and pd.notna(row['оценка_выручки']) else fact_rev
                ws2.append([month, fact_qty, pred_qty, fact_rev, pred_rev])

            ws3 = wb.create_sheet("По категориям")
            ws3.append(["Категория", "Факт: спрос", "Прогноз: спрос", "Факт: выручка", "Прогноз: выручка"])
            for cell in ws3[1]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="D6EAF8")
                cell.alignment = Alignment(horizontal="center")

            for cat in self.selected_categories:
                if cat == "Все категории":
                    cat_df = self.df_full
                else:
                    cat_df = self.df_full[self.df_full['Категория'] == cat]
                if cat_df.empty:
                    continue
                fact_qty = cat_df['Количество продаж'].sum()
                pred_qty = cat_df['прогноз_спроса'].sum() if 'прогноз_спроса' in cat_df.columns else fact_qty
                fact_rev = (cat_df['Количество продаж'] * cat_df['Цена за шт']).sum()
                pred_rev = cat_df['оценка_выручки'].sum() if 'оценка_выручки' in cat_df.columns else fact_rev
                ws3.append([cat, int(fact_qty), int(pred_qty), int(fact_rev), int(pred_rev)])

            ws4 = wb.create_sheet("Товары")
            ws4.append(["Товар", "Категория", "Средняя цена", "Факт: спрос", "Прогноз: спрос", "Факт: выручка", "Прогноз: выручка"])
            for cell in ws4[1]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="D9E1F2")
                cell.alignment = Alignment(horizontal="center")

            df_export = self.df_full.copy()
            if 'прогноз_спроса' not in df_export.columns:
                df_export['прогноз_спроса'] = df_export['Количество продаж']
            if 'оценка_выручки' not in df_export.columns:
                df_export['оценка_выручки'] = df_export['Количество продаж'] * df_export['Цена за шт']

            item_agg = df_export.groupby('Товар', as_index=False).agg({
                'Категория': 'first',
                'Цена за шт': 'mean',
                'Количество продаж': 'sum',
                'прогноз_спроса': 'sum',
                'оценка_выручки': 'sum'
            })

            for _, row in item_agg.iterrows():
                ws4.append([
                    row['Товар'],
                    row['Категория'],
                    round(row['Цена за шт'], 2),
                    int(row['Количество продаж']),
                    int(round(row['прогноз_спроса'])),
                    int(round(row['Количество продаж'] * row['Цена за шт'])),
                    int(round(row['оценка_выручки']))
                ])

            wb.save(path)
            messagebox.showinfo("Успех", f"Прогноз сохранён:\n{os.path.basename(path)}")

        except PermissionError:
            messagebox.showerror("Ошибка доступа", "Нет прав на запись. Попробуйте папку «Документы».")
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda msg=err_msg: messagebox.showerror("Ошибка экспорта", msg))


if __name__ == "__main__":
    root = tk.Tk()
    app = SalesAnalyzerApp(root)
    root.mainloop()