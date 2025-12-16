import tkinter as tk
import tkinter.ttk as ttk
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
plt.rcParams.update({'font.size': 9})


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
        self.root.geometry("1550x580")
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
        tk.Label(cat_frame, text="Категория:", font=("Arial", 11), bg="#f8f9fa").pack(side=tk.LEFT, padx=5)
        self.category_var = tk.StringVar(value="Загрузите файл")
        self.category_combo = ttk.Combobox(
            cat_frame,
            textvariable=self.category_var,
            state="disabled",
            width=25,
            font=("Arial", 10)
        )
        self.category_combo.pack(side=tk.LEFT, padx=5)
        self.category_combo.bind("<<ComboboxSelected>>", self.on_category_change)

        self.charts_frame = tk.Frame(root, bg="#f8f9fa")
        self.charts_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig1, self.ax1 = plt.subplots(figsize=(6, 4.5))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.charts_frame)
        self.canvas1.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.fig2, self.ax2 = plt.subplots(figsize=(6, 4.5))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.charts_frame)
        self.canvas2.get_tk_widget().grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.charts_frame.grid_columnconfigure(0, weight=1)
        self.charts_frame.grid_columnconfigure(1, weight=1)

        self.is_loading = False
        self.anim1 = self.anim2 = None
        self.monthly_df = None
        self.future_monthly = None
        self.df_full = None
        self.current_category = "Все категории"

    def _on_closing(self):
        if self.anim1:
            try:
                self.anim1.event_source.stop()
            except:
                pass
        if self.anim2:
            try:
                self.anim2.event_source.stop()
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
            self.category_combo.config(values=["Загрузите файл"])
            self.category_var.set("Загрузите файл")
            self.category_combo.config(state="disabled", foreground="gray")

    def on_category_change(self, event=None):
        selected = self.category_var.get()
        if selected == "Загрузите файл":
            self.category_combo.config(foreground="gray")
        else:
            self.category_combo.config(foreground="black")
        if not self.is_loading and self.df_full is not None:
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
            self.current_category = "Все категории"

            df = pd.read_excel(path)
            df.columns = df.columns.str.strip()

            required = ['Дата', 'Товар', 'Категория', 'Цена за шт', 'Количество продаж', 'Канал продаж', 'Скидки']
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Отсутствуют столбцы: {missing}")

            selected_cat = self.category_var.get()
            if selected_cat == "Загрузите файл":
                selected_cat = "Все категории"

            if selected_cat != "Все категории":
                df_orig_len = len(df)
                df = df[df['Категория'] == selected_cat].copy()
                if df.empty:
                    raise ValueError(f"Нет данных по категории '{selected_cat}' (всего строк: {df_orig_len})")

            df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Дата'])
            df['Цена за шт'] = pd.to_numeric(df['Цена за шт'], errors='coerce')
            df['Количество продаж'] = pd.to_numeric(df['Количество продаж'], errors='coerce')
            df['Скидки'] = df['Скидки'].fillna('нет').astype(str).str.strip().str.lower()
            df['Скидки'] = df['Скидки'].replace({'нет': 0, '': 0, '-': 0})
            df['Скидки'] = pd.to_numeric(df['Скидки'], errors='coerce').fillna(0)
            df = df.dropna(subset=['Цена за шт', 'Количество продаж'])

            if len(df) < 5:
                raise ValueError(f"Слишком мало данных для обучения (требуется ≥5 записей, имеется: {len(df)})")

            self.df_full = df.copy()
            all_cats = ["Все категории"] + sorted(df['Категория'].dropna().unique().tolist())
            self.root.after(0, lambda: self.category_combo.config(
                values=all_cats, state="readonly", foreground="black"
            ))

            df['месяц'] = df['Дата'].dt.month
            df['квартал'] = df['Дата'].dt.quarter
            df['день_недели'] = df['Дата'].dt.dayofweek
            df['выходной'] = (df['день_недели'] >= 5).astype(int)

            le_cat = LabelEncoder()
            le_channel = LabelEncoder()
            df['код_категории'] = le_cat.fit_transform(df['Категория'])
            df['код_канала'] = le_channel.fit_transform(df['Канал продаж'])

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
                self.df_full = df.copy()
            except Exception as e:
                print(f"Обучение не удалось: {e}. Используем прогноз = факт.")
                self.df_full['прогноз_спроса'] = self.df_full['Количество продаж']
                self.df_full['оценка_выручки'] = self.df_full['Количество продаж'] * self.df_full['Цена за шт']

            self.df_full['месяц_period'] = self.df_full['Дата'].dt.to_period('M')
            monthly = self.df_full.groupby('месяц_period', as_index=False).agg({
                'Количество продаж': 'sum',
                'прогноз_спроса': 'sum',
                'оценка_выручки': 'sum',
                'Цена за шт': 'mean'
            }).reset_index(drop=True)

            for col in ['прогноз_спроса', 'оценка_выручки']:
                if col not in monthly.columns:
                    raise RuntimeError(f"Столбец '{col}' отсутствует после агрегации!")

            monthly['месяц_str'] = monthly['месяц_period'].astype(str)
            monthly = monthly.sort_values('месяц_period').reset_index(drop=True)

            last_date = self.df_full['Дата'].max()
            future_rows = []
            for i in range(1, 13):
                next_date = last_date + pd.DateOffset(months=i)
                hist = self.df_full[self.df_full['Дата'].dt.month == next_date.month]
                if len(hist) == 0:
                    hist = self.df_full.sample(min(50, len(self.df_full)), replace=True)
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
            X_fut_numeric = scaler.transform(future_df[numeric_cols])
            X_fut_categorical = future_df[categorical_cols].values
            X_fut = np.hstack([X_fut_numeric, X_fut_categorical])

            try:
                future_df['прогноз_спроса'] = model.predict(X_fut)
            except:
                avg_demand = self.df_full.groupby(self.df_full['Дата'].dt.month)['Количество продаж'].mean()
                month = next_date.month
                future_df['прогноз_спроса'] = avg_demand.get(month, self.df_full['Количество продаж'].mean())

            future_df['прогноз_выручка'] = future_df['Цена за шт'] * future_df['прогноз_спроса']

            n_months = 12
            group_size = max(1, len(future_df) // n_months)
            future_df['month_id'] = np.arange(len(future_df)) // group_size
            future_df = future_df[future_df['month_id'] < n_months]
            future_monthly = future_df.groupby('month_id')[['прогноз_выручка', 'прогноз_спроса']].sum().reset_index()
            future_monthly = future_monthly.sort_values('month_id').reset_index(drop=True)

            while len(future_monthly) < n_months:
                avg_rev = future_monthly['прогноз_выручка'].mean() if len(future_monthly) > 0 else 50000
                avg_qty = future_monthly['прогноз_спроса'].mean() if len(future_monthly) > 0 else 100
                new_row = pd.DataFrame([{
                    'month_id': len(future_monthly),
                    'прогноз_выручка': avg_rev,
                    'прогноз_спроса': avg_qty
                }])
                future_monthly = pd.concat([future_monthly, new_row], ignore_index=True)
            future_monthly = future_monthly.head(n_months)
            future_monthly['месяц_str'] = [(last_date + pd.DateOffset(months=i + 1)).strftime('%Y-%m') for i in
                                           range(n_months)]

            self.root.after(0, self._update_ui_with_results, monthly, future_monthly, selected_cat)

        except Exception as e:
            import traceback
            print("Ошибка:", traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
        finally:
            self.root.after(0, self._set_ui_loading, False)

    def _update_ui_with_results(self, monthly_df, future_monthly, category):
        try:
            for col in ['Количество продаж', 'прогноз_спроса', 'оценка_выручки', 'месяц_str']:
                if col not in monthly_df.columns:
                    raise ValueError(f"Столбец '{col}' отсутствует в monthly_df")

            self.monthly_df = monthly_df
            self.future_monthly = future_monthly
            self.current_category = category
            self._draw_animation(category)
            self.export_btn.config(state="normal")
        except Exception as e:
            messagebox.showerror("Ошибка данных", f"Невозможно отобразить результаты:\n{str(e)}")

    def _set_ui_loading(self, state):
        self.is_loading = state
        if state:
            self.load_btn.config(state="disabled", text="Анализ...")
            self.browse_btn.config(state="disabled")
            self.category_combo.config(state="disabled")
            self.export_btn.config(state="disabled")
        else:
            self.load_btn.config(state="normal", text="Запустить анализ")
            self.browse_btn.config(state="normal")
            if self.df_full is not None:
                self.category_combo.config(state="readonly")

    def _draw_animation(self, category):
        if self.anim1:
            try:
                self.anim1.event_source.stop()
            except:
                pass
            self.anim1 = None
        if self.anim2:
            try:
                self.anim2.event_source.stop()
            except:
                pass
            self.anim2 = None

        self.fig1.clear()
        self.fig2.clear()
        self.ax1 = self.fig1.add_subplot(111)
        self.ax2 = self.fig2.add_subplot(111)

        df = self.monthly_df
        n = len(df)

        bars = self.ax1.bar(range(n), df['Количество продаж'], width=0.6, color='#76c68f', alpha=0.7,
                            label='Факт: спрос, шт')
        ax1b = self.ax1.twinx()
        line, = ax1b.plot(range(n), df['оценка_выручки'], 'o-', color='#22a7f0', linewidth=2, markersize=6,
                          label='Оценка: выручка, ₽')

        self.ax1.set_ylabel('Спрос, шт', color='#000000', fontsize=10)
        ax1b.set_ylabel('Выручка, ₽', color='#000000', fontsize=10)
        self.ax1.tick_params(axis='y', labelcolor='#000000', labelsize=9)
        ax1b.tick_params(axis='y', labelcolor='#000000', labelsize=9)

        cat_title = "по всем категориям" if category == "Все категории" else f"по категории: {category}"
        self.ax1.set_title(f'Факт и прогноз спроса ({cat_title})', fontweight='bold', fontsize=11)
        self.ax1.set_xlabel('Месяц', fontsize=10)
        self.ax1.grid(True, linestyle='--', alpha=0.5, linewidth=0.7)
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        ax1b.spines['top'].set_visible(False)

        x_labels = df['месяц_str'].tolist()
        self.ax1.set_xticks(range(n))
        self.ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)

        bars_leg = self.ax1.legend(handles=[bars], loc='upper left', fontsize=9)
        line_leg = ax1b.legend(handles=[line], loc='upper right', fontsize=9)
        self.ax1.add_artist(bars_leg)

        def animate1(frame):
            k = min(frame + 1, n)
            for i, bar in enumerate(bars):
                bar.set_height(df['Количество продаж'].iloc[i] if i < k else 0)
            line.set_data(range(k), df['оценка_выручки'].iloc[:k])
            return [line] + [bar for bar in bars[:k]]

        self.anim1 = FuncAnimation(self.fig1, animate1, frames=n, interval=180, blit=True, repeat=False,
                                   cache_frame_data=False)
        self.canvas1.draw()

        self.ax2.clear()
        x_vals = list(range(12))
        y_vals = self.future_monthly['прогноз_выручка'].values[:12].astype(float)
        while len(y_vals) < 12:
            y_vals = np.append(y_vals, y_vals[-1] if len(y_vals) > 0 else 50000)
        month_labels = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]

        line_fut, = self.ax2.plot([], [], 's-', color='#a6d75b', linewidth=2.5, markersize=7, label='2026')
        self.ax2.set_title(f'Прогноз выручки на 2026 ({cat_title})', fontweight='bold', fontsize=11)
        self.ax2.set_xlabel('Месяц', fontsize=10)
        self.ax2.set_ylabel('Выручка (₽)', color='#000000', fontsize=10)
        self.ax2.tick_params(axis='y', labelcolor='#000000', labelsize=9)
        self.ax2.grid(True, linestyle='--', alpha=0.5, linewidth=0.7)
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.set_xlim(-0.5, 11.5)
        self.ax2.set_ylim(0, max(y_vals) * 1.15 if y_vals.size > 0 else 100000)
        self.ax2.set_xticks(x_vals)
        self.ax2.set_xticklabels(month_labels, fontsize=9)
        self.ax2.legend(loc='upper left', fontsize=9)

        def animate2(frame):
            k = min(frame + 1, 12)
            line_fut.set_data(x_vals[:k], y_vals[:k])
            return [line_fut]

        self.anim2 = FuncAnimation(self.fig2, animate2, frames=12, interval=300, blit=True, repeat=False,
                                   cache_frame_data=False)
        self.canvas2.draw()

    def export_to_excel(self):
        if self.future_monthly is None or self.monthly_df is None or self.df_full is None:
            messagebox.showwarning("Внимание", "Сначала запустите анализ.")
            return

        try:
            cat_name = self.current_category
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

            for _, row in self.future_monthly.iterrows():
                month = str(row['месяц_str'])
                rev = int(round(row['прогноз_выручка'])) if pd.notna(row['прогноз_выручка']) else 0
                qty = int(round(row['прогноз_спроса'])) if pd.notna(row['прогноз_спроса']) else 0
                ws1.append([month, rev, qty])

            ws2 = wb.create_sheet("Оценка по месяцам")
            ws2.append(["Месяц", "Факт: спрос (шт)", "Оценка: спрос (шт)", "Факт: выручка (₽)", "Оценка: выручка (₽)"])
            for cell in ws2[1]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="E2F0D9")
                cell.alignment = Alignment(horizontal="center")

            for _, row in self.monthly_df.iterrows():
                month = str(row['месяц_str'])
                fact_qty = int(row['Количество продаж'])
                pred_qty = int(round(row['прогноз_спроса'])) if pd.notna(row['прогноз_спроса']) else 0
                fact_rev = int(fact_qty * row['Цена за шт'])
                pred_rev = int(round(row['оценка_выручки'])) if pd.notna(row['оценка_выручки']) else 0
                ws2.append([month, fact_qty, pred_qty, fact_rev, pred_rev])

            ws3 = wb.create_sheet("Товары")
            ws3.append(["Товар", "Категория", "Средняя цена", "Факт: спрос", "Прогноз: спрос", "Факт: выручка",
                        "Прогноз: выручка"])
            for cell in ws3[1]:
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
                ws3.append([
                    row['Товар'],
                    row['Категория'],
                    round(row['Цена за шт'], 2),
                    int(row['Количество продаж']),
                    int(round(row['прогноз_спроса'])),
                    int(round(row['Количество продаж'] * row['Цена за шт'])),
                    int(round(row['оценка_выручки']))
                ])

            wb.save(path)
            messagebox.showinfo("Успех", f"Прогноз по категории '{cat_name}' сохранён:\n{os.path.basename(path)}")

        except PermissionError:
            messagebox.showerror("Ошибка доступа", "Нет прав на запись. Попробуйте выбрать папку «Документы».")
        except Exception as e:
            messagebox.showerror("Ошибка экспорта", f"Не удалось сохранить файл:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SalesAnalyzerApp(root)
    root.mainloop()