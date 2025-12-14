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
import re

pd.set_option('future.no_silent_downcasting', True)
plt.rcParams.update({'font.size': 9})


def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename).strip()


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
        self.root.geometry("1550x550")
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

        control_frame = tk.Frame(root, bg="#f8f9fa", pady=10)
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
            df = pd.read_excel(path)
            df.columns = df.columns.str.strip()

            required = ['Дата', 'Товар', 'Категория', 'Цена за шт', 'Количество продаж', 'Канал продаж', 'Скидки']
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Отсутствуют столбцы: {missing}")

            df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Дата'])
            df['Цена за шт'] = pd.to_numeric(df['Цена за шт'], errors='coerce')
            df['Количество продаж'] = pd.to_numeric(df['Количество продаж'], errors='coerce')
            df['Скидки'] = df['Скидки'].fillna('нет').astype(str).str.strip().str.lower()
            df['Скидки'] = df['Скидки'].replace({'нет': 0, '': 0, '-': 0})
            df['Скидки'] = pd.to_numeric(df['Скидки'], errors='coerce').fillna(0)
            df = df.dropna(subset=['Цена за шт', 'Количество продаж'])

            self.df_full = df.copy()

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

            model = NNBoost(n_estimators=50, learning_rate=0.03)
            model.fit(X, y)

            pred = model.predict(X)
            df['прогноз_спроса'] = pred
            df['оценка_выручки'] = df['Цена за шт'] * df['прогноз_спроса']

            df['месяц_period'] = df['Дата'].dt.to_period('M')
            monthly = df.groupby('месяц_period', as_index=False).agg({
                'Количество продаж': 'sum',
                'оценка_выручки': 'sum',
                'Цена за шт': 'mean'
            })
            monthly['месяц_str'] = monthly['месяц_period'].astype(str)
            monthly = monthly.sort_values('месяц_period').reset_index(drop=True)

            last_date = df['Дата'].max()
            future_rows = []
            for i in range(1, 13):
                next_date = last_date + pd.DateOffset(months=i)
                hist = df[df['Дата'].dt.month == next_date.month]
                if len(hist) == 0:
                    hist = df.sample(50, replace=True)
                for _, row in hist.iterrows():
                    new_price = row['Цена за шт'] * (1.01 ** i)
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
            future_df['прогноз_спроса'] = model.predict(X_fut)
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
                future_monthly = future_monthly.append(
                    {'month_id': len(future_monthly), 'прогноз_выручка': avg_rev, 'прогноз_спроса': avg_qty},
                    ignore_index=True
                )
            future_monthly = future_monthly.head(n_months)
            future_monthly['месяц_str'] = [(last_date + pd.DateOffset(months=i+1)).strftime('%Y-%m') for i in range(n_months)]

            self.root.after(0, self._update_ui_with_results, monthly, future_monthly)

        except Exception as e:
            import traceback
            err_msg = str(e)
            print("Ошибка:", traceback.format_exc())
            self.root.after(0, messagebox.showerror, "Ошибка", err_msg)
        finally:
            self.root.after(0, self._set_ui_loading, False)

    def _update_ui_with_results(self, monthly_df, future_monthly):
        try:
            self.fig1.clear()
            self.fig2.clear()
            self.ax1 = self.fig1.add_subplot(111)
            self.ax2 = self.fig2.add_subplot(111)

            self.monthly_df = monthly_df
            self.future_monthly = future_monthly

            self._draw_animation()
            self.export_btn.config(state="normal")

        except Exception as e:
            messagebox.showerror("Ошибка визуализации", str(e))

    def _set_ui_loading(self, state):
        self.is_loading = state
        if state:
            self.load_btn.config(state="disabled", text="Анализ...")
            self.browse_btn.config(state="disabled")
            self.export_btn.config(state="disabled")
        else:
            self.load_btn.config(state="normal", text="Запустить анализ")
            self.browse_btn.config(state="normal")

    def _draw_animation(self):
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

        bars = self.ax1.bar(range(n), df['Количество продаж'],
                           width=0.6, color='#76c68f', alpha=0.7, label='Факт: спрос, шт')
        ax1b = self.ax1.twinx()
        line, = ax1b.plot(range(n), df['оценка_выручки'],
                         'o-', color='#22a7f0', linewidth=2, markersize=6, label='Оценка: выручка, ₽')

        self.ax1.set_ylabel('Спрос, шт', color='#000000', fontsize=10)
        ax1b.set_ylabel('Выручка, ₽', color='#000000', fontsize=10)
        self.ax1.tick_params(axis='y', labelcolor='#000000', labelsize=9)
        ax1b.tick_params(axis='y', labelcolor='#000000', labelsize=9)
        self.ax1.set_title('Фактический спрос и оценка выручки', fontweight='bold', fontsize=11)
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

        self.anim1 = FuncAnimation(
            self.fig1, animate1, frames=n,
            interval=180, blit=True, repeat=False,
            cache_frame_data=False
        )
        self.canvas1.draw()

        self.ax2.clear()
        x_vals = list(range(12))
        y_vals = self.future_monthly['прогноз_выручка'].values[:12].astype(float)
        while len(y_vals) < 12:
            y_vals = np.append(y_vals, y_vals[-1] if len(y_vals) > 0 else 50000)
        month_labels = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]

        line_fut, = self.ax2.plot([], [], 's-', color='#a6d75b', linewidth=2.5, markersize=7, label='2026')

        self.ax2.set_title('Прогноз выручки на 2026 год', fontweight='bold', fontsize=11)
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

        self.anim2 = FuncAnimation(
            self.fig2, animate2, frames=12,
            interval=300, blit=True, repeat=False,
            cache_frame_data=False
        )
        self.canvas2.draw()

    def export_to_excel(self):
        if self.future_monthly is None or self.monthly_df is None or self.df_full is None:
            messagebox.showwarning("Внимание", "Сначала запустите анализ.")
            return

        try:
            default_name = sanitize_filename("Прогноз_спроса")
            path = filedialog.asksaveasfilename(
                initialfile=default_name,
                defaultextension=".xlsx",
                filetypes=[("Excel", "*.xlsx")],
                title="Сохранить прогноз"
            )
            if not path:
                return
            if not path.endswith('.xlsx'):
                path += '.xlsx'

            df_forecast = pd.DataFrame({
                'Месяц': self.future_monthly['месяц_str'],
                'Прогноз выручки (₽)': self.future_monthly['прогноз_выручка'].round(0),
                'Прогноз спроса (шт)': self.future_monthly['прогноз_спроса'].round(0)
            })

            df_eval = pd.DataFrame({
                'Месяц': self.monthly_df['месяц_str'],
                'Факт: спрос (шт)': self.monthly_df['Количество продаж'],
                'Оценка: спрос (шт)': self.monthly_df['прогноз_спроса'].round(0),
                'Факт: выручка (₽)': (self.monthly_df['Количество продаж'] * self.monthly_df['Цена за шт']).round(0),
                'Оценка: выручка (₽)': self.monthly_df['оценка_выручки'].round(0)
            })

            df_elast = []
            grouped = self.df_full.groupby(['Товар', 'Дата'])
            for (item, date), group in grouped:
                if len(group) >= 2:
                    g = group.sort_values('Цена за шт').head(2)
                    if len(g) == 2 and g.iloc[0]['Цена за шт'] != g.iloc[1]['Цена за шт']:
                        p1, q1 = g.iloc[0]['Цена за шт'], g.iloc[0]['Количество продаж']
                        p2, q2 = g.iloc[1]['Цена за шт'], g.iloc[1]['Количество продаж']
                        if p1 > 0 and q1 > 0:
                            dp = (p2 - p1) / p1 * 100
                            dq = (q2 - q1) / q1 * 100
                            elast = round(dq / dp, 2) if dp != 0 else 0
                            df_elast.append([
                                item, g.iloc[0]['Категория'], date.strftime('%d.%m.%Y'),
                                p1, q1, p2, q2,
                                round(dp, 1), round(dq, 1), elast
                            ])

            df_elasticity = pd.DataFrame(df_elast, columns=[
                "Товар", "Категория", "Дата",
                "Цена₁ (₽)", "Спрос₁ (шт)",
                "Цена₂ (₽)", "Спрос₂ (шт)",
                "ΔЦена (%)", "ΔСпрос (%)", "Эластичность"
            ])

            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                df_forecast.to_excel(writer, sheet_name='Прогноз на 2026', index=False)
                df_eval.to_excel(writer, sheet_name='Оценка по месяцам', index=False)
                if not df_elasticity.empty:
                    df_elasticity.to_excel(writer, sheet_name='Ценовая эластичность', index=False)

            messagebox.showinfo("Успех", f"Прогноз сохранён:\n{os.path.basename(path)}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SalesAnalyzerApp(root)
    root.mainloop()