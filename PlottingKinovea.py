import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import seaborn as sns
import matplotlib.colors as mcolors

# Configuration parameters
DATA_FILE = r"C:\Users\TRIOS\OneDrive - KTH\Skrivbordet\Collection_data1.xlsx"
DATA_SHEET = "Min"
IMAGE_FILE = r"C:\Users\TRIOS\Downloads\Skull lateral zone.jpg"
GAUSSIAN_SIGMA = 20  # Smoothing parameter for the heatmap

# Zones and their positions on the image
ZONES = ['F', 'L', 'B', 'T']
ZONE_POSITIONS = {
    'F': (1200, 240),  # Frontal
    'L': (600, 240),   # Lateral
    'B': (220, 440),   # Back
    'T': (700, 60)     # Top
}

# Mapping sub-zones to macro zones
ZONE_MAPPING = {
    'F1': 'F', 'F2': 'F',
    'L1': 'L', 'L2': 'L', 'L3': 'L', 'L4': 'L', 'L5': 'L', 'L6': 'L', 'L8': 'L', 'L9': 'L', 'L10': 'L',
    'B1': 'B', 'B2': 'B', 'B3': 'B',
    'T': 'T', 'T1': 'T'
}


def load_data(file_path: str, sheet_name: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_excel(file_path, sheet_name)

def load_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return plt.imread(image_path)

class HeatmapGenerator:
    def __init__(self, df: pd.DataFrame, head_image, zones: list, zone_positions: dict, zone_mapping: dict):
        self.df = df
        self.head_image = head_image
        self.zones = zones
        self.zone_positions = zone_positions
        self.zone_mapping = zone_mapping

    def map_zone(self, zone):
        return self.zone_mapping.get(zone, None)

    def compute_severity(self, group_data: pd.DataFrame) -> dict:
        severity = {}
        for _, row in group_data.iterrows():
            if pd.notna(row['Heatmap']) and pd.notna(row['VI(m/S)']):
                for zone in str(row['Heatmap']).split('-'):
                    zone = zone.strip()
                    macro_zone = self.map_zone(zone)
                    if macro_zone:
                        if macro_zone not in severity:
                            severity[macro_zone] = {'count': 0, 'vi_sum': 0.0}
                        severity[macro_zone]['count'] += 1
                        severity[macro_zone]['vi_sum'] += row['VI(m/S)']
        return {zone: data['vi_sum'] for zone, data in severity.items()}

    def compute_frequency(self, group_data: pd.DataFrame) -> dict:
        freq = {}
        for _, row in group_data.iterrows():
            if pd.notna(row['Heatmap']):
                for zone in str(row['Heatmap']).split('-'):
                    zone = zone.strip()
                    macro_zone = self.map_zone(zone)
                    if macro_zone:
                        freq[macro_zone] = freq.get(macro_zone, 0) + 1
        return freq

    def compute_intensity(self, group_data: pd.DataFrame) -> dict:
        data = {}
        for _, row in group_data.iterrows():
            if pd.notna(row['Heatmap']) and pd.notna(row['VI(m/S)']):
                for zone in str(row['Heatmap']).split('-'):
                    zone = zone.strip()
                    macro_zone = self.map_zone(zone)
                    if macro_zone:
                        if macro_zone not in data:
                            data[macro_zone] = {'count': 0, 'vi_sum': 0.0}
                        data[macro_zone]['count'] += 1
                        data[macro_zone]['vi_sum'] += row['VI(m/S)']
        return {zone: d['vi_sum'] / d['count'] for zone, d in data.items() if d['count'] > 0}

    def compute_std(self, group_data: pd.DataFrame) -> dict:
        std_data = {}
        for _, row in group_data.iterrows():
            if pd.notna(row['Heatmap']) and pd.notna(row['VI(m/S)']):
                for zone in str(row['Heatmap']).split('-'):
                    zone = zone.strip()
                    macro_zone = self.map_zone(zone)
                    if macro_zone:
                        if macro_zone not in std_data:
                            std_data[macro_zone] = {'count': 0, 'vi_sum': 0.0, 'vi_squared_sum': 0.0}
                        std_data[macro_zone]['count'] += 1
                        std_data[macro_zone]['vi_sum'] += row['VI(m/S)']
                        std_data[macro_zone]['vi_squared_sum'] += row['VI(m/S)'] ** 2

        result = {}
        for zone, data in std_data.items():
            count = data['count']
            if count > 1:
                mean_vi = data['vi_sum'] / count
                variance = (data['vi_squared_sum'] / count) - (mean_vi ** 2)
                result[zone] = np.sqrt(variance)
        return result

    def create_metric_map(self, groupby_cols: list, title_suffix: str, metric_fn, cmap='viridis', metric_name='Metric'):
        groups = self.df.groupby(groupby_cols)
        ncols = 3

        valid_groups = [(name, data) for name, data in groups if not any('Average' in str(n) for n in (name if isinstance(name, tuple) else (name,)))]
        n_groups = len(valid_groups)
        nrows = int(np.ceil(n_groups / ncols))

        fig, axs = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
        fig.suptitle(f'MEAN - {metric_name} Map by {title_suffix}', fontsize=16)

        metric_data = []
        global_max = 0

        for group_name, group_data in valid_groups:
            values = metric_fn(group_data)
            if values:
                global_max = max(global_max, max(values.values()))
            metric_data.append((group_name, values))

        for i, (group_name, values) in enumerate(metric_data):
            ax = axs.flatten()[i]
            heatmap = np.zeros(self.head_image.shape[:2])

            for zone, val in values.items():
                if zone in self.zone_positions:
                    x, y = self.zone_positions[zone]
                    x = int(np.clip(x, 0, heatmap.shape[1] - 1))
                    y = int(np.clip(y, 0, heatmap.shape[0] - 1))
                    heatmap[y, x] = val

            heatmap_smoothed = gaussian_filter(heatmap, sigma=GAUSSIAN_SIGMA)
            heatmap_norm = heatmap_smoothed / global_max if global_max > 0 else heatmap_smoothed

            ax.imshow(self.head_image)
            ax.imshow(heatmap_norm, cmap=cmap, alpha=0.7, vmin=0, vmax=1)

            
            import matplotlib.colors as mcolors  # assicurati che sia in cima al file

            norm = mcolors.Normalize(vmin=0, vmax=global_max)
            cmap = plt.get_cmap("RdYlGn_r")  # rosso = alto, verde = basso

            for zone, (x, y) in self.zone_positions.items():
                if zone in values:
                    color = cmap(norm(values[zone]))
                    text_color = 'black' if mcolors.rgb_to_hsv(color[:3])[2] > 0.8 else 'white'
                    ax.text(x, y, f"{zone}\n{values[zone]:.1f}",
                            ha='center', va='center', fontsize=16, fontweight='bold',
                            color=text_color,
                            bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.5', linewidth=1.5))

        

            group_title = group_name if isinstance(group_name, str) else " - ".join(map(str, group_name))
            total_val = sum(values.values())

            if metric_name == 'Impact Frequency':
                ax.set_title(f"{group_title}\nTotal Impacts: {int(total_val)}")
            else:
                ax.set_title(f"{group_title}")

            ax.axis('off')

        for j in range(len(metric_data), nrows * ncols):
            axs.flatten()[j].axis('off')

        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        plt.show()

    def create_std_map(self, groupby_cols: list, title_suffix: str):
        self.create_metric_map(groupby_cols, title_suffix, self.compute_std, cmap='YlGnBu', metric_name='Standard Deviation (VI)')

    def create_severity_map(self, groupby_cols: list, title_suffix: str):
        self.create_metric_map(groupby_cols, title_suffix, self.compute_severity, cmap='YlGnBu', metric_name='Severity Index (Count × Mean VI)')

    def create_frequency_map(self, groupby_cols: list, title_suffix: str):
        self.create_metric_map(groupby_cols, title_suffix, self.compute_frequency, cmap='YlGnBu', metric_name='Impact Frequency')

    def create_intensity_map(self, groupby_cols: list, title_suffix: str):
        self.create_metric_map(groupby_cols, title_suffix, self.compute_intensity, cmap='YlGnBu', metric_name='Mean VI Intensity')


    def clean_velocity_analysis(self):
        sns.set(style="whitegrid")
        df_clean = self.df.dropna(subset=['Sex', 'Discipline', 'VI(m/S)'])
        mean_values = df_clean.groupby(['Sex', 'Discipline'])['VI(m/S)'].mean().reset_index()

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(data=mean_values, x='Sex', y='VI(m/S)', hue='Discipline', palette='Set2')

        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=9)

        ax.set_title('Mean VI (m/s) by Sex and Discipline', fontsize=16)
        ax.set_xlabel('Sex', fontsize=12)
        ax.set_ylabel('Mean VI (m/s)', fontsize=12)
        ax.legend(title='Discipline', fontsize=10, title_fontsize=11)

        plt.tight_layout()
        plt.show()

    def velocity_mean_std_analysis(self):
        sns.set(style="whitegrid")
        df_clean = self.df.dropna(subset=['Sex', 'Discipline', 'VI(m/S)'])
        stats_df = df_clean.groupby(['Sex', 'Discipline'])['VI(m/S)'].agg(['mean', 'std']).reset_index()

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=stats_df, x='Sex', y='mean', hue='Discipline', palette='Set2', errorbar=None)

        for i, row in stats_df.iterrows():
            x_pos = i % len(stats_df['Sex'].unique())
            discipline_offset = list(stats_df['Discipline'].unique()).index(row['Discipline']) * 0.2 - 0.2
            ax.errorbar(x=x_pos + discipline_offset, y=row['mean'], yerr=row['std'], fmt='none', c='black', capsize=5)

        ax.set_title('Mean VI (m/s) with Standard Deviation by Sex and Discipline', fontsize=16)
        ax.set_xlabel('Sex', fontsize=12)
        ax.set_ylabel('Mean VI (m/s)', fontsize=12)
        ax.legend(title='Discipline', fontsize=10, title_fontsize=11)

        plt.tight_layout()
        plt.show()

# --- MAIN ---
if __name__ == '__main__':
    df = load_data(DATA_FILE, DATA_SHEET)
    head_image = load_image(IMAGE_FILE)

    heatmap_gen = HeatmapGenerator(df, head_image, ZONES, ZONE_POSITIONS, ZONE_MAPPING)

    heatmap_gen.clean_velocity_analysis()
    heatmap_gen.velocity_mean_std_analysis()
    heatmap_gen.create_std_map(['Sex', 'Discipline'], "Sex and Discipline")
    heatmap_gen.create_frequency_map(['Sex', 'Discipline'], "Sex and Discipline")
    heatmap_gen.create_intensity_map(['Sex', 'Discipline'], "Sex and Discipline")
    heatmap_gen.create_severity_map(['Sex', 'Discipline'], "Sex and Discipline")
