import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")

def preprocess_special_columns(df):
    # Convert acquisition cost to float
    if 'Acquisition_Cost' in df.columns:
        df['Acquisition_Cost'] = (
            df['Acquisition_Cost']
            .replace(r'\$', '', regex=True)
            .astype(float)
        )

    # Convert date to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Date_Month'] = df['Date'].dt.to_period('M').astype(str)

    return df

def plot_data_distribution(df, filename):
    df = preprocess_special_columns(df)

    # Include transformed Date_Month if exists
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    total_plots = len(num_cols) + len(cat_cols)
    cols = 3
    rows = (total_plots + cols - 1) // cols

    plt.figure(figsize=(cols * 6, rows * 4))
    
    plot_num = 1
    for col in num_cols:
        plt.subplot(rows, cols, plot_num)
        sns.histplot(df[col].dropna(), kde=True, bins=30, color='skyblue')
        plt.title(f"Histogram: {col}")
        plot_num += 1

    for col in cat_cols:
        plt.subplot(rows, cols, plot_num)
        sns.countplot(data=df, x=col, hue=col, order=df[col].value_counts().index, palette='Set2', legend=False)
        plt.title(f"Bar Plot: {col}")
        plt.xticks(rotation=45)
        plot_num += 1

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def sample_data(file_path, sample_size, min_class_size=2):
    df = pd.read_csv(file_path)

    # Plot original data distribution before any processing
    plot_data_distribution(df.copy(), 'original_data_distribution.png')

    # Fill missing values in categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df[categorical_cols] = df[categorical_cols].fillna("Missing")

    if len(categorical_cols) > 0:
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            rare_classes = value_counts[value_counts < min_class_size].index.tolist()
            rare_classes_excluding_singles = [cls for cls in rare_classes if value_counts[cls] > 1]
            df[col] = df[col].replace(rare_classes_excluding_singles, "Other")

        try:
            df_sample, _ = train_test_split(df, train_size=sample_size, stratify=df[categorical_cols], random_state=42)
        except ValueError:
            weights = df[categorical_cols].apply(lambda x: x.map(x.value_counts(normalize=True))).mean(axis=1)
            df_sample = df.sample(n=sample_size, random_state=42, weights=weights)
    else:
        df_sample = df.sample(n=sample_size, random_state=42)

    plot_data_distribution(df_sample.copy(), 'sampled_data_distribution.png')
    
    return df_sample

if __name__ == "__main__":
    sampled_df = sample_data('Social_Media_Advertising.csv', sample_size=8000)
    sampled_df.to_csv('sampled_data.csv', index=False)
