import os
import numpy as np
import pandas as pd

# 1. Resolve dynamic project paths
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# Input from data/raw/
INPUT_CSV = os.path.join(RAW_DATA_DIR, "updated_crop_recommendation.csv")
if not os.path.exists(INPUT_CSV):
    INPUT_CSV = os.path.join(RAW_DATA_DIR, "crop_recommendation.csv")

# Output to data/
OUTPUT_CSV = os.path.join(DATA_DIR, "model_b_agroforestry_carbon.csv")


def generate_model_b_dataset():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Input file not found at: {INPUT_CSV}\n"
            f"Please place 'Crop_recommendation.csv' inside '{RAW_DATA_DIR}'."
        )

    print(f"Loading raw dataset from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.lower()

    # 2. Baseline Mean Above-Ground Biomass Growth Rate (t/ha/yr)
    species_biomass_baselines = {
        'bamboo': 24.5, 'eucalyptus': 20.0, 'poplar': 16.5, 'subabul': 15.0,
        'acacia': 13.0, 'teak': 10.5, 'mahogany': 8.5, 'sandalwood': 6.0,
        'mango': 9.0, 'coconut': 11.0, 'apple': 7.5, 'orange': 7.0, 'papaya': 12.0
    }
    default_crop_biomass = 4.5  # Baseline for standard annual agricultural crops

    # 3. Soil Quality & Climate Suitability Factors
    max_npk = df['n'].max() + df['p'].max() + df['k'].max()
    soil_fertility = (df['n'] + df['p'] + df['k']) / max_npk
    moisture_index = (df['rainfall'] / df['rainfall'].max()) * 0.7 + (df['humidity'] / 100.0) * 0.3

    np.random.seed(42)
    noise = np.random.normal(1.0, 0.05, size=len(df))

    # 4. Target 1: Biomass Yield (t/ha/yr)
    base_yield = df['label'].map(species_biomass_baselines).fillna(default_crop_biomass)
    df['biomass_yield_t_ha_yr'] = np.round(
        base_yield * (0.5 + 0.5 * soil_fertility * moisture_index) * noise, 2
    )

    # 5. Target 2: Carbon Sequestration Potential (t CO2e/ha/yr)
    df['carbon_sequestration_t_co2e_ha_yr'] = np.round(
        df['biomass_yield_t_ha_yr'] * 0.5 * 3.67, 2
    )

    # 6. Target 3: Time to Maturity / Harvest (Years)
    maturity_map = {
        'bamboo': 4, 'subabul': 5, 'poplar': 6, 'eucalyptus': 7, 
        'acacia': 8, 'teak': 20, 'mahogany': 25, 'sandalwood': 15
    }
    df['time_to_maturity_years'] = df['label'].map(maturity_map).fillna(1)

    # Save output dataset
    df.to_csv(OUTPUT_CSV, index=False)
    print("SUCCESS: Model B Dataset generated successfully!")
    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Total rows: {len(df)} | Columns: {list(df.columns)}")


if __name__ == "__main__":
    generate_model_b_dataset()