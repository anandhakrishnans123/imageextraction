import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Streamlit UI for synthetic data generation
st.title("Synthetic Data Generator")

# Upload the Excel file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Load all sheets from the Excel file
    excel_data = pd.ExcelFile(uploaded_file)
    sheets = excel_data.sheet_names

    # Multi-select box for selecting sheets
    selected_sheets = st.multiselect("Select sheets to process", sheets)

    # Initialize dictionaries for user inputs and configurations
    rows_to_generate = {}
    columns_for_sampling = {}
    sampling_values = {}
    data_type_choices = {}
    numeric_ranges = {}

    if selected_sheets:
        for sheet in selected_sheets:
            st.markdown("---")
            # Display settings for each sheet
            st.subheader(f"Settings for Sheet: {sheet}")

            rows_to_generate[sheet] = st.number_input(
                f"Number of synthetic rows for sheet '{sheet}'",
                min_value=1,
                step=1,
                value=10,
                key=f"rows_{sheet}",
            )

            # Allow user to select columns
            columns = pd.read_excel(uploaded_file, sheet_name=sheet).columns
            selected_columns = st.multiselect(
                f"Select columns to sample data for sheet '{sheet}'",
                columns,
                key=f"columns_{sheet}",
            )
            columns_for_sampling[sheet] = selected_columns

            column_values = {}
            column_data_types = {}
            column_numeric_ranges = {}
            for col in selected_columns:
                st.subheader(f"Column settings for: {col}")

                # Data type selection
                data_type = st.selectbox(
                    f"Select the data type for column '{col}' in sheet '{sheet}'",
                    options=["Select", "Numerical", "Categorical", "Date"],
                    index=0,
                    key=f"data_type_{sheet}_{col}",
                )
                column_data_types[col] = data_type

                # Data type-specific inputs
                if data_type == "Numerical":
                    min_value = st.number_input(
                        f"Enter the minimum value for column '{col}' in sheet '{sheet}'",
                        value=0.0,
                        key=f"min_{sheet}_{col}",
                    )
                    max_value = st.number_input(
                        f"Enter the maximum value for column '{col}' in sheet '{sheet}'",
                        value=100.0,
                        key=f"max_{sheet}_{col}",
                    )
                    column_numeric_ranges[col] = (min_value, max_value)

                elif data_type == "Categorical":
                    value = st.text_input(
                        f"Enter the sampling values (comma-separated) for column '{col}' in sheet '{sheet}'",
                        key=f"value_{sheet}_{col}",
                    )
                    column_values[col] = value

                elif data_type == "Date":
                    value = st.text_input(
                        f"Enter the comma-separated date values for column '{col}' in sheet '{sheet}' (e.g., 2024-01-01, 2024-01-02)",
                        key=f"value_{sheet}_{col}",
                    )
                    column_values[col] = value

                st.markdown("---")

            sampling_values[sheet] = column_values
            data_type_choices[sheet] = column_data_types
            numeric_ranges[sheet] = column_numeric_ranges

    if st.button("Generate and Save Augmented Data"):
        # Dictionary to store augmented data for all sheets
        all_augmented_data = {}

        for sheet_name in selected_sheets:
            st.subheader(f"Processing Data for Sheet: {sheet_name}")

            original_data = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            data_without_header = original_data[1:]

            synthetic_data = {}
            num_synthetic_rows = rows_to_generate[sheet_name]

            for column in data_without_header.columns:
                if column in columns_for_sampling[sheet_name]:
                    data_type = data_type_choices[sheet_name].get(column, "Select")
                    value = sampling_values[sheet_name].get(column, None)

                    if data_type == "Numerical":
                        min_value, max_value = numeric_ranges[sheet_name].get(column, (0, 100))
                        synthetic_data[column] = np.random.uniform(min_value, max_value, num_synthetic_rows).tolist()

                    elif data_type == "Categorical":
                        if value is not None:
                            unique_values = [val.strip() for val in value.split(',') if val.strip()]
                            if unique_values:
                                synthetic_data[column] = [np.random.choice(unique_values) for _ in range(num_synthetic_rows)]
                            else:
                                synthetic_data[column] = ["Undefined"] * num_synthetic_rows

                    elif data_type == "Date":
                        if value is not None:
                            try:
                                date_values = [pd.to_datetime(date.strip()) for date in value.split(',')]
                                synthetic_data[column] = [np.random.choice(date_values) for _ in range(num_synthetic_rows)]
                            except ValueError:
                                st.error(f"Invalid date format for column '{column}' in sheet '{sheet_name}'.")
                                synthetic_data[column] = [pd.to_datetime("2024-01-01")] * num_synthetic_rows

                else:
                    if pd.api.types.is_numeric_dtype(data_without_header[column]):
                        column_data = data_without_header[column].dropna()
                        if len(column_data) > 0:
                            mean = column_data.mean()
                            std_dev = column_data.std()
                            synthetic_data[column] = np.random.normal(mean, std_dev, num_synthetic_rows).tolist()
                        else:
                            synthetic_data[column] = [0] * num_synthetic_rows

                    elif pd.api.types.is_categorical_dtype(data_without_header[column]) or data_without_header[column].dtype == "object":
                        column_data = data_without_header[column].dropna()
                        if len(column_data) > 0:
                            category_counts = column_data.value_counts(normalize=True)
                            categories = category_counts.index.tolist()
                            probabilities = category_counts.values.tolist()
                            synthetic_data[column] = np.random.choice(categories, num_synthetic_rows, p=probabilities).tolist()
                        else:
                            synthetic_data[column] = [None] * num_synthetic_rows

                    elif pd.api.types.is_datetime64_any_dtype(data_without_header[column]):
                        column_data = data_without_header[column].dropna()
                        if len(column_data) > 0:
                            min_date = column_data.min()
                            max_date = column_data.max()
                            time_range = (max_date - min_date).days
                            random_days = np.random.randint(0, time_range, num_synthetic_rows)
                            synthetic_data[column] = [min_date + pd.Timedelta(days=days) for days in random_days]
                        else:
                            synthetic_data[column] = [pd.to_datetime("2024-01-01")] * num_synthetic_rows

            synthetic_data_df = pd.DataFrame(synthetic_data)
            augmented_data = pd.concat([original_data, synthetic_data_df], ignore_index=True)
            all_augmented_data[sheet_name] = augmented_data

            st.markdown(f"### Augmented Data for Sheet: {sheet_name}")
            st.write(augmented_data.head())
            st.markdown("---")

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for sheet in sheets:
                if sheet in all_augmented_data:
                    all_augmented_data[sheet].to_excel(writer, index=False, sheet_name=sheet)
                else:
                    pd.read_excel(uploaded_file, sheet_name=sheet).to_excel(writer, index=False, sheet_name=sheet)

        output.seek(0)
        st.download_button(
            label="Download File with Augmented Sheets",
            data=output,
            file_name="augmented_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.success("Augmented data has been saved and is ready for download!")
