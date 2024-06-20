import pandas as pd

def extract_tags(file_path, tab, target):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)

    direction_filter =  target.split('-')[1]
    target =  target.split('-')[0]

    # Load the specified sheet
    pa_zone_df = pd.read_excel(file_path, sheet_name=tab)

    if direction_filter == 'R':
        pa_zone_df = pa_zone_df.loc[:, ~pa_zone_df.columns.str.contains('L')]
    elif direction_filter == 'L':
        pa_zone_df = pa_zone_df.loc[:, ~pa_zone_df.columns.str.contains('R')]

    # Filter rows containing the target string in the 'direction' column and offset 0.0 in the 'OFFSET:' column
    filtered_pa_zone_df = pa_zone_df[
        (pa_zone_df['direction'].astype(str).str.contains(target)) & 
        (pa_zone_df['OFFSET:'] == 0.0)
    ]
    
    # Extract numerical tags from these filtered rows, excluding 0.0
    filtered_tags = []
    for _, row in filtered_pa_zone_df.iterrows():
        for item in row:
            if isinstance(item, (int, float)) and item != 0.0 and not pd.isnull(item):
                filtered_tags.append(item)
    
    
    # Create a DataFrame from the filtered tags
    filtered_tags_df = pd.DataFrame(filtered_tags, columns=['Tag'])
    
    
    return filtered_tags_df

# Parameters
file_path = 'TAGMAP_DHL-HD-Bangkok2-NEW.xlsx'
tab = 'PA ZONE'
target = 'PAF-R'

# Run the function
filtered_tags_df = extract_tags(file_path, tab, target)

# Display the filtered tags DataFrame
print(filtered_tags_df['Tag'].tolist())
