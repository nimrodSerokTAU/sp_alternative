import pandas as pd
import glob

folder = '/groups/pupko/kseniap/'
output_dir = f"{folder}/DataSets/OrthoMaM_v12a_new/Guidance_Filtering_Paper/pairwise_dpos_res/"
output_file = f"{folder}/DataSets/OrthoMaM_v12a_new/Guidance_Filtering_Paper/mean_pairwise_dpos_v1.csv"

aligners = {
    'MUSCLE': ['muscle', 'MUSCLE'],
    'PRANK': ['prank', 'PRANK'],
    'MAFFT': ['mafft', 'MAFFT'],
    'BALIPHY': ['baliphy','bali_phy', 'BALIPHY']
}


if __name__ == '__main__':
    results = []
    csv_files = glob.glob(f'{output_dir}/*.csv')

    for file in csv_files:
        code = file.split('/')[-1].split('_')[1].split('.')[0]  # Extract code from file name
        df = pd.read_csv(file)
        df_filtered = df[~df['MSAs'].str.contains(r'(_alt_|_concat|_orig)', case=False, na=False)]

        mean_values = {'code': code}

        for aligner, substrings in aligners.items():
            if aligner == 'MAFFT':
                filtered_df = df_filtered[~df_filtered['MSAs'].str.contains('|'.join(['prank', 'muscle', 'bali_phy']), case=False, na=False)]
            else:
                filtered_df = df_filtered[df_filtered['MSAs'].str.contains('|'.join(substrings), case=False, na=False)]

            mean_dpos = filtered_df['dpos'].mean() if not filtered_df.empty else None
            mean_values[aligner] = mean_dpos

        results.append(mean_values)

    mean_dpos_df = pd.DataFrame(results)
    mean_dpos_df.to_csv(output_file, index=False)
    print(f"Mean dpos values saved to {output_file}")
