import pandas as pd 
import matplotlib.pyplot as plt


if __name__ == '__main__': 
    RNA = pd.read_csv("data/pdb_rna_year.csv")
    RNA  = RNA.set_index('Year')
    #Fill missing years with the previous year's value
    full_index = pd.Index(range(2024, 1978-1, -1), name = 'Year')
    RNA = RNA.reindex(full_index).reset_index()
    RNA = RNA.bfill()
    #Convert to integers and filter out years before 1990
    RNA = RNA.replace(',', '', regex=True).astype(int)
    RNA = RNA[RNA['Year'] >= 1990]


    protein = pd.read_csv("data/pdb_protein_year.csv")
    #Convert to integers and filter out years before 1990
    protein = protein.replace(',', '', regex=True).astype(int)
    protein = protein[protein['Year'] >= 1990]

    #Check that the lengths of the dataframes are the same
    assert len(protein) == len(RNA)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(protein['Year'], protein['Total Number of Entries Available'], label='Protein', edgecolor='black', zorder = 3, linewidth=0.7, width=1)
    ax.bar(RNA['Year'], RNA['Total Number of Entries Available'], label='RNA', edgecolor='black', zorder=3, linewidth = 0.7, width=1)

    ax.set_xlabel('Year')
    ax.set_ylabel('Total Number of Entries')
    ax.grid(linestyle = '--', zorder = 1)
    ax.legend(loc='upper left', frameon=False)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.tight_layout

    plt.savefig('figures/pdb_statistics.png', dpi=300, bbox_inches='tight')
    